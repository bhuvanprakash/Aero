cmake_minimum_required(VERSION 3.18)
project(aerotensor_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

include(FetchContent)

# Allow older cmake_minimum_required() in vendored deps.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# ── Options ─────────────────────────────────────────────────────────────────
option(AERO_USE_SYSTEM_ZSTD    "Use system-installed zstd"    OFF)
option(AERO_USE_SYSTEM_MSGPACK "Use system-installed msgpack" OFF)
option(AERO_BUILD_FUZZ         "Build libFuzzer target (Unix, clang)" OFF)

# ── BLAKE3 (C implementation, portable build) ──────────────────────────────
FetchContent_Declare(
    blake3
    GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
    GIT_TAG        1.5.5
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(blake3)

set(BLAKE3_SOURCES
    ${blake3_SOURCE_DIR}/c/blake3.c
    ${blake3_SOURCE_DIR}/c/blake3_dispatch.c
    ${blake3_SOURCE_DIR}/c/blake3_portable.c
)

# On ARM64, include the NEON implementation (always available on aarch64).
# On x86, stick to portable-only to avoid SIMD build complexity.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    list(APPEND BLAKE3_SOURCES ${blake3_SOURCE_DIR}/c/blake3_neon.c)
endif()

add_library(blake3_c STATIC ${BLAKE3_SOURCES})
target_include_directories(blake3_c PUBLIC ${blake3_SOURCE_DIR}/c)

# Disable x86 SIMD everywhere; NEON is handled by source inclusion above.
target_compile_definitions(blake3_c PRIVATE
    BLAKE3_NO_SSE2
    BLAKE3_NO_SSE41
    BLAKE3_NO_AVX2
    BLAKE3_NO_AVX512
)

# Only disable NEON on non-ARM platforms.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    target_compile_definitions(blake3_c PRIVATE BLAKE3_NO_NEON)
endif()

# ── Zstd ────────────────────────────────────────────────────────────────────
if(AERO_USE_SYSTEM_ZSTD)
    find_package(zstd REQUIRED)
    add_library(aero_zstd ALIAS zstd::libzstd_static)
else()
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG        v1.5.6
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  build/cmake
    )
    FetchContent_MakeAvailable(zstd)
    add_library(aero_zstd ALIAS libzstd_static)
endif()

# ── msgpack-c (C++ header-only) ────────────────────────────────────────────
if(AERO_USE_SYSTEM_MSGPACK)
    find_package(msgpack-cxx REQUIRED)
else()
    set(MSGPACK_USE_BOOST     OFF CACHE BOOL "" FORCE)
    set(MSGPACK_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(MSGPACK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        msgpack-cxx
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG        cpp-6.1.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(msgpack-cxx)
endif()

# ── nlohmann/json (header-only, for AEROSET) ───────────────────────────────
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ── Aero static library ────────────────────────────────────────────────────
add_library(aero STATIC src/aero.cpp)
target_include_directories(aero PUBLIC include)
target_link_libraries(aero PUBLIC blake3_c aero_zstd msgpack-cxx)
target_compile_features(aero PUBLIC cxx_std_17)

# ── Aero-set static library (adds JSON / AEROSET support) ──────────────────
add_library(aero_set STATIC src/aeroset.cpp)
target_include_directories(aero_set PUBLIC include)
target_link_libraries(aero_set PUBLIC aero nlohmann_json::nlohmann_json)

# ── Aero C API shared library (libaero.so / libaero.dylib / aero.dll) ───────
# Includes state-dict API (load + dequant, optional GGUF→HF name mapping)
add_library(aero_shared SHARED src/aero.cpp src/aero_capi.cpp src/aero_dequant.cpp src/aero_state_dict.cpp)
target_include_directories(aero_shared PUBLIC include)
target_link_libraries(aero_shared PRIVATE blake3_c aero_zstd msgpack-cxx)
target_compile_features(aero_shared PUBLIC cxx_std_17)
set_target_properties(aero_shared PROPERTIES
    OUTPUT_NAME aero
    SOVERSION 0
    VERSION 0.2.0
    PUBLIC_HEADER "include/aero/aero_capi.h"
)
if(WIN32)
    target_compile_definitions(aero_shared PRIVATE AERO_CAPI_EXPORT)
endif()

# ── CLI tools ───────────────────────────────────────────────────────────────
foreach(tool aero_inspect aero_validate aero_list_tensors aero_extract_tensor)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE aero)
endforeach()

foreach(tool aero_inspect_set aero_extract_tensor_set)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE aero_set)
endforeach()

# ── Fuzz target (optional; Unix + clang, -fsanitize=fuzzer) ──────────────────
if(AERO_BUILD_FUZZ AND NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_executable(fuzz_aero_parse fuzz/fuzz_aero_parse.cpp)
    target_link_libraries(fuzz_aero_parse PRIVATE aero)
    target_include_directories(fuzz_aero_parse PRIVATE include)
    target_compile_options(fuzz_aero_parse PRIVATE -fsanitize=fuzzer -fno-omit-frame-pointer)
    target_link_options(fuzz_aero_parse PRIVATE -fsanitize=fuzzer)
endif()
