# AERO v0.1 – C++ Reader Library

Cross-platform C++ 17 library and CLI tools for reading `.aero` model
container files.

## Building

### Prerequisites

- CMake ≥ 3.18
- A C++17 compiler (GCC, Clang, or MSVC)

All library dependencies are fetched automatically via CMake
`FetchContent`:

| Dependency | Purpose | Source |
|------------|---------|--------|
| [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) (C impl) | Per-chunk integrity | vendored via FetchContent |
| [zstd](https://github.com/facebook/zstd) | Decompress metadata chunks | FetchContent |
| [msgpack-c](https://github.com/msgpack/msgpack-c) | Parse MMSG / TIDX | FetchContent (header-only) |

### macOS / Linux

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j$(nproc)
```

### Windows (MSVC)

```powershell
cmake -S cpp -B cpp\build
cmake --build cpp\build --config Release
```

### Using System Dependencies

```bash
cmake -S cpp -B cpp/build \
  -DAERO_USE_SYSTEM_ZSTD=ON \
  -DAERO_USE_SYSTEM_MSGPACK=ON
```

## CLI Tools

All tools return exit code 0 on success, 1 on error.

### `aero_inspect <file.aero>`

Print header info, chunk table, and model metadata (if available).

### `aero_validate <file.aero> [--full]`

Verify BLAKE3 integrity of metadata chunks.  With `--full`, also
stream-verify weight shards (reads entire file).

### `aero_list_tensors <file.aero> [--limit N]`

Print tensor names, shapes, dtypes, and sizes from the TIDX chunk.

### `aero_extract_tensor <file.aero> "<name>" <out.bin>`

Extract raw bytes for a single tensor to a file.

## Library API

```cpp
#include <aero/aero.hpp>

aero::AeroFile file("model.aero");

// Iterate chunks
for (const auto& c : file.chunks())
    printf("%s %s\n", c.fourcc.c_str(), c.name.c_str());

// Load tensor index
auto tensors = file.load_tidx();

// Read tensor bytes (copy)
auto data = file.read_tensor_bytes("embed.weight");

// Zero-copy mmap view
auto view = file.view_tensor("embed.weight");
// view.data points directly into mmap'd weight shard
```

## C API (single-header)

A minimal C ABI is provided for FFI from any language (Rust, Go, Python, etc.):

```c
#include <aero/aero_capi.h>

aero_handle_t h = aero_load("model.aero");
if (!h) { printf("error: %s\n", aero_error_string()); return 1; }

uint32_t n = aero_tensor_count(h);
const uint8_t* data; uint64_t size;
if (aero_tensor_ptr(h, "layer.0.weight", &data, &size) == 0) {
    /* zero-copy: use data[0..size-1] */
}
aero_close(h);
```

Build the shared library (libaero.so / libaero.dylib / aero.dll) with the same
CMake build; the header is `include/aero/aero_capi.h`. See [docs/RUNTIME_SPEC.md](../docs/RUNTIME_SPEC.md) for the full API.

## Error policy

- **C API:** All functions return error codes (0 = success, non-zero = error).
  No exceptions cross the ABI. Last error message is available via `aero_error_string()`.
- **C++ API:** The core implementation may throw `aero::Error` (or other exceptions)
  on parse or I/O failure. The C wrapper catches these and converts them to error
  codes and error strings.

## Key Implementation Details

- **Zero-copy:** `view_tensor(name)` (C++: `AeroFile::view_tensor`) and `aero_tensor_ptr()` (C API)
  return a pointer into a memory-mapped region; no copy is made. Use this path for
  inference when you want to avoid loading full tensors into RAM.

- **Unaligned mmap offsets**: AERO chunk offsets are 16-byte aligned, but
  OS mapping APIs require page/granularity alignment.  `MMapRegion::map()`
  maps from the nearest aligned base and returns a sliced view at the
  correct offset.

- **Streaming BLAKE3**: For large WTSH chunks, `stream_verify_chunk()`
  hashes in 64 MB blocks without loading the full shard into RAM.

- **Forward-compatible msgpack**: Unknown keys in TIDX / MMSG maps are
  silently ignored, allowing older readers to open files with new fields.

## Fuzz (optional)

Build a libFuzzer target to exercise the parser (Unix, clang):

```bash
cmake -S cpp -B cpp/build -DAERO_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
cmake --build cpp/build --target fuzz_aero_parse
./cpp/build/fuzz_aero_parse /path/to/corpus -max_total_time=60
```
