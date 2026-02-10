/**
 * LibFuzzer target for AERO file parsing.
 * Writes fuzz input to a temp file and opens it with AeroFile to exercise
 * header + TOC + chunk parsing. Build with: -fsanitize=fuzzer
 *
 * Build (from cpp/build):
 *   cmake -S .. -B . -DAERO_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
 *   cmake --build . --target fuzz_aero_parse
 * Run:
 *   ./fuzz_aero_parse /path/to/corpus [options]
 */
#include "aero/aero.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

static const size_t kMaxFuzzInput = 1024 * 1024;  // 1 MB max to avoid filling disk

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (data == nullptr || size > kMaxFuzzInput)
        return 0;

    std::string tmpl = "/tmp/aero_fuzz_XXXXXX";
    std::vector<char> v(tmpl.begin(), tmpl.end());
    v.push_back('\0');
    int fd = mkstemp(v.data());
    if (fd < 0) return 0;
    std::string path(v.data());
    close(fd);

    std::ofstream out(path, std::ios::binary);
    if (!out || !out.write(reinterpret_cast<const char*>(data), size)) {
        std::remove(path.c_str());
        return 0;
    }
    out.close();

    try {
        aero::AeroFile file(path);
        (void)file.chunks();
        (void)file.load_tidx();
    } catch (const std::exception&) {
        // Expected for invalid inputs
    } catch (...) {
    }

    std::remove(path.c_str());
    return 0;
}
