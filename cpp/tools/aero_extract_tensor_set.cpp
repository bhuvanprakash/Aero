// aero_extract_tensor_set – extract a tensor from an AEROSET.
#include "aero/aeroset.hpp"
#include <cstdio>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: aero_extract_tensor_set <model.aeroset.json> "
            "\"<tensor>\" <out.bin>\n");
        return 1;
    }
    try {
        aero::AeroSetFile set(argv[1]);
        auto bytes = set.read_tensor_bytes(argv[2]);
        std::ofstream out(argv[3], std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "cannot open %s for writing\n", argv[3]);
            return 1;
        }
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::printf("Extracted %zu bytes -> %s\n", bytes.size(), argv[3]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
