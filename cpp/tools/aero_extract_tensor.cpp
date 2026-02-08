// aero_extract_tensor – extract raw tensor bytes to a file.

#include "aero/aero.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "Usage: aero_extract_tensor <file.aero> "
                     "\"<tensor.name>\" <out.bin>\n");
        return 1;
    }

    try {
        aero::AeroFile file(argv[1]);
        auto data = file.read_tensor_bytes(argv[2]);

        FILE* out = std::fopen(argv[3], "wb");
        if (!out) {
            std::fprintf(stderr, "Error: cannot open output file: %s\n",
                         argv[3]);
            return 1;
        }
        std::fwrite(data.data(), 1, data.size(), out);
        std::fclose(out);

        std::printf("Extracted %zu bytes for tensor '%s' -> %s\n",
                    data.size(), argv[2], argv[3]);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
