// aero_validate – verify chunk integrity (metadata by default, --full for WTSH).

#include "aero/aero.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    bool full = false;
    const char* path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0)
            full = true;
        else
            path = argv[i];
    }

    if (!path) {
        std::fprintf(stderr, "Usage: aero_validate <file.aero> [--full]\n");
        return 1;
    }

    try {
        aero::AeroFile file(path);
        int errors = 0;

        for (const auto& c : file.chunks()) {
            bool is_wtsh = (c.fourcc == "WTSH");

            if (is_wtsh && !full) {
                std::printf("  SKIP  %s (%s) [use --full to verify]\n",
                            c.fourcc.c_str(), c.name.c_str());
                continue;
            }

            try {
                if (is_wtsh) {
                    if (file.stream_verify_chunk(c)) {
                        std::printf("  OK    %s (%s) [streamed]\n",
                                    c.fourcc.c_str(), c.name.c_str());
                    } else {
                        std::printf("  FAIL  %s (%s) BLAKE3 mismatch\n",
                                    c.fourcc.c_str(), c.name.c_str());
                        ++errors;
                    }
                } else {
                    auto data = file.read_chunk_bytes(c, /*verify=*/true);
                    std::printf("  OK    %s (%s) %zu bytes\n",
                                c.fourcc.c_str(), c.name.c_str(), data.size());
                }
            } catch (const std::exception& e) {
                std::printf("  FAIL  %s (%s): %s\n",
                            c.fourcc.c_str(), c.name.c_str(), e.what());
                ++errors;
            }
        }

        if (errors > 0) {
            std::printf("\n%d chunk(s) failed validation.\n", errors);
            return 1;
        }

        // Also try parsing MMSG + TIDX if present.
        try {
            if (file.find_chunk("MMSG")) file.load_manifest();
        } catch (const std::exception& e) {
            std::printf("  FAIL  MMSG parse: %s\n", e.what());
            ++errors;
        }
        try {
            if (file.find_chunk("TIDX")) file.load_tidx();
        } catch (const std::exception& e) {
            std::printf("  FAIL  TIDX parse: %s\n", e.what());
            ++errors;
        }

        if (errors > 0) {
            std::printf("\n%d error(s).\n", errors);
            return 1;
        }

        std::printf("\nAll chunks verified OK.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
