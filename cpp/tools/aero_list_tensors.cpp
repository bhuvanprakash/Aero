// aero_list_tensors – print tensor index from an .aero file.

#include "aero/aero.hpp"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    int limit = -1;
    const char* path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc)
            limit = std::atoi(argv[++i]);
        else
            path = argv[i];
    }

    if (!path) {
        std::fprintf(stderr,
                     "Usage: aero_list_tensors <file.aero> [--limit N]\n");
        return 1;
    }

    try {
        aero::AeroFile file(path);
        auto tensors = file.load_tidx();

        int count = static_cast<int>(tensors.size());
        int show  = (limit >= 0 && limit < count) ? limit : count;

        std::printf("Tensors (%d, showing %d):\n", count, show);
        for (int i = 0; i < show; ++i) {
            const auto& t = tensors[static_cast<size_t>(i)];
            std::printf("  %-40s  dtype=%-5s  shape=[",
                        t.name.c_str(), aero::dtype_name(t.dtype));
            for (size_t j = 0; j < t.shape.size(); ++j) {
                if (j > 0) std::printf(",");
                std::printf("%" PRId64, t.shape[j]);
            }
            std::printf("]  shard=%d  bytes=%" PRIu64 "\n",
                        t.shard_id, t.data_len);
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
