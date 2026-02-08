// aero_inspect_set – inspect an AEROSET index.
#include "aero/aeroset.hpp"
#include "aero/aero.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: aero_inspect_set <model.aeroset.json>\n");
        return 1;
    }
    try {
        aero::AeroSetFile set(argv[1]);
        std::printf("Model : %s\n", set.model_name().c_str());
        std::printf("Arch  : %s\n", set.architecture().c_str());
        std::printf("Parts : %zu\n\n", set.parts().size());

        for (size_t i = 0; i < set.parts().size(); ++i) {
            auto& p = set.parts()[i];
            std::printf("  [%zu] %s  size=%llu  shards=[",
                        i, p.path.c_str(), (unsigned long long)p.size_bytes);
            for (size_t j = 0; j < p.shards.size(); ++j)
                std::printf("%s%d", j ? "," : "", p.shards[j]);
            std::printf("]\n");
        }

        auto& tensors = set.tensors();
        int limit = (argc >= 3) ? std::atoi(argv[2]) : (int)tensors.size();
        if (limit <= 0 || limit > (int)tensors.size()) limit = (int)tensors.size();
        std::printf("\nTensors (%zu, showing %d):\n", tensors.size(), limit);
        for (int i = 0; i < limit; ++i) {
            auto& t = tensors[i];
            std::printf("  %-40s  dtype=%-5s  shape=[",
                        t.name.c_str(), aero::dtype_name(t.dtype));
            for (size_t j = 0; j < t.shape.size(); ++j)
                std::printf("%s%lld", j ? "," : "", (long long)t.shape[j]);
            std::printf("]  shard=%d  bytes=%llu\n",
                        t.shard_id, (unsigned long long)t.data_len);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
