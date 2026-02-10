// aero_inspect – print header, chunks, and optionally model info.

#include "aero/aero.hpp"
#include <cstdio>
#include <cinttypes>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: aero_inspect <file.aero>\n");
        return 1;
    }

    try {
        aero::AeroFile file(argv[1]);
        const auto& hdr = file.header();

        // UUID
        std::printf("UUID    : ");
        for (int i = 0; i < 16; ++i) std::printf("%02x", hdr.uuid[i]);
        std::printf("\n");

        std::printf("Version : %u.%u\n", hdr.version_major, hdr.version_minor);
        std::printf("Flags   : 0x%016" PRIx64 "\n", hdr.file_flags);
        std::printf("TOC     : offset=%" PRIu64 "  length=%" PRIu64 "\n",
                    hdr.toc_offset, hdr.toc_length);
        std::printf("StrTab  : offset=%" PRIu64 "  length=%" PRIu64 "\n",
                    hdr.string_table_offset, hdr.string_table_length);

        const auto& chunks = file.chunks();
        std::printf("\nChunks (%zu):\n", chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto& c = chunks[i];
            std::printf("  [%zu] %-4s  name=%-24s  off=%-10" PRIu64
                        "  len=%-10" PRIu64 "  ulen=%-10" PRIu64
                        "  flags=0x%04x\n",
                        i, c.fourcc.c_str(), c.name.c_str(),
                        c.offset, c.length, c.ulen, c.flags);
        }

        // Try to load manifest for model info.
        try {
            auto m = file.load_manifest();
            if (!m.model_name.empty()) {
                std::printf("\nModel   : %s\n", m.model_name.c_str());
                std::printf("Arch    : %s\n", m.architecture.c_str());
            }
        } catch (...) {}

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
