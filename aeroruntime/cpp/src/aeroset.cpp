// AERO v0.1 – AEROSET reader implementation
#include "aero/aeroset.hpp"

#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace aero {

namespace {

std::string dirname(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

} // anon

AeroSetFile::AeroSetFile(const std::string& index_json_path) {
    base_dir_ = dirname(index_json_path);

    std::ifstream ifs(index_json_path);
    if (!ifs) throw Error("cannot open " + index_json_path);
    json doc = json::parse(ifs);

    // Model info.
    if (doc.contains("model")) {
        auto& m = doc["model"];
        if (m.contains("name"))         model_name_ = m["name"].get<std::string>();
        if (m.contains("architecture")) arch_       = m["architecture"].get<std::string>();
    }

    // Parts.
    if (doc.contains("parts")) {
        for (auto& p : doc["parts"]) {
            AeroSetPartInfo pi;
            pi.path       = p.value("path", "");
            pi.sha256     = p.value("sha256", "");
            pi.size_bytes = p.value("size_bytes", uint64_t(0));
            if (p.contains("shards")) {
                for (auto& s : p["shards"])
                    pi.shards.push_back(s.get<int32_t>());
            }
            // Build shard → part mapping.
            for (auto sid : pi.shards)
                shard_to_part_[sid] = pi.path;
            parts_.push_back(std::move(pi));
        }
    }

    // Global index AERO.
    if (doc.contains("global_tidx") && doc["global_tidx"].contains("path")) {
        std::string idx_path = base_dir_ + "/" +
                               doc["global_tidx"]["path"].get<std::string>();
        index_aero_ = std::make_unique<AeroFile>(idx_path);
        tensors_ = index_aero_->load_tidx();
    }
}

std::string AeroSetFile::part_for_shard(int32_t shard_id) const {
    auto it = shard_to_part_.find(shard_id);
    if (it == shard_to_part_.end())
        throw Error("shard_id " + std::to_string(shard_id) + " not in any part");
    return it->second;
}

AeroFile& AeroSetFile::get_part(const std::string& rel_path) {
    auto it = part_cache_.find(rel_path);
    if (it != part_cache_.end()) return *it->second;
    std::string full = base_dir_ + "/" + rel_path;
    auto ptr = std::make_unique<AeroFile>(full);
    auto& ref = *ptr;
    part_cache_[rel_path] = std::move(ptr);
    return ref;
}

std::vector<uint8_t> AeroSetFile::read_tensor_bytes(const std::string& name) {
    // Find tensor in global TIDX.
    const TensorEntry* te = nullptr;
    for (auto& t : tensors_) {
        if (t.name == name) { te = &t; break; }
    }
    if (!te) throw Error("tensor not found: " + name);

    // Open the correct part and read.
    std::string rel = part_for_shard(te->shard_id);
    AeroFile& part = get_part(rel);
    return part.read_tensor_bytes(name);
}

} // namespace aero
