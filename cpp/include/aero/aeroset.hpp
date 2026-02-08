// AERO v0.1 – AEROSET (multi-file) reader
// Loads model.aeroset.json, opens parts lazily, reads tensors.
#pragma once

#include "aero.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aero {

struct AeroSetPartInfo {
    std::string          path;      // relative to aeroset dir
    std::string          sha256;
    uint64_t             size_bytes = 0;
    std::vector<int32_t> shards;    // shard_ids inside this part
};

class AeroSetFile {
public:
    /// Construct from path to model.aeroset.json.
    explicit AeroSetFile(const std::string& index_json_path);

    const std::string& model_name()   const { return model_name_; }
    const std::string& architecture() const { return arch_; }
    const std::vector<AeroSetPartInfo>& parts()   const { return parts_; }
    const std::vector<TensorEntry>&     tensors() const { return tensors_; }

    /// Read raw tensor bytes (copies data).
    std::vector<uint8_t> read_tensor_bytes(const std::string& name);

private:
    AeroFile& get_part(const std::string& rel_path);
    std::string part_for_shard(int32_t shard_id) const;

    std::string base_dir_;
    std::string model_name_;
    std::string arch_;
    std::vector<AeroSetPartInfo>  parts_;
    std::vector<TensorEntry>      tensors_;

    // Global index AERO file.
    std::unique_ptr<AeroFile>     index_aero_;

    // shard_id → part relative path
    std::unordered_map<int32_t, std::string> shard_to_part_;

    // Cached open parts.
    std::unordered_map<std::string, std::unique_ptr<AeroFile>> part_cache_;
};

} // namespace aero
