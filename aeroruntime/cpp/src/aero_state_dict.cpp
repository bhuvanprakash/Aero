/**
 * AERO state-dict loader: open .aero, dequant PACKED, optional GGUF→HF name mapping.
 * Exposes aero_load_state_dict, aero_state_dict_get_tensor, aero_state_dict_free.
 */
#include "aero/aero_capi.h"
#include "aero/aero.hpp"
#include "aero/aero_dequant.h"
#include "aero/aero_internal.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace aero;

void set_err(const char* msg) {
    aero_internal_set_error(msg);
}

// ── GGUF → HuggingFace name mapping (match Python loader) ──────────────────
static const std::unordered_map<std::string, std::string> gguf_global = {
    {"token_embd.weight", "model.embed_tokens.weight"},
    {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"},
};

static const std::unordered_map<std::string, std::string> gguf_layer = {
    {"attn_k", "self_attn.k_proj"},
    {"attn_q", "self_attn.q_proj"},
    {"attn_v", "self_attn.v_proj"},
    {"attn_output", "self_attn.o_proj"},
    {"attn_norm", "input_layernorm"},
    {"attn_k_norm", "self_attn.k_norm"},
    {"attn_q_norm", "self_attn.q_norm"},
    {"ffn_gate", "mlp.gate_proj"},
    {"ffn_up", "mlp.up_proj"},
    {"ffn_down", "mlp.down_proj"},
    {"ffn_norm", "post_attention_layernorm"},
};

std::string map_gguf_name(const std::string& name) {
    {
        auto it = gguf_global.find(name);
        if (it != gguf_global.end()) return it->second;
    }
    if (name.size() >= 4 && name.substr(0, 4) == "blk.") {
        size_t dot1 = name.find('.', 4);
        if (dot1 != std::string::npos) {
            std::string layer_id = name.substr(4, dot1 - 4);
            std::string rest = name.substr(dot1 + 1);
            for (const auto& p : gguf_layer) {
                if (rest == p.first + ".weight")
                    return "model.layers." + layer_id + "." + p.second + ".weight";
                if (rest == p.first + ".bias")
                    return "model.layers." + layer_id + "." + p.second + ".bias";
            }
            return "model.layers." + layer_id + "." + rest;
        }
    }
    return name;
}

// ── fp16/bf16 → float (for standard dtypes) ─────────────────────────────────
float fp16_to_f32(uint16_t u) {
    uint32_t sign = (u >> 15) & 1;
    uint32_t exp = (u >> 10) & 0x1f;
    uint32_t mant = u & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.f : 0.f;
        return (sign ? -1.f : 1.f) * std::ldexp(static_cast<float>(mant), -24);
    }
    if (exp == 31)
        return mant ? std::numeric_limits<float>::quiet_NaN() : (sign ? -INFINITY : INFINITY);
    return (sign ? -1.f : 1.f) * std::ldexp(1.f + mant / 1024.f, static_cast<int>(exp) - 15);
}

float bf16_to_f32(uint16_t u) {
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// ── One tensor in the state dict ────────────────────────────────────────────
struct StateDictTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;  // always float32
};

// ── State-dict implementation ───────────────────────────────────────────────
struct StateDict {
    std::vector<StateDictTensor> tensors;
};

} // namespace

extern "C" {

aero_state_dict_handle_t aero_load_state_dict(const char* path, int apply_gguf_name_mapping) {
    if (!path) {
        set_err("path is NULL");
        return nullptr;
    }
    try {
        aero::AeroFile file(path);
        std::vector<aero::TensorEntry> entries = file.load_tidx();
        auto sd = std::make_unique<StateDict>();
        sd->tensors.reserve(entries.size());

        for (const aero::TensorEntry& te : entries) {
            StateDictTensor t;
            t.name = apply_gguf_name_mapping ? map_gguf_name(te.name) : te.name;
            t.shape = te.shape;

            int64_t ne = 1;
            for (int64_t s : te.shape) ne *= s;
            if (ne <= 0) continue;
            size_t num_el = static_cast<size_t>(ne);

            if (te.dtype == static_cast<uint16_t>(aero::DType::DT_PACKED)) {
                if (!aero::is_supported_ggml_type(te.ggml_type)) {
                    set_err(("unsupported ggml_type: " + std::to_string(te.ggml_type)).c_str());
                    return nullptr;
                }
                std::vector<uint8_t> raw = file.read_tensor_bytes(te.name);
                size_t expected = aero::packed_byte_size(te.shape.data(), te.shape.size(), te.ggml_type);
                if (raw.size() < expected) {
                    set_err("PACKED tensor byte size mismatch");
                    return nullptr;
                }
                t.data.resize(num_el);
                if (!aero::dequantize(raw.data(), raw.size(), te.shape.data(), te.shape.size(),
                                     te.ggml_type, t.data.data())) {
                    set_err("dequantize failed");
                    return nullptr;
                }
            } else {
                t.data.resize(num_el);
                std::vector<uint8_t> raw = file.read_tensor_bytes(te.name);
                if (te.dtype == 1) {  // F16
                    if (raw.size() < num_el * 2) { set_err("F16 tensor too small"); return nullptr; }
                    for (size_t i = 0; i < num_el; i++)
                        t.data[i] = fp16_to_f32(read_u16(raw.data() + 2 * i));
                } else if (te.dtype == 2) {  // BF16
                    if (raw.size() < num_el * 2) { set_err("BF16 tensor too small"); return nullptr; }
                    for (size_t i = 0; i < num_el; i++)
                        t.data[i] = bf16_to_f32(read_u16(raw.data() + 2 * i));
                } else if (te.dtype == 0) {  // F32
                    if (raw.size() < num_el * 4) { set_err("F32 tensor too small"); return nullptr; }
                    std::memcpy(t.data.data(), raw.data(), num_el * 4);
                } else {
                    set_err(("unsupported dtype: " + std::to_string(te.dtype)).c_str());
                    return nullptr;
                }
            }

            sd->tensors.push_back(std::move(t));
        }

        return reinterpret_cast<aero_state_dict_handle_t>(sd.release());
    } catch (const std::exception& e) {
        set_err(e.what());
        return nullptr;
    } catch (...) {
        set_err("unknown error");
        return nullptr;
    }
}

uint32_t aero_state_dict_tensor_count(aero_state_dict_handle_t h) {
    if (!h) return 0;
    auto* sd = reinterpret_cast<StateDict*>(h);
    return static_cast<uint32_t>(sd->tensors.size());
}

int aero_state_dict_get_tensor(aero_state_dict_handle_t h, uint32_t i,
                               char* name_buf, size_t name_size,
                               int64_t* shape_out, uint32_t* rank_out,
                               aero_state_dict_dtype_t* dtype_out,
                               const void** data_ptr_out, uint64_t* size_bytes_out) {
    if (!h || !name_buf || !shape_out || !rank_out || !dtype_out || !data_ptr_out || !size_bytes_out) {
        aero_internal_set_error("NULL argument");
        return -1;
    }
    auto* sd = reinterpret_cast<StateDict*>(h);
    if (i >= sd->tensors.size()) {
        aero_internal_set_error("index out of range");
        return -1;
    }
    const StateDictTensor& t = sd->tensors[i];
    size_t name_len = t.name.size();
    if (name_len >= name_size) {
        aero_internal_set_error("name buffer too small");
        return -1;
    }
    std::memcpy(name_buf, t.name.data(), name_len);
    name_buf[name_len] = '\0';

    *rank_out = static_cast<uint32_t>(t.shape.size());
    for (size_t j = 0; j < t.shape.size(); j++)
        shape_out[j] = t.shape[j];

    *dtype_out = AERO_SD_F32;
    *data_ptr_out = t.data.data();
    *size_bytes_out = t.data.size() * sizeof(float);
    return 0;
}

void aero_state_dict_free(aero_state_dict_handle_t h) {
    if (!h) return;
    delete reinterpret_cast<StateDict*>(h);
}

} // extern "C"
