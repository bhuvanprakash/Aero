/**
 * AERO — GGUF dequantization (PACKED tensors to float32).
 * Supports common GGML types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K–Q8_K.
 */
#ifndef AERO_AERO_DEQUANT_H
#define AERO_AERO_DEQUANT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aero {

/** GGML quant type enum (matches GGUF / gguf.quants). */
enum class GGMLType : uint32_t {
    F32 = 0, F16 = 1,
    Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    F64 = 28, BF16 = 30,
};

/**
 * Dequantize packed tensor data to float32.
 *
 * @param data     Raw packed bytes (from AERO tensor).
 * @param size     Byte length (must match expected size for shape + ggml_type).
 * @param shape    Tensor shape (number of elements = product of shape).
 * @param rank     Number of dimensions.
 * @param ggml_type GGUF quant type (e.g. 2=Q4_0, 12=Q4_K).
 * @param out      Output buffer; must have at least (product of shape) floats.
 * @return true on success, false if ggml_type unsupported or size mismatch.
 */
bool dequantize(const uint8_t* data, size_t size,
                const int64_t* shape, size_t rank,
                uint32_t ggml_type,
                float* out);

/**
 * Same as dequantize but returns a vector. Returns empty vector on error.
 */
std::vector<float> dequantize_to_vector(const uint8_t* data, size_t size,
                                        const int64_t* shape, size_t rank,
                                        uint32_t ggml_type);

/** Return expected byte size for packed tensor (product of shape in quant blocks). 0 if unsupported. */
size_t packed_byte_size(const int64_t* shape, size_t rank, uint32_t ggml_type);

/** Return true if ggml_type is supported for dequantization. */
bool is_supported_ggml_type(uint32_t ggml_type);

} // namespace aero

#endif
