/**
 * AERO Runtime — Minimal C ABI
 *
 * Single-header C API for loading .aero files and accessing tensor data
 * with zero copy. Any language can bind via FFI without linking C++.
 *
 * See docs/RUNTIME_SPEC.md for full specification.
 */
#ifndef AERO_AERO_CAPI_H
#define AERO_AERO_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(AERO_CAPI_EXPORT)
#  define AERO_CAPI_API __declspec(dllexport)
#elif defined(_WIN32)
#  define AERO_CAPI_API __declspec(dllimport)
#else
#  define AERO_CAPI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to an opened AERO file. */
typedef struct aero_handle* aero_handle_t;

/** DType enum for tensor element type (matches AERO format). */
typedef enum {
    AERO_DT_F16 = 0,  AERO_DT_F32 = 1,  AERO_DT_BF16 = 2,  AERO_DT_F64 = 3,
    AERO_DT_I8 = 4,   AERO_DT_U8 = 5,   AERO_DT_I16 = 6,   AERO_DT_U16 = 7,
    AERO_DT_I32 = 8,   AERO_DT_U32 = 9,  AERO_DT_I64 = 10, AERO_DT_U64 = 11,
    AERO_DT_BOOL = 12,
    AERO_DT_PACKED = 0x8000
} aero_dtype_t;

/**
 * Open an AERO file and parse header + TOC.
 * Returns a handle on success; on failure returns NULL and sets error string.
 */
AERO_CAPI_API aero_handle_t aero_load(const char* path);

/**
 * Close the file and release all resources (including mmap'd regions).
 * Safe to call with NULL (no-op).
 */
AERO_CAPI_API void aero_close(aero_handle_t h);

/**
 * Number of tensors in the TIDX index.
 * Returns 0 if no TIDX or on error.
 */
AERO_CAPI_API uint32_t aero_tensor_count(aero_handle_t h);

/**
 * Write the tensor name for index i (0-based) into buffer as null-terminated UTF-8.
 * At most buffer_size bytes are written (including '\\0').
 * Returns 0 on success, non-zero if index out of range or buffer too small.
 */
AERO_CAPI_API int aero_tensor_name(aero_handle_t h, uint32_t i, char* buffer, size_t buffer_size);

/**
 * Get dtype, shape, and byte length for a tensor by name.
 * shape_out must point to an array of at least 8 int64_t; rank is written to *rank_out.
 * Returns 0 on success, non-zero if tensor not found.
 */
AERO_CAPI_API int aero_tensor_info(aero_handle_t h, const char* name,
                     aero_dtype_t* dtype_out, int64_t* shape_out, uint32_t* rank_out,
                     uint64_t* data_len_out);

/**
 * Return a pointer to the raw tensor data (read-only).
 * Valid until aero_close(h). No copy; data may be in a memory-mapped region.
 * Returns 0 on success and sets *data_out and *size_out; non-zero if tensor not found.
 */
AERO_CAPI_API int aero_tensor_ptr(aero_handle_t h, const char* name,
                    const uint8_t** data_out, uint64_t* size_out);

/**
 * Copy tensor bytes into a user-provided buffer.
 * buffer_size must be at least the tensor's byte length (use aero_tensor_info).
 * Returns 0 on success; non-zero if tensor not found or buffer too small.
 */
AERO_CAPI_API int aero_tensor_bytes(aero_handle_t h, const char* name,
                      uint8_t* buffer, uint64_t buffer_size);

/**
 * Return the last error message (UTF-8, null-terminated).
 * Valid until the next API call on any handle. Never returns NULL.
 */
AERO_CAPI_API const char* aero_error_string(void);

/* -------------------------------------------------------------------------
 * State-dict API (load + dequant to float, optional GGUF→HF name mapping)
 * ------------------------------------------------------------------------- */

/** Opaque handle to a loaded state dict (all tensors in memory as float). */
typedef struct aero_state_dict* aero_state_dict_handle_t;

/** Output dtype for aero_state_dict_get_tensor. */
typedef enum {
    AERO_SD_F16 = 0,
    AERO_SD_F32 = 1
} aero_state_dict_dtype_t;

/**
 * Load full state dict: open file, dequant PACKED tensors, optionally map GGUF names to HuggingFace.
 * Returns handle or NULL on error; use aero_error_string() for message.
 */
AERO_CAPI_API aero_state_dict_handle_t aero_load_state_dict(const char* path, int apply_gguf_name_mapping);

/**
 * Number of tensors in the state dict. Returns 0 if h is NULL.
 */
AERO_CAPI_API uint32_t aero_state_dict_tensor_count(aero_state_dict_handle_t h);

/**
 * Get tensor at index i (0-based). Fills name (null-terminated), shape_out[0..rank_out-1],
 * rank_out, dtype (AERO_SD_F32 or AERO_SD_F16), pointer to data, and size in bytes.
 * shape_out must point to an array of at least rank_out elements (and at least 8).
 * All pointers valid until aero_state_dict_free(h).
 * Returns 0 on success, non-zero if index out of range or buffer too small.
 */
AERO_CAPI_API int aero_state_dict_get_tensor(aero_state_dict_handle_t h, uint32_t i,
    char* name_buf, size_t name_size,
    int64_t* shape_out, uint32_t* rank_out,
    aero_state_dict_dtype_t* dtype_out,
    const void** data_ptr_out, uint64_t* size_bytes_out);

/**
 * Free state-dict handle and all buffers. No-op if h is NULL.
 */
AERO_CAPI_API void aero_state_dict_free(aero_state_dict_handle_t h);

#ifdef __cplusplus
}
#endif

#endif /* AERO_AERO_CAPI_H */
