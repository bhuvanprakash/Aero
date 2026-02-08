// AERO v0.2 – C++ reader library
// Cross-platform: macOS, Linux, Windows (MSVC)
#pragma once

#define AERO_VERSION_MAJOR 0
#define AERO_VERSION_MINOR 2
#define AERO_VERSION_PATCH 0

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aero {

// ── Error type ─────────────────────────────────────────────────────────────

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ── Packed binary structs (match Python writer exactly) ────────────────────

#pragma pack(push, 1)

struct FileHeaderV1 {
    char     magic[4];              //  0: "AERO"
    uint16_t version_major;         //  4
    uint16_t version_minor;         //  6
    uint32_t header_size;           //  8
    uint64_t toc_offset;            // 12
    uint64_t toc_length;            // 20
    uint64_t string_table_offset;   // 28
    uint64_t string_table_length;   // 36
    uint64_t file_flags;            // 44
    uint8_t  uuid[16];              // 52
    uint8_t  reserved[28];          // 68
};                                  // 96

struct TocHeaderV1 {
    uint32_t entry_count;           //  0
    uint32_t reserved0;             //  4
    uint64_t reserved1;             //  8
};                                  // 16

struct TocEntryV1 {
    char     fourcc[4];             //  0
    uint32_t chunk_flags;           //  4
    uint64_t chunk_offset;          //  8
    uint64_t chunk_length;          // 16
    uint64_t chunk_ulen;            // 24
    uint32_t name_off;              // 32
    uint32_t name_len;              // 36
    uint64_t reserved0;             // 40
    uint8_t  blake3_256[32];        // 48
};                                  // 80

#pragma pack(pop)

static_assert(sizeof(FileHeaderV1) == 96, "FileHeaderV1 must be 96 bytes");
static_assert(sizeof(TocHeaderV1)  == 16, "TocHeaderV1 must be 16 bytes");
static_assert(sizeof(TocEntryV1)   == 80, "TocEntryV1 must be 80 bytes");

// ── Enums ──────────────────────────────────────────────────────────────────

enum ChunkFlag : uint32_t {
    COMPRESSED_ZSTD        = 0x0001,
    CHUNK_IS_MMAP_CRITICAL = 0x0002,
    CHUNK_IS_INDEX         = 0x0004,
    CHUNK_IS_OPTIONAL      = 0x0008,
};

enum DType : uint16_t {
    DT_F16  = 0,  DT_F32  = 1,  DT_BF16 = 2,  DT_F64  = 3,
    DT_I8   = 4,  DT_U8   = 5,  DT_I16  = 6,  DT_U16  = 7,
    DT_I32  = 8,  DT_U32  = 9,  DT_I64  = 10, DT_U64  = 11,
    DT_BOOL = 12,
    DT_PACKED = 0x8000,
};

inline const char* dtype_name(uint16_t d) {
    switch (d) {
        case DT_F16:  return "f16";  case DT_F32:  return "f32";
        case DT_BF16: return "bf16"; case DT_F64:  return "f64";
        case DT_I8:   return "i8";   case DT_U8:   return "u8";
        case DT_I16:  return "i16";  case DT_U16:  return "u16";
        case DT_I32:  return "i32";  case DT_U32:  return "u32";
        case DT_I64:  return "i64";  case DT_U64:  return "u64";
        case DT_BOOL: return "bool"; case DT_PACKED: return "packed";
        default:      return "?";
    }
}

// ── Data structures ────────────────────────────────────────────────────────

struct ChunkRef {
    std::string fourcc;         // 4-char tag
    std::string name;
    uint32_t    flags  = 0;
    uint64_t    offset = 0;
    uint64_t    length = 0;     // stored (possibly compressed) length
    uint64_t    ulen   = 0;     // uncompressed length
    uint8_t     blake3[32] = {};
};

struct TensorEntry {
    std::string            name;
    uint16_t               dtype    = 0;
    std::vector<int64_t>   shape;
    int32_t                shard_id = 0;
    uint64_t               data_off = 0;
    uint64_t               data_len = 0;
    uint32_t               flags    = 0;
};

// ── File I/O abstraction ───────────────────────────────────────────────────

class FileHandle {
public:
    FileHandle() = default;
    ~FileHandle();
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    void     open(const std::string& path);
    uint64_t size() const { return size_; }
    void     pread(void* dst, size_t n, uint64_t off) const;

private:
#ifdef _WIN32
    void* handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(-1));
#else
    int fd_ = -1;
#endif
    uint64_t size_ = 0;
};

// ── Memory-mapped region (handles unaligned offsets) ───────────────────────

class MMapRegion {
public:
    MMapRegion() = default;
    ~MMapRegion();
    MMapRegion(const MMapRegion&) = delete;
    MMapRegion& operator=(const MMapRegion&) = delete;
    MMapRegion(MMapRegion&& o) noexcept;
    MMapRegion& operator=(MMapRegion&& o) noexcept;

    /// Map [offset, offset+length) from the file at *path*.
    /// Aligns the OS mapping base internally.
    static MMapRegion map(const std::string& path,
                          uint64_t offset, uint64_t length);

    const uint8_t* data() const { return data_; }
    uint64_t       size() const { return size_; }

private:
    void unmap();

#ifdef _WIN32
    void* file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(-1));
    void* mapping_     = nullptr;
#else
    int fd_ = -1;
#endif
    void*          base_ptr_ = nullptr;
    size_t         map_len_  = 0;
    const uint8_t* data_     = nullptr;
    uint64_t       size_     = 0;
};

// ── AeroFile ───────────────────────────────────────────────────────────────

class AeroFile {
public:
    explicit AeroFile(const std::string& path);
    ~AeroFile() = default;

    // Header access.
    const FileHeaderV1&       header()  const { return header_; }
    const std::vector<ChunkRef>& chunks() const { return chunks_; }

    // Chunk lookup.
    const ChunkRef* find_chunk(const std::string& fourcc) const;
    const ChunkRef* find_chunk_named(const std::string& fourcc,
                                     const std::string& name) const;

    // Read chunk bytes (decompress + optional BLAKE3 verify).
    std::vector<uint8_t> read_chunk_bytes(const ChunkRef& ref,
                                          bool verify = true) const;

    // Stream-verify a (possibly large) chunk via BLAKE3 without loading
    // the whole payload into RAM.
    bool stream_verify_chunk(const ChunkRef& ref,
                             size_t block_size = 64 * 1024 * 1024) const;

    // Tensor index.
    std::vector<TensorEntry> load_tidx() const;

    // Tensor data access.
    std::vector<uint8_t> read_tensor_bytes(const std::string& name);

    struct TensorView {
        std::shared_ptr<MMapRegion> region;
        const uint8_t* data;
        uint64_t       size;
    };
    TensorView view_tensor(const std::string& name);

    // Manifest info.
    struct ManifestInfo {
        std::string model_name;
        std::string architecture;
    };
    ManifestInfo load_manifest() const;

private:
    void parse();
    void load_tidx_if_needed();

    std::string          path_;
    FileHandle           file_;
    FileHeaderV1         header_{};
    std::vector<ChunkRef> chunks_;

    // Cached tensor index.
    bool                      tidx_loaded_ = false;
    std::vector<TensorEntry>  tensors_;

    // Cached shard mappings: shard_id → shared region.
    std::mutex                                        mmap_mu_;
    std::unordered_map<int32_t,
                       std::shared_ptr<MMapRegion>>   shard_cache_;
};

} // namespace aero
