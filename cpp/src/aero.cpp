// AERO v0.1 – C++ reader implementation
// Handles: parsing, zstd decompression, BLAKE3 verification, mmap, msgpack.

#include "aero/aero.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#include <blake3.h>
#include <msgpack.hpp>
#include <zstd.h>

namespace aero {

// ── Helpers ────────────────────────────────────────────────────────────────

[[noreturn]] static void fail(const std::string& msg) {
    throw Error("aero: " + msg);
}

static std::string fourcc_to_str(const char* p) {
    return std::string(p, 4);
}

static std::vector<uint8_t> blake3_hash(const uint8_t* data, size_t len) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, len);
    std::vector<uint8_t> out(32);
    blake3_hasher_finalize(&hasher, out.data(), 32);
    return out;
}

// Safely extract an integer from a msgpack object.
static int64_t mp_to_i64(const msgpack::object& o) {
    if (o.type == msgpack::type::POSITIVE_INTEGER)
        return static_cast<int64_t>(o.via.u64);
    if (o.type == msgpack::type::NEGATIVE_INTEGER)
        return o.via.i64;
    fail("expected integer in msgpack");
}

static uint64_t mp_to_u64(const msgpack::object& o) {
    if (o.type == msgpack::type::POSITIVE_INTEGER)
        return o.via.u64;
    if (o.type == msgpack::type::NEGATIVE_INTEGER && o.via.i64 >= 0)
        return static_cast<uint64_t>(o.via.i64);
    fail("expected non-negative integer in msgpack");
}

static std::string mp_to_str(const msgpack::object& o) {
    if (o.type == msgpack::type::STR)
        return std::string(o.via.str.ptr, o.via.str.size);
    if (o.type == msgpack::type::BIN)
        return std::string(o.via.bin.ptr, o.via.bin.size);
    fail("expected string in msgpack");
}

// ── FileHandle (POSIX) ────────────────────────────────────────────────────

#ifndef _WIN32

FileHandle::~FileHandle() {
    if (fd_ >= 0) ::close(fd_);
}

void FileHandle::open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) fail("cannot open file: " + path);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) fail("fstat failed");
    size_ = static_cast<uint64_t>(st.st_size);
}

void FileHandle::pread(void* dst, size_t n, uint64_t off) const {
    auto* p = static_cast<uint8_t*>(dst);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = ::pread(fd_, p, remaining, static_cast<off_t>(off));
        if (r <= 0) fail("pread failed");
        p += r;
        off += static_cast<uint64_t>(r);
        remaining -= static_cast<size_t>(r);
    }
}

#else // ── FileHandle (Windows) ────────────────────────────────────────────

static constexpr void* INVALID = reinterpret_cast<void*>(static_cast<intptr_t>(-1));

FileHandle::~FileHandle() {
    if (handle_ != INVALID) CloseHandle(handle_);
}

void FileHandle::open(const std::string& path) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    handle_ = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                          nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) fail("cannot open file: " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(handle_, &sz)) fail("GetFileSizeEx failed");
    size_ = static_cast<uint64_t>(sz.QuadPart);
}

void FileHandle::pread(void* dst, size_t n, uint64_t off) const {
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(off & 0xFFFFFFFF);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    DWORD nread = 0;
    if (!ReadFile(handle_, dst, static_cast<DWORD>(n), &nread, &ov)
        || nread != static_cast<DWORD>(n))
        fail("ReadFile/pread failed");
}

#endif

// ── MMapRegion ─────────────────────────────────────────────────────────────

MMapRegion::~MMapRegion() { unmap(); }

MMapRegion::MMapRegion(MMapRegion&& o) noexcept { *this = std::move(o); }

MMapRegion& MMapRegion::operator=(MMapRegion&& o) noexcept {
    if (this != &o) {
        unmap();
#ifdef _WIN32
        file_handle_ = o.file_handle_; o.file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(-1));
        mapping_     = o.mapping_;     o.mapping_     = nullptr;
#else
        fd_ = o.fd_;   o.fd_ = -1;
#endif
        base_ptr_ = o.base_ptr_; o.base_ptr_ = nullptr;
        map_len_  = o.map_len_;  o.map_len_  = 0;
        data_     = o.data_;     o.data_     = nullptr;
        size_     = o.size_;     o.size_     = 0;
    }
    return *this;
}

void MMapRegion::unmap() {
#ifdef _WIN32
    if (base_ptr_) UnmapViewOfFile(base_ptr_);
    if (mapping_) CloseHandle(mapping_);
    if (file_handle_ != reinterpret_cast<void*>(static_cast<intptr_t>(-1)))
        CloseHandle(file_handle_);
    file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(-1));
    mapping_ = nullptr;
#else
    if (base_ptr_ && base_ptr_ != MAP_FAILED)
        ::munmap(base_ptr_, map_len_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ptr_ = nullptr;
    data_     = nullptr;
    map_len_  = 0;
    size_     = 0;
}

MMapRegion MMapRegion::map(const std::string& path,
                           uint64_t offset, uint64_t length) {
    MMapRegion r;
    r.size_ = length;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uint64_t gran = si.dwAllocationGranularity;

    uint64_t base_off = (offset / gran) * gran;
    uint64_t delta    = offset - base_off;
    size_t   mlen     = static_cast<size_t>(delta + length);

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    r.file_handle_ = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
    if (r.file_handle_ == INVALID_HANDLE_VALUE)
        fail("mmap: cannot open file");
    r.mapping_ = CreateFileMappingW(r.file_handle_, nullptr, PAGE_READONLY,
                                    0, 0, nullptr);
    if (!r.mapping_) fail("mmap: CreateFileMapping failed");

    DWORD off_hi = static_cast<DWORD>(base_off >> 32);
    DWORD off_lo = static_cast<DWORD>(base_off & 0xFFFFFFFF);
    r.base_ptr_ = MapViewOfFile(r.mapping_, FILE_MAP_READ,
                                off_hi, off_lo, mlen);
    if (!r.base_ptr_) fail("mmap: MapViewOfFile failed");

    r.map_len_ = mlen;
    r.data_    = static_cast<const uint8_t*>(r.base_ptr_) + delta;
#else
    long page_size = ::sysconf(_SC_PAGESIZE);
    uint64_t gran  = static_cast<uint64_t>(page_size);

    uint64_t base_off = (offset / gran) * gran;
    uint64_t delta    = offset - base_off;
    size_t   mlen     = static_cast<size_t>(delta + length);

    r.fd_ = ::open(path.c_str(), O_RDONLY);
    if (r.fd_ < 0) fail("mmap: cannot open file");

    r.base_ptr_ = ::mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE,
                         r.fd_, static_cast<off_t>(base_off));
    if (r.base_ptr_ == MAP_FAILED) {
        ::close(r.fd_);
        r.fd_ = -1;
        fail("mmap: mmap failed");
    }

    r.map_len_ = mlen;
    r.data_    = static_cast<const uint8_t*>(r.base_ptr_) + delta;
#endif

    return r;
}

// ── AeroFile ───────────────────────────────────────────────────────────────

AeroFile::AeroFile(const std::string& path) : path_(path) {
    file_.open(path);
    parse();
}

void AeroFile::parse() {
    const uint64_t fsize = file_.size();
    if (fsize < sizeof(FileHeaderV1))
        fail("file too small for header");

    file_.pread(&header_, sizeof(header_), 0);

    if (std::memcmp(header_.magic, "AERO", 4) != 0)
        fail("bad magic");
    if (header_.version_major != 0 || header_.version_minor != 1)
        fail("unsupported version");
    if (header_.header_size != 96)
        fail("header_size mismatch");
    if (header_.file_flags != 0)
        fail("file_flags must be 0 (unsupported value)");

    // TOC bounds.
    if (header_.toc_offset + header_.toc_length > fsize)
        fail("TOC extends past end of file");
    if (header_.toc_length < sizeof(TocHeaderV1))
        fail("TOC too small");

    TocHeaderV1 toc_hdr{};
    file_.pread(&toc_hdr, sizeof(toc_hdr), header_.toc_offset);

    uint32_t n = toc_hdr.entry_count;
    if (n > 1'000'000) fail("insane entry_count");

    uint64_t need = sizeof(TocHeaderV1)
                  + static_cast<uint64_t>(n) * sizeof(TocEntryV1);
    if (header_.toc_length < need)
        fail("TOC length too small for entries");

    // String table.
    if (header_.string_table_offset + header_.string_table_length > fsize)
        fail("string table extends past EOF");
    if (header_.string_table_length > 512ULL * 1024 * 1024)
        fail("string table too large");

    std::vector<uint8_t> st(header_.string_table_length);
    if (!st.empty())
        file_.pread(st.data(), st.size(), header_.string_table_offset);

    // TOC entries.
    chunks_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        TocEntryV1 entry{};
        uint64_t eoff = header_.toc_offset + sizeof(TocHeaderV1)
                      + static_cast<uint64_t>(i) * sizeof(TocEntryV1);
        file_.pread(&entry, sizeof(entry), eoff);

        if (entry.chunk_offset + entry.chunk_length > fsize)
            fail("chunk " + std::to_string(i) + " extends past EOF");
        if (entry.name_off + entry.name_len > header_.string_table_length)
            fail("chunk " + std::to_string(i) + " name exceeds string table");

        ChunkRef ref;
        ref.fourcc = fourcc_to_str(entry.fourcc);
        ref.name   = std::string(
            reinterpret_cast<const char*>(st.data() + entry.name_off),
            entry.name_len);
        ref.flags  = entry.chunk_flags;
        ref.offset = entry.chunk_offset;
        ref.length = entry.chunk_length;
        ref.ulen   = entry.chunk_ulen;
        std::memcpy(ref.blake3, entry.blake3_256, 32);

        chunks_.push_back(std::move(ref));
    }
}

const ChunkRef* AeroFile::find_chunk(const std::string& fourcc) const {
    for (auto& c : chunks_)
        if (c.fourcc == fourcc) return &c;
    return nullptr;
}

const ChunkRef* AeroFile::find_chunk_named(const std::string& fourcc,
                                           const std::string& name) const {
    for (auto& c : chunks_)
        if (c.fourcc == fourcc && c.name == name) return &c;
    return nullptr;
}

// ── Chunk reading + decompression + verification ──────────────────────────

std::vector<uint8_t> AeroFile::read_chunk_bytes(const ChunkRef& ref,
                                                bool verify) const {
    std::vector<uint8_t> stored(ref.length);
    file_.pread(stored.data(), stored.size(), ref.offset);

    std::vector<uint8_t> data;

    if (ref.flags & COMPRESSED_ZSTD) {
        data.resize(ref.ulen);
        size_t rc = ZSTD_decompress(data.data(), data.size(),
                                    stored.data(), stored.size());
        if (ZSTD_isError(rc))
            fail(std::string("zstd: ") + ZSTD_getErrorName(rc));
        if (rc != ref.ulen)
            fail("zstd decompressed size mismatch");
    } else {
        data = std::move(stored);
    }

    if (verify) {
        auto h = blake3_hash(data.data(), data.size());
        if (std::memcmp(h.data(), ref.blake3, 32) != 0)
            fail("BLAKE3 mismatch for " + ref.fourcc + " (" + ref.name + ")");
    }

    return data;
}

bool AeroFile::stream_verify_chunk(const ChunkRef& ref,
                                   size_t block_size) const {
    // Compressed chunks must be fully decompressed for BLAKE3 verification.
    if (ref.flags & COMPRESSED_ZSTD) {
        try {
            read_chunk_bytes(ref, true);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Stream-hash uncompressed chunks in blocks.
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);

    uint64_t remaining = ref.length;
    uint64_t off       = ref.offset;
    std::vector<uint8_t> buf(block_size);

    while (remaining > 0) {
        size_t to_read = static_cast<size_t>(
            std::min(static_cast<uint64_t>(block_size), remaining));
        file_.pread(buf.data(), to_read, off);
        blake3_hasher_update(&hasher, buf.data(), to_read);
        off       += to_read;
        remaining -= to_read;
    }

    uint8_t hash[32];
    blake3_hasher_finalize(&hasher, hash, 32);
    return std::memcmp(hash, ref.blake3, 32) == 0;
}

// ── TIDX parsing ──────────────────────────────────────────────────────────

std::vector<TensorEntry> AeroFile::load_tidx() const {
    auto* ref = find_chunk("TIDX");
    if (!ref) fail("TIDX chunk not found");

    auto data = read_chunk_bytes(*ref, true);

    msgpack::object_handle oh = msgpack::unpack(
        reinterpret_cast<const char*>(data.data()), data.size());
    const msgpack::object& root = oh.get();

    if (root.type != msgpack::type::MAP)
        fail("TIDX: root must be map");

    std::vector<TensorEntry> tensors;
    const auto& rmap = root.via.map;

    for (uint32_t i = 0; i < rmap.size; ++i) {
        std::string key = mp_to_str(rmap.ptr[i].key);
        if (key != "tensors") continue;

        const auto& arr_obj = rmap.ptr[i].val;
        if (arr_obj.type != msgpack::type::ARRAY)
            fail("TIDX: 'tensors' must be array");

        const auto& arr = arr_obj.via.array;
        if (arr.size > 100'000'000)
            fail("TIDX: insane tensor count");

        tensors.reserve(arr.size);
        for (uint32_t j = 0; j < arr.size; ++j) {
            const auto& tobj = arr.ptr[j];
            if (tobj.type != msgpack::type::MAP)
                fail("TIDX: tensor entry must be map");

            TensorEntry te;
            const auto& tmap = tobj.via.map;
            for (uint32_t k = 0; k < tmap.size; ++k) {
                std::string tkey = mp_to_str(tmap.ptr[k].key);
                const auto& val = tmap.ptr[k].val;

                if (tkey == "name") {
                    te.name = mp_to_str(val);
                    if (te.name.size() > 4096)
                        fail("TIDX: tensor name too long");
                } else if (tkey == "dtype") {
                    te.dtype = static_cast<uint16_t>(mp_to_u64(val));
                } else if (tkey == "shape") {
                    if (val.type != msgpack::type::ARRAY)
                        fail("TIDX: shape must be array");
                    if (val.via.array.size > 32)
                        fail("TIDX: rank too high");
                    te.shape.resize(val.via.array.size);
                    for (uint32_t s = 0; s < val.via.array.size; ++s)
                        te.shape[s] = mp_to_i64(val.via.array.ptr[s]);
                } else if (tkey == "shard_id") {
                    te.shard_id = static_cast<int32_t>(mp_to_i64(val));
                } else if (tkey == "data_off") {
                    te.data_off = mp_to_u64(val);
                } else if (tkey == "data_len") {
                    te.data_len = mp_to_u64(val);
                } else if (tkey == "flags") {
                    te.flags = static_cast<uint32_t>(mp_to_u64(val));
                }
                // Unknown keys silently ignored (forward-compat).
            }
            tensors.push_back(std::move(te));
        }
        // Reject duplicate tensor names (conformance with Python writer/reader).
        std::set<std::string> seen;
        for (const auto& t : tensors) {
            if (!seen.insert(t.name).second)
                fail("TIDX: duplicate tensor name: " + t.name);
        }
        break;
    }

    return tensors;
}

// ── Manifest parsing ──────────────────────────────────────────────────────

AeroFile::ManifestInfo AeroFile::load_manifest() const {
    auto* ref = find_chunk("MMSG");
    if (!ref) fail("MMSG chunk not found");

    auto data = read_chunk_bytes(*ref, true);

    msgpack::object_handle oh = msgpack::unpack(
        reinterpret_cast<const char*>(data.data()), data.size());
    const msgpack::object& root = oh.get();

    ManifestInfo info;
    if (root.type != msgpack::type::MAP) return info;

    const auto& rmap = root.via.map;
    for (uint32_t i = 0; i < rmap.size; ++i) {
        std::string key = mp_to_str(rmap.ptr[i].key);
        const auto& val = rmap.ptr[i].val;

        if (key == "model" && val.type == msgpack::type::MAP) {
            const auto& mmap = val.via.map;
            for (uint32_t j = 0; j < mmap.size; ++j) {
                std::string mk = mp_to_str(mmap.ptr[j].key);
                if (mk == "name")
                    info.model_name = mp_to_str(mmap.ptr[j].val);
                else if (mk == "architecture")
                    info.architecture = mp_to_str(mmap.ptr[j].val);
            }
        }
    }
    return info;
}

// ── Tensor data access ────────────────────────────────────────────────────

void AeroFile::load_tidx_if_needed() {
    if (!tidx_loaded_) {
        tensors_     = load_tidx();
        tidx_loaded_ = true;
    }
}

std::vector<uint8_t> AeroFile::read_tensor_bytes(const std::string& name) {
    load_tidx_if_needed();

    const TensorEntry* te = nullptr;
    for (auto& t : tensors_)
        if (t.name == name) { te = &t; break; }
    if (!te) fail("tensor not found: " + name);

    std::string shard_name = "weights.shard" + std::to_string(te->shard_id);
    auto* shard = find_chunk_named("WTSH", shard_name);
    if (!shard) fail("weight shard not found: " + shard_name);

    if (te->data_off + te->data_len > shard->length)
        fail("tensor data exceeds shard bounds");

    uint64_t abs_off = shard->offset + te->data_off;
    std::vector<uint8_t> buf(te->data_len);
    file_.pread(buf.data(), buf.size(), abs_off);
    return buf;
}

AeroFile::TensorView AeroFile::view_tensor(const std::string& name) {
    load_tidx_if_needed();

    const TensorEntry* te = nullptr;
    for (auto& t : tensors_)
        if (t.name == name) { te = &t; break; }
    if (!te) fail("tensor not found: " + name);

    std::string shard_name = "weights.shard" + std::to_string(te->shard_id);
    auto* shard = find_chunk_named("WTSH", shard_name);
    if (!shard) fail("weight shard not found: " + shard_name);

    if (te->data_off + te->data_len > shard->length)
        fail("tensor data exceeds shard bounds");

    // Check / populate shard cache.
    std::shared_ptr<MMapRegion> region;
    {
        std::lock_guard<std::mutex> lock(mmap_mu_);
        auto it = shard_cache_.find(te->shard_id);
        if (it != shard_cache_.end()) {
            region = it->second;
        } else {
            auto r = std::make_shared<MMapRegion>(
                MMapRegion::map(path_, shard->offset, shard->length));
            shard_cache_[te->shard_id] = r;
            region = r;
        }
    }

    TensorView view;
    view.region = region;
    view.data   = region->data() + te->data_off;
    view.size   = te->data_len;
    return view;
}

} // namespace aero
