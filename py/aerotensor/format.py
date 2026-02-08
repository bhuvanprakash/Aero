"""AERO v0.1 binary format constants, structs, and alignment helpers."""

import struct
from enum import IntEnum, IntFlag

# ── Magic & version ─────────────────────────────────────────────────────────

MAGIC = b"AERO"
VERSION_MAJOR = 0
VERSION_MINOR = 1

# ── Fixed sizes (bytes) ────────────────────────────────────────────────────

HEADER_SIZE = 96
TOC_HEADER_SIZE = 16
TOC_ENTRY_SIZE = 80

# ── Struct formats (little-endian) ──────────────────────────────────────────
#
# Header (96 B):
#   magic[4]  ver_major(u16) ver_minor(u16) header_size(u32)
#   toc_offset(u64) toc_length(u64)
#   string_table_offset(u64) string_table_length(u64)
#   file_flags(u64)
#   uuid[16]
#   reserved[28]

HEADER_FMT = "<4sHHIQQQQQ16s28s"
TOC_HEADER_FMT = "<IIQ"             # entry_count, reserved0, reserved1
TOC_ENTRY_FMT = "<4sIQQQIIQ32s"     # fourcc, flags, off, len, ulen,
                                     # name_off, name_len, reserved0, blake3

assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
assert struct.calcsize(TOC_HEADER_FMT) == TOC_HEADER_SIZE
assert struct.calcsize(TOC_ENTRY_FMT) == TOC_ENTRY_SIZE

# ── Chunk flags (u32) ──────────────────────────────────────────────────────


class ChunkFlags(IntFlag):
    COMPRESSED_ZSTD = 0x0001
    CHUNK_IS_MMAP_CRITICAL = 0x0002
    CHUNK_IS_INDEX = 0x0004
    CHUNK_IS_OPTIONAL = 0x0008


# ── Dtype enum (u16) ───────────────────────────────────────────────────────


class DType(IntEnum):
    F16 = 0
    F32 = 1
    BF16 = 2
    F64 = 3
    I8 = 4
    U8 = 5
    I16 = 6
    U16 = 7
    I32 = 8
    U32 = 9
    I64 = 10
    U64 = 11
    BOOL = 12
    PACKED = 0x8000   # quant/codec payload; full codec support in Prompt 3


DTYPE_NAMES: dict[DType, str] = {d: d.name.lower() for d in DType}

DTYPE_SIZES: dict[DType, int] = {
    DType.F16: 2,  DType.F32: 4,  DType.BF16: 2, DType.F64: 8,
    DType.I8: 1,   DType.U8: 1,   DType.I16: 2,  DType.U16: 2,
    DType.I32: 4,  DType.U32: 4,  DType.I64: 8,  DType.U64: 8,
    DType.BOOL: 1,
}

# ── FourCC constants ────────────────────────────────────────────────────────

FOURCC_MJSN = b"MJSN"   # optional human-readable JSON metadata
FOURCC_MMSG = b"MMSG"    # manifest as msgpack
FOURCC_TIDX = b"TIDX"    # tensor index as msgpack
FOURCC_WTSH = b"WTSH"    # weight shard raw bytes

# ── Alignment helpers ──────────────────────────────────────────────────────


def align(offset: int, alignment: int) -> int:
    """Round *offset* up to the next multiple of *alignment*."""
    return (offset + alignment - 1) & ~(alignment - 1)


def align8(offset: int) -> int:
    return align(offset, 8)


def align16(offset: int) -> int:
    return align(offset, 16)


def pad_to(data: bytes, alignment: int) -> bytes:
    """Return *data* padded with zero bytes to reach *alignment*."""
    padded_len = align(len(data), alignment)
    return data + b"\x00" * (padded_len - len(data))


# ── Safety limits ──────────────────────────────────────────────────────────

MAX_ENTRY_COUNT = 1_000_000          # reject > 1M TOC entries
MAX_STRING_TABLE = 512 * 1024 * 1024 # 512 MB string table
MAX_METADATA_ULEN = 2 * 1024 * 1024 * 1024  # 2 GB for metadata chunks


# ── Safe integer / shape utilities ─────────────────────────────────────────


def check_u64(value: int, name: str = "value") -> int:
    """Validate that *value* fits in a u64."""
    if value < 0 or value > (1 << 64) - 1:
        raise ValueError(f"{name} out of u64 range: {value}")
    return value


def shape_nbytes(shape: list, dtype_size: int) -> int:
    """Compute total bytes for a tensor shape, checking for overflow."""
    n = 1
    for d in shape:
        if d < 0:
            raise ValueError(f"negative dimension: {d}")
        n *= d
        if n > (1 << 63):
            raise ValueError(f"shape product overflow: {shape}")
    return n * dtype_size
