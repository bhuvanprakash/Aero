# AERO v0.1 Binary Format Specification

**Version:** AERO binary container is **0.1**. Package version (e.g. `0.2.0`) is separate; AEROSET schema is **0.2** (see [AEROSET.md](AEROSET.md)).

## Overview

AERO is a chunked binary container for storing machine-learning model
weights, metadata, and tensor indices.  Design goals:

- **Fast loading** – flat TOC + string table for O(1) chunk discovery.
- **Zero-copy weights** – WTSH chunks are uncompressed and 16-byte aligned
  for direct mmap.
- **Integrity** – every chunk carries a BLAKE3-256 hash over its
  uncompressed content.
- **Compact metadata** – MessagePack encoding with optional zstd compression.

## File Layout

```
┌──────────────────────────────┐  offset 0
│  File Header  (96 B)        │
├──────────────────────────────┤  96
│  TOC Header   (16 B)        │
│  TOC Entry ×N (80 B each)   │
├──────────────────────────────┤  align8
│  String Table (NUL-sep UTF-8)│
├──────────────────────────────┤  align16
│  Chunk Payloads …            │
└──────────────────────────────┘
```

All multi-byte integers are **little-endian**.

## File Header (96 bytes)

| Offset | Size | Field               | Value / Description              |
|--------|------|---------------------|----------------------------------|
| 0      | 4    | magic               | `"AERO"` (ASCII)                |
| 4      | 2    | version_major       | `0`                              |
| 6      | 2    | version_minor       | `1`                              |
| 8      | 4    | header_size         | `96`                             |
| 12     | 8    | toc_offset          | Byte offset of TOC               |
| 20     | 8    | toc_length          | Total bytes of TOC               |
| 28     | 8    | string_table_offset | Byte offset of string table      |
| 36     | 8    | string_table_length | Bytes of string table            |
| 44     | 8    | file_flags          | Reserved, must be `0`            |
| 52     | 16   | uuid                | File identity (random or fixed)  |
| 68     | 28   | reserved            | Must be zeros                    |

## TOC Header (16 bytes)

| Offset | Size | Field       |
|--------|------|-------------|
| 0      | 4    | entry_count |
| 4      | 4    | reserved0   |
| 8      | 8    | reserved1   |

## TOC Entry (80 bytes)

| Offset | Size | Field       | Description                          |
|--------|------|-------------|--------------------------------------|
| 0      | 4    | fourcc      | Chunk type tag (ASCII)               |
| 4      | 4    | chunk_flags | Bitmask (see Flags)                  |
| 8      | 8    | chunk_offset| Absolute file offset of payload      |
| 16     | 8    | chunk_length| Stored (possibly compressed) size    |
| 24     | 8    | chunk_ulen  | Uncompressed size                    |
| 32     | 4    | name_off    | Offset into string table             |
| 36     | 4    | name_len    | Length of name (bytes)               |
| 40     | 8    | reserved0   | Must be `0`                          |
| 48     | 32   | blake3_256  | BLAKE3-256 of **uncompressed** bytes |

## Chunk Flags (u32)

| Bit    | Name                  | Description                   |
|--------|-----------------------|-------------------------------|
| 0x0001 | COMPRESSED_ZSTD       | Payload is zstd-compressed    |
| 0x0002 | CHUNK_IS_MMAP_CRITICAL| Should be mmap'd for perf     |
| 0x0004 | CHUNK_IS_INDEX        | Contains an index structure   |
| 0x0008 | CHUNK_IS_OPTIONAL     | Readers may skip              |

## String Table

NUL-separated UTF-8 names, padded with zeros to an 8-byte boundary.

## Chunk Types

| FourCC | Role             | Compression | Notes                         |
|--------|------------------|-------------|-------------------------------|
| IHSH   | Control integrity| **Never**   | Optional; BLAKE3 over header+TOC+string table (see [AIP-0001](AIP-0001-IHSH.md)) |
| MJSN   | JSON metadata    | Optional    | Human-readable                |
| MMSG   | Manifest (msgpack)| Recommended | Forward-compatible manifest   |
| PHSH   | Page hashes      | **Never**   | Optional; per-page BLAKE3 for a WTSH shard (see [AIP-0002](AIP-0002-PHSH.md)) |
| TIDX   | Tensor index     | Recommended | Maps tensor name → shard/offset |
| WTSH   | Weight shard     | **Never**   | Raw bytes, mmap-critical      |

## Multi-Shard

A single `.aero` file may contain multiple WTSH chunks named
`weights.shard0`, `weights.shard1`, etc.  The TIDX `shard_id` field
identifies which shard holds each tensor.

## DType Enum (u16)

| Value  | Type   |
|--------|--------|
| 0      | f16    |
| 1      | f32    |
| 2      | bf16   |
| 3      | f64    |
| 4–11   | i8/u8/i16/u16/i32/u32/i64/u64 |
| 12     | bool   |
| 0x8000 | packed (quantized / codec data) |

## TIDX Schema (MessagePack)

```json
{
  "tensors": [
    {
      "name": "layer.0.weight",
      "dtype": 1,
      "shape": [4096, 4096],
      "shard_id": 0,
      "data_off": 0,
      "data_len": 67108864,
      "flags": 0,
      "hash_b3": "abc123..." 
    }
  ]
}
```

### Optional TIDX Fields

- **`quant_id`** (int): Index into `quant_descriptors` for quantized tensors.
- **`quant_params`** (map): Codec-specific parameters (e.g., `{"ggml_type": 2}`).
- **`hash_b3`** (string, v0.2+): BLAKE3-256 hash of the tensor's raw bytes
  as a 64-character hex string. Enables slice integrity verification for
  remote range reads. Optional; if absent, only transport-level integrity
  (TLS) and chunk-level BLAKE3 apply.

## MMSG Schema (MessagePack)

```json
{
  "format": {"name": "AERO", "version": [0, 1]},
  "model": {"name": "…", "architecture": "…"},
  "chunks": [ … ],
  "shards": [ … ]
}
```

## Limits

Readers and writers enforce the following caps so that invalid or malicious files are rejected early.

| Constant | Value | Meaning |
|----------|--------|---------|
| **MAX_ENTRY_COUNT** | 1,000,000 | Maximum TOC entries (chunks) per file. |
| **MAX_STRING_TABLE** | 512 MB | Maximum string table size (chunk names). |
| **MAX_METADATA_ULEN** | 2 GB | Maximum uncompressed size of a metadata chunk (e.g. TIDX, MMSG). |

**Practical tensor count cap:** The TIDX chunk is a single metadata chunk, so its decompressed size must be ≤ MAX_METADATA_ULEN. With roughly 50–80 bytes per tensor entry in MessagePack, this implies a **practical cap of ~40M tensors per file**. Files claiming more than that would require a TIDX larger than 2 GB and are rejected.

**For more than ~40M tensors** you would need one or both of:
- A **streaming TIDX writer** that does not materialise the full index in memory.
- A **multi-level index** (e.g. sharded TIDX or external index) and format extension; not defined in v0.1.

## Optional integrity chunks

- **IHSH (AIP-0001):** Optional chunk storing a BLAKE3-256 digest over the control region (header + TOC + string table). Enables `validate --control` to detect tampering or corruption of the file layout. See [AIP-0001-IHSH.md](AIP-0001-IHSH.md).
- **PHSH (AIP-0002):** Optional chunk per WTSH shard storing per-page BLAKE3-256 digests. Enables page-level verification for streaming or partial reads. See [AIP-0002-PHSH.md](AIP-0002-PHSH.md).

Both are additive and backward compatible; readers that do not support them skip by fourcc/optional flag.
