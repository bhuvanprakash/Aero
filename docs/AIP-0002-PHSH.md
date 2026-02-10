# AIP-0002: PHSH — Page hashes per WTSH shard

**Status:** Implemented (optional chunk)  
**Format version:** AERO v0.1 compatible (additive)

## Summary

PHSH is an **optional** chunk that stores a list of BLAKE3-256 digests, one per **page** (fixed size, e.g. 4 MiB) of a WTSH shard. This allows remote or streaming readers to verify and fetch by page without buffering the full tensor or full shard.

## Motivation

Per-tensor `hash_b3` in TIDX works when the reader fetches **entire** tensor bytes. For partial tensor slices, resumable download, or "verify as you stream," page-level hashes are needed. PHSH makes remote loading closer to content-addressed storage.

## Spec

### FourCC

`PHSH` (Page Hash for WTSH)

### Chunk flags

- `CHUNK_IS_OPTIONAL` (0x0008) — readers that do not support PHSH MUST skip it.

### Naming and association

- One PHSH chunk per WTSH shard. Name: `weights.shard{N}.phsh` for the shard `weights.shard{N}`.
- Order in file is arbitrary; readers look up by name.

### Payload (MessagePack)

| Field | Type | Description |
|-------|------|-------------|
| `shard_name` | str | WTSH chunk name this PHSH applies to (e.g. `weights.shard0`). |
| `page_size` | u64 | Page size in bytes (e.g. 4 MiB). Last page may be shorter. |
| `digests` | list of bin (32 bytes each) | BLAKE3-256 of each page in order. |

### Page layout

- Shard payload is split into contiguous pages of size `page_size`. Last page = remainder.
- `digests[i]` = BLAKE3-256 of bytes `[i * page_size, min((i+1) * page_size, shard_ulen))`.

### Validation

- **Optional:** When reading a WTSH range by page, reader can verify each page’s digest against PHSH before use.
- **`validate --full` / `validate_all()`:** If PHSH is present for a WTSH, the reader verifies all page digests for that shard (see `validate_shard_pages()`). Large shards may take longer to validate.

### Writer behavior

- When requested (e.g. `add_page_hashes=True`), after writing each WTSH shard, writer computes page digests and writes a PHSH chunk `weights.shard{N}.phsh`. Default page size 4 MiB (configurable).

### Backwards compatibility

- Old readers ignore PHSH. WTSH layout unchanged. New readers can use PHSH for verify-by-page when present.
