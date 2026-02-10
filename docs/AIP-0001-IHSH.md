# AIP-0001: IHSH — Control-plane integrity

**Status:** Implemented (optional chunk)  
**Format version:** AERO v0.1 compatible (additive)

## Summary

IHSH is an **optional** chunk that stores a BLAKE3-256 digest over the **control plane**: header, TOC, and string table. This allows readers to detect tampering or corruption of the file layout and metadata index without reading payload chunks.

## Motivation

Per-chunk BLAKE3 (in the TOC) protects **payloads**. Header/TOC/string-table corruption can still cause parse failures or wrong chunk lookup. IHSH upgrades AERO to **container integrity**: one hash over the entire control region.

## Spec

### FourCC

`IHSH` (Integrity Hash for header+TOC+string table)

### Chunk flags

- `CHUNK_IS_OPTIONAL` (0x0008) — readers that do not support IHSH MUST skip it.

### Payload (MessagePack)

| Field | Type | Description |
|-------|------|-------------|
| `algo` | str | Hash algorithm; only `"blake3-256"` is defined. |
| `covered_ranges` | list of [u64, u64] | Byte ranges hashed (inclusive start, exclusive end). Typically `[[0, payloads_start]]`. |
| `digest` | bin (32 bytes) | BLAKE3-256 digest of the concatenation of those ranges. |

### Validation

- **`validate --control`:** Read control region (header + TOC + string table), normalize by zeroing **all** TOC entry hash fields (last 32 bytes of each TOC entry), compute BLAKE3 over that region, compare to IHSH digest. If IHSH is absent, skip (or warn).
- **`validate --full`:** Same as today (all chunks + optional IHSH). If IHSH is present, control-plane check is included.

**Implementation note:** Writer and reader both normalize the control region the same way (all TOC hashes zeroed) so the digest is deterministic and comparable. The writer computes the digest after all other TOC entries (lengths/hashes) are finalized.

### Writer behavior

- When requested (e.g. `add_control_integrity=True`), the writer writes header, TOC (all entry hashes initially zero), string table, then writes all other payloads and patches their TOC hashes. It then reads the control region `[0, payloads_start)`, zeroes all TOC entry hashes, computes BLAKE3, builds the IHSH payload, and writes IHSH as the **first** payload chunk (so TOC lists IHSH first). The IHSH TOC hash is patched last. IHSH is not compressed.

### Backwards compatibility

- Old readers ignore unknown chunks; IHSH is optional. New writers can always add IHSH; old writers never do.
