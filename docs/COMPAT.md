# AERO v0.1 Compatibility Rules

## Version Policy

AERO follows a major.minor versioning scheme stored in the file header.

- **Major version 0** indicates a pre-1.0 format. Breaking changes may
  occur between minor versions but will be documented.
- **Minor version 1** is the first public release.

## Reader Compatibility

A v0.1 reader MUST:

1. Accept any v0.1 file (version_major=0, version_minor=1, header_size=96).
2. Accept unknown chunk FourCC codes by skipping them (unless the reader
   specifically requires them).
3. Accept unknown keys in MessagePack structures (TIDX, MMSG) and ignore
   them — this enables forward-compatible extension.
4. Accept unknown chunk flags bits and ignore bits it doesn't recognize.

A v0.1 reader MUST NOT:

1. Reject a file because it contains chunks with unknown FourCC codes.
2. Reject a TIDX entry that contains extra fields beyond the required set.
3. Allocate memory based solely on untrusted header values without caps.

## Writer Compatibility

A v0.1 writer MUST:

1. Set version_major=0, version_minor=1, header_size=96.
2. Produce files byte-for-byte deterministic when given the same inputs
   and a fixed UUID.
3. Store WTSH chunks uncompressed.
4. Compute BLAKE3-256 hashes over uncompressed chunk bytes.
5. Use 16-byte alignment for chunk payloads.
6. Use 8-byte alignment for the string table.

## Optional Chunks

Chunks with the `CHUNK_IS_OPTIONAL` flag (0x0008) may be safely skipped
by readers that don't understand them. This enables adding new chunk
types without breaking existing readers.

## Chunk Order

Within a file, the standard chunk order is:
MJSN, MMSG, TIDX, WTSH (shard 0), WTSH (shard 1), ...

Readers MUST NOT depend on chunk order — they should look up chunks
by FourCC and name via the TOC.

## AEROSET Compatibility

The AEROSET index JSON uses semantic versioning.  A v0.1 reader should
accept any v0.1 index and skip unknown top-level keys.

Each part in an AEROSET is a standalone valid .aero file.  A reader
that doesn't understand AEROSET can still open individual parts.

## Extension Points

The following fields are reserved for future use:

- **File flags** (u64): currently must be 0; future versions may define bits.
- **TOC entry reserved0** (u64): must be 0; future metadata.
- **Header reserved** (28 bytes): must be zeros.
- **TIDX entry flags** (int): currently 0; future per-tensor flags.
- **TIDX entry quant_id / quant_params**: used for packed quantized tensors.

## Migration Path

When AERO v0.2 is defined, it will:
- Increment version_minor to 2.
- Remain backward compatible: v0.2 readers will accept v0.1 files.
- v0.1 readers will reject v0.2 files (version check).
