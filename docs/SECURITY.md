# AERO Security Considerations

## Defensive Parsing

AERO readers implement bounds checking at every level:

1. **Header validation** – magic, version, header_size must match exactly.
2. **TOC bounds** – `toc_offset + toc_length <= file_size`.
3. **Chunk bounds** – `chunk_offset + chunk_length <= file_size`.
4. **String table bounds** – `name_off + name_len <= string_table_length`.
5. **Tensor bounds** – `data_off + data_len <= shard_chunk_length`.

## Safety Limits

Both Python and C++ readers enforce safety caps to prevent allocation bombs
and denial-of-service via crafted files:

| Resource            | Limit       | Python | C++ | Reason                      |
|---------------------|-------------|--------|-----|-----------------------------|
| entry_count         | 1,000,000   | Yes    | Yes | Prevent TOC allocation bomb |
| string_table_length | 512 MB      | Yes    | Yes | Prevent allocation bomb     |
| metadata chunk ulen | 2 GB        | Yes    | Yes | Metadata should be small    |
| WTSH chunk ulen     | file_size   | N/A    | N/A | Weights may be large        |
| tensor name length  | 4,096       | -      | Yes | Sanity check                |
| TIDX tensor count   | 100,000,000 | -      | Yes | Prevent array bomb          |
| safetensors header  | 100 MB      | Yes    | -   | Converter safety cap        |

## Atomic Writes

The Python writer supports atomic file writes:
1. Write to `path.tmp`
2. `fsync` the file descriptor
3. `os.replace()` for an atomic rename

This prevents partially-written files from being read as valid.

## Integrity

- **BLAKE3-256** hash over uncompressed content for every chunk.
- Verification is **mandatory** for metadata chunks on read.
- WTSH verification is optional (streaming verify with `--full` flag).
- The hash prevents silent corruption from bit-flips, truncation, or
  bad storage.

## Compression Safety

- Only zstd decompression is used (well-audited library).
- Decompressed size is bounded by `chunk_ulen` from the TOC entry.
- For metadata chunks, `chunk_ulen` is capped at 2 GB to prevent
  zstd decompression bombs.
- If decompressed size doesn't match `chunk_ulen`, the reader raises.

## Converter Validation

The safetensors converter performs additional validation:
- **Offset bounds**: `data_off + end <= file_size`
- **Start < end**: data_offsets[0] < data_offsets[1]
- **Size consistency**: `dtype_size * product(shape) == end - start`
- **Header size cap**: safetensors JSON header limited to 100 MB.

## No Code Execution

AERO contains only binary data, MessagePack, and optional JSON.
There is no scripting, no pickle, no arbitrary code execution path.
This is a deliberate design choice — the format is **safe to parse**
from untrusted sources (modulo denial-of-service via large allocations,
which the limits above mitigate).

## Transport Integrity (AEROSET)

- Each `.aero` part file has a SHA-256 hash in the AEROSET index.
- This provides transport-level integrity for downloads.
- Inside each part, BLAKE3 per-chunk provides storage-level integrity.

## Fuzz Testing

The test suite includes a deterministic mutation test (`test_fuzz_mutation.py`)
that applies random byte mutations to valid AERO files and verifies that
the reader rejects corrupted input without crashing, hanging, or
performing unbounded allocations.

## Remote Reading Threat Model (v0.2)

When reading AEROSET files from HTTP(S) URLs:

### Network-Level Threats

| Threat | Mitigation |
|--------|------------|
| Man-in-the-middle | Use HTTPS (TLS) for transport encryption |
| Corrupted download | SHA-256 verification per part (from index JSON) |
| Partial/truncated response | Content-Length and byte count validation |
| Redirect loops | urllib follows redirects with built-in limits |

### Application-Level Threats

| Threat | Mitigation |
|--------|------------|
| Malicious index JSON | Strict schema validation, path resolution checks |
| Huge range requests | 64 MB per-request cap, automatic splitting |
| Unbounded tensor reads | 2 GB per-tensor cap |
| Cache poisoning | Atomic cache writes, URL-based keying |
| Cache exhaustion | LRU cleanup when exceeding max size (2 GB default) |
| SSRF via URLs | Client controls which URLs to trust (no automatic following) |

### Slice Integrity

- **Per-tensor `hash_b3`** (optional): BLAKE3-256 of exact tensor bytes.
  Enables verification of byte-range slices without downloading entire parts.
- **When absent**: Rely on transport (TLS) + part-level SHA-256 + chunk-level
  BLAKE3 for full-part verification.

### Recommendations

1. **Use HTTPS** for all remote URLs.
2. **Enable `tensor_hash`** when writing AEROSETs intended for remote distribution:
   ```bash
   aerotensor convert-safetensors model.safetensors model.aero --tensor-hash
   ```
3. **Verify index JSON** authenticity (e.g., via signed checksums or trusted CDN).
4. **Set cache limits** appropriate for your environment:
   ```bash
   export AEROTENSOR_CACHE_MAX_MB=4096  # 4 GB
   ```
5. **Monitor cache directory** for unexpected growth.
