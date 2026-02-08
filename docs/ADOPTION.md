# Using AERO in Other Projects

This page gives adoption hooks: how to plug AERO into your stack, CI, and when to pick single-file vs AEROSET.

---

## How to write a loader adapter

To use AERO from your own runtime (e.g. C++, Rust, or another Python pipeline):

1. **Discovery** – Open the file, read the 96-byte header, then TOC + string table. You can resolve chunk names and FourCCs without reading any payload.
2. **Tensors** – Find the TIDX chunk, read and decompress it (zstd if flagged), parse the MessagePack `tensors` array. Each entry has `name`, `dtype`, `shape`, `shard_id`, `data_off`, `data_len`; use `weights.shard{shard_id}` and the chunk’s file offset + `data_off` to read raw bytes.
3. **Integrity** – For each chunk you care about, read the stored payload, decompress if needed, then verify BLAKE3 of the *uncompressed* bytes against the 32-byte hash in the TOC entry.
4. **Weights** – WTSH chunks are uncompressed and 16-byte aligned; you can `mmap` the file at the chunk offset and slice to `[data_off, data_off + data_len]` for zero-copy.

Minimal Python pattern:

```python
from aerotensor import AeroReader

with AeroReader("model.aero") as r:
    for t in r.list_tensors():
        # your logic: e.g. load into your framework
        buf = r.read_tensor_bytes(t["name"])
```

For **AEROSET**, open `model.aeroset.json`, load `index.aero` for the global TIDX, and resolve each tensor’s `shard_id` to the correct `part-*.aero` (see [AEROSET.md](AEROSET.md)).

---

## How to convert safetensors / GGUF to AERO in CI

Use the CLI in your pipeline so that release artifacts include an AERO build:

```bash
# From repo root or where your model artifacts live
pip install aerotensor   # or: pip install -e py/

# Single safetensors
aerotensor convert-safetensors model.safetensors dist/model.aero

# Sharded safetensors → AEROSET
aerotensor convert-safetensors-sharded ./checkpoints/sharded dist/aero_set

# GGUF (e.g. from llama.cpp)
aerotensor convert-gguf model.gguf dist/model.aero
```

Optional: add integrity and a quick sanity check:

```bash
aerotensor validate --full dist/model.aero
aerotensor inspect dist/model.aero
```

You can run these in a GitHub Actions (or other CI) step after building or downloading the source model; then publish `dist/*.aero` or `dist/aero_set/` as release assets.

---

## When to choose AEROSET vs single-file

| Use case | Recommendation |
|----------|-----------------|
| **Single file** | Models that fit in one file (e.g. &lt; 50–100 GB), single-node loading, simple distribution. Use one `.aero` with one or more WTSH shards. |
| **AEROSET** | Very large models (100 GB+), resumable or parallel downloads, partial updates, or when you want to serve index + parts from different backends (e.g. CDN for parts, API for index). |
| **Remote URLs** | AEROSET v0.2 supports `base_url` and remote part URLs; use when parts are on HTTP(S) and you want the reader to fetch by range. |

Rule of thumb: start with **single-file AERO**; switch to **AEROSET** when file size or download/update patterns make one big file painful.
