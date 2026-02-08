# AEROSET v0.2 – Multi-File Model Container

## Motivation

Single-file AERO works well for models up to ~50 GB.  For 300B+ parameter
models (hundreds of GB to TB), a single file is impractical:

- **Resumable downloads** – if a transfer fails, only re-fetch one part.
- **Parallel fetch** – download parts concurrently.
- **Partial updates** – replace a single weight part without rewriting all.
- **Filesystem limits** – some filesystems cap file size at 4 GB or 16 GB.

AEROSET solves this with a simple JSON index pointing to multiple `.aero`
part files.

## Directory Structure

```
model/
  model.aeroset.json     ← JSON index
  index.aero             ← tiny AERO with global MMSG + TIDX (no WTSH)
  part-000.aero          ← valid AERO file with some WTSH shards
  part-001.aero
  …
```

## Index Schema (`model.aeroset.json`)

```json
{
  "format": {"name": "AEROSET", "version": [0, 1]},
  "model": {
    "name": "llama-70b",
    "architecture": "llama"
  },
  "parts": [
    {
      "path": "part-000.aero",
      "sha256": "abc123…",
      "size_bytes": 4294967296,
      "shards": [0, 1, 2, 3]
    },
    {
      "path": "part-001.aero",
      "sha256": "def456…",
      "size_bytes": 4294967296,
      "shards": [4, 5, 6, 7]
    }
  ],
  "global_tidx": {
    "path": "index.aero",
    "sha256": "789abc…",
    "size_bytes": 12345
  }
}
```

### v0.2 Remote Extensions

v0.2 adds support for HTTP(S) URLs:

```json
{
  "format": {"name": "AEROSET", "version": [0, 2]},
  "model": {"name": "llama-70b", "architecture": "llama"},
  "base_url": "https://cdn.example.com/models/llama-70b",
  "parts": [
    {
      "path": "part-000.aero",
      "sha256": "abc123…",
      "size_bytes": 4294967296,
      "shards": [0, 1, 2, 3]
    }
  ],
  "global_tidx": {
    "path": "https://cdn.example.com/models/llama-70b/index.aero",
    "sha256": "789abc…",
    "size_bytes": 12345
  },
  "cache": {
    "enabled": true,
    "recommended": true
  }
}
```

**New fields:**

- **`base_url`** (optional): Base URL for resolving relative paths. If a
  part's `path` is relative (doesn't start with `http://` or `https://`),
  it's resolved as `base_url + "/" + path`.
- **`cache`** (optional): Hints for client caching. Clients may cache
  byte ranges locally to minimize bandwidth.
- **Parts can be full URLs**: `"path"` may be `"https://..."` instead of
  a relative path.

**Backward compatibility:** v0.1 readers ignore `base_url` and `cache`.
v0.2 readers accept v0.1 files (all paths local).

## Reading Tensors

1. Load `model.aeroset.json`.
2. Open `index.aero` and parse its TIDX → get tensor metadata.
3. To read tensor `T`:
   - Look up `T.shard_id` in TIDX.
   - Find which part contains that shard (from `parts[].shards`).
   - Open that part's `.aero` file.
   - Read from the WTSH chunk named `weights.shard{shard_id}` at
     offset `T.data_off`.

Parts are opened lazily and cached for subsequent reads.

## Writing an AEROSET

The Python API:

```python
from aerotensor.aeroset import write_aeroset
from aerotensor.writer import TensorSpec

write_aeroset(
    "output_dir/",
    tensor_specs,
    model_name="my-model",
    architecture="transformer",
    max_shard_bytes=2 * 1024**3,   # 2 GB per shard
    max_part_shards=4,             # 4 shards per part file
)
```

## CLI

```bash
# Local
aerotensor inspect-set model.aeroset.json
aerotensor validate model.aeroset.json --full

# Remote (v0.2)
aerotensor fetch-tensor https://cdn.example.com/model.aeroset.json "layer.0.wq" out.bin
aerotensor validate https://cdn.example.com/model.aeroset.json --remote
```

## C++ API

```cpp
aero::AeroSetFile set("model/model.aeroset.json");
auto bytes = set.read_tensor_bytes("layer.0.weight");
```

## Remote HTTP Range Reading (v0.2)

AEROSET v0.2 enables efficient remote model loading via HTTP Range requests:

### How It Works

1. **Fetch index JSON** — small file, lists all parts and global TIDX location.
2. **Fetch index.aero header + TOC** — minimal bytes to parse tensor index.
3. **On-demand tensor fetch** — for each tensor:
   - Locate its part and shard from global TIDX
   - Compute byte range: `part_url + chunk_offset + data_off` for `data_len` bytes
   - Issue HTTP Range request: `Range: bytes=start-end`
   - Verify `hash_b3` if present (per-tensor integrity)

### Disk Caching

The Python client caches fetched ranges in `~/.cache/aerotensor/`:

- **Cache key**: `(url_sha256, start, length)`
- **LRU cleanup**: When cache exceeds 2 GB (configurable), oldest files deleted
- **Atomic writes**: Cache entries written atomically to prevent corruption

### Safety Limits

- **Max range per request**: 64 MB (splits larger requests automatically)
- **Max tensor size**: 2 GB
- **Transport integrity**: SHA-256 per part + TLS
- **Slice integrity**: BLAKE3 per tensor (if `hash_b3` in TIDX)

## Design Decisions

- **JSON index** chosen over binary for human debuggability.
- **SHA-256 per part** for transport integrity (BLAKE3 is per-chunk inside
  each part).
- **Global TIDX** lets you list all tensors without touching weight files.
- Each part is a **standalone valid `.aero` file** with its own TIDX and
  MMSG, enabling independent validation.
- **HTTP Range for remote** (v0.2) — fetch only required bytes, not entire parts.
