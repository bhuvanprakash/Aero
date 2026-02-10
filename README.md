# AERO

**AERO** is a chunked model container format and runtime ecosystem for packaging
machine-learning model weights and metadata into a single file (`.aero`) or
multi-file set (AEROSET).  It is designed for **fast loading**, **safe parsing**,
and **zero-copy weight access** via `mmap`.

**Versioning (no confusion):**
- **AERO binary container:** version **0.1** (header magic, TOC, chunk layout).
- **AEROSET schema:** version **0.2** (remote URLs, `base_url`, cache guidance).
- **Package (PyPI / this repo):** **X.Y.Z** (e.g. `0.2.0`); follows semantic versioning.

---

## Try it in 2 minutes

From a **fresh clone**:

```bash
git clone https://github.com/Aero-HQ/Aero.git && cd Aero
pip install -e format/
aerotensor make-test-vector /tmp/test.aero
aerotensor validate --full /tmp/test.aero
aerotensor inspect /tmp/test.aero
```

(PyPI: coming soon. For now use the source install above.)

### Repository layout (organization-style)

| Folder | Contents | PyPI |
|--------|----------|------|
| **format/** | AERO format library (read/write `.aero`, convert GGUF/safetensors) | `aerotensor` |
| **runtime/** | Inference runtime — load `.aero` and run models | `aero-runtime` |
| **packaging/** | PyPI meta package and publish instructions | `aero` (optional bundle) |
| **cpp/** | C++ reader, tools, shared library | — |

One-line install for users: `pip install aero` (from [packaging/](packaging/)) or install format and runtime separately.

### Run inference from .aero (aero-runtime)

Install the runtime in one go and run models directly from `.aero` files (no conversion back to GGUF):

```bash
python scripts/install_runtime.py   # friendly progress + time (recommended)
# or: pip install -e format/ && pip install -e runtime/
aero-run validation/ollama_bench_out/qwen3_1_7b.aero --prompt "What is 2+2?" --model-id Qwen/Qwen3-1.7B
```

Or from the repo with the demo script:

```bash
./run_aero_demo.sh validation/ollama_bench_out/qwen3_1_7b.aero --prompt "Hello!"
```

See [runtime/README.md](runtime/README.md) for the full aero-runtime API and options.

Prebuilt C++ tools and Python wheels are attached to [GitHub Releases](https://github.com/Aero-HQ/Aero/releases) on each tag.

---

## Why AERO?

### Chunked TOC + String Table = Fast Discovery

Every AERO file begins with a fixed 96-byte header followed by a compact
Table of Contents (TOC).  Each TOC entry is exactly 80 bytes and tells you
the FourCC type, stored/uncompressed sizes, name offset, flags, and a BLAKE3
hash — all in one flat, seekable structure.  A reader can learn everything
about the file by reading **header + TOC + string table** (typically < 4 KB),
without touching any payload data.

### BLAKE3 Per-Chunk Integrity

Every chunk carries a 32-byte BLAKE3-256 digest computed over the
*uncompressed* content.  You can verify individual chunks without
reading the whole file.  BLAKE3 is fast, has one main implementation,
and gives strong integrity.

### Metadata May Be Compressed; Weights Stay Raw

Metadata chunks (manifest, tensor index, JSON) may be zstd-compressed
because they are small and benefit from the space savings.  Weight shards
are stored **uncompressed** and at 16-byte aligned offsets so they can be
directly `mmap`'d into GPU/CPU memory without a copy or decompression step.

---

## Features

| Feature | Status |
|---------|--------|
| Single-file `.aero` with multi-shard WTSH | ✓ |
| Multi-file AEROSET (index + parts) | ✓ |
| Convert safetensors → AERO | ✓ |
| Convert sharded safetensors → AEROSET | ✓ |
| Convert GGUF → AERO (quant blocks preserved) | ✓ |
| Python reader + writer | ✓ |
| C++ reader (cross-platform) | ✓ |
| Per-chunk BLAKE3 integrity | ✓ |
| Streaming write (one tensor in memory at a time) | ✓ |
| CI: Linux / macOS / Windows | ✓ |
| **aero-runtime** — run inference from .aero | ✓ |
| **C ABI** (aero_load, aero_tensor_ptr, etc.) + shared lib (libaero.so / .dylib / .dll) | ✓ |
| **Single-header** (aero_capi.h / aero_runtime.h) for embedding | ✓ |

**Benchmarks:** Load time and peak memory (open / read_all / mmap) — see [BENCHMARKS.md](BENCHMARKS.md). *Why AERO?* → [docs/WHY_AERO.md](docs/WHY_AERO.md). Minimal example: [examples/inference_with_aero.py](examples/inference_with_aero.py).

---

## File Layout

```
┌──────────────────────────────┐  0
│  Header (96 B, LE)           │
├──────────────────────────────┤  96
│  TOC Header (16 B)           │
│  TOC Entry × N (80 B each)   │
├──────────────────────────────┤  align8
│  String Table (NUL-sep UTF8) │
├──────────────────────────────┤  align16
│  Chunk Payloads …            │
│  (each at 16-byte alignment) │
└──────────────────────────────┘
```

### Chunk Types

| FourCC | Role | Compression | Notes |
|--------|------|-------------|-------|
| `MJSN` | Human-readable JSON metadata | optional | Usually uncompressed |
| `MMSG` | Manifest (msgpack) | zstd OK | Model name, architecture, shard list |
| `TIDX` | Tensor index (msgpack) | zstd OK | Tensor names, shapes, dtypes, offsets |
| `WTSH` | Weight shard (raw bytes) | **never** | Must be mmap-safe |

---

## Python Quick Start

```bash
cd py
pip install -e .

# Generate a golden test vector
aerotensor make-test-vector /tmp/test.aero

# Inspect the file
aerotensor inspect /tmp/test.aero

# Validate all chunk hashes
aerotensor validate /tmp/test.aero

# Zero-copy: use read_tensor_view() for a memoryview; read_tensor_bytes() copies.

# Compare two .aero files (tensor set + byte/numeric diff)
aerotensor diff model_a.aero model_b.aero --verbose --numeric

# Run tests
pytest tests/ -q
```

### Single-File AERO

```python
from aerotensor import AeroWriter, TensorSpec, DType
import numpy as np

# Stream tensors into a multi-shard .aero file
w = np.random.randn(4096, 4096).astype("<f4")
specs = [TensorSpec(
    name="layer.0.weight", dtype=int(DType.F32),
    shape=[4096, 4096], data_len=w.nbytes,
    read_data=w.tobytes,
)]

writer = AeroWriter()
writer.write_model("model.aero", specs, "my-model", "transformer",
                   max_shard_bytes=2 * 1024**3)
```

### Reading

```python
from aerotensor import AeroReader
import numpy as np

with AeroReader("model.aero") as r:
    for t in r.list_tensors():
        print(t["name"], t["shape"], t["dtype"])

    data = r.read_tensor_bytes("layer.0.weight")
    w = np.frombuffer(data, dtype="<f4").reshape(4096, 4096)
```

### Multi-File AEROSET

```python
from aerotensor import write_aeroset, AeroSetReader

# Write
write_aeroset("output_dir/", specs, "big-model", "transformer",
              max_shard_bytes=2<<30, max_part_shards=4)

# Read
with AeroSetReader("output_dir/model.aeroset.json") as r:
    data = r.read_tensor_bytes("layer.0.weight")
```

### Converters

```bash
# safetensors → AERO
aerotensor convert-safetensors model.safetensors model.aero

# Sharded safetensors → AEROSET
aerotensor convert-safetensors-sharded model_dir/ output_dir/

# GGUF → AERO (preserves quantized blocks)
aerotensor convert-gguf model.gguf model.aero
```

---

## C++ Quick Start

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j

# Single-file tools
./cpp/build/aero_inspect      model.aero
./cpp/build/aero_validate     model.aero --full
./cpp/build/aero_list_tensors model.aero
./cpp/build/aero_extract_tensor model.aero "layer.0.weight" out.bin

# AEROSET tools
./cpp/build/aero_inspect_set       model.aeroset.json
./cpp/build/aero_extract_tensor_set model.aeroset.json "layer.0.weight" out.bin
```

See [`cpp/README_CPP.md`](cpp/README_CPP.md) for build options and
dependency details.

---

## Testing

- **Unit/integration:** `pytest format/tests/ -v`
- **Format validation:** `python validation/format_validation.py`
- **Format benchmark:** `python validation/format_benchmark.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full test and C++ build instructions.

---

## Documentation

- [FORMAT.md](docs/FORMAT.md) – AERO v0.1 binary format specification
- [AEROSET.md](docs/AEROSET.md) – Multi-file container specification (schema v0.2)
- [CONVERTERS.md](docs/CONVERTERS.md) – Converter usage and supported formats
- [SECURITY.md](docs/SECURITY.md) – Bounds checking and safety philosophy
- [COMPAT.md](docs/COMPAT.md) – v0.1 compatibility rules and extension policy
- [ADOPTION.md](docs/ADOPTION.md) – Using AERO in other projects (loader adapter, CI convert, AEROSET vs single-file)
- [RUNTIME_SPEC.md](docs/RUNTIME_SPEC.md) – Minimal C ABI for load/tensor access (zero-copy, FFI-friendly) + future GPU API design
- [RUNTIME_ROADMAP.md](docs/RUNTIME_ROADMAP.md) – Runtime roadmap (current state, C ABI wrapper, zero-copy GPU, native loaders)
- [DROP_IN_REPLACEMENT.md](docs/DROP_IN_REPLACEMENT.md) – Replace safetensors/GGUF with AERO in Hugging Face, llama.cpp, vLLM
- [GROWTH_ROADMAP.md](docs/GROWTH_ROADMAP.md) – 7-phase plan: runtime → benchmarks → adoption → aero.dev → premium → monetization
- [BENCHMARKS.md](BENCHMARKS.md) – How to run load/memory benchmarks and compare AERO vs GGUF/safetensors
- [docs/WHY_AERO.md](docs/WHY_AERO.md) – “Why we built AERO” (blog-style)
- [docs/WHY_AERO_BENCHMARKS.md](docs/WHY_AERO_BENCHMARKS.md) – Benchmark scripts and evidence
- [examples/](examples/) – Minimal inference example
- [CONTRIBUTING.md](CONTRIBUTING.md) – How to run tests, build C++, and submit changes

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| MessagePack for metadata | Faster and more compact than JSON; deterministic packing |
| JSON chunk optional | Human-readable fallback for debugging |
| Fixed 80-byte TOC entries | Seekable without parsing variable-length data |
| 16-byte payload alignment | Works for SIMD loads and GPU DMA |
| Uncompressed weights | Required for `mmap` / zero-copy |
| Per-chunk BLAKE3 | Verify individual chunks without full-file scan |
| UUID in header | Distinguish file instances without path dependency |
| Streaming writer | Handles 100GB+ models without loading all weights |
| AEROSET for huge models | Resumable downloads, parallel fetch, partial updates |

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

Third-party license notices are in [LICENSES/NOTICES.md](LICENSES/NOTICES.md).
