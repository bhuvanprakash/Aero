# aero-runtime — Run Models from .aero Files

**aero-runtime** is the inference runtime for the [AERO](https://github.com/Aero-HQ/Aero) chunked model container format.

**Repo:** [Aero-HQ/aeroruntime](https://github.com/Aero-HQ/aeroruntime)

It loads weights directly from `.aero` files, dequantizes quantized tensors in memory, and runs generation using HuggingFace Transformers — all without converting back to GGUF or safetensors.

## Install

**From source (this repo)** — base install (no PyTorch). Inspect `.aero` files; add inference when needed:

```bash
git clone https://github.com/Aero-HQ/aeroruntime.git && cd aeroruntime
# Build C++ lib (optional, for faster state-dict load): cd cpp && cmake -B build && cmake --build build && cd ..
pip install -e .
# Then: aero-run model.aero --info
# When you want to run models: pip install -e ".[inference]"
```

**With inference (PyTorch + Transformers)** — needed for generation and `AeroModel`:

```bash
pip install aero-runtime[inference]
```

**Minimal (no PyTorch)** — inspect files only (`aero-run model.aero --info` works; generation needs `[inference]`):

```bash
pip install aero-runtime
```

**From source with inference** — from this repo root:

```bash
pip install -e ".[inference]"
```

**Note:** PyTorch is a Python library; there is no `torch` shell command. To check it: `python -c "import torch; print(torch.__version__)"`. Use the same `python` for both install and run (e.g. `python -m aero_runtime.cli ...`).

## Quick Start

### CLI

```bash
# Inspect an .aero file
aero-run model.aero --info

# Generate text
aero-run model.aero --prompt "What is 2+2?" --model-id Qwen/Qwen3-1.7B

# Interactive chat
aero-run model.aero --interactive --model-id Qwen/Qwen3-1.7B

# Use float16 to save memory
aero-run model.aero --prompt "Hello!" --model-id Qwen/Qwen3-1.7B --dtype float16

# GPU inference
aero-run model.aero --prompt "Explain AI" --model-id Qwen/Qwen3-1.7B --device cuda
```

### Python API

```python
from aero_runtime import AeroModel

# Load and run
model = AeroModel("model.aero", model_id="Qwen/Qwen3-1.7B")
output = model.generate("What is 2+2?", max_new_tokens=50)
print(output)

# With custom settings
model = AeroModel(
    "model.aero",
    model_id="Qwen/Qwen3-1.7B",
    device="cuda",
    dtype=torch.float16,
)
output = model.generate("Explain gravity.", temperature=0.5, max_new_tokens=200)
```

### Low-level: Load State Dict

```python
from aero_runtime import load_state_dict, AeroModelLoader

# Quick: get a PyTorch state_dict
state_dict = load_state_dict("model.aero")

# Detailed: inspect metadata first
loader = AeroModelLoader("model.aero")
print(loader.model_name)      # e.g. "qwen3"
print(loader.architecture)    # e.g. "qwen2"
print(loader.source_format)   # "gguf" or "safetensors"
print(loader.tensor_count)    # 311
print(loader.file_size_mb)    # 1290.64

state_dict = loader.load_state_dict(target_dtype=torch.float16)
```

## How It Works

```
┌──────────────────────────────────────────────────────┐
│                    .aero file                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │  Header   │  │   TOC    │  │  Weight Shards     │ │
│  │  (96 B)   │  │  + TIDX  │  │  (mmap-aligned)    │ │
│  └──────────┘  └──────────┘  └────────────────────┘ │
└──────────────────┬───────────────────────────────────┘
                   │ AeroModelLoader
                   ▼
        ┌─────────────────────┐
        │  1. Read tensors    │
        │  2. Dequantize      │  (PACKED → float via gguf)
        │  3. Map names       │  (GGUF → HuggingFace)
        └─────────┬───────────┘
                  │ state_dict
                  ▼
        ┌─────────────────────┐
        │  HuggingFace Model  │
        │  (AutoModelForCLM)  │
        └─────────┬───────────┘
                  │
                  ▼
            model.generate()
```

**No GGUF file is written or used.** The `.aero` file is the single source of weights.

## Supported Models

aero-runtime supports any model that can be:
1. Converted to `.aero` (from GGUF or safetensors via `aerotensor`)
2. Loaded into a HuggingFace Transformers model

Tested architectures:
- **Qwen3** (1.7B, 7B) — `Qwen/Qwen3-1.7B`
- **Qwen2.5** (1.5B, 7B) — `Qwen/Qwen2.5-1.5B-Instruct`
- **Llama 3.1** (8B) — `meta-llama/Llama-3.1-8B-Instruct`
- **SmolLM2** (360M, 1.7B) — `HuggingFaceTB/SmolLM2-360M-Instruct`

## Pipeline

```
Full-precision model          Quantized model             AERO runtime
(safetensors on HF)     →     (GGUF via llama.cpp)    →   (.aero file)
                                                           ↓
                                                      aero-run model.aero
                                                           ↓
                                                      "Hello! I am ..."
```

## Format compatibility

aero-runtime loads `.aero` files produced by **aerotensor**. The format supports optional integrity chunks:

- **IHSH (AIP-0001)** — control-plane BLAKE3 digest (header + TOC + string table). In aerotensor, use `validate --control` to verify; fully implemented in reader and writer.
- **PHSH (AIP-0002)** — per-page BLAKE3 digests for each WTSH shard. In aerotensor, `validate_all()` verifies PHSH when present.

Readers that do not use these chunks simply skip them (optional flag); existing and extended files both work.

## Requirements

- Python 3.9+
- `aerotensor` — AERO format reader/writer
- `torch` — PyTorch
- `transformers` — HuggingFace Transformers
- `gguf` — for dequantizing quantized models
- `numpy`

## License

Apache-2.0
