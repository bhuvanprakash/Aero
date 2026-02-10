# aero

Meta package for the [AERO](https://github.com/Aero-HQ/Aero) ecosystem. Installing `aero` pulls in:

- **aerotensor** — AERO format library (read/write `.aero`, convert GGUF/safetensors)
- **aero-runtime[inference]** — Run models from `.aero` (PyTorch + Transformers)

## Install

```bash
pip install aero
```

Then:

```bash
# Inspect an .aero file
aero-run model.aero --info

# Generate (requires model-id for config/tokenizer)
aero-run model.aero --prompt "Hello!" --model-id Qwen/Qwen3-1.7B
```

For a minimal install (format only, no PyTorch), install the packages separately:

```bash
pip install aerotensor
pip install aero-runtime[inference]   # when you need inference
```

## Links

- **Format library:** [aerotensor](https://github.com/Aero-HQ/aerotensor) (PyPI coming soon)
- **Runtime:** [aeroruntime](https://github.com/Aero-HQ/aeroruntime) (PyPI coming soon)
- **Main repo:** [github.com/Aero-HQ/Aero](https://github.com/Aero-HQ/Aero)
