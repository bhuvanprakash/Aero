# AERO Converters

## safetensors → AERO

Convert a single safetensors file to a single AERO file.

```bash
aerotensor convert-safetensors model.safetensors model.aero
aerotensor convert-safetensors model.safetensors model.aero --max-shard-bytes 2147483648
```

**What it does:**
1. Parses the safetensors JSON header (tensor names, dtypes, shapes, offsets).
2. Sorts tensors by name for deterministic output.
3. Streams tensor data into sharded WTSH chunks via `write_model()`.
4. Maps safetensors dtypes (F32, F16, BF16, etc.) to AERO DType enum.

**Streaming:** Only one tensor's bytes are in memory at a time.

## Sharded safetensors → AEROSET

Convert a sharded safetensors model (common HuggingFace format) to an
AEROSET directory.

```bash
aerotensor convert-safetensors-sharded ./model_dir/ ./aero_out/
aerotensor convert-safetensors-sharded model.safetensors.index.json ./aero_out/
```

**Input format:**
```
model_dir/
  model.safetensors.index.json
  model-00001-of-00005.safetensors
  model-00002-of-00005.safetensors
  …
```

The index JSON maps tensor names to shard files via `weight_map`.

**Output:**
```
aero_out/
  model.aeroset.json
  index.aero
  part-000.aero
  part-001.aero
  …
```

## GGUF → AERO

Convert a GGUF file (llama.cpp format) to AERO.

```bash
aerotensor convert-gguf model.gguf model.aero
```

**Quantized tensors** are preserved as-is using `DType.PACKED`:

- The raw quantized block bytes are copied verbatim into WTSH.
- Each tensor's TIDX entry includes `quant_id` and `quant_params` fields.
- The manifest includes a `quant_descriptors` array mapping `quant_id` to
  GGML type info.

**Supported GGML types:**
- Native: F32, F16, BF16, F64, I8, I16, I32, I64
- Quantized (preserved as packed): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
  Q2_K through Q8_K

**GGUF metadata** (general.name, general.architecture) is mapped to the
AERO manifest.

## Python API

```python
from aerotensor.convert.safetensors_to_aero import convert_safetensors
from aerotensor.convert.gguf_to_aero import convert_gguf

convert_safetensors("model.safetensors", "model.aero")
convert_gguf("model.gguf", "model.aero")
```

## Notes

- All converters produce **deterministic** output: same input → same bytes
  (given the same UUID).
- Tensor order is always sorted by name.
- WTSH chunks are never compressed.
- Metadata chunks (MMSG, TIDX) are zstd-compressed.
- BLAKE3 hashes are computed over uncompressed content.
