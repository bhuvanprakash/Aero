# Format validation and benchmarking

Two scripts: one runs a full AERO format check (single file, sharded, AEROSET, integrity, determinism, corruption); the other compares AERO to safetensors, GGUF, and optionally ONNX (speed, fidelity, corruption).

## Scripts

| Script | What it does |
|--------|----------------|
| **format_validation.py** | Writes a small model (single file, sharded, AEROSET), then runs the full check: validate, read-back, metadata, dtypes, determinism, chunk layout, corruption/truncation, AEROSET SHA-256, duplicate-name rejection. Exit 0 = pass. |
| **format_benchmark.py** | Benchmarks AERO vs safetensors / GGUF / ONNX: write and read times (9 runs averaged), file size, round-trip fidelity, corruption detection. Uses small up to 1000-tensor sets. |

## Quick start

From repo root:

```bash
# Full format validation (single / sharded / AEROSET, integrity, determinism, etc.)
python validation/format_validation.py

# Format comparison (AERO vs safetensors / GGUF / ONNX)
python validation/format_benchmark.py
```

Or from inside `validation/`:

```bash
cd validation
python format_validation.py
python format_benchmark.py
```

## Outputs

`format_validation.py` leaves `demo_single.aero`, `demo_sharded.aero`, and `demo_aeroset/` here (all in `.gitignore`). `format_benchmark.py` uses temp files and deletes them when done.

## ONNX

If you want ONNX in the benchmark:

```bash
pip install onnx
python validation/format_benchmark.py
```

## CLI examples (after running format_validation.py)

```bash
python -m aerotensor.cli inspect validation/demo_single.aero
python -m aerotensor.cli validate --full validation/demo_single.aero
python -m aerotensor.cli inspect-set validation/demo_aeroset/model.aeroset.json
python -m aerotensor.cli validate validation/demo_aeroset/model.aeroset.json --full
```
