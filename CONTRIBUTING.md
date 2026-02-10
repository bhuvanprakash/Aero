# Contributing to AERO

We’re on Apache 2.0. Contributions are welcome.

**Repo:** [github.com/Aero-HQ/Aero](https://github.com/Aero-HQ/Aero)

## Get started

1. **Clone the repo**
   ```bash
   git clone https://github.com/Aero-HQ/Aero.git
   cd Aero
   ```

2. **Install the Python package (editable)**
   ```bash
   pip install -e "format/.[dev]"
   ```
   This installs `aerotensor` and dev dependencies (e.g. pytest).

3. **Quick smoke test**
   ```bash
   aerotensor make-test-vector /tmp/test.aero
   aerotensor validate --full /tmp/test.aero
   aerotensor inspect /tmp/test.aero
   ```

## Running tests

- **Unit / integration tests**
  ```bash
  pytest format/tests/ -v
  ```

- **Format validation** (single/sharded/AEROSET, integrity, determinism, corruption checks)
  ```bash
  python validation/format_validation.py
  ```

- **Format benchmark** (AERO vs safetensors / GGUF / ONNX)
  ```bash
  python validation/format_benchmark.py
  ```

## C++ build (optional)

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/aero_validate /tmp/test.aero --full
```

See `cpp/README_CPP.md` for dependencies and options.

## Submitting changes

1. For bigger changes, open an issue first or comment on an existing one.
2. Fork, branch, make your changes.
3. Run the tests above (at least `pytest format/tests/` and `python validation/format_validation.py`).
4. Open a pull request. In the description, mention what changed and how you tested it.

By submitting a PR, you agree that your contributions are licensed under the Apache License 2.0.

## Code and docs

- **Python:** Follow the existing style (e.g. `format/aerotensor/`). Type hints and docstrings are welcome.
- **C++:** Match the style in `cpp/src/` and `cpp/include/`.
- **Docs:** Update `docs/` and `README.md` if you change behaviour or add features.

## Before pushing to GitHub

- [ ] Clone URLs in README and this file point to your repo (Aero-HQ/Aero).
- [ ] Run `pytest format/tests/ -v` and `python validation/format_validation.py`; both should pass.
- [ ] Don’t commit `.env`, API keys, or other secrets.
- [ ] If you still have a `modeltesting/` folder locally, you can delete it—everything lives in `validation/` now and `modeltesting/` is gitignored.

## CI

Every push and PR runs Python tests (Linux, macOS, Windows; several Python versions), C++ build and integration tests. Workflows live in `.github/workflows/`.
