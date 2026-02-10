# Packaging & PyPI publishing

This folder holds **publishable PyPI packages** and instructions for the AERO organization layout.

## Repo layout

| Folder        | Package (PyPI) | Purpose |
|---------------|----------------|---------|
| **format/**   | `aerotensor`   | AERO format library: read/write `.aero`, convert GGUF/safetensors |
| **runtime/**  | `aero-runtime` | Inference runtime: load `.aero` and run models (optional PyTorch/Transformers) |
| **packaging/**| `aero`         | Meta package: `pip install aero` installs format + runtime (convenience) |

## Publishing to PyPI

### 1. Format library (`aerotensor`)

```bash
cd format
python -m pip install build twine
python -m build
python -m twine check dist/*
# Upload (requires PYPI_API_TOKEN):
python -m twine upload dist/*
```

Bump version in `format/pyproject.toml` and tag (e.g. `v0.2.1`) before publishing.

### 2. Runtime (`aero-runtime`)

```bash
cd runtime
python -m pip install build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Bump version in `runtime/pyproject.toml` and tag before publishing.

### 3. Meta package (`aero`)

The `aero` package in this folder has no code; it only depends on `aerotensor` and `aero-runtime[inference]`. Users can:

- `pip install aero` — installs format + runtime with inference (PyTorch, Transformers)
- Or install separately: `pip install aerotensor` and `pip install aero-runtime[inference]`

To publish the meta package:

```bash
cd packaging/aero
python -m build
python -m twine upload dist/*
```

Keep `aero/pyproject.toml` version and dependencies in sync with `format/` and `runtime/` when cutting releases.

## CI/CD

GitHub Actions (`.github/workflows/release.yml`) can be extended to build and upload all three packages on tag push. Ensure secrets (e.g. `PYPI_API_TOKEN`) are set for the organization or repo.

## Versioning

- **format** (`aerotensor`): Semantic version; bump when format or API changes.
- **runtime** (`aero-runtime`): Follow format when possible, or independent semver.
- **packaging/aero**: Bump when you want to release a new “bundle” (e.g. after updating format or runtime deps).
