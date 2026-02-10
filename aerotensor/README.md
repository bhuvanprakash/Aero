# aerotensor

**Repo:** [Aero-HQ/aerotensor](https://github.com/Aero-HQ/aerotensor)

Python package for **AERO** — a chunked model container format for ML weights (single-file `.aero` or multi-file AEROSET).

## Install

```bash
pip install -e .
# Optional: checksums with pip install -e ".[verify]"
```

## Quick start

```bash
aerotensor make-test-vector /tmp/test.aero
aerotensor validate --full /tmp/test.aero
# Control-plane integrity (IHSH): aerotensor validate --control /tmp/test.aero
```

## Integrity (IHSH & PHSH)

- **IHSH (AIP-0001):** Optional control-plane digest. Use `add_control_integrity=True` when writing and `validate --control` when reading. See [AIP-0001-IHSH.md](../docs/AIP-0001-IHSH.md).
- **PHSH (AIP-0002):** Optional per-page hashes per WTSH shard. Use `add_page_hashes=True` when writing; `validate_all()` verifies PHSH when present. See [AIP-0002-PHSH.md](../docs/AIP-0002-PHSH.md).

This repo also contains the **aero** meta package under `packaging/aero` (one install for format + runtime). The main AERO project lives at [Aero-HQ/Aero](https://github.com/Aero-HQ/Aero).
