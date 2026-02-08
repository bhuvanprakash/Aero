# aerotensor

Python package for **AERO** – a chunked model container format for ML weights (single-file `.aero` or multi-file AEROSET).

Full documentation and quick start: [README in repository root](../README.md).

```bash
pip install -e .
aerotensor make-test-vector /tmp/test.aero
aerotensor validate --full /tmp/test.aero
```
