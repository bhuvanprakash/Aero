"""Corruption / integrity tests for AERO v0.1."""

import numpy as np
import pytest

from aerotensor.format import DType, FOURCC_MMSG, FOURCC_WTSH
from aerotensor.reader import AeroError, AeroReader, IntegrityError
from aerotensor.writer import (
    AeroWriter,
    make_json_chunk,
    make_manifest_chunk,
    make_tidx_chunk,
    make_wtsh_chunk,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_file(path: str) -> None:
    """Create a tiny valid .aero file."""
    fixed_uuid = b"\x11" * 16
    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    bias = np.array([0.5, -0.5], dtype="<f4")
    shard_data = weight.tobytes() + bias.tobytes()

    tidx = {
        "tensors": [
            {
                "name": "linear.weight",
                "dtype": int(DType.F32),
                "shape": [2, 3],
                "shard_id": 0,
                "data_off": 0,
                "data_len": len(weight.tobytes()),
                "flags": 0,
            },
            {
                "name": "linear.bias",
                "dtype": int(DType.F32),
                "shape": [2],
                "shard_id": 0,
                "data_off": len(weight.tobytes()),
                "data_len": len(bias.tobytes()),
                "flags": 0,
            },
        ]
    }
    manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": "test", "architecture": "linear"},
        "chunks": [],
        "shards": [],
    }
    meta = {"k": "v"}
    chunks = [
        make_json_chunk(meta),
        make_manifest_chunk(manifest),
        make_tidx_chunk(tidx),
        make_wtsh_chunk(shard_data),
    ]
    AeroWriter(file_uuid=fixed_uuid).write(path, chunks)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_corrupt_metadata_chunk(tmp_path):
    """Flipping a byte in a compressed metadata chunk triggers an error."""
    path = str(tmp_path / "corrupt.aero")
    _make_file(path)

    # Find MMSG chunk offset and flip a byte inside the stored payload.
    with AeroReader(path) as r:
        mmsg = r.get_chunk(FOURCC_MMSG)
        assert mmsg is not None
        flip_offset = mmsg.offset + 2  # inside compressed payload

    data = bytearray(open(path, "rb").read())
    data[flip_offset] ^= 0xFF
    with open(path, "wb") as f:
        f.write(data)

    with AeroReader(path) as r:
        mmsg = r.get_chunk(FOURCC_MMSG)
        with pytest.raises((IntegrityError, Exception)):
            r.read_chunk_bytes(mmsg, verify_hash=True)


def test_corrupt_wtsh_chunk(tmp_path):
    """Flipping a byte in WTSH triggers IntegrityError on verify."""
    path = str(tmp_path / "corrupt_w.aero")
    _make_file(path)

    with AeroReader(path) as r:
        wtsh = r.get_chunk(FOURCC_WTSH)
        assert wtsh is not None
        flip_offset = wtsh.offset + 4

    data = bytearray(open(path, "rb").read())
    data[flip_offset] ^= 0xFF
    with open(path, "wb") as f:
        f.write(data)

    with AeroReader(path) as r:
        wtsh = r.get_chunk(FOURCC_WTSH)
        with pytest.raises(IntegrityError):
            r.read_chunk_bytes(wtsh, verify_hash=True)


def test_bad_magic(tmp_path):
    """File with wrong magic is rejected."""
    path = str(tmp_path / "bad.aero")
    _make_file(path)

    data = bytearray(open(path, "rb").read())
    data[0:4] = b"NOPE"
    with open(path, "wb") as f:
        f.write(data)

    with pytest.raises(AeroError, match="bad magic"):
        AeroReader(path)


def test_truncated_file(tmp_path):
    """File shorter than header size is rejected."""
    path = str(tmp_path / "tiny.aero")
    with open(path, "wb") as f:
        f.write(b"AERO" + b"\x00" * 10)

    with pytest.raises(AeroError, match="too small"):
        AeroReader(path)
