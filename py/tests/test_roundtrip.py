"""Round-trip tests for AERO v0.1 writer + reader."""

import numpy as np
import pytest

from aerotensor.format import (
    DType,
    FOURCC_MJSN,
    FOURCC_MMSG,
    FOURCC_TIDX,
    FOURCC_WTSH,
)
from aerotensor.reader import AeroReader
from aerotensor.writer import (
    AeroWriter,
    make_json_chunk,
    make_manifest_chunk,
    make_tidx_chunk,
    make_wtsh_chunk,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def test_aero(tmp_path):
    """Create a golden-style .aero file with 2 tensors."""
    out = str(tmp_path / "test.aero")
    fixed_uuid = b"\x11" * 16

    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    bias = np.array([0.5, -0.5], dtype="<f4")
    weight_bytes = weight.tobytes()
    bias_bytes = bias.tobytes()
    shard_data = weight_bytes + bias_bytes

    tidx = {
        "tensors": [
            {
                "name": "linear.weight",
                "dtype": int(DType.F32),
                "shape": [2, 3],
                "shard_id": 0,
                "data_off": 0,
                "data_len": len(weight_bytes),
                "flags": 0,
            },
            {
                "name": "linear.bias",
                "dtype": int(DType.F32),
                "shape": [2],
                "shard_id": 0,
                "data_off": len(weight_bytes),
                "data_len": len(bias_bytes),
                "flags": 0,
            },
        ]
    }
    manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": "test-linear", "architecture": "linear"},
        "chunks": [
            {"role": "json", "fourcc": "MJSN", "toc_index": 0},
            {"role": "manifest", "fourcc": "MMSG", "toc_index": 1},
            {"role": "tensors", "fourcc": "TIDX", "toc_index": 2},
            {"role": "weights", "fourcc": "WTSH", "toc_index": 3, "shard_id": 0},
        ],
        "shards": [
            {
                "shard_id": 0,
                "toc_index": 3,
                "size_bytes": len(shard_data),
                "tensor_count": 2,
                "mmap_critical": True,
            },
        ],
    }
    meta = {"description": "test", "model_name": "test-linear"}

    chunks = [
        make_json_chunk(meta),
        make_manifest_chunk(manifest),
        make_tidx_chunk(tidx),
        make_wtsh_chunk(shard_data),
    ]
    AeroWriter(file_uuid=fixed_uuid).write(out, chunks)
    return out


# ── Tests ───────────────────────────────────────────────────────────────────


def test_list_chunks(test_aero):
    with AeroReader(test_aero) as r:
        chunks = r.list_chunks()
        fourccs = {c.fourcc for c in chunks}
        assert FOURCC_MJSN in fourccs
        assert FOURCC_MMSG in fourccs
        assert FOURCC_TIDX in fourccs
        assert FOURCC_WTSH in fourccs


def test_list_tensors(test_aero):
    with AeroReader(test_aero) as r:
        tensors = r.list_tensors()
        assert len(tensors) == 2
        names = {t["name"] for t in tensors}
        assert "linear.weight" in names
        assert "linear.bias" in names


def test_tensor_bytes(test_aero):
    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    bias = np.array([0.5, -0.5], dtype="<f4")

    with AeroReader(test_aero) as r:
        assert r.read_tensor_bytes("linear.weight") == weight.tobytes()
        assert r.read_tensor_bytes("linear.bias") == bias.tobytes()


def test_reconstruct_numpy(test_aero):
    with AeroReader(test_aero) as r:
        w = np.frombuffer(
            r.read_tensor_bytes("linear.weight"), dtype="<f4"
        ).reshape(2, 3)
        np.testing.assert_array_equal(w, [[1, 2, 3], [4, 5, 6]])

        b = np.frombuffer(
            r.read_tensor_bytes("linear.bias"), dtype="<f4"
        )
        np.testing.assert_array_equal(b, [0.5, -0.5])


def test_uuid_and_version(test_aero):
    with AeroReader(test_aero) as r:
        assert r.uuid == b"\x11" * 16
        assert r.version == (0, 1)


def test_validate_all(test_aero):
    with AeroReader(test_aero) as r:
        errors = r.validate_all()
        assert errors == []


def test_determinism(tmp_path):
    """Same inputs ⇒ identical bytes."""
    uuid = b"\x22" * 16
    weight = np.array([1.0, 2.0], dtype="<f4")
    shard = weight.tobytes()
    tidx = {
        "tensors": [
            {
                "name": "w",
                "dtype": int(DType.F32),
                "shape": [2],
                "shard_id": 0,
                "data_off": 0,
                "data_len": len(shard),
                "flags": 0,
            }
        ]
    }
    manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": "t", "architecture": "t"},
        "chunks": [],
        "shards": [],
    }
    meta = {"k": "v"}

    def _write(p):
        chunks = [
            make_json_chunk(meta),
            make_manifest_chunk(manifest),
            make_tidx_chunk(tidx),
            make_wtsh_chunk(shard),
        ]
        AeroWriter(file_uuid=uuid).write(str(p), chunks)

    p1 = tmp_path / "a.aero"
    p2 = tmp_path / "b.aero"
    _write(p1)
    _write(p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_mmap_view(test_aero):
    """read_tensor_view returns a valid memoryview; closing invalidates it."""
    with AeroReader(test_aero) as r:
        view = r.read_tensor_view("linear.weight")
        data = bytes(view.view)
        expected = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype="<f4"
        ).tobytes()
        assert data == expected
        view.close()
