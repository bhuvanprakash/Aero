"""Multi-shard write_model tests."""

import numpy as np
import pytest

from aerotensor.format import DType, FOURCC_WTSH
from aerotensor.reader import AeroReader
from aerotensor.writer import AeroWriter, TensorSpec


def _specs():
    """Three small tensors, each 12 bytes (3 x f32)."""
    arrays = [
        np.array([1.0, 2.0, 3.0], dtype="<f4"),
        np.array([4.0, 5.0, 6.0], dtype="<f4"),
        np.array([7.0, 8.0, 9.0], dtype="<f4"),
    ]
    specs = []
    for i, a in enumerate(arrays):
        raw = a.tobytes()
        specs.append(TensorSpec(
            name=f"t{i}", dtype=int(DType.F32), shape=[3],
            data_len=len(raw), read_data=lambda r=raw: r,
        ))
    return specs


@pytest.fixture
def multishard_aero(tmp_path):
    out = str(tmp_path / "multi.aero")
    writer = AeroWriter(file_uuid=b"\x33" * 16)
    # 12 bytes per tensor, cap at 15 → one tensor per shard
    writer.write_model(out, _specs(), "test", "mlp", max_shard_bytes=15)
    return out


def test_multiple_wtsh_chunks(multishard_aero):
    with AeroReader(multishard_aero) as r:
        wtsh = [c for c in r.list_chunks() if c.fourcc == FOURCC_WTSH]
        assert len(wtsh) == 3
        names = {c.name for c in wtsh}
        assert names == {"weights.shard0", "weights.shard1", "weights.shard2"}


def test_tensor_readback(multishard_aero):
    with AeroReader(multishard_aero) as r:
        for i in range(3):
            raw = r.read_tensor_bytes(f"t{i}")
            arr = np.frombuffer(raw, dtype="<f4")
            expected = np.array([1 + 3 * i, 2 + 3 * i, 3 + 3 * i], dtype="<f4")
            np.testing.assert_array_equal(arr, expected)


def test_validate_multishard(multishard_aero):
    with AeroReader(multishard_aero) as r:
        assert r.validate_all() == []


def test_find_shard_chunk(multishard_aero):
    with AeroReader(multishard_aero) as r:
        for sid in range(3):
            ref = r.find_shard_chunk(sid)
            assert ref is not None
            assert ref.name == f"weights.shard{sid}"


def test_single_shard_when_cap_large(tmp_path):
    """All tensors fit in one shard when cap is large."""
    out = str(tmp_path / "single.aero")
    writer = AeroWriter(file_uuid=b"\x44" * 16)
    writer.write_model(out, _specs(), "test", "mlp", max_shard_bytes=1 << 30)
    with AeroReader(out) as r:
        wtsh = [c for c in r.list_chunks() if c.fourcc == FOURCC_WTSH]
        assert len(wtsh) == 1
