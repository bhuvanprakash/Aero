"""AEROSET write + read tests."""

import json
import os

import numpy as np
import pytest

from aerotensor.aeroset import AeroSetReader, write_aeroset
from aerotensor.format import DType
from aerotensor.writer import TensorSpec


def _make_specs(n=4):
    specs = []
    for i in range(n):
        a = np.array([float(i)] * 4, dtype="<f4")
        raw = a.tobytes()
        specs.append(TensorSpec(
            name=f"layer.{i}.weight", dtype=int(DType.F32), shape=[4],
            data_len=len(raw), read_data=lambda r=raw: r,
        ))
    return specs


@pytest.fixture
def aeroset_dir(tmp_path):
    out = str(tmp_path / "set")
    write_aeroset(
        out, _make_specs(), "test-set", "transformer",
        max_shard_bytes=20,     # 16 bytes per tensor → 1 per shard
        max_part_shards=2,      # 2 shards per part → 2 parts
        uuid=b"\x55" * 16,
    )
    return out


def test_aeroset_structure(aeroset_dir):
    idx = os.path.join(aeroset_dir, "model.aeroset.json")
    assert os.path.exists(idx)
    with open(idx) as f:
        data = json.load(f)
    assert data["format"]["name"] == "AEROSET"
    assert len(data["parts"]) == 2
    assert os.path.exists(os.path.join(aeroset_dir, "index.aero"))
    assert os.path.exists(os.path.join(aeroset_dir, "part-000.aero"))
    assert os.path.exists(os.path.join(aeroset_dir, "part-001.aero"))


def test_aeroset_reader(aeroset_dir):
    idx = os.path.join(aeroset_dir, "model.aeroset.json")
    with AeroSetReader(idx) as reader:
        tensors = reader.list_tensors()
        assert len(tensors) == 4
        for i in range(4):
            raw = reader.read_tensor_bytes(f"layer.{i}.weight")
            arr = np.frombuffer(raw, dtype="<f4")
            np.testing.assert_array_equal(arr, [float(i)] * 4)


def test_aeroset_model_info(aeroset_dir):
    idx = os.path.join(aeroset_dir, "model.aeroset.json")
    with AeroSetReader(idx) as reader:
        assert reader.model_name == "test-set"
        assert reader.architecture == "transformer"
