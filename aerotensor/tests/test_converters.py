"""Converter tests – create fake source files and verify round-trip."""

import json
import os
import struct

import numpy as np
import pytest

from aerotensor.format import DType
from aerotensor.reader import AeroReader


# ── Helpers: create fake safetensors ────────────────────────────────────────


def _make_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file."""
    header: dict = {}
    data = bytearray()
    for name in sorted(tensors):
        arr = tensors[name]
        raw = arr.tobytes()
        dtype_map = {np.float32: "F32", np.float16: "F16"}
        dt = dtype_map[arr.dtype.type]
        header[name] = {
            "dtype": dt,
            "shape": list(arr.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data.extend(raw)
    hdr_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(data)


# ── Helpers: create fake GGUF ───────────────────────────────────────────────


def _make_gguf(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal GGUF v3 file with F32 tensors."""
    buf = bytearray()
    buf.extend(b"GGUF")                            # magic
    buf.extend(struct.pack("<I", 3))                # version
    buf.extend(struct.pack("<Q", len(tensors)))     # tensor_count
    buf.extend(struct.pack("<Q", 0))                # kv_count (no metadata)

    # Tensor infos
    data_parts: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = tensors[name]
        raw = arr.tobytes()
        name_bytes = name.encode("utf-8")
        buf.extend(struct.pack("<Q", len(name_bytes)))
        buf.extend(name_bytes)
        buf.extend(struct.pack("<I", len(arr.shape)))       # ndim
        for d in arr.shape:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<I", 0))                    # ggml_type F32
        buf.extend(struct.pack("<Q", offset))               # offset
        data_parts.append(raw)
        offset += len(raw)

    # Alignment (default 32 bytes)
    alignment = 32
    pos = len(buf)
    pad = ((pos + alignment - 1) // alignment) * alignment - pos
    buf.extend(b"\x00" * pad)
    for part in data_parts:
        buf.extend(part)

    with open(path, "wb") as f:
        f.write(buf)


# ── safetensors → aero ──────────────────────────────────────────────────────


def test_convert_safetensors(tmp_path):
    from aerotensor.convert.safetensors_to_aero import convert_safetensors

    st_path = str(tmp_path / "model.safetensors")
    _make_safetensors(st_path, {
        "fc.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "fc.bias": np.array([0.1, 0.2], dtype=np.float32),
    })

    aero_path = str(tmp_path / "model.aero")
    convert_safetensors(st_path, aero_path)

    with AeroReader(aero_path) as r:
        assert r.validate_all() == []
        tensors = r.list_tensors()
        assert len(tensors) == 2
        w = np.frombuffer(r.read_tensor_bytes("fc.weight"), dtype="<f4").reshape(2, 2)
        np.testing.assert_array_equal(w, [[1, 2], [3, 4]])
        b = np.frombuffer(r.read_tensor_bytes("fc.bias"), dtype="<f4")
        np.testing.assert_array_almost_equal(b, [0.1, 0.2])


# ── safetensors sharded → aeroset ──────────────────────────────────────────


def test_convert_safetensors_sharded(tmp_path):
    from aerotensor.convert.safetensors_sharded_to_aeroset import (
        convert_safetensors_sharded,
    )
    from aerotensor.aeroset import AeroSetReader

    # Create two shard files + index.
    shard_dir = str(tmp_path / "shards")
    os.makedirs(shard_dir)
    _make_safetensors(os.path.join(shard_dir, "s-001.safetensors"), {
        "a.weight": np.array([1.0, 2.0], dtype=np.float32),
    })
    _make_safetensors(os.path.join(shard_dir, "s-002.safetensors"), {
        "b.weight": np.array([3.0, 4.0], dtype=np.float32),
    })
    index = {
        "metadata": {},
        "weight_map": {
            "a.weight": "s-001.safetensors",
            "b.weight": "s-002.safetensors",
        },
    }
    with open(os.path.join(shard_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    out_dir = str(tmp_path / "out")
    convert_safetensors_sharded(shard_dir, out_dir)

    with AeroSetReader(os.path.join(out_dir, "model.aeroset.json")) as reader:
        tensors = reader.list_tensors()
        assert len(tensors) == 2
        a = np.frombuffer(reader.read_tensor_bytes("a.weight"), dtype="<f4")
        np.testing.assert_array_equal(a, [1, 2])
        b = np.frombuffer(reader.read_tensor_bytes("b.weight"), dtype="<f4")
        np.testing.assert_array_equal(b, [3, 4])


# ── GGUF → aero ────────────────────────────────────────────────────────────


def test_convert_gguf(tmp_path):
    from aerotensor.convert.gguf_to_aero import convert_gguf

    gguf_path = str(tmp_path / "model.gguf")
    _make_gguf(gguf_path, {
        "blk.0.w": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "blk.1.w": np.array([4.0, 5.0], dtype=np.float32),
    })

    aero_path = str(tmp_path / "model.aero")
    convert_gguf(gguf_path, aero_path)

    with AeroReader(aero_path) as r:
        assert r.validate_all() == []
        tensors = r.list_tensors()
        assert len(tensors) == 2
        a = np.frombuffer(r.read_tensor_bytes("blk.0.w"), dtype="<f4")
        np.testing.assert_array_equal(a, [1, 2, 3])
