"""Integration tests: Python writes → C++ reads.

Skips automatically if C++ tools are not built.
"""

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from aerotensor.format import DType
from aerotensor.writer import AeroWriter, TensorSpec
from aerotensor.aeroset import write_aeroset

# ── Locate C++ tools ────────────────────────────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD = os.path.join(_REPO, "cpp", "build")


def _tool(name: str) -> str:
    """Return path to C++ tool binary, or skip if not found."""
    candidates = [
        os.path.join(_BUILD, name),
        os.path.join(_BUILD, "Release", name),
        os.path.join(_BUILD, "Release", name + ".exe"),
        os.path.join(_BUILD, name + ".exe"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    pytest.skip(f"C++ tool {name!r} not found in {_BUILD}")


def _run(tool_path: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [tool_path, *args],
        capture_output=True, text=True, timeout=30,
    )


# ── Single-file tests ──────────────────────────────────────────────────────


class TestSingleFile:
    @pytest.fixture
    def aero_file(self, tmp_path):
        path = str(tmp_path / "test.aero")
        weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
        bias = np.array([0.5, -0.5], dtype="<f4")
        wb, bb = weight.tobytes(), bias.tobytes()
        specs = [
            TensorSpec("linear.weight", int(DType.F32), [2, 3], len(wb),
                       lambda r=wb: r),
            TensorSpec("linear.bias", int(DType.F32), [2], len(bb),
                       lambda r=bb: r),
        ]
        AeroWriter(file_uuid=b"\x11" * 16).write_model(
            path, specs, "test", "linear")
        return path

    def test_validate(self, aero_file):
        result = _run(_tool("aero_validate"), aero_file)
        assert result.returncode == 0, result.stderr

    def test_validate_full(self, aero_file):
        result = _run(_tool("aero_validate"), "--full", aero_file)
        assert result.returncode == 0, result.stderr

    def test_list_tensors(self, aero_file):
        result = _run(_tool("aero_list_tensors"), aero_file)
        assert result.returncode == 0
        assert "linear.weight" in result.stdout
        assert "linear.bias" in result.stdout

    def test_extract_tensor(self, aero_file, tmp_path):
        out = str(tmp_path / "w.bin")
        result = _run(_tool("aero_extract_tensor"), aero_file,
                      "linear.weight", out)
        assert result.returncode == 0, result.stderr
        expected = struct.pack("<6f", 1, 2, 3, 4, 5, 6)
        assert open(out, "rb").read() == expected

    def test_inspect(self, aero_file):
        result = _run(_tool("aero_inspect"), aero_file)
        assert result.returncode == 0
        assert "UUID" in result.stdout
        assert "WTSH" in result.stdout


# ── Multi-shard tests ──────────────────────────────────────────────────────


class TestMultiShard:
    @pytest.fixture
    def multi_file(self, tmp_path):
        path = str(tmp_path / "multi.aero")
        specs = []
        for i in range(3):
            a = np.array([float(i + 1)] * 3, dtype="<f4")
            raw = a.tobytes()
            specs.append(TensorSpec(
                f"t{i}", int(DType.F32), [3], len(raw),
                lambda r=raw: r,
            ))
        AeroWriter(file_uuid=b"\x77" * 16).write_model(
            path, specs, "test", "mlp", max_shard_bytes=15)
        return path

    def test_validate_multi(self, multi_file):
        result = _run(_tool("aero_validate"), multi_file)
        assert result.returncode == 0, result.stderr

    def test_extract_from_shard(self, multi_file, tmp_path):
        out = str(tmp_path / "t1.bin")
        result = _run(_tool("aero_extract_tensor"), multi_file, "t1", out)
        assert result.returncode == 0, result.stderr
        expected = struct.pack("<3f", 2, 2, 2)
        assert open(out, "rb").read() == expected


# ── AEROSET tests ──────────────────────────────────────────────────────────


class TestAeroSet:
    @pytest.fixture
    def aeroset_dir(self, tmp_path):
        out = str(tmp_path / "set")
        specs = []
        for i in range(4):
            a = np.array([float(i)] * 4, dtype="<f4")
            raw = a.tobytes()
            specs.append(TensorSpec(
                f"l{i}", int(DType.F32), [4], len(raw),
                lambda r=raw: r,
            ))
        write_aeroset(out, specs, "test", "mlp",
                      max_shard_bytes=20, max_part_shards=2,
                      uuid=b"\x55" * 16)
        return out

    def test_inspect_set(self, aeroset_dir):
        idx = os.path.join(aeroset_dir, "model.aeroset.json")
        result = _run(_tool("aero_inspect_set"), idx)
        assert result.returncode == 0, result.stderr
        assert "test" in result.stdout

    def test_extract_from_set(self, aeroset_dir, tmp_path):
        idx = os.path.join(aeroset_dir, "model.aeroset.json")
        out = str(tmp_path / "l2.bin")
        result = _run(_tool("aero_extract_tensor_set"), idx, "l2", out)
        assert result.returncode == 0, result.stderr
        expected = struct.pack("<4f", 2, 2, 2, 2)
        assert open(out, "rb").read() == expected
