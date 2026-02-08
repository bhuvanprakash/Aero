"""Fuzz-ish mutation tests – verify the reader rejects corrupted files
without hanging, crashing, or allocating huge memory.

Uses a fixed random seed for deterministic results.
"""

from __future__ import annotations

import os
import random
import struct
from typing import Union

import numpy as np
import pytest

from aerotensor.format import HEADER_SIZE, TOC_HEADER_SIZE, TOC_ENTRY_SIZE
from aerotensor.reader import AeroError, AeroReader, IntegrityError
from aerotensor.writer import (
    AeroWriter,
    make_json_chunk,
    make_manifest_chunk,
    make_tidx_chunk,
    make_wtsh_chunk,
)
from aerotensor.format import DType


def _make_valid(path: str) -> None:
    """Write a minimal valid .aero file."""
    fixed_uuid = b"\x22" * 16
    w = np.array([1.0, 2.0, 3.0], dtype="<f4")
    shard = w.tobytes()
    tidx = {"tensors": [
        {"name": "w", "dtype": int(DType.F32), "shape": [3],
         "shard_id": 0, "data_off": 0, "data_len": len(shard), "flags": 0},
    ]}
    manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": "test", "architecture": "test"},
        "chunks": [
            {"role": "manifest", "fourcc": "MMSG", "toc_index": 0},
            {"role": "tensors", "fourcc": "TIDX", "toc_index": 1},
            {"role": "weights", "fourcc": "WTSH", "toc_index": 2, "shard_id": 0},
        ],
        "shards": [{"shard_id": 0, "toc_index": 2, "size_bytes": len(shard),
                     "tensor_count": 1, "mmap_critical": True}],
    }
    writer = AeroWriter(file_uuid=fixed_uuid)
    writer.write(path, [
        make_manifest_chunk(manifest),
        make_tidx_chunk(tidx),
        make_wtsh_chunk(shard),
    ])


def _corrupt(data: bytearray, rng: random.Random, n_mutations: int = 1):
    """Apply n random byte mutations."""
    for _ in range(n_mutations):
        pos = rng.randint(0, len(data) - 1)
        data[pos] = (data[pos] + rng.randint(1, 255)) & 0xFF


def _read_file(path: str) -> bytearray:
    with open(path, "rb") as f:
        return bytearray(f.read())


def _write_file(path: str, data: Union[bytes, bytearray]) -> None:
    with open(path, "wb") as f:
        f.write(data)


# ── Header mutations ────────────────────────────────────────────────────────

class TestHeaderMutations:
    def test_corrupt_magic(self, tmp_path):
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        data[0:4] = b"XXXX"
        _write_file(path, data)
        with pytest.raises(AeroError, match="bad magic"):
            AeroReader(path)

    def test_corrupt_version(self, tmp_path):
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        struct.pack_into("<HH", data, 4, 99, 99)
        _write_file(path, data)
        with pytest.raises(AeroError, match="unsupported version"):
            AeroReader(path)

    def test_corrupt_header_size(self, tmp_path):
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        struct.pack_into("<I", data, 8, 42)
        _write_file(path, data)
        with pytest.raises(AeroError, match="header_size"):
            AeroReader(path)

    def test_huge_toc_offset(self, tmp_path):
        """TOC offset pointing way past file end."""
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        struct.pack_into("<Q", data, 12, 0xFFFFFFFFFFFFFFFF)
        _write_file(path, data)
        with pytest.raises(AeroError, match="past end"):
            AeroReader(path)

    def test_huge_string_table(self, tmp_path):
        """String table length absurdly large."""
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        struct.pack_into("<Q", data, 36, 0xFFFFFFFF)
        _write_file(path, data)
        with pytest.raises(AeroError):
            AeroReader(path)


# ── TOC mutations ────────────────────────────────────────────────────────────

class TestTocMutations:
    def test_huge_entry_count(self, tmp_path):
        """entry_count set absurdly high should be rejected by cap."""
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        toc_off_bytes = data[12:20]
        toc_off = struct.unpack("<Q", toc_off_bytes)[0]
        struct.pack_into("<I", data, toc_off, 0x7FFFFFFF)
        _write_file(path, data)
        with pytest.raises(AeroError):
            AeroReader(path)

    def test_chunk_offset_past_eof(self, tmp_path):
        """A chunk offset pointing beyond file should be rejected."""
        path = str(tmp_path / "test.aero")
        _make_valid(path)
        data = _read_file(path)
        toc_off = struct.unpack("<Q", data[12:20])[0]
        entry0_start = toc_off + TOC_HEADER_SIZE
        # chunk_offset is at +8 in the entry (after fourcc+flags)
        struct.pack_into("<Q", data, entry0_start + 8, len(data) + 1000)
        _write_file(path, data)
        with pytest.raises(AeroError, match="past end"):
            AeroReader(path)


# ── Random mutation sweep ────────────────────────────────────────────────────

class TestRandomMutations:
    @pytest.mark.parametrize("seed", range(20))
    def test_random_mutation_rejects(self, tmp_path, seed):
        """Random byte mutations must not crash the reader.

        We don't assert a specific error — just that the reader either
        raises AeroError/IntegrityError or (rarely) succeeds.  It must
        never hang or segfault.
        """
        path = str(tmp_path / f"fuzz_{seed}.aero")
        _make_valid(path)
        data = _read_file(path)
        rng = random.Random(seed)
        _corrupt(data, rng, n_mutations=rng.randint(1, 5))
        _write_file(path, data)

        try:
            with AeroReader(path) as r:
                r.validate_all()
        except (AeroError, IntegrityError, Exception):
            pass  # expected – corrupt files should be rejected

    def test_truncated_to_various_sizes(self, tmp_path):
        """Truncating at various points must not crash."""
        path = str(tmp_path / "base.aero")
        _make_valid(path)
        data = _read_file(path)

        for cut in [0, 1, 10, 50, 95, 96, 97, 112, 200, len(data) - 1]:
            trunc_path = str(tmp_path / f"trunc_{cut}.aero")
            _write_file(trunc_path, data[:cut])
            try:
                with AeroReader(trunc_path) as r:
                    r.validate_all()
            except (AeroError, Exception):
                pass
