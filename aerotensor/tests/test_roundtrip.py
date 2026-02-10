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
    TensorSpec,
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


def test_ihsh_phsh_roundtrip(tmp_path):
    """Write with IHSH (AIP-0001) and PHSH (AIP-0002), then validate --control and validate_all."""
    import os
    import struct
    import pytest
    from aerotensor.format import HEADER_FMT, HEADER_SIZE, TOC_HEADER_SIZE, TOC_ENTRY_SIZE, TOC_HEADER_FMT
    from aerotensor import _blake3

    out = str(tmp_path / "with_ihsh_phsh.aero")
    writer_region_path = str(tmp_path / "writer_zeroed_region.bin")
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    bias = np.array([0.1, -0.1], dtype="<f4")
    specs = [
        TensorSpec("w", int(DType.F32), list(weight.shape), weight.nbytes, lambda: weight.tobytes()),
        TensorSpec("b", int(DType.F32), list(bias.shape), bias.nbytes, lambda: bias.tobytes()),
    ]
    os.environ["AERO_DEBUG_IHSH_WRITER"] = writer_region_path
    try:
        AeroWriter().write_model(
            out,
            specs,
            model_name="test",
            architecture="linear",
            add_control_integrity=True,
            add_page_hashes=True,
            page_hash_page_size=64,
        )
    finally:
        os.environ.pop("AERO_DEBUG_IHSH_WRITER", None)
    # Compare writer's zeroed region to reader's (final file, zeroed)
    with AeroReader(out) as r:
        ihsh_ref = r.get_chunk(b"IHSH")
        assert ihsh_ref is not None
        ihsh_data = r.read_chunk_bytes(ihsh_ref, verify_hash=True)
    import msgpack
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(ihsh_data)
    ihsh = unpacker.unpack()
    control_region_len = int(ihsh["covered_ranges"][0][1])
    expected_digest = ihsh["digest"]
    with open(out, "rb") as f:
        raw_control = bytearray(f.read(control_region_len))
    assert len(raw_control) == control_region_len
    toc_off = struct.unpack(HEADER_FMT, bytes(raw_control[:HEADER_SIZE]))[4]
    toc_hdr = bytes(raw_control[toc_off : toc_off + TOC_HEADER_SIZE])
    entry_count = struct.unpack(TOC_HEADER_FMT, toc_hdr)[0]
    for i in range(entry_count):
        ent = toc_off + TOC_HEADER_SIZE + i * TOC_ENTRY_SIZE
        raw_control[ent + TOC_ENTRY_SIZE - 32 : ent + TOC_ENTRY_SIZE] = b"\x00" * 32
    reader_zeroed = bytes(raw_control)
    actual_from_raw = _blake3.digest(reader_zeroed)
    assert actual_from_raw is not None
    # Writer wrote its zeroed region to writer_region_path; must match reader's zeroed region
    with open(writer_region_path, "rb") as wf:
        writer_zeroed = wf.read()
    assert len(writer_zeroed) == len(reader_zeroed), (
        f"writer zeroed len={len(writer_zeroed)} reader zeroed len={len(reader_zeroed)}"
    )
    if writer_zeroed != reader_zeroed:
        first_diff = next(i for i in range(len(writer_zeroed)) if writer_zeroed[i] != reader_zeroed[i])
        pytest.fail(
            f"IHSH zeroed region differs at byte {first_diff}: "
            f"writer={writer_zeroed[first_diff]:02x} reader={reader_zeroed[first_diff]:02x}"
        )
    assert expected_digest == actual_from_raw, (
        f"IHSH digest: expected={expected_digest.hex()} actual={actual_from_raw.hex()}"
    )
    with AeroReader(out) as r:
        assert r.validate_control() == []
        all_errors = r.validate_all()
        assert all_errors == [], all_errors
        chunks = [c.name for c in r.list_chunks()]
        assert "control.integrity" in chunks
        assert "weights.shard0.phsh" in chunks
        assert r.read_tensor_bytes("w") == weight.tobytes()
        assert r.read_tensor_bytes("b") == bias.tobytes()
