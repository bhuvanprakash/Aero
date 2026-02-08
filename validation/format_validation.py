#!/usr/bin/env python3
"""
AERO format validation: write a minimal model (single file, sharded, AEROSET),
then run validate, read-back, metadata roundtrip, dtypes, determinism,
chunk layout, corruption/truncation checks, AEROSET SHA-256, and duplicate-name
rejection. Exit 0 if all pass, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

_REPO_PY = os.path.join(os.path.dirname(__file__), "..", "py")
if os.path.isdir(_REPO_PY) and _REPO_PY not in sys.path:
    sys.path.insert(0, _REPO_PY)

import numpy as np

from aerotensor import AeroReader, AeroSetReader, AeroWriter
from aerotensor.format import DType, FOURCC_MJSN, FOURCC_MMSG, FOURCC_TIDX, FOURCC_WTSH
from aerotensor.writer import TensorSpec
from aerotensor.aeroset import write_aeroset
import json as _json


MODEL_NAME = "demo-model"
ARCHITECTURE = "demo"


def make_demo_tensors():
    """Build TensorSpecs with multiple dtypes and one oversized tensor."""
    specs = []
    # F32 embedding
    embed = np.arange(8, dtype=np.float32)
    raw = embed.tobytes()
    specs.append(
        TensorSpec(
            name="embed.weight",
            dtype=int(DType.F32),
            shape=list(embed.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    # F32 layer
    wq = np.ones((4, 4), dtype=np.float32) * 0.5
    raw = wq.tobytes()
    specs.append(
        TensorSpec(
            name="layer.0.wq",
            dtype=int(DType.F32),
            shape=list(wq.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    wv = np.zeros((4, 4), dtype=np.float32)
    wv[0, 0] = 1.0
    raw = wv.tobytes()
    specs.append(
        TensorSpec(
            name="layer.0.wv",
            dtype=int(DType.F32),
            shape=list(wv.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    # F16 (half) tensor
    scale = np.array([1.0, 2.0], dtype=np.float16)
    raw = scale.tobytes()
    specs.append(
        TensorSpec(
            name="layer.0.scale",
            dtype=int(DType.F16),
            shape=list(scale.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    # I32 tensor (e.g. indices)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    raw = indices.tobytes()
    specs.append(
        TensorSpec(
            name="layer.0.indices",
            dtype=int(DType.I32),
            shape=list(indices.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    # One tensor larger than 64 bytes (tests single-tensor shard when max_shard_bytes=64)
    big = np.ones((20, 4), dtype=np.float32)  # 320 bytes
    raw = big.tobytes()
    specs.append(
        TensorSpec(
            name="layer.0.big",
            dtype=int(DType.F32),
            shape=list(big.shape),
            data_len=len(raw),
            read_data=lambda r=raw: r,
        )
    )
    # Extra dtypes: BF16, F64, I8, U8
    bf16 = np.array([0x3F80, 0x4000], dtype=np.uint16)  # 1.0, 2.0 bf16
    raw = bf16.tobytes()
    specs.append(TensorSpec("layer.0.bf16", int(DType.BF16), [2], len(raw), lambda r=raw: r))
    f64 = np.array([1.0, 2.0], dtype=np.float64)
    raw = f64.tobytes()
    specs.append(TensorSpec("layer.0.f64", int(DType.F64), list(f64.shape), len(raw), lambda r=raw: r))
    i8 = np.array([-1, 0, 1], dtype=np.int8)
    raw = i8.tobytes()
    specs.append(TensorSpec("layer.0.i8", int(DType.I8), list(i8.shape), len(raw), lambda r=raw: r))
    u8 = np.array([0, 1, 255], dtype=np.uint8)
    raw = u8.tobytes()
    specs.append(TensorSpec("layer.0.u8", int(DType.U8), list(u8.shape), len(raw), lambda r=raw: r))
    return specs


def run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run aerotensor CLI; return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "aerotensor.cli"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.returncode, result.stdout, result.stderr


def file_sha256(path: str) -> str:
    """Return SHA-256 hex digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_metadata_roundtrip(path: str) -> bool:
    """Verify manifest has expected model name and architecture."""
    with AeroReader(path) as r:
        manifest = r.read_msgpack(FOURCC_MMSG)
        model = manifest.get("model", {})
        name = model.get("name")
        arch = model.get("architecture")
        if name != MODEL_NAME or arch != ARCHITECTURE:
            print(f"   FAIL metadata: expected name={MODEL_NAME!r} arch={ARCHITECTURE!r}, got name={name!r} arch={arch!r}")
            return False
    return True


def check_chunk_structure(path: str) -> bool:
    """Verify expected chunks exist: MJSN, MMSG, TIDX, at least one WTSH."""
    with AeroReader(path) as r:
        chunks = r.list_chunks()
        fourccs = {c.fourcc for c in chunks}
        if FOURCC_MJSN not in fourccs or FOURCC_MMSG not in fourccs or FOURCC_TIDX not in fourccs:
            print(f"   FAIL chunks: missing MJSN/MMSG/TIDX, got {fourccs}")
            return False
        wtsh = [c for c in chunks if c.fourcc == FOURCC_WTSH]
        if not wtsh:
            print("   FAIL chunks: no WTSH chunk")
            return False
    return True


def check_numpy_roundtrip(path: str, ref_arrays: dict) -> bool:
    """Read tensor bytes, view as numpy with stored dtype/shape, compare to ref."""
    with AeroReader(path) as r:
        for t in r.list_tensors():
            name = t["name"]
            if name not in ref_arrays:
                continue
            raw = r.read_tensor_bytes(name)
            dtype_id = t["dtype"]
            shape = t["shape"]
            _dt = {
                0: np.float16, 1: np.float32, 2: np.uint16, 3: np.float64,
                4: np.int8, 5: np.uint8, 6: np.int16, 7: np.uint16,
                8: np.int32, 9: np.uint32, 10: np.int64, 11: np.uint64, 12: np.bool_,
            }
            dtype = _dt.get(dtype_id)
            if dtype is None:
                continue
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
            ref = ref_arrays[name]
            if not np.array_equal(arr, ref):
                print(f"   FAIL numpy roundtrip: {name} values differ")
                return False
    return True


def check_list_tensors_match(path: str, specs: list) -> bool:
    """Every list_tensors() entry must match a spec (name, dtype, shape)."""
    expected = {(s.name, s.dtype, tuple(s.shape)) for s in specs}
    with AeroReader(path) as r:
        for t in r.list_tensors():
            key = (t["name"], t["dtype"], tuple(t["shape"]))
            if key not in expected:
                print(f"   FAIL list_tensors: unexpected {key}")
                return False
            expected.discard(key)
    if expected:
        print(f"   FAIL list_tensors: missing {expected}")
        return False
    return True


def check_mjsn_if_present(path: str) -> bool:
    """If MJSN chunk exists, it must be valid JSON."""
    with AeroReader(path) as r:
        c = r.get_chunk(FOURCC_MJSN)
        if c is None:
            return True
        raw = r.read_chunk_bytes(c, verify_hash=True)
        try:
            _json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"   FAIL MJSN: {e}")
            return False
    return True


def check_corruption_n_offsets(path: str, n: int = 3) -> bool:
    """Corrupt at n chunk-payload offsets; validate must fail each time."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    with AeroReader(path) as r:
        ranges = [(c.offset, c.offset + c.length) for c in r.list_chunks()]
    seen = set()
    offsets = []
    for start, end in ranges:
        if start < end and end <= len(data):
            mid = (start + end) // 2
            for off in (start, start + 1, mid, end - 1):
                if start <= off < end and off not in seen:
                    seen.add(off)
                    offsets.append(off)
            if len(offsets) >= n:
                break
    if len(offsets) < n:
        return True
    for off in offsets[:n]:
        corrupted = bytearray(data)
        corrupted[off] ^= 0xFF
        tmp = path + ".corrupt_demo"
        with open(tmp, "wb") as f:
            f.write(corrupted)
        rc, _, _ = run_cli(["validate", tmp, "--full"])
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if rc == 0:
            print(f"   FAIL corruption at offset {off} was accepted")
            return False
    return True


def check_truncation_rejected(path: str) -> bool:
    """Truncated file must fail validate (no crash)."""
    with open(path, "rb") as f:
        data = f.read()
    trunc_at = len(data) // 2
    if trunc_at <= 0:
        return True
    tmp = path + ".trunc_demo"
    with open(tmp, "wb") as f:
        f.write(data[:trunc_at])
    try:
        rc, _, _ = run_cli(["validate", tmp, "--full"])
        return rc != 0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def check_view_vs_bytes(path: str, tensor_name: str, expected_bytes: bytes) -> bool:
    """read_tensor_view bytes must match read_tensor_bytes."""
    with AeroReader(path) as r:
        direct = r.read_tensor_bytes(tensor_name)
        view = r.read_tensor_view(tensor_name)
        try:
            view_bytes = bytes(view.view)
            if view_bytes != direct or view_bytes != expected_bytes:
                print("   FAIL view vs bytes mismatch")
                return False
        finally:
            view.close()
    return True


def main() -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    single_path = os.path.join(base, "demo_single.aero")
    sharded_path = os.path.join(base, "demo_sharded.aero")
    single2_path = os.path.join(base, "demo_single2.aero")
    aeroset_dir = os.path.join(base, "demo_aeroset")

    specs = make_demo_tensors()
    uuid_bytes = b"\x11\x22\x33\x44" * 4

    ref_arrays = {
        "embed.weight": np.arange(8, dtype=np.float32),
        "layer.0.wq": np.ones((4, 4), dtype=np.float32) * 0.5,
        "layer.0.wv": np.zeros((4, 4), dtype=np.float32),
        "layer.0.scale": np.array([1.0, 2.0], dtype=np.float16),
        "layer.0.indices": np.array([0, 1, 2, 3], dtype=np.int32),
        "layer.0.big": np.ones((20, 4), dtype=np.float32),
        "layer.0.bf16": np.array([0x3F80, 0x4000], dtype=np.uint16),
        "layer.0.f64": np.array([1.0, 2.0], dtype=np.float64),
        "layer.0.i8": np.array([-1, 0, 1], dtype=np.int8),
        "layer.0.u8": np.array([0, 1, 255], dtype=np.uint8),
    }
    ref_arrays["layer.0.wv"][0, 0] = 1.0

    failed = []
    print("AERO format validation")
    print("=" * 56)

    print("\n1. Write single-file AERO: demo_single.aero")
    writer = AeroWriter(file_uuid=uuid_bytes)
    writer.write_model(
        single_path, specs, MODEL_NAME, ARCHITECTURE, max_shard_bytes=2 << 30,
    )
    print("   Done.")

    print("\n2. Write multi-shard AERO: demo_sharded.aero (max 64 B/shard)")
    writer = AeroWriter(file_uuid=uuid_bytes)
    writer.write_model(
        sharded_path, specs, MODEL_NAME, ARCHITECTURE, max_shard_bytes=64,
    )
    print("   Done.")

    print("\n3. Determinism: write same model again → same file hash")
    writer = AeroWriter(file_uuid=uuid_bytes)
    writer.write_model(
        single2_path, specs, MODEL_NAME, ARCHITECTURE, max_shard_bytes=2 << 30,
    )
    h1 = file_sha256(single_path)
    h2 = file_sha256(single2_path)
    if h1 != h2:
        print(f"   FAIL determinism: hashes differ\n      {h1}\n      {h2}")
        failed.append("determinism")
    else:
        print("   OK   same hash")
    try:
        os.unlink(single2_path)
    except OSError:
        pass

    print("\n4. Write AEROSET: demo_aeroset/")
    os.makedirs(aeroset_dir, exist_ok=True)
    write_aeroset(
        aeroset_dir, specs, MODEL_NAME, ARCHITECTURE,
        max_shard_bytes=64, max_part_shards=2, uuid=uuid_bytes,
    )
    print("   Done.")

    print("\n5. Validate (CLI --full)")
    for label, path in [
        ("demo_single.aero", single_path),
        ("demo_sharded.aero", sharded_path),
        ("AEROSET", os.path.join(aeroset_dir, "model.aeroset.json")),
    ]:
        rc, out, err = run_cli(["validate", path, "--full"])
        if rc != 0:
            print(f"   FAIL {label}: {err or out}")
            failed.append(f"validate_{label}")
        else:
            print(f"   OK   {label}")

    print("\n6. Metadata roundtrip (model name, architecture)")
    if not check_metadata_roundtrip(single_path):
        failed.append("metadata")
    else:
        print("   OK   single")
    if not check_metadata_roundtrip(sharded_path):
        failed.append("metadata_sharded")
    else:
        print("   OK   sharded")

    print("\n7. Chunk structure (MJSN, MMSG, TIDX, WTSH)")
    if not check_chunk_structure(single_path):
        failed.append("chunks_single")
    else:
        print("   OK   single")
    if not check_chunk_structure(sharded_path):
        failed.append("chunks_sharded")
    else:
        print("   OK   sharded (multiple WTSH)")

    print("\n7b. list_tensors() match (name, dtype, shape) for every tensor")
    if not check_list_tensors_match(single_path, specs):
        failed.append("list_tensors")
    else:
        print("   OK")

    print("\n7c. MJSN valid JSON (if present)")
    if not check_mjsn_if_present(single_path):
        failed.append("mjsn")
    else:
        print("   OK")

    print("\n8. Read-back: bytes match across single / sharded / aeroset")
    ref_bytes = {}
    with AeroReader(single_path) as r:
        for t in r.list_tensors():
            ref_bytes[t["name"]] = r.read_tensor_bytes(t["name"])

    for name in ref_bytes:
        with AeroReader(sharded_path) as r:
            got = r.read_tensor_bytes(name)
            if got != ref_bytes[name]:
                print(f"   FAIL sharded: {name}")
                failed.append("readback_sharded")
                break
    else:
        print("   OK   sharded")

    aeroset_index = os.path.join(aeroset_dir, "model.aeroset.json")
    with AeroSetReader(aeroset_index) as r:
        for name in ref_bytes:
            got = r.read_tensor_bytes(name)
            if got != ref_bytes[name]:
                print(f"   FAIL aeroset: {name}")
                failed.append("readback_aeroset")
                break
        else:
            print("   OK   aeroset")

    print("\n9. NumPy roundtrip (bytes → array → compare to original)")
    if not check_numpy_roundtrip(single_path, ref_arrays):
        failed.append("numpy_roundtrip")
    else:
        print("   OK")

    print("\n10. Corruption detection (twelve payload offsets → validate must fail)")
    if not check_corruption_n_offsets(single_path, n=12):
        failed.append("corruption")
    else:
        print("   OK   all twelve corruptions rejected")

    print("\n10b. Truncation (truncated file must fail validate)")
    if not check_truncation_rejected(single_path):
        failed.append("truncation")
    else:
        print("   OK   truncated file rejected")

    print("\n10c. read_tensor_view bytes match read_tensor_bytes")
    ref_bytes_one = {}
    with AeroReader(single_path) as r:
        for t in r.list_tensors():
            ref_bytes_one[t["name"]] = r.read_tensor_bytes(t["name"])
    first_name = next(iter(ref_bytes_one))
    if not check_view_vs_bytes(single_path, first_name, ref_bytes_one[first_name]):
        failed.append("view_vs_bytes")
    else:
        print("   OK   view matches bytes")

    print("\n10d. CLI inspect exit 0")
    rc, _, _ = run_cli(["inspect", single_path])
    if rc != 0:
        failed.append("cli_inspect")
    else:
        print("   OK")

    print("\n10e. Sharded determinism (write sharded twice → same hash)")
    sharded2_path = os.path.join(base, "demo_sharded2.aero")
    writer = AeroWriter(file_uuid=uuid_bytes)
    writer.write_model(sharded2_path, specs, MODEL_NAME, ARCHITECTURE, max_shard_bytes=64)
    h1 = file_sha256(sharded_path)
    h2 = file_sha256(sharded2_path)
    try:
        os.unlink(sharded2_path)
    except OSError:
        pass
    if h1 != h2:
        print("   FAIL sharded hashes differ")
        failed.append("sharded_determinism")
    else:
        print("   OK   same hash")

    print("\n10f. list_tensors() names identical (single vs sharded vs aeroset)")
    with AeroReader(single_path) as r:
        names_single = {t["name"] for t in r.list_tensors()}
    with AeroReader(sharded_path) as r:
        names_sharded = {t["name"] for t in r.list_tensors()}
    with AeroSetReader(aeroset_index) as r:
        names_aeroset = {t["name"] for t in r.list_tensors()}
    if names_single != names_sharded or names_single != names_aeroset:
        print("   FAIL name sets differ")
        failed.append("names_identical")
    else:
        print("   OK   same names")

    print("\n10g. Every tensor: read_tensor_view bytes == read_tensor_bytes")
    with AeroReader(single_path) as r:
        for t in r.list_tensors():
            name = t["name"]
            direct = r.read_tensor_bytes(name)
            view = r.read_tensor_view(name)
            try:
                view_bytes = bytes(view.view)
                if view_bytes != direct:
                    print(f"   FAIL view vs bytes: {name}")
                    failed.append("all_view_bytes")
                    break
            finally:
                view.close()
        else:
            print("   OK   all tensors")

    print("\n10h. AEROSET validate (SHA-256 + shard refs)")
    with AeroSetReader(aeroset_index) as r:
        errors = r.validate()
    if errors:
        for e in errors:
            print(f"   FAIL: {e}")
        failed.append("aeroset_validate")
    else:
        print("   OK   AEROSET SHA-256 + shard refs validated")

    print("\n10i. Duplicate tensor name → writer must reject")
    dup_specs = [
        TensorSpec(name="dup", dtype=int(DType.F32), shape=[1],
                   data_len=4, read_data=lambda: b"\x00" * 4),
        TensorSpec(name="dup", dtype=int(DType.F32), shape=[1],
                   data_len=4, read_data=lambda: b"\x00" * 4),
    ]
    try:
        dup_path = os.path.join(base, "demo_dup.aero")
        w = AeroWriter(file_uuid=uuid_bytes)
        w.write_model(dup_path, dup_specs, "dup", "test")
        print("   FAIL duplicate tensor name accepted")
        failed.append("dup_tensor_reject")
        try:
            os.unlink(dup_path)
        except OSError:
            pass
    except ValueError:
        print("   OK   rejected with ValueError")

    print("\n" + "=" * 56)
    if failed:
        print("RESULT: FAILED –", ", ".join(failed))
        return 1
    print("RESULT: All checks passed.")
    print("\nGenerated: demo_single.aero, demo_sharded.aero, demo_aeroset/")
    print("Try: python -m aerotensor.cli inspect demo_single.aero")
    print("     python -m aerotensor.cli inspect-set demo_aeroset/model.aeroset.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
