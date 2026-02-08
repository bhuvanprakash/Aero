#!/usr/bin/env python3
"""
Compare AERO with safetensors, GGUF, and optionally ONNX on:
  - Speed (write time, read time) – 9 runs, report avg
  - Accuracy / fidelity (round-trip: same bytes or max float diff)
  - Corruption – AERO: 12 chunk-payload offsets; others: single offset

Runs small, medium, large, xlarge (500 tensors), and xxlarge (1000 tensors). No external model files.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time

# Add repo py/ for aerotensor
_REPO_PY = os.path.join(os.path.dirname(__file__), "..", "py")
if os.path.isdir(_REPO_PY) and _REPO_PY not in sys.path:
    sys.path.insert(0, _REPO_PY)

import numpy as np

from aerotensor import AeroReader, AeroWriter
from aerotensor.format import DType, FOURCC_WTSH
from aerotensor.writer import TensorSpec


# ── Tensor sets: small, medium, large (same for all formats) ─────────────────

def get_tensor_dict_small():
    """Same tensors as demo: F32 + F16 + I32, one larger tensor."""
    wv = np.zeros((4, 4), dtype=np.float32)
    wv[0, 0] = 1.0
    return {
        "embed.weight": np.arange(8, dtype=np.float32),
        "layer.0.wq": np.ones((4, 4), dtype=np.float32) * 0.5,
        "layer.0.wv": wv,
        "layer.0.scale": np.array([1.0, 2.0], dtype=np.float16),
        "layer.0.indices": np.array([0, 1, 2, 3], dtype=np.int32),
        "layer.0.big": np.ones((20, 4), dtype=np.float32),
    }


def get_tensor_dict_medium():
    """~30 tensors, mixed dtypes, one 64KB tensor."""
    rng = np.random.default_rng(42)
    out = {}
    for i in range(8):
        for key in ["wq", "wk", "wv"]:
            name = f"layer.{i}.{key}"
            out[name] = rng.standard_normal((16, 16)).astype(np.float32)
    out["embed.scale"] = np.array([1.0, 2.0], dtype=np.float16)
    out["embed.indices"] = np.arange(10, dtype=np.int32)
    n = 64 * 1024 // 4
    out["embed.large"] = rng.standard_normal(n).astype(np.float32)
    return out


def get_tensor_dict_large():
    """~80 tensors + 0.3 MB tensor."""
    rng = np.random.default_rng(123)
    out = {}
    for i in range(20):
        for key in ["wq", "wk", "wv", "wo"]:
            out[f"layer.{i}.{key}"] = rng.standard_normal((32, 32)).astype(np.float32)
    n = int(0.3 * 1024 * 1024 / 4)
    out["embed.huge"] = rng.standard_normal(n).astype(np.float32)
    return out


def get_tensor_dict_xlarge():
    """500 tensors + 0.5 MB tensor (stress count and size)."""
    rng = np.random.default_rng(456)
    out = {}
    for i in range(125):
        for key in ["wq", "wk", "wv", "wo"]:
            out[f"layer.{i}.{key}"] = rng.standard_normal((16, 16)).astype(np.float32)
    n = int(0.5 * 1024 * 1024 / 4)
    out["embed.xhuge"] = rng.standard_normal(n).astype(np.float32)
    return out


def get_tensor_dict_xxlarge():
    """1000 tensors + 0.3 MB tensor (extra stress scale)."""
    rng = np.random.default_rng(789)
    out = {}
    for i in range(250):
        for key in ["wq", "wk", "wv", "wo"]:
            out[f"layer.{i}.{key}"] = rng.standard_normal((12, 12)).astype(np.float32)
    n = int(0.3 * 1024 * 1024 / 4)
    out["embed.xxhuge"] = rng.standard_normal(n).astype(np.float32)
    return out


def get_tensor_dict():
    """Alias for small (back compat)."""
    return get_tensor_dict_small()


def tensor_dict_to_ref_bytes(tensors: dict) -> dict[str, bytes]:
    """Reference bytes for fidelity check (name -> raw bytes)."""
    ref = {}
    for name, arr in tensors.items():
        ref[name] = arr.tobytes()
    return ref


# ── Safetensors (minimal write + read, no safetensors package) ───────────────

def _write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    data = bytearray()
    for name in sorted(tensors):
        arr = tensors[name]
        raw = arr.tobytes()
        dtype_map = {np.float32: "F32", np.float16: "F16", np.int32: "I32"}
        dt = dtype_map.get(arr.dtype.type, "F32")
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


def _read_safetensors(path: str) -> dict[str, bytes]:
    """Read all tensor bytes from a safetensors file (no checksum)."""
    from aerotensor.convert.safetensors_to_aero import _parse_safetensors_header

    with open(path, "rb") as fd:
        header, data_off = _parse_safetensors_header(fd)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        length = end - start
        with open(path, "rb") as fd:
            fd.seek(data_off + start)
            out[name] = fd.read(length)
    return out


# ── GGUF (minimal v3 write, read via our converter) ──────────────────────────

def _write_gguf(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Minimal GGUF v3; store as F32 (cast F16/I32 for GGUF compatibility)."""
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", len(tensors)))
    buf.extend(struct.pack("<Q", 0))
    data_parts = []
    offset = 0
    for name in sorted(tensors):
        arr = tensors[name]
        # GGUF minimal writer uses F32 only
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        raw = arr.tobytes()
        name_bytes = name.encode("utf-8")
        buf.extend(struct.pack("<Q", len(name_bytes)))
        buf.extend(name_bytes)
        buf.extend(struct.pack("<I", len(arr.shape)))
        for d in arr.shape:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<I", 0))
        buf.extend(struct.pack("<Q", offset))
        data_parts.append(raw)
        offset += len(raw)
    alignment = 32
    pos = len(buf)
    pad = ((pos + alignment - 1) // alignment) * alignment - pos
    buf.extend(b"\x00" * pad)
    for part in data_parts:
        buf.extend(part)
    with open(path, "wb") as f:
        f.write(buf)


def _read_gguf_via_aero(gguf_path: str, aero_path: str) -> dict[str, bytes]:
    """Convert GGUF -> AERO, then read tensors from AERO."""
    from aerotensor.convert.gguf_to_aero import convert_gguf

    convert_gguf(gguf_path, aero_path)
    with AeroReader(aero_path) as r:
        return {t["name"]: r.read_tensor_bytes(t["name"]) for t in r.list_tensors()}


# ── AERO ────────────────────────────────────────────────────────────────────

def _write_aero(path: str, tensors: dict[str, np.ndarray]) -> None:
    specs = []
    for name in sorted(tensors):
        arr = tensors[name]
        raw = arr.tobytes()
        dtype_map = {np.float32: DType.F32, np.float16: DType.F16, np.int32: DType.I32}
        dt = int(dtype_map.get(arr.dtype.type, DType.F32))
        specs.append(
            TensorSpec(
                name=name, dtype=dt, shape=list(arr.shape),
                data_len=len(raw), read_data=lambda r=raw: r,
            )
        )
    writer = AeroWriter(file_uuid=b"\x11\x22\x33\x44" * 4)
    writer.write_model(path, specs, "compare-model", "demo", max_shard_bytes=2 << 30)


def _read_aero(path: str) -> dict[str, bytes]:
    with AeroReader(path) as r:
        return {t["name"]: r.read_tensor_bytes(t["name"]) for t in r.list_tensors()}


def _validate_aero(path: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "aerotensor.cli", "validate", path, "--full"],
        capture_output=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return r.returncode == 0


def _aero_corruption_detected_at_payload_offsets(path: str, num_offsets: int = 3) -> bool:
    """Corrupt at num_offsets different chunk payload offsets; validate must fail each time."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    with AeroReader(path) as r:
        chunk_ranges = [(c.offset, c.offset + c.length) for c in r.list_chunks()]
    offsets_tried = []
    for start, end in chunk_ranges:
        for off in (start, (start + end) // 2, end - 1):
            if start <= off < end and off < len(data) and off not in offsets_tried:
                offsets_tried.append(off)
                if len(offsets_tried) >= num_offsets:
                    break
        if len(offsets_tried) >= num_offsets:
            break
    for off in offsets_tried[:num_offsets]:
        corrupted = bytearray(data)
        corrupted[off] ^= 0xFF
        tmp = path + ".corrupt_tmp"
        with open(tmp, "wb") as f:
            f.write(corrupted)
        ok = _validate_aero(tmp)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if ok:
            return False
    return True


# ── Fidelity and corruption ─────────────────────────────────────────────────

def fidelity(ref: dict[str, bytes], got: dict[str, bytes]) -> tuple[bool, float]:
    """Exact match and max abs diff (for floats). Returns (exact, max_diff)."""
    if set(ref) != set(got):
        return False, float("nan")
    max_diff = 0.0
    for name in ref:
        if ref[name] != got[name]:
            if len(ref[name]) == len(got[name]):
                a = np.frombuffer(ref[name], dtype=np.uint8)
                b = np.frombuffer(got[name], dtype=np.uint8)
                if np.any(a != b) and len(ref[name]) % 4 == 0:
                    fa = np.frombuffer(ref[name], dtype=np.float32)
                    fb = np.frombuffer(got[name], dtype=np.float32)
                    if len(fa) == len(fb):
                        max_diff = max(max_diff, float(np.max(np.abs(fa - fb))))
            return False, max_diff
    return True, 0.0


def corrupt_file(path: str, offset: int = 500) -> None:
    """Flip one byte at offset (in-place)."""
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        if b:
            f.seek(offset)
            f.write(bytes([b[0] ^ 1]))


# ── ONNX (optional) ──────────────────────────────────────────────────────────

def _onnx_available() -> bool:
    try:
        import onnx
        return True
    except ImportError:
        return False


def _write_onnx(path: str, tensors: dict[str, np.ndarray]) -> None:
    import onnx
    from onnx import TensorProto, helper

    initializers = []
    for name in sorted(tensors):
        arr = tensors[name]
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        t = helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten().tolist())
        initializers.append(t)
    graph = helper.make_graph([], "weights", [], [], initializer=initializers)
    model = helper.make_model(graph)
    onnx.save(model, path)


def _read_onnx(path: str) -> dict[str, bytes]:
    import onnx
    model = onnx.load(path)
    out = {}
    for init in model.graph.initializer:
        from onnx import numpy_helper
        arr = numpy_helper.to_array(init)
        out[init.name] = arr.tobytes()
    return out


# ── Benchmark one format ────────────────────────────────────────────────────

def run_format(
    name: str,
    base_dir: str,
    tensors: dict,
    ref_bytes: dict,
    write_fn,
    read_fn,
    validate_fn=None,
    extra_cleanup: list = None,
    num_runs: int = 3,
) -> dict:
    """Write, read, check fidelity, test corruption. Return metrics dict.
    If num_runs > 1, write/read times are averaged over num_runs."""
    path = os.path.join(base_dir, f"compare_{name}")
    extra = extra_cleanup or []
    results = {"format": name, "write_ms": None, "read_ms": None, "size_bytes": None,
               "fidelity_exact": None, "max_diff": None, "corruption_detected": None}

    # Write
    if name == "gguf":
        path_gguf = path + ".gguf"
        path_aero = path + "_from_gguf.aero"
        t0 = time.perf_counter()
        _write_gguf(path_gguf, tensors)
        results["size_bytes"] = os.path.getsize(path_gguf)
        results["write_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        got = _read_gguf_via_aero(path_gguf, path_aero)
        results["read_ms"] = (time.perf_counter() - t0) * 1000
        # GGUF stores F32 only; compare only F32 tensors for fidelity
        ref_f32 = {k: v for k, v in ref_bytes.items() if tensors[k].dtype == np.float32}
        got_f32 = {k: got[k] for k in ref_f32 if k in got}
        exact, max_diff = fidelity(ref_f32, got_f32)
        results["fidelity_exact"] = exact
        results["max_diff"] = max_diff if not exact else 0.0
        extra = [path_aero]
    else:
        ext = {"aero": ".aero", "safetensors": ".safetensors", "onnx": ".onnx"}[name]
        path = path + ext
        write_times, read_times = [], []
        for _ in range(max(1, num_runs)):
            t0 = time.perf_counter()
            write_fn(path, tensors)
            write_times.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            got = read_fn(path)
            read_times.append((time.perf_counter() - t0) * 1000)
        results["write_ms"] = sum(write_times) / len(write_times)
        results["read_ms"] = sum(read_times) / len(read_times)
        results["size_bytes"] = os.path.getsize(path)

    if name != "gguf":
        exact, max_diff = fidelity(ref_bytes, got)
        results["fidelity_exact"] = exact
        results["max_diff"] = max_diff if not exact else 0.0

    # Corruption: AERO = 12 payload offsets; others = single offset
    if name == "aero" and validate_fn:
        results["corruption_detected"] = _aero_corruption_detected_at_payload_offsets(path, num_offsets=12)
    elif name == "safetensors":
        corrupt_path = path + ".corrupted"
        with open(path, "rb") as f:
            data = bytearray(f.read())
        if len(data) > 600:
            data[600] ^= 1
        with open(corrupt_path, "wb") as f:
            f.write(data)
        try:
            got_c = _read_safetensors(corrupt_path)
            exact_c, _ = fidelity(ref_bytes, got_c)
            results["corruption_detected"] = not exact_c  # no checksum; we detect wrong data
        except Exception:
            results["corruption_detected"] = True
        try:
            os.unlink(corrupt_path)
        except OSError:
            pass
    elif name == "gguf":
        results["corruption_detected"] = None  # skip (converter path)
    elif name == "onnx":
        corrupt_path = path + ".corrupted"
        with open(path, "rb") as f:
            data = bytearray(f.read())
        if len(data) > 600:
            data[600] ^= 1
        with open(corrupt_path, "wb") as f:
            f.write(data)
        try:
            _read_onnx(corrupt_path)
            results["corruption_detected"] = False
        except Exception:
            results["corruption_detected"] = True
        try:
            os.unlink(corrupt_path)
        except OSError:
            pass

    for p in [path, path + ".gguf"] + extra:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def _run_one_size(tensors: dict, ref_bytes: dict, base: str, num_runs: int) -> list[dict]:
    """Run comparison for one tensor set; return list of result dicts."""
    results = []
    results.append(run_format("aero", base, tensors, ref_bytes, _write_aero, _read_aero, _validate_aero, num_runs=num_runs))
    results.append(run_format("safetensors", base, tensors, ref_bytes, _write_safetensors, _read_safetensors, num_runs=num_runs))
    results.append(run_format("gguf", base, tensors, ref_bytes, _write_gguf, None, num_runs=num_runs))
    if _onnx_available():
        results.append(run_format("onnx", base, tensors, ref_bytes, _write_onnx, _read_onnx, num_runs=num_runs))
    return results


def main() -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    num_runs = 9

    print("Format comparison: AERO vs safetensors vs GGUF vs ONNX")
    print("Metrics: speed (write/read ms, avg over {} runs), size, fidelity, corruption (AERO: 12 payload offsets)".format(num_runs))
    print("=" * 72)

    all_passed = True

    for label, get_tensors in [
        ("small", get_tensor_dict_small),
        ("medium", get_tensor_dict_medium),
        ("large", get_tensor_dict_large),
        ("xlarge (500 tensors)", get_tensor_dict_xlarge),
        ("xxlarge (1000 tensors)", get_tensor_dict_xxlarge),
    ]:
        tensors = get_tensors()
        ref_bytes = tensor_dict_to_ref_bytes(tensors)
        print("\n--- {} ({} tensors, {:.1f} KB ref) ---".format(
            label, len(tensors), sum(len(v) for v in ref_bytes.values()) / 1024))
        results = _run_one_size(tensors, ref_bytes, base, num_runs)
        print(f"{'Format':<14} {'Write ms':>10} {'Read ms':>10} {'Size':>10} {'Fidelity':>10} {'Corrupt':>8}")
        print("-" * 72)
        for r in results:
            fid = "exact" if r["fidelity_exact"] else f"diff={r['max_diff']:.2e}"
            corrupt = "yes" if r["corruption_detected"] is True else ("no" if r["corruption_detected"] is False else "-")
            print(f"{r['format']:<14} {r['write_ms']:>10.2f} {r['read_ms']:>10.2f} {r['size_bytes']:>10} {fid:>10} {corrupt:>8}")
        aero = next((x for x in results if x["format"] == "aero"), None)
        if not (aero and aero["fidelity_exact"] and aero["corruption_detected"]):
            print("  FAIL: AERO must have exact fidelity and corruption detection for {}.".format(label))
            all_passed = False
        required_failed = [r["format"] for r in results if r["format"] != "onnx" and not r["fidelity_exact"]]
        if required_failed:
            print("  Fidelity FAIL (required):", required_failed)
            all_passed = False

    if not _onnx_available():
        print("\n(Install 'onnx' for ONNX comparison: pip install onnx)")

    print()
    if all_passed:
        print("AERO: exact round-trip and corruption (12 payload offsets) verified for small/medium/large/xlarge/xxlarge.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
