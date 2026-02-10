"""AERO v0.2 CLI – inspect, validate, convert, and generate test vectors."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

from . import __version__
from .format import ChunkFlags, DType, DTYPE_NAMES, FOURCC_TIDX
from .reader import AeroReader
from .writer import (
    AeroWriter,
    TensorSpec,
    make_json_chunk,
    make_manifest_chunk,
    make_tidx_chunk,
    make_wtsh_chunk,
)


# ── Terminal UI (stdlib only: colors when TTY, Unicode tables) ───────────────

def _color_enabled() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return os.environ.get("NO_COLOR", "").strip() == ""

_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
}

def _c(name: str, text: str) -> str:
    if not _color_enabled() or name not in _COLORS:
        return text
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"

def _section(title: str) -> str:
    return _c("cyan", f"\n  ◆ {title}")

def _ok(msg: str) -> str:
    return _c("green", "✓ ") + msg

def _fail(msg: str) -> str:
    return _c("red", "✗ ") + msg

def _table(headers: list[str], rows: list[list[str]], padding: int = 1) -> list[str]:
    """Return lines for a UTF-8 box table. Column widths from content."""
    if not headers and not rows:
        return []
    col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
    widths = [0] * col_count
    for i, h in enumerate(headers):
        if i < col_count:
            widths[i] = max(widths[i], len(h))
    for row in rows:
        for i, cell in enumerate(row):
            if i < col_count:
                widths[i] = max(widths[i], len(str(cell)))
    pad = " " * padding
    total = sum(widths) + (padding * 2 * col_count) + (col_count - 1) * 3  # │ and spaces
    lines = []
    top = "╭" + "┬".join("─" * (w + 2 * padding) for w in widths) + "╮"
    lines.append(top)
    h_cells = [pad + (str(headers[i]) if i < len(headers) else "").ljust(widths[i]) + pad for i in range(col_count)]
    lines.append("│" + "│".join(h_cells) + "│")
    lines.append("├" + "┼".join("─" * (w + 2 * padding) for w in widths) + "┤")
    for row in rows:
        cells = []
        for i in range(col_count):
            s = str(row[i]) if i < len(row) else ""
            if len(s) > widths[i]:
                s = s[: widths[i] - 1] + "…"
            cells.append(pad + s.ljust(widths[i]) + pad)
        lines.append("│" + "│".join(cells) + "│")
    lines.append("╰" + "┴".join("─" * (w + 2 * padding) for w in widths) + "╯")
    return lines


# ── inspect ─────────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    with AeroReader(args.file) as reader:
        print(_c("bold", f"\n  AERO  {reader.version[0]}.{reader.version[1]}  ") + _c("dim", args.file))
        print(_section("File"))
        print(f"    UUID    {reader.uuid.hex()}")
        print(f"    Version {reader.version[0]}.{reader.version[1]}")
        print(f"    Flags   0x{reader.file_flags:016x}")

        chunks = reader.list_chunks()
        print(_section(f"Chunks ({len(chunks)})"))
        flag_parts_fn = lambda c: ",".join(
            x for x, bit in [
                ("zstd", ChunkFlags.COMPRESSED_ZSTD),
                ("mmap", ChunkFlags.CHUNK_IS_MMAP_CRITICAL),
                ("index", ChunkFlags.CHUNK_IS_INDEX),
                ("opt", ChunkFlags.CHUNK_IS_OPTIONAL),
            ] if c.flags & bit
        )
        ch_headers = ["#", "FourCC", "Name", "Offset", "Len", "ULen", "Flags"]
        ch_rows = [
            [str(i), c.fourcc.decode().strip(), c.name[:28] + "…" if len(c.name) > 29 else c.name,
             str(c.offset), str(c.length), str(c.uncompressed_length), flag_parts_fn(c)]
            for i, c in enumerate(chunks)
        ]
        for line in _table(ch_headers, ch_rows):
            print("  " + line)

        tidx_ref = reader.get_chunk(FOURCC_TIDX)
        if tidx_ref is not None:
            tensors = reader.list_tensors()
            max_t = args.max_tensors if args.max_tensors else len(tensors)
            show = min(max_t, len(tensors))
            print(_section(f"Tensors ({len(tensors)}, showing {show})"))
            th = ["Name", "Dtype", "Shape", "Shard", "Bytes"]
            tr = []
            for t in tensors[:show]:
                try:
                    dtype_name = DTYPE_NAMES[DType(t["dtype"])]
                except (ValueError, KeyError):
                    dtype_name = f"?{t['dtype']}"
                name = t["name"]
                if len(name) > 42:
                    name = name[:41] + "…"
                tr.append([name, dtype_name, str(t["shape"]), str(t["shard_id"]), str(t["data_len"])])
            for line in _table(th, tr):
                print("  " + line)
            if len(tensors) > show:
                print(_c("dim", f"\n  … and {len(tensors) - show} more (use --max-tensors to show more)"))
        print()


# ── validate ────────────────────────────────────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> None:
    path = args.file
    remote = getattr(args, "remote", False)
    cache_dir = getattr(args, "cache_dir", None)
    
    # Auto-detect aeroset vs single-file
    if path.endswith(".aeroset.json"):
        return _validate_aeroset(path, full=args.full, control=getattr(args, "control", False),
                                 remote=remote, cache_dir=cache_dir)
    try:
        with AeroReader(path) as reader:
            if getattr(args, "control", False):
                errors = reader.validate_control()
            elif args.full:
                errors = reader.validate_all()
            else:
                # Quick mode: only validate metadata chunks.
                errors = []
                for ref in reader.list_chunks():
                    if ref.fourcc in (b"MMSG", b"TIDX", b"MJSN"):
                        try:
                            reader.read_chunk_bytes(ref, verify_hash=True)
                        except Exception as exc:
                            errors.append(f"{ref.fourcc.decode()} ({ref.name}): {exc}")
            if errors:
                for e in errors:
                    print(_fail(e), file=sys.stderr)
                sys.exit(1)
            if getattr(args, "control", False):
                print(_ok("Control-plane integrity (IHSH) OK."))
            else:
                mode = "full" if args.full else "quick"
                print(_ok(f"{len(reader.list_chunks())} chunks ({mode} verify)."))
    except Exception as exc:
        print(_fail(str(exc)), file=sys.stderr)
        sys.exit(1)


def _validate_aeroset(path: str, full: bool = False, control: bool = False,
                     remote: bool = False, cache_dir: str = None) -> None:
    """Validate an AEROSET by checking index and all parts."""
    import json
    import os
    from .aeroset import AeroSetReader

    try:
        if path.startswith(("http://", "https://")):
            if not remote:
                print(_fail("Remote URL requires --remote flag"), file=sys.stderr)
                sys.exit(1)

        with AeroSetReader(path, enable_remote=remote, cache_dir=cache_dir) as reader:
            if control:
                errors = reader._idx_reader.validate_control()
                if errors:
                    for e in errors:
                        print(_fail(e), file=sys.stderr)
                    sys.exit(1)
                print(_ok("Control-plane integrity (IHSH) OK."))
                return
            tensors = reader.list_tensors()
            if full:
                for t in tensors:
                    reader.read_tensor_bytes(t["name"])
            mode = "full" if full else "quick"
            print(_ok(f"AEROSET with {len(tensors)} tensors ({mode} verify)."))
    except Exception as exc:
        print(_fail(str(exc)), file=sys.stderr)
        sys.exit(1)


# ── make-test-vector ────────────────────────────────────────────────────────


def cmd_make_test_vector(args: argparse.Namespace) -> None:
    fixed_uuid = b"\x11" * 16
    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    bias = np.array([0.5, -0.5], dtype="<f4")
    weight_bytes = weight.tobytes()
    bias_bytes = bias.tobytes()
    shard_data = weight_bytes + bias_bytes

    tidx = {"tensors": [
        {"name": "linear.weight", "dtype": int(DType.F32),
         "shape": [2, 3], "shard_id": 0,
         "data_off": 0, "data_len": len(weight_bytes), "flags": 0},
        {"name": "linear.bias", "dtype": int(DType.F32),
         "shape": [2], "shard_id": 0,
         "data_off": len(weight_bytes), "data_len": len(bias_bytes), "flags": 0},
    ]}
    manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": "test-linear", "architecture": "linear"},
        "chunks": [
            {"role": "json", "fourcc": "MJSN", "toc_index": 0},
            {"role": "manifest", "fourcc": "MMSG", "toc_index": 1},
            {"role": "tensors", "fourcc": "TIDX", "toc_index": 2},
            {"role": "weights", "fourcc": "WTSH", "toc_index": 3, "shard_id": 0},
        ],
        "shards": [{"shard_id": 0, "toc_index": 3,
                     "size_bytes": len(shard_data),
                     "tensor_count": 2, "mmap_critical": True}],
    }
    meta = {"description": "AERO v0.1 test vector", "model_name": "test-linear"}
    chunks = [
        make_json_chunk(meta),
        make_manifest_chunk(manifest),
        make_tidx_chunk(tidx),
        make_wtsh_chunk(shard_data),
    ]
    writer = AeroWriter(file_uuid=fixed_uuid)
    writer.write(args.output, chunks)

    sha = hashlib.sha256()
    with open(args.output, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            sha.update(block)
    print(_ok(f"Wrote {args.output}"))
    print(_c("dim", f"  SHA-256  {sha.hexdigest()}"))


# ── inspect-set ─────────────────────────────────────────────────────────────


def cmd_inspect_set(args: argparse.Namespace) -> None:
    from .aeroset import AeroSetReader

    with AeroSetReader(args.file) as reader:
        print(_c("bold", "\n  AEROSET  ") + _c("dim", args.file))
        print(_section("Index"))
        print(f"    Model  {reader.model_name}")
        print(f"    Arch   {reader.architecture}")
        tensors = reader.list_tensors()
        max_t = args.max_tensors if args.max_tensors else len(tensors)
        show = min(max_t, len(tensors))
        print(_section(f"Tensors ({len(tensors)}, showing {show})"))
        th = ["Name", "Dtype", "Shape", "Shard", "Bytes"]
        tr = []
        for t in tensors[:show]:
            try:
                dtype_name = DTYPE_NAMES[DType(t["dtype"])]
            except (ValueError, KeyError):
                dtype_name = f"?{t['dtype']}"
            name = t["name"]
            if len(name) > 42:
                name = name[:41] + "…"
            tr.append([name, dtype_name, str(t["shape"]), str(t["shard_id"]), str(t["data_len"])])
        for line in _table(th, tr):
            print("  " + line)
        if len(tensors) > show:
            print(_c("dim", f"\n  … and {len(tensors) - show} more (use --max-tensors to show more)"))
        print()


# ── convert-safetensors ─────────────────────────────────────────────────────


def cmd_convert_safetensors(args: argparse.Namespace) -> None:
    from .convert.safetensors_to_aero import convert_safetensors

    convert_safetensors(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(_ok(f"Converted {args.input} → {args.output}"))


# ── convert-safetensors-sharded ─────────────────────────────────────────────


def cmd_convert_safetensors_sharded(args: argparse.Namespace) -> None:
    from .convert.safetensors_sharded_to_aeroset import convert_safetensors_sharded

    path = convert_safetensors_sharded(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(_ok(f"Wrote aeroset: {path}"))


# ── convert-gguf ────────────────────────────────────────────────────────────


def cmd_convert_gguf(args: argparse.Namespace) -> None:
    from .convert.gguf_to_aero import convert_gguf

    convert_gguf(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(_ok(f"Converted {args.input} → {args.output}"))


# ── convert (auto) ──────────────────────────────────────────────────────────


def cmd_convert_auto(args: argparse.Namespace) -> None:
    """Auto-detect input format and route to appropriate converter."""
    import os
    
    inp = args.input
    if inp.endswith(".safetensors"):
        return cmd_convert_safetensors(args)
    elif inp.endswith(".gguf"):
        return cmd_convert_gguf(args)
    elif os.path.isdir(inp) or inp.endswith(".index.json"):
        # Assume sharded safetensors
        return cmd_convert_safetensors_sharded(args)
    else:
        print(_fail(f"Cannot auto-detect format for {inp}"), file=sys.stderr)
        print(_c("dim", "  Supported: .safetensors, .gguf, directory with .index.json"), file=sys.stderr)
        sys.exit(1)


# ── diff ─────────────────────────────────────────────────────────────────────


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two .aero files: tensor set and optionally byte/numeric diff."""
    from .format import DType, DTYPE_NAMES

    path_a = args.file_a
    path_b = args.file_b
    verbose = getattr(args, "verbose", False)
    numeric = getattr(args, "numeric", False)

    def tensor_set(path: str) -> tuple[list[dict], set[str]]:
        with AeroReader(path) as r:
            tensors = r.list_tensors()
            names = {t["name"] for t in tensors}
            return tensors, names

    try:
        tensors_a, names_a = tensor_set(path_a)
        tensors_b, names_b = tensor_set(path_b)
    except Exception as exc:
        print(_fail(str(exc)), file=sys.stderr)
        sys.exit(1)

    only_a = names_a - names_b
    only_b = names_b - names_a
    common = names_a & names_b

    # Compare common tensors by bytes (and optionally numeric)
    differing: list[str] = []
    numeric_diffs: list[tuple[str, float]] = []

    if common:
        with AeroReader(path_a) as ra, AeroReader(path_b) as rb:
            tensor_map_a = {t["name"]: t for t in ra.list_tensors()}
            tensor_map_b = {t["name"]: t for t in rb.list_tensors()}
            for name in sorted(common):
                ba = ra.read_tensor_bytes(name)
                bb = rb.read_tensor_bytes(name)
                if ba != bb:
                    differing.append(name)
                    if numeric and len(ba) == len(bb):
                        ta = tensor_map_a[name]
                        tb = tensor_map_b[name]
                        dtype_val = ta.get("dtype", 1)
                        try:
                            dt = DType(dtype_val)
                        except ValueError:
                            dt = DType.F32
                        if dt in (DType.F16, DType.F32, DType.BF16, DType.F64):
                            import numpy as np
                            dtype_np = {
                                DType.F16: np.float16,
                                DType.F32: np.float32,
                                DType.BF16: np.float16,  # treat as f16 for diff
                                DType.F64: np.float64,
                            }.get(dt, np.float32)
                            try:
                                arr_a = np.frombuffer(ba, dtype=dtype_np)
                                arr_b = np.frombuffer(bb, dtype=dtype_np)
                                if arr_a.size == arr_b.size:
                                    max_diff = float(np.max(np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))))
                                    numeric_diffs.append((name, max_diff))
                            except Exception:
                                pass

    # Report
    exit_code = 0
    print(_c("bold", "\n  diff  ") + _c("dim", f"{path_a}  vs  {path_b}"))
    if only_a:
        exit_code = 1
        print(_section(f"Only in first ({len(only_a)})"))
        if verbose:
            for n in sorted(only_a):
                print(_c("red", f"    − {n}"))
        else:
            preview = ", ".join(sorted(only_a)[:10]) + (" …" if len(only_a) > 10 else "")
            print(f"    {preview}")
    if only_b:
        exit_code = 1
        print(_section(f"Only in second ({len(only_b)})"))
        if verbose:
            for n in sorted(only_b):
                print(_c("yellow", f"    + {n}"))
        else:
            preview = ", ".join(sorted(only_b)[:10]) + (" …" if len(only_b) > 10 else "")
            print(f"    {preview}")

    if differing:
        exit_code = 1
        print(_section(f"Differing bytes ({len(differing)})"))
        if verbose or len(differing) <= 20:
            for n in differing:
                extra = ""
                for nm, diff in numeric_diffs:
                    if nm == n:
                        extra = _c("dim", f"  max_abs_diff={diff:.2e}")
                        break
                print(_c("red", f"    ≠ {n}") + extra)
        else:
            preview = ", ".join(differing[:10]) + " …"
            print(f"    {preview}")
        if numeric_diffs and (verbose or len(numeric_diffs) <= 20):
            print(_c("dim", "\n  Numeric (max abs diff):"))
            for n, d in sorted(numeric_diffs, key=lambda x: -x[1])[:15]:
                print(f"    {n}: {d:.2e}")
    else:
        if not only_a and not only_b:
            print(_ok("All common tensors have identical bytes."))
    print()
    sys.exit(exit_code)


# ── fetch-tensor ────────────────────────────────────────────────────────────


def cmd_fetch_tensor(args: argparse.Namespace) -> None:
    """Fetch a tensor from local or remote aeroset."""
    from .aeroset import AeroSetReader

    cache_dir = args.cache_dir if hasattr(args, "cache_dir") else None
    with AeroSetReader(args.aeroset, enable_remote=True, cache_dir=cache_dir) as reader:
        data = reader.read_tensor_bytes(args.tensor)
        with open(args.output, "wb") as f:
            f.write(data)
        print(_ok(f"Fetched {args.tensor} ({len(data):,} bytes) → {args.output}"))


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aerotensor", description="AERO v0.2 CLI"
    )
    parser.add_argument(
        "--version", action="version", version=f"aerotensor {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    # inspect / info
    p = sub.add_parser("inspect", help="Inspect an .aero file", aliases=["info"])
    p.add_argument("file")
    p.add_argument("--max-tensors", type=int, default=None)

    # validate
    p = sub.add_parser("validate", help="Validate an .aero file or .aeroset.json")
    p.add_argument("file")
    p.add_argument("--control", action="store_true",
                   help="Verify only IHSH control-plane integrity (AIP-0001) if present")
    p.add_argument("--full", action="store_true",
                   help="Verify all chunks including WTSH (slow for large files)")
    p.add_argument("--remote", action="store_true",
                   help="Enable remote URL loading for aeroset")
    p.add_argument("--cache-dir", help="Cache directory for remote reads")

    # make-test-vector
    p = sub.add_parser("make-test-vector", help="Generate a golden test vector")
    p.add_argument("output")

    # inspect-set
    p = sub.add_parser("inspect-set", help="Inspect an .aeroset.json")
    p.add_argument("file")
    p.add_argument("--max-tensors", type=int, default=None)

    # convert-safetensors
    p = sub.add_parser("convert-safetensors", help="safetensors -> aero")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--max-shard-bytes", type=int, default=2 << 30)
    p.add_argument("--tensor-hash", action="store_true",
                   help="Compute per-tensor BLAKE3 hashes for slice integrity")

    # convert-safetensors-sharded
    p = sub.add_parser("convert-safetensors-sharded",
                       help="sharded safetensors -> aeroset")
    p.add_argument("input", help="path to index.json or directory")
    p.add_argument("output", help="output directory")
    p.add_argument("--max-shard-bytes", type=int, default=2 << 30)
    p.add_argument("--tensor-hash", action="store_true",
                   help="Compute per-tensor BLAKE3 hashes")

    # convert-gguf
    p = sub.add_parser("convert-gguf", help="GGUF -> aero")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--max-shard-bytes", type=int, default=2 << 30)
    p.add_argument("--tensor-hash", action="store_true",
                   help="Compute per-tensor BLAKE3 hashes")

    # convert auto
    p = sub.add_parser("convert", help="Auto-detect format and convert")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--max-shard-bytes", type=int, default=2 << 30)
    p.add_argument("--tensor-hash", action="store_true",
                   help="Compute per-tensor BLAKE3 hashes")

    # fetch-tensor (remote)
    p = sub.add_parser("fetch-tensor", help="Fetch tensor from local or remote aeroset")
    p.add_argument("aeroset", help="path or URL to model.aeroset.json")
    p.add_argument("tensor", help="tensor name")
    p.add_argument("output", help="output file")
    p.add_argument("--cache-dir", help="cache directory for remote reads")

    # diff
    p = sub.add_parser("diff", help="Compare two .aero files (tensor set and byte/numeric diff)")
    p.add_argument("file_a", help="first .aero file")
    p.add_argument("file_b", help="second .aero file")
    p.add_argument("--verbose", "-v", action="store_true", help="list every tensor name")
    p.add_argument("--numeric", action="store_true", help="report max abs diff for float tensors")

    args = parser.parse_args()

    cmds = {
        "inspect": cmd_inspect,
        "info": cmd_inspect,  # alias
        "validate": cmd_validate,
        "make-test-vector": cmd_make_test_vector,
        "inspect-set": cmd_inspect_set,
        "convert-safetensors": cmd_convert_safetensors,
        "convert-safetensors-sharded": cmd_convert_safetensors_sharded,
        "convert-gguf": cmd_convert_gguf,
        "convert": cmd_convert_auto,
        "fetch-tensor": cmd_fetch_tensor,
        "diff": cmd_diff,
    }
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
