"""AERO v0.2 CLI – inspect, validate, convert, and generate test vectors."""

from __future__ import annotations

import argparse
import hashlib
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


# ── inspect ─────────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    with AeroReader(args.file) as reader:
        print(f"UUID    : {reader.uuid.hex()}")
        print(f"Version : {reader.version[0]}.{reader.version[1]}")
        print(f"Flags   : 0x{reader.file_flags:016x}")
        print()

        chunks = reader.list_chunks()
        print(f"Chunks ({len(chunks)}):")
        for i, c in enumerate(chunks):
            flag_parts: list[str] = []
            if c.flags & ChunkFlags.COMPRESSED_ZSTD:
                flag_parts.append("zstd")
            if c.flags & ChunkFlags.CHUNK_IS_MMAP_CRITICAL:
                flag_parts.append("mmap")
            if c.flags & ChunkFlags.CHUNK_IS_INDEX:
                flag_parts.append("index")
            if c.flags & ChunkFlags.CHUNK_IS_OPTIONAL:
                flag_parts.append("opt")
            flag_str = ",".join(flag_parts) if flag_parts else "-"
            print(
                f"  [{i}] {c.fourcc.decode():4s}  "
                f"name={c.name!r:24s}  "
                f"off={c.offset:<10d}  len={c.length:<10d}  "
                f"ulen={c.uncompressed_length:<10d}  "
                f"flags={flag_str}"
            )

        tidx_ref = reader.get_chunk(FOURCC_TIDX)
        if tidx_ref is not None:
            tensors = reader.list_tensors()
            max_t = args.max_tensors if args.max_tensors else len(tensors)
            show = min(max_t, len(tensors))
            print(f"\nTensors ({len(tensors)}, showing {show}):")
            for t in tensors[:show]:
                dtype_val = t["dtype"]
                try:
                    dtype_name = DTYPE_NAMES[DType(dtype_val)]
                except (ValueError, KeyError):
                    dtype_name = f"?{dtype_val}"
                print(
                    f"  {t['name']:40s}  "
                    f"dtype={dtype_name:5s}  shape={t['shape']}  "
                    f"shard={t['shard_id']}  bytes={t['data_len']}"
                )


# ── validate ────────────────────────────────────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> None:
    path = args.file
    remote = getattr(args, "remote", False)
    cache_dir = getattr(args, "cache_dir", None)
    
    # Auto-detect aeroset vs single-file
    if path.endswith(".aeroset.json"):
        return _validate_aeroset(path, full=args.full, remote=remote, cache_dir=cache_dir)
    try:
        with AeroReader(path) as reader:
            if args.full:
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
                    print(f"FAIL: {e}", file=sys.stderr)
                sys.exit(1)
            mode = "full" if args.full else "quick"
            print(f"OK – {len(reader.list_chunks())} chunks ({mode} verify).")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


def _validate_aeroset(path: str, full: bool = False, remote: bool = False, cache_dir: str = None) -> None:
    """Validate an AEROSET by checking index and all parts."""
    import json
    import os
    from .aeroset import AeroSetReader

    try:
        if path.startswith(("http://", "https://")):
            if not remote:
                print("ERROR: Remote URL requires --remote flag", file=sys.stderr)
                sys.exit(1)

        with AeroSetReader(path, enable_remote=remote, cache_dir=cache_dir) as reader:
            tensors = reader.list_tensors()
            if full:
                # Verify all tensors are readable
                for t in tensors:
                    reader.read_tensor_bytes(t["name"])
            mode = "full" if full else "quick"
            print(f"OK – AEROSET with {len(tensors)} tensors ({mode} verify).")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
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
    print(f"Wrote {args.output}")
    print(f"SHA-256: {sha.hexdigest()}")


# ── inspect-set ─────────────────────────────────────────────────────────────


def cmd_inspect_set(args: argparse.Namespace) -> None:
    from .aeroset import AeroSetReader

    with AeroSetReader(args.file) as reader:
        print(f"Model : {reader.model_name}")
        print(f"Arch  : {reader.architecture}")
        tensors = reader.list_tensors()
        max_t = args.max_tensors if args.max_tensors else len(tensors)
        show = min(max_t, len(tensors))
        print(f"\nTensors ({len(tensors)}, showing {show}):")
        for t in tensors[:show]:
            dtype_val = t["dtype"]
            try:
                dtype_name = DTYPE_NAMES[DType(dtype_val)]
            except (ValueError, KeyError):
                dtype_name = f"?{dtype_val}"
            print(
                f"  {t['name']:40s}  "
                f"dtype={dtype_name:5s}  shape={t['shape']}  "
                f"shard={t['shard_id']}  bytes={t['data_len']}"
            )


# ── convert-safetensors ─────────────────────────────────────────────────────


def cmd_convert_safetensors(args: argparse.Namespace) -> None:
    from .convert.safetensors_to_aero import convert_safetensors

    convert_safetensors(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(f"Converted {args.input} -> {args.output}")


# ── convert-safetensors-sharded ─────────────────────────────────────────────


def cmd_convert_safetensors_sharded(args: argparse.Namespace) -> None:
    from .convert.safetensors_sharded_to_aeroset import convert_safetensors_sharded

    path = convert_safetensors_sharded(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(f"Wrote aeroset: {path}")


# ── convert-gguf ────────────────────────────────────────────────────────────


def cmd_convert_gguf(args: argparse.Namespace) -> None:
    from .convert.gguf_to_aero import convert_gguf

    convert_gguf(
        args.input, args.output,
        max_shard_bytes=args.max_shard_bytes,
        tensor_hash=getattr(args, "tensor_hash", False),
    )
    print(f"Converted {args.input} -> {args.output}")


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
        print(f"ERROR: Cannot auto-detect format for {inp}", file=sys.stderr)
        print("Supported: .safetensors, .gguf, directory with .index.json", file=sys.stderr)
        sys.exit(1)


# ── fetch-tensor ────────────────────────────────────────────────────────────


def cmd_fetch_tensor(args: argparse.Namespace) -> None:
    """Fetch a tensor from local or remote aeroset."""
    from .aeroset import AeroSetReader

    cache_dir = args.cache_dir if hasattr(args, "cache_dir") else None
    with AeroSetReader(args.aeroset, enable_remote=True, cache_dir=cache_dir) as reader:
        data = reader.read_tensor_bytes(args.tensor)
        with open(args.output, "wb") as f:
            f.write(data)
        print(f"Fetched {args.tensor} ({len(data)} bytes) -> {args.output}")


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
    }
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
