"""Convert a sharded safetensors set to AEROSET.

Expects either:
- A path to ``model.safetensors.index.json``, or
- A directory containing one.

The index JSON has ``{"weight_map": {"tensor": "shard_file", …}}``.
"""

from __future__ import annotations

import json
import os
import struct

from ..format import DType
from ..writer import TensorSpec
from ..aeroset import write_aeroset
from .safetensors_to_aero import _parse_safetensors_header, _ST_DTYPE


def convert_safetensors_sharded(
    index_path_or_dir: str,
    out_dir: str,
    *,
    max_shard_bytes: int = 2 << 30,
    max_part_shards: int = 4,
    uuid: bytes | None = None,
    tensor_hash: bool = False,
) -> str:
    """Convert sharded safetensors to AEROSET; return aeroset.json path."""
    # Locate index file.
    if os.path.isdir(index_path_or_dir):
        candidates = [
            f for f in os.listdir(index_path_or_dir)
            if f.endswith(".index.json")
        ]
        if not candidates:
            raise FileNotFoundError("no .index.json found in directory")
        index_path = os.path.join(index_path_or_dir, sorted(candidates)[0])
    else:
        index_path = index_path_or_dir
    base_dir = os.path.dirname(os.path.abspath(index_path))

    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index.get("weight_map", {})
    meta = index.get("metadata", {})

    # Group tensors by source shard file.
    file_tensors: dict[str, list[str]] = {}
    for tname, fname in weight_map.items():
        file_tensors.setdefault(fname, []).append(tname)

    # Parse each shard file header to get dtype/shape/offsets.
    shard_headers: dict[str, tuple[dict, int]] = {}
    for fname in sorted(file_tensors):
        fpath = os.path.join(base_dir, fname)
        with open(fpath, "rb") as fd:
            hdr, doff = _parse_safetensors_header(fd)
        hdr.pop("__metadata__", None)
        shard_headers[fname] = (hdr, doff)

    # Build TensorSpecs in sorted name order for determinism.
    all_names = sorted(weight_map.keys())
    specs: list[TensorSpec] = []

    for tname in all_names:
        fname = weight_map[tname]
        fpath = os.path.join(base_dir, fname)
        hdr, doff = shard_headers[fname]
        info = hdr[tname]
        dt = _ST_DTYPE.get(info["dtype"])
        if dt is None:
            raise ValueError(f"unsupported dtype: {info['dtype']}")
        start, end = info["data_offsets"]
        data_len = end - start
        abs_offset = doff + start

        def _reader(p=fpath, o=abs_offset, n=data_len):
            with open(p, "rb") as f:
                f.seek(o)
                return f.read(n)

        specs.append(TensorSpec(
            name=tname, dtype=int(dt), shape=info["shape"],
            data_len=data_len, read_data=_reader,
        ))

    model_name = meta.get("model_name", "converted-sharded")
    arch = meta.get("architecture", "unknown")

    return write_aeroset(
        out_dir, specs, model_name, arch,
        max_shard_bytes=max_shard_bytes,
        max_part_shards=max_part_shards,
        uuid=uuid,
        tensor_hash=tensor_hash,
    )
