"""Convert a single safetensors file to AERO.

Safetensors layout::

    [8 B header_len as u64 LE]
    [header_len B  JSON header]
    [rest         raw tensor data]

The JSON header maps tensor names to ``{dtype, shape, data_offsets}``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import BinaryIO

from ..format import DType, DTYPE_SIZES
from ..writer import AeroWriter, TensorSpec

_ST_DTYPE = {
    "F16": DType.F16, "F32": DType.F32, "BF16": DType.BF16,
    "F64": DType.F64, "I8": DType.I8, "U8": DType.U8,
    "I16": DType.I16, "U16": DType.U16, "I32": DType.I32,
    "I64": DType.I64, "BOOL": DType.BOOL,
}

MAX_HEADER = 100 * 1024 * 1024  # 100 MB safety cap


def _parse_safetensors_header(fd: BinaryIO):
    """Return (header_dict, data_section_offset)."""
    raw = fd.read(8)
    if len(raw) < 8:
        raise ValueError("truncated safetensors file")
    header_len = struct.unpack("<Q", raw)[0]
    if header_len > MAX_HEADER:
        raise ValueError(f"safetensors header too large: {header_len}")
    header_bytes = fd.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError("truncated safetensors header")
    header = json.loads(header_bytes)
    return header, 8 + header_len


def _validate_tensor_info(name: str, info: dict, file_size: int,
                          data_off: int) -> None:
    """Validate a single tensor entry from the safetensors header."""
    if "data_offsets" not in info:
        raise ValueError(f"tensor {name!r}: missing data_offsets")
    if "dtype" not in info:
        raise ValueError(f"tensor {name!r}: missing dtype")
    if "shape" not in info:
        raise ValueError(f"tensor {name!r}: missing shape")

    start, end = info["data_offsets"]
    if start < 0 or end < 0:
        raise ValueError(f"tensor {name!r}: negative offset")
    if start > end:
        raise ValueError(
            f"tensor {name!r}: start offset ({start}) > end ({end})")
    if data_off + end > file_size:
        raise ValueError(
            f"tensor {name!r}: data extends past end of file "
            f"(offset {data_off + end} > file size {file_size})")

    # Validate dtype × shape product matches byte span.
    dt = _ST_DTYPE.get(info["dtype"])
    if dt is not None and dt in DTYPE_SIZES:
        element_size = DTYPE_SIZES[dt]
        n_elements = 1
        for d in info["shape"]:
            if d < 0:
                raise ValueError(f"tensor {name!r}: negative dimension {d}")
            n_elements *= d
        expected = n_elements * element_size
        actual = end - start
        if expected != actual:
            raise ValueError(
                f"tensor {name!r}: dtype {info['dtype']} × shape "
                f"{info['shape']} = {expected} bytes, but data span "
                f"is {actual} bytes"
            )


def convert_safetensors(
    in_path: str,
    out_path: str,
    *,
    max_shard_bytes: int = 2 << 30,
    uuid: bytes | None = None,
    tensor_hash: bool = False,
) -> None:
    """Convert a safetensors file to AERO, streaming tensor data."""
    file_size = os.path.getsize(in_path)

    with open(in_path, "rb") as fd:
        header, data_off = _parse_safetensors_header(fd)

    # Collect tensor specs sorted by name for determinism.
    meta = header.pop("__metadata__", {})
    names = sorted(header.keys())

    # Validate all tensors first.
    for name in names:
        _validate_tensor_info(name, header[name], file_size, data_off)

    specs: list[TensorSpec] = []
    for name in names:
        info = header[name]
        dt = _ST_DTYPE.get(info["dtype"])
        if dt is None:
            raise ValueError(f"unsupported safetensors dtype: {info['dtype']}")
        start, end = info["data_offsets"]
        data_len = end - start
        abs_offset = data_off + start

        def _reader(p=in_path, o=abs_offset, n=data_len):
            with open(p, "rb") as f:
                f.seek(o)
                return f.read(n)

        specs.append(TensorSpec(
            name=name, dtype=int(dt), shape=info["shape"],
            data_len=data_len, read_data=_reader,
        ))

    model_name = meta.get("model_name", "converted")
    arch = meta.get("architecture", "unknown")

    writer = AeroWriter(file_uuid=uuid)
    writer.write_model(
        out_path, specs, model_name, arch,
        max_shard_bytes=max_shard_bytes,
        extra_meta={"source_format": "safetensors"},
        tensor_hash=tensor_hash,
    )
