"""Convert a GGUF file to AERO, preserving packed quant blocks.

Implements a minimal GGUF v2/v3 parser – no external dependency required.
"""

from __future__ import annotations

import struct

from ..format import DType
from ..writer import AeroWriter, TensorSpec

# ── GGML type → (block_bytes, block_elements) ──────────────────────────────
_GGML_BLOCK = {
    0: (4, 1),       # F32
    1: (2, 1),       # F16
    2: (18, 32),     # Q4_0
    3: (20, 32),     # Q4_1
    6: (22, 32),     # Q5_0
    7: (24, 32),     # Q5_1
    8: (34, 32),     # Q8_0
    9: (40, 32),     # Q8_1
    10: (84, 256),   # Q2_K
    11: (110, 256),  # Q3_K
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
    15: (292, 256),  # Q8_K
    24: (1, 1),      # I8
    25: (2, 1),      # I16
    26: (4, 1),      # I32
    27: (8, 1),      # I64
    28: (8, 1),      # F64
    30: (2, 1),      # BF16
}

_GGML_TO_AERO = {
    0: DType.F32, 1: DType.F16, 28: DType.F64, 30: DType.BF16,
    24: DType.I8, 25: DType.I16, 26: DType.I32, 27: DType.I64,
}

_GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 28: "F64", 30: "BF16",
}


def _ggml_nbytes(gtype: int, shape: list[int]) -> int:
    if gtype not in _GGML_BLOCK:
        raise ValueError(f"unknown GGML type {gtype}")
    bsz, bel = _GGML_BLOCK[gtype]
    n = 1
    for d in shape:
        n *= d
    if bel == 1:
        return n * bsz
    return (n // bel) * bsz


# ── Minimal GGUF parser ────────────────────────────────────────────────────


_MAX_GGUF_TENSORS = 10_000_000    # reject files claiming > 10M tensors
_MAX_GGUF_KV = 10_000_000        # reject files claiming > 10M KV pairs
_MAX_GGUF_NDIM = 32               # max dimensions
_MAX_GGUF_STRING = 10 * 1024 * 1024  # 10 MB string cap


class _GGUFReader:
    """Minimal GGUF v2/v3 reader (little-endian only)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.fd = open(path, "rb")
        self.metadata: dict = {}
        self.tensors: list[dict] = []
        self.data_offset = 0
        self._parse()

    def close(self) -> None:
        self.fd.close()

    def __enter__(self) -> "_GGUFReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── parsing ──────────────────────────────────────────────────────────

    def _parse(self) -> None:
        magic = self.fd.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        version = self._u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        self._v3 = version >= 3
        tc = self._u64() if self._v3 else self._u32()
        kc = self._u64() if self._v3 else self._u32()

        if tc > _MAX_GGUF_TENSORS:
            raise ValueError(
                f"GGUF tensor_count {tc} exceeds safety cap "
                f"({_MAX_GGUF_TENSORS})")
        if kc > _MAX_GGUF_KV:
            raise ValueError(
                f"GGUF kv_count {kc} exceeds safety cap ({_MAX_GGUF_KV})")

        for _ in range(kc):
            k = self._string()
            v = self._value()
            self.metadata[k] = v
        for _ in range(tc):
            name = self._string()
            ndim = self._u32()
            if ndim > _MAX_GGUF_NDIM:
                raise ValueError(
                    f"GGUF tensor {name!r}: ndim {ndim} exceeds cap "
                    f"({_MAX_GGUF_NDIM})")
            dims = [self._u64() if self._v3 else self._u32()
                    for _ in range(ndim)]
            gtype = self._u32()
            offset = self._u64()
            self.tensors.append({
                "name": name, "shape": dims,
                "type": gtype, "offset": offset,
            })
        alignment = self.metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1:
            alignment = 32
        pos = self.fd.tell()
        self.data_offset = ((pos + alignment - 1) // alignment) * alignment

        # Compute byte sizes from offsets.
        file_size = self.fd.seek(0, 2)
        by_off = sorted(range(len(self.tensors)),
                        key=lambda i: self.tensors[i]["offset"])
        for idx, i in enumerate(by_off):
            if idx + 1 < len(by_off):
                nxt = self.tensors[by_off[idx + 1]]["offset"]
                n_bytes = nxt - self.tensors[i]["offset"]
            else:
                n_bytes = (
                    file_size - self.data_offset - self.tensors[i]["offset"]
                )
            if n_bytes < 0:
                raise ValueError(
                    f"GGUF tensor {self.tensors[i]['name']!r}: "
                    f"computed negative byte size ({n_bytes})")
            self.tensors[i]["n_bytes"] = n_bytes

    # ── primitives ───────────────────────────────────────────────────────

    def _u8(self) -> int:
        return struct.unpack("<B", self.fd.read(1))[0]

    def _i8(self) -> int:
        return struct.unpack("<b", self.fd.read(1))[0]

    def _u16(self) -> int:
        return struct.unpack("<H", self.fd.read(2))[0]

    def _i16(self) -> int:
        return struct.unpack("<h", self.fd.read(2))[0]

    def _u32(self) -> int:
        return struct.unpack("<I", self.fd.read(4))[0]

    def _i32(self) -> int:
        return struct.unpack("<i", self.fd.read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self.fd.read(8))[0]

    def _i64(self) -> int:
        return struct.unpack("<q", self.fd.read(8))[0]

    def _f32(self) -> float:
        return struct.unpack("<f", self.fd.read(4))[0]

    def _f64(self) -> float:
        return struct.unpack("<d", self.fd.read(8))[0]

    def _string(self) -> str:
        length = self._u64() if self._v3 else self._u32()
        if length > _MAX_GGUF_STRING:
            raise ValueError(
                f"GGUF string length {length} exceeds cap "
                f"({_MAX_GGUF_STRING})")
        raw = self.fd.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"GGUF string at offset {self.fd.tell() - length} "
                f"contains invalid UTF-8: {exc}"
            ) from exc

    def _value(self):
        vtype = self._u32()
        return self._typed(vtype)

    def _typed(self, vt: int):
        if vt == 0:   return self._u8()
        if vt == 1:   return self._i8()
        if vt == 2:   return self._u16()
        if vt == 3:   return self._i16()
        if vt == 4:   return self._u32()
        if vt == 5:   return self._i32()
        if vt == 6:   return self._f32()
        if vt == 7:   return bool(self._u8())
        if vt == 8:   return self._string()
        if vt == 9:                                     # ARRAY
            at = self._u32()
            n = self._u64() if self._v3 else self._u32()
            return [self._typed(at) for _ in range(n)]
        if vt == 10:  return self._u64()
        if vt == 11:  return self._i64()
        if vt == 12:  return self._f64()
        raise ValueError(f"unknown GGUF value type {vt}")


# ── Converter ───────────────────────────────────────────────────────────────


def convert_gguf(
    in_path: str,
    out_path: str,
    *,
    max_shard_bytes: int = 2 << 30,
    uuid: bytes | None = None,
    tensor_hash: bool = False,
) -> None:
    """Convert a GGUF file to AERO, preserving packed quant blocks."""
    import os
    file_size = os.path.getsize(in_path)

    with _GGUFReader(in_path) as reader:
        # Sort tensors by name for determinism.
        ordered = sorted(reader.tensors, key=lambda t: t["name"])

        specs: list[TensorSpec] = []
        quant_descriptors: list[dict] = []
        seen_quant: dict[int, int] = {}   # ggml_type → quant_id

        for t in ordered:
            gtype = t["type"]
            n_bytes = t["n_bytes"]
            abs_off = reader.data_offset + t["offset"]

            if abs_off + n_bytes > file_size:
                raise ValueError(
                    f"GGUF tensor {t['name']!r}: data at offset {abs_off} + "
                    f"{n_bytes} bytes exceeds file size {file_size}"
                )

            if gtype in _GGML_TO_AERO:
                aero_dtype = int(_GGML_TO_AERO[gtype])
                extra: dict = {}
            else:
                aero_dtype = int(DType.PACKED)
                if gtype not in seen_quant:
                    qid = len(quant_descriptors)
                    seen_quant[gtype] = qid
                    quant_descriptors.append({
                        "id": qid,
                        "quant_name": _GGML_TYPE_NAME.get(gtype, f"ggml_{gtype}"),
                        "codec": "ggml",
                        "params": {"ggml_type": gtype},
                    })
                extra = {
                    "quant_id": seen_quant[gtype],
                    "quant_params": {"ggml_type": gtype},
                }

            def _read(p=in_path, o=abs_off, n=n_bytes):
                with open(p, "rb") as f:
                    f.seek(o)
                    return f.read(n)

            specs.append(TensorSpec(
                name=t["name"], dtype=aero_dtype, shape=t["shape"],
                data_len=n_bytes, read_data=_read, extra=extra,
            ))

        model_name = reader.metadata.get("general.name", "converted-gguf")
        arch = reader.metadata.get("general.architecture", "unknown")

        extra_meta: dict = {"source_format": "gguf"}
        if quant_descriptors:
            extra_meta["quant_descriptors"] = quant_descriptors

    writer = AeroWriter(file_uuid=uuid)
    writer.write_model(
        out_path, specs, model_name, arch,
        max_shard_bytes=max_shard_bytes, extra_meta=extra_meta,
        tensor_hash=tensor_hash,
    )
