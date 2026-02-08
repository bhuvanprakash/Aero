"""AERO v0.1 writer – produces deterministic .aero files.

Supports both low-level chunk assembly (``write``) and high-level
streaming model write (``write_model``) with automatic sharding.
"""

from __future__ import annotations

import json
import os
import struct
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Callable

import blake3
import msgpack
import zstandard as zstd

from .format import (
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
    HEADER_SIZE,
    TOC_HEADER_SIZE,
    TOC_ENTRY_SIZE,
    HEADER_FMT,
    TOC_HEADER_FMT,
    TOC_ENTRY_FMT,
    ChunkFlags,
    FOURCC_MJSN,
    FOURCC_MMSG,
    FOURCC_TIDX,
    FOURCC_WTSH,
    MAX_ENTRY_COUNT,
    MAX_STRING_TABLE,
    align8,
    align16,
    pad_to,
)


# ── Chunk dataclass ─────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single chunk to be written into an .aero file."""

    fourcc: bytes          # 4-byte ASCII tag, e.g. b"WTSH"
    name: str              # human-readable name stored in string table
    data: bytes            # raw (uncompressed) payload
    flags: int = 0         # ChunkFlags bitmask (additional to auto-set ones)
    compress_zstd: bool = False

    def __post_init__(self) -> None:
        if len(self.fourcc) != 4:
            raise ValueError(f"fourcc must be exactly 4 bytes, got {len(self.fourcc)}")


# ── TensorSpec (for streaming write_model) ──────────────────────────────────


@dataclass
class TensorSpec:
    """Describes a tensor for :meth:`AeroWriter.write_model`.

    *read_data* is called exactly once per write and must return
    exactly *data_len* bytes.  This enables lazy / streaming I/O.
    """

    name: str
    dtype: int
    shape: list
    data_len: int
    read_data: Callable[[], bytes]
    extra: dict = field(default_factory=dict)   # quant_id, quant_params, …


# ── Shard planning ──────────────────────────────────────────────────────────


def plan_shards(
    specs: list[TensorSpec],
    max_shard_bytes: int,
) -> list[tuple[int, list[TensorSpec]]]:
    """Assign tensors to shards deterministically.

    Each tensor is appended to the current shard until adding it would
    exceed *max_shard_bytes*.  An empty shard always accepts the next
    tensor (handles tensors larger than the cap).
    """
    shards: list[tuple[int, list[TensorSpec]]] = []
    current: list[TensorSpec] = []
    current_size = 0
    sid = 0
    for spec in specs:
        if current and current_size + spec.data_len > max_shard_bytes:
            shards.append((sid, current))
            sid += 1
            current = []
            current_size = 0
        current.append(spec)
        current_size += spec.data_len
    if current:
        shards.append((sid, current))
    return shards


# ── AeroWriter ──────────────────────────────────────────────────────────────


class AeroWriter:
    """Assemble chunks into a single .aero file.

    Two write paths:

    * :meth:`write` – low-level, takes pre-built :class:`Chunk` list.
    * :meth:`write_model` – high-level, streams tensor data with
      automatic multi-shard support.  Only one tensor's bytes are held
      in memory at a time.
    """

    def __init__(
        self,
        file_uuid: bytes | None = None,
        file_flags: int = 0,
    ) -> None:
        if file_uuid is None:
            file_uuid = _uuid.uuid4().bytes
        if len(file_uuid) != 16:
            raise ValueError("file_uuid must be exactly 16 bytes")
        self._uuid = file_uuid
        self._file_flags = file_flags

    # ── Low-level write ────────────────────────────────────────────────────

    def write(self, out_path: str, chunks: list[Chunk], *, atomic: bool = True) -> None:
        dest = out_path
        tmp_path = out_path + ".tmp" if atomic else out_path

        # Safety checks.
        if len(chunks) > MAX_ENTRY_COUNT:
            raise ValueError(
                f"chunk count {len(chunks)} exceeds safety cap "
                f"({MAX_ENTRY_COUNT})"
            )
        seen_names: set[tuple[bytes, str]] = set()
        for chunk in chunks:
            key = (chunk.fourcc, chunk.name)
            if key in seen_names:
                raise ValueError(
                    f"duplicate chunk: fourcc={chunk.fourcc!r} "
                    f"name={chunk.name!r}"
                )
            seen_names.add(key)

        # 1. Build string table.
        string_table = bytearray()
        name_entries: list[tuple[int, int]] = []
        for chunk in chunks:
            name_bytes = chunk.name.encode("utf-8")
            name_entries.append((len(string_table), len(name_bytes)))
            string_table.extend(name_bytes)
            string_table.append(0)
        string_table_raw = pad_to(bytes(string_table), 8)

        if len(string_table_raw) > MAX_STRING_TABLE:
            raise ValueError(
                f"string table size {len(string_table_raw)} exceeds safety cap "
                f"({MAX_STRING_TABLE})"
            )

        # 2. Compress & hash.
        compressor = zstd.ZstdCompressor(level=6)
        stored_payloads: list[bytes] = []
        blake3_hashes: list[bytes] = []
        stored_flags: list[int] = []
        ulens: list[int] = []

        for chunk in chunks:
            raw = chunk.data
            blake3_hashes.append(blake3.blake3(raw).digest())
            ulens.append(len(raw))
            flags = chunk.flags
            if chunk.compress_zstd:
                stored = compressor.compress(raw)
                flags |= ChunkFlags.COMPRESSED_ZSTD
            else:
                stored = raw
            stored_payloads.append(stored)
            stored_flags.append(flags)

        # 3. Layout.
        n = len(chunks)
        toc_offset = HEADER_SIZE
        toc_length = TOC_HEADER_SIZE + n * TOC_ENTRY_SIZE
        string_table_offset = align8(toc_offset + toc_length)
        payloads_start = align16(string_table_offset + len(string_table_raw))

        chunk_offsets: list[int] = []
        offset = payloads_start
        for payload in stored_payloads:
            offset = align16(offset)
            chunk_offsets.append(offset)
            offset += len(payload)

        # 4. TOC entries.
        toc_entries = bytearray()
        for i, chunk in enumerate(chunks):
            toc_entries.extend(struct.pack(
                TOC_ENTRY_FMT,
                chunk.fourcc, stored_flags[i], chunk_offsets[i],
                len(stored_payloads[i]), ulens[i],
                name_entries[i][0], name_entries[i][1],
                0, blake3_hashes[i],
            ))

        # 5. Assemble file (atomic: write .tmp → fsync → rename).
        try:
            with open(tmp_path, "wb") as f:
                f.write(struct.pack(
                    HEADER_FMT, MAGIC, VERSION_MAJOR, VERSION_MINOR,
                    HEADER_SIZE, toc_offset, toc_length,
                    string_table_offset, len(string_table_raw),
                    self._file_flags, self._uuid, b"\x00" * 28,
                ))
                f.write(struct.pack(TOC_HEADER_FMT, n, 0, 0))
                f.write(toc_entries)
                _pad_to_offset(f, string_table_offset)
                f.write(string_table_raw)
                _pad_to_offset(f, payloads_start)
                for i, payload in enumerate(stored_payloads):
                    _pad_to_offset(f, chunk_offsets[i])
                    f.write(payload)
                if atomic:
                    f.flush()
                    os.fsync(f.fileno())
            if atomic:
                os.replace(tmp_path, dest)
        except BaseException:
            if atomic and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ── High-level streaming write ───────────────────────────────────────

    def write_model(
        self,
        out_path: str,
        tensor_specs,
        model_name: str,
        architecture: str,
        *,
        max_shard_bytes: int = 2 << 30,
        extra_meta: dict | None = None,
        atomic: bool = True,
        tensor_hash: bool = False,
    ) -> None:
        """Write a complete .aero file with automatic multi-shard WTSH.

        *tensor_specs* is an iterable of :class:`TensorSpec`.  Each
        tensor's bytes are read exactly once via ``spec.read_data()``,
        keeping peak memory proportional to a single tensor.

        If *tensor_hash* is True, computes BLAKE3 hash for each tensor
        and stores as "hash_b3" (hex string) in TIDX for slice integrity.
        """
        specs = list(tensor_specs)
        if not specs:
            raise ValueError("no tensors provided")

        # Validate tensor specs.
        seen_names: set[str] = set()
        for spec in specs:
            if spec.name in seen_names:
                raise ValueError(f"duplicate tensor name: {spec.name!r}")
            seen_names.add(spec.name)
            if spec.data_len < 0:
                raise ValueError(
                    f"tensor {spec.name!r}: negative data_len ({spec.data_len})"
                )
            for i, dim in enumerate(spec.shape):
                if dim < 0:
                    raise ValueError(
                        f"tensor {spec.name!r}: negative dimension "
                        f"at axis {i} ({dim})"
                    )

        shard_plan = plan_shards(specs, max_shard_bytes)

        # ── Build TIDX (placeholder for hash_b3 if enabled) ─────────────
        tidx_entries: list[dict] = []
        tensor_hash_map: dict[str, int] = {}  # name -> tidx index
        for sid, shard_specs in shard_plan:
            off = 0
            for spec in shard_specs:
                entry = {
                    "name": spec.name, "dtype": spec.dtype,
                    "shape": list(spec.shape), "shard_id": sid,
                    "data_off": off, "data_len": spec.data_len, "flags": 0,
                }
                entry.update(spec.extra)
                if tensor_hash:
                    # Placeholder; will be filled during WTSH write
                    entry["hash_b3"] = ""
                    tensor_hash_map[spec.name] = len(tidx_entries)
                tidx_entries.append(entry)
                off += spec.data_len
        tidx = {"tensors": tidx_entries}

        # ── Build manifest ───────────────────────────────────────────────
        chunks_desc = [
            {"role": "json", "fourcc": "MJSN", "toc_index": 0},
            {"role": "manifest", "fourcc": "MMSG", "toc_index": 1},
            {"role": "tensors", "fourcc": "TIDX", "toc_index": 2},
        ]
        shard_infos = []
        for idx, (sid, shard_specs) in enumerate(shard_plan):
            toc_idx = 3 + idx
            chunks_desc.append({
                "role": "weights", "fourcc": "WTSH",
                "toc_index": toc_idx, "shard_id": sid,
            })
            shard_infos.append({
                "shard_id": sid, "toc_index": toc_idx,
                "size_bytes": sum(s.data_len for s in shard_specs),
                "tensor_count": len(shard_specs),
                "mmap_critical": True,
            })
        manifest = {
            "format": {"name": "AERO", "version": [0, 1]},
            "model": {"name": model_name, "architecture": architecture},
            "chunks": chunks_desc,
            "shards": shard_infos,
        }
        meta_dict: dict = {"model_name": model_name, "architecture": architecture}
        if extra_meta:
            meta_dict.update(extra_meta)

        # ── Process metadata chunks (small, in-memory) ───────────────────
        meta_chunks = [
            make_json_chunk(meta_dict),
            make_manifest_chunk(manifest),
            make_tidx_chunk(tidx),
        ]
        compressor = zstd.ZstdCompressor(level=6)
        processed: list[tuple] = []          # (fourcc, name, stored, ulen, flags, hash)
        for c in meta_chunks:
            raw = c.data
            h = blake3.blake3(raw).digest()
            flags = c.flags
            if c.compress_zstd:
                stored = compressor.compress(raw)
                flags |= int(ChunkFlags.COMPRESSED_ZSTD)
            else:
                stored = raw
            processed.append((c.fourcc, c.name, stored, len(raw), flags, h))

        # ── WTSH descriptors (data NOT loaded) ───────────────────────────
        wtsh_list: list[tuple] = []          # (fourcc, name, size, flags, specs)
        for sid, shard_specs in shard_plan:
            wtsh_list.append((
                FOURCC_WTSH,
                f"weights.shard{sid}",
                sum(s.data_len for s in shard_specs),
                int(ChunkFlags.CHUNK_IS_MMAP_CRITICAL),
                shard_specs,
            ))

        # ── String table ─────────────────────────────────────────────────
        all_names = [p[1] for p in processed] + [w[1] for w in wtsh_list]
        st_buf = bytearray()
        name_ents: list[tuple[int, int]] = []
        for nm in all_names:
            nb = nm.encode("utf-8")
            name_ents.append((len(st_buf), len(nb)))
            st_buf.extend(nb)
            st_buf.append(0)
        st_raw = pad_to(bytes(st_buf), 8)

        if len(st_raw) > MAX_STRING_TABLE:
            raise ValueError(
                f"string table size {len(st_raw)} exceeds safety cap "
                f"({MAX_STRING_TABLE})"
            )

        # ── Layout ───────────────────────────────────────────────────────
        n_total = len(processed) + len(wtsh_list)

        if n_total > MAX_ENTRY_COUNT:
            raise ValueError(
                f"total chunk count {n_total} exceeds safety cap "
                f"({MAX_ENTRY_COUNT})"
            )
        toc_off = HEADER_SIZE
        toc_len = TOC_HEADER_SIZE + n_total * TOC_ENTRY_SIZE
        st_off = align8(toc_off + toc_len)
        pay_start = align16(st_off + len(st_raw))

        offsets: list[int] = []
        cur = pay_start
        for _, _, stored, *_ in processed:
            cur = align16(cur)
            offsets.append(cur)
            cur += len(stored)
        for _, _, sz, *_ in wtsh_list:
            cur = align16(cur)
            offsets.append(cur)
            cur += sz

        # ── Write file (streaming WTSH with seek-back for hashes) ────────
        _BLAKE3_OFF = TOC_ENTRY_SIZE - 32     # blake3 field is last 32 B
        dest = out_path
        tmp_path = out_path + ".tmp" if atomic else out_path

        try:
            with open(tmp_path, "wb") as f:
                # Header
                f.write(struct.pack(
                    HEADER_FMT, MAGIC, VERSION_MAJOR, VERSION_MINOR,
                    HEADER_SIZE, toc_off, toc_len, st_off, len(st_raw),
                    self._file_flags, self._uuid, b"\x00" * 28,
                ))
                # TOC header
                f.write(struct.pack(TOC_HEADER_FMT, n_total, 0, 0))

                # TOC entries (record file positions for WTSH patch)
                toc_positions: list[int] = []
                for i in range(n_total):
                    toc_positions.append(f.tell())
                    if i < len(processed):
                        fourcc, _, stored, ulen, flags, h = processed[i]
                        slen = len(stored)
                    else:
                        wi = i - len(processed)
                        fourcc, _, sz, flags, _ = wtsh_list[wi]
                        slen, ulen, h = sz, sz, b"\x00" * 32
                    f.write(struct.pack(
                        TOC_ENTRY_FMT, fourcc, flags, offsets[i],
                        slen, ulen, name_ents[i][0], name_ents[i][1], 0, h,
                    ))

                # String table
                _pad_to_offset(f, st_off)
                f.write(st_raw)
                _pad_to_offset(f, pay_start)

                # Metadata payloads
                for i, (_, _, stored, *_) in enumerate(processed):
                    _pad_to_offset(f, offsets[i])
                    f.write(stored)

                # WTSH payloads – streamed one tensor at a time
                tensor_hashes: dict[str, str] = {}  # name -> hex hash
                for wi, (_, _, _, _, shard_specs) in enumerate(wtsh_list):
                    ci = len(processed) + wi
                    _pad_to_offset(f, offsets[ci])
                    shard_hasher = blake3.blake3()
                    for spec in shard_specs:
                        data = spec.read_data()
                        if len(data) != spec.data_len:
                            raise ValueError(
                                f"{spec.name}: expected {spec.data_len}B, "
                                f"got {len(data)}B"
                            )
                        f.write(data)
                        shard_hasher.update(data)
                        # Compute per-tensor hash if enabled
                        if tensor_hash:
                            tensor_hasher = blake3.blake3(data)
                            tensor_hashes[spec.name] = tensor_hasher.hexdigest()
                        del data
                    # Patch BLAKE3 hash in TOC entry
                    end = f.tell()
                    f.seek(toc_positions[ci] + _BLAKE3_OFF)
                    f.write(shard_hasher.digest())
                    f.seek(end)

                # If tensor_hash enabled, update TIDX with computed hashes
                if tensor_hash and tensor_hashes:
                    for name, hex_hash in tensor_hashes.items():
                        idx = tensor_hash_map.get(name)
                        if idx is not None:
                            tidx_entries[idx]["hash_b3"] = hex_hash
                    # Re-pack TIDX with hashes
                    tidx_with_hashes = msgpack.packb(tidx, use_bin_type=True)
                    tidx_hash = blake3.blake3(tidx_with_hashes).digest()
                    # Find TIDX TOC position and update
                    tidx_toc_idx = 2  # MJSN=0, MMSG=1, TIDX=2
                    tidx_stored = tidx_with_hashes
                    if meta_chunks[2].compress_zstd:
                        tidx_stored = compressor.compress(tidx_with_hashes)
                    # Seek and rewrite TIDX payload
                    f.seek(offsets[tidx_toc_idx])
                    f.write(tidx_stored)
                    # Update TOC entry for TIDX
                    f.seek(toc_positions[tidx_toc_idx] + 16)  # chunk_length offset
                    f.write(struct.pack("<Q", len(tidx_stored)))
                    f.seek(toc_positions[tidx_toc_idx] + 24)  # chunk_ulen offset
                    f.write(struct.pack("<Q", len(tidx_with_hashes)))
                    f.seek(toc_positions[tidx_toc_idx] + _BLAKE3_OFF)
                    f.write(tidx_hash)

                if atomic:
                    f.flush()
                    os.fsync(f.fileno())

            # Atomic rename outside the with block
            if atomic:
                os.replace(tmp_path, dest)
        except BaseException:
            if atomic and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ── Helpers ─────────────────────────────────────────────────────────────────


def _pad_to_offset(f, target: int) -> None:
    pos = f.tell()
    if pos < target:
        f.write(b"\x00" * (target - pos))


# ── Chunk builders ──────────────────────────────────────────────────────────


def make_json_chunk(
    obj: object,
    fourcc: bytes = FOURCC_MJSN,
    name: str = "meta.json",
    compressed: bool = False,
) -> Chunk:
    """Create a JSON metadata chunk (MJSN by default)."""
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return Chunk(fourcc=fourcc, name=name, data=data, compress_zstd=compressed)


def make_manifest_chunk(
    manifest_dict: dict,
    name: str = "manifest",
    compressed: bool = True,
) -> Chunk:
    """Create the MMSG manifest chunk (msgpack, zstd-compressed by default)."""
    data = msgpack.packb(manifest_dict, use_bin_type=True)
    return Chunk(fourcc=FOURCC_MMSG, name=name, data=data, compress_zstd=compressed)


def make_tidx_chunk(
    tidx_dict: dict,
    name: str = "tensors",
    compressed: bool = True,
) -> Chunk:
    """Create the TIDX tensor-index chunk (msgpack, zstd-compressed)."""
    data = msgpack.packb(tidx_dict, use_bin_type=True)
    return Chunk(
        fourcc=FOURCC_TIDX, name=name, data=data,
        flags=int(ChunkFlags.CHUNK_IS_INDEX), compress_zstd=compressed,
    )


def make_wtsh_chunk(
    raw_bytes: bytes,
    shard_name: str = "weights.shard0",
) -> Chunk:
    """Create a WTSH weight-shard chunk (uncompressed, mmap-critical)."""
    return Chunk(
        fourcc=FOURCC_WTSH, name=shard_name, data=raw_bytes,
        flags=int(ChunkFlags.CHUNK_IS_MMAP_CRITICAL), compress_zstd=False,
    )
