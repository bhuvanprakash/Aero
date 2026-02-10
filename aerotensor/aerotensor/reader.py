"""AERO v0.1 reader – parse, validate, and access .aero files."""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any

import msgpack
from . import _blake3
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
    FOURCC_IHSH,
    FOURCC_PHSH,
    FOURCC_TIDX,
    FOURCC_WTSH,
    MAX_ENTRY_COUNT,
    MAX_STRING_TABLE,
    MAX_METADATA_ULEN,
)


# ── Exceptions ──────────────────────────────────────────────────────────────


class AeroError(Exception):
    """Base exception for AERO format / parse errors."""


class IntegrityError(AeroError):
    """BLAKE3 hash mismatch on chunk content."""


# ── ChunkRef ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkRef:
    """Lightweight reference to a TOC entry (no payload loaded)."""

    fourcc: bytes
    name: str
    flags: int
    offset: int
    length: int
    uncompressed_length: int
    blake3_256: bytes


# ── MMapView ────────────────────────────────────────────────────────────────


class MMapView:
    """Owns a memory-mapped region and exposes a :class:`memoryview` slice.

    Uses ``mmap.ALLOCATIONGRANULARITY`` to handle unaligned chunk offsets:
    the actual mapping starts at the nearest aligned base so that the OS
    mapping constraint is satisfied, and we slice to the exact window the
    caller requested.
    """

    def __init__(
        self,
        mm: mmap.mmap,
        base_offset: int,
        delta: int,
        length: int,
    ) -> None:
        self._mm = mm
        self._base = base_offset
        self._delta = delta
        self._length = length

    @property
    def view(self) -> memoryview:
        return memoryview(self._mm)[self._delta : self._delta + self._length]

    def close(self) -> None:
        self._mm.close()

    def __enter__(self) -> MMapView:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── AeroReader ──────────────────────────────────────────────────────────────


class AeroReader:
    """Read and validate an .aero file.

    Usage::

        with AeroReader("model.aero") as r:
            for c in r.list_chunks():
                print(c.fourcc, c.name)
            tensors = r.list_tensors()
            data = r.read_tensor_bytes("embed.weight")
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd = open(path, "rb")  # noqa: SIM115
        self._file_size = os.fstat(self._fd.fileno()).st_size
        self._chunks: list[ChunkRef] = []
        self._header_fields: dict[str, Any] = {}
        self._tensor_list: list[dict] | None = None
        self._tensor_map: dict[str, dict] | None = None
        self._parse()

    # ── Public properties ────────────────────────────────────────────────

    @property
    def uuid(self) -> bytes:
        return self._header_fields["uuid"]

    @property
    def file_flags(self) -> int:
        return self._header_fields["file_flags"]

    @property
    def version(self) -> tuple[int, int]:
        return (
            self._header_fields["version_major"],
            self._header_fields["version_minor"],
        )

    # ── Chunk discovery ──────────────────────────────────────────────────

    def list_chunks(self) -> list[ChunkRef]:
        return list(self._chunks)

    def get_chunk(self, fourcc: bytes) -> ChunkRef | None:
        """Return the first chunk matching *fourcc*, or ``None``."""
        for c in self._chunks:
            if c.fourcc == fourcc:
                return c
        return None

    def get_chunk_named(self, fourcc: bytes, name: str) -> ChunkRef | None:
        for c in self._chunks:
            if c.fourcc == fourcc and c.name == name:
                return c
        return None

    def find_shard_chunk(self, shard_id: int) -> ChunkRef | None:
        """Find the WTSH chunk for the given *shard_id*."""
        return self.get_chunk_named(FOURCC_WTSH, f"weights.shard{shard_id}")

    # ── Chunk I/O ────────────────────────────────────────────────────────

    def read_chunk_bytes(
        self,
        chunk_ref: ChunkRef,
        verify_hash: bool = True,
    ) -> bytes:
        """Read stored bytes, decompress if needed, verify BLAKE3."""
        self._fd.seek(chunk_ref.offset)
        stored = self._fd.read(chunk_ref.length)
        if len(stored) != chunk_ref.length:
            raise AeroError("truncated chunk data")

        if chunk_ref.flags & ChunkFlags.COMPRESSED_ZSTD:
            # Cap decompression size for metadata to prevent zstd bombs.
            ulen = chunk_ref.uncompressed_length
            if chunk_ref.fourcc != FOURCC_WTSH and ulen > MAX_METADATA_ULEN:
                raise AeroError(
                    f"metadata chunk ulen {ulen} exceeds safety cap "
                    f"({MAX_METADATA_ULEN}); refusing to decompress"
                )
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(
                stored, max_output_size=ulen
            )
            if len(data) != chunk_ref.uncompressed_length:
                raise AeroError(
                    f"zstd decompressed size mismatch: "
                    f"got {len(data)}, expected {chunk_ref.uncompressed_length}"
                )
        else:
            data = stored

        if verify_hash:
            h = _blake3.digest(data)
            if h is not None and h != chunk_ref.blake3_256:
                raise IntegrityError(
                    f"BLAKE3 mismatch for chunk "
                    f"{chunk_ref.fourcc!r} ({chunk_ref.name})"
                )
        return data

    def read_msgpack(self, fourcc: bytes = FOURCC_TIDX) -> Any:
        ref = self.get_chunk(fourcc)
        if ref is None:
            raise AeroError(f"chunk {fourcc!r} not found")
        data = self.read_chunk_bytes(ref, verify_hash=True)
        return msgpack.unpackb(data, raw=False)

    # ── Tensor helpers ───────────────────────────────────────────────────

    def list_tensors(self) -> list[dict]:
        if self._tensor_list is None:
            tidx = self.read_msgpack(FOURCC_TIDX)
            self._tensor_list = tidx.get("tensors", [])
            self._tensor_map = {t["name"]: t for t in self._tensor_list}
        return list(self._tensor_list)

    def read_tensor_bytes(self, tensor_name: str) -> bytes:
        """Read raw bytes for one tensor (safe copy).

        Verifies per-tensor BLAKE3 hash (``hash_b3``) if present in TIDX.
        """
        tensor = self._find_tensor(tensor_name)

        shard_name = f"weights.shard{tensor['shard_id']}"
        shard_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
        if shard_ref is None:
            raise AeroError(f"weight shard {shard_name!r} not found")

        data_off: int = tensor["data_off"]
        data_len: int = tensor["data_len"]
        if data_off + data_len > shard_ref.length:
            raise AeroError("tensor data exceeds shard bounds")

        abs_off = shard_ref.offset + data_off
        self._fd.seek(abs_off)
        raw = self._fd.read(data_len)
        if len(raw) != data_len:
            raise AeroError("truncated tensor data")

        # Verify per-tensor hash if present (v0.2)
        if "hash_b3" in tensor and tensor["hash_b3"]:
            expected = tensor["hash_b3"]
            actual = _blake3.hexdigest(raw)
            if actual is not None and actual != expected:
                raise IntegrityError(
                    f"tensor {tensor_name!r} hash_b3 mismatch: "
                    f"expected {expected}, got {actual}"
                )

        return raw

    def read_tensor_view(self, tensor_name: str) -> MMapView:
        """Return a **zero-copy** :class:`MMapView` for a tensor.

        This is the preferred path when you need tensor data without copying:
        the returned view references memory-mapped file bytes. Use
        :meth:`read_tensor_bytes` only when you need a copy (e.g. to modify
        or pass to code that does not support memoryview).

        The caller **must** close the returned handle when done.
        """
        tensor = self._find_tensor(tensor_name)

        shard_name = f"weights.shard{tensor['shard_id']}"
        shard_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
        if shard_ref is None:
            raise AeroError(f"weight shard {shard_name!r} not found")

        data_off: int = tensor["data_off"]
        data_len: int = tensor["data_len"]
        if data_off + data_len > shard_ref.length:
            raise AeroError("tensor data exceeds shard bounds")

        abs_off = shard_ref.offset + data_off

        # Map from a granularity-aligned base so the OS is happy.
        gran = mmap.ALLOCATIONGRANULARITY
        base_off = (abs_off // gran) * gran
        delta = abs_off - base_off
        map_len = delta + data_len

        fd_no = os.open(self._path, os.O_RDONLY)
        try:
            mm = mmap.mmap(
                fd_no, map_len, access=mmap.ACCESS_READ, offset=base_off
            )
        finally:
            os.close(fd_no)

        return MMapView(mm, base_off, delta, data_len)

    # ── Validation ───────────────────────────────────────────────────────

    def validate_control(self) -> list[str]:
        """Verify IHSH (control-plane integrity) if present (AIP-0001). Return error list."""
        errors: list[str] = []
        ihsh_ref = self.get_chunk(FOURCC_IHSH)
        if ihsh_ref is None:
            return errors  # no IHSH chunk, nothing to check
        try:
            ihsh_data = self.read_chunk_bytes(ihsh_ref, verify_hash=True)
            unpacker = msgpack.Unpacker(raw=False)
            unpacker.feed(ihsh_data)
            ihsh = unpacker.unpack()
            covered = ihsh.get("covered_ranges")
            if not covered or not isinstance(covered[0], (list, tuple)) or len(covered[0]) < 2:
                errors.append("IHSH: invalid or missing covered_ranges")
                return errors
            control_region_len = int(covered[0][1])
            control_region = bytearray(self._read(0, control_region_len))
            if len(control_region) != control_region_len:
                errors.append("IHSH: short read of control region")
                return errors
            # Digest is over control region with all TOC entry hashes zeroed (writer hashes before any patch)
            toc_off = struct.unpack(HEADER_FMT, bytes(control_region[:HEADER_SIZE]))[4]
            toc_hdr = bytes(control_region[toc_off : toc_off + TOC_HEADER_SIZE])
            entry_count = struct.unpack(TOC_HEADER_FMT, toc_hdr)[0]
            for i in range(entry_count):
                ent = toc_off + TOC_HEADER_SIZE + i * TOC_ENTRY_SIZE
                control_region[ent + TOC_ENTRY_SIZE - 32 : ent + TOC_ENTRY_SIZE] = b"\x00" * 32
            actual_digest = _blake3.digest(bytes(control_region))
            if actual_digest is None:
                errors.append("IHSH: BLAKE3 not available")
                return errors
            if ihsh.get("algo") != "blake3-256":
                errors.append(f"IHSH: unsupported algo {ihsh.get('algo')!r}")
                return errors
            expected = ihsh.get("digest")
            if isinstance(expected, bytes) and len(expected) == 32:
                if expected != actual_digest:
                    errors.append(
                        "IHSH: control-plane digest mismatch "
                        "(header/TOC/string table tampered or corrupted)"
                    )
            else:
                errors.append("IHSH: invalid digest in payload")
        except Exception as exc:
            errors.append(f"IHSH: {exc}")
        return errors

    def validate_shard_pages(self, shard_name: str) -> list[str]:
        """Verify WTSH shard page-by-page using PHSH if present (AIP-0002)."""
        errors: list[str] = []
        wtsh_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
        if wtsh_ref is None:
            return errors
        phsh_ref = self.get_chunk_named(FOURCC_PHSH, shard_name + ".phsh")
        if phsh_ref is None:
            return errors
        try:
            phsh_data = self.read_chunk_bytes(phsh_ref, verify_hash=True)
            phsh = msgpack.unpackb(phsh_data, raw=False)
            page_size = phsh.get("page_size") or 0
            digests = phsh.get("digests") or []
            if page_size <= 0 or not digests:
                errors.append(f"PHSH {shard_name}.phsh: invalid page_size or digests")
                return errors
            shard_data = self.read_chunk_bytes(wtsh_ref, verify_hash=False)
            if len(shard_data) != wtsh_ref.length:
                errors.append(f"PHSH {shard_name}: short read")
                return errors
            n_pages = (len(shard_data) + page_size - 1) // page_size
            if n_pages != len(digests):
                errors.append(
                    f"PHSH {shard_name}: page count mismatch "
                    f"(expected {n_pages}, got {len(digests)})"
                )
                return errors
            digest_fn = _blake3.digest
            if digest_fn is None:
                errors.append("PHSH: BLAKE3 not available")
                return errors
            for i in range(n_pages):
                start = i * page_size
                end = min(start + page_size, len(shard_data))
                page = shard_data[start:end]
                actual = digest_fn(page)
                if actual is None:
                    errors.append(f"PHSH {shard_name}: BLAKE3 failed for page {i}")
                    break
                expected = digests[i] if isinstance(digests[i], bytes) else bytes(digests[i])
                if len(expected) != 32 or expected != actual:
                    errors.append(
                        f"PHSH {shard_name}: page {i} digest mismatch"
                    )
        except Exception as exc:
            errors.append(f"PHSH {shard_name}: {exc}")
        return errors

    def validate_all(self) -> list[str]:
        """Verify every chunk and tensor metadata; return error descriptions."""
        errors: list[str] = []
        errors.extend(self.validate_control())
        for ref in self._chunks:
            try:
                self.read_chunk_bytes(ref, verify_hash=True)
            except Exception as exc:
                errors.append(f"{ref.fourcc.decode()} ({ref.name}): {exc}")

        # Validate tensor entries reference valid shards and stay in bounds.
        try:
            tensors = self.list_tensors()
            seen_names: set[str] = set()
            for t in tensors:
                name = t.get("name", "<missing>")
                if name in seen_names:
                    errors.append(f"TIDX: duplicate tensor name {name!r}")
                seen_names.add(name)
                shard_name = f"weights.shard{t.get('shard_id', -1)}"
                shard_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
                if shard_ref is None:
                    errors.append(
                        f"TIDX: tensor {name!r} references missing "
                        f"shard {shard_name!r}"
                    )
                else:
                    data_off = t.get("data_off", 0)
                    data_len = t.get("data_len", 0)
                    if data_off + data_len > shard_ref.length:
                        errors.append(
                            f"TIDX: tensor {name!r} data_off+data_len "
                            f"exceeds shard {shard_name!r} "
                            f"({data_off}+{data_len} > {shard_ref.length})"
                        )
        except Exception as exc:
            errors.append(f"TIDX validation: {exc}")

        # Optional PHSH (page hashes) verification per WTSH (AIP-0002)
        for ref in self._chunks:
            if ref.fourcc != FOURCC_WTSH:
                continue
            if self.get_chunk_named(FOURCC_PHSH, ref.name + ".phsh") is not None:
                errors.extend(self.validate_shard_pages(ref.name))

        return errors

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._fd.close()

    def __enter__(self) -> AeroReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Internal ─────────────────────────────────────────────────────────

    def _find_tensor(self, name: str) -> dict:
        if self._tensor_map is None:
            self.list_tensors()  # populate cache
        assert self._tensor_map is not None
        tensor = self._tensor_map.get(name)
        if tensor is None:
            raise AeroError(f"tensor {name!r} not found")
        return tensor

    def _parse(self) -> None:
        if self._file_size < HEADER_SIZE:
            raise AeroError("file too small for header")

        raw_header = self._read(0, HEADER_SIZE)
        (
            magic, ver_major, ver_minor, hdr_size,
            toc_off, toc_len, st_off, st_len,
            file_flags, file_uuid, _reserved,
        ) = struct.unpack(HEADER_FMT, raw_header)

        if magic != MAGIC:
            raise AeroError(f"bad magic: {magic!r}")
        if ver_major != VERSION_MAJOR or ver_minor != VERSION_MINOR:
            raise AeroError(f"unsupported version {ver_major}.{ver_minor}")
        if hdr_size != HEADER_SIZE:
            raise AeroError(f"header_size mismatch: {hdr_size}")
        if file_flags != 0:
            raise AeroError(
                f"unsupported file_flags: 0x{file_flags:x} "
                f"(only 0 is valid in v0.1)"
            )

        self._header_fields = {
            "version_major": ver_major,
            "version_minor": ver_minor,
            "toc_offset": toc_off,
            "toc_length": toc_len,
            "string_table_offset": st_off,
            "string_table_length": st_len,
            "file_flags": file_flags,
            "uuid": file_uuid,
        }

        # Validate TOC bounds.
        if toc_off + toc_len > self._file_size:
            raise AeroError("TOC extends past end of file")
        if toc_len < TOC_HEADER_SIZE:
            raise AeroError("TOC too small for header")

        # Parse TOC header.
        toc_hdr_raw = self._read(toc_off, TOC_HEADER_SIZE)
        entry_count, _r0, _r1 = struct.unpack(TOC_HEADER_FMT, toc_hdr_raw)

        if entry_count > MAX_ENTRY_COUNT:
            raise AeroError(
                f"entry_count {entry_count} exceeds safety cap "
                f"({MAX_ENTRY_COUNT}); refusing to parse"
            )

        expected_toc_len = TOC_HEADER_SIZE + entry_count * TOC_ENTRY_SIZE
        if toc_len < expected_toc_len:
            raise AeroError("TOC length doesn't match entry_count")

        # Validate & read string table.
        if st_off + st_len > self._file_size:
            raise AeroError("string table extends past end of file")
        if st_len > MAX_STRING_TABLE:
            raise AeroError(
                f"string_table_length {st_len} exceeds safety cap "
                f"({MAX_STRING_TABLE}); refusing to parse"
            )
        st_data = self._read(st_off, st_len)

        # Parse TOC entries.
        for i in range(entry_count):
            off = toc_off + TOC_HEADER_SIZE + i * TOC_ENTRY_SIZE
            raw_entry = self._read(off, TOC_ENTRY_SIZE)
            (
                fourcc, flags, chunk_off, chunk_len, chunk_ulen,
                name_off, name_len, _r0, b3hash,
            ) = struct.unpack(TOC_ENTRY_FMT, raw_entry)

            if chunk_off + chunk_len > self._file_size:
                raise AeroError(
                    f"chunk {i} ({fourcc!r}) extends past end of file"
                )
            if name_off + name_len > st_len:
                raise AeroError(
                    f"chunk {i} ({fourcc!r}) name exceeds string table"
                )

            name = st_data[name_off : name_off + name_len].decode("utf-8")
            self._chunks.append(
                ChunkRef(
                    fourcc=fourcc,
                    name=name,
                    flags=flags,
                    offset=chunk_off,
                    length=chunk_len,
                    uncompressed_length=chunk_ulen,
                    blake3_256=b3hash,
                )
            )

    def _read(self, offset: int, length: int) -> bytes:
        self._fd.seek(offset)
        data = self._fd.read(length)
        if len(data) != length:
            raise AeroError(f"short read at offset {offset}")
        return data


# ── AeroReaderFromSource (for remote/range reading) ────────────────────────


class AeroReaderFromSource:
    """Read an .aero file from a RangeSource (local or remote).
    
    Provides the same interface as AeroReader but uses byte-range reads
    instead of file I/O, enabling efficient remote HTTP Range fetching.
    """

    def __init__(self, source) -> None:
        """Initialize from a RangeSource (from aerotensor.remote)."""
        self._source = source
        self._file_size = source.size()
        if self._file_size is None:
            raise AeroError("source must provide known size")
        self._chunks: list[ChunkRef] = []
        self._header_fields: dict[str, Any] = {}
        self._tensor_list: list[dict] | None = None
        self._tensor_map: dict[str, dict] | None = None
        self._parse()

    @property
    def uuid(self) -> bytes:
        return self._header_fields["uuid"]

    @property
    def file_flags(self) -> int:
        return self._header_fields["file_flags"]

    @property
    def version(self) -> tuple[int, int]:
        return (
            self._header_fields["version_major"],
            self._header_fields["version_minor"],
        )

    def list_chunks(self) -> list[ChunkRef]:
        return list(self._chunks)

    def get_chunk(self, fourcc: bytes) -> ChunkRef | None:
        for c in self._chunks:
            if c.fourcc == fourcc:
                return c
        return None

    def get_chunk_named(self, fourcc: bytes, name: str) -> ChunkRef | None:
        for c in self._chunks:
            if c.fourcc == fourcc and c.name == name:
                return c
        return None

    def find_shard_chunk(self, shard_id: int) -> ChunkRef | None:
        return self.get_chunk_named(FOURCC_WTSH, f"weights.shard{shard_id}")

    def read_chunk_bytes(
        self,
        chunk_ref: ChunkRef,
        verify_hash: bool = True,
    ) -> bytes:
        """Read stored bytes, decompress if needed, verify BLAKE3."""
        stored = self._source.read_range(chunk_ref.offset, chunk_ref.length)
        if len(stored) != chunk_ref.length:
            raise AeroError("truncated chunk data")

        if chunk_ref.flags & ChunkFlags.COMPRESSED_ZSTD:
            # Cap decompression size for metadata
            ulen = chunk_ref.uncompressed_length
            if chunk_ref.fourcc != FOURCC_WTSH and ulen > MAX_METADATA_ULEN:
                raise AeroError(
                    f"metadata chunk ulen {ulen} exceeds safety cap "
                    f"({MAX_METADATA_ULEN}); refusing to decompress"
                )
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(stored, max_output_size=ulen)
            if len(data) != ulen:
                raise AeroError(
                    f"zstd decompressed size mismatch: "
                    f"got {len(data)}, expected {ulen}"
                )
        else:
            data = stored

        if verify_hash:
            h = _blake3.digest(data)
            if h is not None and h != chunk_ref.blake3_256:
                raise IntegrityError(
                    f"BLAKE3 mismatch for chunk "
                    f"{chunk_ref.fourcc!r} ({chunk_ref.name})"
                )
        return data

    def read_msgpack(self, fourcc: bytes = FOURCC_TIDX) -> Any:
        ref = self.get_chunk(fourcc)
        if ref is None:
            raise AeroError(f"chunk {fourcc!r} not found")
        data = self.read_chunk_bytes(ref, verify_hash=True)
        return msgpack.unpackb(data, raw=False)

    def list_tensors(self) -> list[dict]:
        if self._tensor_list is None:
            tidx = self.read_msgpack(FOURCC_TIDX)
            self._tensor_list = tidx.get("tensors", [])
            self._tensor_map = {t["name"]: t for t in self._tensor_list}
        return list(self._tensor_list)

    def read_tensor_bytes(self, tensor_name: str) -> bytes:
        """Read raw bytes for one tensor, verifying hash_b3 if present."""
        tensor = self._find_tensor(tensor_name)

        shard_name = f"weights.shard{tensor['shard_id']}"
        shard_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
        if shard_ref is None:
            raise AeroError(f"weight shard {shard_name!r} not found")

        data_off: int = tensor["data_off"]
        data_len: int = tensor["data_len"]
        if data_off + data_len > shard_ref.length:
            raise AeroError("tensor data exceeds shard bounds")

        abs_off = shard_ref.offset + data_off
        raw = self._source.read_range(abs_off, data_len)
        if len(raw) != data_len:
            raise AeroError("truncated tensor data")

        # Verify per-tensor hash if present (v0.2)
        if "hash_b3" in tensor and tensor["hash_b3"]:
            expected = tensor["hash_b3"]
            actual = _blake3.hexdigest(raw)
            if actual is not None and actual != expected:
                raise IntegrityError(
                    f"tensor {tensor_name!r} hash_b3 mismatch: "
                    f"expected {expected}, got {actual}"
                )

        return raw

    def validate_control(self) -> list[str]:
        """Verify IHSH (control-plane integrity) if present (AIP-0001)."""
        errors: list[str] = []
        ihsh_ref = self.get_chunk(FOURCC_IHSH)
        if ihsh_ref is None:
            return errors
        try:
            ihsh_data = self.read_chunk_bytes(ihsh_ref, verify_hash=True)
            unpacker = msgpack.Unpacker(raw=False)
            unpacker.feed(ihsh_data)
            ihsh = unpacker.unpack()
            covered = ihsh.get("covered_ranges")
            if not covered or not isinstance(covered[0], (list, tuple)) or len(covered[0]) < 2:
                errors.append("IHSH: invalid or missing covered_ranges")
                return errors
            control_region_len = int(covered[0][1])
            control_region = bytearray(self._source.read_range(0, control_region_len))
            if len(control_region) != control_region_len:
                errors.append("IHSH: short read of control region")
                return errors
            toc_off = self._header_fields["toc_offset"]
            entry_count = len(self._chunks)
            for i in range(entry_count):
                ent = toc_off + TOC_HEADER_SIZE + i * TOC_ENTRY_SIZE
                control_region[ent + TOC_ENTRY_SIZE - 32 : ent + TOC_ENTRY_SIZE] = b"\x00" * 32
            actual_digest = _blake3.digest(bytes(control_region))
            if actual_digest is None:
                errors.append("IHSH: BLAKE3 not available")
                return errors
            if ihsh.get("algo") != "blake3-256":
                errors.append(f"IHSH: unsupported algo {ihsh.get('algo')!r}")
                return errors
            expected = ihsh.get("digest")
            if isinstance(expected, bytes) and len(expected) == 32:
                if expected != actual_digest:
                    errors.append(
                        "IHSH: control-plane digest mismatch "
                        "(header/TOC/string table tampered or corrupted)"
                    )
            else:
                errors.append("IHSH: invalid digest in payload")
        except Exception as exc:
            errors.append(f"IHSH: {exc}")
        return errors

    def validate_shard_pages(self, shard_name: str) -> list[str]:
        """Verify WTSH shard page-by-page using PHSH if present (AIP-0002)."""
        errors: list[str] = []
        wtsh_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
        if wtsh_ref is None:
            return errors
        phsh_ref = self.get_chunk_named(FOURCC_PHSH, shard_name + ".phsh")
        if phsh_ref is None:
            return errors
        try:
            phsh_data = self.read_chunk_bytes(phsh_ref, verify_hash=True)
            phsh = msgpack.unpackb(phsh_data, raw=False)
            page_size = phsh.get("page_size") or 0
            digests = phsh.get("digests") or []
            if page_size <= 0 or not digests:
                errors.append(f"PHSH {shard_name}.phsh: invalid page_size or digests")
                return errors
            n_pages = len(digests)
            expected_pages = (wtsh_ref.length + page_size - 1) // page_size
            if expected_pages != n_pages:
                errors.append(
                    f"PHSH {shard_name}: page count mismatch "
                    f"(expected {expected_pages}, got {n_pages})"
                )
                return errors
            digest_fn = _blake3.digest
            if digest_fn is None:
                errors.append("PHSH: BLAKE3 not available")
                return errors
            for i in range(n_pages):
                start = i * page_size
                end = min(start + page_size, wtsh_ref.length)
                page = self._source.read_range(wtsh_ref.offset + start, end - start)
                if len(page) != end - start:
                    errors.append(f"PHSH {shard_name}: short read page {i}")
                    break
                actual = digest_fn(page)
                if actual is None:
                    errors.append(f"PHSH {shard_name}: BLAKE3 failed for page {i}")
                    break
                expected = digests[i] if isinstance(digests[i], bytes) else bytes(digests[i])
                if len(expected) != 32 or expected != actual:
                    errors.append(f"PHSH {shard_name}: page {i} digest mismatch")
        except Exception as exc:
            errors.append(f"PHSH {shard_name}: {exc}")
        return errors

    def validate_all(self) -> list[str]:
        """Verify every chunk and tensor metadata; return error descriptions."""
        errors: list[str] = []
        # IHSH (control-plane) if present
        errors.extend(self.validate_control())
        for ref in self._chunks:
            try:
                self.read_chunk_bytes(ref, verify_hash=True)
            except Exception as exc:
                errors.append(f"{ref.fourcc.decode()} ({ref.name}): {exc}")

        # Validate tensor entries reference valid shards and stay in bounds.
        try:
            tensors = self.list_tensors()
            seen_names: set[str] = set()
            for t in tensors:
                name = t.get("name", "<missing>")
                if name in seen_names:
                    errors.append(f"TIDX: duplicate tensor name {name!r}")
                seen_names.add(name)
                shard_name = f"weights.shard{t.get('shard_id', -1)}"
                shard_ref = self.get_chunk_named(FOURCC_WTSH, shard_name)
                if shard_ref is None:
                    # Not an error for index.aero in AEROSET (no WTSH present)
                    pass
                else:
                    data_off = t.get("data_off", 0)
                    data_len = t.get("data_len", 0)
                    if data_off + data_len > shard_ref.length:
                        errors.append(
                            f"TIDX: tensor {name!r} data_off+data_len "
                            f"exceeds shard {shard_name!r} "
                            f"({data_off}+{data_len} > {shard_ref.length})"
                        )
        except Exception as exc:
            errors.append(f"TIDX validation: {exc}")

        # Optional PHSH (page hashes) verification per WTSH (AIP-0002)
        for ref in self._chunks:
            if ref.fourcc != FOURCC_WTSH:
                continue
            if self.get_chunk_named(FOURCC_PHSH, ref.name + ".phsh") is not None:
                errors.extend(self.validate_shard_pages(ref.name))

        return errors

    def close(self) -> None:
        self._source.close()

    def __enter__(self) -> "AeroReaderFromSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _find_tensor(self, name: str) -> dict:
        if self._tensor_map is None:
            self.list_tensors()  # populate cache
        assert self._tensor_map is not None
        tensor = self._tensor_map.get(name)
        if tensor is None:
            raise AeroError(f"tensor {name!r} not found")
        return tensor

    def _parse(self) -> None:
        if self._file_size < HEADER_SIZE:
            raise AeroError("file too small for header")

        raw_header = self._source.read_range(0, HEADER_SIZE)
        (
            magic, ver_major, ver_minor, hdr_size,
            toc_off, toc_len, st_off, st_len,
            file_flags, file_uuid, _reserved,
        ) = struct.unpack(HEADER_FMT, raw_header)

        if magic != MAGIC:
            raise AeroError(f"bad magic: {magic!r}")
        if ver_major != VERSION_MAJOR or ver_minor != VERSION_MINOR:
            raise AeroError(f"unsupported version {ver_major}.{ver_minor}")
        if hdr_size != HEADER_SIZE:
            raise AeroError(f"header_size mismatch: {hdr_size}")
        if file_flags != 0:
            raise AeroError(
                f"unsupported file_flags: 0x{file_flags:x} "
                f"(only 0 is valid in v0.1)"
            )

        self._header_fields = {
            "version_major": ver_major,
            "version_minor": ver_minor,
            "toc_offset": toc_off,
            "toc_length": toc_len,
            "string_table_offset": st_off,
            "string_table_length": st_len,
            "file_flags": file_flags,
            "uuid": file_uuid,
        }

        # Validate TOC bounds
        if toc_off + toc_len > self._file_size:
            raise AeroError("TOC extends past end of file")
        if toc_len < TOC_HEADER_SIZE:
            raise AeroError("TOC too small for header")

        # Parse TOC header
        toc_hdr_raw = self._source.read_range(toc_off, TOC_HEADER_SIZE)
        entry_count, _r0, _r1 = struct.unpack(TOC_HEADER_FMT, toc_hdr_raw)

        if entry_count > MAX_ENTRY_COUNT:
            raise AeroError(
                f"entry_count {entry_count} exceeds safety cap "
                f"({MAX_ENTRY_COUNT}); refusing to parse"
            )

        expected_toc_len = TOC_HEADER_SIZE + entry_count * TOC_ENTRY_SIZE
        if toc_len < expected_toc_len:
            raise AeroError("TOC length doesn't match entry_count")

        # Validate & read string table
        if st_off + st_len > self._file_size:
            raise AeroError("string table extends past end of file")
        if st_len > MAX_STRING_TABLE:
            raise AeroError(
                f"string_table_length {st_len} exceeds safety cap "
                f"({MAX_STRING_TABLE}); refusing to parse"
            )
        st_data = self._source.read_range(st_off, st_len)

        # Parse TOC entries
        entries_data = self._source.read_range(
            toc_off + TOC_HEADER_SIZE,
            entry_count * TOC_ENTRY_SIZE
        )
        for i in range(entry_count):
            off = i * TOC_ENTRY_SIZE
            raw_entry = entries_data[off : off + TOC_ENTRY_SIZE]
            (
                fourcc, flags, chunk_off, chunk_len, chunk_ulen,
                name_off, name_len, _r0, b3hash,
            ) = struct.unpack(TOC_ENTRY_FMT, raw_entry)

            if chunk_off + chunk_len > self._file_size:
                raise AeroError(
                    f"chunk {i} ({fourcc!r}) extends past end of file"
                )
            if name_off + name_len > st_len:
                raise AeroError(
                    f"chunk {i} ({fourcc!r}) name exceeds string table"
                )

            name = st_data[name_off : name_off + name_len].decode("utf-8")
            self._chunks.append(
                ChunkRef(
                    fourcc=fourcc,
                    name=name,
                    flags=flags,
                    offset=chunk_off,
                    length=chunk_len,
                    uncompressed_length=chunk_ulen,
                    blake3_256=b3hash,
                )
            )
