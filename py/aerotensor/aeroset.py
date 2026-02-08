"""AEROSET v0.2 – multi-file model container (index + parts).

An AEROSET directory looks like::

    model.aeroset.json      ← JSON index
    index.aero              ← tiny AERO with global MMSG + TIDX (no WTSH)
    part-000.aero           ← valid AERO with some WTSH shards
    part-001.aero
    …

v0.2 adds support for remote URLs in part paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional
from urllib.parse import urljoin

from .format import DType, FOURCC_WTSH
from .reader import AeroReader
from .writer import (
    AeroWriter,
    TensorSpec,
    make_manifest_chunk,
    make_tidx_chunk,
    plan_shards,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(1 << 20)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


# ── Writer ──────────────────────────────────────────────────────────────────


def write_aeroset(
    output_dir: str,
    tensor_specs,
    model_name: str,
    architecture: str,
    *,
    max_shard_bytes: int = 2 << 30,
    max_part_shards: int = 4,
    uuid: bytes | None = None,
    extra_meta: dict | None = None,
    tensor_hash: bool = False,
) -> str:
    """Write an AEROSET directory and return path to the index JSON.

    *tensor_specs* is an iterable of :class:`TensorSpec`.
    Shards are grouped into parts, each part containing at most
    *max_part_shards* weight shards.
    """
    os.makedirs(output_dir, exist_ok=True)
    specs = list(tensor_specs)
    shard_plan = plan_shards(specs, max_shard_bytes)

    # Group shards into parts.
    parts_plan: list[list[tuple[int, list[TensorSpec]]]] = []
    current_part: list[tuple[int, list[TensorSpec]]] = []
    for shard in shard_plan:
        current_part.append(shard)
        if len(current_part) >= max_part_shards:
            parts_plan.append(current_part)
            current_part = []
    if current_part:
        parts_plan.append(current_part)

    # ── Build global TIDX ────────────────────────────────────────────────
    all_tidx: list[dict] = []
    for sid, shard_specs in shard_plan:
        off = 0
        for spec in shard_specs:
            entry = {
                "name": spec.name, "dtype": spec.dtype,
                "shape": list(spec.shape), "shard_id": sid,
                "data_off": off, "data_len": spec.data_len, "flags": 0,
            }
            entry.update(spec.extra)
            all_tidx.append(entry)
            off += spec.data_len

    # ── Write each part ──────────────────────────────────────────────────
    part_infos: list[dict] = []
    for pid, part_shards in enumerate(parts_plan):
        part_path = os.path.join(output_dir, f"part-{pid:03d}.aero")
        # Collect tensor specs for this part.
        part_specs: list[TensorSpec] = []
        for _sid, shard_specs in part_shards:
            part_specs.extend(shard_specs)

        writer = AeroWriter(file_uuid=uuid)
        writer.write_model(
            part_path, part_specs, model_name, architecture,
            max_shard_bytes=max_shard_bytes, extra_meta=extra_meta,
            tensor_hash=tensor_hash,
        )

        shard_ids = [sid for sid, _ in part_shards]
        part_infos.append({
            "path": f"part-{pid:03d}.aero",
            "sha256": _sha256_file(part_path),
            "size_bytes": os.path.getsize(part_path),
            "shards": shard_ids,
        })

    # ── Write global index.aero (MMSG + TIDX only) ──────────────────────
    index_path = os.path.join(output_dir, "index.aero")
    g_manifest = {
        "format": {"name": "AERO", "version": [0, 1]},
        "model": {"name": model_name, "architecture": architecture},
        "chunks": [
            {"role": "manifest", "fourcc": "MMSG", "toc_index": 0},
            {"role": "tensors", "fourcc": "TIDX", "toc_index": 1},
        ],
        "shards": [],
    }
    idx_writer = AeroWriter(file_uuid=uuid)
    idx_writer.write(index_path, [
        make_manifest_chunk(g_manifest),
        make_tidx_chunk({"tensors": all_tidx}),
    ])

    # ── Write model.aeroset.json ─────────────────────────────────────────
    aeroset = {
        "format": {"name": "AEROSET", "version": [0, 1]},
        "model": {"name": model_name, "architecture": architecture},
        "parts": part_infos,
        "global_tidx": {
            "path": "index.aero",
            "sha256": _sha256_file(index_path),
            "size_bytes": os.path.getsize(index_path),
        },
    }
    json_path = os.path.join(output_dir, "model.aeroset.json")
    with open(json_path, "w") as f:
        json.dump(aeroset, f, indent=2)

    return json_path


# ── Reader ──────────────────────────────────────────────────────────────────


class AeroSetReader:
    """Read tensors from an AEROSET (index JSON + parts).
    
    Supports both local files and remote HTTP(S) URLs (v0.2).
    """

    def __init__(
        self,
        index_json_path: str,
        *,
        enable_remote: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._enable_remote = enable_remote
        self._cache_dir = cache_dir
        
        # Load index JSON (local or remote)
        if index_json_path.startswith("http://") or index_json_path.startswith("https://"):
            if not enable_remote:
                raise ValueError("Remote URLs require enable_remote=True")
            import urllib.request
            with urllib.request.urlopen(index_json_path, timeout=20) as resp:
                self._index = json.loads(resp.read().decode("utf-8"))
            self._base = index_json_path.rsplit("/", 1)[0] if "/" in index_json_path else ""
        else:
            self._base = os.path.dirname(os.path.abspath(index_json_path))
            with open(index_json_path) as f:
                self._index = json.load(f)

        # Resolve base_url if present (v0.2)
        self._base_url = self._index.get("base_url", "")

        # Load global TIDX from index.aero
        idx_path = self._resolve_path(self._index["global_tidx"]["path"])
        self._idx_reader = self._open_reader(idx_path)
        self._tensors = self._idx_reader.list_tensors()
        self._tensor_map: dict[str, dict] = {t["name"]: t for t in self._tensors}

        # Build shard_id → part_path mapping
        self._shard_to_part: dict[int, str] = {}
        for part in self._index["parts"]:
            part_path = self._resolve_path(part["path"])
            for sid in part["shards"]:
                self._shard_to_part[sid] = part_path

        self._part_cache: dict[str, AeroReader] = {}

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to base or base_url."""
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if self._base_url:
            return urljoin(self._base_url + "/", path)
        if self._base.startswith("http://") or self._base.startswith("https://"):
            return urljoin(self._base + "/", path)
        return os.path.join(self._base, path)

    def _open_reader(self, path: str) -> AeroReader:
        """Open AeroReader for local file or remote URL."""
        if (path.startswith("http://") or path.startswith("https://")) and self._enable_remote:
            from .remote import open_range_source
            source = open_range_source(path, cache_dir=self._cache_dir, enable_cache=True)
            from .reader import AeroReaderFromSource
            return AeroReaderFromSource(source)
        return AeroReader(path)

    @property
    def model_name(self) -> str:
        return self._index["model"]["name"]

    @property
    def architecture(self) -> str:
        return self._index["model"]["architecture"]

    def list_tensors(self) -> list[dict]:
        return list(self._tensors)

    def read_tensor_bytes(self, name: str) -> bytes:
        tensor = self._find(name)
        part_path = self._shard_to_part[tensor["shard_id"]]
        reader = self._open_part(part_path)
        return reader.read_tensor_bytes(name)

    def validate(self) -> list[str]:
        """Validate AEROSET integrity: check SHA-256 of parts and index."""
        errors: list[str] = []

        # Validate index.aero SHA-256.
        tidx_info = self._index.get("global_tidx", {})
        if tidx_info and "sha256" in tidx_info:
            idx_path = self._resolve_path(tidx_info["path"])
            if not idx_path.startswith(("http://", "https://")):
                actual = _sha256_file(idx_path)
                if actual != tidx_info["sha256"]:
                    errors.append(
                        f"index.aero SHA-256 mismatch: "
                        f"expected {tidx_info['sha256']}, got {actual}"
                    )

        # Validate each part file SHA-256.
        for part in self._index.get("parts", []):
            part_path = self._resolve_path(part["path"])
            if not part_path.startswith(("http://", "https://")) and "sha256" in part:
                actual = _sha256_file(part_path)
                if actual != part["sha256"]:
                    errors.append(
                        f"{part['path']} SHA-256 mismatch: "
                        f"expected {part['sha256']}, got {actual}"
                    )

        # Validate all shards referenced in TIDX exist in parts.
        for t in self._tensors:
            sid = t.get("shard_id")
            if sid is not None and sid not in self._shard_to_part:
                errors.append(
                    f"tensor {t.get('name', '?')!r}: shard_id {sid} "
                    f"not found in any part"
                )

        return errors

    def close(self) -> None:
        self._idx_reader.close()
        for r in self._part_cache.values():
            r.close()
        self._part_cache.clear()

    def __enter__(self) -> AeroSetReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── internal ─────────────────────────────────────────────────────────

    def _find(self, name: str) -> dict:
        tensor = self._tensor_map.get(name)
        if tensor is None:
            from .reader import AeroError
            raise AeroError(f"tensor {name!r} not found in aeroset")
        return tensor

    def _open_part(self, path: str) -> AeroReader:
        if path not in self._part_cache:
            self._part_cache[path] = self._open_reader(path)
        return self._part_cache[path]
