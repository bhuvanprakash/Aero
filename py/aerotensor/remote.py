"""Remote byte-range reading with HTTP Range support and disk caching.

Enables reading AERO/AEROSET files from HTTP(S) URLs with minimal
bandwidth usage by fetching only required byte ranges.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


# ── Configuration ───────────────────────────────────────────────────────────

MAX_RANGE_BYTES = 64 * 1024 * 1024      # 64 MB per HTTP request
MAX_TENSOR_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per tensor read
DEFAULT_CACHE_MAX_MB = 2048              # 2 GB default cache size
DEFAULT_TIMEOUT = 20                     # seconds


# ── RangeSource Protocol ────────────────────────────────────────────────────


class RangeSource(ABC):
    """Abstract interface for byte-range reading."""

    @abstractmethod
    def size(self) -> Optional[int]:
        """Return total size in bytes, or None if unknown."""

    @abstractmethod
    def read_range(self, start: int, length: int) -> bytes:
        """Read *length* bytes starting at *start*."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def __enter__(self) -> RangeSource:
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── FileRangeSource ─────────────────────────────────────────────────────────


class FileRangeSource(RangeSource):
    """Range source for local files using pread or seek+read."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = open(path, "rb")
        self._size = os.fstat(self._fd.fileno()).st_size

    def size(self) -> int:
        return self._size

    def read_range(self, start: int, length: int) -> bytes:
        if start + length > self._size:
            raise ValueError(
                f"range [{start}, {start + length}) exceeds file size {self._size}"
            )
        # Use pread if available (Unix), else seek+read
        if hasattr(os, "pread"):
            return os.pread(self._fd.fileno(), length, start)
        else:
            self._fd.seek(start)
            data = self._fd.read(length)
            if len(data) != length:
                raise IOError("short read from file")
            return data

    def close(self) -> None:
        self._fd.close()


# ── HttpRangeSource ─────────────────────────────────────────────────────────


class HttpRangeSource(RangeSource):
    """Range source for HTTP(S) URLs with Range request support."""

    def __init__(
        self,
        url: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        headers: Optional[dict] = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}
        self._size: Optional[int] = None
        self._supports_range = False
        self._probe()

    def _probe(self) -> None:
        """Send HEAD request to determine size and Range support."""
        req = urllib.request.Request(self.url, method="HEAD", headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._size = int(resp.headers.get("Content-Length", 0)) or None
                accept_ranges = resp.headers.get("Accept-Ranges", "").lower()
                self._supports_range = accept_ranges == "bytes"
        except urllib.error.URLError:
            # HEAD may fail; we'll try GET with Range later
            pass

    def size(self) -> Optional[int]:
        return self._size

    def read_range(self, start: int, length: int) -> bytes:
        if length > MAX_RANGE_BYTES:
            # Split into multiple requests
            chunks = []
            remaining = length
            offset = start
            while remaining > 0:
                chunk_len = min(remaining, MAX_RANGE_BYTES)
                chunks.append(self._fetch_range(offset, chunk_len))
                offset += chunk_len
                remaining -= chunk_len
            return b"".join(chunks)
        return self._fetch_range(start, length)

    def _fetch_range(self, start: int, length: int) -> bytes:
        """Fetch a single range via HTTP Range header."""
        end = start + length - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={start}-{end}"

        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 206:
                    # Partial content
                    data = resp.read()
                    if len(data) != length:
                        raise IOError(
                            f"HTTP 206 returned {len(data)} bytes, expected {length}"
                        )
                    return data
                elif resp.status == 200:
                    # Server ignores Range; read and slice
                    # Safety: only allow if length is small
                    if length > MAX_RANGE_BYTES:
                        raise IOError(
                            "Server does not support Range requests and "
                            f"requested {length} bytes exceeds safety cap"
                        )
                    data = resp.read()
                    if len(data) < start + length:
                        raise IOError("server returned insufficient data")
                    return data[start : start + length]
                else:
                    raise IOError(f"unexpected HTTP status {resp.status}")
        except urllib.error.URLError as e:
            raise IOError(f"HTTP request failed: {e}")

    def close(self) -> None:
        pass


# ── Disk Cache ──────────────────────────────────────────────────────────────


class DiskCache:
    """Simple disk cache for byte ranges keyed by (url_hash, start, length)."""

    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = DEFAULT_CACHE_MAX_MB) -> None:
        if cache_dir is None:
            cache_dir = os.environ.get(
                "AEROTENSOR_CACHE_DIR",
                os.path.join(Path.home(), ".cache", "aerotensor"),
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def _key_path(self, url: str, start: int, length: int) -> Path:
        """Compute cache file path for a range."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        subdir = self.cache_dir / url_hash
        subdir.mkdir(exist_ok=True)
        return subdir / f"{start}-{length}.bin"

    def get(self, url: str, start: int, length: int) -> Optional[bytes]:
        """Retrieve cached range, or None if not cached."""
        path = self._key_path(url, start, length)
        if path.exists():
            try:
                # Update mtime for LRU
                os.utime(path, None)
                return path.read_bytes()
            except (IOError, OSError):
                return None
        return None

    def put(self, url: str, start: int, length: int, data: bytes) -> None:
        """Store range in cache with atomic write."""
        path = self._key_path(url, start, length)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_bytes(data)
            # Atomic rename
            tmp_path.replace(path)
        except (IOError, OSError):
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Cleanup if cache exceeds max size
        self._cleanup_if_needed()

    def _cleanup_if_needed(self) -> None:
        """Remove oldest files if cache size exceeds limit."""
        try:
            total = 0
            files = []
            for subdir in self.cache_dir.iterdir():
                if not subdir.is_dir():
                    continue
                for f in subdir.iterdir():
                    if f.suffix == ".bin":
                        stat = f.stat()
                        files.append((stat.st_mtime, stat.st_size, f))
                        total += stat.st_size

            if total > self.max_size_bytes:
                # Sort by mtime (oldest first)
                files.sort()
                removed = 0
                for mtime, size, f in files:
                    if total - removed <= self.max_size_bytes:
                        break
                    try:
                        f.unlink()
                        removed += size
                    except OSError:
                        pass
        except (IOError, OSError):
            # Cleanup is best-effort
            pass


# ── CachedRangeSource ───────────────────────────────────────────────────────


class CachedRangeSource(RangeSource):
    """Wrapper that adds disk caching to any RangeSource."""

    def __init__(
        self,
        source: RangeSource,
        cache: DiskCache,
        cache_key: str,
    ) -> None:
        self.source = source
        self.cache = cache
        self.cache_key = cache_key

    def size(self) -> Optional[int]:
        return self.source.size()

    def read_range(self, start: int, length: int) -> bytes:
        # Try cache first
        cached = self.cache.get(self.cache_key, start, length)
        if cached is not None:
            return cached

        # Fetch from source
        data = self.source.read_range(start, length)

        # Store in cache
        try:
            self.cache.put(self.cache_key, start, length, data)
        except (IOError, OSError):
            # Cache write failure is non-fatal
            pass

        return data

    def close(self) -> None:
        self.source.close()


# ── Factory ─────────────────────────────────────────────────────────────────


def open_range_source(
    path_or_url: str,
    *,
    cache_dir: Optional[str] = None,
    enable_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> RangeSource:
    """Open a RangeSource for a local path or HTTP(S) URL.

    If *enable_cache* is True and the source is remote, wraps in a
    CachedRangeSource for disk caching.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        source = HttpRangeSource(path_or_url, timeout=timeout)
        if enable_cache:
            cache = DiskCache(cache_dir=cache_dir)
            source = CachedRangeSource(source, cache, path_or_url)
        return source
    else:
        return FileRangeSource(path_or_url)
