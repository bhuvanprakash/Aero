"""Optional BLAKE3 support. When blake3 is not installed, digest/hexdigest return None."""

from __future__ import annotations

from typing import Optional

_blake3_module = None

def _get_blake3():
    global _blake3_module
    if _blake3_module is None:
        try:
            import blake3 as b
            _blake3_module = b
        except ImportError:
            _blake3_module = False
    return _blake3_module if _blake3_module else None


def digest(data: bytes) -> Optional[bytes]:
    """Return BLAKE3-256 digest of data, or None if blake3 is not installed."""
    b3 = _get_blake3()
    if b3 is None:
        return None
    return b3.blake3(data).digest()


def hexdigest(data: bytes) -> Optional[str]:
    """Return BLAKE3-256 hex digest of data, or None if blake3 is not installed."""
    b3 = _get_blake3()
    if b3 is None:
        return None
    return b3.blake3(data).hexdigest()


def get_module():
    """Return the blake3 module for incremental hashing. Raises ImportError if not installed."""
    required_for_writing()
    return _get_blake3()


def digest_required(data: bytes) -> bytes:
    """Return BLAKE3-256 digest; raise ImportError if blake3 is not installed."""
    required_for_writing()
    return _get_blake3().blake3(data).digest()


def required_for_writing() -> None:
    """Raise ImportError with install hint if blake3 is not available (needed for writing)."""
    if _get_blake3() is None:
        raise ImportError(
            "blake3 is required to write .aero files. Install with: pip install aerotensor[verify]"
        ) from None
