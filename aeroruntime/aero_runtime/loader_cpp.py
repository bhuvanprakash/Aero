"""Load state_dict from .aero via the C++ loader (libaero state-dict API).

Uses ctypes to call aero_load_state_dict, aero_state_dict_get_tensor, aero_state_dict_free.
Returns dict[str, torch.Tensor]. Copy tensor data so state_dict is valid after handle is freed.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_int, c_int64, c_uint32, c_uint64, c_void_p
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_libaero() -> ctypes.CDLL | None:
    """Load libaero (libaero.so / libaero.dylib / aero.dll). Returns None if not found."""
    names = []
    if sys.platform == "darwin":
        names = ["libaero.dylib", "aero.dylib"]
    elif sys.platform == "win32":
        names = ["aero.dll", "libaero.dll"]
    else:
        names = ["libaero.so", "libaero.so.0", "aero.so"]

    # Search order: env, same dir as this package, sys.path, system
    search_dirs = []
    if os.environ.get("AERO_LIB_PATH"):
        search_dirs.append(os.environ["AERO_LIB_PATH"])
    this_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(this_dir)
    repo_root = os.path.dirname(os.path.dirname(this_dir))  # runtime/aero_runtime -> repo root
    repo_cpp = os.path.join(repo_root, "cpp", "build")
    if os.path.isdir(repo_cpp):
        search_dirs.append(repo_cpp)
    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
    try:
        return ctypes.CDLL(names[0])
    except OSError:
        pass
    return None


_lib: ctypes.CDLL | None = None

def _get_lib() -> ctypes.CDLL | None:
    global _lib
    if _lib is None:
        _lib = _find_libaero()
    return _lib


def _has_state_dict_api(lib: ctypes.CDLL) -> bool:
    """Return True if lib exports aero_load_state_dict."""
    try:
        lib.aero_load_state_dict
        return True
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# C types
# ---------------------------------------------------------------------------

# aero_state_dict_handle_t = void*
# aero_state_dict_dtype_t: AERO_SD_F16=0, AERO_SD_F32=1

def _register_api(lib: ctypes.CDLL) -> None:
    lib.aero_load_state_dict.argtypes = [c_char_p, c_int]
    lib.aero_load_state_dict.restype = c_void_p

    lib.aero_state_dict_tensor_count.argtypes = [c_void_p]
    lib.aero_state_dict_tensor_count.restype = c_uint32

    lib.aero_state_dict_get_tensor.argtypes = [
        c_void_p, c_uint32,
        c_char_p, ctypes.c_size_t,
        POINTER(c_int64), POINTER(c_uint32),
        POINTER(c_int),  # dtype_out (aero_state_dict_dtype_t)
        POINTER(c_void_p), POINTER(c_uint64),
    ]
    lib.aero_state_dict_get_tensor.restype = c_int

    lib.aero_state_dict_free.argtypes = [c_void_p]
    lib.aero_state_dict_free.restype = None

    lib.aero_error_string.argtypes = []
    lib.aero_error_string.restype = c_char_p


# ---------------------------------------------------------------------------
# Public: load state_dict via C++
# ---------------------------------------------------------------------------

def load_state_dict_cpp(
    aero_path: str,
    map_names: bool = True,
    target_dtype: Any = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a PyTorch state_dict from an .aero file using the C++ loader.

    Requires libaero with state-dict API. Copies all tensor data so the returned
    state_dict is independent of the C++ handle.

    Args:
        aero_path: Path to .aero file.
        map_names: Apply GGUF -> HuggingFace name mapping.
        target_dtype: If set, convert tensors to this dtype (e.g. torch.float16).
        verbose: If True, log progress.

    Returns:
        dict mapping tensor names to torch.Tensor (float32; then convert if target_dtype).

    Raises:
        RuntimeError: If lib not found, API missing, or load fails.
    """
    import torch

    lib = _get_lib()
    if lib is None:
        raise RuntimeError("libaero not found; set AERO_LIB_PATH or build cpp and add to path")
    if not _has_state_dict_api(lib):
        raise RuntimeError("libaero does not export aero_load_state_dict (rebuild with state-dict support)")
    _register_api(lib)

    path_bytes = aero_path.encode("utf-8") + b"\0"
    handle = lib.aero_load_state_dict(path_bytes, 1 if map_names else 0)
    if handle is None or handle == 0:
        err = lib.aero_error_string()
        msg = err.decode("utf-8", errors="replace") if err else "unknown error"
        raise RuntimeError(f"aero_load_state_dict failed: {msg}")

    try:
        n = lib.aero_state_dict_tensor_count(handle)
        state_dict: dict[str, Any] = {}
        name_buf = ctypes.create_string_buffer(1024)
        shape_buf = (c_int64 * 32)()  # max rank 32 (match TIDX limit)
        rank = c_uint32()
        dtype_out = c_int()
        data_ptr = c_void_p()
        size_bytes = c_uint64()

        for i in range(n):
            rc = lib.aero_state_dict_get_tensor(
                handle, i,
                name_buf, ctypes.sizeof(name_buf),
                shape_buf, byref(rank),
                byref(dtype_out), byref(data_ptr), byref(size_bytes),
            )
            if rc != 0:
                err = lib.aero_error_string()
                msg = err.decode("utf-8", errors="replace") if err else "get_tensor failed"
                raise RuntimeError(f"aero_state_dict_get_tensor({i}): {msg}")

            name = name_buf.value.decode("utf-8")
            r = rank.value
            shape = [shape_buf[j] for j in range(r)]
            # C API returns float32 (AERO_SD_F32=1) or float16 (AERO_SD_F16=0)
            is_f16 = dtype_out.value == 0
            num_el = 1
            for s in shape:
                num_el *= int(s)
            # Cap to avoid OOM from corrupt or malicious shape
            _MAX_TENSOR_ELEMENTS = 2 * (1024**3)  # 2B elements
            if num_el <= 0 or num_el > _MAX_TENSOR_ELEMENTS:
                raise RuntimeError(
                    f"aero_state_dict_get_tensor({i}): invalid num_elements {num_el} for tensor {name!r}"
                )
            expected_bytes = num_el * 2 if is_f16 else num_el * 4
            if data_ptr.value is None or data_ptr.value == 0:
                raise RuntimeError(
                    f"aero_state_dict_get_tensor({i}): null data pointer for tensor {name!r}"
                )
            if size_bytes.value < expected_bytes:
                raise RuntimeError(
                    f"aero_state_dict_get_tensor({i}): size_bytes {size_bytes.value} < expected {expected_bytes} for tensor {name!r}"
                )
            if is_f16:
                arr = np.ctypeslib.as_array(
                    (ctypes.c_uint16 * num_el).from_address(data_ptr.value)
                ).copy()
                t = torch.from_numpy(arr.astype(np.float16)).reshape(shape)
            else:
                arr = np.ctypeslib.as_array(
                    (ctypes.c_float * num_el).from_address(data_ptr.value)
                ).copy()
                t = torch.from_numpy(arr).reshape(shape)

            if target_dtype is not None and t.dtype != target_dtype:
                t = t.to(target_dtype)
            state_dict[name] = t

            if verbose and (i + 1) % 50 == 0:
                import logging
                logging.getLogger("aero_runtime").info("  loaded %d / %d tensors ...", i + 1, n)
    finally:
        lib.aero_state_dict_free(handle)

    return state_dict


def is_available() -> bool:
    """Return True if the C++ state-dict loader is available (lib found and API present)."""
    lib = _get_lib()
    return lib is not None and _has_state_dict_api(lib)
