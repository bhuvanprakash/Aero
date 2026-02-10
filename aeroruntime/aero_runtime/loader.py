"""Load model weights from .aero files with automatic dequantization and name mapping.

Handles:
  - Reading AERO files via aerotensor.AeroReader
  - Dequantizing PACKED (GGUF quantized) tensors to float via the gguf package
  - Mapping GGUF-style tensor names (blk.N.*) to HuggingFace format (model.layers.N.*)
  - Standard dtype tensors (F16, F32, BF16, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from aerotensor import AeroReader
from aerotensor.format import FOURCC_MJSN

logger = logging.getLogger("aero_runtime")

# AERO PACKED dtype code (quantized data)
DTYPE_PACKED = 0x8000

# ---------------------------------------------------------------------------
# GGUF → HuggingFace tensor-name mapping
# ---------------------------------------------------------------------------

# Per-layer sub-key mapping (works for llama, qwen2, qwen3, mistral, phi3, …)
_GGUF_LAYER_MAP: dict[str, str] = {
    "attn_k":       "self_attn.k_proj",
    "attn_q":       "self_attn.q_proj",
    "attn_v":       "self_attn.v_proj",
    "attn_output":  "self_attn.o_proj",
    "attn_norm":    "input_layernorm",
    "attn_k_norm":  "self_attn.k_norm",
    "attn_q_norm":  "self_attn.q_norm",
    "ffn_gate":     "mlp.gate_proj",
    "ffn_up":       "mlp.up_proj",
    "ffn_down":     "mlp.down_proj",
    "ffn_norm":     "post_attention_layernorm",
}

# Global (non-per-layer) name mapping
_GGUF_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight":   "model.embed_tokens.weight",
    "output_norm.weight":  "model.norm.weight",
    "output.weight":       "lm_head.weight",
}


def _map_gguf_name(name: str) -> str:
    """Map a single GGUF tensor name to HuggingFace format."""
    if name in _GGUF_GLOBAL_MAP:
        return _GGUF_GLOBAL_MAP[name]

    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) >= 4:
            layer_id = parts[1]
            rest = ".".join(parts[2:])
            for gguf_key, hf_key in _GGUF_LAYER_MAP.items():
                if rest == f"{gguf_key}.weight":
                    return f"model.layers.{layer_id}.{hf_key}.weight"
                if rest == f"{gguf_key}.bias":
                    return f"model.layers.{layer_id}.{hf_key}.bias"
            # Fallback: keep layer prefix but nest under model.layers
            return f"model.layers.{layer_id}.{rest}"

    return name


def _map_state_dict(state_dict: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    """Map all keys in a state_dict from GGUF to HuggingFace naming."""
    return {_map_gguf_name(k): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_AERO_TO_NP: dict[int, np.dtype] = {
    0:  np.float16,   # F16
    1:  np.float32,   # F32
    3:  np.float64,   # F64
    4:  np.int8,      # I8
    5:  np.uint8,     # U8
    6:  np.int16,     # I16
    7:  np.uint16,    # U16
    8:  np.int32,     # I32
    9:  np.uint32,    # U32
    10: np.int64,     # I64
    11: np.uint64,    # U64
    12: np.bool_,     # BOOL
}


def _decode_tensor(
    name: str,
    raw: bytes,
    shape: tuple[int, ...],
    dtype_val: int,
    tensor_info: dict[str, Any],
    *,
    target_dtype: Any = None,
) -> Any:
    """Decode a raw tensor from AERO bytes into a torch.Tensor."""
    import torch
    if target_dtype is None:
        target_dtype = torch.float32
    if dtype_val == DTYPE_PACKED:
        return _dequantize_packed(name, raw, shape, tensor_info, target_dtype)
    return _decode_standard(raw, shape, dtype_val, target_dtype)


def _dequantize_packed(
    name: str,
    raw: bytes,
    shape: tuple[int, ...],
    tensor_info: dict[str, Any],
    target_dtype: Any,
) -> Any:
    """Dequantize a PACKED (GGUF quantized) tensor to float."""
    try:
        from gguf.quants import (
            GGMLQuantizationType,
            dequantize,
            quant_shape_to_byte_shape,
        )
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required to dequantize quantized AERO models. "
            "Install it with: pip install gguf"
        ) from None

    qparams = tensor_info.get("quant_params") or {}
    ggml_type = qparams.get("ggml_type")
    if ggml_type is None:
        raise ValueError(
            f"Tensor {name!r} is PACKED but missing quant_params.ggml_type"
        )

    qtype = GGMLQuantizationType(ggml_type)
    byte_shape = quant_shape_to_byte_shape(shape, qtype)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(byte_shape)
    import torch
    deq = dequantize(arr, qtype)
    # Avoid redundant copy: from_numpy shares memory; clone only when converting dtype
    t = torch.from_numpy(deq)
    if target_dtype != torch.float32:
        t = t.to(target_dtype)
    return t


def _decode_standard(
    raw: bytes,
    shape: tuple[int, ...],
    dtype_val: int,
    target_dtype: Any,
) -> Any:
    """Decode a standard (non-quantized) tensor."""
    import torch
    if dtype_val == 2:  # BF16
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).clone()
    elif dtype_val in _AERO_TO_NP:
        np_dtype = _AERO_TO_NP[dtype_val]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        t = torch.from_numpy(arr)
    else:
        raise ValueError(
            f"Unsupported AERO dtype for standard tensor: {dtype_val}. "
            "Supported: 0=F16, 1=F32, 2=BF16, 3=F64, 4-12=I8..BOOL, or PACKED."
        )

    if target_dtype is not None and t.dtype != target_dtype:
        t = t.to(target_dtype)
    return t


# ---------------------------------------------------------------------------
# AeroModelLoader
# ---------------------------------------------------------------------------


class AeroModelLoader:
    """Read metadata and weights from an .aero file.

    Example::

        loader = AeroModelLoader("model.aero")
        print(loader.model_name, loader.architecture, loader.tensor_count)
        state_dict = loader.load_state_dict()
    """

    def __init__(self, aero_path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(aero_path))
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"aero_path does not exist: {self.path!r}")
        if not os.path.isfile(self.path):
            raise ValueError(f"aero_path is not a file: {self.path!r}")
        self._metadata: dict[str, Any] = {}
        self._tensor_count: int = 0
        self._file_size_mb: float = 0.0
        self._load_info()

    # -- Properties ----------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def model_name(self) -> str:
        return self._metadata.get("model_name", "unknown")

    @property
    def architecture(self) -> str:
        return self._metadata.get("architecture", "unknown")

    @property
    def source_format(self) -> str | None:
        return self._metadata.get("source_format")

    @property
    def tensor_count(self) -> int:
        return self._tensor_count

    @property
    def file_size_mb(self) -> float:
        return self._file_size_mb

    # -- Loading -------------------------------------------------------------

    def load_state_dict(
        self,
        *,
        map_names: bool | None = None,
        target_dtype: Any = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Load all tensors as a PyTorch state_dict.

        Uses the C++ loader (libaero state-dict API) when available and path is a
        local file; otherwise falls back to Python (AeroReader + gguf dequant).

        Args:
            map_names: Map GGUF names to HuggingFace format.
                       ``None`` (default) auto-detects: maps if source_format == "gguf".
            target_dtype: Convert all tensors to this dtype (default: float32).
            verbose: Print progress.

        Returns:
            A dict mapping tensor names to ``torch.Tensor``.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load state dict. Install with: pip install aero-runtime[inference]"
            ) from None
        if target_dtype is None:
            target_dtype = torch.float32
        if map_names is None:
            map_names = self.source_format == "gguf"

        # Try C++ path when libaero is available and path is a local file
        if os.path.isfile(self.path):
            try:
                from . import loader_cpp
                if loader_cpp.is_available():
                    t0 = time.time()
                    state_dict = loader_cpp.load_state_dict_cpp(
                        self.path,
                        map_names=map_names,
                        target_dtype=target_dtype,
                        verbose=verbose,
                    )
                    elapsed = time.time() - t0
                    if verbose:
                        logger.info(
                            "Loaded %d tensors in %.1f s (target dtype: %s) [C++ path]",
                            len(state_dict), elapsed, target_dtype,
                        )
                    return state_dict
            except Exception as e:
                if verbose:
                    logger.debug("C++ loader failed, falling back to Python: %s", e)

        # Python path
        state_dict: dict[str, Any] = {}
        t0 = time.time()
        with AeroReader(self.path) as r:
            tensors = r.list_tensors()
            if not tensors:
                raise ValueError(
                    f"AERO file has no tensors (TIDX empty or invalid): {self.path!r}"
                )
            for t in tensors:
                if not isinstance(t.get("name"), str) or "shape" not in t or "dtype" not in t:
                    raise ValueError(
                        "Malformed TIDX: each tensor must have 'name', 'shape', and 'dtype'"
                    )
            total = len(tensors)
            for idx, t in enumerate(tensors):
                name = t["name"]
                shape = tuple(t["shape"])
                dtype_val = t["dtype"]
                # Prefer zero-copy view for standard dtypes (mmap path); fallback to read_tensor_bytes
                if dtype_val != DTYPE_PACKED:
                    try:
                        with r.read_tensor_view(name) as v:
                            raw = v.view.tobytes()
                    except Exception:
                        raw = r.read_tensor_bytes(name)
                else:
                    raw = r.read_tensor_bytes(name)
                tensor = _decode_tensor(
                    name, raw, shape, dtype_val, t,
                    target_dtype=target_dtype,
                )
                del raw  # free immediately

                mapped_name = _map_gguf_name(name) if map_names else name
                state_dict[mapped_name] = tensor

                if verbose and (idx + 1) % 50 == 0:
                    logger.info("  loaded %d / %d tensors ...", idx + 1, total)

        elapsed = time.time() - t0
        if verbose:
            logger.info(
                "Loaded %d tensors in %.1f s (target dtype: %s)",
                total, elapsed, target_dtype,
            )
        return state_dict

    # -- Internal ------------------------------------------------------------

    def _load_info(self) -> None:
        """Read metadata and tensor count from the AERO file (fast, no weight data)."""
        try:
            self._file_size_mb = os.path.getsize(self.path) / (1024 * 1024)
        except OSError:
            self._file_size_mb = 0.0

        try:
            with AeroReader(self.path) as r:
                # Read JSON metadata chunk
                mjsn = r.get_chunk(FOURCC_MJSN)
                if mjsn is not None:
                    try:
                        raw = r.read_chunk_bytes(mjsn, verify_hash=True)
                        self._metadata = json.loads(raw)
                    except Exception:
                        pass

                # Count tensors
                try:
                    self._tensor_count = len(r.list_tensors())
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Could not read AERO metadata from %r: %s", self.path, e)
            # Keep defaults: _metadata={}, _tensor_count=0, _file_size_mb set above


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def load_state_dict(
    aero_path: str,
    *,
    map_names: bool | None = None,
    target_dtype: Any = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a PyTorch state_dict from an .aero file.

    Shortcut for ``AeroModelLoader(path).load_state_dict(...)``.
    """
    loader = AeroModelLoader(aero_path)
    return loader.load_state_dict(
        map_names=map_names,
        target_dtype=target_dtype,
        verbose=verbose,
    )
