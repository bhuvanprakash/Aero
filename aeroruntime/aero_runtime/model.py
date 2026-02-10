"""High-level AeroModel — load from .aero and generate text.

Example::

    from aero_runtime import AeroModel

    model = AeroModel("model.aero", model_id="Qwen/Qwen3-1.7B")
    print(model.generate("What is 2+2?"))
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .loader import AeroModelLoader

if TYPE_CHECKING:
    import torch


def _get_torch():
    """Import torch; raise with install hint if not installed (inference extra)."""
    try:
        import torch as t
        return t
    except ImportError:
        raise ImportError(
            "PyTorch is required for inference. Install with:\n"
            "  pip install aero-runtime[inference]\n"
            "or: pip install torch"
        ) from None

logger = logging.getLogger("aero_runtime")

# ---------------------------------------------------------------------------
# Heuristic HF model-ID detection from AERO metadata
# ---------------------------------------------------------------------------

# (name_contains, architecture, suggested_hf_id)
_HF_HINTS: list[tuple[str, str | None, str]] = [
    ("qwen3",       "qwen2",  "Qwen/Qwen3-1.7B"),
    ("qwen2.5",     "qwen2",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2",       "qwen2",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("llama-3.1-8b","llama",  "meta-llama/Llama-3.1-8B-Instruct"),
    ("llama-3",     "llama",  "meta-llama/Llama-3.1-8B-Instruct"),
    ("llama",       "llama",  "meta-llama/Llama-3.1-8B-Instruct"),
    ("smollm2",     None,     "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("smollm",      None,     "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("phi-3",       "phi3",   "microsoft/Phi-3-mini-4k-instruct"),
    ("phi3",        "phi3",   "microsoft/Phi-3-mini-4k-instruct"),
    ("mistral",     "llama",  "mistralai/Mistral-7B-Instruct-v0.3"),
    ("gemma",       "gemma",  "google/gemma-2b"),
]


def _guess_hf_model_id(metadata: dict[str, Any]) -> str | None:
    """Best-effort guess of HuggingFace model ID from AERO metadata."""
    model_name = (metadata.get("model_name") or "").lower()
    architecture = (metadata.get("architecture") or "").lower()

    for name_hint, arch_hint, hf_id in _HF_HINTS:
        if name_hint in model_name:
            if arch_hint is None or arch_hint == architecture:
                return hf_id
    # Fallback: try architecture alone
    for _, arch_hint, hf_id in _HF_HINTS:
        if arch_hint and arch_hint == architecture:
            return hf_id
    return None


# ---------------------------------------------------------------------------
# AeroModel
# ---------------------------------------------------------------------------


class AeroModel:
    """Load a model from an .aero file and run generation.

    Args:
        aero_path: Path to the ``.aero`` file.
        model_id: HuggingFace model ID for config and tokenizer.
                  If ``None``, the runtime tries to auto-detect from
                  the AERO metadata (best-effort).
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"`` (default).
        dtype: Torch dtype for weights after loading (default: ``torch.float32``).
               Use ``torch.float16`` to halve memory usage.

    Example::

        model = AeroModel("qwen3_1_7b.aero", model_id="Qwen/Qwen3-1.7B")
        print(model.generate("Explain gravity in one sentence."))
    """

    def __init__(
        self,
        aero_path: str,
        model_id: str | None = None,
        *,
        device: str = "auto",
        dtype: Any = None,
    ) -> None:
        torch = _get_torch()
        if dtype is None:
            dtype = torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)
        self._loader = AeroModelLoader(aero_path)
        self._model_id = model_id
        self._device_str = device
        self._dtype = dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

        self._resolve_model_id()
        self._resolve_device()
        # Weights and HF model/tokenizer are loaded on first generate() via _ensure_loaded()

    # -- Public properties ---------------------------------------------------

    @property
    def model_id(self) -> str | None:
        """HuggingFace model ID used for config / tokenizer (None if auto-detect failed)."""
        return self._model_id

    @property
    def loader(self) -> AeroModelLoader:
        """The underlying :class:`AeroModelLoader`."""
        return self._loader

    @property
    def device(self) -> Any:
        if self._device is None:
            raise RuntimeError("Model not loaded; device is set after first generate() or load_weights()")
        return self._device

    # -- Generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from a prompt.

        Returns the full generated string (prompt + completion).
        """
        self._ensure_loaded()

        torch = _get_torch()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -- Info ----------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return a summary dict about the loaded model."""
        dev = str(self._device) if self._device is not None else self._device_str
        return {
            "aero_path": self._loader.path,
            "model_id": self._model_id,
            "model_name": self._loader.model_name,
            "architecture": self._loader.architecture,
            "source_format": self._loader.source_format,
            "tensor_count": self._loader.tensor_count,
            "file_size_mb": round(self._loader.file_size_mb, 2),
            "device": dev,
            "dtype": str(self._dtype),
        }

    # -- Internal ------------------------------------------------------------

    def _resolve_model_id(self) -> None:
        """Determine the HuggingFace model ID (or leave None if auto-detect fails)."""
        if self._model_id is not None:
            return  # user-provided

        guessed = _guess_hf_model_id(self._loader.metadata)
        if guessed is not None:
            logger.info("Auto-detected HF model ID: %s", guessed)
            self._model_id = guessed
        else:
            self._model_id = None
            logger.warning(
                "Could not auto-detect HuggingFace model ID (model_name=%r, architecture=%r). "
                "Provide model_id=... for inference.",
                self._loader.model_name,
                self._loader.architecture,
            )

    def _resolve_device(self) -> None:
        """Set _device from _device_str (so device property works before load)."""
        torch = _get_torch()
        if self._device_str == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = self._device_str
        self._device = torch.device(dev)

    def _ensure_loaded(self) -> None:
        """Load weights and build model on first use."""
        if self._model is not None:
            return
        self._load()

    def _load(self) -> None:
        """Load config, tokenizer, weights, and build the model."""
        if self._model_id is None:
            raise ValueError(
                "model_id is required for loading weights and config. "
                "Could not auto-detect from AERO metadata; provide model_id=... when creating AeroModel."
            )
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for inference. "
                "Install with: pip install aero-runtime[inference]"
            ) from None

        torch = _get_torch()

        # 1. Config & tokenizer from HuggingFace
        print(f"[aero-runtime] Loading config/tokenizer from {self._model_id} ...")
        config = AutoConfig.from_pretrained(self._model_id, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )

        # 2. Load state_dict from AERO
        print(
            f"[aero-runtime] Loading weights from {self._loader.path} "
            f"({self._loader.file_size_mb:.0f} MB, "
            f"{self._loader.tensor_count} tensors) ..."
        )
        t0 = time.time()
        state_dict = self._loader.load_state_dict(
            target_dtype=self._dtype, verbose=False
        )
        load_time = time.time() - t0
        print(f"[aero-runtime] Weights loaded in {load_time:.1f} s")

        # 3. Build model from config and inject weights
        print("[aero-runtime] Building model ...")
        model = AutoModelForCausalLM.from_config(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del state_dict  # free memory

        if missing:
            logger.warning("  %d missing keys (may be expected)", len(missing))
        if unexpected:
            logger.warning("  %d unexpected keys", len(unexpected))

        # 4. Move to device
        if self._device_str == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = self._device_str
        self._device = torch.device(dev)

        model.eval()
        model = model.to(self._device)
        self._model = model

        print(f"[aero-runtime] Ready on {self._device}")

    def load_weights(self) -> None:
        """Explicitly load weights and build the model (optional; called automatically on first generate())."""
        self._ensure_loaded()
