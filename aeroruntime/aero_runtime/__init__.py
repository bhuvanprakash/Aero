"""aero-runtime — Run inference directly from .aero model files.

Usage (CLI):
    pip install aero-runtime
    aero-run model.aero --prompt "Hello!" --model-id Qwen/Qwen3-1.7B

Usage (Python):
    from aero_runtime import AeroModel

    model = AeroModel("model.aero", model_id="Qwen/Qwen3-1.7B")
    print(model.generate("What is 2+2?"))
"""

__version__ = "0.1.0"

from .loader import AeroModelLoader, load_state_dict
from .model import AeroModel

__all__ = [
    "__version__",
    "AeroModel",
    "AeroModelLoader",
    "load_state_dict",
]
