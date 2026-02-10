"""AERO v0.2 – a chunked model container format."""

__version__ = "0.2.0"

from .format import (
    MAGIC, VERSION_MAJOR, VERSION_MINOR,
    ChunkFlags, DType, DTYPE_NAMES,
    FOURCC_MJSN, FOURCC_MMSG, FOURCC_TIDX, FOURCC_WTSH,
)
from .reader import AeroError, AeroReader, ChunkRef, IntegrityError
from .writer import AeroWriter, Chunk, TensorSpec, plan_shards
from .aeroset import AeroSetReader, write_aeroset

__all__ = [
    "__version__",
    "MAGIC", "VERSION_MAJOR", "VERSION_MINOR",
    "ChunkFlags", "DType", "DTYPE_NAMES",
    "FOURCC_MJSN", "FOURCC_MMSG", "FOURCC_TIDX", "FOURCC_WTSH",
    "AeroReader", "AeroWriter", "AeroError", "IntegrityError",
    "ChunkRef", "Chunk", "TensorSpec", "plan_shards",
    "AeroSetReader", "write_aeroset",
]
