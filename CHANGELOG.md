# Changelog

All notable changes to AERO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-09

### Added
- **AEROSET v0.2**: Remote HTTP Range fetching support for distributed model loading
  - Parts can now be URLs (https://...) for remote access
  - Optional `base_url` field for relative path resolution
  - HTTP Range request support with disk caching
  - Per-tensor BLAKE3 hashes (`hash_b3`) for slice integrity verification
  - New `aerotensor fetch-tensor` CLI command for remote tensor extraction
  - Disk cache with configurable size limits and automatic cleanup
- Remote range reading infrastructure (`aerotensor.remote` module)
  - `FileRangeSource` and `HttpRangeSource` implementations
  - Range coalescing for efficient batch reads
  - Safety caps: 64MB per request, 2GB per tensor
- New CLI commands:
  - `aerotensor --version` to display version information
  - `aerotensor info` as alias for `inspect`
  - `aerotensor convert auto` for automatic format detection
  - `aerotensor fetch-tensor` for remote tensor extraction
- Writer support for optional per-tensor hashing (`--tensor-hash` flag)
- Comprehensive remote range tests with local HTTP server

### Changed
- AEROSET validation now auto-detects and validates remote URLs
- `validate` command supports `--remote` flag for URL-based aerosets
- Improved CLI help text and error messages

### Fixed
- None

## [0.1.1] - 2025-02-08

### Added
- Atomic file writes with fsync and temporary file handling
- Comprehensive safety limits and hardening:
  - Entry count cap (1M entries)
  - String table cap (512 MB)
  - Metadata uncompressed length cap (2 GB)
  - GGUF parser safety caps (10M tensors, 10M KV pairs, 32 dimensions, 10MB strings)
- Safetensors converter validation:
  - Offset bounds checking
  - Data span consistency validation (dtype × shape == byte span)
  - Missing field detection
- Fuzz mutation tests for robustness verification
- C++ integration tests (Python writes → C++ reads)
- `validate --full` flag for complete chunk verification including WTSH
- AEROSET auto-detection in validate command
- Documentation:
  - `LICENSE` (Apache-2.0)
  - `LICENSES/NOTICES.md` (third-party notices)
  - `docs/COMPAT.md` (compatibility rules)
  - `docs/SECURITY.md` (updated with safety limits)

### Changed
- Reader now enforces strict bounds checking at all parsing levels
- GGUF converter validates tensor offsets before reading
- Improved error messages with specific context

### Fixed
- Python 3.9 compatibility in type annotations

## [0.1.0] - 2025-02-07

### Added
- Initial AERO v0.1 binary format specification
- Single-file `.aero` container with multi-shard WTSH support
- AEROSET v0.1 multi-file mode (index + parts)
- Python implementation:
  - Writer with streaming support (one tensor in memory at a time)
  - Reader with mmap support for zero-copy access
  - CLI tools (inspect, validate, make-test-vector)
  - AEROSET writer and reader
- Converters:
  - safetensors → AERO (streaming)
  - Sharded safetensors → AEROSET (streaming)
  - GGUF → AERO (preserves packed quantization blocks)
- C++ cross-platform implementation:
  - Reader library (macOS, Linux, Windows)
  - CLI tools (inspect, validate, list_tensors, extract_tensor)
  - AEROSET support (inspect_set, extract_tensor_set)
  - Memory-mapped I/O with unaligned offset handling
- Per-chunk BLAKE3-256 integrity verification
- Deterministic output (same inputs → same bytes)
- Comprehensive test suite (60 tests)
- CI workflows:
  - Python tests (Linux, macOS, Windows)
  - C++ build and smoke tests
  - Integration tests (Python ↔ C++)
- Documentation:
  - `docs/FORMAT.md` (binary format specification)
  - `docs/AEROSET.md` (multi-file container specification)
  - `docs/CONVERTERS.md` (converter usage)
  - `docs/SECURITY.md` (safety philosophy)
  - `README.md` (comprehensive guide)

[0.2.0]: https://github.com/Aero-HQ/Aero/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Aero-HQ/Aero/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Aero-HQ/Aero/releases/tag/v0.1.0
