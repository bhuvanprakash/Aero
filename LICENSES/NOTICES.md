# Licenses

This project is licensed under the **Apache License 2.0**. See the [LICENSE](../LICENSE) file in the repository root.

---

# Third-Party Licenses

This project uses the following open-source libraries.

## BLAKE3

- **Repository:** https://github.com/BLAKE3-team/BLAKE3
- **License:** CC0-1.0 / Apache-2.0 (dual-licensed)
- **Usage:** C implementation vendored via CMake FetchContent for
  per-chunk integrity hashing.

## Zstandard (zstd)

- **Repository:** https://github.com/facebook/zstd
- **License:** BSD-3-Clause / GPL-2.0 (dual-licensed)
- **Usage:** Metadata chunk compression/decompression.

## msgpack-c

- **Repository:** https://github.com/msgpack/msgpack-c
- **License:** BSL-1.0 (Boost Software License)
- **Usage:** MessagePack parsing for TIDX and MMSG chunks.

## nlohmann/json

- **Repository:** https://github.com/nlohmann/json
- **License:** MIT
- **Usage:** JSON parsing for AEROSET index files (C++ only).

## Python Dependencies

- **blake3** (PyPI): CC0-1.0 / Apache-2.0
- **zstandard** (PyPI): BSD-3-Clause
- **msgpack** (PyPI): Apache-2.0
- **numpy** (PyPI): BSD-3-Clause
