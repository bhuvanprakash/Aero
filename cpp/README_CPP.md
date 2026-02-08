# AERO v0.1 – C++ Reader Library

Cross-platform C++ 17 library and CLI tools for reading `.aero` model
container files.

## Building

### Prerequisites

- CMake ≥ 3.18
- A C++17 compiler (GCC, Clang, or MSVC)

All library dependencies are fetched automatically via CMake
`FetchContent`:

| Dependency | Purpose | Source |
|------------|---------|--------|
| [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) (C impl) | Per-chunk integrity | vendored via FetchContent |
| [zstd](https://github.com/facebook/zstd) | Decompress metadata chunks | FetchContent |
| [msgpack-c](https://github.com/msgpack/msgpack-c) | Parse MMSG / TIDX | FetchContent (header-only) |

### macOS / Linux

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j$(nproc)
```

### Windows (MSVC)

```powershell
cmake -S cpp -B cpp\build
cmake --build cpp\build --config Release
```

### Using System Dependencies

```bash
cmake -S cpp -B cpp/build \
  -DAERO_USE_SYSTEM_ZSTD=ON \
  -DAERO_USE_SYSTEM_MSGPACK=ON
```

## CLI Tools

All tools return exit code 0 on success, 1 on error.

### `aero_inspect <file.aero>`

Print header info, chunk table, and model metadata (if available).

### `aero_validate <file.aero> [--full]`

Verify BLAKE3 integrity of metadata chunks.  With `--full`, also
stream-verify weight shards (reads entire file).

### `aero_list_tensors <file.aero> [--limit N]`

Print tensor names, shapes, dtypes, and sizes from the TIDX chunk.

### `aero_extract_tensor <file.aero> "<name>" <out.bin>`

Extract raw bytes for a single tensor to a file.

## Library API

```cpp
#include <aero/aero.hpp>

aero::AeroFile file("model.aero");

// Iterate chunks
for (const auto& c : file.chunks())
    printf("%s %s\n", c.fourcc.c_str(), c.name.c_str());

// Load tensor index
auto tensors = file.load_tidx();

// Read tensor bytes (copy)
auto data = file.read_tensor_bytes("embed.weight");

// Zero-copy mmap view
auto view = file.view_tensor("embed.weight");
// view.data points directly into mmap'd weight shard
```

## Key Implementation Details

- **Unaligned mmap offsets**: AERO chunk offsets are 16-byte aligned, but
  OS mapping APIs require page/granularity alignment.  `MMapRegion::map()`
  maps from the nearest aligned base and returns a sliced view at the
  correct offset.

- **Streaming BLAKE3**: For large WTSH chunks, `stream_verify_chunk()`
  hashes in 64 MB blocks without loading the full shard into RAM.

- **Forward-compatible msgpack**: Unknown keys in TIDX / MMSG maps are
  silently ignored, allowing older readers to open files with new fields.
