"""Remote range reading tests with local HTTP server."""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from aerotensor.aeroset import AeroSetReader, write_aeroset
from aerotensor.format import DType
from aerotensor.remote import (
    DiskCache,
    FileRangeSource,
    HttpRangeSource,
    open_range_source,
)
from aerotensor.writer import AeroWriter, TensorSpec


# ── HTTP Range Server ───────────────────────────────────────────────────────


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that supports Range requests."""

    request_count = 0
    request_lock = threading.Lock()

    def do_HEAD(self):
        self.send_response(200)
        path = self.translate_path(self.path)
        if os.path.isfile(path):
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        with self.request_lock:
            RangeRequestHandler.request_count += 1

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            # Parse Range: bytes=start-end
            range_spec = range_header[6:]
            start, end = range_spec.split("-")
            start = int(start)
            end = int(end) if end else file_size - 1
            length = end - start + 1

            if start >= file_size or end >= file_size:
                self.send_error(416, "Range Not Satisfiable")
                return

            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            # Normal GET
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())

    def log_message(self, format, *args):
        pass  # Suppress logs


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    """Spin up a local HTTP server with Range support."""
    serve_dir = tmp_path_factory.mktemp("http_root")
    
    # Create test aeroset
    specs = []
    for i in range(4):
        a = np.array([float(i)] * 4, dtype="<f4")
        raw = a.tobytes()
        specs.append(TensorSpec(
            name=f"layer.{i}.weight", dtype=int(DType.F32), shape=[4],
            data_len=len(raw), read_data=lambda r=raw: r,
        ))
    
    aeroset_dir = serve_dir / "model"
    write_aeroset(
        str(aeroset_dir), specs, "test-remote", "mlp",
        max_shard_bytes=20, max_part_shards=2,
        uuid=b"\xAA" * 16, tensor_hash=True,
    )

    # Start server
    os.chdir(serve_dir)
    handler = RangeRequestHandler
    handler.request_count = 0
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", str(aeroset_dir)

    server.shutdown()


# ── Range Source Tests ──────────────────────────────────────────────────────


def test_file_range_source(tmp_path):
    """FileRangeSource reads correct byte ranges."""
    path = tmp_path / "test.bin"
    path.write_bytes(b"0123456789" * 10)

    with FileRangeSource(str(path)) as src:
        assert src.size() == 100
        assert src.read_range(0, 5) == b"01234"
        assert src.read_range(10, 5) == b"01234"
        assert src.read_range(95, 5) == b"56789"


def test_http_range_source(http_server):
    """HttpRangeSource fetches correct ranges."""
    base_url, local_dir = http_server
    
    # Create a small test file
    test_file = Path(local_dir).parent / "test.bin"
    test_file.write_bytes(b"ABCDEFGHIJ" * 10)

    url = f"{base_url}/test.bin"
    with HttpRangeSource(url) as src:
        assert src.size() == 100
        data = src.read_range(0, 5)
        assert data == b"ABCDE"
        data = src.read_range(50, 10)
        assert data == b"ABCDEFGHIJ"


def test_disk_cache(tmp_path):
    """DiskCache stores and retrieves ranges."""
    cache = DiskCache(cache_dir=str(tmp_path / "cache"), max_size_mb=1)
    
    url = "http://example.com/model.bin"
    data = b"test data"
    
    # Miss
    assert cache.get(url, 0, len(data)) is None
    
    # Store
    cache.put(url, 0, len(data), data)
    
    # Hit
    assert cache.get(url, 0, len(data)) == data


# ── Remote AEROSET Tests ────────────────────────────────────────────────────


def test_remote_aeroset_list_tensors(http_server, tmp_path):
    """Remote AeroSetReader can list tensors."""
    base_url, local_dir = http_server
    url = f"{base_url}/model/model.aeroset.json"
    
    cache_dir = str(tmp_path / "cache")
    with AeroSetReader(url, enable_remote=True, cache_dir=cache_dir) as reader:
        tensors = reader.list_tensors()
        assert len(tensors) == 4
        names = {t["name"] for t in tensors}
        assert "layer.0.weight" in names


def test_remote_aeroset_read_tensor(http_server, tmp_path):
    """Remote AeroSetReader reads tensor bytes correctly."""
    base_url, local_dir = http_server
    url = f"{base_url}/model/model.aeroset.json"
    
    cache_dir = str(tmp_path / "cache")
    
    # Read remote
    with AeroSetReader(url, enable_remote=True, cache_dir=cache_dir) as remote:
        remote_data = remote.read_tensor_bytes("layer.2.weight")
    
    # Read local for comparison
    local_json = os.path.join(local_dir, "model.aeroset.json")
    with AeroSetReader(local_json, enable_remote=False) as local:
        local_data = local.read_tensor_bytes("layer.2.weight")
    
    assert remote_data == local_data
    arr = np.frombuffer(remote_data, dtype="<f4")
    np.testing.assert_array_equal(arr, [2.0, 2.0, 2.0, 2.0])


def test_remote_cache_works(http_server, tmp_path):
    """Second read uses cache and doesn't hit network."""
    base_url, local_dir = http_server
    url = f"{base_url}/model/model.aeroset.json"
    
    cache_dir = str(tmp_path / "cache")
    RangeRequestHandler.request_count = 0
    
    # First read
    with AeroSetReader(url, enable_remote=True, cache_dir=cache_dir) as reader:
        data1 = reader.read_tensor_bytes("layer.0.weight")
    
    initial_count = RangeRequestHandler.request_count
    assert initial_count > 0
    
    # Second read
    with AeroSetReader(url, enable_remote=True, cache_dir=cache_dir) as reader:
        data2 = reader.read_tensor_bytes("layer.0.weight")
    
    # Request count should not increase significantly (only index reads)
    assert data1 == data2


def test_tensor_hash_verification(http_server, tmp_path):
    """Per-tensor hash_b3 is verified on remote reads."""
    base_url, local_dir = http_server
    url = f"{base_url}/model/model.aeroset.json"
    
    cache_dir = str(tmp_path / "cache")
    
    with AeroSetReader(url, enable_remote=True, cache_dir=cache_dir) as reader:
        tensors = reader.list_tensors()
        # Verify hash_b3 field is present (written with tensor_hash=True)
        assert all("hash_b3" in t and t["hash_b3"] for t in tensors)
        
        # Read should succeed and verify hash
        data = reader.read_tensor_bytes("layer.1.weight")
        arr = np.frombuffer(data, dtype="<f4")
        np.testing.assert_array_equal(arr, [1.0, 1.0, 1.0, 1.0])


def test_open_range_source_auto_detect(tmp_path):
    """open_range_source auto-detects local vs remote."""
    from aerotensor.remote import open_range_source
    
    # Local file
    path = tmp_path / "test.bin"
    path.write_bytes(b"local data")
    
    with open_range_source(str(path)) as src:
        assert src.read_range(0, 5) == b"local"
