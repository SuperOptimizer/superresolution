"""Dependency-light reader for the Vesuvius OME-Zarr volumes.

The level-0 arrays are uncompressed raw u8 with 128^3 chunks and
``dimension_separator: "/"`` (chunk key ``{level}/{z}/{y}/{x}``). Because chunks
are raw bytes with no codec, a chunk is just a plain object we can fetch directly
-- no zarr library, boto3, or s3fs required. We use the AWS CLI (`aws s3 cp ... -`)
with ``--no-sign-request`` against the public bucket, with an optional boto3 fast
path and a local-filesystem mode for pre-downloaded volumes.

Missing chunks (NoSuchKey) are treated as `fill_value` blocks -- this is how the
zarr format encodes the all-zero padding around the central ROI.

NOTE: only the no-compressor / C-order / raw-u8 case is supported, which is what
this dataset uses. We assert it rather than silently mis-decode.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------
# Backends: object fetch returning bytes or None (missing)
# ----------------------------------------------------------------------------
class _Backend:
    def get(self, key: str) -> Optional[bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_text(self, key: str) -> str:
        data = self.get(key)
        if data is None:
            raise FileNotFoundError(key)
        return data.decode("utf-8")


class LocalBackend(_Backend):
    """Read a zarr directory already on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, key: str) -> Optional[bytes]:
        p = self.root / key
        if not p.exists():
            return None
        return p.read_bytes()


class S3CLIBackend(_Backend):
    """Fetch objects via the `aws` CLI with --no-sign-request (no boto3 needed)."""

    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{key}"

    def get(self, key: str) -> Optional[bytes]:
        proc = subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", "--quiet", self._uri(key), "-"],
            capture_output=True,
        )
        if proc.returncode != 0:
            # Missing key (padding) or transient error -> treat as absent.
            return None
        return proc.stdout


class S3Boto3Backend(_Backend):
    """Faster S3 fetch when boto3 is installed (unsigned public access)."""

    def __init__(self, bucket: str, prefix: str):
        import boto3  # type: ignore
        from botocore import UNSIGNED  # type: ignore
        from botocore.config import Config  # type: ignore

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self._err = __import__("botocore").exceptions.ClientError

    def get(self, key: str) -> Optional[bytes]:
        full = f"{self.prefix}/{key}"
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=full)
            return obj["Body"].read()
        except self._err as e:  # NoSuchKey etc.
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "AccessDenied"):
                return None
            raise


def _make_s3_backend(bucket: str, prefix: str) -> _Backend:
    try:
        return S3Boto3Backend(bucket, prefix)
    except Exception:
        return S3CLIBackend(bucket, prefix)


# ----------------------------------------------------------------------------
# Zarr array
# ----------------------------------------------------------------------------
class Zarr3D:
    """A single 3-D zarr array (one multiscale level) over a pluggable backend.

    `level_prefix` is the key prefix of the array within the backend's root, e.g.
    "0" for multiscale level 0. Chunk keys are then "0/z/y/x".
    """

    def __init__(self, backend: _Backend, level: int = 0, cache_chunks: int = 64):
        self.backend = backend
        self.level = int(level)
        self.lp = str(level)
        meta = json.loads(backend.get_text(f"{self.lp}/.zarray"))
        self.shape = tuple(int(s) for s in meta["shape"])
        self.chunks = tuple(int(c) for c in meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill_value = meta.get("fill_value", 0)
        self.sep = meta.get("dimension_separator", ".")
        # Only the raw / C-order case is supported (what this dataset uses).
        assert meta.get("compressor") is None, "only uncompressed chunks supported"
        assert meta.get("filters") in (None, []), "filters not supported"
        assert meta.get("order", "C") == "C", "only C-order supported"
        assert len(self.shape) == 3, "Zarr3D expects a 3-D array"
        self._chunk_elems = int(np.prod(self.chunks))
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_cap = cache_chunks
        self._lock = threading.Lock()

    # -- chunk access -------------------------------------------------------
    def _chunk_key(self, cz: int, cy: int, cx: int) -> str:
        s = self.sep
        return f"{self.lp}/{cz}{s}{cy}{s}{cx}"

    def _fill_chunk(self) -> np.ndarray:
        return np.full(self.chunks, self.fill_value, dtype=self.dtype)

    def get_chunk(self, cz: int, cy: int, cx: int) -> np.ndarray:
        """Return a full chunk (chunk-shaped). Missing -> fill_value block.

        Thread-safe: the cache dict ops are locked; the (slow) network fetch runs
        outside the lock so concurrent fetches actually parallelize.
        """
        key = (cz, cy, cx)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        raw = self.backend.get(self._chunk_key(cz, cy, cx))  # outside lock
        if raw is None:
            arr = self._fill_chunk()
        else:
            buf = np.frombuffer(raw, dtype=self.dtype)
            if buf.size != self._chunk_elems:
                raise ValueError(
                    f"chunk {key}: got {buf.size} elems, expected {self._chunk_elems}"
                )
            arr = buf.reshape(self.chunks).copy()
        with self._lock:
            self._cache[key] = arr
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)
        return arr

    def chunk_occupied(self, cz: int, cy: int, cx: int) -> bool:
        """True if the chunk has any nonzero voxel. Missing chunks are empty.

        Cheap occupancy used to skip all-zero padding without reading full data
        into the training/inference path (here it still reads the chunk; for the
        real run, precompute an occupancy mask from chunk presence + a min/max scan).
        """
        raw = self.backend.get(self._chunk_key(cz, cy, cx))
        if raw is None:
            return False
        return np.frombuffer(raw, dtype=self.dtype).any()

    # -- region access ------------------------------------------------------
    def read_region(self, z0: int, y0: int, x0: int, dz: int, dy: int, dx: int,
                    workers: int = 16) -> np.ndarray:
        """Read an arbitrary box [z0:z0+dz, ...], assembling from chunks.

        The box must lie within bounds. Returns an array of shape (dz, dy, dx).
        Chunk fetches are I/O-bound (subprocess/network) so we parallelize them
        across a thread pool (`workers`); set workers<=1 for serial.
        """
        for o, d, s in zip((z0, y0, x0), (dz, dy, dx), self.shape):
            if o < 0 or o + d > s:
                raise IndexError(f"region out of bounds: start {o}, size {d}, dim {s}")
        out = np.empty((dz, dy, dx), dtype=self.dtype)
        cz0, cy0, cx0 = (z0 // self.chunks[0], y0 // self.chunks[1], x0 // self.chunks[2])
        cz1 = (z0 + dz - 1) // self.chunks[0]
        cy1 = (y0 + dy - 1) // self.chunks[1]
        cx1 = (x0 + dx - 1) // self.chunks[2]
        coords = [(cz, cy, cx)
                  for cz in range(cz0, cz1 + 1)
                  for cy in range(cy0, cy1 + 1)
                  for cx in range(cx0, cx1 + 1)]

        def place(cell):
            cz, cy, cx = cell
            chunk = self.get_chunk(cz, cy, cx)
            gz, gy, gx = cz * self.chunks[0], cy * self.chunks[1], cx * self.chunks[2]
            sz0, sz1 = max(z0, gz), min(z0 + dz, gz + self.chunks[0])
            sy0, sy1 = max(y0, gy), min(y0 + dy, gy + self.chunks[1])
            sx0, sx1 = max(x0, gx), min(x0 + dx, gx + self.chunks[2])
            out[sz0 - z0 : sz1 - z0, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = chunk[
                sz0 - gz : sz1 - gz, sy0 - gy : sy1 - gy, sx0 - gx : sx1 - gx]

        if workers and workers > 1 and len(coords) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(place, coords))
            return out
        for cz in range(cz0, cz1 + 1):
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self.get_chunk(cz, cy, cx)
                    # chunk's global extent
                    gz, gy, gx = cz * self.chunks[0], cy * self.chunks[1], cx * self.chunks[2]
                    # overlap of [z0,z0+dz) with [gz, gz+chunk_z)
                    sz0, sz1 = max(z0, gz), min(z0 + dz, gz + self.chunks[0])
                    sy0, sy1 = max(y0, gy), min(y0 + dy, gy + self.chunks[1])
                    sx0, sx1 = max(x0, gx), min(x0 + dx, gx + self.chunks[2])
                    out[sz0 - z0 : sz1 - z0, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = chunk[
                        sz0 - gz : sz1 - gz, sy0 - gy : sy1 - gy, sx0 - gx : sx1 - gx
                    ]
        return out

    def n_chunks(self) -> tuple[int, int, int]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))  # ceil-div


# ----------------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------------
def open_s3(bucket: str, scroll: str, zarr_path: str, level: int = 0) -> Zarr3D:
    """Open a level of an OME-Zarr volume on S3.

    `zarr_path` is relative to the scroll, e.g. "volumes/2025...-masked.zarr".
    """
    prefix = f"{scroll}/{zarr_path}".strip("/")
    return Zarr3D(_make_s3_backend(bucket, prefix), level=level)


def open_local(zarr_root: str | Path, level: int = 0) -> Zarr3D:
    """Open a level of a zarr directory already on disk."""
    return Zarr3D(LocalBackend(zarr_root), level=level)
