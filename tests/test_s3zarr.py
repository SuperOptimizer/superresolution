import json

import numpy as np

from superres.s3zarr import Zarr3D, _Backend


class DictBackend(_Backend):
    """In-memory backend for testing -- no network. Maps key -> bytes."""

    def __init__(self, store):
        self.store = store

    def get(self, key):
        return self.store.get(key)


def _make_store(shape, chunks, sep="/", fill=0):
    zarray = {
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": "|u1",
        "fill_value": fill,
        "order": "C",
        "filters": None,
        "compressor": None,
        "dimension_separator": sep,
        "zarr_format": 2,
    }
    return {"0/.zarray": json.dumps(zarray).encode()}


def test_zarray_parsing():
    store = _make_store((10, 10, 10), (4, 4, 4))
    z = Zarr3D(DictBackend(store))
    assert z.shape == (10, 10, 10)
    assert z.chunks == (4, 4, 4)
    assert z.dtype == np.dtype("u1")
    assert z.sep == "/"
    assert z.n_chunks() == (3, 3, 3)


def test_missing_chunk_is_fill_value():
    store = _make_store((8, 8, 8), (4, 4, 4), fill=0)
    z = Zarr3D(DictBackend(store))
    # no chunk objects present -> all fill (padding)
    chunk = z.get_chunk(0, 0, 0)
    assert chunk.shape == (4, 4, 4)
    assert np.all(chunk == 0)
    assert not z.chunk_occupied(0, 0, 0)


def test_chunk_reshaping_and_region_assembly():
    shape = (8, 8, 8)
    chunks = (4, 4, 4)
    store = _make_store(shape, chunks)
    # build a ground-truth volume and split into raw chunks
    gt = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    for cz in range(2):
        for cy in range(2):
            for cx in range(2):
                block = gt[cz * 4 : cz * 4 + 4, cy * 4 : cy * 4 + 4, cx * 4 : cx * 4 + 4]
                store[f"0/{cz}/{cy}/{cx}"] = block.tobytes()
    z = Zarr3D(DictBackend(store))
    # full read reconstructs the volume
    out = z.read_region(0, 0, 0, 8, 8, 8)
    assert np.array_equal(out, gt)
    # a cross-chunk subregion
    sub = z.read_region(2, 2, 2, 4, 4, 4)
    assert np.array_equal(sub, gt[2:6, 2:6, 2:6])
    assert z.chunk_occupied(1, 1, 1)


def test_region_out_of_bounds_raises():
    store = _make_store((8, 8, 8), (4, 4, 4))
    z = Zarr3D(DictBackend(store))
    import pytest

    with pytest.raises(IndexError):
        z.read_region(6, 0, 0, 4, 4, 4)


def test_dimension_separator_dot():
    store = _make_store((4, 4, 4), (4, 4, 4), sep=".")
    gt = np.ones((4, 4, 4), dtype=np.uint8) * 7
    store["0/0.0.0"] = gt.tobytes()
    z = Zarr3D(DictBackend(store))
    assert np.array_equal(z.read_region(0, 0, 0, 4, 4, 4), gt)
