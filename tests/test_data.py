import json

import numpy as np

from superres.data import (
    SyntheticPatchDataset,
    PatchDataset,
    robust_normalize,
    random_symmetry,
)
from superres.s3zarr import Zarr3D
from tests.test_s3zarr import DictBackend


def test_synthetic_dataset_yields_pairs():
    ds = SyntheticPatchDataset(volume_shape=(48, 48, 48), patch=(32, 32, 32), length=4, seed=1)
    x, y = ds[0]
    assert x.shape == (1, 32, 32, 32)
    assert y.shape == x.shape           # same grid
    assert float((x - y).abs().mean()) > 0   # input genuinely degraded
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0001


def test_robust_normalize_range_and_outlier_resistance():
    rng = np.random.default_rng(0)
    v = rng.uniform(50.0, 150.0, size=(10, 10, 10)).astype(np.float32)  # signal spread
    v[5, 5, 5] = 1e6  # single extreme outlier
    out = robust_normalize(v)
    assert out.min() >= 0 and out.max() <= 1.0001
    # min/max normalization would crush the real signal to ~0 because of the 1e6
    # outlier; percentile clipping keeps the bulk of the signal spread across [0,1].
    bulk = out[v < 1e5]
    assert bulk.std() > 0.1
    assert bulk.mean() > 0.1


def test_random_symmetry_preserves_content():
    rng = np.random.default_rng(0)
    v = rng.random((8, 8, 8)).astype(np.float32)
    out = random_symmetry(v, np.random.default_rng(3), inplane_only=False)
    assert out.shape == v.shape
    assert np.isclose(out.sum(), v.sum())  # permutation/flip conserves voxels


def test_inplane_only_keeps_z_axis():
    v = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    # in-plane only: z dimension size must be unchanged
    out = random_symmetry(v, np.random.default_rng(5), inplane_only=True)
    assert out.shape[0] == 2


def _occupancy_store():
    # 16^3 volume of 4^3 chunks; the lower half in z is solid signal, upper half
    # is all-zero padding. Importance sampling should land patches in the signal.
    shape, chunks = (16, 16, 16), (4, 4, 4)
    zarray = {
        "shape": list(shape), "chunks": list(chunks), "dtype": "|u1",
        "fill_value": 0, "order": "C", "filters": None, "compressor": None,
        "dimension_separator": "/", "zarr_format": 2,
    }
    store = {"0/.zarray": json.dumps(zarray).encode()}
    block = np.full(chunks, 200, dtype=np.uint8)
    for cz in range(2):                 # z chunks 0,1 == lower half occupied
        for cy in range(4):
            for cx in range(4):
                store[f"0/{cz}/{cy}/{cx}"] = block.tobytes()
    return store


def test_importance_sampling_prefers_occupied():
    z = Zarr3D(DictBackend(_occupancy_store()))
    # half the volume is signal; with occupancy_min=0.5 a meaningful fraction of
    # start positions qualify, so importance sampling reliably finds signal.
    ds = PatchDataset(
        z, patch=(4, 4, 4), occupancy_min=0.5, length=8, seed=0, augment=False,
        max_reject=200,
    )
    # every returned clean patch must clear the occupancy floor -- pure air rejected
    for i in range(6):
        x, y = ds[i]
        assert float((y > 0).float().mean()) >= 0.5
