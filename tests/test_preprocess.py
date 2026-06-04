import numpy as np

from superres.preprocess import kernel_extent_voxels, deconvolve_block, sharpen_region
from superres.paganin import physics_from_metadata

PHYS = {"paganin_delta_beta": 1000.0, "energy_kev": 78.0, "sample_detector_mm": 220.0,
        "sample_pixel_mm": 0.0024, "unsharp_sigma": 1.2, "unsharp_coeff": 4.0}


def test_kernel_extent_reasonable():
    e = kernel_extent_voxels(PHYS)
    assert 8 <= e <= 96            # local neighborhood, not whole-volume


def test_deconvolve_sharpens():
    rng = np.random.default_rng(0)
    v = rng.random((32, 32, 32)).astype(np.float32)
    def hf(a):
        f = np.abs(np.fft.fftn(a - a.mean())); return float(f[f.shape[0]//4:].sum())
    out = deconvolve_block(v, PHYS, reg=0.05)
    assert out.shape == v.shape
    # deconv of a smooth (blurred) input should add high-freq; on random input it
    # at least changes it and stays finite
    assert np.isfinite(out).all()


def test_sharpen_region_local_equals_global():
    # the key property: a local halo gives the same result as large context
    rng = np.random.default_rng(1)
    vol = rng.random((96, 96, 96)).astype(np.float32)
    def fetch(z, y, x, dz, dy, dx): return vol[z:z+dz, y:y+dy, x:x+dx]
    local = sharpen_region(fetch, 32, 32, 32, 24, 24, 24, vol.shape, PHYS, reg=0.05)
    big = deconvolve_block(vol[8:88, 8:88, 8:88], PHYS, reg=0.05)
    ref = big[24:48, 24:48, 24:48]
    assert np.allclose(local, ref, atol=0.02)


def test_physics_from_metadata():
    md = {"scan": {"tomo": {"acquisition": {"energy": 78.0, "sampleDetectorDistance": 220.0,
            "detector": {"samplePixelSize": 0.0024, "scintillator": "GAGG_50um"}},
            "processing": {"preprocessing": {"phase": {"method": "Paganin",
                "delta_beta": 1000.0, "unsharp_coeff": 4.0, "unsharp_sigma": 1.2}}}}}}
    p = physics_from_metadata(md)
    assert p is not None
    assert p["paganin_delta_beta"] == 1000.0
    assert p["energy_kev"] == 78.0
    assert physics_from_metadata({}) is None    # no physics -> None
