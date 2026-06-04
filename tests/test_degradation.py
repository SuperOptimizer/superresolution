import numpy as np

from superres.degradation import (
    RandomDegradation,
    DegradationRanges,
    anisotropic_blur,
    quantize_u8,
    forward_blur_only,
    measure_psf_from_edge,
)


def _hf_energy(v):
    f = np.abs(np.fft.fftn(v - v.mean()))
    return float(f[f.shape[0] // 4 :].sum())


def test_degradation_is_shape_preserving():
    rng = np.random.default_rng(0)
    deg = RandomDegradation(DegradationRanges())
    clean = rng.random((24, 24, 24)).astype("float32")
    out, params = deg.apply(clean, rng)
    assert out.shape == clean.shape  # x1: no grid change
    assert out.dtype == np.float32
    assert set(["sigma_z", "sigma_xy", "noise_sigma"]).issubset(params)


def test_degradation_reduces_high_frequency_energy():
    rng = np.random.default_rng(1)
    clean = rng.random((32, 32, 32)).astype("float32")
    # blur-only (no added noise) must reduce HF energy
    blurred = anisotropic_blur(clean, 1.5, 1.0)
    assert _hf_energy(blurred) < _hf_energy(clean)


def test_degradation_deterministic_under_seed():
    clean = np.random.default_rng(2).random((16, 16, 16)).astype("float32")
    a, pa = RandomDegradation(DegradationRanges()).apply(clean, np.random.default_rng(7))
    b, pb = RandomDegradation(DegradationRanges()).apply(clean, np.random.default_rng(7))
    assert np.allclose(a, b)
    assert pa["sigma_z"] == pb["sigma_z"]


def test_quantize_u8_roundtrip_levels():
    v = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, 1, 256)
    q = quantize_u8(v)
    assert q.min() >= 0 and q.max() <= 1
    # quantized values land on the 1/255 grid
    assert np.allclose(q * 255, np.round(q * 255))


def test_forward_blur_only_matches_params():
    rng = np.random.default_rng(3)
    v = rng.random((16, 16, 16)).astype("float32")
    params = {"sigma_z": 1.2, "sigma_xy": 0.8}
    out = forward_blur_only(v, params)
    assert out.shape == v.shape
    assert _hf_energy(out) < _hf_energy(v)


def test_measure_psf_from_clean_edge():
    # an oversampled blurred step edge -> recover roughly the blur sigma
    x = np.arange(101)
    from scipy.ndimage import gaussian_filter1d

    sigma_true = 3.0
    esf = gaussian_filter1d((x >= 50).astype(float), sigma_true)
    res = measure_psf_from_edge(esf)
    assert abs(res["sigma"] - sigma_true) < 0.6
    assert res["mtf"][0] == 1.0  # normalized DC
