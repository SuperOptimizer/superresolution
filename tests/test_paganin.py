import numpy as np

from superres.paganin import (
    wavelength_m, paganin_transfer, unsharp_transfer, recon_transfer,
    apply_recon_blur, deconvolve_recon,
)

PHYS = {"paganin_delta_beta": 1000.0, "energy_kev": 78.0,
        "sample_detector_mm": 220.0, "sample_pixel_mm": 0.0024,
        "unsharp_sigma": 1.2, "unsharp_coeff": 4.0}


def test_wavelength():
    # 78 keV -> ~0.0159 nm
    assert 1.4e-11 < wavelength_m(78.0) < 1.7e-11


def test_paganin_is_lowpass():
    k = np.array([0.01, 0.1, 0.25, 0.5])
    T = paganin_transfer(k, 1000.0, 78.0, 220.0, 2.4)
    assert np.all(np.diff(T) < 0)        # strictly decreasing
    assert T[0] <= 1.0 and T[-1] > 0


def test_unsharp_boosts_high_freq():
    k = np.array([0.01, 0.25, 0.5])
    U = unsharp_transfer(k, 1.2, 4.0)
    assert U[-1] > U[0] > 1.0            # boosts, increasing


def test_forward_blur_softens():
    rng = np.random.default_rng(0)
    clean = rng.random((32, 32, 32)).astype(np.float32)
    def hf(v):
        f = np.abs(np.fft.fftn(v - v.mean())); return float(f[f.shape[0]//4:].sum())
    blurred = apply_recon_blur(clean, PHYS, strength=1.0)
    assert blurred.shape == clean.shape
    assert hf(blurred) < hf(clean)       # net low-pass


def test_deconvolution_recovers():
    rng = np.random.default_rng(1)
    clean = rng.random((32, 32, 32)).astype(np.float32)
    data = apply_recon_blur(clean, PHYS, strength=1.0)
    rec = deconvolve_recon(data, PHYS, reg=0.01)
    def hf(v):
        f = np.abs(np.fft.fftn(v - v.mean())); return float(f[f.shape[0]//4:].sum())
    assert hf(rec) > hf(data)            # recovers high-freq vs the blurred data
