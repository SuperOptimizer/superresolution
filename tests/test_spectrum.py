import numpy as np

from superres.spectrum import radial_power_spectrum, overshoot_metric
from superres.degradation import anisotropic_blur


def test_radial_spectrum_shape_and_range():
    rng = np.random.default_rng(0)
    v = rng.random((32, 32, 32)).astype(np.float32)
    freqs, power = radial_power_spectrum(v, nbins=32)
    assert freqs.shape == power.shape == (32,)
    assert freqs[0] >= 0 and freqs[-1] <= 0.5  # up to grid Nyquist
    assert np.all(power >= 0)


def test_blurred_has_less_high_freq_power():
    rng = np.random.default_rng(1)
    v = rng.random((32, 32, 32)).astype(np.float32)
    blurred = anisotropic_blur(v, 2.0, 2.0)
    _, p_clean = radial_power_spectrum(v)
    _, p_blur = radial_power_spectrum(blurred)
    # in the top frequency shells, blur attenuates power
    assert p_blur[-8:].sum() < p_clean[-8:].sum()


def test_overshoot_metric_flags_invented_high_freq():
    rng = np.random.default_rng(2)
    target = rng.random((24, 24, 24)).astype(np.float32)
    # a volume with EXTRA high-freq energy (white noise added) should overshoot
    inflated = target + 0.5 * rng.standard_normal(target.shape).astype(np.float32)
    m = overshoot_metric(inflated, target)
    assert m["overshoot_ratio"] > 1.1  # detector fires on invented HF


def test_overshoot_metric_low_for_matching_volume():
    rng = np.random.default_rng(3)
    target = rng.random((24, 24, 24)).astype(np.float32)
    m = overshoot_metric(target.copy(), target)
    assert m["overshoot_ratio"] < 1.01
    assert m["mean_log_gap"] < 1e-6
