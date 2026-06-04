"""Synthetic degradation operator: clean (native grid) -> realistic degraded input.

This is the load-bearing piece of the whole project. We never expand the grid
(this is the x1 / restoration case): every operation keeps the array shape fixed.
The model learns to invert this operator, so the operator MUST resemble the real
acquisition or the model inverts a fiction and fails at inference.

Pipeline (all in normalized float, [0,1]-ish):
    intensity jitter -> anisotropic Gaussian PSF blur -> sub-voxel shift
    -> additive (optionally colored) noise -> u8 quantization round-trip

Calibration philosophy: center every parameter on the MEASURED value (see
`measure_psf_from_edge`) and randomize with a modest spread. Too narrow overfits
one PSF; too wide (Real-ESRGAN-style) wastes capacity on artifacts you never see.

Deliberately NO adversarial / perceptual term anywhere -- conservative by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift


# ----------------------------------------------------------------------------
# Individual operators (each shape-preserving)
# ----------------------------------------------------------------------------
def anisotropic_blur(vol: np.ndarray, sigma_z: float, sigma_xy: float) -> np.ndarray:
    """Separable Gaussian PSF with independent through-plane / in-plane widths.

    `vol` is (z, y, x). CT PSF is almost never isotropic; z is usually wider.
    """
    if sigma_z <= 0 and sigma_xy <= 0:
        return vol
    return gaussian_filter(vol, sigma=(sigma_z, sigma_xy, sigma_xy), mode="reflect")


def add_noise(
    vol: np.ndarray,
    sigma: float,
    color: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive Gaussian noise, optionally spatially correlated ("colored").

    Real CT noise is correlated by the reconstruction, not white. `color` is the
    std-dev of a Gaussian smoothing applied to a white field; 0 == white noise.
    The field is re-scaled after smoothing so the requested `sigma` is preserved.
    """
    if sigma <= 0:
        return vol
    field = rng.standard_normal(vol.shape).astype(np.float32)
    if color > 0:
        field = gaussian_filter(field, sigma=color, mode="reflect")
        std = float(field.std())
        if std > 1e-8:
            field /= std  # restore unit variance lost to smoothing
    return vol + sigma * field


def subvoxel_shift(vol: np.ndarray, dz: float, dy: float, dx: float) -> np.ndarray:
    """Sub-voxel translation so the model isn't tied to grid phase."""
    if dz == 0 and dy == 0 and dx == 0:
        return vol
    return nd_shift(vol, shift=(dz, dy, dx), order=1, mode="reflect")


def quantize_u8(vol: np.ndarray) -> np.ndarray:
    """Round-trip through u8 to model the precision loss of u8 storage.

    Assumes `vol` is in normalized [0,1] units. Clips, scales to 0..255, rounds,
    scales back. This eats exactly the near-noise-floor precision downstream ink
    cues live on -- which is why training should ideally keep higher bit depth,
    but inference data is u8, so the model must be robust to this.
    """
    q = np.clip(vol, 0.0, 1.0) * 255.0
    q = np.round(q) / 255.0
    return q.astype(np.float32)


# ----------------------------------------------------------------------------
# Randomized operator
# ----------------------------------------------------------------------------
@dataclass
class DegradationRanges:
    """(low, high) ranges, sampled uniformly per patch. Defaults are placeholders;
    set from a measured PSF/MTF for your data."""

    sigma_xy: tuple[float, float] = (0.6, 1.3)
    sigma_z: tuple[float, float] = (0.9, 1.8)
    noise_sigma: tuple[float, float] = (0.005, 0.03)
    noise_color: tuple[float, float] = (0.0, 1.2)
    shift: tuple[float, float] = (-0.5, 0.5)
    intensity_gain: tuple[float, float] = (0.9, 1.1)
    intensity_bias: tuple[float, float] = (-0.03, 0.03)
    u8_quantize: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "DegradationRanges":
        def pair(key, default):
            v = cfg.get(key, default)
            return (float(v[0]), float(v[1]))

        return cls(
            sigma_xy=pair("sigma_xy", (0.6, 1.3)),
            sigma_z=pair("sigma_z", (0.9, 1.8)),
            noise_sigma=pair("noise_sigma", (0.005, 0.03)),
            noise_color=pair("noise_color", (0.0, 1.2)),
            shift=pair("shift", (-0.5, 0.5)),
            intensity_gain=pair("intensity_gain", (0.9, 1.1)),
            intensity_bias=pair("intensity_bias", (-0.03, 0.03)),
            u8_quantize=bool(cfg.get("u8_quantize", True)),
        )


class RandomDegradation:
    """Samples a fresh degradation per call. `params` (returned) records the draw,
    so a deterministic forward operator can be reconstructed for data-consistency."""

    def __init__(self, ranges: DegradationRanges):
        self.r = ranges

    def sample_params(self, rng: np.random.Generator) -> dict:
        def u(lo_hi):
            return float(rng.uniform(lo_hi[0], lo_hi[1]))

        return {
            "sigma_z": u(self.r.sigma_z),
            "sigma_xy": u(self.r.sigma_xy),
            "noise_sigma": u(self.r.noise_sigma),
            "noise_color": u(self.r.noise_color),
            "dz": u(self.r.shift),
            "dy": u(self.r.shift),
            "dx": u(self.r.shift),
            "gain": u(self.r.intensity_gain),
            "bias": u(self.r.intensity_bias),
            "u8_quantize": self.r.u8_quantize,
        }

    def apply(
        self, clean: np.ndarray, rng: np.random.Generator, params: dict | None = None,
        voxel_um: float | None = None, ref_um: float = 2.4,
    ) -> tuple[np.ndarray, dict]:
        """Return (degraded, params). `clean` is float (z,y,x), nominally [0,1].

        If `voxel_um` is given, the PSF sigmas are scaled by (ref_um/voxel_um) so
        the blur is physically consistent across resolution tiers (a fixed-micron
        PSF spans more voxels at finer resolution). Matches gpu_degrade.GPUDegradation.
        """
        if params is None:
            params = self.sample_params(rng)
            if voxel_um is not None:
                s = ref_um / float(voxel_um)
                params = {**params, "sigma_z": params["sigma_z"] * s,
                          "sigma_xy": params["sigma_xy"] * s}
        v = clean.astype(np.float32)
        # intensity jitter (per-scan normalization differences)
        v = v * params["gain"] + params["bias"]
        # PSF blur
        v = anisotropic_blur(v, params["sigma_z"], params["sigma_xy"])
        # sub-voxel grid-phase shift
        v = subvoxel_shift(v, params["dz"], params["dy"], params["dx"])
        # noise
        v = add_noise(v, params["noise_sigma"], params["noise_color"], rng)
        # u8 precision round-trip
        if params["u8_quantize"]:
            v = quantize_u8(v)
        return v.astype(np.float32), params


def forward_blur_only(vol: np.ndarray, params: dict) -> np.ndarray:
    """Deterministic forward operator for data-consistency: just the PSF blur from
    a recorded `params` draw (noise/quantization are not invertible constraints).
    Used to check that a restored volume, re-blurred, reproduces the measurement.
    """
    return anisotropic_blur(vol, params["sigma_z"], params["sigma_xy"])


# ----------------------------------------------------------------------------
# Calibration tool
# ----------------------------------------------------------------------------
def measure_psf_from_edge(edge_profile: np.ndarray) -> dict:
    """Estimate PSF width from an edge-spread function (ESF).

    Pull a 1-D intensity profile across a sharp papyrus/air boundary (an oversampled
    ESF), pass it here. We differentiate to the line-spread function (LSF), fit a
    Gaussian std-dev, and report it (in voxels) plus the MTF via FFT. Do this
    separately along z and within-plane to get the anisotropy, then center the
    degradation `sigma_*` ranges on these measured values.

    This is a calibration helper, not part of the training hot path.

    Returns dict with keys: sigma (Gaussian std in voxels), lsf, mtf, freqs.
    """
    esf = np.asarray(edge_profile, dtype=np.float64)
    esf = esf - esf.min()
    if esf.max() > 0:
        esf = esf / esf.max()
    lsf = np.gradient(esf)
    lsf = np.abs(lsf)
    s = lsf.sum()
    if s <= 0:
        raise ValueError("Degenerate edge profile: LSF integrates to zero.")
    lsf = lsf / s
    # Gaussian std from the second central moment of the LSF.
    x = np.arange(lsf.size, dtype=np.float64)
    mean = float((x * lsf).sum())
    var = float(((x - mean) ** 2 * lsf).sum())
    sigma = float(np.sqrt(max(var, 0.0)))
    mtf = np.abs(np.fft.rfft(lsf))
    if mtf[0] > 0:
        mtf = mtf / mtf[0]
    freqs = np.fft.rfftfreq(lsf.size)  # cycles/voxel; 0.5 == grid Nyquist
    return {"sigma": sigma, "lsf": lsf, "mtf": mtf, "freqs": freqs}
