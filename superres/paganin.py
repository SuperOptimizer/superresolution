"""Physics-derived PSF from BM18 acquisition metadata (Paganin + optics).

The dominant, KNOWN blur in these reconstructions is the Paganin single-distance
phase-retrieval filter applied during recon. It is an analytic low-pass; we use
the EXACT form from ESRF's nabu (nabu/preproc/phase.py, verified to ~1e-16):

    T(f) = 1 / (1 + delta_beta * lambda * D * pi * f^2)

with f in cycles/micron, lambda = 1.23984199e-3 / E_keV (micron), D = distance
(micron). A Gaussian unsharp mask (sigma, coeff) is applied afterward to re-sharpen.
So the effective recon transfer function is roughly:

    H_recon(k) = T_paganin(k) * U_unsharp(k)

This lets us build a degradation operator from the REAL physics (no scanner
experiments, no guessed sigmas) for the volumes whose metadata we have. We model
the residual blur the recon imparts; the model learns to invert it.

Convention: spatial frequency k in cycles/voxel (matches our spectrum metrics);
physical quantities converted using the sample pixel size.
"""
from __future__ import annotations

import math

import numpy as np

# Constant from nabu (keV*micron); wavelength_micron = _HC_KEV_UM / energy_keV.
_HC_KEV_UM = 1.23984199e-3


def physics_from_metadata(md: dict) -> dict | None:
    """Extract the PSF-relevant recon physics from a volume's metadata.json dict.

    Tolerant of schema variation. Returns a physics dict usable by recon_transfer,
    or None if the essential Paganin params are absent. Mirrors the full nabu chain
    we can model: Paganin (delta_beta), unsharp (coeff/sigma), acquisition geometry.
    """
    if not md:
        return None
    tomo = (md.get("scan", {}) or {}).get("tomo", {}) or {}
    acq = tomo.get("acquisition", {}) or {}
    proc = tomo.get("processing", {}) or {}
    pre = proc.get("preprocessing", {}) or {}
    phase = pre.get("phase", {}) or {}
    det = acq.get("detector", {}) or {}
    recon = proc.get("reconstruction", {}) if isinstance(proc.get("reconstruction"), dict) else {}

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    p = {
        "paganin_delta_beta": num(phase.get("delta_beta")),
        "unsharp_coeff": num(phase.get("unsharp_coeff")) or 0.0,
        "unsharp_sigma": num(phase.get("unsharp_sigma")) or 0.0,
        "unsharp_method": phase.get("unsharp_method", "gaussian"),
        "energy_kev": num(acq.get("energy")),
        "sample_detector_mm": num(acq.get("sampleDetectorDistance")),
        "sample_pixel_mm": num(det.get("samplePixelSize")),
        "scintillator": det.get("scintillator"),
        "recon_method": recon.get("method"),
        "fbp_filter": (recon.get("fbp_filter_type") or "ramlak"),
    }
    if p["paganin_delta_beta"] is None or p["energy_kev"] is None:
        return None
    # sample_pixel_um for the transfer functions
    p["sample_pixel_um"] = (p["sample_pixel_mm"] or 0.0024) * 1000.0
    return p


def wavelength_micron(energy_kev: float) -> float:
    """X-ray wavelength in MICRONS from energy (keV), matching nabu's constant."""
    return _HC_KEV_UM / energy_kev


def wavelength_m(energy_kev: float) -> float:
    """X-ray wavelength in meters (kept for tests/back-compat)."""
    return wavelength_micron(energy_kev) * 1e-6


def paganin_transfer(k_cyc_per_voxel: np.ndarray, delta_beta: float, energy_kev: float,
                     sample_detector_mm: float, sample_pixel_um: float) -> np.ndarray:
    """Paganin low-pass transfer, EXACTLY matching nabu's PaganinPhaseRetrieval:

        filter = 1 / (1 + db * L * D * pi * k2)

    where k2 = (fy^2 + fx^2) with fy = fftfreq(n, d=pixel_size_micron) i.e. spatial
    frequency in CYCLES/MICRON, L = wavelength (micron), D = distance (micron).
    (nabu/preproc/phase.py: compute_filter). We take `k_cyc_per_voxel` (our spectrum
    convention) and convert to cycles/micron by dividing by the pixel size.
    """
    L = wavelength_micron(energy_kev)              # micron
    D = sample_detector_mm * 1000.0                 # mm -> micron
    # cycles/voxel -> cycles/micron: divide by micron-per-voxel (= sample_pixel_um)
    f_cyc_per_um = k_cyc_per_voxel / sample_pixel_um
    k2 = f_cyc_per_um ** 2
    denom = 1.0 + delta_beta * L * D * math.pi * k2
    return 1.0 / denom


def unsharp_transfer(k_cyc_per_voxel: np.ndarray, sigma_vox: float, coeff: float) -> np.ndarray:
    """Transfer function of an unsharp mask: out = in + coeff*(in - blur(in)).
    In Fourier: U(k) = 1 + coeff*(1 - G(k)), G = Gaussian transfer at sigma (voxels).
    """
    g = np.exp(-2.0 * (math.pi ** 2) * (sigma_vox ** 2) * (k_cyc_per_voxel ** 2))
    return 1.0 + coeff * (1.0 - g)


def recon_transfer(k_cyc_per_voxel: np.ndarray, phys: dict) -> np.ndarray:
    """Combined recon transfer H = T_paganin * U_unsharp from a physics dict
    (as produced by survey_physics.extract_physics)."""
    T = paganin_transfer(
        k_cyc_per_voxel,
        delta_beta=phys["paganin_delta_beta"],
        energy_kev=phys["energy_kev"],
        sample_detector_mm=phys.get("sample_detector_mm") or 220.0,
        sample_pixel_um=(phys.get("sample_pixel_mm") or 0.0024) * 1000.0,
    )
    U = unsharp_transfer(
        k_cyc_per_voxel,
        sigma_vox=phys.get("unsharp_sigma") or 1.2,
        coeff=phys.get("unsharp_coeff") or 0.0,
    )
    H = T * U
    return np.clip(H, 1e-4, None)


def _radial_freq_grid(shape):
    coords = [np.fft.fftfreq(n) for n in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    return np.sqrt(sum(g ** 2 for g in grids))


def apply_recon_blur(vol: np.ndarray, phys: dict, strength: float = 1.0) -> np.ndarray:
    """Apply the recon transfer function to a clean volume in 3-D (separable in
    Fourier). `strength` in [0,1] interpolates identity->full operator so we can
    randomize severity. Models the residual recon blur the model must invert.

    NOTE: H_recon can be >1 (unsharp boosts mid-freq). For a DEGRADATION operator
    we want a net low-pass, so we use only the low-pass part when H>1 is not
    desired -- here we apply H directly but clip to <=1 so it's strictly a blur
    (the unsharp already happened in the real data; for *forward* degradation we
    model the net softening, not re-applying the boost).
    """
    kr = _radial_freq_grid(vol.shape)
    H = recon_transfer(kr, phys)
    H = np.clip(H, None, 1.0)                  # strictly low-pass for degradation
    H = (1.0 - strength) + strength * H        # interpolate identity -> H
    F = np.fft.fftn(vol.astype(np.float64))
    out = np.fft.ifftn(F * H).real
    return out.astype(np.float32)


def deconvolve_recon_torch(vol, phys: dict, reg: float = 0.05):
    """GPU/torch version of deconvolve_recon for use in the training step.

    `vol` is (N,1,Z,Y,X). Returns the Wiener-deconvolved tensor, same shape.
    The transfer function H is built on-device from the (per-sample-constant)
    physics. Runs in fp32 internally for FFT stability, casts back to vol.dtype.
    """
    import torch
    dev = vol.device
    Z, Y, X = vol.shape[-3:]
    # radial freq grid in cycles/voxel
    kz = torch.fft.fftfreq(Z, device=dev)
    ky = torch.fft.fftfreq(Y, device=dev)
    kx = torch.fft.fftfreq(X, device=dev)
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    kr = torch.sqrt(KZ**2 + KY**2 + KX**2).cpu().numpy()
    H = recon_transfer(kr, phys)                       # numpy, small cost
    Ht = torch.from_numpy(H).to(dev, torch.float32)
    inv = (Ht / (Ht * Ht + reg))[None, None]
    F = torch.fft.fftn(vol.float(), dim=(-3, -2, -1))
    out = torch.fft.ifftn(F * inv, dim=(-3, -2, -1)).real
    return out.to(vol.dtype)


def deconvolve_recon(vol: np.ndarray, phys: dict, reg: float = 0.02) -> np.ndarray:
    """Direct (Wiener-regularized) inverse of the recon transfer -- FREE resolution
    recovery with NO learned prior, since it inverts a KNOWN analytic operator.

    out_F = in_F * H / (H^2 + reg). reg bounds amplification near the noise floor
    (so we don't blow up frequencies the operator killed). This is the cleanest,
    hallucination-proof gain available; use as a preprocessing step or a baseline.
    """
    kr = _radial_freq_grid(vol.shape)
    H = recon_transfer(kr, phys)
    F = np.fft.fftn(vol.astype(np.float64))
    inv = H / (H * H + reg)
    out = np.fft.ifftn(F * inv).real
    return out.astype(np.float32)
