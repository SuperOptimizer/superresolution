"""Spectrum-based validation.

The core validation idea: a correct restoration's radially-averaged power spectrum
should climb toward the clean target's spectrum and FLATTEN at grid Nyquist -- it
must not OVERSHOOT past the target, because energy above what the target contains
is invented (the network reading itself, not the scroll). With an L1 / no-adversarial
restorer it won't overshoot; this function quantifies it.
"""
from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB). Standard restoration fidelity metric.

    NOTE for this task: PSNR is dominated by low frequencies (which were never
    degraded), so it's INSENSITIVE to the high-freq band we actually recover. Use
    alongside hf_psnr / spectral gap, not alone."""
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(data_range * data_range / mse)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Global SSIM (single-window over the whole patch -- cheap, no sliding window).

    Captures luminance/contrast/structure agreement. CAVEAT: SSIM can REWARD
    invented-but-plausible texture, so it cannot distinguish real recovery from
    hallucination -- pair it with overshoot_ratio."""
    a = pred.astype(np.float64); b = target.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def hf_psnr(pred: np.ndarray, target: np.ndarray, cutoff: float = 0.25,
            data_range: float = 1.0) -> float:
    """PSNR computed ONLY on the high-frequency band (freq > cutoff cyc/voxel).

    This is the sharpest measure of real restoration GAIN: it strips the low-freq
    bulk that masks improvement in plain PSNR and reports fidelity exactly in the
    band we restore. High-pass both volumes via FFT, then PSNR on the residual."""
    def highpass(v):
        v = v.astype(np.float64)
        F = np.fft.fftn(v)
        coords = [np.fft.fftfreq(n) for n in v.shape]
        grids = np.meshgrid(*coords, indexing="ij")
        kr = np.sqrt(sum(g ** 2 for g in grids))
        F[kr < cutoff] = 0.0
        return np.fft.ifftn(F).real
    return psnr(highpass(pred), highpass(target), data_range=data_range)


def quality_metrics(pred: np.ndarray, target: np.ndarray, degraded: np.ndarray | None = None,
                    data_range: float = 1.0) -> dict:
    """Bundle of restoration metrics. If `degraded` (the input) is given, also
    reports the GAIN over doing nothing (restored vs degraded)."""
    out = {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
        "hf_psnr": hf_psnr(pred, target, data_range=data_range),
    }
    if degraded is not None:
        out["psnr_gain"] = out["psnr"] - psnr(degraded, target, data_range)
        out["hf_psnr_gain"] = out["hf_psnr"] - hf_psnr(degraded, target, data_range=data_range)
        out["ssim_gain"] = out["ssim"] - ssim(degraded, target, data_range)
    return out


def effective_resolution(vol: np.ndarray, nbins: int = 64,
                         noise_frac: float = 0.05) -> float:
    """Effective resolution as a fraction of grid Nyquist (0..1).

    The idea (the "how much real resolution is actually here" measure): a volume
    samples up to Nyquist (freq 0.5 cyc/voxel) but its SIGNAL only fills part of
    that band -- beyond f_eff the spectrum is just noise. f_eff/Nyquist is the
    effective resolution. A blurred 1000^3 volume whose signal dies at half Nyquist
    genuinely holds only ~500^3 of information.

    We find f_eff as the highest frequency where radial power still exceeds the
    noise floor (estimated as `noise_frac` of peak power, after subtracting the
    flat high-freq tail). Returns f_eff / 0.5 in [0,1].
    """
    freqs, power = radial_power_spectrum(vol, nbins)
    if power.max() <= 0:
        return 0.0
    # noise floor: median of the top-quartile (highest-freq) bins
    tail = power[int(0.75 * nbins):]
    floor = float(np.median(tail)) if tail.size else 0.0
    sig = power - floor
    peak = sig.max()
    if peak <= 0:
        return 0.0
    thresh = noise_frac * peak
    above = np.where(sig > thresh)[0]
    if above.size == 0:
        return 0.0
    f_eff = freqs[above[-1]]          # highest freq still above noise
    return float(f_eff / 0.5)         # fraction of Nyquist


def resolution_gain(restored: np.ndarray, degraded: np.ndarray,
                    nbins: int = 64) -> dict:
    """The "Nx super-resolution" number: effective-resolution ratio after vs before.

    eff_before = degraded input's effective resolution (frac of Nyquist)
    eff_after  = restored output's effective resolution
    factor     = eff_after / eff_before  (e.g. 1.6 == "1.6x effective resolution")

    Bounded by Nyquist (eff<=1), so it CANNOT report runaway gain -- a built-in
    honesty cap. To convert to physical microns: phys_eff = voxel_um / eff.
    """
    eff_before = effective_resolution(degraded, nbins)
    eff_after = effective_resolution(restored, nbins)
    factor = (eff_after / eff_before) if eff_before > 1e-6 else 0.0
    return {"eff_before": eff_before, "eff_after": eff_after, "factor": factor}


def radial_power_spectrum(vol: np.ndarray, nbins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 3-D power spectrum.

    Returns (freqs, power) where freqs are in cycles/voxel on [0, 0.5] (0.5 = grid
    Nyquist) and power is the mean |FFT|^2 in each radial frequency shell.
    """
    v = vol.astype(np.float64)
    v = v - v.mean()
    F = np.fft.fftn(v)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)

    coords = [np.fft.fftshift(np.fft.fftfreq(n)) for n in v.shape]
    grids = np.meshgrid(*coords, indexing="ij")
    kr = np.sqrt(sum(g ** 2 for g in grids))  # radial freq magnitude, up to ~0.866

    kmax = 0.5  # only report up to grid Nyquist
    bins = np.linspace(0.0, kmax, nbins + 1)
    idx = np.digitize(kr.ravel(), bins) - 1
    power = np.zeros(nbins)
    counts = np.zeros(nbins)
    flatP = P.ravel()
    valid = (idx >= 0) & (idx < nbins)
    np.add.at(power, idx[valid], flatP[valid])
    np.add.at(counts, idx[valid], 1.0)
    counts[counts == 0] = 1.0
    power /= counts
    freqs = 0.5 * (bins[:-1] + bins[1:])
    return freqs, power


def overshoot_metric(
    restored: np.ndarray, target: np.ndarray, nbins: int = 64
) -> dict:
    """Compare restored vs target spectra.

    Returns dict with:
      freqs, restored_power, target_power,
      overshoot_ratio: max over frequency shells of restored/target power
                       (>1 + tolerance means the net is inventing high-freq energy),
      mean_log_gap:    mean |log10(restored/target)| (lower == closer match).
    """
    fr, pr = radial_power_spectrum(restored, nbins)
    _, pt = radial_power_spectrum(target, nbins)
    eps = 1e-12
    ratio = pr / (pt + eps)
    log_gap = np.abs(np.log10((pr + eps) / (pt + eps)))
    return {
        "freqs": fr,
        "restored_power": pr,
        "target_power": pt,
        "overshoot_ratio": float(np.max(ratio)),
        "mean_log_gap": float(np.mean(log_gap)),
    }


def save_spectrum_report(metrics: dict, path: str) -> None:
    """Write a CSV (always) and a PNG plot (if matplotlib is available)."""
    import csv

    fr = metrics["freqs"]
    with open(path + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_cyc_per_voxel", "restored_power", "target_power"])
        for i in range(len(fr)):
            w.writerow([fr[i], metrics["restored_power"][i], metrics["target_power"][i]])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.semilogy(fr, metrics["target_power"], label="target (clean)")
        plt.semilogy(fr, metrics["restored_power"], label="restored")
        plt.axvline(0.5, ls="--", c="k", lw=0.7, label="grid Nyquist")
        plt.xlabel("frequency (cycles/voxel)")
        plt.ylabel("power")
        plt.legend()
        plt.title(
            f"overshoot={metrics['overshoot_ratio']:.2f}  gap={metrics['mean_log_gap']:.3f}"
        )
        plt.savefig(path + ".png", dpi=110, bbox_inches="tight")
        plt.close()
    except Exception:
        pass  # CSV is enough; plotting is best-effort
