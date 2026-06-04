"""Spectrum-based validation.

The core validation idea: a correct restoration's radially-averaged power spectrum
should climb toward the clean target's spectrum and FLATTEN at grid Nyquist -- it
must not OVERSHOOT past the target, because energy above what the target contains
is invented (the network reading itself, not the scroll). With an L1 / no-adversarial
restorer it won't overshoot; this function quantifies it.
"""
from __future__ import annotations

import numpy as np


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
