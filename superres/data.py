"""Datasets: lazily sample clean 3-D patches, degrade them on the fly, return
(degraded_input, clean_target) pairs on the SAME grid (x1 restoration).

Two datasets:
  - PatchDataset: samples from a Zarr3D (S3 or local), with papyrus importance
    sampling (reject air) and robust percentile normalization.
  - SyntheticPatchDataset: procedural fiber-like volumes so tests and the CPU
    smoke run need no network.

Augmentation note (anisotropy guard): because we degrade AFTER the geometric
augmentation, the full 48-element cubic symmetry group is safe even with an
anisotropic PSF -- the PSF is applied in the patch's post-rotation frame, so z<->xy
swaps don't teach isotropic blur. Set `inplane_only=True` to restrict to the 8
in-plane symmetries if you ever pre-degrade instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationRanges, RandomDegradation


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------
def robust_normalize(
    vol: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5
) -> np.ndarray:
    """Percentile-clip then scale to ~[0,1]. Never min/max (outlier-fragile).

    Computed over nonzero voxels when possible so the air background doesn't
    dominate the percentiles.
    """
    v = vol.astype(np.float32)
    sample = v[v > 0] if np.any(v > 0) else v
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    if hi <= lo:
        # Degenerate percentile span (e.g. near-constant patch). Fall back to a
        # 0..max scaling so constant nonzero signal maps to a constant nonzero
        # value rather than collapsing to all-zero (which would read as "air").
        vmax = float(v.max())
        if vmax <= 0:
            return np.zeros_like(v)
        return np.clip(v / vmax, 0.0, 1.0).astype(np.float32)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Cubic-symmetry augmentation (octahedral group, 48 elements)
# ----------------------------------------------------------------------------
def random_symmetry(vol: np.ndarray, rng: np.random.Generator, inplane_only: bool) -> np.ndarray:
    """Apply a random axis-permutation + flips. Label-preserving, no resampling."""
    v = vol
    if inplane_only:
        # only swap the in-plane (y, x) axes; keep z fixed
        if rng.random() < 0.5:
            v = np.swapaxes(v, 1, 2)
        flips = [ax for ax in (1, 2) if rng.random() < 0.5]
    else:
        # random permutation of the 3 axes (24 rotations + reflections via flips)
        perm = rng.permutation(3)
        v = np.transpose(v, perm)
        flips = [ax for ax in (0, 1, 2) if rng.random() < 0.5]
    if flips:
        v = np.flip(v, axis=tuple(flips))
    return np.ascontiguousarray(v)


# ----------------------------------------------------------------------------
# Base dataset (shared degrade/normalize/augment logic)
# ----------------------------------------------------------------------------
class _BasePatchDataset(Dataset):
    def __init__(
        self,
        patch: tuple[int, int, int],
        degradation: RandomDegradation,
        low_pct: float,
        high_pct: float,
        augment: bool,
        inplane_only: bool,
        length: int,
        seed: int,
        return_params: bool = False,
    ):
        self.patch = tuple(patch)
        self.degradation = degradation
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.augment = augment
        self.inplane_only = inplane_only
        self.length = length
        self.seed = seed
        self.return_params = return_params

    def __len__(self) -> int:
        return self.length

    def _sample_clean(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _rng(self, idx: int) -> np.random.Generator:
        # Per-(epoch-agnostic)-item RNG; varies with idx so workers diverge.
        return np.random.default_rng(self.seed * 1_000_003 + idx)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        clean = self._sample_clean(rng)                # raw patch (any dtype)
        clean = robust_normalize(clean, self.low_pct, self.high_pct)
        if self.augment:
            clean = random_symmetry(clean, rng, self.inplane_only)
        degraded, params = self.degradation.apply(clean, rng)
        x = torch.from_numpy(degraded[None]).float()   # (1, z, y, x)
        y = torch.from_numpy(clean[None]).float()
        if self.return_params:
            return x, y, params
        return x, y


# ----------------------------------------------------------------------------
# Real-volume dataset
# ----------------------------------------------------------------------------
class PatchDataset(_BasePatchDataset):
    """Sample patches from a Zarr3D with papyrus importance sampling."""

    def __init__(
        self,
        zarr,                              # Zarr3D
        patch=(128, 128, 128),
        degradation: Optional[RandomDegradation] = None,
        occupancy_min: float = 0.5,
        low_pct: float = 0.5,
        high_pct: float = 99.5,
        augment: bool = True,
        inplane_only: bool = False,
        length: int = 100_000,
        seed: int = 0,
        max_reject: int = 50,
        return_params: bool = False,
    ):
        if degradation is None:
            degradation = RandomDegradation(DegradationRanges())
        super().__init__(patch, degradation, low_pct, high_pct, augment,
                         inplane_only, length, seed, return_params)
        self.z = zarr
        self.occupancy_min = occupancy_min
        self.max_reject = max_reject

    def _sample_clean(self, rng: np.random.Generator) -> np.ndarray:
        pz, py, px = self.patch
        sz, sy, sx = self.z.shape
        for _ in range(self.max_reject):
            z0 = int(rng.integers(0, sz - pz + 1))
            y0 = int(rng.integers(0, sy - py + 1))
            x0 = int(rng.integers(0, sx - px + 1))
            patch = self.z.read_region(z0, y0, x0, pz, py, px)
            occ = float((patch > 0).mean())
            if occ >= self.occupancy_min:
                return patch
        # give up after max_reject -- return last patch regardless (rare)
        return patch


# ----------------------------------------------------------------------------
# Synthetic dataset (no network) -- procedural fiber-like texture
# ----------------------------------------------------------------------------
def make_synthetic_volume(shape, rng: np.random.Generator) -> np.ndarray:
    """Procedural papyrus-ish volume: oriented fibrous high-frequency texture.

    Not physically accurate -- just rich broadband structure so a restoration net
    has something with real high-frequency content to recover. Returns u8-scaled.
    """
    from scipy.ndimage import gaussian_filter

    z, y, x = shape
    # base low-frequency tissue
    base = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=3.0)
    # fiber bundles: sum of a few anisotropically-smoothed noise fields
    fibers = np.zeros(shape, dtype=np.float32)
    for _ in range(4):
        n = rng.standard_normal(shape).astype(np.float32)
        sig = rng.uniform(0.3, 0.8, size=3)
        sig[int(rng.integers(0, 3))] *= 8.0  # elongate along one axis
        fibers += gaussian_filter(n, sigma=tuple(sig))
    vol = 0.6 * base + 1.4 * fibers
    vol -= vol.min()
    vol /= max(vol.max(), 1e-6)
    return (vol * 255.0).astype(np.uint8)


class SyntheticPatchDataset(_BasePatchDataset):
    """Crops patches from one procedurally-generated volume. No network/disk."""

    def __init__(
        self,
        volume_shape=(96, 96, 96),
        patch=(32, 32, 32),
        degradation: Optional[RandomDegradation] = None,
        low_pct: float = 0.5,
        high_pct: float = 99.5,
        augment: bool = True,
        inplane_only: bool = False,
        length: int = 64,
        seed: int = 0,
        return_params: bool = False,
    ):
        if degradation is None:
            degradation = RandomDegradation(DegradationRanges())
        super().__init__(patch, degradation, low_pct, high_pct, augment,
                         inplane_only, length, seed, return_params)
        self.volume = make_synthetic_volume(volume_shape, np.random.default_rng(seed))

    def _sample_clean(self, rng: np.random.Generator) -> np.ndarray:
        pz, py, px = self.patch
        sz, sy, sx = self.volume.shape
        z0 = int(rng.integers(0, sz - pz + 1))
        y0 = int(rng.integers(0, sy - py + 1))
        x0 = int(rng.integers(0, sx - px + 1))
        return self.volume[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]


class CachedVolumeDataset(_BasePatchDataset):
    """Sample patches from a local cached cube (.npy on fast disk) produced by
    scripts/cache_roi.py. Memory-maps the file so workers share it without copying.
    Importance-samples toward occupied (papyrus) regions, same as PatchDataset, but
    with no per-patch network cost.
    """

    def __init__(
        self,
        npy_path: str,
        patch=(128, 128, 128),
        degradation: Optional[RandomDegradation] = None,
        occupancy_min: float = 0.5,
        low_pct: float = 0.5,
        high_pct: float = 99.5,
        augment: bool = True,
        inplane_only: bool = False,
        length: int = 100_000,
        seed: int = 0,
        max_reject: int = 50,
        return_params: bool = False,
    ):
        if degradation is None:
            degradation = RandomDegradation(DegradationRanges())
        super().__init__(patch, degradation, low_pct, high_pct, augment,
                         inplane_only, length, seed, return_params)
        # mmap so each DataLoader worker maps the same pages (no per-worker copy)
        self.volume = np.load(npy_path, mmap_mode="r")
        self.occupancy_min = occupancy_min
        self.max_reject = max_reject

    def _sample_clean(self, rng: np.random.Generator) -> np.ndarray:
        pz, py, px = self.patch
        sz, sy, sx = self.volume.shape
        patch = None
        for _ in range(self.max_reject):
            z0 = int(rng.integers(0, sz - pz + 1))
            y0 = int(rng.integers(0, sy - py + 1))
            x0 = int(rng.integers(0, sx - px + 1))
            patch = np.asarray(self.volume[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px])
            if float((patch > 0).mean()) >= self.occupancy_min:
                return patch
        return patch
