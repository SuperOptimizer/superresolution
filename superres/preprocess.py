"""Standalone physics-based volume preprocessor (NO ML).

Sharpens a scroll volume by inverting the KNOWN reconstruction operators (Paganin
phase-retrieval low-pass + unsharp), derived exactly from the volume's metadata
(matches ESRF nabu source). This is a pure-physics deblur usable for ANY downstream
task -- segmentation, ink detection, human reading -- independent of any ML model.

Locality: the recon transfer is a convolution, so each output voxel depends only on
a finite neighborhood (~kernel extent, tens of voxels), NOT the whole volume. So we
process in HALO-TILED blocks: read tile + margin, deconvolve, keep the inner region.
This scales to arbitrarily large volumes and is embarrassingly parallel.
"""
from __future__ import annotations

import numpy as np

from .paganin import recon_transfer, _radial_freq_grid


def kernel_extent_voxels(phys: dict, thresh: float = 0.01) -> int:
    """Estimate the real-space half-extent (voxels) of the recon kernel, so we can
    size the tile halo. We find where the inverse filter's impulse response decays
    below `thresh` of its peak along one axis."""
    n = 256
    k = np.fft.fftfreq(n)
    H = recon_transfer(np.abs(k), phys)
    H = np.clip(H, 1e-4, None)
    inv = H / (H * H + 0.05)               # the deconvolution filter
    psf = np.abs(np.fft.ifft(inv).real)
    psf = np.fft.fftshift(psf)
    peak = psf.max()
    above = np.where(psf > thresh * peak)[0]
    if above.size == 0:
        return 16
    half = int(max(abs(above[0] - n // 2), abs(above[-1] - n // 2)))
    return int(np.clip(half, 8, 96))


def deconvolve_block(block: np.ndarray, phys: dict, reg: float = 0.05) -> np.ndarray:
    """Wiener-deconvolve one block with the recon transfer (numpy/FFT)."""
    kr = _radial_freq_grid(block.shape)
    H = recon_transfer(kr, phys)
    F = np.fft.fftn(block.astype(np.float64))
    inv = H / (H * H + reg)
    return np.fft.ifftn(F * inv).real.astype(np.float32)


def sharpen_region(fetch, z, y, x, dz, dy, dx, shape, phys, reg: float = 0.05,
                   halo: int | None = None):
    """Viewer-friendly: sharpen exactly the region [z:z+dz, y:y+dy, x:x+dx] for
    display (e.g. vc3d post-processing). Fetches the region + a halo internally so
    the result has no edge artifacts, deconvolves, returns ONLY the requested region.

    `fetch(z,y,x,dz,dy,dx) -> ndarray` reads from the volume (zarr/array). `shape`
    is the full volume (Z,Y,X) for halo clamping. `reg` is the sharpening-strength
    dial (lower = sharper+noisier; expose as a UI slider). The halo (~15 voxels for
    BM18 Paganin) makes this cheap -- you only read a little beyond what's shown.

    Cost: one FFT over (region + 2*halo). A 512x512 slab is ~150ms on CPU, <10ms on
    GPU -- interactive. Physics comes from the volume's metadata once.
    """
    if halo is None:
        halo = kernel_extent_voxels(phys)
    Z, Y, X = shape
    rz0, ry0, rx0 = max(0, z - halo), max(0, y - halo), max(0, x - halo)
    rz1, ry1, rx1 = min(Z, z + dz + halo), min(Y, y + dy + halo), min(X, x + dx + halo)
    blk = fetch(rz0, ry0, rx0, rz1 - rz0, ry1 - ry0, rx1 - rx0)
    dec = deconvolve_block(blk.astype(np.float32), phys, reg=reg)
    iz, iy, ix = z - rz0, y - ry0, x - rx0
    return dec[iz:iz + dz, iy:iy + dy, ix:ix + dx]


def preprocess_volume(
    read_region,                 # callable(z,y,x,dz,dy,dx)->ndarray
    write_region,                # callable(z,y,x, ndarray)->None
    shape,                       # (Z,Y,X) of the volume
    phys: dict,
    tile: int = 256,
    halo: int | None = None,
    reg: float = 0.05,
    normalize=None,              # optional callable to normalize a tile before deconv
    occupancy_skip: bool = True,
    progress=lambda *_: None,
):
    """Halo-tiled physics deconvolution over a large volume.

    For each interior tile, read tile+halo, deconvolve, write back only the inner
    tile (halo discarded -> no seam artifacts since edge voxels had real context).
    """
    Z, Y, X = shape
    if halo is None:
        halo = kernel_extent_voxels(phys)
    nz = -(-Z // tile); ny = -(-Y // tile); nx = -(-X // tile)
    total = nz * ny * nx
    done = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                z0, y0, x0 = iz * tile, iy * tile, ix * tile
                tz, ty, tx = min(tile, Z - z0), min(tile, Y - y0), min(tile, X - x0)
                # read with halo, clamped to bounds
                rz0, ry0, rx0 = max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)
                rz1 = min(Z, z0 + tz + halo); ry1 = min(Y, y0 + ty + halo); rx1 = min(X, x0 + tx + halo)
                blk = read_region(rz0, ry0, rx0, rz1 - rz0, ry1 - ry0, rx1 - rx0)
                done += 1
                progress(done, total)
                if occupancy_skip and not np.any(blk):
                    continue
                fblk = normalize(blk) if normalize else blk.astype(np.float32)
                dec = deconvolve_block(fblk, phys, reg=reg)
                # inner region offset within the (haloed) block
                iz0, iy0, ix0 = z0 - rz0, y0 - ry0, x0 - rx0
                inner = dec[iz0:iz0 + tz, iy0:iy0 + ty, ix0:ix0 + tx]
                write_region(z0, y0, x0, inner)
    return total
