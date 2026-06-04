"""GPU-side degradation: same operator as superres.degradation, in torch.

When the CPU DataLoader can't keep a fast GPU fed (the L4 sat at 0% util because
scipy's per-patch Gaussian blur cost ~70 ms), move the degradation onto the GPU.
The loader then only reads + normalizes clean patches (cheap); the train step
applies PSF blur + colored noise + sub-voxel shift + u8 quantization on-device.

This mirrors superres.degradation.RandomDegradation but batched and on tensors.
Per-sample params are sampled on CPU (cheap scalars) and returned so the
data-consistency loss can reuse the exact sigmas.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .degradation import DegradationRanges


def _kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur1axis(vol: torch.Tensor, axis: int, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return vol
    k = _kernel1d(sigma, vol.device, vol.dtype)
    shape = [1, 1, 1, 1, 1]
    shape[axis] = k.numel()
    kernel = k.view(shape)
    slot = {2: 4, 3: 2, 4: 0}[axis]
    pad = [0, 0, 0, 0, 0, 0]
    r = k.numel() // 2
    pad[slot] = pad[slot + 1] = r
    out = F.pad(vol, pad, mode="reflect")
    return F.conv3d(out, kernel)


class GPUDegradation:
    """Batched degradation on (N,1,Z,Y,X) clean tensors in [0,1]."""

    def __init__(self, ranges: DegradationRanges):
        self.r = ranges

    def _u(self, lo_hi, rng):
        return float(rng.uniform(lo_hi[0], lo_hi[1]))

    def apply(self, clean: torch.Tensor, rng: np.random.Generator):
        """Return (degraded, params). Operates per-sample so each gets its own draw.

        Because separable conv with a per-sample kernel isn't a single batched op,
        we loop over the (small) batch -- still vastly cheaper than CPU scipy and
        fully on-device.
        """
        n = clean.shape[0]
        outs = []
        sig_z, sig_xy = [], []
        for i in range(n):
            v = clean[i : i + 1]
            sz = self._u(self.r.sigma_z, rng)
            sxy = self._u(self.r.sigma_xy, rng)
            gain = self._u(self.r.intensity_gain, rng)
            bias = self._u(self.r.intensity_bias, rng)
            nz = self._u(self.r.noise_sigma, rng)
            ncol = self._u(self.r.noise_color, rng)
            dz, dy, dx = (self._u(self.r.shift, rng) for _ in range(3))

            v = v * gain + bias
            # anisotropic PSF blur
            v = _blur1axis(v, 2, sz)
            v = _blur1axis(v, 3, sxy)
            v = _blur1axis(v, 4, sxy)
            # sub-voxel shift via grid_sample (trilinear)
            if dz or dy or dx:
                v = _subvoxel_shift(v, dz, dy, dx)
            # noise (optionally colored)
            if nz > 0:
                field = torch.randn_like(v)
                if ncol > 0:
                    field = _blur1axis(field, 2, ncol)
                    field = _blur1axis(field, 3, ncol)
                    field = _blur1axis(field, 4, ncol)
                    std = field.std()
                    if std > 1e-8:
                        field = field / std
                v = v + nz * field
            # u8 quantization round-trip
            if self.r.u8_quantize:
                v = (torch.clamp(v, 0.0, 1.0) * 255.0).round() / 255.0
            outs.append(v)
            sig_z.append(sz)
            sig_xy.append(sxy)
        degraded = torch.cat(outs, dim=0)
        return degraded, {"sigma_z": sig_z, "sigma_xy": sig_xy}


def _subvoxel_shift(v: torch.Tensor, dz: float, dy: float, dx: float) -> torch.Tensor:
    n, c, Z, Y, X = v.shape
    # base identity grid in normalized [-1,1] coords
    zs = torch.linspace(-1, 1, Z, device=v.device, dtype=v.dtype)
    ys = torch.linspace(-1, 1, Y, device=v.device, dtype=v.dtype)
    xs = torch.linspace(-1, 1, X, device=v.device, dtype=v.dtype)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    # shift in voxels -> normalized offset (2/size per voxel)
    gx = gx + (2.0 * dx / max(X - 1, 1))
    gy = gy + (2.0 * dy / max(Y - 1, 1))
    gz = gz + (2.0 * dz / max(Z - 1, 1))
    grid = torch.stack([gx, gy, gz], dim=-1)[None]  # (1,Z,Y,X,3), order x,y,z
    return F.grid_sample(v, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=True)
