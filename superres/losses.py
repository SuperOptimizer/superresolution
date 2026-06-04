"""Losses: Charbonnier (smooth L1), optional air-masking, and data-consistency.

Data-consistency: re-degrade the prediction through the KNOWN forward PSF and
compare to the degraded input. This bounds the network to the operator's null
space -- it can interpolate within the unresolved band but cannot contradict the
measurement. Implemented as a differentiable separable Gaussian blur in torch so
it backprops.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3,
                mask: torch.Tensor | None = None) -> torch.Tensor:
    """sqrt((pred-target)^2 + eps^2), optionally masked (air excluded)."""
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom
    return loss.mean()


def air_mask(target: torch.Tensor, thresh: float = 0.02) -> torch.Tensor:
    """1 where there's papyrus signal, 0 in background air. Shaped like target."""
    return (target > thresh).float()


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def differentiable_blur(vol: torch.Tensor, sigma_z: float, sigma_xy: float) -> torch.Tensor:
    """Separable anisotropic Gaussian blur on (N,1,Z,Y,X). Mirrors degradation PSF."""
    out = vol
    for axis, sigma in zip((2, 3, 4), (sigma_z, sigma_xy, sigma_xy)):
        if sigma <= 0:
            continue
        k = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
        shape = [1, 1, 1, 1, 1]
        shape[axis] = k.numel()
        kernel = k.view(shape)
        pad = [0, 0, 0, 0, 0, 0]
        # F.pad order is (x_l,x_r, y_l,y_r, z_l,z_r); map axis->pad slot
        slot = {2: 4, 3: 2, 4: 0}[axis]
        r = k.numel() // 2
        pad[slot] = pad[slot + 1] = r
        out = F.pad(out, pad, mode="reflect")
        out = F.conv3d(out, kernel)
    return out


def data_consistency_loss(pred: torch.Tensor, degraded_input: torch.Tensor,
                          params: dict, eps: float = 1e-3) -> torch.Tensor:
    """Re-blur the prediction with the forward PSF; compare to the degraded input.

    `params` carries the per-sample sigma draws. For a batch we use the mean sigma
    (cheap, adequate -- the constraint only needs to be approximately right).
    """
    sz = float(_mean(params, "sigma_z"))
    sxy = float(_mean(params, "sigma_xy"))
    reblurred = differentiable_blur(pred, sz, sxy)
    return charbonnier(reblurred, degraded_input, eps=eps)


def _mean(params: dict, key: str) -> float:
    v = params[key]
    if torch.is_tensor(v):
        return v.float().mean().item()
    try:
        return float(sum(v) / len(v))
    except TypeError:
        return float(v)
