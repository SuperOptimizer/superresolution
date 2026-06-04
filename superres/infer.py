"""Tiled streaming inference over a large volume.

Key pieces from the design:
  - overlapping windows with COSINE-FEATHER blend (halo >= PSF support) so tile
    seams -- the single most common way a working model produces ugly output --
    disappear.
  - occupancy skip: all-zero tiles are passed through untouched (no model call).
    For the real run, skip the READ too (not just compute); here we operate on an
    in-memory array, so we skip compute.
  - bf16 autocast only when CUDA is present; CPU runs fp32.

`infer_volume` works on a numpy array (already normalized to ~[0,1]); for the real
multi-TB case, wrap it to stream tiles from a Zarr3D and write to an output zarr,
reusing the same window/blend logic per tile.
"""
from __future__ import annotations

import numpy as np
import torch


def _cosine_window(shape: tuple[int, int, int], overlap: int) -> np.ndarray:
    """Separable cosine taper ramping ~0->1 over `overlap` voxels at each edge.

    Half-voxel-offset cosine so two ramps offset by `overlap` are complementary
    and sum to 1 in the interior (partition of unity). A small positive floor keeps
    the weight strictly > 0 everywhere, so the acc/wacc normalization in
    `infer_volume` is exact even where a voxel is covered by a single tile (e.g. the
    outermost edges of the whole volume), avoiding divide-by-zero seams.
    """
    floor = 1e-3

    def ramp(n: int) -> np.ndarray:
        w = np.ones(n, dtype=np.float64)
        o = min(overlap, n // 2)
        if o > 0:
            # sample at half-voxel centers: positions 0.5/o .. (o-0.5)/o
            t = (np.arange(o) + 0.5) / o
            edge = (1 - np.cos(np.pi * t)) / 2  # complementary cosine, 0<edge<1
            w[:o] = edge
            w[-o:] = edge[::-1]
        return floor + (1.0 - floor) * w

    wz, wy, wx = (ramp(s) for s in shape)
    return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)


def _starts(total: int, window: int, overlap: int) -> list[int]:
    step = max(1, window - overlap)
    if total <= window:
        return [0]
    pts = list(range(0, total - window + 1, step))
    if pts[-1] != total - window:
        pts.append(total - window)
    return pts


@torch.no_grad()
def infer_volume(
    model: torch.nn.Module,
    vol: np.ndarray,
    window=(256, 256, 256),
    overlap: int = 32,
    occupancy_skip: bool = True,
    device: str | None = None,
    amp_bf16: bool = True,
) -> np.ndarray:
    """Restore a normalized float volume (z,y,x) with overlap-blend tiling.

    Returns a float array the same shape as `vol`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    use_amp = amp_bf16 and device == "cuda"

    win = tuple(min(window[i], vol.shape[i]) for i in range(3))
    acc = np.zeros(vol.shape, dtype=np.float32)
    wacc = np.zeros(vol.shape, dtype=np.float32)
    blend = _cosine_window(win, overlap)

    zs = _starts(vol.shape[0], win[0], overlap)
    ys = _starts(vol.shape[1], win[1], overlap)
    xs = _starts(vol.shape[2], win[2], overlap)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                tile = vol[z0 : z0 + win[0], y0 : y0 + win[1], x0 : x0 + win[2]]
                if occupancy_skip and not tile.any():
                    out = tile  # all-air: pass through, no model call
                else:
                    t = torch.from_numpy(np.ascontiguousarray(tile))[None, None].float().to(device)
                    if use_amp:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out = model(t)
                    else:
                        out = model(t)
                    out = out[0, 0].float().cpu().numpy()
                acc[z0 : z0 + win[0], y0 : y0 + win[1], x0 : x0 + win[2]] += out * blend
                wacc[z0 : z0 + win[0], y0 : y0 + win[1], x0 : x0 + win[2]] += blend

    wacc[wacc == 0] = 1.0
    return acc / wacc
