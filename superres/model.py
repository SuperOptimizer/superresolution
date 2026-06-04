"""3-D residual U-Net for x1 restoration.

Design choices (from the project's design rationale):
  - SHALLOW (few downsampling levels). Deep pooling throws away exactly the high
    frequencies we want to recover; restoration is local-to-medium-range, not
    semantic. Default 3 levels.
  - RESIDUAL PREDICTION: out = in + net(in). The net only models the correction,
    so low frequencies pass through untouched and it can be small/shallow.
  - GroupNorm (not BatchNorm): batch-size-robust on CPU (batch 1-2) and H100 alike.
  - ~8-12M params at default width. Small model is a regularizer here -- it limits
    capacity to memorize/hallucinate given few source volumes.
  - NO adversarial/perceptual components. Trained with Charbonnier (see losses.py).

NAFNet-3D is the documented upgrade path; not implemented in v1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = 1
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    """Conv-GN-act x2 with a residual skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm1 = _gn(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = _gn(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int):
        super().__init__()
        self.pool = nn.Conv3d(in_ch, out_ch, 2, stride=2)   # learned downsample
        self.body = nn.Sequential(*[ResBlock(out_ch) for _ in range(blocks)])

    def forward(self, x):
        return self.body(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, blocks: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.reduce = nn.Conv3d(out_ch + skip_ch, out_ch, 1)
        self.body = nn.Sequential(*[ResBlock(out_ch) for _ in range(blocks)])

    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd-sized (defensive; patches are usually even)
        if x.shape[2:] != skip.shape[2:]:
            diff = [s - x.shape[2 + i] for i, s in enumerate(skip.shape[2:])]
            x = nn.functional.pad(x, [p for d in reversed(diff) for p in (0, d)])
        x = torch.cat([x, skip], dim=1)
        return self.body(self.reduce(x))


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_width: int = 32,
        levels: int = 3,
        blocks_per_level: int = 2,
        residual: bool = True,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.grad_checkpoint = grad_checkpoint
        w = base_width
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, w, 3, padding=1), _gn(w), nn.SiLU(inplace=True),
            *[ResBlock(w) for _ in range(blocks_per_level)],
        )
        widths = [w * (2 ** i) for i in range(levels + 1)]  # e.g. [32,64,128,256]
        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1], blocks_per_level) for i in range(levels)]
        )
        self.ups = nn.ModuleList(
            [Up(widths[i + 1], widths[i], widths[i], blocks_per_level)
             for i in reversed(range(levels))]
        )
        self.head = nn.Conv3d(w, in_ch, 3, padding=1)
        # Zero-init the head so the net starts as identity (residual=True).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        inp = x
        ckpt = self.grad_checkpoint and self.training and x.requires_grad
        if ckpt:
            from torch.utils.checkpoint import checkpoint

            def run(mod, *a):
                return checkpoint(mod, *a, use_reentrant=False)
        else:
            def run(mod, *a):
                return mod(*a)

        h = run(self.stem, x)
        skips = [h]
        for d in self.downs:
            h = run(d, h)
            skips.append(h)
        h = skips[-1]
        for i, u in enumerate(self.ups):
            h = run(u, h, skips[-(i + 2)])
        delta = self.head(h)
        return inp + delta if self.residual else delta


def build_model(cfg: dict) -> ResUNet3D:
    return ResUNet3D(
        in_ch=int(cfg.get("in_ch", 1)),
        base_width=int(cfg.get("base_width", 32)),
        levels=int(cfg.get("levels", 3)),
        blocks_per_level=int(cfg.get("blocks_per_level", 2)),
        residual=bool(cfg.get("residual", True)),
        grad_checkpoint=bool(cfg.get("grad_checkpoint", False)),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
