"""Config-driven training loop.

Charbonnier (air-masked) + data-consistency, AdamW, GroupNorm model (so CPU batch
works), bf16 autocast guarded to CUDA. `--smoke` runs a few steps on synthetic data
to prove the loop end-to-end on CPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import PatchDataset, SyntheticPatchDataset
from .degradation import DegradationRanges, RandomDegradation
from .model import build_model, count_params
from .losses import charbonnier, air_mask, data_consistency_loss


def build_dataset(cfg: dict, smoke: bool, length: int, return_params: bool = True):
    d = cfg["data"]
    deg = RandomDegradation(DegradationRanges.from_config(cfg["degradation"]))
    common = dict(
        degradation=deg,
        low_pct=d.get("norm_low_pct", 0.5),
        high_pct=d.get("norm_high_pct", 99.5),
        augment=d.get("augment", True),
        inplane_only=d.get("augment_inplane_only", False),
        length=length,
        seed=cfg.get("seed", 0),
        return_params=return_params,
    )
    source = "synthetic" if smoke else d.get("source", "synthetic")
    if source == "synthetic":
        return SyntheticPatchDataset(
            volume_shape=tuple(d.get("synthetic_shape", [96, 96, 96])),
            patch=tuple(d["patch"]),
            **common,
        )
    # real volume (S3 or local)
    from .s3zarr import open_s3, open_local

    if d.get("local_path"):
        z = open_local(d["local_path"], level=d.get("level", 0))
    else:
        z = open_s3(d["bucket"], d["scroll"], d["zarr"], level=d.get("level", 0))
    return PatchDataset(
        z,
        patch=tuple(d["patch"]),
        occupancy_min=d.get("occupancy_min", 0.5),
        **common,
    )


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    params = [b[2] for b in batch]
    # gather per-sample sigmas for the data-consistency term
    merged = {
        "sigma_z": [p["sigma_z"] for p in params],
        "sigma_xy": [p["sigma_xy"] for p in params],
    }
    return xs, ys, merged


def train(config_path: str, smoke: bool = False, max_steps: int | None = None):
    cfg = load_config(config_path)
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    tcfg = cfg["train"]
    lcfg = cfg["loss"]
    steps = max_steps if max_steps is not None else int(tcfg["steps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(tcfg.get("amp_bf16", False)) and device == "cuda"

    ds = build_dataset(cfg, smoke, length=steps * int(tcfg["batch_size"]))
    loader = DataLoader(
        ds,
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg.get("num_workers", 0)),
        collate_fn=_collate,
        drop_last=True,
    )

    model = build_model(cfg["model"]).to(device)
    print(f"model params: {count_params(model):,}  device: {device}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )

    ckpt_dir = Path(tcfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dc_w = float(lcfg.get("data_consistency_weight", 0.0))
    eps = float(lcfg.get("charbonnier_eps", 1e-3))
    mask_air = bool(lcfg.get("mask_air", True))

    model.train()
    step = 0
    last_loss = None
    for x, y, params in loader:
        if step >= steps:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
        with ctx:
            pred = model(x)
            mask = air_mask(y) if mask_air else None
            loss = charbonnier(pred, y, eps=eps, mask=mask)
            if dc_w > 0:
                loss = loss + dc_w * data_consistency_loss(pred, x, params, eps=eps)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step % max(1, int(tcfg.get("val_every", 1000))) == 0:
            print(f"step {step:6d}  loss {last_loss:.5f}")
        if (step + 1) % int(tcfg.get("ckpt_every", 2000)) == 0:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "step": step},
                ckpt_dir / "latest.pt",
            )
        step += 1

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt_dir / "latest.pt")
    print(f"done. final loss {last_loss}. checkpoint -> {ckpt_dir/'latest.pt'}")
    return last_loss


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
