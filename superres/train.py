"""Config-driven training loop.

Charbonnier (air-masked) + data-consistency, AdamW, GroupNorm model (so CPU batch
works), bf16 autocast on CUDA. Supports three data sources (cfg["data"]["source"]):
  - "synthetic": procedural volumes, no network (also forced by --smoke)
  - "cached":    a local .npy cube produced by scripts/cache_roi.py (fast NVMe)
  - "s3"/local:  per-patch reads from a Zarr3D (slow; prefer "cached" for real runs)

Real-run features: throughput logging, gradient clipping, periodic spectrum
validation on a held-out patch, checkpoint + report sync to S3, and resume.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

# Reduce CUDA fragmentation; must be set before the CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import (
    PatchDataset,
    SyntheticPatchDataset,
    CachedVolumeDataset,
    robust_normalize,
)
from .degradation import DegradationRanges, RandomDegradation
from .model import build_model, count_params
from .losses import charbonnier, air_mask, data_consistency_loss
from .spectrum import overshoot_metric


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
    if source == "cached":
        return CachedVolumeDataset(
            d["cache_path"],
            patch=tuple(d["patch"]),
            occupancy_min=d.get("occupancy_min", 0.5),
            **common,
        )
    if source == "multicube":
        from .data import MultiCubeDataset
        return MultiCubeDataset(
            d["manifest"],
            patch=tuple(d["patch"]),
            occupancy_min=d.get("occupancy_min", 0.5),
            **common,
        )
    # real volume (S3 or local) -- per-patch reads
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
    merged = {
        "sigma_z": [p["sigma_z"] for p in params],
        "sigma_xy": [p["sigma_xy"] for p in params],
    }
    return xs, ys, merged


def _s3_sync_blocking(local_dir: Path, s3_prefix: str):
    try:
        subprocess.run(
            ["aws", "s3", "sync", str(local_dir), s3_prefix, "--only-show-errors"],
            check=False, timeout=600,
        )
        print(f"  [s3] synced {local_dir} -> {s3_prefix}", flush=True)
    except Exception as e:  # never let a sync failure kill training
        print(f"  [s3] sync failed: {e}", flush=True)


def _s3_sync(local_dir: Path, s3_prefix: str, background: bool = True):
    """Upload a directory to S3. Backgrounded by default so the (network-bound)
    upload never blocks the training loop -- the checkpoint is already on local
    disk, so we don't need to wait for it to reach S3 to keep training."""
    if not s3_prefix:
        return
    if background:
        t = threading.Thread(target=_s3_sync_blocking, args=(local_dir, s3_prefix),
                             daemon=True)
        t.start()
    else:
        _s3_sync_blocking(local_dir, s3_prefix)


def _sample_val_clean(cfg, rng):
    """Return (clean_patch_normalized, voxel_um or None) for validation.

    Averages over several patches at the call site; this returns one. For the
    multicube source we draw from a random cube (so validation spans scales).
    """
    d = cfg["data"]
    patch = tuple(d["patch"])
    pz, py, px = patch
    src = d.get("source", "synthetic")
    occ_min = d.get("occupancy_min", 0.5)

    def crop_occupied(vol):
        sz, sy, sx = vol.shape
        c = None
        for _ in range(40):
            z0 = int(rng.integers(0, sz - pz + 1)); y0 = int(rng.integers(0, sy - py + 1)); x0 = int(rng.integers(0, sx - px + 1))
            c = np.asarray(vol[z0:z0+pz, y0:y0+py, x0:x0+px])
            if (c > 0).mean() >= occ_min:
                return c
        return c

    if src == "multicube" and Path(d.get("manifest", "")).exists():
        import json
        entries = json.loads(Path(d["manifest"]).read_text())
        e = entries[int(rng.integers(0, len(entries)))]
        vol = np.load(e["path"], mmap_mode="r")
        clean = robust_normalize(crop_occupied(vol), d.get("norm_low_pct", 0.5), d.get("norm_high_pct", 99.5))
        return clean, float(e["voxel_um"])
    if src == "cached" and Path(d.get("cache_path", "")).exists():
        vol = np.load(d["cache_path"], mmap_mode="r")
        clean = robust_normalize(crop_occupied(vol), d.get("norm_low_pct", 0.5), d.get("norm_high_pct", 99.5))
        return clean, None
    from .data import make_synthetic_volume
    return robust_normalize(make_synthetic_volume(patch, rng)), None


@torch.no_grad()
def _spectrum_check(model, cfg, device, use_amp, n_patches: int = 4) -> dict:
    """Degrade -> restore several held-out patches; report AVERAGED overshoot/gap.

    Averaging over patches makes the metric stable enough to trust over a long
    run (single-patch spectrum is ~+-0.04 noisy). Scale-conditioned: passes each
    patch's voxel_um to the model.
    """
    rng = np.random.default_rng(98765)
    scale_cond = bool(cfg["model"].get("scale_cond", False))
    eval_model = getattr(model, "_orig_mod", model)  # uncompiled (avoid recompile)
    deg = RandomDegradation(DegradationRanges.from_config(cfg["degradation"]))
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
    from .spectrum import quality_metrics, resolution_gain
    in_gaps, out_gaps, overshoots = [], [], []
    psnrs, ssims, hf_gains = [], [], []
    eff_befores, eff_afters, factors = [], [], []
    for _ in range(n_patches):
        clean, vum = _sample_val_clean(cfg, rng)
        degraded, _ = deg.apply(clean, rng)
        x = torch.from_numpy(degraded[None, None]).float().to(device)
        vt = (torch.tensor([float(vum)], device=device)
              if (scale_cond and vum is not None) else None)
        with ctx:
            pred = eval_model(x, voxel_um=vt) if scale_cond else eval_model(x)
        restored = pred[0, 0].float().cpu().numpy()
        in_gaps.append(overshoot_metric(degraded, clean)["mean_log_gap"])
        m_out = overshoot_metric(restored, clean)
        out_gaps.append(m_out["mean_log_gap"])
        overshoots.append(m_out["overshoot_ratio"])
        qm = quality_metrics(restored, clean, degraded=degraded)
        psnrs.append(qm["psnr"]); ssims.append(qm["ssim"]); hf_gains.append(qm["hf_psnr_gain"])
        rg = resolution_gain(restored, degraded)
        eff_befores.append(rg["eff_before"]); eff_afters.append(rg["eff_after"]); factors.append(rg["factor"])
    return {"in_gap": float(np.mean(in_gaps)), "out_gap": float(np.mean(out_gaps)),
            "overshoot": float(np.mean(overshoots)), "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)), "hf_psnr_gain": float(np.mean(hf_gains)),
            "eff_before": float(np.mean(eff_befores)), "eff_after": float(np.mean(eff_afters)),
            "res_factor": float(np.mean(factors))}


def train(config_path: str, smoke: bool = False, max_steps: int | None = None,
          resume: str | None = None):
    cfg = load_config(config_path)
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    tcfg = cfg["train"]
    lcfg = cfg["loss"]
    steps = max_steps if max_steps is not None else int(tcfg["steps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(tcfg.get("amp_bf16", device == "cuda")) and device == "cuda"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # GPU-side degradation: loader returns clean-only patches (cheap), the train
    # step applies PSF/noise/quant on-device. Avoids the CPU loader starving the GPU.
    gpu_degrade = bool(tcfg.get("gpu_degrade", device == "cuda")) and device == "cuda"
    ds = build_dataset(cfg, smoke, length=steps * int(tcfg["batch_size"]))
    gdeg = None
    if gpu_degrade:
        from .gpu_degrade import GPUDegradation
        ds.clean_only = True
        gdeg = GPUDegradation(DegradationRanges.from_config(cfg["degradation"]))
    nw = int(tcfg.get("num_workers", 0))
    # Force 'fork' so workers inherit the in-RAM cube via copy-on-write instead of
    # pickling it (Python 3.14 defaults to forkserver/spawn, which tries to pickle
    # the multi-GB array to each worker and fails / wastes RAM).
    mp_ctx = torch.multiprocessing.get_context("fork") if nw > 0 else None
    loader = DataLoader(
        ds,
        batch_size=int(tcfg["batch_size"]),
        num_workers=nw,
        collate_fn=(None if gpu_degrade else _collate),
        drop_last=True,
        pin_memory=(device == "cuda"),
        persistent_workers=bool(nw),
        prefetch_factor=(int(tcfg.get("prefetch_factor", 4)) if nw else None),
        multiprocessing_context=mp_ctx,
    )

    model = build_model(cfg["model"]).to(device)
    scale_cond = bool(cfg["model"].get("scale_cond", False))
    # NOTE: channels_last_3d measured ~2x SLOWER for these 3D convs on the L4
    # (Ada) -- it selects worse cuDNN kernels. Keep contiguous. Opt in per-config.
    chlast = bool(tcfg.get("channels_last", False)) and device == "cuda"
    if chlast:
        model = model.to(memory_format=torch.channels_last_3d)
    n_params = count_params(model)
    # torch.compile: fuse the many small 3D-conv kernels (better SM occupancy) and,
    # with mode="reduce-overhead", capture CUDA graphs to remove per-step dispatch
    # gaps -- both push us toward compute saturation. Gated by train.compile.
    compile_mode = tcfg.get("compile", None)
    if compile_mode and device == "cuda":
        # static shapes (patches are fixed-size) + fullgraph (no silent eager
        # fallback) let Inductor specialize hardest. Both default on when compiling.
        c_dynamic = bool(tcfg.get("compile_dynamic", False))
        c_fullgraph = bool(tcfg.get("compile_fullgraph", True))
        print(f"torch.compile(mode={compile_mode!r}, dynamic={c_dynamic}, "
              f"fullgraph={c_fullgraph}) -- first steps slow (autotune)")
        model = torch.compile(model, mode=compile_mode,
                              dynamic=c_dynamic, fullgraph=c_fullgraph)
    print(f"model params: {n_params:,}  device: {device}  bf16={use_amp}  "
          f"compile={compile_mode}  source={cfg['data'].get('source')}")
    # fused=True runs the AdamW update as a single fused CUDA kernel (less launch
    # overhead than the default for-loop / foreach paths). Free on CUDA.
    opt_kwargs = dict(lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    if device == "cuda" and bool(tcfg.get("fused_optimizer", True)):
        opt_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    # LR schedule: linear warmup then cosine decay to lr_min. A flat LR doesn't
    # settle over a long convergence run; cosine decay is what lets the loss
    # actually flatten rather than bounce. Enabled by train.lr_schedule="cosine".
    import math as _math
    sched_kind = tcfg.get("lr_schedule", "cosine")
    warmup = int(tcfg.get("warmup_steps", min(500, int(tcfg["steps"]) // 20)))
    lr_min_frac = float(tcfg.get("lr_min_frac", 0.05))
    total_steps = int(tcfg["steps"])

    def lr_factor(s):
        if sched_kind != "cosine":
            return 1.0
        if s < warmup:
            return (s + 1) / max(1, warmup)
        prog = (s - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, prog)
        return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + _math.cos(_math.pi * prog))

    start_step = 0
    if resume and Path(resume).exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {resume} at step {start_step}")

    ckpt_dir = Path(tcfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    s3_prefix = tcfg.get("s3_sync", "")
    dc_w = float(lcfg.get("data_consistency_weight", 0.0))
    eps = float(lcfg.get("charbonnier_eps", 1e-3))
    mask_air = bool(lcfg.get("mask_air", True))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = max(1, int(tcfg.get("log_every", tcfg.get("val_every", 50))))
    val_every = max(1, int(tcfg.get("val_every", 1000)))
    ckpt_every = int(tcfg.get("ckpt_every", 2000))

    def save(step, tag="latest", background=True):
        path = ckpt_dir / f"{tag}.pt"
        # Snapshot to CPU on the main thread (cheap, must read live GPU state),
        # then write to disk in a thread so the (I/O-bound) save doesn't stall
        # training. torch.compile wraps the model -- unwrap for a clean state_dict.
        m = getattr(model, "_orig_mod", model)
        snap = {"model": {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()},
                "opt": opt.state_dict(), "cfg": cfg, "step": step}
        if background:
            threading.Thread(target=torch.save, args=(snap, path), daemon=True).start()
        else:
            torch.save(snap, path)
        return path

    deg_rng = np.random.default_rng(cfg.get("seed", 0) + 777)
    model.train()
    step = start_step
    last_loss = None
    t_win = time.time()
    seen = 0
    for batch in loader:
        if step >= steps:
            break
        # apply LR schedule for this step
        lr_now = float(tcfg["lr"]) * lr_factor(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        vum = None
        if gpu_degrade:
            # clean-only loader: batch is either clean, or (clean, voxel_um)
            if isinstance(batch, (list, tuple)):
                y, vum = batch
                vum = vum.to(device, non_blocking=True)
            else:
                y = batch
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                x, params = gdeg.apply(y, deg_rng)            # degrade on GPU
        else:
            x, y, params = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        if chlast:
            x = x.to(memory_format=torch.channels_last_3d)
            y = y.to(memory_format=torch.channels_last_3d)
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
        with ctx:
            pred = model(x, voxel_um=vum) if scale_cond else model(x)
            mask = air_mask(y) if mask_air else None
            loss = charbonnier(pred, y, eps=eps, mask=mask)
            if dc_w > 0:
                loss = loss + dc_w * data_consistency_loss(pred, x, params, eps=eps)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        seen += 1

        # Only sync the loss to CPU on log steps -- doing it every step forces a
        # GPU->CPU stall that serializes the async pipeline.
        if step % log_every == 0:
            last_loss = float(loss.detach().cpu())
            dt = time.time() - t_win
            ips = seen / dt if dt > 0 else 0.0
            print(f"step {step:6d}  loss {last_loss:.5f}  {ips:.2f} it/s  "
                  f"({ips*int(tcfg['batch_size']):.1f} patches/s)", flush=True)
            t_win = time.time(); seen = 0

        if step > start_step and step % val_every == 0:
            model.eval()
            s = _spectrum_check(model, cfg, device, use_amp)
            model.train()
            print(f"  [val] RES GAIN={s['res_factor']:.2f}x "
                  f"(eff {s['eff_before']:.2f}->{s['eff_after']:.2f} Nyq)  "
                  f"overshoot={s['overshoot']:.2f}  "
                  f"PSNR={s['psnr']:.1f}dB +{s['hf_psnr_gain']:.1f}HF  "
                  f"SSIM={s['ssim']:.3f}  gap {s['in_gap']:.2f}->{s['out_gap']:.2f}",
                  flush=True)
            t_win = time.time(); seen = 0   # don't count the val pause against throughput

        if (step + 1) % ckpt_every == 0:
            p = save(step)                  # CPU snapshot + threaded write (non-blocking)
            print(f"  [ckpt] step {step} -> {p}", flush=True)
            _s3_sync(ckpt_dir, s3_prefix)   # backgrounded upload
            t_win = time.time(); seen = 0
        step += 1

    last_loss = float(loss.detach().cpu())
    save(step, background=False)            # final save must complete
    _s3_sync(ckpt_dir, s3_prefix, background=False)
    print(f"done. final loss {last_loss}. checkpoint -> {ckpt_dir/'latest.pt'}")
    return last_loss


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
