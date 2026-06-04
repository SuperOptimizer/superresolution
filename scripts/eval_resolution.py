#!/usr/bin/env python3
"""Evaluate a trained checkpoint's resolution-gain multiplier, per resolution tier.

Reports the "Nx effective resolution" number (res_factor) plus PSNR/SSIM/overshoot,
broken down by voxel-size tier so you can see where the model helps most.

    python scripts/eval_resolution.py --ckpt /mnt/data/checkpoints/scale_converge/latest.pt \
        --manifest /mnt/data/cubes/cubes.json --n 6
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.model import build_model
from superres.degradation import RandomDegradation, DegradationRanges
from superres.data import robust_normalize
from superres.spectrum import resolution_gain, quality_metrics, overshoot_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=6, help="patches per tier")
    ap.add_argument("--patch", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    m = build_model(cfg["model"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    scale_cond = bool(cfg["model"].get("scale_cond", False))
    deg = RandomDegradation(DegradationRanges.from_config(cfg["degradation"]))
    rng = np.random.default_rng(7)
    P = args.patch
    print(f"checkpoint step {ck['step']}  scale_cond={scale_cond}\n")

    entries = json.loads(Path(args.manifest).read_text())
    by_tier = defaultdict(list)
    for e in entries:
        by_tier[round(e["voxel_um"], 3)].append(e)

    print(f"{'tier(um)':>9} {'RES GAIN':>9} {'eff_b->eff_a':>14} {'overshoot':>10} "
          f"{'PSNR':>6} {'HFgain':>7} {'SSIM':>6}")
    all_f = []
    for vum in sorted(by_tier):
        facs, ovs, ps, hf, ss = [], [], [], [], []
        for e in by_tier[vum]:
            vol = np.load(e["path"], mmap_mode="r")
            sz, sy, sx = vol.shape
            for _ in range(args.n):
                z = int(rng.integers(0, sz - P)); y = int(rng.integers(0, sy - P)); x0 = int(rng.integers(0, sx - P))
                c = np.asarray(vol[z:z+P, y:y+P, x0:x0+P])
                if (c > 0).mean() < 0.5:
                    continue
                clean = robust_normalize(c)
                degraded, _ = deg.apply(clean, rng)
                xt = torch.from_numpy(degraded[None, None]).float().to(dev)
                vt = torch.tensor([vum], device=dev) if scale_cond else None
                with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else _null()):
                    pred = m(xt, voxel_um=vt) if scale_cond else m(xt)
                r = pred[0, 0].float().cpu().numpy()
                rg = resolution_gain(r, degraded)
                facs.append(rg["factor"]); ovs.append(overshoot_metric(r, clean)["overshoot_ratio"])
                qm = quality_metrics(r, clean, degraded=degraded)
                ps.append(qm["psnr"]); hf.append(qm["hf_psnr_gain"]); ss.append(qm["ssim"])
        if not facs:
            continue
        all_f.extend(facs)
        print(f"{vum:>9.3f} {np.mean(facs):>8.2f}x {'':>14} {np.mean(ovs):>10.2f} "
              f"{np.mean(ps):>6.1f} {np.mean(hf):>+7.2f} {np.mean(ss):>6.3f}")
    print(f"\nOVERALL mean resolution gain: {np.mean(all_f):.2f}x")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
