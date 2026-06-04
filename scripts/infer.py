#!/usr/bin/env python3
"""Run tiled restoration inference.

Smoke (synthetic volume in/out, no network):
    python scripts/infer.py --config configs/smoke.yaml --ckpt checkpoints_smoke/latest.pt --synthetic

Real volume from a local .npy (already normalized float32 z,y,x):
    python scripts/infer.py --config configs/default.yaml --ckpt checkpoints/latest.pt \
        --input vol.npy --output restored.npy

Spectrum report vs a clean reference (optional):
    add --target clean.npy to write a spectrum CSV/PNG.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.config import load_config
from superres.model import build_model
from superres.infer import infer_volume
from superres.data import make_synthetic_volume, robust_normalize
from superres.degradation import RandomDegradation, DegradationRanges
from superres.spectrum import overshoot_metric, save_spectrum_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", help="path to .npy float volume (z,y,x)")
    ap.add_argument("--output", default="restored.npy")
    ap.add_argument("--target", help="optional clean .npy for spectrum report")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate a degraded synthetic volume as input (no network)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["infer"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg["model"])
    model.load_state_dict(ck["model"])

    target = None
    if args.synthetic:
        rng = np.random.default_rng(123)
        shape = tuple(cfg["data"].get("synthetic_shape", [64, 64, 64]))
        clean = robust_normalize(make_synthetic_volume(shape, rng))
        deg = RandomDegradation(DegradationRanges.from_config(cfg["degradation"]))
        vol, _ = deg.apply(clean, rng)
        target = clean
    else:
        vol = np.load(args.input).astype(np.float32)
        if args.target:
            target = np.load(args.target).astype(np.float32)

    restored = infer_volume(
        model, vol,
        window=tuple(icfg["window"]),
        overlap=int(icfg["overlap"]),
        occupancy_skip=bool(icfg.get("occupancy_skip", True)),
        amp_bf16=bool(icfg.get("amp_bf16", False)),
    )
    np.save(args.output, restored)
    print(f"wrote {args.output}  shape {restored.shape}")

    if target is not None:
        m = overshoot_metric(restored, target)
        save_spectrum_report(m, args.output + ".spectrum")
        print(f"spectrum: overshoot_ratio={m['overshoot_ratio']:.3f} "
              f"(>~1.1 = inventing HF), mean_log_gap={m['mean_log_gap']:.3f}")
        # input vs target gap for reference
        if args.synthetic:
            mi = overshoot_metric(vol, target)
            print(f"  (degraded input mean_log_gap={mi['mean_log_gap']:.3f}; "
                  f"restoration should reduce this)")


if __name__ == "__main__":
    main()
