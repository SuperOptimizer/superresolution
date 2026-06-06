#!/usr/bin/env python3
"""Validate the WHOLE preprocess chain stage-by-stage on the full quality basket, across
several 128^3 regions (on-disk, cached calibration). Shows what each stage adds/costs so we
can pick the best chain config and catch any stage that hurts. Math in C; Python glues."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fastmetrics as fm
from superres import s3zarr, fysics_pipeline as fp
from scipy.ndimage import binary_erosion

CUBE = "/home/forrest/paris4_2um_cube"
REGIONS = [(384,384,384),(640,384,384),(200,200,200),(384,640,640),(512,512,512)]

# stage configs (cumulative): each adds one stage
CONFIGS = [
    ("raw",            dict(do_deconv=False, do_denoise=False)),
    ("deconv",         dict(do_deconv=True,  do_denoise=False)),
    ("deconv+denoise", dict(do_deconv=True,  do_denoise=True)),
    ("+air-zero",      dict(do_deconv=True,  do_denoise=True, _air=True)),
]


def score(f, out):
    pap = binary_erosion(f > (130/255), iterations=2)
    vd = fm.valley_depth(out)
    return dict(
        HSJ   = fm.hsj(out)[0],
        sharp = fm.edge_sharpness(out)/(fm.edge_sharpness(f)+1e-9),
        dynrng= fm.dynamic_range(out),
        noise = fm.flat_noise(out)/(fm.flat_noise(f)+1e-9),
        paptex= fm.hp_texture(out, pap)/(fm.hp_texture(f, pap)+1e-9) if pap.sum()>500 else 0,
        contr = fm.midband_ratio(out, f),
        keep  = (out > 0.001).mean()*100,
        corelost = (out[pap] <= 0.001).mean()*100 if pap.sum()>500 else 0,
    )


def main():
    z = s3zarr.open_local(CUBE, 0)
    cal = fm.fast_cal(CUBE)
    print(f"cal: reg={cal.deconv_reg} eps={cal.guided_eps:.4g} air_cut~dark_mode\n")
    # average each config's metrics over all regions
    agg = {name: [] for name, _ in CONFIGS}
    for c in REGIONS:
        f = z.read_region(*c, 128,128,128, workers=8).astype(np.float32)/255
        if (f>0.02).mean() < 0.3: continue
        u8 = np.clip(f*255,0,255).astype(np.uint8)
        for name, cfg in CONFIGS:
            cal.do_air_zero = cfg.get("_air", False)
            out = fp.process_tile(u8, cal, do_deconv=cfg["do_deconv"], do_denoise=cfg["do_denoise"]).astype(np.float32)/255
            agg[name].append(score(f, out))
            cal.do_air_zero = False
    print(f"{'config':>16} {'HSJ':>5} {'sharp':>6} {'dynrng':>7} {'noise':>6} {'paptex':>7} {'contr':>6} {'keep%':>6} {'corelost':>8}")
    for name, _ in CONFIGS:
        rows = agg[name]; m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{name:>16} {m['HSJ']:>5.1f} {m['sharp']:>6.2f} {m['dynrng']:>7.2f} {m['noise']:>6.2f} "
              f"{m['paptex']:>7.2f} {m['contr']:>6.2f} {m['keep']:>5.0f}% {m['corelost']:>7.2f}%", flush=True)
    print("\\nread DOWN each column to see what each added stage does to that metric.")
    print("want: contr up, noise down, paptex~1, corelost~0, dynrng up, HSJ up")


if __name__ == "__main__":
    main()
