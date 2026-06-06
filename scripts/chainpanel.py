#!/usr/bin/env python3
"""Fast whole-preprocess-chain metric panel on 128^3 cubes. Calibrate ONCE, process N regions
through the full process_tile chain, score the C-backed metric basket. The harness we iterate on
as we integrate stages into the pipeline. Usage: python chainpanel.py [n_regions]"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fastmetrics as fm
from superres import s3zarr, fysics_pipeline as fp
from scipy.ndimage import binary_erosion

CUBE = "/home/forrest/paris4_2um_cube"
REGIONS = [(384,384,384),(640,384,384),(200,200,200),(384,640,640),(512,512,512),(300,700,400)]


def panel(f, out):
    """score one (raw f, processed out) pair across the full quality basket. f,out in [0,1]."""
    pap = binary_erosion(f > (130/255), iterations=2)
    vd = fm.valley_depth(out)
    return dict(
        depth = vd[3] if vd else 0.0,
        hsj   = fm.hsj(out)[0],
        frc   = fm.fsc_res(out),
        sharp = fm.edge_sharpness(out)/(fm.edge_sharpness(f)+1e-9),
        dynrng= fm.dynamic_range(out),
        noise = fm.flat_noise(out)/(fm.flat_noise(f)+1e-9),
        paptex= fm.hp_texture(out, pap)/(fm.hp_texture(f, pap)+1e-9) if pap.sum()>500 else 0,
        contr = fm.midband_ratio(out, f),
        keep  = (out > 0.001).mean()*100,
        corelost = (out[pap] <= 0.001).mean()*100 if pap.sum()>500 else 0,
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(REGIONS)
    z = s3zarr.open_local(CUBE, 0)
    read = lambda a,b,c,d,e,g: z.read_region(a,b,c,d,e,g,workers=12)
    t0 = time.time()
    cal = fm.fast_cal(CUBE)   # calibrate on the current 128^3 cube (cached -> instant after 1st)
    print(f"calibrated in {time.time()-t0:.0f}s: deconv reg={cal.deconv_reg} eps={cal.guided_eps:.4g} air={cal.air_thresh}")
    print(f"{'region':>15} {'depth':>6} {'HSJ':>5} {'FRCo':>5} {'FRC':>5} {'paptex':>6} {'contr':>6} {'keep%':>6} {'corelost':>8} {'t':>5}")
    for c in REGIONS[:n]:
        f = read(*c,128,128,128).astype(np.float32)/255
        if (f>0.02).mean() < 0.3: print(f"{str(c):>15}  air-skip"); continue
        u8 = np.clip(f*255,0,255).astype(np.uint8)
        t = time.time(); out = fp.process_tile(u8, cal).astype(np.float32)/255; dt = time.time()-t
        p = panel(f, out)
        print(f"{str(c):>15} {p['depth']:>6.3f} {p['hsj']:>5.1f} {p['frc_o']:>5.2f} {p['frc']:>5.2f} "
              f"{p['paptex']:>6.2f} {p['contr']:>6.2f} {p['keep']:>5.0f}% {p['corelost']:>7.2f}% {dt:>4.1f}s", flush=True)
    print("\\nGOOD = FRC~FRCo (resolution kept), paptex~1, corelost~0, depth up, HSJ up")


if __name__ == "__main__":
    main()
