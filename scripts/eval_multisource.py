#!/usr/bin/env python3
"""Evaluate the optimized pipeline across a DIVERSE set of open-bucket volumes (resolutions,
energies, scrolls). For each: find an occupied 128^3 chunk near center, derive physics from
the bucket index metadata, calibrate on that chunk, run the full chain, score the quality
basket. Flags any failure or quality regression. Streams from S3 (one chunk/volume)."""
import os, sys, json, re, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
import scripts.fastmetrics as fm
from superres import s3zarr, fysics_pipeline as fp
from scipy.ndimage import binary_erosion

BUCKET = "vesuvius-challenge-open-data"
INDEX = "/tmp/vesuvius_index.json"


def pick_diverse(allvols, n_per_res=1):
    """one volume per (resolution) bucket, spread across scrolls -- a diverse coverage set."""
    by_res = {}
    for v in allvols:
        by_res.setdefault(v[3], []).append(v)
    picked = []
    seen_scrolls = set()
    for res in sorted(by_res):
        # prefer a scroll not yet covered
        cands = sorted(by_res[res], key=lambda v: v[0] in seen_scrolls)
        for v in cands[:n_per_res]:
            picked.append(v); seen_scrolls.add(v[0])
    return picked


def build_md(v, idx, z):
    """Build md_phys using the VOLUME'S OWN metadata.json (available per-volume) when present,
    falling back PER FIELD to the bucket-index window + filename-parsed res/energy + sensible
    defaults when metadata is missing/incomplete. Returns (md, source) where source notes what
    was used ('meta' = full volume metadata, 'partial' = some fields fell back, 'fallback' =
    no usable volume metadata). Never crashes on missing metadata."""
    scroll, vid, lid, res_fn, en_fn = v
    cm = idx[scroll]["volumes"][vid].get("creation", {}).get("metadata", {})  # index window
    res = float(res_fn or 2.4)
    dist_default = 11000.0 if res > 40 else (1200.0 if res > 5 else 220.0)
    # filename/index-derived fallback (always available)
    md = {"energy_kev": float(en_fn or 78), "pixel_um": res, "distance_mm": dist_default,
          "delta_beta": 1000.0,
          "window_f32_min": cm.get("target_window_f32_min"),
          "window_f32_max": cm.get("target_window_f32_max")}
    source = "fallback"
    # try the volume's own metadata.json (the authoritative source)
    try:
        vmd = fp.load_md_phys(None, backend=z.backend)
        used = []
        for k in ("energy_kev", "pixel_um", "distance_mm", "delta_beta",
                  "unsharp_sigma", "unsharp_coeff", "window_f32_min", "window_f32_max"):
            val = vmd.get(k)
            if val is not None:
                md[k] = val; used.append(k)
        # 'meta' if the load-bearing physics (delta_beta + distance) came from metadata
        if "delta_beta" in used and "distance_mm" in used:
            source = "meta"
        elif used:
            source = "partial"
    except Exception:
        pass   # metadata missing/unreadable -> keep the fallback md
    return md, source


def find_occupied(z, n=128, tries=9):
    """probe near center for an occupied n^3 chunk; return origin or None."""
    Z, Y, X = z.shape
    cz, cy, cx = Z//2, Y//2, X//2
    offs = [(0,0,0),(0,-n,-n),(0,n,n),(-n,0,0),(n,0,0),(0,-n,n),(0,n,-n),(-n,-n,0),(n,n,0)]
    for dz,dy,dx in offs[:tries]:
        z0=min(max(cz+dz-n//2,0),Z-n); y0=min(max(cy+dy-n//2,0),Y-n); x0=min(max(cx+dx-n//2,0),X-n)
        try:
            blk = z.read_region(z0,y0,x0,n,n,n,workers=12)
        except Exception:
            continue
        if (blk>0).mean() > 0.5:
            return (z0,y0,x0), blk
    return None, None


def main():
    idx = json.load(open(INDEX))
    allvols=[]
    for scroll,info in idx.items():
        for vid,v in info.get("volumes",{}).items():
            lid=v.get("long_id","")
            m=re.search(r'-([\d.]+)um-.*?-(\d+)keV',lid)
            if not m: continue
            allvols.append((scroll,vid,lid,float(m.group(1)),int(m.group(2))))
    targets = pick_diverse(allvols)
    print(f"evaluating {len(targets)} volumes spanning resolutions {sorted(set(t[3] for t in targets))}\n")
    out = open("/tmp/multisource.txt","w")
    out.write(f"{'scroll':>16} {'res':>6} {'keV':>4} {'md_src':>8} {'db':>6} {'contr':>6} {'noise':>6} {'sharp':>6} {'corelost':>8} {'verdict':>8}\n")
    for scroll,vid,lid,res,en in targets:
        try:
            z = s3zarr.open_s3(BUCKET, scroll, f"volumes/{lid}", 0)
            (org, blk) = find_occupied(z)
            if blk is None:
                out.write(f"{scroll:>16} {res:>6} {en:>4} {'NO-OCC':>8}\n"); out.flush(); continue
            md, md_src = build_md((scroll,vid,lid,res,en), idx, z)
            f = blk.astype(np.float32)/255.0
            cal = fp.calibrate_prepass(md, [blk], verbose=False)
            cal.air_thresh = fp.air_thresh_from_physics(md); cal.do_air_zero = True; cal.scratch_passes = 5
            o = fp.process_tile(np.ascontiguousarray(blk,np.uint8), cal).astype(np.float32)/255.0
            pap = binary_erosion(f>(130/255), iterations=2)
            contr = fm.midband_ratio(o,f); noise = fm.flat_noise(o)/(fm.flat_noise(f)+1e-9)
            sharp = fm.edge_sharpness(o)/(fm.edge_sharpness(f)+1e-9)
            corelost = (o[pap]<=0.001).mean()*100 if pap.sum()>500 else 0
            verdict = "OK" if (contr>=1.1 and corelost<1.0 and np.isfinite(o).all()) else "CHECK"
            out.write(f"{scroll:>16} {res:>6} {en:>4} {md_src:>8} {md['delta_beta']:>6.0f} {contr:>6.2f} {noise:>6.2f} {sharp:>6.2f} {corelost:>7.2f}% {verdict:>8}\n"); out.flush()
        except Exception as e:
            out.write(f"{scroll:>16} {res:>6} {en:>4} ERROR {type(e).__name__}: {str(e)[:50]}\n"); out.flush()
    out.write("\nverdict OK = contrast>=1.1, corelost<1%, finite output\n")
    out.close()


if __name__ == "__main__":
    main()
