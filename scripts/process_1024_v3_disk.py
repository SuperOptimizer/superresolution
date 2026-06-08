#!/usr/bin/env python3
"""v3 export, ROBUST: (1) cache the raw 1024^3 region from S3 to LOCAL DISK once (with retry
on flaky/oversized chunk reads), then (2) process entirely OFF DISK with preprocess_volume
(global valley anchor + bounded local air cut). Faster and immune to mid-run S3 failures."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
from superres import s3zarr, fysics_pipeline as fp

BUCKET = "vesuvius-challenge-open-data"; SCROLL = "PHercParis4"
VOL = "volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
N = 1024
RAW = os.path.expanduser("~/paris4_2um_v3_raw")        # cached raw region
OUT = os.path.expanduser("~/paris4_2um_processed_v3")  # processed output


def cache_raw():
    """Download the raw 1024^3 region to a local raw-zarr, chunk by chunk, with retry."""
    src = s3zarr.open_s3(BUCKET, SCROLL, VOL, 0)
    Z, Y, X = src.shape
    z0 = (Z//2 - N//2)//128*128; y0 = (Y//2 - N//2)//128*128; x0 = (X//2 - N//2)//128*128
    print(f"caching raw region [{z0}:{z0+N}, {y0}:{y0+N}, {x0}:{x0+N}] -> {RAW}", flush=True)
    lvl = os.path.join(RAW, "0"); os.makedirs(lvl, exist_ok=True)
    json.dump({"chunks": [128,128,128], "compressor": None, "dtype": "|u1", "fill_value": 0,
               "order": "C", "shape": [N,N,N], "zarr_format": 2, "dimension_separator": "/"},
              open(os.path.join(lvl, ".zarray"), "w"))
    cz0, cy0, cx0 = z0//128, y0//128, x0//128
    nc = N//128; total = nc**3; done = 0; t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    def fetch(coord):
        dz, dy, dx = coord
        p = os.path.join(lvl, str(dz), str(dy), str(dx))
        if os.path.exists(p) and os.path.getsize(p) == 128**3:
            return  # already cached (resume)
        key = src._chunk_key(cz0+dz, cy0+dy, cx0+dx)
        for attempt in range(4):
            raw = src.backend.get(key)
            if raw is None:
                np.zeros(128**3, np.uint8).tofile(p); return       # missing -> air
            if len(raw) == 128**3:                                  # correct size
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(raw); return
            # bad/truncated read -> retry; if persistent, take the first 128^3 bytes
            if attempt == 3:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                buf = np.frombuffer(raw, np.uint8)[:128**3]
                if buf.size < 128**3:
                    buf = np.concatenate([buf, np.zeros(128**3-buf.size, np.uint8)])
                buf.tofile(p)
                print(f"  WARN chunk {dz}/{dy}/{dx}: bad size {len(raw)}, truncated to 128^3", flush=True)
    coords = [(dz,dy,dx) for dz in range(nc) for dy in range(nc) for dx in range(nc)]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, _ in enumerate(ex.map(fetch, coords)):
            done += 1
            if done % 64 == 0:
                print(f"  cached {done}/{total}  {time.time()-t0:.0f}s", flush=True)
    print(f"raw cached in {time.time()-t0:.0f}s", flush=True)
    return src


def main():
    if not (os.path.exists(os.path.join(RAW, "0", ".zarray"))
            and len([f for f in _allfiles(os.path.join(RAW, "0")) if not f.endswith(".zarray")]) >= 512):
        src = cache_raw()
        meta = json.loads(src.backend.get("metadata.json").decode("utf-8"))
        json.dump(meta, open(os.path.join(RAW, "metadata.json"), "w"))
    print("processing off local disk...", flush=True)
    z = s3zarr.open_local(RAW, 0)
    meta = json.load(open(os.path.join(RAW, "metadata.json")))
    read = lambda a,b,c,d,e,g: z.read_region(a,b,c,d,e,g, workers=8)

    lvl = os.path.join(OUT, "0"); os.makedirs(lvl, exist_ok=True)
    json.dump({"chunks": [128,128,128], "compressor": None, "dtype": "|u1", "fill_value": 0,
               "order": "C", "shape": [N,N,N], "zarr_format": 2, "dimension_separator": "/"},
              open(os.path.join(lvl, ".zarray"), "w"))
    def write(zz, yy, xx, blk):
        p = os.path.join(lvl, str(zz//128), str(yy//128)); os.makedirs(p, exist_ok=True)
        np.ascontiguousarray(blk, np.uint8).tofile(os.path.join(p, str(xx//128)))

    samp = fp.select_sample_tiles(read, (N,N,N), n=6, tile=128, candidates=27)
    t0 = time.time()
    def prog(tag, val=None):
        if tag in ("calibrated", "pass1_done"): print(f"  {tag}: {val}  {time.time()-t0:.0f}s", flush=True)
    stats, cal = fp.preprocess_volume(read, write, (N,N,N), meta, samp, tile=128,
                                      do_air_zero=True, scratch_passes=5, progress=prog)
    print(f"GLOBAL cal: reg={cal.deconv_reg} eps={cal.guided_eps:.4g} "
          f"air_cut_u8={cal.air_cut_u8} band=+/-{cal.air_cut_band}", flush=True)
    print(f"DONE -> {OUT}/0/  ({stats}, {time.time()-t0:.0f}s processing)", flush=True)


def _allfiles(d):
    for r, _, fs in os.walk(d):
        for f in fs: yield f


if __name__ == "__main__":
    main()
