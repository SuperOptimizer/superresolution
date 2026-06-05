#!/usr/bin/env python3
"""Run the whole-volume fysics pipeline over a LOCAL zarr (level-0), streaming, on disk.

Reads chunks from a local zarr dir (fast random disk reads), processes tile-by-tile
(constant RAM), writes a processed output zarr. Proves the pipeline on a real complete
native volume without S3 latency. Memory stays bounded -> works on 20TB.

  python scripts/run_local_volume.py --in /home/forrest/paris4_local --out /home/forrest/paris4_out
"""
import argparse, json, os, sys, time, resource
import numpy as np
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from superres import fysics_pipeline as fp


class LocalZarr:
    """Minimal raw-uint8 zarr reader/writer (compressor=null, dimension_separator='/')."""
    def __init__(self, root, level="0", mode="r", shape=None, chunks=(128, 128, 128)):
        self.root = root; self.level = level; self.lvldir = os.path.join(root, level)
        if mode == "r":
            za = json.load(open(os.path.join(self.lvldir, ".zarray")))
            self.shape = tuple(za["shape"]); self.chunks = tuple(za["chunks"])
            assert za.get("compressor") is None, "expected raw (uncompressed) chunks"
        else:
            os.makedirs(self.lvldir, exist_ok=True)
            self.shape = tuple(shape); self.chunks = tuple(chunks)
            json.dump({"chunks": list(self.chunks), "compressor": None,
                       "dtype": "|u1", "fill_value": 0, "order": "C",
                       "shape": list(self.shape), "zarr_format": 2,
                       "dimension_separator": "/"},
                      open(os.path.join(self.lvldir, ".zarray"), "w"))
        self.ncz, self.ncy, self.ncx = [-(-s // c) for s, c in zip(self.shape, self.chunks)]

    def _path(self, cz, cy, cx):
        return os.path.join(self.lvldir, str(cz), str(cy), str(cx))

    def get_chunk(self, cz, cy, cx):
        p = self._path(cz, cy, cx)
        if not os.path.exists(p):
            return None  # missing -> fill (air)
        b = open(p, "rb").read()
        n = int(np.prod(self.chunks))
        if len(b) != n:
            return None
        return np.frombuffer(b, np.uint8).reshape(self.chunks)

    def put_chunk(self, cz, cy, cx, arr):
        # only write full chunks (edge handled by caller padding to chunk size)
        p = self._path(cz, cy, cx); os.makedirs(os.path.dirname(p), exist_ok=True)
        np.ascontiguousarray(arr, np.uint8).tofile(p)

    def read_region(self, z, y, x, dz, dy, dx):
        o = np.zeros((dz, dy, dx), np.uint8); cz0 = z // 128; cy0 = y // 128; cx0 = x // 128
        for cz in range(z // 128, (z + dz - 1) // 128 + 1):
            for cy in range(y // 128, (y + dy - 1) // 128 + 1):
                for cx in range(x // 128, (x + dx - 1) // 128 + 1):
                    c = self.get_chunk(cz, cy, cx)
                    if c is None: continue
                    gz, gy, gx = cz * 128, cy * 128, cx * 128
                    a0, b0, c0 = max(z, gz), max(y, gy), max(x, gx)
                    a1, b1, c1 = min(z + dz, gz + 128), min(y + dy, gy + 128), min(x + dx, gx + 128)
                    o[a0 - z:a1 - z, b0 - y:b1 - y, c0 - x:c1 - x] = c[a0 - gz:a1 - gz, b0 - gy:b1 - gy, c0 - gx:c1 - gx]
        return o


def write_region_zarr(zo: LocalZarr):
    """Returns a write_region that buffers inner tiles aligned to output chunks. Since
    the pipeline writes inner tiles (tile-aligned, tile=multiple of 128), and chunks are
    128, we can write each 128-subblock directly."""
    def wr(z, y, x, blk):
        dz, dy, dx = blk.shape
        for cz in range(z // 128, (z + dz - 1) // 128 + 1):
            for cy in range(y // 128, (y + dy - 1) // 128 + 1):
                for cx in range(x // 128, (x + dx - 1) // 128 + 1):
                    gz, gy, gx = cz * 128, cy * 128, cx * 128
                    sub = np.zeros((128, 128, 128), np.uint8)
                    a0, b0, c0 = max(z, gz), max(y, gy), max(x, gx)
                    a1, b1, c1 = min(z + dz, gz + 128), min(y + dy, gy + 128), min(x + dx, gx + 128)
                    sub[a0 - gz:a1 - gz, b0 - gy:b1 - gy, c0 - gx:c1 - gx] = blk[a0 - z:a1 - z, b0 - y:b1 - y, c0 - x:c1 - x]
                    zo.put_chunk(cz, cy, cx, sub)
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--no-deconv", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--diffusion", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="parallel tile workers (default cores-2; 1=serial)")
    args = ap.parse_args()

    zin = LocalZarr(args.inp, "0", "r")
    SH = zin.shape
    print(f"input {SH} = {np.prod(SH)/1e9:.1f}GB level-0", flush=True)
    zout = LocalZarr(args.out, "0", "w", shape=SH, chunks=(128, 128, 128))

    md = json.load(open(os.path.join(args.inp, "metadata.json")))
    t = md.get("scan", {}).get("tomo", md.get("tomo", {})); a = t.get("acquisition", {})
    ph = t.get("processing", {}).get("preprocessing", {}).get("phase", {})
    md_phys = dict(delta_beta=ph.get("delta_beta", 1000), energy_kev=a.get("energy", 110),
                   distance_mm=a.get("sampleDetectorDistance", 11000),
                   pixel_um=a.get("detector", {}).get("samplePixelSize", 0.045532) * 1000,
                   unsharp_sigma=ph.get("unsharp_sigma", 1.2), unsharp_coeff=ph.get("unsharp_coeff", 4.0))
    print("beam mA:", a.get("machineCurrentStart"), "->", a.get("machineCurrentStop"), flush=True)
    # calibration PRE-PASS: sample a few textured tiles, tune the whole chain across
    # a bucket of metrics (resolution-adaptive; turns off stages that can't help).
    samples = []
    for (sz, sy, sx) in [(SH[0]//2, SH[1]//2, SH[2]//2), (SH[0]//2, SH[1]//3, SH[2]//2),
                         (SH[0]//3, SH[1]//2, SH[2]//3), (2*SH[0]//3, SH[1]//2, SH[2]//2)]:
        t = zin.read_region(sz, sy, sx, 128, 128, 128)
        if t.std() > 20 and (t > 20).mean() > 0.5:
            samples.append(t)
    if not samples:
        samples = [zin.read_region(SH[0]//2, SH[1]//2, SH[2]//2, 128, 128, 128)]
    cal = fp.calibrate_prepass(md_phys, samples, verbose=True)
    print(f"calib db_scale={cal.db_scale:.2f} noise_ref={cal.noise_ref:.4f} eps={cal.guided_eps:.4f} halo={cal.halo}", flush=True)

    t0 = time.time(); last = [0]
    def prog(p, f):
        if isinstance(f, float) and time.time() - last[0] > 20:
            last[0] = time.time()
            print(f"  {p} {f*100:.0f}% {time.time()-t0:.0f}s RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
    runner = (fp.run_pipeline_2pass if args.workers == 1 else fp.run_pipeline_2pass_parallel)
    kw = {} if args.workers == 1 else {"workers": args.workers}
    stats = runner(zin.read_region, write_region_zarr(zout), SH, cal,
                   tile=args.tile, do_normalize=True, do_zdrift=True,
                   do_deconv=not args.no_deconv, do_denoise=not args.no_denoise,
                   do_diffusion=args.diffusion, progress=prog, **kw)
    print(f"DONE {stats} in {time.time()-t0:.0f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
    print(f"zdrift drift_frac={getattr(cal,'zdrift_drift_frac',0):.3f} applied={cal.zdrift_factor is not None}; "
          f"norm {cal.norm_lo}/{cal.norm_hi} air={cal.air_thresh:.3f}", flush=True)


if __name__ == "__main__":
    main()
