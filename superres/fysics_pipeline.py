"""Whole-volume preprocessing pipeline driven by the fysics C kernels.

This is the ORCHESTRATOR: fysics does the per-chunk math (deconv, denoise, ...);
this streams a (possibly 20TB) volume chunk-by-chunk, feeds each tile+halo through
the validated, auto-calibrated chain, and writes the inner tile back. The caller
owns I/O via read_region/write_region callables (works on S3 zarr, local zarr, npy).

Pipeline (the validated order):
  pass 0  CALIBRATE (once, from metadata + a few sampled chunks):
            - physics (delta_beta, energy, distance, u8<->phys window) from metadata.json
            - auto delta_beta scale (partial inversion on fine volumes)
            - noise level (for auto-denoise strength)
  pass 1  STREAM each occupied tile (read tile+halo):
            u8 -> [dewindow to phys] -> deconvolve(auto reg, scaled db) ->
            denoise(calibrated) -> [optional: coherence-diffusion / MUSICA] ->
            -> back to u8 -> write inner tile (halo discarded => seam-free)

Everything heavy is in C (libfysics) called via ctypes; Python only orchestrates.
"""
from __future__ import annotations
import ctypes as C
import os
from pathlib import Path

import numpy as np

_LIB = os.environ.get("FYSICS_LIB", "/home/forrest/fysics/build/libfysics.so")


# ---------------------------------------------------------------- ctypes bindings
class _Phys(C.Structure):
    _fields_ = [("delta_beta", C.c_double), ("energy_kev", C.c_double),
                ("distance_mm", C.c_double), ("pixel_um", C.c_double),
                ("unsharp_sigma", C.c_double), ("unsharp_coeff", C.c_double),
                ("psf_sigma_vox", C.c_double)]


class _NoiseModel(C.Structure):
    _fields_ = [("g", C.c_double), ("b", C.c_double), ("noise_ref", C.c_double),
                ("ref_intensity", C.c_double), ("n_bins_used", C.c_int)]


class _HistState(C.Structure):
    _fields_ = [("hist", C.c_long * 256), ("total", C.c_long)]


def _load():
    lib = C.CDLL(_LIB)
    f32p = C.POINTER(C.c_float)
    u8p = C.POINTER(C.c_ubyte)
    lib.fy_u8_to_phys.argtypes = [u8p, f32p, C.c_size_t, C.c_double, C.c_double]
    lib.fy_phys_to_u8.argtypes = [f32p, u8p, C.c_size_t, C.c_double, C.c_double]
    lib.fy_deconvolve.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int,
                                  C.POINTER(_Phys), C.c_double]
    lib.fy_deconvolve.restype = C.c_int
    lib.fy_auto_deltabeta_scale.argtypes = [C.POINTER(_Phys)]
    lib.fy_auto_deltabeta_scale.restype = C.c_double
    lib.fy_estimate_noise.argtypes = [f32p, C.c_int, C.c_int, C.c_int, C.c_int,
                                      C.c_double, C.c_double, C.POINTER(_NoiseModel)]
    lib.fy_estimate_noise.restype = C.c_int
    lib.fy_guided_eps_for_noise.argtypes = [C.c_double]
    lib.fy_guided_eps_for_noise.restype = C.c_double
    lib.fy_guided_denoise.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int,
                                      C.c_int, C.c_double]
    lib.fy_guided_denoise.restype = C.c_int
    lib.fy_coherence_diffusion_auto.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int, C.c_int]
    lib.fy_coherence_diffusion_auto.restype = C.c_int
    lib.fy_kernel_halo.argtypes = [C.POINTER(_Phys)]
    lib.fy_kernel_halo.restype = C.c_int
    # two-pass global stats
    lib.fy_hist_init.argtypes = [C.POINTER(_HistState)]
    lib.fy_hist_accumulate_u8.argtypes = [C.POINTER(_HistState), u8p, C.c_size_t]
    lib.fy_hist_percentile_u8.argtypes = [C.POINTER(_HistState), C.c_double]
    lib.fy_hist_percentile_u8.restype = C.c_int
    lib.fy_norm_apply_u8.argtypes = [u8p, f32p, C.c_size_t, C.c_ubyte, C.c_ubyte]
    lib.fy_auto_air_thresh.argtypes = [C.POINTER(_HistState)]
    lib.fy_auto_air_thresh.restype = C.c_float
    # z-drift (shading / beam-current decay across z)
    dp = C.POINTER(C.c_double); lp = C.POINTER(C.c_long); fp_ = C.POINTER(C.c_float)
    lib.fy_zdrift_accumulate.argtypes = [f32p, C.c_int, C.c_int, C.c_int, C.c_int, dp, lp, C.c_float]
    lib.fy_zdrift_finalize.argtypes = [dp, lp, C.c_int, fp_]
    lib.fy_zdrift_apply.argtypes = [f32p, C.c_int, C.c_int, C.c_int, C.c_int, fp_]
    return lib


_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def _fp(a):  # float32 contiguous -> c_float*
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(C.POINTER(C.c_float))


# ---------------------------------------------------------------- physics from metadata
def physics_struct(md_phys: dict) -> _Phys:
    """Build the fysics _Phys from a physics dict (delta_beta, energy_kev, ...)."""
    return _Phys(
        float(md_phys.get("delta_beta", 1000.0)),
        float(md_phys.get("energy_kev", md_phys.get("energy", 78.0))),
        float(md_phys.get("distance_mm", md_phys.get("distance", 220.0))),
        float(md_phys.get("pixel_um", md_phys.get("pixel", 2.4))),
        float(md_phys.get("unsharp_sigma", 1.2)),
        float(md_phys.get("unsharp_coeff", 4.0)),
        0.0,
    )


# ---------------------------------------------------------------- pass 0: calibration
class Calibration:
    """Per-volume calibration computed once and reused for every tile."""
    def __init__(self, phys: _Phys, db_scale: float, noise_ref: float,
                 guided_eps: float, halo: int, window: tuple | None,
                 guided_eps_raw: float = None,
                 norm_lo: int = None, norm_hi: int = None,
                 air_thresh: float = None, zdrift_factor=None):
        self.phys = phys
        self.db_scale = db_scale
        self.noise_ref = noise_ref
        self.guided_eps = guided_eps          # post-deconv strength (in-pipeline)
        self.guided_eps_raw = guided_eps_raw  # raw-noise strength (denoise-only use)
        self.halo = halo
        self.window = window  # (f32_min, f32_max) for u8<->phys, or None to keep u8 scale
        # ---- whole-volume stats (set by the pass-1 finalize; None until then) ----
        self.norm_lo = norm_lo        # u8 lo for global normalization
        self.norm_hi = norm_hi        # u8 hi for global normalization
        self.air_thresh = air_thresh  # Otsu air threshold [0,1]
        self.zdrift_factor = zdrift_factor  # per-z correction factor array (float32)

    def scaled_phys(self) -> _Phys:
        p = _Phys(self.phys.delta_beta * self.db_scale, self.phys.energy_kev,
                  self.phys.distance_mm, self.phys.pixel_um,
                  self.phys.unsharp_sigma, self.phys.unsharp_coeff, 0.0)
        return p


def calibrate(md_phys: dict, sample_chunks, window=None,
              auto_deltabeta=True) -> Calibration:
    """Compute per-volume calibration from metadata physics + a few sampled chunks.

    md_phys: physics dict from metadata. sample_chunks: iterable of small float32
    arrays (in [0,1]) from TEXTURED regions, used to estimate the noise level.
    window: (f32_min, f32_max) from metadata zarr_export if dewindowing to physical
    units; None keeps the u8/[0,1] scale.
    """
    L = lib()
    phys = physics_struct(md_phys)
    db_scale = L.fy_auto_deltabeta_scale(C.byref(phys)) if auto_deltabeta else 1.0
    halo = L.fy_kernel_halo(C.byref(phys))

    # noise level: median noise_ref over the sampled textured chunks
    refs = []
    for ch in sample_chunks:
        arr, p = _fp(ch)
        nm = _NoiseModel()
        nz, ny, nx = arr.shape
        if L.fy_estimate_noise(p, nz, ny, nx, 5, 10.0, 0.4, C.byref(nm)) == 0 and nm.noise_ref > 0:
            refs.append(nm.noise_ref)
    noise_ref = float(np.median(refs)) if refs else 0.02
    # NOTE: deconv runs BEFORE denoise and amplifies the noise ~3-4x (measured on real
    # PHerc data: flat-region noise 1.56 -> 5.54 u8 levels after deconv). The base
    # fy_guided_eps_for_noise is calibrated to the RAW noise, so in-pipeline (post-
    # deconv) it under-denoises. Scale eps for the post-deconv noise: eps ~ noise^2, so
    # a ~3.5x noise rise -> ~12x eps; empirically x4 on the linear noise (=>~ the same)
    # restores flat noise BELOW raw while keeping texture. Skip the boost if no deconv.
    POST_DECONV_NOISE_GAIN = 3.5
    guided_eps_raw = L.fy_guided_eps_for_noise(noise_ref)
    guided_eps = L.fy_guided_eps_for_noise(noise_ref * POST_DECONV_NOISE_GAIN)
    return Calibration(phys, db_scale, noise_ref, guided_eps, halo, window,
                       guided_eps_raw=guided_eps_raw)


# ---------------------------------------------------------------- the per-tile chain
def process_tile(block_u8: np.ndarray, cal: Calibration,
                 do_deconv=True, do_denoise=True, do_diffusion=False,
                 diffusion_strength=2) -> np.ndarray:
    """Run the full chain on one tile (with halo). Input/output uint8.

    Order: u8 -> [phys] -> deconv(scaled db, auto reg) -> denoise -> [diffusion] ->
    -> u8. The HALO is part of block_u8; the caller crops the inner tile after.
    """
    L = lib()
    nz, ny, nx = block_u8.shape
    n = nz * ny * nx

    # to float [0,1] (the kernels operate in this normalized scale; dewindow is a
    # linear reparam that the deconv/denoise are invariant to up to scale, so we
    # keep [0,1] for the in-pipeline math and only dewindow when PHYSICAL units are
    # requested downstream).
    cur = (block_u8.astype(np.float32) / 255.0)

    if do_deconv:
        inp, ip = _fp(cur)
        out = np.empty_like(inp); op = out.ctypes.data_as(C.POINTER(C.c_float))
        ph = cal.scaled_phys()
        rc = L.fy_deconvolve(ip, op, nz, ny, nx, C.byref(ph), -1.0)  # reg<=0 -> auto
        if rc != 0:
            raise RuntimeError("fy_deconvolve failed")
        cur = out

    if do_denoise:
        inp, ip = _fp(cur)
        out = np.empty_like(inp); op = out.ctypes.data_as(C.POINTER(C.c_float))
        rc = L.fy_guided_denoise(ip, op, nz, ny, nx, 2, cal.guided_eps)
        if rc != 0:
            raise RuntimeError("fy_guided_denoise failed")
        cur = out

    if do_diffusion:
        inp, ip = _fp(cur)
        out = np.empty_like(inp); op = out.ctypes.data_as(C.POINTER(C.c_float))
        rc = L.fy_coherence_diffusion_auto(ip, op, nz, ny, nx, int(diffusion_strength))
        if rc != 0:
            raise RuntimeError("fy_coherence_diffusion_auto failed")
        cur = out

    # back to u8
    cur = np.clip(cur * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cur


# ---------------------------------------------------------------- pass 1: stream
def run_pipeline(read_region, write_region, shape, cal: Calibration,
                 tile=256, occupancy_skip=True,
                 do_deconv=True, do_denoise=True, do_diffusion=False,
                 progress=lambda *_: None):
    """Halo-tiled whole-volume processing. read_region(z,y,x,dz,dy,dx)->u8 ndarray;
    write_region(z,y,x,ndarray)->None. Streams; never holds the volume in RAM.

    For each occupied tile: read tile+halo, run the chain, write only the inner
    tile (halo discarded -> seam-free, since edge voxels were processed with real
    context).
    """
    Z, Y, X = shape
    halo = cal.halo
    nz = -(-Z // tile); ny = -(-Y // tile); nx = -(-X // tile)
    total = nz * ny * nx; done = 0; processed = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                z0, y0, x0 = iz * tile, iy * tile, ix * tile
                tz = min(tile, Z - z0); ty = min(tile, Y - y0); tx = min(tile, X - x0)
                rz0, ry0, rx0 = max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)
                rz1 = min(Z, z0 + tz + halo); ry1 = min(Y, y0 + ty + halo); rx1 = min(X, x0 + tx + halo)
                blk = read_region(rz0, ry0, rx0, rz1 - rz0, ry1 - ry0, rx1 - rx0)
                done += 1; progress(done, total)
                if occupancy_skip and not np.any(blk):
                    continue
                out = process_tile(blk, cal, do_deconv=do_deconv,
                                    do_denoise=do_denoise, do_diffusion=do_diffusion)
                iz0, iy0, ix0 = z0 - rz0, y0 - ry0, x0 - rx0
                inner = out[iz0:iz0 + tz, iy0:iy0 + ty, ix0:ix0 + tx]
                write_region(z0, y0, x0, inner)
                processed += 1
    return {"tiles_total": total, "tiles_processed": processed}


# ---------------------------------------------------------------- pass 1: global stats
def accumulate_global_stats(read_region, shape, cal: Calibration, tile=256,
                            want_norm=True, want_zdrift=True,
                            norm_lo_pct=0.5, norm_hi_pct=99.5,
                            zdrift_min_frac=0.05,
                            progress=lambda *_: None):
    """PASS 1 (streaming, cheap): accumulate WHOLE-VOLUME statistics so pass 2 can apply
    a CONSISTENT global mapping -- the proper way to normalize a volume too big for RAM.

      - global histogram -> lo/hi percentiles for normalization + Otsu air threshold
      - per-z papyrus mean -> beam-current / shading DRIFT correction (intensity ranges
        across the volume; metadata machineCurrentStart/Stop confirms ~1.5-13.7% decay).
    State is tiny (256-bin histogram + 2 arrays of length Z). Mutates `cal` in place
    (sets norm_lo/hi, air_thresh, zdrift_factor). Reads NO halo -- plain tiling.
    """
    L = lib()
    Z, Y, X = shape
    hist = _HistState(); L.fy_hist_init(C.byref(hist))
    sums = np.zeros(Z, np.float64); counts = np.zeros(Z, np.int64)
    sums_p = sums.ctypes.data_as(C.POINTER(C.c_double))
    counts_p = counts.ctypes.data_as(C.POINTER(C.c_long))
    # a papyrus threshold for zdrift, in [0,1]; use a low fixed value (air is near 0)
    pap_thr = 0.10
    nz = -(-Z // tile); ny = -(-Y // tile); nx = -(-X // tile)
    total = nz * ny * nx; done = 0
    for iz in range(nz):
        z0 = iz * tile; tz = min(tile, Z - z0)
        for iy in range(ny):
            y0 = iy * tile; ty = min(tile, Y - y0)
            for ix in range(nx):
                x0 = ix * tile; tx = min(tile, X - x0)
                blk = read_region(z0, y0, x0, tz, ty, tx)
                done += 1; progress(done, total)
                if not np.any(blk):
                    continue
                u8 = np.ascontiguousarray(blk, np.uint8)
                if want_norm:
                    L.fy_hist_accumulate_u8(C.byref(hist),
                                            u8.ctypes.data_as(C.POINTER(C.c_ubyte)), u8.size)
                if want_zdrift:
                    f = np.ascontiguousarray(u8.astype(np.float32) / 255.0)
                    L.fy_zdrift_accumulate(f.ctypes.data_as(C.POINTER(C.c_float)),
                                           tz, ty, tx, z0, sums_p, counts_p, C.c_float(pap_thr))
    if want_norm and hist.total > 0:
        cal.norm_lo = int(L.fy_hist_percentile_u8(C.byref(hist), norm_lo_pct))
        cal.norm_hi = int(L.fy_hist_percentile_u8(C.byref(hist), norm_hi_pct))
        cal.air_thresh = float(L.fy_auto_air_thresh(C.byref(hist)))
    if want_zdrift and counts.sum() > 0:
        factor = np.zeros(Z, np.float32)
        L.fy_zdrift_finalize(sums_p, counts_p, Z,
                             factor.ctypes.data_as(C.POINTER(C.c_float)))
        # GATE: only keep the correction if the drift is SIGNIFICANT. On a volume with
        # little drift the smoothed factor just fits noise and ADDS spread (measured:
        # PHercParis4 45um has ~3% drift -> correction hurt). The factor's own range is
        # the measured drift magnitude; require > a threshold (default 5%). The metadata
        # beam-current delta is the physical confirmation (set cal.beam_drift_frac).
        fr = float(np.nanmax(factor)) - float(np.nanmin(factor))
        cal.zdrift_drift_frac = fr
        if fr >= zdrift_min_frac:
            cal.zdrift_factor = factor
        else:
            cal.zdrift_factor = None  # negligible drift -> don't correct
    return cal


def run_pipeline_2pass(read_region, write_region, shape, cal: Calibration,
                       tile=256, occupancy_skip=True,
                       do_normalize=True, do_zdrift=True,
                       do_deconv=True, do_denoise=True, do_diffusion=False,
                       progress=lambda *_: None):
    """Full TWO-PASS whole-volume pipeline.
      pass 1: accumulate_global_stats (global histogram + z-drift profile)
      pass 2: per tile -> z-drift correct + global normalize -> deconv -> denoise ->
              [diffusion] -> write inner tile.
    Gives CONSISTENT intensity across the whole volume (drift-corrected + globally
    normalized) AND per-tile restoration, all streaming. The z-drift/normalize are
    applied to the WHOLE haloed block before the local ops so seams stay clean.
    """
    L = lib()
    Z, Y, X = shape
    if do_normalize or do_zdrift:
        progress("pass1", 0)
        accumulate_global_stats(read_region, shape, cal, tile=tile,
                                want_norm=do_normalize, want_zdrift=do_zdrift,
                                progress=lambda d, t: progress("pass1", d / t))
    halo = cal.halo
    nz = -(-Z // tile); ny = -(-Y // tile); nx = -(-X // tile)
    total = nz * ny * nx; done = 0; processed = 0
    lo = cal.norm_lo if (do_normalize and cal.norm_lo is not None) else 0
    hi = cal.norm_hi if (do_normalize and cal.norm_hi is not None) else 255
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                z0, y0, x0 = iz * tile, iy * tile, ix * tile
                tz = min(tile, Z - z0); ty = min(tile, Y - y0); tx = min(tile, X - x0)
                rz0, ry0, rx0 = max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)
                rz1 = min(Z, z0 + tz + halo); ry1 = min(Y, y0 + ty + halo); rx1 = min(X, x0 + tx + halo)
                blk = read_region(rz0, ry0, rx0, rz1 - rz0, ry1 - ry0, rx1 - rx0)
                done += 1; progress("pass2", done / total)
                if occupancy_skip and not np.any(blk):
                    continue
                # --- global intensity corrections on the haloed block (float [0,1]) ---
                u8 = np.ascontiguousarray(blk, np.uint8)
                if do_normalize and cal.norm_lo is not None:
                    f = np.empty(u8.size, np.float32)
                    L.fy_norm_apply_u8(u8.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                       f.ctypes.data_as(C.POINTER(C.c_float)), u8.size,
                                       C.c_ubyte(lo), C.c_ubyte(hi))
                    fcur = f.reshape(u8.shape)
                else:
                    fcur = u8.astype(np.float32) / 255.0
                if do_zdrift and cal.zdrift_factor is not None:
                    fcur = np.ascontiguousarray(fcur)
                    L.fy_zdrift_apply(fcur.ctypes.data_as(C.POINTER(C.c_float)),
                                      fcur.shape[0], fcur.shape[1], fcur.shape[2], rz0,
                                      cal.zdrift_factor.ctypes.data_as(C.POINTER(C.c_float)))
                # back to u8 for the per-tile chain (which re-normalizes to [0,1])
                blk_corr = np.clip(fcur * 255.0 + 0.5, 0, 255).astype(np.uint8)
                out = process_tile(blk_corr, cal, do_deconv=do_deconv,
                                   do_denoise=do_denoise, do_diffusion=do_diffusion)
                iz0, iy0, ix0 = z0 - rz0, y0 - ry0, x0 - rx0
                inner = out[iz0:iz0 + tz, iy0:iy0 + ty, ix0:ix0 + tx]
                write_region(z0, y0, x0, inner)
                processed += 1
    return {"tiles_total": total, "tiles_processed": processed,
            "norm_lo": cal.norm_lo, "norm_hi": cal.norm_hi,
            "air_thresh": cal.air_thresh}


# ---------------------------------------------------------------- PARALLEL (bounded)
def _tile_coords(shape, tile):
    Z, Y, X = shape
    nz = -(-Z // tile); ny = -(-Y // tile); nx = -(-X // tile)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                yield iz * tile, iy * tile, ix * tile


def run_pipeline_2pass_parallel(read_region, write_region, shape, cal: Calibration,
                                tile=256, workers=None, occupancy_skip=True,
                                do_normalize=True, do_zdrift=True,
                                do_deconv=True, do_denoise=True, do_diffusion=False,
                                progress=lambda *_: None):
    """Tile-PARALLEL two-pass pipeline. Same result as run_pipeline_2pass (tiles are
    independent -> seam-free is preserved), but processes `workers` tiles concurrently.

    MEMORY stays BOUNDED: peak RAM ~= workers * (one tile+halo working set). With
    workers = cores-2 and a 256^3 tile that's a few GB, NOT volume-sized -> still
    20TB-safe. The C kernels release the GIL during compute (pure-C ctypes calls), so
    threads give real parallelism. Each worker forces single-threaded C (OMP=1) so we
    parallelize over TILES, not nested threads (no oversubscription).

    read_region/write_region MUST be thread-safe (concurrent calls with disjoint
    regions). For a local/S3 zarr where each chunk is a separate file/object and tiles
    write disjoint chunks, this holds.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 2)
    L = lib()
    Z, Y, X = shape
    halo = cal.halo

    # ---- PASS 1: parallel accumulate with per-worker partial state, then merge ----
    if do_normalize or do_zdrift:
        coords = list(_tile_coords(shape, tile))
        # one state per worker thread, kept in a dict keyed by thread id so the main
        # thread can merge them after the pool closes (threadlocal would be empty here).
        states = {}; states_lock = threading.Lock()

        def _worker_state():
            tid = threading.get_ident()
            st = states.get(tid)
            if st is None:
                st = type("S", (), {})()
                st.hist = _HistState(); L.fy_hist_init(C.byref(st.hist))
                st.sums = np.zeros(Z, np.float64); st.counts = np.zeros(Z, np.int64)
                with states_lock:
                    states[tid] = st
            return st

        def p1(coord):
            z0, y0, x0 = coord
            tz = min(tile, Z - z0); ty = min(tile, Y - y0); tx = min(tile, X - x0)
            blk = read_region(z0, y0, x0, tz, ty, tx)
            if not np.any(blk):
                return
            st = _worker_state()
            u8 = np.ascontiguousarray(blk, np.uint8)
            if do_normalize:
                L.fy_hist_accumulate_u8(C.byref(st.hist),
                                        u8.ctypes.data_as(C.POINTER(C.c_ubyte)), u8.size)
            if do_zdrift:
                f = np.ascontiguousarray(u8.astype(np.float32) / 255.0)
                L.fy_zdrift_accumulate(f.ctypes.data_as(C.POINTER(C.c_float)),
                                       tz, ty, tx, z0,
                                       st.sums.ctypes.data_as(C.POINTER(C.c_double)),
                                       st.counts.ctypes.data_as(C.POINTER(C.c_long)),
                                       C.c_float(0.10))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, _ in enumerate(ex.map(p1, coords)):
                progress("pass1", (i + 1) / len(coords))
        states = list(states.values())
        # merge per-thread states
        merged = _HistState(); L.fy_hist_init(C.byref(merged))
        msum = np.zeros(Z, np.float64); mcnt = np.zeros(Z, np.int64)
        for st in states:
            L.fy_hist_merge(C.byref(merged), C.byref(st.hist))
            msum += st.sums; mcnt += st.counts
        if do_normalize and merged.total > 0:
            cal.norm_lo = int(L.fy_hist_percentile_u8(C.byref(merged), 0.5))
            cal.norm_hi = int(L.fy_hist_percentile_u8(C.byref(merged), 99.5))
            cal.air_thresh = float(L.fy_auto_air_thresh(C.byref(merged)))
        if do_zdrift and mcnt.sum() > 0:
            factor = np.zeros(Z, np.float32)
            L.fy_zdrift_finalize(msum.ctypes.data_as(C.POINTER(C.c_double)),
                                 mcnt.ctypes.data_as(C.POINTER(C.c_long)), Z,
                                 factor.ctypes.data_as(C.POINTER(C.c_float)))
            fr = float(np.nanmax(factor)) - float(np.nanmin(factor))
            cal.zdrift_drift_frac = fr
            cal.zdrift_factor = factor if fr >= 0.05 else None

    # ---- PASS 2: parallel per-tile processing ----
    coords = list(_tile_coords(shape, tile))
    lo = cal.norm_lo if (do_normalize and cal.norm_lo is not None) else 0
    hi = cal.norm_hi if (do_normalize and cal.norm_hi is not None) else 255
    done = [0]; processed = [0]; lock = threading.Lock(); total = len(coords)

    def p2(coord):
        z0, y0, x0 = coord
        tz = min(tile, Z - z0); ty = min(tile, Y - y0); tx = min(tile, X - x0)
        rz0, ry0, rx0 = max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)
        rz1 = min(Z, z0 + tz + halo); ry1 = min(Y, y0 + ty + halo); rx1 = min(X, x0 + tx + halo)
        blk = read_region(rz0, ry0, rx0, rz1 - rz0, ry1 - ry0, rx1 - rx0)
        with lock:
            done[0] += 1; progress("pass2", done[0] / total)
        if occupancy_skip and not np.any(blk):
            return
        u8 = np.ascontiguousarray(blk, np.uint8)
        if do_normalize and cal.norm_lo is not None:
            f = np.empty(u8.size, np.float32)
            L.fy_norm_apply_u8(u8.ctypes.data_as(C.POINTER(C.c_ubyte)),
                               f.ctypes.data_as(C.POINTER(C.c_float)), u8.size,
                               C.c_ubyte(lo), C.c_ubyte(hi))
            fcur = f.reshape(u8.shape)
        else:
            fcur = u8.astype(np.float32) / 255.0
        if do_zdrift and cal.zdrift_factor is not None:
            fcur = np.ascontiguousarray(fcur)
            L.fy_zdrift_apply(fcur.ctypes.data_as(C.POINTER(C.c_float)),
                              fcur.shape[0], fcur.shape[1], fcur.shape[2], rz0,
                              cal.zdrift_factor.ctypes.data_as(C.POINTER(C.c_float)))
        blk_corr = np.clip(fcur * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out = process_tile(blk_corr, cal, do_deconv=do_deconv,
                           do_denoise=do_denoise, do_diffusion=do_diffusion)
        iz0, iy0, ix0 = z0 - rz0, y0 - ry0, x0 - rx0
        inner = out[iz0:iz0 + tz, iy0:iy0 + ty, ix0:ix0 + tx]
        write_region(z0, y0, x0, inner)
        with lock:
            processed[0] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(p2, coords))
    return {"tiles_total": total, "tiles_processed": processed[0], "workers": workers,
            "norm_lo": cal.norm_lo, "norm_hi": cal.norm_hi, "air_thresh": cal.air_thresh}
