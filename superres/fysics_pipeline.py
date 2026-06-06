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
    lib.fy_coherence_diffusion_halo.argtypes = [C.c_double, C.c_double, C.c_int]
    lib.fy_coherence_diffusion_halo.restype = C.c_int
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
    # papyrus/air masking (keep sharpening on papyrus; leave air flat)
    lib.fy_papyrus_mask.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int,
                                    C.c_float, C.c_float, C.c_float, C.c_float, C.c_int]
    lib.fy_papyrus_mask.restype = C.c_int
    # bilateral (used for the throwaway-mask scratch denoise) + histogram metrics
    lib.fy_bilateral_denoise.argtypes = [f32p, f32p, C.c_int, C.c_int, C.c_int,
                                         C.c_double, C.c_double, C.c_int]
    lib.fy_bilateral_denoise.restype = C.c_int
    lib.fy_valley_depth.argtypes = [C.POINTER(C.c_long), C.POINTER(C.c_int),
                                    C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.fy_valley_depth.restype = C.c_double
    # whole-chain quality metrics (used to reconcile the calibration objective with the
    # validated quality basket: noise floor, edge/sheet sharpness)
    lib.fy_edge_sharpness.argtypes = [f32p, C.c_int, C.c_int, C.c_int]
    lib.fy_edge_sharpness.restype = C.c_double
    lib.fy_flat_noise.argtypes = [f32p, C.c_int, C.c_int, C.c_int, C.c_int]
    lib.fy_flat_noise.restype = C.c_double
    lib.fy_apply_mask.argtypes = [f32p, f32p, f32p, f32p,
                                  C.c_int, C.c_int, C.c_int, C.c_float]
    lib.fy_apply_mask.restype = C.c_int
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
def md_phys_from_metadata(metadata: dict) -> dict:
    """Extract the physics dict the pipeline needs from a volume's metadata.json.

    ALWAYS prefer this over hardcoding: the metadata records the EXACT acquisition +
    nabu Paganin phase-retrieval parameters (energy, propagation distance, pixel size,
    delta/beta, unsharp) that we are partially inverting. Getting delta_beta wrong (e.g.
    assuming 1000 when nabu used 2000) mis-specifies the forward model and the deconv.

    Layout (ESRF/nabu export): scan.tomo.acquisition.{energy, sampleDetectorDistance,
    detector.samplePixelSize} and scan.tomo.processing.preprocessing.phase.{method,
    delta_beta, unsharp_coeff, unsharp_sigma}. Falls back across a couple of nestings."""
    scan = metadata.get("scan", {})
    tomo = scan.get("tomo", metadata.get("tomo", scan))
    acq = tomo.get("acquisition", {})
    det = acq.get("detector", {})
    proc = tomo.get("processing", {})
    phase = proc.get("preprocessing", {}).get("phase", {}) if proc else {}
    zx = metadata.get("zarr_export", {})
    # coalesce: use the default when the key is MISSING **or present-but-None** (incomplete
    # metadata). dict.get(k, default) only covers missing keys, not null values.
    def g(d, k, default):
        val = d.get(k, None)
        return default if val is None else val
    # samplePixelSize is in mm in the ESRF metadata -> convert to um
    px_mm = g(det, "samplePixelSize", None)
    md = {
        "energy_kev": float(g(acq, "energy", 78.0)),
        "distance_mm": float(g(acq, "sampleDetectorDistance", 220.0)),
        "pixel_um": float(px_mm * 1000.0) if px_mm else 2.4,
        "delta_beta": float(g(phase, "delta_beta", 1000.0)),
        "unsharp_sigma": float(g(phase, "unsharp_sigma", 1.2)),
        "unsharp_coeff": float(g(phase, "unsharp_coeff", 4.0)),
        "phase_method": phase.get("method"),
        # beam-current drift over the scan (used to gate z-drift correction)
        "machine_current_start": acq.get("machineCurrentStart"),
        "machine_current_stop": acq.get("machineCurrentStop"),
        # u8 export window (physical units): u8 = 255*(mu - wlo)/(whi - wlo). This is the
        # KEY to a physics-derived air threshold -- air's attenuation ~= the window floor.
        "window_f32_min": (float(zx["target_window_f32_min"])
                           if "target_window_f32_min" in zx else None),
        "window_f32_max": (float(zx["target_window_f32_max"])
                           if "target_window_f32_max" in zx else None),
    }
    # RECON-MEASURED physical distribution (nabu's own histogram, pre-windowing) -- the REAL
    # air/material range, better than the nominal window for a physics-derived class boundary.
    h32 = proc.get("32bitsData", {}).get("histogram", {}) if proc else {}
    if h32:
        md["recon_air_floor"] = h32.get("min_0p002_percentile")        # ~0 = air baseline
        md["recon_material_p998"] = h32.get("max_0p998_percentile")    # densest material
    conv = proc.get("postprocessing", {}).get("32BitsConversion", {}) if proc else {}
    if conv:
        md["recon_used_min"] = conv.get("dataset_used_min")
        md["recon_used_max"] = conv.get("dataset_used_max")
    # reconstruction method (GHBP != generic FBP -> its filter has its own transfer fn)
    md["recon_method"] = proc.get("reconstruction", {}).get("method") if proc else None
    return md


def air_thresh_from_physics(md_phys: dict) -> float | None:
    """Physics-derived air threshold in [0,1] (u8/255), or None if the window is unknown.

    The u8 export maps physical attenuation mu linearly: u8 = 255*(mu - wlo)/(whi - wlo),
    with (wlo, whi) = the metadata zarr_export window. Air's linear attenuation is ~0, and
    the window FLOOR (wlo) is set right at the air/material baseline -- so air clips to the
    very bottom of the u8 range. The air/papyrus boundary sits a small margin above the
    floor. We place the threshold a few percent of the window span above the floor.

    This is principled where Otsu is NOT: Otsu on a mostly-material tile finds a within-
    papyrus split (measured: u8 93 on a dense corner, deep inside papyrus -> would mask
    real papyrus as air). The physics value is ~u8 5-15 (just above the air baseline)."""
    wlo, whi = md_phys.get("window_f32_min"), md_phys.get("window_f32_max")
    if wlo is None or whi is None or whi <= wlo:
        return None
    # air sits at the floor; the boundary to material is a small margin up the window span.
    # 5% of span above the floor is comfortably above air noise yet far below papyrus bulk.
    frac_air = 0.05
    return float(frac_air)  # since u8 = span_frac, and floor maps to 0 -> thresh = frac


def load_md_phys(zarr_root: str, backend=None) -> dict:
    """Read a volume's metadata.json (local dir or via an s3zarr backend) -> physics dict.

    If metadata.json EXISTS we ALWAYS use it (policy). Raises FileNotFoundError if the
    caller asked for metadata-derived physics but none is present, rather than silently
    falling back to hardcoded defaults (which masked a 2x delta_beta error before)."""
    import json as _json
    if backend is not None:  # s3zarr backend: metadata.json sits beside the level dirs
        txt = backend.get("metadata.json")
        if txt is None:
            raise FileNotFoundError("metadata.json not found in backend")
        meta = _json.loads(txt.decode("utf-8") if isinstance(txt, (bytes, bytearray)) else txt)
    else:
        p = os.path.join(zarr_root, "metadata.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"metadata.json not found at {p}")
        meta = _json.load(open(p))
    return md_phys_from_metadata(meta)


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
        self.air_thresh = air_thresh  # air threshold [0,1] (physics-derived preferred)
        self.air_thresh_physics = None  # set by calibrate() from the export window
        self.zdrift_factor = zdrift_factor  # per-z correction factor array (float32)
        # ---- TUNED per-stage config (set by calibrate_prepass; sensible defaults) ----
        self.do_deconv = True
        self.deconv_reg = -1.0          # <=0 -> kernel auto
        self.deconv_lo = 0.0            # GLOBAL deconv-output range for seam-safe rescale
        self.deconv_hi = 1.0            # (measured in the pre-pass; same for every tile)
        self.do_denoise = True
        self.denoise_radius = 2
        self.do_diffusion = False
        self.diffusion_strength = 2
        # ---- AIR-ZERO (throwaway-mask) config ----
        # Validated air separation: denoise a SCRATCH copy (iterated gentle bilateral) to
        # decide the air/papyrus mask, then ZERO air in the processed output (papyrus keeps
        # full processing). air_cut_u8 = the threshold on the scratch (set from the clean
        # scratch's fitted dark mode, dark_mu+0.5sigma); scratch_passes = bilateral iterations.
        self.do_air_zero = False
        self.air_cut_u8 = None        # u8 threshold on the denoised scratch (None -> derive)
        self.scratch_passes = 5
        self.tuning = {}                # per-stage chosen value + metric panel (for inspection)

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
    # NOTE: deconv runs BEFORE denoise and amplifies the noise ~3-4x. The base
    # fy_guided_eps_for_noise is calibrated to the RAW noise, so in-pipeline (post-deconv)
    # it under-denoises -> boost eps. NOTE this only seeds guided_eps_raw; the joint search
    # in calibrate_prepass picks the FINAL eps from its EPS_GRID via _metric_panel.
    # OPEN FINDING (2026-06-06): the whole-chain quality basket [[whole-pipeline-validation]]
    # shows the joint search OVER-DENOISES (picks eps~0.009; the noise<=1 & sharp>=1 knee is
    # eps~0.004-0.005, which keeps +20% contrast and stays net-sharp). Root cause: _metric_panel
    # rewards noise reduction more than the whole-chain basket. FIX = reconcile _metric_panel
    # with the basket (constrain noise<=raw, sharp>=1); not a one-line gain change (verified:
    # changing this gain doesn't move the joint-search result).
    POST_DECONV_NOISE_GAIN = 3.5
    guided_eps_raw = L.fy_guided_eps_for_noise(noise_ref)
    guided_eps = L.fy_guided_eps_for_noise(noise_ref * POST_DECONV_NOISE_GAIN)
    cal = Calibration(phys, db_scale, noise_ref, guided_eps, halo, window,
                      guided_eps_raw=guided_eps_raw)
    # PHYSICS-DERIVED air threshold from the export window (preferred over Otsu, which on a
    # mostly-material tile finds a within-papyrus split). None if the window is unknown.
    cal.air_thresh_physics = air_thresh_from_physics(md_phys)
    return cal


# ---------------------------------------------------------------- the per-tile chain
# ---------------------------------------------------------------- calibration pre-pass
def _metric_panel(ref, out, ref_cache=None):
    """Score a processed tile `out` vs the (deconvolved) reference `ref`, both float
    [0,1]-ish, across a BUCKET of metrics. Returns a dict of normalized sub-scores in
    [0,1] (1 = good) plus the raw numbers. Roughly-equal-weight by design; hard
    constraints are flagged separately. Metrics span the ways processing helps or harms:
    detail-retention, noise, contrast, sheet/gap structure, and safety.

    ref_cache: optional dict. The reference-only quantities (its radial PSD, texture std,
    flat-noise level) depend solely on `ref`, which is CONSTANT across a parameter sweep on
    the same tile -- pass a per-tile dict to compute them once and reuse, ~halving the FFT
    work during calibration. Safe to omit (computed fresh)."""
    from numpy.fft import fftn, fftshift
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None
    def rpsd(v):
        v = v - v.mean(); F = np.abs(fftshift(fftn(v))) ** 2
        n = v.shape[0]; c = n // 2
        z, y, x = np.indices(v.shape) - c; r = np.sqrt(z*z + y*y + x*x).astype(int)
        return (np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[:c]
    def hp(v): return v - gaussian_filter(v, 1.0) if gaussian_filter else v - v.mean()
    def flatnoise(v):
        if gaussian_filter is None: return float(v.std())
        s = []
        for z in range(0, v.shape[0]-16, 16):
            for y in range(0, v.shape[1]-16, 16):
                for x in range(0, v.shape[2]-16, 16):
                    p = v[z:z+16, y:y+16, x:x+16]
                    if (p > 0.06).mean() > 0.9 and p.std() < 0.1:
                        s.append(hp(p).std())
        return float(np.median(s)) if s else float(hp(v).std())
    # reference-only quantities: compute once per tile, reuse across the sweep.
    if ref_cache is not None and "pr" in ref_cache:
        pr = ref_cache["pr"]; tex_r = ref_cache["tex_r"]; n_r = ref_cache["n_r"]
    else:
        pr = rpsd(ref); tex_r = float(hp(ref).std()); n_r = flatnoise(ref)
        if ref_cache is not None:
            ref_cache["pr"] = pr; ref_cache["tex_r"] = tex_r; ref_cache["n_r"] = n_r
    po = rpsd(out); nyq = len(pr) - 1
    def band(p, a, b): return p[int(a*nyq):int(b*nyq)].sum() + 1e-12
    midret = band(po, 0.17, 0.5) / band(pr, 0.17, 0.5)     # >=1 keeps texture band
    hiret = band(po, 0.5, 1.0) / band(pr, 0.5, 1.0)
    tex_o = float(hp(out).std())
    n_o = flatnoise(out)
    noise_ratio = n_o / (n_r + 1e-9)                        # <1 = denoised
    tex_ratio = tex_o / (tex_r + 1e-9)
    legibility = (tex_o / (n_o + 1e-9)) / (tex_r / (n_r + 1e-9) + 1e-9)
    clip = float((out <= 0.001).mean() + (out >= 0.999).mean())
    finite = bool(np.isfinite(out).all())
    # A "good" result = MORE legible (texture rises relative to noise) + texture/detail
    # not collapsed + no clipping. Several metrics, roughly equal weight. The bucket is
    # built so that REMOVING NOISE (which lowers raw mid-band power) is rewarded, not
    # penalized -- the detail guard is on TEXTURE structure, not raw spectral power.
    sub = {
        "legibility":  min(legibility / 3.0, 1.0),          # tex/noise gain (the headline)
        "denoise":     min(max(0.0, 1.0 - noise_ratio) / 0.5, 1.0),  # noise removed
        "texture_keep": min(tex_ratio / 0.8, 1.0),          # keep texture (1.0 by tex_ratio>=0.8)
        "contrast":    min(midret / 1.5, 1.0) if midret > 1 else midret,  # contrast restored (deconv)
        "no_clip":     max(0.0, 1.0 - clip * 8.0),
    }
    score = sum(sub.values()) / len(sub) if finite else 0.0
    # HARD CONSTRAINTS: reject only genuinely BAD outcomes -- legibility must IMPROVE
    # (>1.0) and texture must not COLLAPSE (tex_ratio not tiny) and no clip pile-up.
    # This accepts processing that helps (45um forced-on: legibility 2.53->7.23) while
    # rejecting over-smoothing (eps=0.78: texture 0.19x -> tex_ratio fails).
    ok = (finite and legibility >= 1.05 and tex_ratio >= 0.55 and clip < 0.08)
    # SAFETY-only subset (finite + no clip pile-up + texture not OBLITERATED): used by the
    # quality-basket-reconciled joint search, which owns the QUALITY decision. The full `ok`
    # above (legibility>=1.05, tex>=0.55) is the legacy quality gate -- too strict for coarse
    # volumes (45um: legibility 1.04<1.05 rejected everything -> all OFF). The basket decides
    # quality; the panel only vetoes genuinely unsafe output here.
    safe = (finite and clip < 0.08 and tex_ratio >= 0.30)
    return {"score": score, "ok": ok, "safe": safe, "sub": sub,
            "raw": {"midret": midret, "hiret": hiret, "noise_ratio": noise_ratio,
                    "tex_ratio": tex_ratio, "legibility": legibility, "clip": clip}}


def select_sample_tiles(read_region, shape, n=6, tile=128, candidates=24,
                        min_occupancy=0.6, seed=0):
    """SMART SAMPLING: pick the `n` most textured, representative tiles for calibration.

    The pre-pass is only as good as its samples -- a few fixed coords can land on air or
    an atypical region and skew every downstream pick. We probe `candidates` deterministic
    locations across the volume, keep those with enough papyrus (occupancy >= min), rank by
    TEXTURE ENERGY (high-pass std -- the signal calibration actually optimizes), and return
    a spread of the most textured tiles. Deterministic (seeded) so calibration is
    reproducible. read_region(z,y,x,dz,dy,dx)->u8; returns list of u8 tiles."""
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None
    Z, Y, X = shape
    # deterministic quasi-grid of candidate origins, biased toward the volume interior
    rng = np.random.RandomState(seed)
    g = max(1, int(round(candidates ** (1.0 / 3.0))))
    origins = []
    for iz in range(g):
        for iy in range(g):
            for ix in range(g):
                fz = (iz + 0.5) / g; fy = (iy + 0.5) / g; fx = (ix + 0.5) / g
                z0 = int(np.clip(fz * (Z - tile), 0, max(0, Z - tile)))
                y0 = int(np.clip(fy * (Y - tile), 0, max(0, Y - tile)))
                x0 = int(np.clip(fx * (X - tile), 0, max(0, X - tile)))
                origins.append((z0, y0, x0))
    # small deterministic jitter so we don't always sit on chunk boundaries
    scored = []
    for (z0, y0, x0) in origins:
        jz = int(rng.randint(0, max(1, tile // 4)))
        z0 = min(z0 + jz, max(0, Z - tile))
        blk = read_region(z0, y0, x0, min(tile, Z - z0), min(tile, Y - y0), min(tile, X - x0))
        occ = float((blk > 0).mean())
        if occ < min_occupancy:
            continue
        v = blk.astype(np.float32) / 255.0
        hp = (v - gaussian_filter(v, 1.0)) if gaussian_filter else (v - v.mean())
        tex = float(hp.std())
        scored.append((tex, occ, (z0, y0, x0), blk))
    if not scored:  # nothing textured enough -> fall back to the most-occupied probes
        return []
    scored.sort(key=lambda r: r[0], reverse=True)
    # take the top-textured tiles but spread across the ranked list so they aren't all
    # from one hot spot (take every k-th of the top 2n)
    top = scored[: max(n, 2 * n)]
    step = max(1, len(top) // n)
    picked = [top[i] for i in range(0, len(top), step)][:n]
    return [p[3] for p in picked]


def calibrate_prepass(md_phys: dict, sample_tiles, auto_deltabeta=True, verbose=False,
                      allow_diffusion=False, refine=True):
    """CALIBRATION PRE-PASS: measure the volume on sample tiles and TUNE the whole chain
    across a BUCKET of metrics (roughly equal weight, hard safety constraints).

    Strategy (improved over the old greedy stage-by-stage sweep):
      * MEMOIZED: the expensive deconv FFT is computed once per (tile, reg) and reused
        across the sweep, the rescale-range measurement, and the joint denoise search --
        the old code recomputed each deconv ~3x.
      * JOINT (deconv x denoise): instead of locking deconv then denoising on top, we score
        the CHAIN end-to-end over a grid of (reg, eps) pairs. The best pair is often not
        best-deconv-alone + best-denoise-after (a stronger deconv can pay for a stronger
        denoise). OFF for each stage is just reg/eps = none in the same grid.
      * REFINED: after the coarse grid we bisect once around the winning reg and eps to
        land between grid points.
      * EXPLAINABLE: cal.tuning records, per stage, the winner, the runner-up margin, and
        -- when a stage is OFF -- WHY (which hard constraint failed on the best candidate).

    This stays RESOLUTION-ADAPTIVE by measurement: coarse/noisy 45um -> deconv-only (denoise
    candidates fail detail-retention); fine 2.4um -> deconv + gentle denoise. sample_tiles:
    list of uint8 textured tiles (>=64^3); use select_sample_tiles() to pick good ones.

    Returns a fully-tuned Calibration ready for the streaming pass."""
    L = lib()
    # base calibration (physics, db_scale, halo, noise_ref)
    samp01 = [t.astype(np.float32) / 255.0 for t in sample_tiles]
    cal = calibrate(md_phys, samp01, auto_deltabeta=auto_deltabeta)
    f32p = C.POINTER(C.c_float)
    raw = samp01

    # -- MEMOIZED deconv: one FFT per (tile-index, reg), reused everywhere. The raw (un-
    #    rescaled) output is cached; rescale is a cheap pure-numpy transform on top. --
    _dec_cache = {}
    def deconv_raw(ti, reg):
        key = (ti, round(reg, 6))
        o = _dec_cache.get(key)
        if o is None:
            a = np.ascontiguousarray(raw[ti], np.float32); o = np.empty_like(a)
            ph = cal.scaled_phys()
            L.fy_deconvolve(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p),
                            *a.shape, C.byref(ph), C.c_double(reg))
            _dec_cache[key] = o
        return o
    def rescale01(o, lo=None, hi=None):
        if lo is None:
            lo, hi = np.percentile(o, 0.1), np.percentile(o, 99.9)
        if hi - lo > 1e-6:
            return np.clip((o - lo) / (hi - lo), 0, 1)
        return o
    def guided(v, eps):
        a = np.ascontiguousarray(v, np.float32); o = np.empty_like(a)
        L.fy_guided_denoise(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape, 2, eps)
        return o
    def diffuse(v, s):
        a = np.ascontiguousarray(v, np.float32); o = np.empty_like(a)
        L.fy_coherence_diffusion_auto(a.ctypes.data_as(f32p), o.ctypes.data_as(f32p), *a.shape, int(s))
        return o

    # ---- WHOLE-CHAIN QUALITY BASKET helpers (C-backed; reconciles the calibration objective
    #      with the validated quality basket -- noise floor & sheet sharpness, in physical
    #      units the basket uses). See [[whole-pipeline-validation]]. ----
    def _edge_sharp(v):
        a = np.ascontiguousarray(v, np.float32)
        return float(L.fy_edge_sharpness(a.ctypes.data_as(f32p), *a.shape))
    def _flat_noise(v):
        a = np.ascontiguousarray(v, np.float32)
        return float(L.fy_flat_noise(a.ctypes.data_as(f32p), *a.shape, 8))
    def _midband(v):
        from numpy.fft import fftn, fftshift
        x = v - v.mean(); F = np.abs(fftshift(fftn(x))) ** 2; n = v.shape[0]; c = n // 2
        z, y, xx = np.indices(v.shape) - c; r = np.sqrt(z*z + y*y + xx*xx).astype(int)
        rp = (np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[:c]
        nyq = len(rp) - 1; return rp[int(0.17*nyq):int(0.5*nyq)].sum() + 1e-12
    # per-tile raw references for the quality ratios (computed once)
    _raw_noise = [_flat_noise(t) for t in raw]
    _raw_sharp = [_edge_sharp(t) for t in raw]
    _raw_mid = [_midband(t) for t in raw]
    def quality_basket(out_tiles):
        """Validated whole-chain quality score over the sample tiles. Returns (score, ok):
        HARD CONSTRAINTS noise<=1.05*raw (deconv amplification undone) AND sharp>=0.98*raw
        (not net-blurred); OBJECTIVE = mean mid-band contrast ratio (maximize). Mirrors the
        constrained joint tune that picked reg=0.15/eps~0.004 over the over-denoised 0.009."""
        noises, sharps, contrs = [], [], []
        for i, o in enumerate(out_tiles):
            noises.append(_flat_noise(o) / (_raw_noise[i] + 1e-9))
            sharps.append(_edge_sharp(o) / (_raw_sharp[i] + 1e-9))
            contrs.append(_midband(o) / (_raw_mid[i] + 1e-9))
        noise = float(np.median(noises)); sharp = float(np.median(sharps)); contr = float(np.median(contrs))
        # CONSTRAINTS (resolution-robust, multi-source validated 1.1-45um). Two failure modes
        # to avoid: (a) hard noise<=raw rejected good deconv on fine/noisy AND coarse volumes
        # (-> everything OFF); (b) pure legibility>=1 let deconv-ONLY win everywhere (noise
        # 1.76x, no denoise). Balance: deconv may amplify noise, but only PROPORTIONAL to the
        # contrast it buys -- noise ceiling = 1 + 0.5*(contrast-1), capped at 2.0. Reject net-
        # blur (sharp<0.95). OBJECTIVE (caller) = legibility (contrast/noise), so among passing
        # cells it PREFERS the denoised one (controls noise) over noisy deconv-only.
        noise_ceiling = min(1.0 + 0.5 * max(contr - 1.0, 0.0), 2.0)
        legibility = contr / max(noise, 1e-6)
        ok = (sharp >= 0.95 and noise <= noise_ceiling and contr >= 1.0)
        return legibility, ok, dict(noise=noise, sharp=sharp, contrast=contr, legibility=legibility)

    tuning = {}

    # ---------- JOINT (deconv reg x denoise eps) search over the full chain ----------
    # candidate axes. reg=None -> deconv OFF; eps=0 -> denoise OFF (both reachable here).
    REG_GRID = [None, -1.0, 0.005, 0.015, 0.05, 0.15]   # None=off, -1=kernel-auto
    eps_base = cal.guided_eps_raw
    EPS_GRID = [0.0] + [eps_base * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]

    # one reference-PSD/texture/noise cache per tile (ref = raw[ti], constant across cells)
    _ref_caches = [{} for _ in raw]
    def chain_score(reg, eps):
        """End-to-end score for the (reg, eps) chain. RECONCILED with the validated whole-chain
        quality basket: the OBJECTIVE is the quality basket's mid-band contrast (maximize), and
        a candidate is `ok` only if it passes BOTH the panel's safety constraints (legibility,
        texture-not-collapsed, no clip) AND the basket's quality constraints (noise<=raw,
        sharp>=raw). This stops the old over-denoising: panel-alone rewarded noise reduction and
        picked eps~0.009 (net-blur); the basket constraint rejects net-blur -> picks eps~0.004-5."""
        outs, per = [], []
        for ti in range(len(raw)):
            cur = raw[ti]
            if reg is not None:
                cur = rescale01(deconv_raw(ti, reg))
            if eps > 0:
                cur = guided(cur, eps)
            outs.append(cur)
            per.append(_metric_panel(raw[ti], cur, ref_cache=_ref_caches[ti]))
        panel_safe = all(m["safe"] for m in per)   # safety only (finite/no-clip/tex-not-gone)
        sc, basket_ok, qm = quality_basket(outs)    # quality OWNED by the basket; sc=legibility
        # ok = panel SAFETY (not the legacy quality floor) AND basket quality. The basket's
        # contrast/noise/sharp constraints make the quality call; the panel only vetoes unsafe.
        ok = panel_safe and basket_ok
        for m in per:
            m["quality"] = qm
        return sc, ok, per

    # COARSE-TO-FINE joint search (cheaper than the full REGxEPS grid while still JOINT):
    #   stage A: score every reg at a couple of probe eps (off + one mid) to rank the regs
    #            jointly (a reg that only shines WITH denoise still ranks here).
    #   stage B: fully sweep eps only at the top-K regs.
    # This keeps the joint optimum reachable (best (reg,eps) need not be best-reg-alone)
    # but evaluates ~|REG|*2 + K*|EPS| cells instead of |REG|*|EPS|.
    results = []  # (score, ok, reg, eps, per)
    seen = set()
    def _eval(reg, eps):
        k = (reg, round(eps, 8))
        if k in seen: return
        seen.add(k)
        sc, ok, per = chain_score(reg, eps)
        results.append((sc, ok, reg, eps, per))
        if verbose:
            rg = "off" if reg is None else f"{reg:g}"
            print(f"  reg={rg:>5} eps={eps:.4f}: score={sc:.3f} ok={ok}")
    probe_eps = [0.0]
    if len(EPS_GRID) > 1:
        probe_eps.append(EPS_GRID[len(EPS_GRID) // 2])  # a representative mid eps
    for reg in REG_GRID:
        for eps in probe_eps:
            _eval(reg, eps)
    # rank regs by their best probe score; fully sweep eps at the top-K (K=3 incl. off-reg)
    reg_best = {}
    for sc, ok, reg, eps, per in results:
        if reg not in reg_best or sc > reg_best[reg]:
            reg_best[reg] = sc
    top_regs = [r for r, _ in sorted(reg_best.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    for reg in top_regs:
        for eps in EPS_GRID:
            _eval(reg, eps)
    ok_cells = [r for r in results if r[1]]
    baseline = chain_score(None, 0.0)[0]  # identity score = panel(raw,raw)
    if ok_cells:
        ok_cells.sort(key=lambda r: r[0], reverse=True)
        best = ok_cells[0]
        runner = ok_cells[1] if len(ok_cells) > 1 else None
    else:
        # nothing passed the hard constraints -> everything OFF; explain using the best-
        # scoring (constraint-failing) candidate so "all off" is never a mystery.
        results.sort(key=lambda r: r[0], reverse=True)
        best = (baseline, True, None, 0.0, None)
        runner = None
        worst_cand = results[0]
        # find which constraint failed on that top candidate (first failing tile)
        why = "no candidate beat the safety constraints"
        if worst_cand[4]:
            for m in worst_cand[4]:
                if not m["ok"]:
                    rr = m["raw"]
                    fails = []
                    if rr["legibility"] < 1.05: fails.append(f"legibility {rr['legibility']:.2f}<1.05")
                    if rr["tex_ratio"] < 0.55: fails.append(f"tex_ratio {rr['tex_ratio']:.2f}<0.55")
                    if rr["clip"] >= 0.08:      fails.append(f"clip {rr['clip']:.3f}>=0.08")
                    why = "; ".join(fails) or "non-finite output"
                    break
        tuning["all_off_reason"] = why
        if verbose: print(f"  ALL STAGES OFF -- best candidate failed: {why}")

    best_reg, best_eps = best[2], best[3]

    # ---------- REFINEMENT: bisect once around the winning reg and eps ----------
    if refine and ok_cells:
        # refine reg: try the midpoints to the neighbours on the (sorted-numeric) reg axis
        if best_reg is not None and best_reg > 0:
            numeric_regs = sorted(r for r in REG_GRID if isinstance(r, float) and r > 0)
            i = numeric_regs.index(best_reg)
            neigh = []
            if i > 0: neigh.append((numeric_regs[i-1] + best_reg) / 2)
            if i < len(numeric_regs) - 1: neigh.append((best_reg + numeric_regs[i+1]) / 2)
            for rg in neigh:
                sc, ok, per = chain_score(rg, best_eps)
                if verbose: print(f"  refine reg={rg:.4f} eps={best_eps:.4f}: score={sc:.3f} ok={ok}")
                if ok and sc > best[0]:
                    best = (sc, ok, rg, best_eps, per); best_reg = rg
        # refine eps similarly
        if best_eps > 0:
            pos_eps = sorted(e for e in EPS_GRID if e > 0)
            i = pos_eps.index(best_eps)
            neigh = []
            if i > 0: neigh.append((pos_eps[i-1] + best_eps) / 2)
            if i < len(pos_eps) - 1: neigh.append((best_eps + pos_eps[i+1]) / 2)
            for ep in neigh:
                sc, ok, per = chain_score(best_reg, ep)
                if verbose: print(f"  refine reg={best_reg} eps={ep:.4f}: score={sc:.3f} ok={ok}")
                if ok and sc > best[0]:
                    best = (sc, ok, best_reg, ep, per); best_eps = ep

    # ---------- commit deconv ----------
    cal.do_deconv = best_reg is not None
    cal.deconv_reg = best_reg if best_reg is not None else -1.0
    margin = (best[0] - runner[0]) if runner else None
    tuning["deconv"] = {"on": cal.do_deconv, "reg": cal.deconv_reg,
                        "chain_score": best[0], "off_score": baseline,
                        "runner_up_margin": margin}

    # GLOBAL deconv-output rescale (seam-safe): measure the deconv output's robust range
    # ONCE across the samples (using the memoized raw deconv at the chosen reg) and rescale
    # EVERY tile by the SAME [lo,hi] so no data is clipped AND all tiles map identically.
    if cal.do_deconv:
        vals = np.concatenate([deconv_raw(ti, cal.deconv_reg).ravel() for ti in range(len(raw))])
        cal.deconv_lo = float(np.percentile(vals, 0.1))
        cal.deconv_hi = float(np.percentile(vals, 99.9))
        if cal.deconv_hi - cal.deconv_lo < 1e-6:
            cal.deconv_lo, cal.deconv_hi = 0.0, 1.0
        tuning["deconv"]["rescale"] = [cal.deconv_lo, cal.deconv_hi]

    # ---------- commit denoise ----------
    cal.do_denoise = best_eps > 0
    cal.guided_eps = best_eps if best_eps > 0 else 0.0
    tuning["denoise"] = {"on": cal.do_denoise, "eps": cal.guided_eps, "chain_score": best[0]}

    # working set after the chosen deconv+denoise (diffusion operates on this)
    decd = [rescale01(deconv_raw(ti, cal.deconv_reg)) if cal.do_deconv else raw[ti]
            for ti in range(len(raw))]
    dnd = [guided(t, cal.guided_eps) if cal.do_denoise else t for t in decd]

    # ---- 3. DIFFUSION (clean sheets). NOTE: coherence diffusion is ITERATIVE -- its
    #         domain of dependence grows per iteration beyond a fixed tile halo, so it is
    #         NOT cleanly tileable / seam-free in the streaming pipeline (verified: tiled
    #         vs whole mismatches by ~30 u8). It is therefore DISABLED in the streaming
    #         pre-pass by default; use it as a standalone (whole-region) filter where
    #         seams don't matter. Set allow_diffusion=True only if you accept tile seams
    #         or run it un-tiled. ----
    if allow_diffusion:
        def _diff_score(fn):
            per = [_metric_panel(t, fn(t)) for t in dnd]
            return float(np.mean([m["score"] for m in per])), all(m["ok"] for m in per)
        diff_best = (0, _diff_score(lambda t: t)[0])
        for s in [1, 2]:
            sc, ok = _diff_score(lambda t, ss=s: diffuse(t, ss))
            if verbose: print(f"  diffusion strength={s}: score={sc:.3f} ok={ok}")
            if ok and sc > diff_best[1] * 1.02:
                diff_best = (s, sc)
        cal.do_diffusion = diff_best[0] > 0; cal.diffusion_strength = diff_best[0] or 2
    else:
        cal.do_diffusion = False; cal.diffusion_strength = 2
    tuning["diffusion"] = {"on": cal.do_diffusion, "strength": cal.diffusion_strength,
                           "note": "disabled in streaming (not seam-free); standalone only"}

    # ---- HALO = max over ENABLED stages (seam-freeness depends on the LARGEST reach).
    # deconv halo is cal.halo (already set); coherence diffusion needs a MUCH bigger halo
    # (3*sigma+3*rho+n_iters ~ 26-49 vox). Use the diffusion halo helper for a safe upper
    # bound when diffusion is on, else the deconv halo. ----
    halo = cal.halo
    if cal.do_diffusion:
        iters = {1: 12, 2: 24, 3: 40}.get(cal.diffusion_strength, 24)
        # auto-CED uses sigma up to 1.5, rho up to 6 -> use the worst case for safety
        dh = L.fy_coherence_diffusion_halo(1.5, 6.0, iters)
        halo = max(halo, int(dh))
    cal.halo = halo
    tuning["halo"] = halo

    cal.tuning = tuning
    if verbose:
        print(f"TUNED: deconv={cal.do_deconv}(reg={cal.deconv_reg}) "
              f"denoise={cal.do_denoise}(eps={cal.guided_eps:.4f}) "
              f"diffusion={cal.do_diffusion}(s={cal.diffusion_strength})")
    return cal


def process_tile(block_u8: np.ndarray, cal: Calibration,
                 do_deconv=None, do_denoise=None, do_diffusion=None,
                 diffusion_strength=None, do_mask=None) -> np.ndarray:
    """Run the full chain on one tile (with halo). Input/output uint8.

    Order: u8 -> [phys] -> deconv -> denoise -> [diffusion] -> [air-mask] -> u8. The
    per-stage on/off and STRENGTHS come from the (calibrated) `cal` by default -- pass
    explicit args only to override. The HALO is part of block_u8; the caller crops the
    inner tile after.

    AIR MASKING (do_mask): papyrus is bright+textured, air is dark+flat. After processing,
    blend the PROCESSED result on papyrus with the ORIGINAL (un-sharpened) tile in air, so
    deconv/denoise don't amplify noise in empty gaps. Gated on cal.air_thresh (set by the
    pass-1 finalize); the mask is built per-tile and is local (radius-bounded) so it stays
    seam-safe. Default ON whenever an air threshold is available."""
    L = lib()
    nz, ny, nx = block_u8.shape
    # use the TUNED config unless explicitly overridden
    do_deconv = cal.do_deconv if do_deconv is None else do_deconv
    do_denoise = cal.do_denoise if do_denoise is None else do_denoise
    do_diffusion = cal.do_diffusion if do_diffusion is None else do_diffusion
    diffusion_strength = cal.diffusion_strength if diffusion_strength is None else diffusion_strength
    if do_mask is None:
        do_mask = cal.air_thresh is not None

    orig = (block_u8.astype(np.float32) / 255.0)   # keep for the air blend
    cur = orig

    if do_deconv:
        inp, ip = _fp(cur)
        out = np.empty_like(inp); op = out.ctypes.data_as(C.POINTER(C.c_float))
        ph = cal.scaled_phys()
        rc = L.fy_deconvolve(ip, op, nz, ny, nx, C.byref(ph), C.c_double(cal.deconv_reg))
        if rc != 0:
            raise RuntimeError("fy_deconvolve failed")
        # seam-safe GLOBAL rescale of the deconv overshoot to [0,1] (don't hard-clip)
        if cal.deconv_hi - cal.deconv_lo > 1e-6:
            out = (out - cal.deconv_lo) / (cal.deconv_hi - cal.deconv_lo)
        # NOTE: the export window-clips the u8 (saturated material -> 255, masked/below-window
        # -> 0). Those rails are not true intensities, and deconv rings against the clipped
        # plateau edge (~50% of voxels in the 3-vox ring around a saturated voxel overshoot to
        # >0.99, vs 0.01% far away). MEASURED but NOT guarded: it touches only ~0.05% of voxels,
        # and a clean fix needs clip-inpainting before deconv (not a local cap -- the local
        # values are themselves railed, so any cap is a no-op in the ring). Documented as a
        # known limitation in memory [[zarr-export-baked-processing]] rather than half-fixed.
        cur = out

    if do_denoise and cal.guided_eps > 0:
        inp, ip = _fp(cur)
        out = np.empty_like(inp); op = out.ctypes.data_as(C.POINTER(C.c_float))
        rc = L.fy_guided_denoise(ip, op, nz, ny, nx, cal.denoise_radius, cal.guided_eps)
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

    # AIR-ZERO (throwaway-mask). Air noise and papyrus fiber texture are spectrally
    # ENTANGLED (validated: texture Fisher ~0), so no denoiser separates them on the kept
    # data without destroying papyrus. The fix: denoise a SCRATCH copy (iterated gentle
    # bilateral -- deepens the air|papyrus histogram valley, de-saturates) ONLY to DECIDE
    # the mask, then ZERO air in the PROCESSED output. The kept papyrus keeps full deconv+
    # denoise detail (the scratch smoothing is discarded). Threshold from the clean scratch's
    # dark mode (or cal.air_cut_u8). Validated 128^3: papyrus texture preserved, near-0 specks.
    if cal.do_air_zero:
        # SCRATCH denoise to decide the mask. Use the GUIDED filter (O(N) box-based), NOT
        # bilateral: guided x5 is ~28x faster (0.7s vs 20s/128^3) and finds the SAME dark
        # mode (u8 53), which is all the threshold needs. The scratch is discarded -- only
        # the mask DECISION matters, so guided's slightly shallower valley is irrelevant.
        scratch = orig.copy()
        for _ in range(int(cal.scratch_passes)):
            si, sp = _fp(scratch)
            so = np.empty_like(si); sop = so.ctypes.data_as(C.POINTER(C.c_float))
            if L.fy_guided_denoise(sp, sop, nz, ny, nx, 2, C.c_double(0.01)) != 0:
                break
            scratch = so
        # threshold: explicit air_cut_u8, else the clean scratch's VALLEY (the histogram
        # THRESHOLD PRIORITY: (1) explicit cal.air_cut_u8 (e.g. a GLOBAL volume-wide cut --
        # preferred; consistent across all chunks), else (2) anchor to the scratch's DARK MODE
        # + a small margin. We anchor to the DARK MODE, NOT the valley: per-128^3-chunk the
        # valley is UNSTABLE (it depends on each chunk's papyrus/air RATIO -- measured 55..95
        # across chunks of one volume), and when it lands high (95, deep in papyrus) it ZEROS
        # REAL PAPYRUS. The dark mode is STABLE (air/void attenuation is constant: 48-49 across
        # the same chunks) so dark+margin gives a consistent, papyrus-safe cut everywhere.
        if cal.air_cut_u8 is not None:
            cut = int(cal.air_cut_u8)              # global / explicit -> consistent
        else:
            su8 = np.ascontiguousarray(np.clip(scratch * 255 + 0.5, 0, 255).astype(np.uint8))
            hist = np.bincount(su8.ravel(), minlength=256).astype(np.int64)
            dark = C.c_int(0); light = C.c_int(0); valley = C.c_int(0)
            d = L.fy_valley_depth(hist.ctypes.data_as(C.POINTER(C.c_long)),
                                  C.byref(dark), C.byref(light), C.byref(valley))
            if d >= 0:
                # cut just above the dark 'other' mode; cap WELL below the valley so we never
                # reach into papyrus even when the (unstable) valley sits high.
                cut = min(dark.value + 8, (dark.value + valley.value) // 2)
            else:
                cut = int((cal.air_thresh or 0.05) * 255)
        air = scratch < (cut / 255.0)        # decided on the clean scratch
        cur = np.where(air, 0.0, cur)         # zero air in the PROCESSED output

    # back to u8
    cur = np.clip(cur * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cur


# ---------------------------------------------------------------- interactive / vc3d API
def calibrate_for_volume(metadata: dict, sample_chunk_u8: np.ndarray) -> Calibration:
    """Compute the per-volume Calibration ONCE (call this when a volume is opened), then reuse
    it for every chunk via preprocess_chunk(). `metadata` = the volume's metadata.json (dict);
    physics is derived from it (delta_beta/energy/distance/pixel/window). `sample_chunk_u8` = a
    representative occupied chunk (e.g. a 128^3 from the volume center) used to tune the chain.
    Air-zero is enabled by default (cut at the histogram valley, auto-derived)."""
    md = md_phys_from_metadata(metadata)
    cal = calibrate_prepass(md, [np.ascontiguousarray(sample_chunk_u8, np.uint8)], verbose=False)
    cal.air_thresh = air_thresh_from_physics(md)
    cal.do_air_zero = cal.air_thresh is not None
    return cal


def preprocess_chunk(chunk_u8: np.ndarray, cal: Calibration) -> np.ndarray:
    """Preprocess ONE chunk (e.g. vc3d's bare 32^3) -> u8. Interactive: ~3-8ms per 32^3.

    SEAMS: this processes the chunk WITHOUT a halo, so the chunk INTERIOR is fully correct
    (contrast restored, papyrus preserved, 0% core loss; median voxel matches the seam-free
    result exactly) but a thin shell at chunk boundaries differs by ~15 u8 -> faint seam lines
    between adjacent chunks. For interactive viewing that's acceptable. For SEAM-FREE output
    (segmentation/measurement), feed a halo-padded block instead (32^3 + ~cal.halo on each
    side, ~96^3 at 2.4um) and keep the inner 32^3. Same call either way -- just pad the input."""
    return process_tile(np.ascontiguousarray(chunk_u8, np.uint8), cal)


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
                            want_norm=True, want_zdrift=True, want_air_cut=True,
                            norm_lo_pct=0.5, norm_hi_pct=99.5,
                            zdrift_min_frac=0.05,
                            progress=lambda *_: None):
    """PASS 1 (streaming, cheap): accumulate WHOLE-VOLUME statistics so pass 2 can apply
    a CONSISTENT global mapping -- the proper way to process a volume too big for RAM, and
    the FIX for per-chunk inconsistency (a per-128^3 air valley is unstable -> zeros real
    papyrus on dense chunks; a GLOBAL cut is consistent and papyrus-safe).

      - global histogram -> lo/hi percentiles for normalization
      - GLOBAL AIR CUT: histogram of the SCRATCH-DENOISED volume -> ONE dark-mode-anchored
        air threshold for the whole volume (-> cal.air_cut_u8, used by every pass-2 chunk).
      - per-z papyrus mean -> beam-current / shading DRIFT correction.
    State is tiny (two 256-bin histograms + 2 arrays of length Z). Mutates `cal` in place
    (sets norm_lo/hi, air_cut_u8, zdrift_factor). Reads NO halo -- plain tiling.
    """
    L = lib()
    f32p_ = C.POINTER(C.c_float)
    Z, Y, X = shape
    hist = _HistState(); L.fy_hist_init(C.byref(hist))
    air_hist = np.zeros(256, np.int64)   # histogram of scratch-denoised values (for the air cut)
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
                if want_air_cut and cal.do_air_zero:
                    # scratch-denoise this tile (same as process_tile's air-mask scratch) and
                    # accumulate its histogram -> the GLOBAL air cut is derived from the whole
                    # volume's denoised distribution, not each chunk's.
                    sc = np.ascontiguousarray(u8.astype(np.float32) / 255.0)
                    for _ in range(int(cal.scratch_passes)):
                        so = np.empty_like(sc)
                        if L.fy_guided_denoise(sc.ctypes.data_as(f32p_), so.ctypes.data_as(f32p_),
                                               *sc.shape, 2, C.c_double(0.01)) != 0:
                            break
                        sc = so
                    su8 = np.clip(sc * 255 + 0.5, 0, 255).astype(np.uint8)
                    air_hist += np.bincount(su8.ravel(), minlength=256).astype(np.int64)
                if want_zdrift:
                    f = np.ascontiguousarray(u8.astype(np.float32) / 255.0)
                    L.fy_zdrift_accumulate(f.ctypes.data_as(C.POINTER(C.c_float)),
                                           tz, ty, tx, z0, sums_p, counts_p, C.c_float(pap_thr))
    if want_norm and hist.total > 0:
        cal.norm_lo = int(L.fy_hist_percentile_u8(C.byref(hist), norm_lo_pct))
        cal.norm_hi = int(L.fy_hist_percentile_u8(C.byref(hist), norm_hi_pct))
        cal.air_thresh = (cal.air_thresh_physics if cal.air_thresh_physics is not None
                          else float(L.fy_auto_air_thresh(C.byref(hist))))
    # GLOBAL AIR CUT: derive ONE dark-mode-anchored cut from the whole-volume scratch
    # histogram -> every pass-2 chunk uses the SAME papyrus-safe threshold (fixes per-chunk
    # valley instability that zeroed real papyrus).
    if want_air_cut and cal.do_air_zero and air_hist.sum() > 0:
        ah = np.ascontiguousarray(air_hist)
        dark = C.c_int(0); light = C.c_int(0); valley = C.c_int(0)
        d = L.fy_valley_depth(ah.ctypes.data_as(C.POINTER(C.c_long)),
                              C.byref(dark), C.byref(light), C.byref(valley))
        if d >= 0:
            cal.air_cut_u8 = min(dark.value + 8, (dark.value + valley.value) // 2)
        else:
            cal.air_cut_u8 = int((cal.air_thresh or 0.05) * 255)
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
    L = lib()
    Z, Y, X = shape
    halo = cal.halo

    # ---- MEMORY-BUDGETED worker count (must never OOM) ----
    # The deconv is the RAM hog: a tile+halo FFT-pads to the next power of two and holds
    # re+im float32 buffers + input/output copies. Estimate per-worker peak and cap the
    # worker count to a fraction of FREE system RAM so N_workers * per_tile <= budget.
    def _next_pow2(x):
        p = 1
        while p < x:
            p *= 2
        return p
    pad = _next_pow2(tile + 2 * halo)
    # deconv: re+im (2) + a couple of work copies (~3) of pad^3 float32
    per_tile_gb = pad ** 3 * 4 * 5 / 1e9 if do_deconv else (tile + 2 * halo) ** 3 * 4 * 4 / 1e9
    try:
        with open("/proc/meminfo") as f:
            free_kb = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
        free_gb = free_kb / 1e6
    except Exception:
        free_gb = 8.0
    budget_gb = max(2.0, 0.6 * free_gb)            # use at most 60% of available RAM
    mem_cap = max(1, int(budget_gb / max(per_tile_gb, 0.05)))
    core_cap = max(1, (os.cpu_count() or 4) - 2)
    auto_workers = min(core_cap, mem_cap)
    if workers is None:
        workers = auto_workers
    else:
        # honor an explicit request but NEVER let it exceed the memory cap (no OOM)
        workers = min(workers, mem_cap)
    progress("workers", float(workers))

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
            cal.air_thresh = (cal.air_thresh_physics if cal.air_thresh_physics is not None
                              else float(L.fy_auto_air_thresh(C.byref(merged))))
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
