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
                 guided_eps: float, halo: int, window: tuple | None):
        self.phys = phys
        self.db_scale = db_scale
        self.noise_ref = noise_ref
        self.guided_eps = guided_eps
        self.halo = halo
        self.window = window  # (f32_min, f32_max) for u8<->phys, or None to keep u8 scale

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
    guided_eps = L.fy_guided_eps_for_noise(noise_ref)
    return Calibration(phys, db_scale, noise_ref, guided_eps, halo, window)


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
