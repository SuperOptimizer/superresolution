# superresolution

ML-based **restoration ("×1 super-resolution")** for Vesuvius Challenge scroll
volumes. The goal is to **deblur / denoise / recover attenuated high-frequency
signal at the native voxel grid** — *not* to expand the grid (no 1000³ → 2000³).
Input and target live on the **same grid**; the model inverts the acquisition MTF
rolloff and noise, filling attenuated frequency bands up to grid Nyquist with
structure that is **data-defined, not hallucinated**.

## Why this and not "make more voxels"

Your sampling grid is finer than your *effective* resolution (set by the imaging
MTF), so the high-frequency bins are attenuated, not empty. Restoration pushes
energy back into those bands. Going below the native voxel size onto a finer grid
would be pure invention with no ground truth to validate against — explicitly a
non-goal here.

Conservative by construction: **L1/Charbonnier loss, no adversarial/perceptual
term, optional data-consistency against the measured PSF, and a spectrum-overshoot
audit** that flags any invented high-frequency energy.

## Install

```bash
pip install -r requirements.txt
```

Core deps: `numpy scipy torch pyyaml pytest`. Optional: `matplotlib` (spectrum
plots), `boto3` (faster S3). The public S3 bucket needs no credentials; without
`boto3` we use the `aws` CLI with `--no-sign-request`.

## Quickstart (CPU, no network)

```bash
# 1. Run the test suite (all CPU, no network)
pytest -q

# 2. Train a few steps on synthetic data (seconds on CPU)
python scripts/train.py --config configs/smoke.yaml --smoke

# 3. Restore a synthetic degraded volume + spectrum audit
python scripts/infer.py --config configs/smoke.yaml \
    --ckpt checkpoints_smoke/latest.pt --synthetic --output restored.npy
```

The inference report prints `overshoot_ratio` (>~1.1 means the model is inventing
high-frequency energy) and `mean_log_gap` (lower = restored spectrum closer to the
clean target).

## Verify against the live data bucket

```bash
# Lists volumes/resolutions and reads one real 128³ chunk from S3
python scripts/probe_s3.py PHerc0500P2
```

## Data

Public bucket `s3://vesuvius-challenge-open-data` (us-east-1, CC BY-NC 4.0).
Layout: `{SCROLL}/volumes/{timestamp}-{res}um-...-masked.zarr/{level}/` — OME-Zarr
v0.4 multiscale, dtype `u8`, **128³ chunks**, uncompressed (raw bytes), so chunks
are fetchable directly with no zarr/boto3 dependency. `fill_value: 0` encodes the
zero-padding around the central ROI.

Same-scroll multi-resolution volumes exist (e.g. **PHerc0500P2** at
2.215 / 4.317 / 9.362 µm; **PHerc1203** at 2.403 / 9.362 µm). v1 trains
**synthetic-only** (degrade the finest volume, learn to invert); these real coarser
scans are reserved for spectrum validation / domain-gap checks in a later iteration.

## How it works

1. **Degradation operator** (`superres/degradation.py`) — the load-bearing piece.
   Turns a clean native-grid patch into a realistic same-grid degraded input:
   anisotropic Gaussian PSF (z ≠ xy), colored noise, sub-voxel shift, u8
   quantization, intensity jitter. Every parameter is **centered on the measured
   value with a modest spread** (`measure_psf_from_edge` calibrates from an
   edge-spread function). Get this right and the model transfers to real data; get
   it wrong and it inverts a fiction.
2. **Dataset** (`superres/data.py`) — lazily samples patches (S3 or synthetic),
   papyrus importance-sampling (skip air), robust percentile normalization,
   cubic-symmetry augmentation (full 48-group is safe because we degrade *after*
   augmenting, so the anisotropic PSF stays in the post-rotation frame).
3. **Model** (`superres/model.py`) — shallow **3-D residual U-Net** (~8–12M params
   at default width), residual prediction (`out = in + net(in)`, identity at init),
   GroupNorm. Shallow on purpose: deep pooling discards the high frequencies we want.
4. **Loss** (`superres/losses.py`) — Charbonnier (air-masked) + optional
   data-consistency (re-blur prediction through the known PSF, match the input,
   bounding the net to the operator null space).
5. **Inference** (`superres/infer.py`) — tiled streaming with cosine-feather
   overlap-blend (no seams) and occupancy-skip for all-zero tiles.
6. **Validation** (`superres/spectrum.py`) — radially-averaged power spectrum;
   restored spectrum should climb toward the target and **flatten at Nyquist**, not
   overshoot. Overshoot = invented detail.

## Scaling to real data / H100

This box is CPU-only; the code runs end-to-end here on tiny patches and scales up.
For the real run:

- **bf16, not fp8.** fp8's ~3 mantissa bits quantize away the near-noise-floor
  signal this task exists to recover. bf16 is barely slower for conv and preserves
  the dynamic range.
- `torch.compile`, `channels_last_3d`, **batch over patch size** (128³ at batch
  8–16 beats 256³ at batch 1–2; the patch-size floor is PSF support, ~128³).
- The bottleneck is **data loading / I/O**, not the GPU. Profile the `DataLoader`
  to keep utilization >90%; move degradation to the GPU if the loader starves;
  align reads to the 128³ chunking.
- **Occupancy-skip the *reads***, not just compute — a 72 TB volume at ~30%
  occupancy is ~21.6 TB of actual signal. Stream tile → infer → write; never
  materialize the whole volume. Multi-GPU DDP for inference is trivially parallel
  (different tiles per card) — add it when wall-clock matters, not before.

## Upgrade paths (documented, not built in v1)

NAFNet-3D backbone; channel-attention (Restormer-style) hybrid if underfitting the
texture prior; scale-conditioned multi-resolution single model; real co-registered
pairs for supervised validation; diffusion-posterior-with-data-consistency only if
L1 is measurably too soft *and* the degradation calibration is confirmed correct
(it reopens hallucination headroom, so it's the deliberate last step).

## Non-goals (v1)

No grid expansion / upscaling. No adversarial / perceptual / diffusion / transformer.
No real-pair co-registration. No scale-conditioning. These are upgrade paths above.

## License

Code: see `LICENSE`. Data: CC BY-NC 4.0 (Vesuvius Challenge / EduceLab) — attribute
and do not use commercially.
