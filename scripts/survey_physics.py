#!/usr/bin/env python3
"""Survey per-volume acquisition/reconstruction physics from metadata.json.

For each volume in sources.json, fetch metadata.json and extract the parameters
that define the real PSF -- so we can build a physics-derived degradation operator
instead of guessing Gaussian sigmas. Reports which volumes have usable physics and
the parameter spread. Writes a physics manifest.

    python scripts/survey_physics.py --sources configs/sources.json --out physics.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fetch_metadata(bucket, scroll, zarr):
    uri = f"s3://{bucket}/{scroll}/{zarr}/metadata.json"
    # pipe to stdout (cp to a file fails under the noisy awscli build)
    proc = subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--quiet", uri, "-"],
                          capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return json.loads(proc.stdout)
    except Exception:
        return None


def extract_physics(md):
    """Pull the PSF-relevant params, tolerant of schema variation."""
    if not md:
        return None
    scan = md.get("scan", {}) or {}
    tomo = scan.get("tomo", {}) or {}
    acq = tomo.get("acquisition", {}) or {}
    proc = tomo.get("processing", {}) or {}
    pre = proc.get("preprocessing", {}) or {}
    phase = pre.get("phase", {}) or {}
    det = acq.get("detector", {}) or {}

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    p = {
        "paganin_delta_beta": num(phase.get("delta_beta")),
        "unsharp_coeff": num(phase.get("unsharp_coeff")),
        "unsharp_sigma": num(phase.get("unsharp_sigma")),
        "energy_kev": num(acq.get("energy")),
        "sample_detector_mm": num(acq.get("sampleDetectorDistance")),
        "source_sample_mm": num(acq.get("sourceSampleDistance")),
        "sample_pixel_mm": num(det.get("samplePixelSize")),
        "scintillator": det.get("scintillator"),
        "recon_method": proc.get("reconstruction", {}).get("method") if isinstance(proc.get("reconstruction"), dict) else None,
        "phase_method": phase.get("method"),
    }
    has = p["paganin_delta_beta"] is not None and p["energy_kev"] is not None
    p["usable"] = bool(has)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--out", default="physics.json")
    args = ap.parse_args()

    src = json.loads(Path(args.sources).read_text())
    bucket = src["bucket"]
    results = []
    for v in src["volumes"]:
        md = fetch_metadata(bucket, v["scroll"], v["zarr"])
        phys = extract_physics(md)
        if phys is None:
            print(f"[--- ] {v['scroll']:12s} no metadata.json")
            results.append({**v, "physics": None})
            continue
        tag = "phys" if phys["usable"] else "part"
        print(f"[{tag}] {v['scroll']:12s} vum={v['voxel_um']:.3f} "
              f"db={phys['paganin_delta_beta']} E={phys['energy_kev']} "
              f"sdd={phys['sample_detector_mm']} uns_s={phys['unsharp_sigma']} "
              f"scint={phys['scintillator']}")
        results.append({**v, "physics": phys})

    Path(args.out).write_text(json.dumps(results, indent=2))
    usable = sum(1 for r in results if r.get("physics") and r["physics"].get("usable"))
    print(f"\n{usable}/{len(results)} volumes have usable physics. wrote {args.out}")
    # spread of key params
    dbs = [r["physics"]["paganin_delta_beta"] for r in results
           if r.get("physics") and r["physics"].get("paganin_delta_beta")]
    if dbs:
        print(f"delta_beta values: {sorted(set(dbs))}")


if __name__ == "__main__":
    main()
