#!/usr/bin/env python3
"""Prove the dependency-light S3 reader works against the live public bucket.

Lists volumes/resolutions for a scroll and reads ONE 128^3 chunk from level 0.
Requires network + the `aws` CLI (or boto3). Skipped in CI.

    python scripts/probe_s3.py PHerc0500P2
    python scripts/probe_s3.py PHerc0500P2 volumes/20250526151718-2.215um-0.4m-111keV-masked.zarr
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.s3zarr import open_s3

BUCKET = "vesuvius-challenge-open-data"


def list_volumes(scroll: str):
    uri = f"s3://{BUCKET}/{scroll}/volumes/"
    out = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", uri], capture_output=True, text=True
    )
    vols = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("PRE ") and line.endswith(".zarr/"):
            vols.append("volumes/" + line[4:].rstrip("/"))
    return vols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scroll")
    ap.add_argument("zarr", nargs="?", help="volumes/...zarr (defaults to first found)")
    args = ap.parse_args()

    vols = list_volumes(args.scroll)
    print(f"{args.scroll}: {len(vols)} volume(s)")
    for v in vols:
        print("  ", v)
    zarr_path = args.zarr or (vols[0] if vols else None)
    if not zarr_path:
        print("no volumes found")
        return

    print(f"\nopening level 0 of {zarr_path} ...")
    z = open_s3(BUCKET, args.scroll, zarr_path, level=0)
    print(f"  shape={z.shape} chunks={z.chunks} dtype={z.dtype} "
          f"n_chunks={z.n_chunks()} sep={z.sep!r}")

    # read one central 128^3 chunk-aligned region
    cz, cy, cx = (s // 2 // c * c for s, c in zip(z.shape, z.chunks))
    dz, dy, dx = z.chunks
    region = z.read_region(cz, cy, cx, dz, dy, dx)
    nz = int((region > 0).sum())
    print(f"  read region @({cz},{cy},{cx}) size {region.shape}: "
          f"nonzero={nz}/{region.size} ({100*nz/region.size:.1f}% occupancy)")
    print("OK: dependency-light S3 reader works.")


if __name__ == "__main__":
    main()
