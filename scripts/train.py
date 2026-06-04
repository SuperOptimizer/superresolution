#!/usr/bin/env python3
"""Train the restoration model.

    python scripts/train.py --config configs/smoke.yaml --smoke
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superres.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="force synthetic data (no network); for CPU smoke testing")
    ap.add_argument("--steps", type=int, default=None, help="override training steps")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    args = ap.parse_args()
    train(args.config, smoke=args.smoke, max_steps=args.steps, resume=args.resume)


if __name__ == "__main__":
    main()
