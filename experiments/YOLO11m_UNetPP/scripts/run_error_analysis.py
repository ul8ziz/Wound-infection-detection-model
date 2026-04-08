#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from combined.error_analysis import run_error_analysis
from pipeline_utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-images", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(SCRIPT_DIR / "config.yaml")
    run_error_analysis(cfg, SCRIPT_DIR, split=args.split, max_images=args.max_images)


if __name__ == "__main__":
    main()
