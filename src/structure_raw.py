#!/usr/bin/env python

"""
src/structure_raw.py

CLI script: load one raw EEG CSV, apply montage, and write a structured CSV.

Usage (manually):
    python -m src.structure_raw \
        --input data/raw/251120-..._c1_f22_v100.csv \
        --output data/processed/251120-..._c1_f22_v100.structured.csv \
        --config config/config.yaml \
        --montages config/montages.yaml
"""

import argparse
from pathlib import Path

from .config import load_main_config, load_montages, get_montage
from .io import load_eeg_csv


def main():
    parser = argparse.ArgumentParser(description="Structure raw FreeEEG32 CSV using YAML-configured montage.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Path to write structured CSV")
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    args = parser.parse_args()

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    rec = load_eeg_csv(args.input, montage)

    print(f"Loaded {args.input}")
    print(f"  fs ≈ {rec.fs:.2f} Hz")
    print(f"  good channels: {rec.good_channels}")
    print(f"  dead channels: {rec.dead_channels}")

    # Save structured data: time, marker, and all channels
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec.df.to_csv(out_path, index=False)
    print(f"Structured CSV written to {out_path}")


if __name__ == "__main__":
    main()
