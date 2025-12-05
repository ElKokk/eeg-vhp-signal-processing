#!/usr/bin/env python

"""src/structure_raw.py

CLI script: load one raw EEG CSV, apply montage, and write a structured CSV.

Usage (manually):

    python -m src.structure_raw \
        --input data/raw_streaming/251127-..._c1_f75_v100.csv \
        --output data/processed_streaming/251127-..._c1_f75_v100.structured.csv \
        --config config/config_streaming.yaml \
        --montages config/montages.yaml
"""

import argparse
from pathlib import Path
import sys

from .config import load_main_config, load_montages, get_montage
from .io import load_eeg_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw EEG CSV into a structured CSV using a montage definition."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to raw CSV file recorded by the board.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output structured CSV.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to main config YAML (config.yaml or config_streaming.yaml).",
    )
    parser.add_argument(
        "--montages",
        "-m",
        required=True,
        help="Path to montages.yaml file.",
    )
    return parser.parse_args()


def main() -> None:
    # Debug: see what Snakemake actually passes
    print("[structure_raw] sys.argv:", sys.argv)

    args = parse_args()

    # Load configuration and montage
    main_cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(main_cfg, montages)

    print("[structure_raw] --------")
    print(f"[structure_raw] Input   : {args.input}")
    print(f"[structure_raw] Output  : {args.output}")
    print(f"[structure_raw] Montage : {montage.name}")

    # Load and structure the EEG recording
    rec = load_eeg_csv(args.input, montage)

    print(f"[structure_raw] fs ≈ {rec.fs:.2f} Hz")
    print(f"[structure_raw] good channels ({len(rec.good_channels)}): {rec.good_channels}")
    print(f"[structure_raw] dead channels ({len(rec.dead_channels)}): {rec.dead_channels}")

    # Save structured data: timestamp, t_rel, marker, channel columns
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec.df.to_csv(out_path, index=False)

    print(f"[structure_raw] Structured CSV written to {out_path}")
    print("[structure_raw] --------")


if __name__ == "__main__":
    main()
