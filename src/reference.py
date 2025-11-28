#!/usr/bin/env python

"""
src/reference.py

Apply Common Average Reference (CAR) to EEG channels, and optionally add
bipolar derivations defined in config.yaml.

1. Identifies EEG channels from montage.
2. CAR: Subtracts the mean of all EEG channels from each EEG channel.
3. Bipolar: Computes differences (e.g., C3-C4) and adds them as new columns.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from src.config import load_main_config, load_montages, get_montage


def main():
    parser = argparse.ArgumentParser(
        description="Apply CAR and optional bipolar derivations."
    )
    parser.add_argument("--input", required=True, help="Input preprocessed CSV")
    parser.add_argument("--output", required=True, help="Output rereferenced CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--montages", default="config/montages.yaml")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load configuration
    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    # Load data
    df = pd.read_csv(in_path)

    # 1. Identify EEG channels present in the data
    eeg_channels = [ch for ch in montage.channel_map.values() if ch in df.columns]

    if not eeg_channels:
        raise ValueError("No EEG channels found in input CSV.")

    print(f"[REREF] Applying Common Average Reference (CAR) to {len(eeg_channels)} channels.")

    # 2. Apply CAR (Common Average Reference)
    # Calculate the average across all EEG channels for each timepoint
    common_avg = df[eeg_channels].mean(axis=1)

    # Subtract average from each channel
    df[eeg_channels] = df[eeg_channels].sub(common_avg, axis=0)

    # 3. Add Bipolar Derivations
    reref_cfg = cfg.get("analysis", {}).get("reref", {})
    bipolar_pairs = reref_cfg.get("bipolar_pairs", [])

    if bipolar_pairs:
        print(f"[REREF] Computing bipolar pairs: {bipolar_pairs}")

    def parse_pair(pair):
        """Helper to parse ['C3', 'C4'] or 'C3-C4'."""
        if isinstance(pair, str):
            if "-" not in pair:
                return None
            a, b = pair.split("-", 1)
            return a.strip(), b.strip()
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return str(pair[0]).strip(), str(pair[1]).strip()
        return None

    for raw_pair in bipolar_pairs:
        parsed = parse_pair(raw_pair)
        if parsed is None:
            print(f"[REREF] Warning: Skipping invalid pair format: {raw_pair}")
            continue

        anode, cathode = parsed

        # Check if both channels exist
        if anode in df.columns and cathode in df.columns:
            col_name = f"{anode}-{cathode}"
            # Compute difference (since data is already CAR, this is valid)
            df[col_name] = df[anode] - df[cathode]
            print(f"[REREF] Created channel: {col_name}")
        else:
            print(f"[REREF] Warning: Missing channels for pair {anode}-{cathode}")

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[REREF] Saved to {out_path}")


if __name__ == "__main__":
    main()