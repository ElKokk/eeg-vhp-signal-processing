#!/usr/bin/env python

"""
src/psd.py

Compute PSDs (Welch) for EEG + Bipolar channels.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.config import load_main_config, load_montages, get_montage


def _get_channels_with_bipolar(df: pd.DataFrame, cfg: dict, montage) -> List[str]:
    """Return list of channels to analyse = montage EEG + any bipolar in df."""
    # 1. Base EEG channels
    base_eeg = [ch for ch in montage.channel_map.values() if ch in df.columns]

    # 2. Bipolar channels defined in config
    reref_cfg = cfg.get("analysis", {}).get("reref", {})
    bipolar_pairs = reref_cfg.get("bipolar_pairs", []) or []

    def parse_pair(pair):
        if isinstance(pair, str):
            if "-" not in pair: return None
            a, b = pair.split("-", 1)
            return a.strip(), b.strip()
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return str(pair[0]).strip(), str(pair[1]).strip()
        return None

    extra_channels = []
    for pair in bipolar_pairs:
        parsed = parse_pair(pair)
        if parsed:
            a, b = parsed
            col_name = f"{a}-{b}"
            if col_name in df.columns:
                extra_channels.append(col_name)

    # Combine and deduplicate
    all_channels = base_eeg + extra_channels
    return list(dict.fromkeys(all_channels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fig", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--montages", default="config/montages.yaml")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    df = pd.read_csv(in_path)

    # Estimate fs
    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' not found.")
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else 512.0
    print(f"[PSD] fs ≈ {fs:.2f} Hz")

    # Get channels (including C3-C4 if present)
    channels = _get_channels_with_bipolar(df, cfg, montage)
    print(f"[PSD] Analyzing channels: {channels}")

    # Config params
    psd_cfg = cfg.get("analysis", {}).get("psd", {})
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", 60.0))
    nperseg = int(psd_cfg.get("nperseg", 2048))

    out_csv = Path(args.out_csv)
    out_fig = Path(args.out_fig)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for ch in channels:
        sig = df[ch].to_numpy().astype(float)

        # Handle short signals
        eff_nperseg = min(nperseg, len(sig))
        freqs, psd_vals = welch(sig, fs=fs, nperseg=eff_nperseg, scaling='density')

        mask = (freqs >= fmin) & (freqs <= fmax)
        f_sel = freqs[mask]
        p_sel = psd_vals[mask]
        p_db = 10 * np.log10(p_sel + 1e-20)

        ax.plot(f_sel, p_db, label=ch, linewidth=1.0, alpha=0.9)

        for f, p, db in zip(f_sel, p_sel, p_db):
            records.append({"freq": f, "channel": ch, "psd": p, "psd_db": db})

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(f"PSD (Welch) - {in_path.stem}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"[PSD] Done. Saved {out_csv}")


if __name__ == "__main__":
    main()