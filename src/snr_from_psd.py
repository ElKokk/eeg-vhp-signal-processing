#!/usr/bin/env python

"""
src/snr_from_psd.py

Compute SNR around target frequencies from a PSD CSV and plot
SNR vs frequency with one line per channel.

Input CSV:
  - columns: freq, channel, [psd_col], ...
    where psd_col is usually 'psd_on' (ON-block PSD).

For each channel and each target frequency f0:

  - Signal power = PSD at bin closest to f0.
  - Noise power = median PSD in [f0 - bw, f0 + bw], excluding
    [f0 - exclude, f0 + exclude] around the target.

Outputs:
  - CSV: channel, target, freq_peak, snr_db
  - PNG: line plot of SNR vs target frequency for all channels.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compute SNR from PSD CSV.")
    parser.add_argument("--psd_csv", required=True, help="Input PSD CSV (e.g. ON-block PSD)")
    parser.add_argument("--targets", nargs="+", type=float, required=True, help="Target freqs (Hz)")
    parser.add_argument("--bw", type=float, default=2.0, help="+/- bandwidth for noise window")
    parser.add_argument("--exclude", type=float, default=0.5, help="+/- exclusion around f0")
    parser.add_argument("--col", default="psd_on", help="PSD column name to use")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--out_fig", required=True, help="Output PNG path")
    args = parser.parse_args()

    in_path = Path(args.psd_csv)
    df = pd.read_csv(in_path)

    if "freq" not in df.columns:
        raise ValueError(f"Column 'freq' not found in {in_path}")
    if "channel" not in df.columns:
        raise ValueError(f"Column 'channel' not found in {in_path}")
    if args.col not in df.columns:
        raise ValueError(f"PSD column '{args.col}' not in {in_path}")

    targets = sorted(args.targets)
    print(f"[SNR] Using PSD column: {args.col}")
    print(f"[SNR] Targets: {targets}")

    rows = []
    channels = sorted(df["channel"].unique())

    # --- Compute SNRs ---
    for ch in channels:
        df_ch = df[df["channel"] == ch]
        freqs = df_ch["freq"].values
        psd = df_ch[args.col].values

        for f0 in targets:
            # closest bin to f0
            idx_peak = int(np.argmin(np.abs(freqs - f0)))
            f_peak = float(freqs[idx_peak])
            signal = float(psd[idx_peak])

            # noise window around f0
            mask_noise = (
                (freqs >= f0 - args.bw)
                & (freqs <= f0 + args.bw)
                & ~((freqs >= f0 - args.exclude) & (freqs <= f0 + args.exclude))
            )
            noise_vals = psd[mask_noise]
            if noise_vals.size == 0:
                continue
            noise = float(np.median(noise_vals))

            snr_db = 10.0 * np.log10((signal + 1e-20) / (noise + 1e-20))
            rows.append(
                {
                    "channel": ch,
                    "target": f0,
                    "freq_peak": f_peak,
                    "snr_db": snr_db,
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[SNR] Summary written to {out_csv}")

    # --- Line plot: SNR vs frequency, one line per channel ---
    if out_df.empty:
        print("[SNR] No SNR values computed, skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for ch in channels:
        df_ch = out_df[out_df["channel"] == ch]
        if df_ch.empty:
            continue
        # ensure targets are in the same order on x-axis
        df_ch = df_ch.sort_values("target")
        x = df_ch["target"].values
        y = df_ch["snr_db"].values
        ax.plot(x, y, marker="o", label=ch)

    ax.set_xlabel("Target frequency (Hz)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"SNR vs frequency — {in_path.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[SNR] Figure written to {out_fig}")


if __name__ == "__main__":
    main()
