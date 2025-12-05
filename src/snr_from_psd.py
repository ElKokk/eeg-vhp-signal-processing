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
  - Noise power = median PSD in [f0 - bw, f0 + bw], optionally
    excluding narrow bands around specified artefact frequencies
    (e.g. 50, 100, 150 Hz).

SNR (dB) = 10 * log10(signal_power / noise_power).

This script supports both:
  - --bw (old name, half-width of noise window)
  - --bandwidth (alias of --bw, overrides it if given)
  - --exclude f1 f2 ... (absolute frequencies to exclude
    from the noise window, e.g. 50 100 150 200).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SNR from PSD CSV.")

    parser.add_argument(
        "--psd_csv",
        required=True,
        help="Input PSD CSV (e.g. ON-block PSD)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        required=True,
        help="Target frequencies (Hz)",
    )
    # Old name (kept for compatibility)
    parser.add_argument(
        "--bw",
        type=float,
        default=None,
        help="Half-width of noise window around each target (Hz)",
    )
    # New alias used in your Snakefile
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Alias of --bw; if provided, overrides --bw.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Frequencies (Hz) to exclude from the noise window, e.g. 50 100 150 200.\n"
            "Any PSD bins whose frequency is within 0.5 Hz of one of these values are"
            " removed from the noise estimate."
        ),
    )
    parser.add_argument(
        "--col",
        default="psd_on",
        help="Name of PSD column to use (default: psd_on)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path for SNR table",
    )
    parser.add_argument(
        "--out_fig",
        required=True,
        help="Output PNG path for SNR plot",
    )

    args = parser.parse_args()

    # Resolve bandwidth (Hz)
    if args.bandwidth is not None:
        bw = float(args.bandwidth)
    elif args.bw is not None:
        bw = float(args.bw)
    else:
        bw = 2.0  # sensible default
    if bw <= 0:
        raise ValueError("Noise window half-width (--bw/--bandwidth) must be > 0")

    exclude_freqs = np.array(args.exclude, dtype=float) if args.exclude else np.array([])
    print(f"[SNR] Noise window half-width (bw): ±{bw:.2f} Hz")
    if exclude_freqs.size > 0:
        print(f"[SNR] Excluding artefact freqs (±0.5 Hz): {exclude_freqs.tolist()}")
    else:
        print("[SNR] No artefact frequencies excluded.")

    in_path = Path(args.psd_csv)
    df = pd.read_csv(in_path)

    if args.col not in df.columns:
        raise ValueError(f"PSD column '{args.col}' not in {in_path}")

    targets = sorted(args.targets)
    print(f"[SNR] Using PSD column: {args.col}")
    print(f"[SNR] Targets: {targets}")

    rows = []
    channels = sorted(df["channel"].unique())

    # ------------------------------------------------------------------
    # Compute SNRs
    # ------------------------------------------------------------------
    for ch in channels:
        df_ch = df[df["channel"] == ch]
        freqs = df_ch["freq"].to_numpy(dtype=float)
        psd = df_ch[args.col].to_numpy(dtype=float)

        # sanity: remove NaNs from PSD
        valid = np.isfinite(psd) & np.isfinite(freqs)
        freqs = freqs[valid]
        psd = psd[valid]
        if freqs.size == 0:
            print(f"[SNR] Channel {ch}: no valid PSD samples, skipping.")
            continue

        for f0 in targets:
            # closest bin to f0 = signal
            idx_peak = int(np.argmin(np.abs(freqs - f0)))
            f_peak = float(freqs[idx_peak])
            signal = float(psd[idx_peak])

            # candidate noise window around f0
            noise_mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)

            # remove bins near artefact frequencies (powerline etc.)
            if exclude_freqs.size > 0:
                for f_art in exclude_freqs:
                    noise_mask &= np.abs(freqs - f_art) > 0.5  # 0.5-Hz veto band

            noise_freqs = freqs[noise_mask]
            noise_psd = psd[noise_mask]

            if noise_psd.size < 3:
                # Not enough points to estimate noise robustly
                print(
                    f"[SNR] Channel {ch}, f0={f0:.2f} Hz: too few noise bins (n={noise_psd.size}), skipping."
                )
                continue

            noise = float(np.median(noise_psd))
            if noise <= 0 or not np.isfinite(noise) or not np.isfinite(signal):
                print(
                    f"[SNR] Channel {ch}, f0={f0:.2f} Hz: invalid signal/noise (signal={signal}, noise={noise}), skipping."
                )
                continue

            snr_db = 10.0 * np.log10(signal / noise)

            rows.append(
                {
                    "channel": ch,
                    "target": float(f0),
                    "freq_peak": f_peak,
                    "signal": signal,
                    "noise": noise,
                    "snr_db": snr_db,
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[SNR] Summary written to {out_csv}")

    # ------------------------------------------------------------------
    # Line plot: SNR vs frequency, one line per channel
    # ------------------------------------------------------------------
    if out_df.empty:
        print("[SNR] No SNR values computed, skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for ch in channels:
        df_ch = out_df[out_df["channel"] == ch]
        if df_ch.empty:
            continue
        df_ch = df_ch.sort_values("target")
        x = df_ch["target"].to_numpy(dtype=float)
        y = df_ch["snr_db"].to_numpy(dtype=float)
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
