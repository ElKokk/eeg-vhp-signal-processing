#!/usr/bin/env python

"""
src/psd_onblocks.py

Compute PSD averaged ONLY over ON-blocks (marker 1 -> 11).
Uses CAR/Bipolar data if provided.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.config import load_main_config, load_montages, get_montage


def _get_channels_with_bipolar(df: pd.DataFrame, cfg: dict, montage) -> List[str]:
    base_eeg = [ch for ch in montage.channel_map.values() if ch in df.columns]
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

    extra = []
    for pair in bipolar_pairs:
        parsed = parse_pair(pair)
        if parsed:
            col = f"{parsed[0]}-{parsed[1]}"
            if col in df.columns: extra.append(col)

    return list(dict.fromkeys(base_eeg + extra))


def find_on_intervals(markers: np.ndarray) -> List[Tuple[int, int]]:
    on_starts = np.where(markers == 1)[0]
    off_starts = np.where(markers == 11)[0]

    intervals = []
    if len(on_starts) == 0:
        # Fallback: if no markers, use whole file
        return [(0, len(markers))]

    for start in on_starts:
        future_offs = off_starts[off_starts > start]
        if len(future_offs) > 0:
            intervals.append((start, future_offs[0]))
        else:
            intervals.append((start, len(markers)))
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fig", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--montages", default="config/montages.yaml")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists(): raise FileNotFoundError(in_path)

    cfg = load_main_config(args.config)
    montage = get_montage(cfg, load_montages(args.montages))

    df = pd.read_csv(in_path)
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else 512.0

    channels = _get_channels_with_bipolar(df, cfg, montage)
    print(f"[PSD_ON] Channels: {channels}")

    markers = df["marker"].to_numpy().astype(int)
    intervals = find_on_intervals(markers)
    print(f"[PSD_ON] Found {len(intervals)} ON intervals")

    psd_cfg = cfg.get("analysis", {}).get("psd", {})
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", 60.0))
    # Use 1-second epochs for stability
    nperseg = int(fs)

    accumulated_psd = {ch: [] for ch in channels}
    freqs_ref = None

    for (start, end) in intervals:
        dur = end - start
        if dur < nperseg: continue

        for ch in channels:
            sig = df[ch].iloc[start:end].to_numpy().astype(float)
            freqs, Pxx = welch(sig, fs=fs, nperseg=nperseg)
            if freqs_ref is None: freqs_ref = freqs
            accumulated_psd[ch].append(Pxx)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 6))

    if freqs_ref is not None:
        mask = (freqs_ref >= fmin) & (freqs_ref <= fmax)
        f_sel = freqs_ref[mask]

        for ch in channels:
            if not accumulated_psd[ch]: continue

            # Mean across all blocks
            P_mean = np.mean(np.stack(accumulated_psd[ch]), axis=0)
            P_db = 10 * np.log10(P_mean + 1e-20)

            p_sel = P_mean[mask]
            db_sel = P_db[mask]

            ax.plot(f_sel, db_sel, label=ch)
            for f, p, db in zip(f_sel, p_sel, db_sel):
                rows.append({"freq": f, "channel": ch, "psd_on": p, "psd_on_db": db})

    ax.set_title(f"ON-Block PSD: {in_path.stem}")
    ax.set_ylabel("dB")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    plt.close()

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[PSD_ON] Saved {args.out_csv}")


if __name__ == "__main__":
    main()