#!/usr/bin/env python

"""
src/psd.py

Compute PSDs (Welch) for each EEG channel in a preprocessed CSV and save:

- A long-format CSV with columns: freq, channel, psd, psd_db
- A simple multi-channel PSD plot (PNG)

Files are written into:
  results/psd/<condition>/<stem>.psd.csv
  results/psd/<condition>/<stem>.psd.png

where <condition> is:
  - "baseline_vhp_off"                for baseline_with_VHP_powered_OFF
  - "baseline_vhp_on_no_contact"      for baseline_with_VHP_powered_ON_stim_ON_no_contact...
  - "<freq>Hz"                        for stim runs (c1_f22, c1_f24, etc.)
  - "baseline_other" / "other"        as fallbacks
"""

import argparse
from pathlib import Path
from typing import List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from .config import load_main_config, load_montages, get_montage


def infer_condition(stem: str) -> str:
    """
    Infer condition label from filename stem.

    Order matters: we check specific baselines BEFORE generic _fXX_ matches.
    """
    lower = stem.lower()

    if "baseline_with_vhp_powered_off" in lower:
        return "baseline_vhp_off"

    if "baseline_with_vhp_powered_on_stim_on_no_contact" in lower:
        return "baseline_vhp_on_no_contact"

    # Generic stim runs with fXX
    m = re.search(r"_f(\d+)", stem)
    if m:
        return f"{m.group(1)}Hz"

    if "baseline" in lower:
        return "baseline_other"

    return "other"


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSDs per channel from preprocessed EEG CSV."
    )
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Base output directory for PSD files (subfolders per condition)",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    # ---- load preprocessed data ----
    df = pd.read_csv(in_path)

    # Sampling rate from timestamps
    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' not found in preprocessed CSV.")
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else float(cfg["eeg"]["sampling_rate"])
    print(f"[PSD] {in_path.name}: fs ≈ {fs:.2f} Hz")

    # EEG channels: those defined in montage and actually present
    eeg_channels: List[str] = [
        ch for ch in montage.channel_map.values() if ch in df.columns
    ]
    if not eeg_channels:
        raise ValueError(
            "No EEG channels found in preprocessed CSV. "
            "Check montage and preprocess output."
        )
    print(f"[PSD] EEG channels: {eeg_channels}")

    # PSD params from config (safe defaults)
    psd_cfg = cfg.get("analysis", {}).get("psd", {})
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", 60.0))
    nperseg = int(psd_cfg.get("nperseg", 4096))

    print(f"[PSD] Params: fmin={fmin}, fmax={fmax}, nperseg={nperseg}")

    all_rows = []
    plt.figure(figsize=(10, 6))

    for ch in eeg_channels:
        sig = df[ch].to_numpy().astype(float)
        nperseg_eff = min(nperseg, len(sig))
        if nperseg_eff < 16:
            raise ValueError(
                f"Signal too short for PSD: channel {ch}, n_samples={len(sig)}"
            )

        freqs, Pxx = welch(sig, fs=fs, nperseg=nperseg_eff)

        # Restrict frequency range
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_sel = freqs[mask]
        Pxx_sel = Pxx[mask]
        PdB = 10.0 * np.log10(Pxx_sel + 1e-20)

        # Save rows for CSV
        for f, p, pd_val in zip(freqs_sel, Pxx_sel, PdB):
            all_rows.append(
                {"freq": f, "channel": ch, "psd": p, "psd_db": pd_val}
            )

        # Plot
        plt.plot(freqs_sel, PdB, label=ch)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"PSD (Welch) — {in_path.name}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize="small", loc="upper right")
    plt.tight_layout()

    # ---- decide output folder & names ----
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem  # e.g. 251120-..._c1_f24_v100.preproc
    cond_label = infer_condition(stem)
    cond_dir = base_out_dir / cond_label
    cond_dir.mkdir(parents=True, exist_ok=True)

    out_csv = cond_dir / f"{stem}.psd.csv"
    out_fig = cond_dir / f"{stem}.psd.png"

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print(f"[PSD] CSV written to {out_csv}")
    print(f"[PSD] Figure written to {out_fig}")
    print(f"[PSD] Condition='{cond_label}'")


if __name__ == "__main__":
    main()
