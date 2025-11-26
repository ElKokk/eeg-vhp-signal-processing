#!/usr/bin/env python

"""
src/spectrogram.py

Compute spectrograms (time–frequency) for each EEG channel in a preprocessed CSV
and save one PNG per channel, with event markers (ON/OFF) overlaid.

Output layout:
  results/spectrograms/<condition>/<stem>_<channel>_spec.png
"""

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from .config import load_main_config, load_montages, get_montage


def infer_condition(stem: str) -> str:
    """
    Infer condition label from filename stem.

    Same logic as in src.psd:
      - baseline_with_VHP_powered_OFF                -> baseline_vhp_off
      - baseline_with_VHP_powered_ON_stim_ON_no_contact -> baseline_vhp_on_no_contact
      - _fXX_ stim files                             -> XXHz
      - other baselines                              -> baseline_other
      - fallback                                     -> other
    """
    lower = stem.lower()

    if "baseline_with_vhp_powered_off" in lower:
        return "baseline_vhp_off"

    if "baseline_with_vhp_powered_on_stim_on_no_contact" in lower:
        return "baseline_vhp_on_no_contact"

    m = re.search(r"_f(\d+)", stem)
    if m:
        return f"{m.group(1)}Hz"

    if "baseline" in lower:
        return "baseline_other"

    return "other"


def main():
    parser = argparse.ArgumentParser(
        description="Compute spectrograms per channel from preprocessed EEG CSV."
    )
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Base output directory for spectrogram PNGs (subfolders per condition)",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    args = parser.parse_args()

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    in_path = Path(args.input)
    df = pd.read_csv(in_path)

    # ----- sampling rate -----
    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' not found in preprocessed CSV.")
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else float(cfg["eeg"]["sampling_rate"])
    print(f"[SPEC] {in_path.name}: fs ≈ {fs:.2f} Hz")

    # ----- EEG channels -----
    eeg_channels: List[str] = [
        ch for ch in montage.channel_map.values() if ch in df.columns
    ]
    if not eeg_channels:
        raise ValueError(
            "No EEG channels found in preprocessed CSV. "
            "Check montage and preprocess output."
        )
    print(f"[SPEC] EEG channels: {eeg_channels}")

    # ----- spectrogram params -----
    spec_cfg = cfg.get("analysis", {}).get("spectrogram", {})
    fmin = float(spec_cfg.get("fmin", 1.0))
    fmax = float(spec_cfg.get("fmax", 60.0))
    nperseg = int(spec_cfg.get("nperseg", 4096))
    noverlap_ratio = float(spec_cfg.get("noverlap_ratio", 0.75))

    print(
        f"[SPEC] Params: fmin={fmin}, fmax={fmax}, "
        f"nperseg={nperseg}, noverlap_ratio={noverlap_ratio}"
    )

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Decide condition subfolder from filename stem
    stem = in_path.stem
    cond_label = infer_condition(stem)
    out_dir = base_out_dir / cond_label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SPEC] Condition '{cond_label}', saving figures into: {out_dir}")

    # ----- event markers -----
    if "marker" not in df.columns or "t_rel" not in df.columns:
        raise ValueError("Columns 'marker' or 't_rel' not found in preprocessed CSV.")

    markers = df["marker"].values
    t_rel = df["t_rel"].values
    event_idx = np.where(markers != 0)[0]
    event_onsets = t_rel[event_idx]
    event_labels = [str(int(markers[i])) for i in event_idx]

    # ----- per-channel spectrogram -----
    for ch in eeg_channels:
        sig = df[ch].to_numpy().astype(float)

        nperseg_eff = min(nperseg, len(sig))
        if nperseg_eff < 16:
            raise ValueError(
                f"Signal too short for spectrogram: channel {ch}, n_samples={len(sig)}"
            )
        noverlap = int(nperseg_eff * noverlap_ratio)

        f, t, Sxx = spectrogram(
            sig,
            fs=fs,
            nperseg=nperseg_eff,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        # Restrict frequency range
        fmask = (f >= fmin) & (f <= fmax)
        f_sel = f[fmask]
        Sxx_sel = Sxx[fmask, :]

        # Convert to relative dB
        Sxx_db = 10.0 * np.log10(Sxx_sel + 1e-20)
        Sxx_db_rel = Sxx_db - np.median(Sxx_db)

        fig, ax = plt.subplots(figsize=(10, 5))
        pcm = ax.pcolormesh(t, f_sel, Sxx_db_rel, shading="gouraud", cmap="inferno")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(fmin, fmax)
        ax.set_title(f"Spectrogram ({ch}) — {in_path.name}")
        fig.colorbar(pcm, ax=ax, label="Relative power (dB)")

        # Event markers as dashed vertical lines
        for onset, label in zip(event_onsets, event_labels):
            if onset >= t.min() and onset <= t.max():
                ax.axvline(onset, linestyle="--", linewidth=0.8, color="white")
                y_text = fmin + 0.02 * (fmax - fmin)
                ax.text(
                    onset,
                    y_text,
                    label,
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                    color="white",
                )

        plt.tight_layout()
        out_file = out_dir / f"{stem}_{ch}_spec.png"
        plt.savefig(out_file, dpi=200)
        plt.close()

        print(f"[SPEC] Saved spectrogram for {ch} to {out_file}")


if __name__ == "__main__":
    main()
