#!/usr/bin/env python

"""
src/spectrogram.py

Compute spectrograms for EEG + Bipolar channels.
"""

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

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


def infer_condition(stem: str) -> str:
    m = re.search(r"_f(\d+)", stem)
    if m: return f"{m.group(1)}Hz"
    if "baseline" in stem.lower(): return "baseline"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_flag", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--montages", default="config/montages.yaml")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists(): raise FileNotFoundError(in_path)

    cfg = load_main_config(args.config)
    montage = get_montage(cfg, load_montages(args.montages))

    spec_cfg = cfg.get("analysis", {}).get("spectrogram", {})
    fmin = float(spec_cfg.get("fmin", 1.0))
    fmax = float(spec_cfg.get("fmax", 60.0))
    nperseg = int(spec_cfg.get("nperseg", 2048))
    noverlap = int(nperseg * float(spec_cfg.get("noverlap_ratio", 0.75)))

    df = pd.read_csv(in_path)
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else 512.0

    channels = _get_channels_with_bipolar(df, cfg, montage)
    print(f"[SPECTROGRAM] Channels: {channels}")

    condition = infer_condition(in_path.stem)
    out_dir = Path(args.out_flag).parent / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # Event markers
    events = df[df["marker"] != 0] if "marker" in df.columns else pd.DataFrame()

    t_start = df["t_rel"].iloc[0]

    for ch in channels:
        sig = df[ch].to_numpy().astype(float)
        eff_nperseg = min(nperseg, len(sig))

        f, t_spec, Sxx = spectrogram(sig, fs=fs, nperseg=eff_nperseg, noverlap=noverlap)
        t_spec += t_start  # shift time

        fmask = (f >= fmin) & (f <= fmax)
        f_sel = f[fmask]
        Sxx_db = 10 * np.log10(Sxx[fmask, :] + 1e-20)

        # Relative Power (subtract median to see changes)
        Sxx_db_rel = Sxx_db - np.median(Sxx_db)

        fig, ax = plt.subplots(figsize=(12, 5))
        pcm = ax.pcolormesh(t_spec, f_sel, Sxx_db_rel, shading="gouraud", cmap="inferno")
        fig.colorbar(pcm, ax=ax, label="Relative dB")
        ax.set_title(f"Spectrogram: {ch} ({condition})")
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")

        # Overlay events
        if not events.empty:
            for t_ev, lab in zip(events["t_rel"], events["marker"]):
                if t_spec.min() <= t_ev <= t_spec.max():
                    ax.axvline(t_ev, color='white', linestyle='--', alpha=0.7)
                    ax.text(t_ev, fmax, str(int(lab)), color='white', verticalalignment='top')

        plt.tight_layout()
        plt.savefig(out_dir / f"{in_path.stem}_{ch}_spec.png", dpi=100)
        plt.close()

    with open(args.out_flag, "w") as f:
        f.write("ok")
    print(f"[SPECTROGRAM] Done. {out_dir}")


if __name__ == "__main__":
    main()