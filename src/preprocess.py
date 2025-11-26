#!/usr/bin/env python

"""
src/preprocess.py

Preprocessing for EEG:

- Load structured CSV (with t_rel, timestamp, marker, EEG channels).
- Band-pass filter (e.g. 1–80 Hz).
- Notch filter at mains frequencies (e.g. 50, 100, 150 Hz).
- Save preprocessed CSV.

This is basic signal cleaning before PSD/spectrogram analysis.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

from .config import load_main_config, load_montages, get_montage


def design_bandpass(l_freq: float | None, h_freq: float | None, fs: float, order: int = 4):
    """Design a Butterworth band-pass (or high/low) filter as SOS."""
    nyq = fs / 2.0
    if l_freq is None and h_freq is None:
        return None

    if l_freq is None:
        wn = h_freq / nyq
        btype = "lowpass"
    elif h_freq is None:
        wn = l_freq / nyq
        btype = "highpass"
    else:
        wn = [l_freq / nyq, h_freq / nyq]
        btype = "bandpass"

    sos = butter(order, wn, btype=btype, output="sos")
    return sos


def apply_bandpass(data: np.ndarray, fs: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase bandpass filter channel-wise (axis=0: time, axis=1: channels).
    data: shape (n_samples, n_channels)
    """
    sos = design_bandpass(l_freq, h_freq, fs, order=order)
    if sos is None:
        return data
    # sosfiltfilt: zero-phase filtering
    return sosfiltfilt(sos, data, axis=0)


def apply_notch(data: np.ndarray, fs: float, freqs: List[float], Q: float = 30.0) -> np.ndarray:
    """
    Apply a cascade of notch filters for each frequency in freqs.
    data: shape (n_samples, n_channels)
    """
    out = data.copy()
    for f0 in freqs:
        if f0 >= fs / 2.0:
            # can't filter above Nyquist
            continue
        # iirnotch in SciPy supports fs argument (modern versions)
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        out = filtfilt(b, a, out, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEG (band-pass + notch) from structured CSV.")
    parser.add_argument("--input", required=True, help="Path to structured CSV")
    parser.add_argument("--output", required=True, help="Path for preprocessed CSV")
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    args = parser.parse_args()

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)

    # Re-estimate fs from timestamps (robust to config mismatch)
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else float(cfg["eeg"]["sampling_rate"])
    print(f"Preprocessing {in_path}")
    print(f"  estimated fs ≈ {fs:.2f} Hz")

    # Determine which columns are EEG channels (from montage)
    eeg_channels = [ch for ch in montage.channel_map.values() if ch in df.columns]
    if not eeg_channels:
        raise ValueError("No EEG channels found in structured CSV. Check montage and structure_raw outputs.")

    print("  EEG channels:", eeg_channels)

    # Extract EEG data as (n_samples, n_channels)
    data = df[eeg_channels].to_numpy().astype(float)

    # Get preprocessing params from config
    bp_cfg = cfg["preprocessing"]["bandpass"]
    l_freq = float(bp_cfg.get("l_freq", 1.0))
    h_freq = float(bp_cfg.get("h_freq", 80.0))
    order = int(bp_cfg.get("order", 4))

    notch_cfg = cfg["preprocessing"]["notch"]
    notch_freqs = [float(f) for f in notch_cfg.get("freqs", [])]
    Q = float(notch_cfg.get("Q", 30.0))

    print(f"  Band-pass: {l_freq}–{h_freq} Hz (order={order})")
    print(f"  Notch freqs: {notch_freqs} (Q={Q})")

    # 1) Band-pass
    data_bp = apply_bandpass(data, fs, l_freq=l_freq, h_freq=h_freq, order=order)

    # 2) Notch
    data_clean = apply_notch(data_bp, fs, freqs=notch_freqs, Q=Q)

    # Put filtered data back into DataFrame
    df_out = df.copy()
    df_out[eeg_channels] = data_clean

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"Preprocessed CSV written to {out_path}")


if __name__ == "__main__":
    main()
