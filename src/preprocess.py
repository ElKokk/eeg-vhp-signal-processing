#!/usr/bin/env python

"""
src/preprocess.py

Preprocessing for EEG:

- Load structured CSV (with t_rel, timestamp, marker, EEG channels).
- Band-pass filter (default 1–80 Hz, but highcut is automatically raised
  if needed to include the stimulation frequency).
- Optional notch filters (e.g. 50, 100, 150 Hz).
- Save preprocessed CSV.

This is basic signal cleaning before PSD/spectrogram analysis.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

from .config import load_main_config, load_montages, get_montage


def design_bandpass(
    l_freq: Optional[float],
    h_freq: Optional[float],
    fs: float,
    order: int = 4,
):
    """Design a Butterworth band-pass (or low/high-pass) filter as SOS."""
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

    return butter(order, wn, btype=btype, output="sos")


def apply_bandpass(
    data: np.ndarray,
    fs: float,
    l_freq: Optional[float],
    h_freq: Optional[float],
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase bandpass filter channel-wise (axis=0: time, axis=1: channels).
    data: shape (n_samples, n_channels)
    """
    sos = design_bandpass(l_freq, h_freq, fs, order=order)
    if sos is None:
        return data
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
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        out = filtfilt(b, a, out, axis=0)
    return out


def _parse_stim_freq_from_name(stem: str) -> Optional[float]:
    """
    Extract stimulation frequency from filename stem.

    Matches patterns like:
      "..._f22_", "..._f275_v100", "..._f36"
    """
    m = re.search(r"_f(\d+)(?:_v|_|$)", stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


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

    # Target sampling rate from config
    eeg_cfg = cfg.get("eeg", {})
    target_fs = float(eeg_cfg.get("sampling_rate", 512.0))

    # Estimate fs from timestamps if available
    if "timestamp" in df.columns:
        dt = df["timestamp"].diff().dropna()
        if len(dt) > 0:
            dt_med = float(dt.median())
            fs_est = 1.0 / dt_med if dt_med > 0 else target_fs
        else:
            fs_est = target_fs
    else:
        fs_est = target_fs

    print(f"[PREPROC] Input: {in_path}")
    if fs_est < 0.5 * target_fs or fs_est > 2.0 * target_fs:
        print(f"[PREPROC] estimated fs ≈ {fs_est:.2f} Hz, config says {target_fs:.2f} Hz → using config fs.")
        fs = target_fs
    else:
        fs = fs_est
        print(f"[PREPROC] fs ≈ {fs:.2f} Hz")

    # Parse stimulation frequency from filename
    stim_freq = _parse_stim_freq_from_name(in_path.stem)
    if stim_freq is not None:
        print(f"[PREPROC] Parsed stimulation frequency: {stim_freq:.1f} Hz")

    # EEG channels from montage
    eeg_channels = [ch for ch in montage.channel_map.values() if ch in df.columns]
    if not eeg_channels:
        raise ValueError("[PREPROC] No EEG channels found in structured CSV. Check montage and structure_raw outputs.")
    print("[PREPROC] EEG channels:", eeg_channels)

    data = df[eeg_channels].to_numpy(dtype=float)

    # Preprocessing config (robust defaults if missing)
    pre_cfg = cfg.get("preprocessing", {})
    bp_cfg = pre_cfg.get("bandpass", {})
    l_freq = float(bp_cfg.get("l_freq", 1.0))
    h_freq = float(bp_cfg.get("h_freq", 80.0))
    order = int(bp_cfg.get("order", 4))

    # Adapt highcut so we do NOT filter out the stimulation frequency
    if stim_freq is not None:
        nyq = fs / 2.0
        desired_h = stim_freq * 1.5  # leave some margin
        new_h = min(nyq * 0.99, max(h_freq, desired_h))
        if new_h > h_freq:
            print(
                f"[PREPROC] Raising band-pass highcut from {h_freq:.1f} Hz to {new_h:.1f} Hz "
                f"to include stim at {stim_freq:.1f} Hz"
            )
            h_freq = new_h

    notch_cfg = pre_cfg.get("notch", {})
    notch_freqs = [float(f) for f in notch_cfg.get("freqs", [])]
    Q = float(notch_cfg.get("Q", 30.0))

    print(f"[PREPROC] Band-pass: {l_freq:.1f}–{h_freq:.1f} Hz (order={order})")
    print(f"[PREPROC] Notch freqs: {notch_freqs} (Q={Q})")

    # 1) Band-pass
    data_bp = apply_bandpass(data, fs, l_freq=l_freq, h_freq=h_freq, order=order)

    # 2) Notch (optional)
    data_clean = apply_notch(data_bp, fs, freqs=notch_freqs, Q=Q) if notch_freqs else data_bp

    # Save
    df_out = df.copy()
    df_out[eeg_channels] = data_clean

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"[PREPROC] Preprocessed CSV written to {out_path}")


if __name__ == "__main__":
    main()
