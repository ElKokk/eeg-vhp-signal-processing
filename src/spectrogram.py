#!/usr/bin/env python

"""
src/spectrogram.py

Compute spectrograms for EEG (+ bipolar / reref channels) from a
preprocessed structured CSV.

Key features:
- Uses sampling rate from config (eeg.sampling_rate).
- Color scheme: 'inferno' (as requested).
- Frequency (y) range dynamically extended so the stimulation frequency is
  always visible.
- Horizontal line at stim frequency.
- Percentile-based color scaling (1–99%) -> richer, more contrasted
  spectrograms.
- Saves PNGs into per-Hz subfolders, inferred from filename, e.g. "..._f24_..."
  → "<results_root>/24Hz/...".
- Optional --channels C3 Cz C4 to restrict which electrodes are processed.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp_spectrogram

from .config import load_main_config, load_montages, get_montage


def _get_sampling_rate(cfg: dict) -> float:
    eeg_cfg = cfg.get("eeg", {})
    return float(eeg_cfg.get("sampling_rate", 1000.0))


def _get_spectrogram_cfg(cfg: dict) -> dict:
    spec_cfg = cfg.get("analysis", {}).get("spectrogram", {})

    win_sec = float(spec_cfg.get("win_sec", 2.0))
    step_sec = float(spec_cfg.get("step_sec", 0.25))
    fmin_base = float(spec_cfg.get("fmin", 0.0))
    fmax_base = float(spec_cfg.get("fmax", 80.0))

    return {
        "win_sec": win_sec,
        "step_sec": step_sec,
        "fmin_base": fmin_base,
        "fmax_base": fmax_base,
    }


def _parse_stim_freq_from_name(stem: str) -> Optional[float]:
    """Extract stimulation frequency from filename stem like '..._f24_', '..._f275_v100'."""
    m = re.search(r"_f(\d+)(?:_v|_|$)", stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _get_freq_range(fs: float, spec_cfg: dict, stim_freq: Optional[float]) -> tuple[float, float]:
    fmin = spec_cfg["fmin_base"]
    fmax_base = spec_cfg["fmax_base"]
    nyq = fs / 2.0

    if stim_freq is not None:
        desired_max = stim_freq * 1.5
        fmax = min(nyq, max(fmax_base, desired_max))
    else:
        fmax = min(fmax_base, nyq)

    return fmin, fmax


def _find_channels(df: pd.DataFrame, montage) -> List[str]:
    base = [ch for ch in montage.channel_map.values() if ch in df.columns]
    bipolar = [
        c
        for c in df.columns
        if "-" in c and c not in base and c not in ("timestamp", "t_rel", "marker")
    ]
    return base + bipolar


def main():
    parser = argparse.ArgumentParser(description="Compute spectrograms for EEG channels.")
    parser.add_argument("--input", required=True, help="Preprocessed structured CSV.")
    parser.add_argument(
        "--out_flag",
        required=True,
        help="Path to sentinel file that indicates completion; PNGs are written next to it.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML.")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML.")
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Optional list of channel names to process (e.g. --channels C3 Cz C4). "
             "If omitted, all montage EEG + bipolar channels are used.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_flag = Path(args.out_flag)
    out_root = out_flag.parent
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    fs = _get_sampling_rate(cfg)
    spec_cfg = _get_spectrogram_cfg(cfg)

    stim_freq = _parse_stim_freq_from_name(in_path.stem)
    fmin, fmax = _get_freq_range(fs, spec_cfg, stim_freq)

    print("[SPECTROGRAM] Input   :", in_path)
    print("[SPECTROGRAM] fs      :", fs)
    print("[SPECTROGRAM] stim f0 :", stim_freq)
    print("[SPECTROGRAM] f-range :", fmin, "→", fmax, "Hz")

    df = pd.read_csv(in_path)

    # Time base (used only for plotting / event alignment)
    if "t_rel" in df.columns:
        t_rel = df["t_rel"].to_numpy(dtype=float)
    elif "timestamp" in df.columns:
        t0 = float(df["timestamp"].iloc[0])
        t_rel = df["timestamp"].to_numpy(dtype=float) - t0
    else:
        n_samples = len(df)
        t_rel = np.arange(n_samples, dtype=float) / fs

    # Events (optional)
    if "marker" in df.columns:
        events = df[
            ["t_rel" if "t_rel" in df.columns else "timestamp", "marker"]
        ].rename(columns={"t_rel": "t_rel", "timestamp": "t_rel"})
    else:
        events = pd.DataFrame(columns=["t_rel", "marker"])

    # Channels
    channels = _find_channels(df, montage)
    if args.channels:
        requested = {ch.strip() for ch in args.channels}
        channels = [ch for ch in channels if ch in requested]
    if not channels:
        raise ValueError("[SPECTROGRAM] No channels selected/found for spectrogram.")
    print("[SPECTROGRAM] Channels:", channels)

    # Spectrogram parameters
    win_sec = spec_cfg["win_sec"]
    step_sec = spec_cfg["step_sec"]

    nperseg = int(round(win_sec * fs))
    if nperseg < 2:
        nperseg = 2
    noverlap = int(round((win_sec - step_sec) * fs))
    if noverlap < 0:
        noverlap = 0
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    # Per-Hz folder
    if stim_freq is not None:
        hz_folder = f"{int(round(stim_freq))}Hz"
    else:
        hz_folder = "no_freq"
    out_dir = out_root / hz_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for ch in channels:
        x = df[ch].to_numpy(dtype=float)

        # Handle NaNs / short signals
        not_nan = np.isfinite(x)
        n_finite = int(np.count_nonzero(not_nan))
        if n_finite == 0:
            print(f"[SPECTROGRAM] Channel {ch}: all NaNs, skipping.")
            continue
        if n_finite < nperseg:
            print(
                f"[SPECTROGRAM] Channel {ch}: finite samples shorter than window "
                f"(finite={n_finite}, nperseg={nperseg}), skipping."
            )
            continue

        mean_val = np.nanmean(x)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        x = np.where(np.isfinite(x), x, mean_val)

        # Spectrogram (PSD)
        f, t_spec, Sxx = sp_spectrogram(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        if Sxx.size == 0 or f.size == 0 or t_spec.size == 0:
            print(f"[SPECTROGRAM] Channel {ch}: empty spectrogram output, skipping.")
            continue

        freq_mask = np.logical_and(f >= fmin, f <= fmax)
        if not np.any(freq_mask):
            print(
                f"[SPECTROGRAM] Channel {ch}: no frequencies in "
                f"[{fmin}, {fmax}] Hz, skipping."
            )
            continue

        f_sel = f[freq_mask]
        S_sel = Sxx[freq_mask, :]

        if S_sel.size == 0:
            print(f"[SPECTROGRAM] Channel {ch}: zero-size S_sel, skipping.")
            continue

        # Convert to dB and use percentile-based scaling for richer contrast
        S_db = 10.0 * np.log10(S_sel + 1e-20)
        vmin, vmax = np.percentile(S_db, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = vmax = None

        fig, ax = plt.subplots(figsize=(8, 4))
        pcm = ax.pcolormesh(
            t_spec,
            f_sel,
            S_db,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(pcm, ax=ax, label="Power (dB)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(fmin, fmax)

        title_extra = f" (f0={stim_freq:.0f} Hz)" if stim_freq is not None else ""
        ax.set_title(f"Spectrogram ({ch}) — {in_path.stem}{title_extra}")

        # Horizontal line at stim frequency
        if stim_freq is not None and fmin <= stim_freq <= fmax:
            ax.axhline(stim_freq, color="white", linestyle="--", linewidth=1.0, alpha=0.9)

        # Vertical event markers
        if not events.empty:
            for t_ev, lab in zip(events["t_rel"], events["marker"]):
                if t_spec.min() <= t_ev <= t_spec.max():
                    ax.axvline(t_ev, color="white", linestyle=":", alpha=0.6)
                    try:
                        lab_str = str(int(lab))
                    except Exception:
                        lab_str = str(lab)
                    ax.text(
                        t_ev,
                        fmax,
                        lab_str,
                        color="white",
                        verticalalignment="top",
                        fontsize=7,
                    )

        plt.tight_layout()
        out_png = out_dir / f"{in_path.stem}_{ch}_spec.png"
        plt.savefig(out_png, dpi=140)
        plt.close()
        print(f"[SPECTROGRAM] Saved {out_png}")

    # Touch sentinel flag
    out_flag.parent.mkdir(parents=True, exist_ok=True)
    with out_flag.open("w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"[SPECTROGRAM] Done. Flag written to {out_flag}")


if __name__ == "__main__":
    main()
