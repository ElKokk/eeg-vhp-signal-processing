#!/usr/bin/env python

"""
src/spectrogram.py

Compute spectrograms (time–frequency) for selected EEG channels in a
preprocessed CSV and save one PNG per channel, with event markers
(ON/OFF) and a horizontal line at the stimulation frequency.

This version matches the "old" look:
- nperseg / noverlap_ratio (e.g. nperseg=4096, 75% overlap)
- relative dB (subtract global median)
- shading="gouraud", cmap="inferno"
- figsize=(10, 5), dpi=200

Plus:
- dynamic frequency range so stim frequency is always visible
- per-Hz output folders (e.g. results/.../24Hz/...)
- optional --channels to select which electrodes to plot
- Snakemake integration via --out_flag sentinel file
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _parse_stim_freq_from_name(stem: str) -> Optional[float]:
    """
    Extract stimulation frequency from filename stem like:
      "..._f24_", "..._f275_v100", "..._f36"
    Returns float Hz or None.
    """
    m = re.search(r"_f(\d+)(?:_v|_|$)", stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _get_fs(df: pd.DataFrame, cfg: dict) -> float:
    """
    Estimate sampling rate from timestamps; if not possible or implausible,
    fall back to cfg['eeg']['sampling_rate'] (default 1000.0 for your setup).
    """
    # Default fallback is 1000 Hz because your streaming montage runs at 1000 Hz.
    cfg_fs = float(cfg.get("eeg", {}).get("sampling_rate", 1000.0))

    if "timestamp" in df.columns:
        dt = df["timestamp"].diff().dropna()
        if len(dt) > 0:
            dt_med = float(dt.median())
            if dt_med > 0:
                fs_est = 1.0 / dt_med
            else:
                fs_est = cfg_fs
        else:
            fs_est = cfg_fs
    else:
        fs_est = cfg_fs

    # If the estimate is crazy, just use config value
    if fs_est < 0.5 * cfg_fs or fs_est > 2.0 * cfg_fs:
        print(
            f"[SPECTROGRAM] estimated fs ≈ {fs_est:.2f} Hz, "
            f"config says {cfg_fs:.2f} Hz → using config fs"
        )
        return cfg_fs

    print(f"[SPECTROGRAM] fs ≈ {fs_est:.2f} Hz (config={cfg_fs:.2f} Hz)")
    return fs_est


def _get_spec_cfg(cfg: dict) -> dict:
    """
    Read spectrogram settings from config.

    Supports BOTH:
      - "old style"  nperseg + noverlap_ratio
      - "new style"  win_sec + step_sec (used only if nperseg missing)
    """
    spec_cfg = cfg.get("analysis", {}).get("spectrogram", {})

    fmin = float(spec_cfg.get("fmin", 1.0))
    fmax_base = float(spec_cfg.get("fmax", 60.0))

    # Old-style settings (preferred for 'sharp' look)
    nperseg = spec_cfg.get("nperseg", None)
    noverlap_ratio = spec_cfg.get("noverlap_ratio", 0.75)

    # New-style (fallback) in seconds
    win_sec = spec_cfg.get("win_sec", None)
    step_sec = spec_cfg.get("step_sec", None)

    return {
        "fmin": fmin,
        "fmax_base": fmax_base,
        "nperseg": int(nperseg) if nperseg is not None else None,
        "noverlap_ratio": float(noverlap_ratio),
        "win_sec": float(win_sec) if win_sec is not None else None,
        "step_sec": float(step_sec) if step_sec is not None else None,
    }


def _get_freq_range(fs: float, spec_cfg: dict, stim_freq: Optional[float]) -> tuple[float, float]:
    """
    Base fmin/fmax from config; ensure that fmax is high enough to include
    the stimulation frequency (stim * 1.5), but never above Nyquist.
    """
    fmin = spec_cfg["fmin"]
    fmax_base = spec_cfg["fmax_base"]
    nyq = fs / 2.0

    if stim_freq is not None:
        desired_max = stim_freq * 1.5
        fmax = min(nyq, max(fmax_base, desired_max))
    else:
        fmax = min(fmax_base, nyq)

    return fmin, fmax


def _get_channels(df: pd.DataFrame, montage, requested: Optional[List[str]]) -> List[str]:
    """
    Get list of channels to plot.

    - base EEG channels from montage
    - plus bipolar channels (names with '-') if present
    - optionally restricted to 'requested' list from --channels
    """
    base = [ch for ch in montage.channel_map.values() if ch in df.columns]
    bipolar = [
        c
        for c in df.columns
        if "-" in c and c not in base and c not in ("timestamp", "t_rel", "marker")
    ]
    chans = base + bipolar
    if requested:
        requested_set = {ch.strip() for ch in requested}
        chans = [ch for ch in chans if ch in requested_set]
    return chans


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute spectrograms per channel from preprocessed EEG CSV."
    )
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument(
        "--out_flag",
        required=True,
        help=(
            "Path to sentinel file that indicates completion; PNGs are written "
            "next to it (inside per-Hz subfolders)."
        ),
    )
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    parser.add_argument(
        "--channels",
        nargs="+",
        help=(
            "Optional list of channel names to process "
            "(e.g. --channels C3 Cz C4). If omitted, all montage EEG + bipolar channels are used."
        ),
    )
    args = parser.parse_args()

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)

    # Sampling rate
    fs = _get_fs(df, cfg)

    # Spectrogram config
    spec_cfg = _get_spec_cfg(cfg)

    # Stim frequency from filename
    stim_freq = _parse_stim_freq_from_name(in_path.stem)
    if stim_freq is not None:
        print(f"[SPECTROGRAM] Parsed stimulation frequency: {stim_freq:.1f} Hz")

    fmin, fmax = _get_freq_range(fs, spec_cfg, stim_freq)
    print(f"[SPECTROGRAM] Frequency range: {fmin} → {fmax} Hz")

    # Time base for events
    if "t_rel" in df.columns:
        t_rel = df["t_rel"].to_numpy(dtype=float)
    elif "timestamp" in df.columns:
        t0 = float(df["timestamp"].iloc[0])
        t_rel = df["timestamp"].to_numpy(dtype=float) - t0
    else:
        t_rel = np.arange(len(df), dtype=float) / fs

    # Event markers (robust to NaNs)
    if "marker" in df.columns:
        markers = df["marker"].to_numpy()
        # Only keep finite, non-zero markers as events
        valid = np.isfinite(markers) & (markers != 0)
        event_idx = np.where(valid)[0]
        event_onsets = t_rel[event_idx]
        event_labels = [str(int(round(float(markers[i])))) for i in event_idx]
    else:
        event_onsets = np.array([], dtype=float)
        event_labels = []

    # Channels to process
    channels = _get_channels(df, montage, args.channels)
    if not channels:
        raise ValueError("[SPECTROGRAM] No channels selected/found for spectrogram.")
    print("[SPECTROGRAM] Channels:", channels)

    # Output directories
    out_flag = Path(args.out_flag)
    root_dir = out_flag.parent
    root_dir.mkdir(parents=True, exist_ok=True)

    if stim_freq is not None:
        hz_folder = f"{int(round(stim_freq))}Hz"
    else:
        hz_folder = "no_freq"
    out_dir = root_dir / hz_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Windowing parameters (prefer old nperseg style)
    if spec_cfg["nperseg"] is not None:
        base_nperseg = spec_cfg["nperseg"]
        noverlap_ratio = spec_cfg["noverlap_ratio"]
        print(
            f"[SPECTROGRAM] Using nperseg={base_nperseg}, "
            f"noverlap_ratio={noverlap_ratio}"
        )
    else:
        # Fallback: compute nperseg from win_sec
        win_sec = spec_cfg["win_sec"] or 2.0
        step_sec = spec_cfg["step_sec"] or 0.25
        base_nperseg = int(round(win_sec * fs))
        noverlap_ratio = 1.0 - step_sec / win_sec
        print(
            f"[SPECTROGRAM] Using win_sec={win_sec}, step_sec={step_sec} "
            f"→ nperseg={base_nperseg}, noverlap_ratio={noverlap_ratio:.2f}"
        )

    # -----------------------------------------------------------------
    # Per-channel spectrograms
    # -----------------------------------------------------------------
    for ch in channels:
        sig = df[ch].to_numpy().astype(float)

        # Handle NaNs
        if not np.any(np.isfinite(sig)):
            print(f"[SPECTROGRAM] Channel {ch}: all NaNs, skipping.")
            continue
        mean_val = np.nanmean(sig)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        sig = np.where(np.isfinite(sig), sig, mean_val)

        nperseg_eff = min(base_nperseg, len(sig))
        if nperseg_eff < 2:
            print(f"[SPECTROGRAM] Channel {ch}: signal too short, skipping.")
            continue

        noverlap = int(nperseg_eff * noverlap_ratio)
        if noverlap >= nperseg_eff:
            noverlap = nperseg_eff - 1

        f, t, Sxx = sp_spectrogram(
            sig,
            fs=fs,
            nperseg=nperseg_eff,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        if Sxx.size == 0 or f.size == 0 or t.size == 0:
            print(f"[SPECTROGRAM] Channel {ch}: empty spectrogram output, skipping.")
            continue

        # Restrict frequency range
        fmask = (f >= fmin) & (f <= fmax)
        if not np.any(fmask):
            print(
                f"[SPECTROGRAM] Channel {ch}: no freqs in [{fmin},{fmax}] Hz, skipping."
            )
            continue

        f_sel = f[fmask]
        Sxx_sel = Sxx[fmask, :]

        # Relative dB (old behaviour)
        Sxx_db = 10.0 * np.log10(Sxx_sel + 1e-20)
        Sxx_db_rel = Sxx_db - np.median(Sxx_db)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        pcm = ax.pcolormesh(
            t,
            f_sel,
            Sxx_db_rel,
            shading="gouraud",
            cmap="inferno",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(fmin, fmax)

        title_extra = f" (f0={stim_freq:.0f} Hz)" if stim_freq is not None else ""
        ax.set_title(f"Spectrogram ({ch}) — {in_path.name}{title_extra}")

        fig.colorbar(pcm, ax=ax, label="Relative power (dB)")

        # Horizontal line at stim frequency
        if stim_freq is not None and fmin <= stim_freq <= fmax:
            ax.axhline(
                stim_freq,
                linestyle="--",
                linewidth=1.0,
                color="white",
                alpha=0.9,
            )

        # Event markers (vertical dashed lines + labels)
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
        out_file = out_dir / f"{in_path.stem}_{ch}_spec.png"
        plt.savefig(out_file, dpi=200)
        plt.close()

        print(f"[SPECTROGRAM] Saved spectrogram for {ch} to {out_file}")

    # Touch sentinel flag for Snakemake
    out_flag.parent.mkdir(parents=True, exist_ok=True)
    with out_flag.open("w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"[SPECTROGRAM] Done. Flag written to {out_flag}")


if __name__ == "__main__":
    main()
