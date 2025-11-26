###############################################
# Snakefile – EEG VHP pipeline
#
# Steps:
#   1) structure_raw: raw CSV -> structured CSV (t_rel, channel names)
#   2) preprocess   : structured -> preproc (band-pass + notch)
#   3) psd          : preproc -> PSD CSV + PNG (in per-condition subfolders)
#   4) spectrogram  : preproc -> per-channel spectrograms (per-condition folders)
###############################################

import os

# Use this YAML as the global config
configfile: "config/config.yaml"

# Shortcuts for paths & lists from config
RAW_DIR      = config["paths"]["raw_dir"]
PROC_DIR     = config["paths"]["processed_dir"]
RESULTS_DIR  = config["paths"]["results_dir"]
RECORDINGS   = config["recordings"]

PSD_DIR      = os.path.join(RESULTS_DIR, "psd")
SPEC_DIR     = os.path.join(RESULTS_DIR, "spectrograms")

###############################################
# Final targets
###############################################

rule all:
    input:
        # 1) Preprocessed CSVs for each recording
        expand(
            f"{PROC_DIR}" + "/{rec}.preproc.csv",
            rec=RECORDINGS
        ),
        # 2) PSD jobs done for each recording
        expand(
            f"{PSD_DIR}" + "/{rec}.psd.done",
            rec=RECORDINGS
        ),
        # 3) Spectrogram jobs done for each recording
        expand(
            f"{SPEC_DIR}" + "/{rec}.spec.done",
            rec=RECORDINGS
        )

###############################################
# 1) structure_raw
###############################################

rule structure_raw:
    """
    Convert raw FreeEEG32 CSV -> structured CSV with t_rel and channel names.
    """
    input:
        f"{RAW_DIR}" + "/{rec}.csv"
    output:
        f"{PROC_DIR}" + "/{rec}.structured.csv"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.structure_raw \
            --input {input} \
            --output {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 2) preprocess
###############################################

rule preprocess:
    """
    Preprocess structured EEG: band-pass + notch filter.
    Input:  structured CSV from structure_raw
    Output: preprocessed CSV with cleaned EEG channels
    """
    input:
        f"{PROC_DIR}" + "/{rec}.structured.csv"
    output:
        f"{PROC_DIR}" + "/{rec}.preproc.csv"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.preprocess \
            --input {input} \
            --output {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 3) PSD (per-condition subfolders)
###############################################

rule psd:
    """
    Compute PSD (Welch) for each EEG channel from preprocessed CSV.
    Creates:
      - CSV + PNG in results/psd/<condition>/
      - a small .psd.done file as a flag so Snakemake knows it's done
    """
    input:
        f"{PROC_DIR}" + "/{rec}.preproc.csv"
    output:
        f"{PSD_DIR}" + "/{rec}.psd.done"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml",
        outdir=PSD_DIR
    shell:
        """
        python -m src.psd \
            --input {input} \
            --out_dir {params.outdir} \
            --config {params.cfg} \
            --montages {params.mont} \
        && echo "ok" > {output}
        """

###############################################
# 4) Spectrogram (per-condition subfolders)
###############################################

rule spectrogram:
    """
    Compute spectrograms per channel from preprocessed CSV.
    Creates:
      - multiple PNGs in results/spectrograms/<condition>/
      - a small .spec.done file as a flag so Snakemake knows it's done
    """
    input:
        f"{PROC_DIR}" + "/{rec}.preproc.csv"
    output:
        f"{SPEC_DIR}" + "/{rec}.spec.done"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml",
        outdir=SPEC_DIR
    shell:
        """
        python -m src.spectrogram \
            --input {input} \
            --out_dir {params.outdir} \
            --config {params.cfg} \
            --montages {params.mont} \
        && echo "ok" > {output}
        """
