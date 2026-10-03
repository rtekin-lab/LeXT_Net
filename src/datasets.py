# -*- coding: utf-8 -*-
"""
datasets.py  —  LeXT-Net data loading utilities
================================================
Verified format details (from sample files):

PTB-XL
  Path  : .../records100/<XXXXX>/<ecg_id>_lr.{dat,hea}
  FS    : 100 Hz  (already at target — no resampling needed)
  Shape : (1000, 12) per record  ← confirmed from 00001_lr.hea
  Lead order in file: I, II, III, AVR, AVL, AVF, V1..V6
  Note  : AVR/AVL/AVF in header; remapped to standard aVR/aVL/aVF internally.

Chapman (ECGData_Chapman/ECGData)
  Path  : .../ECGData/MUSE_*.csv
  FS    : 500 Hz  → resampled to 100 Hz
  Shape : (5000, 12) raw = 10 s × 500 Hz  → (1000, 12) after resample
  Lead order: I, II, III, aVR, aVL, aVF, V1..V6  (from CSV header row)
  Note  : CSV has a BOM header row with column names; skip with header=0.

CPSC2018
  Path  : .../cpsc_2018/A*.{mat,hea}
  FS    : 500 Hz  → resampled to 100 Hz
  Shape : (7500, 12) raw = 15 s × 500 Hz  → crop to 5000 (10 s) → (1000, 12)
  Lead order: I, II, III, aVR, aVL, aVF, V1..V6  (from A0001.hea)

All loaders return:
  signals  : np.ndarray  (N, 1000, 12)  float32
  rec_ids  : list[str]
"""

import os
import glob
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FS  = 100     # Hz — unified sampling rate
TARGET_LEN = 1000    # samples — 10 s × 100 Hz
N_LEADS    = 12

# Standard lead order (paper & model convention)
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resample_to_target(sig: np.ndarray, fs_in: int) -> np.ndarray:
    """
    Resample (T, 12) from fs_in to TARGET_FS.
    If fs_in == TARGET_FS, return unchanged.
    Uses scipy.signal.resample (FFT-based, preserves spectral content below Nyquist).
    """
    if fs_in == TARGET_FS:
        return sig
    n_out = int(round(sig.shape[0] * TARGET_FS / fs_in))
    resampled = sp_signal.resample(sig, n_out, axis=0)
    return resampled.astype(np.float32)


def _ensure_10s(sig: np.ndarray, source_label: str = "") -> Optional[np.ndarray]:
    """
    Guarantee exactly TARGET_LEN (1000) samples.

    Strategy:
      - sig has exactly 1000 samples  → return as-is
      - sig has MORE than 1000        → crop to first 1000 (= first 10 s)
      - sig has LESS than 950         → discard (too short, <9.5 s — log warning)
      - sig has 950..999              → zero-pad on the right (marginal length difference)
    """
    t = sig.shape[0]
    if t == TARGET_LEN:
        return sig
    if t > TARGET_LEN:
        return sig[:TARGET_LEN]
    # t < TARGET_LEN
    if t < int(TARGET_LEN * 0.95):   # < 950 samples
        warnings.warn(f"Record {source_label} has only {t} samples "
                      f"after resampling — discarded.")
        return None
    # Marginal shortfall — pad with zeros
    pad = np.zeros((TARGET_LEN - t, sig.shape[1]), dtype=sig.dtype)
    return np.vstack([sig, pad])


def _zscore_normalize(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-lead z-score normalisation. sig: (T, C)."""
    mu  = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0,  keepdims=True) + eps
    return ((sig - mu) / std).astype(np.float32)


# ── PTB-XL ─────────────────────────────────────────────────────────────────────

def load_ptbxl(
    ptbxl_path: str | Path,
    normalize: bool = True,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Load PTB-XL 100 Hz recordings.

    Directory layout (confirmed):
        records100/
          00000/   00001_lr.dat  00001_lr.hea
          01000/   01001_lr.dat  ...
          ...

    Parameters
    ----------
    ptbxl_path : path to the dataset root (contains ptbxl_database.csv
                 AND records100/).
    normalize  : per-record per-lead z-score.
    max_records: optional cap for quick testing.
    verbose    : progress printing.

    Returns
    -------
    signals  : (N, 1000, 12) float32
    rec_ids  : list of ecg_id strings
    metadata : pd.DataFrame (ptbxl_database.csv)
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb:  pip install wfdb")

    ptbxl_path = Path(ptbxl_path)
    csv_path   = ptbxl_path.parent / "ptbxl_database.csv"
    if not csv_path.exists():
        # Try one level up
        csv_path = ptbxl_path.parent.parent / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found near {ptbxl_path}.\n"
            f"Expected at: {ptbxl_path.parent / 'ptbxl_database.csv'}"
        )

    metadata   = pd.read_csv(csv_path, index_col="ecg_id")
    metadata   = metadata.sort_index()

    # Build list of (ecg_id, absolute_record_path)
    # filename_lr column contains paths like "records100/00000/00001_lr"
    # relative to the dataset root (parent of records100).
    dataset_root = ptbxl_path.parent   # .../ptb-xl-...-1.0.3/
    records  = metadata["filename_lr"].tolist()
    rec_ids  = [str(i) for i in metadata.index.tolist()]

    if max_records is not None:
        records  = records[:max_records]
        rec_ids  = rec_ids[:max_records]

    signals, valid_ids = [], []
    n = len(records)

    for i, (rel_path, rid) in enumerate(zip(records, rec_ids)):
        if verbose and (i % 2000 == 0 or i == n - 1):
            print(f"\r  PTB-XL  {i+1:>6}/{n}", end="", flush=True)
        try:
            rec_path = str(dataset_root / rel_path)
            data, header = wfdb.rdsamp(rec_path)     # (1000, 12) in mV at 100 Hz
            sig = data[:, :N_LEADS].astype(np.float32)

            # PTB-XL records100 are already at 100 Hz — but verify
            fs = header["fs"]
            if fs != TARGET_FS:
                sig = _resample_to_target(sig, fs)

            sig = _ensure_10s(sig, source_label=rid)
            if sig is None:
                continue

            if normalize:
                sig = _zscore_normalize(sig)
            signals.append(sig)
            valid_ids.append(rid)
        except Exception as e:
            warnings.warn(f"PTB-XL record {rid}: {e}")

    if verbose:
        print(f"\n  PTB-XL: loaded {len(signals):,} / {n:,} records.")

    valid_int_ids = [int(i) for i in valid_ids]
    return (
        np.stack(signals, axis=0),
        valid_ids,
        metadata.loc[valid_int_ids],
    )


# ── Chapman-Shaoxing ───────────────────────────────────────────────────────────

# Confirmed from MUSE_20180111_155115_19000.csv:
#   - Row 0  : header  "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6"  (BOM prefix ﻿)
#   - Rows 1..5000 : 5000 data rows (500 Hz × 10 s)
#   - 12 columns matching LEAD_ORDER exactly
#   - Values in µV (MUSE system uses µV × 1000 convention — leave as-is for normalisation)

def load_chapman(
    chapman_path: str | Path,
    normalize: bool = True,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load Chapman-Shaoxing ECG dataset.

    Directory layout (confirmed):
        ECGData_Chapman/ECGData/
          MUSE_20180111_155115_19000.csv
          MUSE_20180111_155154_74000.csv
          ...

    Each CSV:
      - Header row : I, II, III, aVR, aVL, aVF, V1..V6  (BOM-prefixed)
      - Data rows  : 5000  (500 Hz × 10 s)
      → Resampled to 1000 (100 Hz × 10 s)

    Parameters
    ----------
    chapman_path : path to the ECGData folder (contains MUSE_*.csv files).
    normalize    : per-record per-lead z-score.
    max_records  : optional cap.
    verbose      : progress.

    Returns
    -------
    signals : (N, 1000, 12)  float32
    rec_ids : list of CSV stem names
    """
    chapman_path = Path(chapman_path)
    csv_files    = sorted(chapman_path.glob("*.csv"))
    if not csv_files:
        # Try one level deeper
        csv_files = sorted(chapman_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under {chapman_path}")

    if max_records is not None:
        csv_files = csv_files[:max_records]

    signals, rec_ids = [], []
    n = len(csv_files)

    for i, csv_file in enumerate(csv_files):
        if verbose and (i % 1000 == 0 or i == n - 1):
            print(f"\r  Chapman  {i+1:>6}/{n}", end="", flush=True)
        try:
            # encoding='utf-8-sig' strips the BOM automatically
            df  = pd.read_csv(csv_file, header=0, encoding="utf-8-sig")
            # Verify columns match expected lead order
            if list(df.columns) != LEAD_ORDER:
                warnings.warn(
                    f"Chapman {csv_file.stem}: unexpected columns "
                    f"{list(df.columns)} — skipping."
                )
                continue

            sig = df.values.astype(np.float32)         # (5000, 12)

            # Chapman raw shape should be 5000 (500 Hz × 10 s)
            # Resample to 100 Hz → 1000 samples
            sig = _resample_to_target(sig, fs_in=500)  # → (1000, 12)
            sig = _ensure_10s(sig, source_label=csv_file.stem)
            if sig is None:
                continue

            if normalize:
                sig = _zscore_normalize(sig)
            signals.append(sig)
            rec_ids.append(csv_file.stem)
        except Exception as e:
            warnings.warn(f"Chapman {csv_file.stem}: {e}")

    if verbose:
        print(f"\n  Loaded {len(signals):,} / {n:,} Chapman records.")

    return np.stack(signals, axis=0), rec_ids


# ── CPSC 2018 ──────────────────────────────────────────────────────────────────

# Confirmed from A0001.hea:
#   "A0001 12 500 7500"  → 12 leads, 500 Hz, 7500 samples = 15 s
#   Lead order: I, II, III, aVR, aVL, aVF, V1..V6  (matches LEAD_ORDER)
#   Data file : A0001.mat  (WFDB MATLAB format, 16-bit, gain 1000 µV/mV)
#   Strategy  : take first 5000 samples (0–10 s), then resample → 1000

def load_cpsc2018(
    cpsc_path: str | Path,
    normalize: bool = True,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load the CPSC 2018 Challenge dataset.

    Directory layout (confirmed):
        cpsc_2018/
          A0001.hea   A0001.mat
          A0002.hea   A0002.mat
          ...

    Each record:
      - 12 leads, 500 Hz, 7500 samples (15 s)
      → Crop first 5000 samples (0–10 s), then resample to 1000 (100 Hz × 10 s)

    Parameters
    ----------
    cpsc_path  : directory containing A*.hea / A*.mat files.
    normalize  : per-record per-lead z-score.
    max_records: optional cap.
    verbose    : progress.

    Returns
    -------
    signals : (N, 1000, 12)  float32
    rec_ids : list of record stem names
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb:  pip install wfdb")

    cpsc_path = Path(cpsc_path)
    hea_files = sorted(cpsc_path.glob("*.hea"))
    if not hea_files:
        raise FileNotFoundError(f"No .hea files found in {cpsc_path}")

    if max_records is not None:
        hea_files = hea_files[:max_records]

    signals, rec_ids = [], []
    n = len(hea_files)

    for i, hea in enumerate(hea_files):
        if verbose and (i % 500 == 0 or i == n - 1):
            print(f"\r  CPSC2018  {i+1:>5}/{n}", end="", flush=True)
        try:
            rec_path        = str(hea.with_suffix(""))
            data, header    = wfdb.rdsamp(rec_path)    # (7500, 12) at 500 Hz typically
            sig             = data[:, :N_LEADS].astype(np.float32)
            fs              = header["fs"]              # should be 500

            # Crop to 10 s BEFORE resampling (avoids edge artefacts)
            ten_sec_samples = int(fs * 10)
            if sig.shape[0] > ten_sec_samples:
                sig = sig[:ten_sec_samples]             # (5000, 12) if 500 Hz

            # Resample to TARGET_FS
            sig = _resample_to_target(sig, fs_in=fs)   # → (1000, 12)
            sig = _ensure_10s(sig, source_label=hea.stem)
            if sig is None:
                continue

            if normalize:
                sig = _zscore_normalize(sig)
            signals.append(sig)
            rec_ids.append(hea.stem)
        except Exception as e:
            warnings.warn(f"CPSC2018 {hea.stem}: {e}")

    if verbose:
        print(f"\n  Loaded {len(signals):,} / {n:,} CPSC2018 records.")

    return np.stack(signals, axis=0), rec_ids
