# -*- coding: utf-8 -*-
"""
metrics.py  —  Evaluation metrics
===================================
Computes the eight metrics reported in Table 1 of the paper:

    R2, ssim, pcc, msc, mae, rmse, nrmse, snr

Public API
----------
compute_lead_metrics_batch(y_true, y_pred) -> dict
    Returns per-lead metric averages over N records.
    Shape: { lead_name: { metric: float } }
"""

import warnings
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as _ssim

LEAD_ORDER  = ["I", "II", "III", "AVR", "AVL", "AVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
METRIC_KEYS = ["R2", "ssim", "pcc", "msc", "mae", "rmse", "nrmse", "snr"]
FS          = 100   # Hz — unified sampling rate
BANDS       = [(0.5, 4), (4, 15), (15, 40)]   # MSC frequency bands (Hz)


# ── Single lead, single record ────────────────────────────────────────────────

def _single_lead_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all eight metrics for one 1-D signal pair."""
    eps    = 1e-12
    y_true = np.nan_to_num(y_true.astype(np.float64))
    y_pred = np.nan_to_num(y_pred.astype(np.float64))

    # R²
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = -1.0
    r2 = float(np.nan_to_num(r2, nan=-1.0))

    # PCC
    pcc = 0.0
    if np.std(y_true) > eps and np.std(y_pred) > eps:
        try:
            pcc = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pass
    pcc = float(np.nan_to_num(pcc))

    # SSIM (1-D, sliding window)
    dr = float(max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min()))
    if dr < eps:
        dr = 1.0
    try:
        win = max(3, min(101, (len(y_true) // 2) * 2 - 1))
        # win_size must be odd
        if win % 2 == 0:
            win -= 1
        ssim_val = float(_ssim(y_true, y_pred, data_range=dr, win_size=win))
    except Exception:
        ssim_val = 0.0
    ssim_val = float(np.nan_to_num(ssim_val))

    # MSC (band-averaged magnitude-squared coherence)
    try:
        f, Cxy = sp_signal.coherence(y_true, y_pred, fs=FS,
                                     nperseg=FS, noverlap=FS // 2)
        band_vals = [
            float(np.nanmean(Cxy[(f >= lo) & (f <= hi)]))
            for lo, hi in BANDS
            if np.any((f >= lo) & (f <= hi))
        ]
        msc = float(np.mean(band_vals)) if band_vals else 0.0
    except Exception:
        msc = 0.0
    msc = float(np.nan_to_num(msc))

    # Amplitude error metrics
    err   = y_pred - y_true
    mae   = float(np.mean(np.abs(err)))
    mse   = float(np.mean(err ** 2))
    rmse  = float(np.sqrt(mse))
    rng   = float(y_true.max() - y_true.min())
    nrmse = float(rmse / (rng + eps))
    sig_p = float(np.sum(y_true ** 2))
    err_p = float(np.sum(err ** 2))
    snr   = float(10.0 * np.log10((sig_p + eps) / (err_p + eps)))

    for v in [mae, rmse, nrmse, snr]:
        v = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=-100.0))

    return {"R2": r2, "ssim": ssim_val, "pcc": pcc, "msc": msc,
            "mae": mae, "rmse": rmse, "nrmse": nrmse, "snr": snr}


# ── N records × 12 leads → per-lead average ──────────────────────────────────

def compute_lead_metrics_batch(
    y_true: np.ndarray,   # (N, T, 12)
    y_pred: np.ndarray,   # (N, T, 12)
    verbose: bool = True,
) -> dict:
    """
    Average metrics over N records for each of the 12 leads.

    Parameters
    ----------
    y_true, y_pred : float32 arrays of shape (N, T, 12)
    verbose        : print per-record progress

    Returns
    -------
    { lead_name: { metric_name: float } }
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    N   = len(y_true)
    acc = {ch: {m: [] for m in METRIC_KEYS} for ch in range(12)}

    for i in range(N):
        if verbose and (i % 200 == 0 or i == N - 1):
            print(f"\r    Metrics: {i+1}/{N} records", end="", flush=True)
        for ch in range(12):
            m = _single_lead_metrics(y_true[i, :, ch], y_pred[i, :, ch])
            for k in METRIC_KEYS:
                acc[ch][k].append(m[k])
    if verbose:
        print()

    return {
        lead: {k: float(np.mean(acc[ch][k])) for k in METRIC_KEYS}
        for ch, lead in enumerate(LEAD_ORDER)
    }
