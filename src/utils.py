# -*- coding: utf-8 -*-
"""
utils.py  —  Shared utilities
================================
ECGDataset, per-record normalisation / denormalisation,
and the full lead-wise ECG metric computation pipeline.
No external project-specific dependencies.
"""

import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal as sp_signal
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as _ssim

from src.metrics import (compute_lead_metrics_batch,  # re-export
                          LEAD_ORDER, METRIC_KEYS, FS, BANDS,
                          _single_lead_metrics)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """Returns (input, target, filename) triples."""
    def __init__(self, data_A, data_B, filenames=None):
        self.A = torch.tensor(data_A, dtype=torch.float32)
        self.B = torch.tensor(data_B, dtype=torch.float32)
        self.filenames = (filenames if filenames is not None
                          else [str(i) for i in range(len(data_A))])

    def __len__(self):  return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.filenames[idx]


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_per_record_leadwise(x: np.ndarray, mode: str = "zscore",
                                   eps: float = 1e-8):
    """
    Per-record, per-lead normalisation.

    Parameters
    ----------
    x    : float array of shape (N, T, C)
    mode : "zscore" | "minmax"

    Returns
    -------
    x_norm : normalised array, same shape as x
    stats  : dict of normalisation parameters (for denormalisation)
    """
    assert x.ndim == 3
    x = x.astype(np.float32, copy=True)
    if mode == "zscore":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1,  keepdims=True) + eps
        return ((x - mu) / sd).astype(np.float32), {"mode": "zscore", "mu": mu, "sd": sd}
    if mode == "minmax":
        mn  = x.min(axis=1, keepdims=True)
        mx  = x.max(axis=1, keepdims=True)
        rng = (mx - mn) + eps
        return ((x - mn) / rng).astype(np.float32), {"mode": "minmax", "mn": mn, "rng": rng}
    raise ValueError(f"Unknown normalisation mode: {mode}")


def denormalize_per_record_leadwise(x_norm: np.ndarray, stats: dict):
    """Inverse of normalize_per_record_leadwise."""
    assert x_norm.ndim == 3
    mode = stats["mode"]
    if mode == "zscore":
        return (x_norm * stats["sd"] + stats["mu"]).astype(np.float32)
    if mode == "minmax":
        return (x_norm * stats["rng"] + stats["mn"]).astype(np.float32)
    raise ValueError(f"Unknown normalisation mode: {mode}")


# ── Full evaluation pipeline (legacy helper) ──────────────────────────────────

def evaluate_performanceFull(model, test_loader, device,
                              stats=None, test_indices=None):
    """
    Run model on test_loader, optionally denormalise, compute all metrics.

    Returns a list of rows: [filename, metric1_lead1, ..., metric13_lead12]
    Compatible with the original calisma1_libs interface.
    """
    import scipy.stats as scipy_stats
    from sklearn.metrics import r2_score as _r2
    from skimage.metrics import structural_similarity as _ssim2

    model.eval()
    trues, preds, fnames = [], [], []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels, fn = data
            trues.append(labels.cpu().numpy())
            preds.append(model(inputs.to(device)).cpu().numpy())
            fnames.extend(list(fn))

    trues = np.concatenate(trues)    # (N, T, 12)
    preds = np.concatenate(preds)

    if stats is not None and test_indices is not None:
        stats_te = {k: (v[test_indices] if isinstance(v, np.ndarray) else v)
                    for k, v in stats.items()}
        trues = denormalize_per_record_leadwise(trues, stats_te)
        preds = denormalize_per_record_leadwise(preds, stats_te)

    results = []
    N = len(trues)
    eps = 1e-12

    METRIC_FNS = [
        "pcorr", "scorr", "kcorr", "r2", "mse",
        "ncc", "msc", "ssim", "mae", "rmse", "snr", "nrmse",
    ]

    for i, (t, p, fn) in enumerate(zip(trues, preds, fnames)):
        print(f"\r  Evaluating: {i+1}/{N}", end="", flush=True)
        row = [fn]
        for mfn in METRIC_FNS:
            for ch in range(12):
                tc = np.nan_to_num(t[:, ch].astype(np.float64))
                pc = np.nan_to_num(p[:, ch].astype(np.float64))
                try:
                    if mfn == "pcorr":
                        v = float(scipy_stats.pearsonr(tc, pc)[0]) if tc.std() > eps else 0.0
                    elif mfn == "scorr":
                        v = float(scipy_stats.spearmanr(tc, pc)[0]) if tc.std() > eps else 0.0
                    elif mfn == "kcorr":
                        v = float(scipy_stats.kendalltau(tc, pc)[0]) if tc.std() > eps else 0.0
                    elif mfn == "r2":
                        v = float(_r2(tc, pc))
                    elif mfn == "mse":
                        v = float(np.mean((tc - pc) ** 2))
                    elif mfn == "ncc":
                        xz = (tc - tc.mean()) / (tc.std() + eps)
                        yz = (pc - pc.mean()) / (pc.std() + eps)
                        v  = float(np.max(np.abs(np.correlate(xz, yz, "full")))
                                   / (len(xz) + eps))
                    elif mfn == "msc":
                        f, Cxy = sp_signal.coherence(tc, pc, fs=FS, nperseg=FS)
                        bv = [float(np.nanmean(Cxy[(f >= lo) & (f <= hi)]))
                              for lo, hi in BANDS
                              if np.any((f >= lo) & (f <= hi))]
                        v = float(np.mean(bv)) if bv else 0.0
                    elif mfn == "ssim":
                        dr = float(max(tc.max(), pc.max()) - min(tc.min(), pc.min()))
                        if dr < eps: dr = 1.0
                        v = float(_ssim2(tc, pc, data_range=dr,
                                         win_size=min(101, len(tc) // 2 * 2 - 1)))
                    elif mfn == "mae":
                        v = float(np.mean(np.abs(tc - pc)))
                    elif mfn == "rmse":
                        v = float(np.sqrt(np.mean((tc - pc) ** 2)))
                    elif mfn == "snr":
                        v = float(10 * np.log10(
                            (np.sum(tc ** 2) + eps) / (np.sum((tc - pc) ** 2) + eps)))
                    elif mfn == "nrmse":
                        rng = float(tc.max() - tc.min())
                        v   = float(np.sqrt(np.mean((tc - pc) ** 2)) / (rng + eps))
                    else:
                        v = 0.0
                except Exception:
                    v = 0.0
                row.append(float(np.nan_to_num(v, nan=0.0)))
        results.append(row)
    print()
    return results
