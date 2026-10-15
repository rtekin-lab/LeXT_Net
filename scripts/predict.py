#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py  —  12-lead ECG reconstruction from 3-lead input (no training needed)
==================================================================================
Loads a pre-trained LeXT-Net checkpoint and reconstructs a full 12-lead ECG
from a 3-lead input recording.

Usage examples:
    # Single CSV file (Chapman format: 5000 rows × 12 cols, 500 Hz)
    python scripts/predict.py --input my_ecg.csv

    # PTB-XL WFDB record
    python scripts/predict.py --input path/to/00001_lr --format wfdb

    # Specify strategy and model explicitly
    python scripts/predict.py --input my_ecg.csv \\
        --dataset ptbxl --strategy EGV3 --model leXT-netv1s

    # Use all 5 folds and average (ensemble)
    python scripts/predict.py --input my_ecg.csv --ensemble

    # Save reconstructed signal as CSV
    python scripts/predict.py --input my_ecg.csv --output reconstructed.csv

    # Plot original 3-lead input and reconstructed 12-lead
    python scripts/predict.py --input my_ecg.csv --plot

Input formats supported:
    csv   : CSV file, rows=samples, cols=12 leads (Chapman format)
            OR rows=samples, cols=3 leads (already reduced)
    wfdb  : WFDB record path (without extension), e.g. records100/00000/00001_lr

Default model: leXT-netv2 (paper model) with EGV3 strategy
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.leXT_net  import TCNTrunk12UNetHeadsExtendedFast
from src.transforms       import apply_strategy
from src.datasets         import _resample_to_target, _ensure_10s, _zscore_normalize

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]

MODEL_CFG = {
    "leXT-netv1s": dict(
        trunk_feat_ch=128, trunk_layers=6, trunk_k=7, trunk_pdrop=0.0,
        use_temporal_gate=False,
        head_base=64, head_pdrop=0.0,
        uniform_depth=True, uniform_depth_val=2,
    ),
    "leXT-netv2s": dict(
        trunk_feat_ch=128, trunk_layers=6, trunk_k=7, trunk_pdrop=0.0,
        use_temporal_gate=True, gate_win=7,
        head_base=64, head_pdrop=0.0,
        limb_depth=1, chest_depth=2,
    ),
    "leXT-netv1": dict(
        trunk_feat_ch=256, trunk_layers=8, trunk_k=9, trunk_pdrop=0.0,
        use_temporal_gate=False,
        head_base=64, head_pdrop=0.0,
        uniform_depth=True, uniform_depth_val=2,
    ),
    "leXT-netv2": dict(
        trunk_feat_ch=256, trunk_layers=8, trunk_k=9, trunk_pdrop=0.0,
        use_temporal_gate=True, gate_win=9,
        head_base=64, head_pdrop=0.0,
        limb_depth=1, chest_depth=3,
    ),
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LeXT-Net: reconstruct 12-lead ECG from 3-lead input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",    required=True,
                   help="Input file path (CSV or WFDB record path)")
    p.add_argument("--format",   default="auto",
                   choices=["auto", "csv", "wfdb"],
                   help="Input format (default: auto-detect from extension)")
    p.add_argument("--fs",       type=int, default=None,
                   help="Input sampling rate in Hz. Recommended for custom data; otherwise inferred from record length.")
    p.add_argument("--dataset",  default="ptbxl",
                   choices=["ptbxl", "chapman", "cpsc2018"],
                   help="Dataset the model was trained on (default: ptbxl)")
    p.add_argument("--strategy", default="EGV3",
                   choices=["EGV1", "EGV2", "EGV3", "QSI"],
                   help="3-lead input strategy used during training (default: EGV3)")
    p.add_argument("--model",    default="leXT-netv2",
                   choices=list(MODEL_CFG.keys()),
                   help="Model variant (default: leXT-netv2)")
    p.add_argument("--fold",     type=int, default=1,
                   choices=[1, 2, 3, 4, 5],
                   help="Which fold checkpoint to use (default: 1)")
    p.add_argument("--ensemble", action="store_true",
                   help="Average predictions from all 5 folds")
    p.add_argument("--weights_dir", default=None,
                   help="Override default weights directory")
    p.add_argument("--output",   default=None,
                   help="Save reconstructed signal as CSV")
    p.add_argument("--plot",     action="store_true",
                   help="Plot input and reconstructed signals")
    p.add_argument("--no_cuda",  action="store_true")
    return p.parse_args()

# ── Input loading ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> np.ndarray:
    """Load CSV ECG file. Returns (T, 12) or (T, 3) float32."""
    import pandas as pd
    df = pd.read_csv(path, header=0 if _has_header(path) else None,
                     encoding="utf-8-sig")
    sig = df.values.astype(np.float32)
    return sig

def _has_header(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
    try:
        float(first.split(",")[0])
        return False
    except ValueError:
        return True

def load_wfdb(path: str) -> np.ndarray:
    """Load WFDB record. Returns (T, 12) float32."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb: pip install wfdb")
    data, header = wfdb.rdsamp(path)
    return data[:, :12].astype(np.float32)

def preprocess(sig: np.ndarray, fs_in: int = None) -> np.ndarray:
    """
    Preprocess raw signal to (1, 1000, 12) model-ready format.
    Handles both 12-lead and 3-lead inputs.
    """
    # Auto-detect sampling rate from shape
    if fs_in is None:
        if sig.shape[0] == 1000:   fs_in = 100
        elif sig.shape[0] == 5000: fs_in = 500
        elif sig.shape[0] == 7500: fs_in = 500
        else:                       fs_in = 500   # assume 500 Hz

    # Resample to 100 Hz if needed
    sig = _resample_to_target(sig, fs_in)          # (≤1000, C)

    # Crop to 10 s
    sig = _ensure_10s(sig, source_label="input")   # (1000, C)
    if sig is None:
        raise ValueError("Signal too short (< 9.5 s after resampling)")

    # Z-score normalise per lead
    sig = _zscore_normalize(sig)                   # (1000, C)

    # If 12-lead, keep as-is; if 3-lead, already reduced
    return sig[np.newaxis]   # (1, 1000, C)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, dataset, strategy, fold, weights_dir, device):
    cfg   = MODEL_CFG[model_name]
    model = TCNTrunk12UNetHeadsExtendedFast(
        input_size=3, output_size=12, **cfg)

    if weights_dir is not None:
        wdir = Path(weights_dir)
    else:
        wdir = ROOT / "results" / "weights" / dataset / strategy / model_name

    pth = wdir / f"fold{fold}.pth"
    if not pth.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {pth}\n"
            f"Run first:  python scripts/download_weights.py "
            f"--dataset {dataset} --strategy {strategy} --model {model_name}"
        )
    state = torch.load(pth, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, x3: np.ndarray, device) -> np.ndarray:
    """x3: (1, 1000, 3) → returns (1000, 12) float32"""
    t = torch.from_numpy(x3).to(device)   # (1, 1000, 3)
    y = model(t)                           # (1, 1000, 12)
    return y.squeeze(0).cpu().numpy()      # (1000, 12)

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_result(x3: np.ndarray, y12: np.ndarray, strategy: str):
    """Plot 3-lead input and 12-lead reconstruction side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    from src.transforms import STRATEGY_NAMES
    t = np.arange(1000) / 100   # seconds

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"LeXT-Net Reconstruction  |  Input strategy: {strategy}",
                 fontsize=13, fontweight="bold")

    for i, (ax, lead) in enumerate(zip(axes, LEAD_ORDER)):
        ax.plot(t, y12[:, i], color="#1A7A8A", linewidth=0.9, label="Reconstructed")
        ax.set_ylabel(lead, fontsize=8, rotation=0, labelpad=22)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, color="#cccccc")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cpu" if (args.no_cuda or
                           not torch.cuda.is_available()) else "cuda")

    print(f"\nLeXT-Net Inference")
    print(f"  Model    : {args.model}")
    print(f"  Dataset  : {args.dataset}   Strategy: {args.strategy}")
    print(f"  Device   : {device}")
    print(f"  Input    : {args.input}\n")

    # ── Load input ────────────────────────────────────────────────
    fmt = args.format
    if fmt == "auto":
        fmt = "wfdb" if not args.input.endswith(".csv") else "csv"

    if fmt == "csv":
        sig = load_csv(args.input)
        # Auto-detect fs from row count
        fs  = args.fs if args.fs is not None else (500 if sig.shape[0] >= 3000 else 100)
    else:
        sig = load_wfdb(args.input)
        # WFDB header has fs — assume 100 for PTB-XL lr, 500 otherwise
        fs  = args.fs if args.fs is not None else (100 if sig.shape[0] == 1000 else 500)

    print(f"  Raw shape: {sig.shape}  (fs={fs} Hz)")

    # ── Preprocess → 12-lead (1, 1000, 12) ───────────────────────
    if sig.shape[1] == 12:
        full12 = preprocess(sig, fs_in=fs)   # (1, 1000, 12)
        x3     = apply_strategy(full12, args.strategy)  # (1, 1000, 3)
        print(f"  12-lead input detected — applying {args.strategy} to get 3-lead input")
    elif sig.shape[1] == 3:
        x3     = preprocess(sig, fs_in=fs)   # already 3-lead
        full12 = None
        print(f"  3-lead input detected — using as-is")
    else:
        raise ValueError(f"Expected 3 or 12 columns, got {sig.shape[1]}")

    # ── Load model(s) and infer ───────────────────────────────────
    folds = list(range(1, 6)) if args.ensemble else [args.fold]
    preds = []

    for fold in folds:
        model = load_model(args.model, args.dataset, args.strategy,
                           fold, args.weights_dir, device)
        n_p = sum(p.numel() for p in model.parameters())
        if fold == folds[0]:
            print(f"  Parameters: {n_p:,}")
        pred = infer(model, x3, device)   # (1000, 12)
        preds.append(pred)
        print(f"  Fold {fold} ✓")

    y12 = np.mean(preds, axis=0)   # (1000, 12) — ensemble average

    if args.ensemble:
        print(f"  Ensemble: averaged {len(folds)} fold predictions")

    # ── Output ────────────────────────────────────────────────────
    print(f"\n  Reconstruction complete — shape: {y12.shape}")
    print("  Output space: per-record/per-lead normalized units (z-score).")
    print("  Physical target-lead amplitudes cannot be recovered from 3 input leads alone under this training normalization.")

    if args.output:
        import pandas as pd
        df_out = pd.DataFrame(y12, columns=LEAD_ORDER)
        df_out.to_csv(args.output, index=False)
        print(f"  Saved to: {args.output}")

    if args.plot:
        plot_result(x3.squeeze(0), y12, args.strategy)

    if not args.output and not args.plot:
        # Print quick summary stats
        print(f"\n  Lead-wise amplitude summary (reconstructed):")
        print(f"  {'Lead':6s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}")
        print(f"  {'─'*36}")
        for i, lead in enumerate(LEAD_ORDER):
            mn, mx, sd = y12[:,i].min(), y12[:,i].max(), y12[:,i].std()
            print(f"  {lead:6s}  {mn:>8.4f}  {mx:>8.4f}  {sd:>8.4f}")
        print(f"\n  Tip: use --output out.csv to save, --plot to visualise")


if __name__ == "__main__":
    main()
