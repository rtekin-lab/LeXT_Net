#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_multidataset.py  —  LeXT-Net evaluation across multiple datasets
========================================================================
Trains LeXT-Net (v1 and/or v2) on PTB-XL, Chapman, or CPSC2018 and
writes per-dataset results to all_results_Norm_difDataSet.xlsx.

Usage:
    python scripts/train_multidataset.py --dataset ptbxl    --strategy ALL
    python scripts/train_multidataset.py --dataset chapman  --strategy ALL --model ALL
    python scripts/train_multidataset.py --dataset cpsc2018 --strategy EGV3 --model leXT-netv2

Supported --dataset : ptbxl | cpsc2018 | chapman | custom
Supported --strategy: EGV1 | EGV2 | EGV3 | QSI | ALL
Supported --model   : leXT-netv1 | leXT-netv2 | ALL  (default: leXT-netv2)

Output:
    results/all_results_Norm_difDataSet.xlsx
      PTB-XL    → LeXT-Net(ptbxl)
      Chapman   → LeXT-Net(chapman)
      CPSC2018  → LeXT-Net(cpsc2018)
    results/weights/<dataset>/<strategy>/<model>/fold<k>.pth
"""

import os, sys, argparse, time, warnings
warnings.filterwarnings("ignore", "The epoch parameter in", UserWarning)
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets   import load_ptbxl, load_cpsc2018, load_chapman
from src.custom_data import load_custom_manifest
from src.transforms import apply_strategy
from src.crossval   import sequential_patient_folds, get_fold
from src.utils      import (ECGDataset, compute_lead_metrics_batch,
                             LEAD_ORDER, METRIC_KEYS)
from configs.paths  import PATHS
from src.models.leXT_net import (TCNTrunk12UNetHeadsExtended,
                                  TCNTrunk12UNetHeads,
                                  TCNTrunk12UNetHeadsExtendedFast,
                                  CombinedLoss)

# ── GPU / AMP / Compile ───────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")

try:
    from torch.amp import autocast, GradScaler
    def _amp(dev): return autocast(device_type=dev, enabled=(dev=="cuda"))
    def _scaler(dev): return GradScaler(device=dev, enabled=(dev=="cuda"))
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    def _amp(dev): return autocast(enabled=(dev=="cuda"))
    def _scaler(dev): return GradScaler(enabled=(dev=="cuda"))


class TensorLoader:
    """
    GPU-resident data loader — zero CPU-GPU transfer overhead during training.

    Loads the entire dataset into GPU memory once at construction time.
    Each __iter__ call yields (x_batch, y_batch, dummy_fnames) slices
    directly from GPU tensors — no DataLoader workers, no per-batch transfer.

    Memory estimate:
        21,837 records × 1,000 samples × 12 leads × 4 bytes ≈ 1.0 GB
        21,837 records × 1,000 samples × 3  leads × 4 bytes ≈ 0.26 GB
        Total ≈ 1.3 GB — fits comfortably in 4 GB VRAM alongside the model.

    Falls back to CPU if the data does not fit in GPU memory.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 batch_size: int = 32, shuffle: bool = True,
                 device: torch.device = torch.device("cuda")):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.device     = device
        self.n          = len(x)

        # Move to GPU once — pinned→device for maximum bandwidth
        try:
            self.x = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
            self.y = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
        except RuntimeError:
            # Fallback: not enough VRAM — stay on CPU, transfer per batch
            self.x = torch.from_numpy(x)
            self.y = torch.from_numpy(y)
            self.device = torch.device("cpu")

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        idx = (torch.randperm(self.n, device=self.device)
               if self.shuffle else torch.arange(self.n, device=self.device))
        for start in range(0, self.n, self.batch_size):
            sl = idx[start : start + self.batch_size]
            # yield dummy filename list to match ECGDataset interface
            yield self.x[sl], self.y[sl], [""] * len(sl)

def maybe_compile(model, use_compile=True):
    """torch.compile() — disabled by default on Windows.

    torch.compile requires Triton which is only available on Linux/WSL2.
    On native Windows this function always returns the model unchanged.
    The speed gains from BatchedHeadGroup + TensorLoader make compile
    less critical anyway.

    To enable on Linux/WSL2, pass --compile explicitly.
    """
    if not use_compile:
        return model
    # Check platform first — avoid long Triton init on Windows
    import platform
    if platform.system() == "Windows":
        return model   # Triton not supported on native Windows
    try:
        import triton  # noqa: F401
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model

# ── Excel map ─────────────────────────────────────────────────────────────────
STRATEGY_ORDER = ["EGV2", "EGV3", "EGV1", "QSI"]
STRATEGY_COL   = {"EGV2": 3, "EGV3": 11, "EGV1": 19, "QSI": 27}
LEAD_ROWS_MEAN = {"I":4,"II":5,"III":6,"AVR":7,"AVL":8,"AVF":9,
                   "V1":10,"V2":11,"V3":12,"V4":13,"V5":14,"V6":15}
LEAD_ROWS_STD  = {k: v+18 for k,v in LEAD_ROWS_MEAN.items()}
DATASET_SHEET  = {"ptbxl":"LeXT-Net(ptbxl)",
                   "chapman":"LeXT-Net(chapman)",
                   "cpsc2018":"LeXT-Net(cpsc2018)"}
EXCEL_PATH     = ROOT / "results" / "all_results_Norm_difDataSet.xlsx"

# ── Model configurations ──────────────────────────────────────────────────────
MODEL_CFG = {
    "leXT-netv1": dict(
        trunk_feat_ch=256, trunk_layers=8, trunk_k=9, trunk_pdrop=0.1,
        use_temporal_gate=False,
        head_base=64, head_pdrop=0.0,
        uniform_depth=True, uniform_depth_val=2,
    ),
    "leXT-netv1s": dict(   # Slim: ~4x faster, F=128 L=6 k=7
        trunk_feat_ch=128, trunk_layers=6, trunk_k=7, trunk_pdrop=0.1,
        use_temporal_gate=False,
        head_base=64, head_pdrop=0.0,
        uniform_depth=True, uniform_depth_val=2,
    ),
    "leXT-netv2": dict(
        trunk_feat_ch=256, trunk_layers=8, trunk_k=9, trunk_pdrop=0.1,
        use_temporal_gate=True, gate_win=9,
        head_base=64, head_pdrop=0.0,
        limb_depth=1, chest_depth=3,
    ),
    "leXT-netv2s": dict(   # Slim: ~4.2x faster, F=128 L=6 k=7 chest_d=2
        trunk_feat_ch=128, trunk_layers=6, trunk_k=7, trunk_pdrop=0.1,
        use_temporal_gate=True, gate_win=7,
        head_base=64, head_pdrop=0.0,
        limb_depth=1, chest_depth=2,
    ),
}
MODEL_ORDER = ["leXT-netv1", "leXT-netv1s", "leXT-netv2", "leXT-netv2s"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LeXT-Net multi-dataset evaluation")
    p.add_argument("--dataset",    default="chapman",
                   choices=["ptbxl","cpsc2018","chapman","custom"])
    p.add_argument("--model",      default="leXT-netv2",
                   choices=["leXT-netv1","leXT-netv1s","leXT-netv2","leXT-netv2s","ALL"],
                   help="leXT-netv1 | leXT-netv1s | leXT-netv2 | leXT-netv2s | ALL")
    p.add_argument("--strategy",   default="ALL",
                   choices=["EGV1","EGV2","EGV3","QSI","ALL"])
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--folds",      type=int,   default=5,
                   help="Number of CV folds used to define the split (paper protocol: 5)")
    p.add_argument("--fold",       type=int, default=None, choices=[1,2,3,4,5],
                   help="Train/evaluate only this fold from the fixed n-fold split; default: all folds")
    p.add_argument("--manifest",   default=None,
                   help="Manifest CSV for --dataset custom")
    p.add_argument("--data_root",  default=None,
                   help="Optional base directory for relative paths in a custom manifest")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no_cuda",    action="store_true")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile() (use if PyTorch < 2.0)")
    p.add_argument("--no_excel",   action="store_true")
    p.add_argument("--max_rec",    type=int,   default=None)
    return p.parse_args()

# ── Progress ──────────────────────────────────────────────────────────────────

def log(msg="", end="\n"): print(msg, end=end, flush=True)

def bar(cur, total, msg="", w=28):
    f = int(w * cur / max(total,1))
    print(f"\r  [{'█'*f}{'░'*(w-f)}] {cur/max(total,1)*100:5.1f}%  {msg}",
          end="", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(name="leXT-netv2"):
    cfg = MODEL_CFG[name]
    return TCNTrunk12UNetHeadsExtendedFast(input_size=3, output_size=12, **cfg)

# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(model, x_tr, y_tr, model_name, epochs, batch_size, device,
               n_workers, use_compile):
    dev  = device.type
    sc   = _scaler(dev)
    model = model.to(device)
    if use_compile:
        model = maybe_compile(model)

    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs-5), eta_min=1e-6)
    sch    = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[5])
    crit   = CombinedLoss(w_mae=0.70, w_pcc=0.25, w_cons=0.05,
                          w_smooth=0.02, ramp_epochs=20)

    # TensorLoader: pre-load entire fold to GPU — zero per-batch CPU transfer
    t_load = time.time()
    dl = TensorLoader(x_tr, y_tr, batch_size=batch_size,
                      shuffle=True, device=device)
    log(f"  Data pre-loaded to {dl.device} in {time.time()-t_load:.1f}s  "
        f"({x_tr.nbytes/1e6:.0f}+{y_tr.nbytes/1e6:.0f} MB)")

    t0 = time.time()
    for epoch in range(epochs):
        model.train(); ep = torch.tensor(0.0, device=device)
        for step, (xb, yb, _) in enumerate(dl):
            # xb, yb already on GPU (TensorLoader pre-loaded)
            opt.zero_grad(set_to_none=True)
            with _amp(dev):
                loss = crit(model(xb), yb, epoch=epoch)
            sc.scale(loss).backward()
            sc.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt); sc.update()
            ep += loss.detach()  # no GPU→CPU sync per step
        sch.step()
        avg = ep.item() / dl.n  # single GPU→CPU sync per epoch
        eta = (time.time()-t0)/(epoch+1) * (epochs-epoch-1)
        bar(epoch+1, epochs,
            f"epoch {epoch+1:3d}/{epochs}  loss={avg:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  ETA {eta/60:.1f}min")
    print()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, x_te, y_te, device, batch=64):
    model.eval()
    dl  = TensorLoader(x_te, y_te, batch_size=batch,
                       shuffle=False, device=device)
    out = []
    for i, (xb, _, _) in enumerate(dl):
        out.append(model(xb).cpu().numpy())  # xb already on GPU
        bar(min((i+1)*batch, len(x_te)), len(x_te), "Inference")
    print()
    return np.concatenate(out)

# ── Excel write ───────────────────────────────────────────────────────────────

def write_excel(dataset, strategy, lead_means, lead_stds, model_name="leXT-netv2"):
    if not EXCEL_PATH.exists():
        warnings.warn(f"Excel not found: {EXCEL_PATH}"); return
    wb  = openpyxl.load_workbook(str(EXCEL_PATH))
    sn  = DATASET_SHEET[dataset]   # always the LeXT-Net sheet for this dataset
    if sn not in wb.sheetnames:
        warnings.warn(f"Sheet '{sn}' not found"); wb.close(); return
    ws  = wb[sn]; c0 = STRATEGY_COL[strategy]
    # Write metrics
    for lead in LEAD_ORDER:
        for mi, metric in enumerate(METRIC_KEYS):
            ws.cell(LEAD_ROWS_MEAN[lead], c0+mi, lead_means[lead][metric])
            ws.cell(LEAD_ROWS_STD[lead],  c0+mi, lead_stds[lead][metric])
    # Write model version note in row 1 (above the strategy header row 2)
    # Cell: same column block as strategy, row 1
    ws.cell(row=1, column=c0, value=f"Model: {model_name}")
    wb.save(str(EXCEL_PATH)); wb.close()
    log(f"  Excel updated — sheet='{sn}'  strategy={strategy}  model={model_name}")

# ── Single strategy CV ────────────────────────────────────────────────────────

def run_strategy(strategy, model_name, signals, rec_ids, patient_ids, dataset, args, device):
    log(f"\n{'='*62}")
    log(f"  STRATEGY: {strategy}   Dataset: {dataset}   Model: {model_name}")
    log(f"  CV folds: {args.folds}   Selected fold: {args.fold or 'ALL'}   Epochs: {args.epochs}   Batch: {args.batch}")
    log(f"{'='*62}")

    inputs  = apply_strategy(signals, strategy)
    targets = signals
    folds   = sequential_patient_folds(rec_ids, patient_ids, n_folds=args.folds)
    acc     = {ld:{m:[] for m in METRIC_KEYS} for ld in LEAD_ORDER}
    wdir    = ROOT/"results"/"weights"/dataset/strategy/model_name
    wdir.mkdir(parents=True, exist_ok=True)
    use_compile = not args.no_compile

    selected = list(enumerate(folds, start=1))
    if args.fold is not None:
        if args.fold > len(folds):
            raise ValueError(f"--fold {args.fold} exceeds --folds {args.folds}")
        selected = [(args.fold, folds[args.fold - 1])]

    for fold_no, fold in selected:
        t0 = time.time()
        k = fold_no - 1
        log(f"\n  -- Fold {fold_no}/{args.folds} {'─'*42}")
        x_tr, x_te, y_tr, y_te = get_fold(inputs, targets, fold)
        log(f"  Train: {len(x_tr):,}   Test: {len(x_te):,}")

        model = build_model(model_name)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Parameters: {n_p:,}")

        log("  [Training]")
        model = train_fold(model, x_tr, y_tr, model_name,
                           args.epochs, args.batch,
                           device, args.workers, use_compile)
        torch.save(model.state_dict(), str(wdir/f"fold{fold_no}.pth"))

        log("  [Inference]")
        preds = predict(model, x_te, y_te, device, args.batch)

        log("  [Metrics]")
        lm = compute_lead_metrics_batch(y_te, preds, verbose=True)
        for ld in LEAD_ORDER:
            for m in METRIC_KEYS:
                acc[ld][m].append(lm[ld][m])

        r2  = np.mean([lm[ld]["R2"]  for ld in LEAD_ORDER])
        snr = np.mean([lm[ld]["snr"] for ld in LEAD_ORDER])
        log(f"  Fold {fold_no} done  mAll R²={r2:.4f}  SNR={snr:.2f} dB  "
            f"[{(time.time()-t0)/60:.1f} min]")

    means = {ld:{m:float(np.mean(acc[ld][m]))for m in METRIC_KEYS}
             for ld in LEAD_ORDER}
    stds  = {ld:{m:float(np.std( acc[ld][m]))for m in METRIC_KEYS}
             for ld in LEAD_ORDER}

    log(f"\n  ┌─ Summary: {strategy} / {dataset} {'─'*28}┐")
    for m in METRIC_KEYS:
        mu  = np.mean([means[ld][m] for ld in LEAD_ORDER])
        std = np.mean([stds[ld][m]  for ld in LEAD_ORDER])
        log(f"  │ {m:8s}  {mu:>12.6f} ± {std:<10.6f}│")
    log(f"  └{'─'*50}┘")

    if not args.no_excel and dataset in DATASET_SHEET and args.fold is None:
        write_excel(dataset, strategy, means, stds, model_name=model_name)
    elif not args.no_excel and dataset in DATASET_SHEET and args.fold is not None:
        log("  Single-fold run: Excel template update skipped to avoid overwriting 5-fold summary tables.")
    elif dataset == "custom" and not args.no_excel:
        log("  Custom dataset: Excel template update skipped; weights and console metrics were saved.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cpu" if (args.no_cuda or
                           not torch.cuda.is_available()) else "cuda")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    models     = MODEL_ORDER if args.model == "ALL" else [args.model]
    strategies = STRATEGY_ORDER if args.strategy == "ALL" else [args.strategy]

    log(f"\n{'='*62}")
    log(f"  Device: {device}   Dataset: {args.dataset}")
    log(f"  Models: {models}")
    log(f"  Strategies: {strategies}")
    log(f"  torch.compile: {'disabled' if args.no_compile else 'enabled (Linux/WSL2 only)'}")
    log(f"{'='*62}")

    log(f"\n[1] Loading {args.dataset} ...")
    t0 = time.time(); patient_ids = None

    if args.dataset == "ptbxl":
        signals, rec_ids, meta = load_ptbxl(
            PATHS["ptbxl"], normalize=True,
            max_records=args.max_rec, verbose=True)
        try:
            id2p = {str(int(e)): str(int(meta.loc[int(e), "patient_id"]))
                    for e in rec_ids}
            patient_ids = [id2p.get(r, r) for r in rec_ids]
            log(f"  Patient-wise split: {len(set(patient_ids)):,} patients")
        except Exception as e:
            warnings.warn(f"Patient IDs unavailable ({e})")
    elif args.dataset == "cpsc2018":
        signals, rec_ids = load_cpsc2018(
            PATHS["cpsc2018"], normalize=True,
            max_records=args.max_rec, verbose=True)
    elif args.dataset == "chapman":
        signals, rec_ids = load_chapman(
            PATHS["chapman"], normalize=True,
            max_records=args.max_rec, verbose=True)
    elif args.dataset == "custom":
        if not args.manifest:
            raise ValueError("--manifest is required when --dataset custom")
        signals, rec_ids, patient_ids = load_custom_manifest(
            args.manifest, data_root=args.data_root, normalize=True,
            max_records=args.max_rec, verbose=True)
        log(f"  Patient-wise split: {len(set(patient_ids)):,} patient/group IDs")

    log(f"  {signals.shape}  [{(time.time()-t0)/60:.1f} min]")
    assert signals.shape[1]==1000 and signals.shape[2]==12

    t_all = time.time()
    total = len(models) * len(strategies)
    done  = 0
    for model_name in models:
        for si, strategy in enumerate(strategies):
            done += 1
            log(f"\n[{done}/{total}] {model_name} × {strategy}")
            run_strategy(strategy, model_name, signals, rec_ids,
                         patient_ids, args.dataset, args, device)

    log(f"\n{'='*62}")
    log(f"  ALL DONE  [{(time.time()-t_all)/60:.1f} min]")
    log(f"  Excel: {EXCEL_PATH}")
    log(f"{'='*62}\n")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
