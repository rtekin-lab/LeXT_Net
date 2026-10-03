# LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23123820-blue)](https://doi.org/10.5281/zenodo.23123820)

Official implementation of the paper:

> **LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction**
> Ramazan Tekin, Hazret Tekin, Mine Onat, Merve Akkuş, Yılmaz Kaya
> *Scientific Reports* (under review) · [Zenodo](https://doi.org/10.5281/zenodo.23123820)

---

## Overview

Standard 12-lead ECG acquisition requires precise placement of ten electrodes by trained personnel — a constraint that cannot always be met in ambulatory monitoring, wearable sensing, or resource-limited settings. LeXT-Net reconstructs a full 12-lead ECG from only three input leads in a single forward pass.

**Architecture:**
- **[P1] Shared TCN backbone** — dilated residual blocks for global temporal context
- **[H1] Batched Temporal Attention Gate** — per-lead temporal weighting, single grouped-convolution call
- **[H2] Batched Projection** — channel reduction for all 12 leads simultaneously
- **[P3] Lead-adaptive U-Net heads** — shallow for Einthoven-derivable limb leads, deeper for precordial
- **Physics-informed loss** — Einthoven–Goldberger constraints + precordial smoothness + curriculum ramp

---

## Quick Start — No Training Required

Pre-trained weights are on Zenodo. Download and run inference immediately.

```bash
git clone https://github.com/rtekin-lab/LeXT-Net.git
cd LeXT-Net
pip install -r requirements.txt

# Download EGV3 weights (~100 MB)
python scripts/download_weights.py --strategy EGV3

# Reconstruct 12-lead ECG from a CSV file
python scripts/predict.py --input your_ecg.csv --plot

# Save output
python scripts/predict.py --input your_ecg.csv --output reconstructed.csv --ensemble
```

See [WEIGHTS.md](WEIGHTS.md) for the full weight catalogue and manual download instructions.

---

## Repository Structure

```
LeXT-Net/
├── src/
│   ├── models/
│   │   ├── leXT_net.py        # LeXT-Net variants + CombinedLoss
│   │   └── baselines.py       # BiLSTM, TCN, ECG-Transformer, UNet++, AtUNET
│   ├── datasets.py            # PTB-XL, CPSC2018, Chapman loaders
│   ├── transforms.py          # EGV1 / EGV2 / EGV3 / QSI input strategies
│   ├── crossval.py            # Sequential 5-fold CV, patient-wise
│   └── utils.py               # Dataset, normalisation, metrics
├── scripts/
│   ├── predict.py             # ← Inference with pre-trained weights
│   ├── download_weights.py    # ← Download weights from Zenodo
│   ├── train_ptbxl.py         # PTB-XL: all models comparison
│   └── train_multidataset.py  # LeXT-Net: multi-dataset evaluation
├── configs/
│   └── paths.py               # ← Set dataset paths here before training
├── results/
│   ├── all_results_Norm_R.xlsx
│   └── all_results_Norm_difDataSet.xlsx
├── WEIGHTS.md
└── requirements.txt
```

---

## predict.py

```bash
# Default: EGV3 strategy, leXT-netv1s, PTB-XL weights, fold 1
python scripts/predict.py --input ecg.csv

# Ensemble (average all 5 folds — more robust)
python scripts/predict.py --input ecg.csv --ensemble --output out.csv

# WFDB record (PTB-XL format)
python scripts/predict.py --input records100/00000/00001_lr --format wfdb

# Different strategy
python scripts/predict.py --input ecg.csv --strategy EGV1
```

**Supported input formats:**

| Format | Columns | Sampling rate |
|---|---|---|
| CSV (Chapman / MUSE) | 12 or 3 | 500 Hz → auto-resampled to 100 Hz |
| WFDB `.dat` + `.hea` | 12 | 100 Hz (PTB-XL records100) |

If a 12-lead CSV is provided, the selected strategy is applied to extract the 3-lead input automatically.

---

## Training From Scratch

Edit `configs/paths.py` first, then:

```bash
# PTB-XL — all models × all strategies × 5-fold CV
python scripts/train_ptbxl.py --model ALL --strategy ALL

# Single model (recommended first run)
python scripts/train_ptbxl.py --model leXT-netv1s --strategy EGV3

# Quick test
python scripts/train_ptbxl.py --model leXT-netv1s --strategy EGV3 \
    --max_rec 200 --epochs 3

# Multi-dataset (LeXT-Net only)
python scripts/train_multidataset.py --dataset chapman  --strategy ALL --model leXT-netv2s
python scripts/train_multidataset.py --dataset cpsc2018 --strategy EGV3
```

`--model` options: `bilstm` | `tcn` | `atunet` | `unetpp` | `leXT-netv1` | `leXT-netv1s` | `leXT-netv2` | `leXT-netv2s` | `ALL`

---

## Datasets

| Dataset | Records | Fs | Download |
|---|---|---|---|
| **PTB-XL** | 21,837 | 100 Hz | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) — DOI: 10.13026/kfzx-aw45 |
| **CPSC 2018** | 6,877 | 500 → 100 Hz | [ICBEB 2018](http://2018.icbeb.org/Challenge.html) |
| **Chapman-Shaoxing** | 10,646 | 500 → 100 Hz | [Figshare](https://figshare.com/collections/ChapmanECG/4560497) |

Datasets are not distributed with this repository.

---

## Model Variants

| Model | Params | Speed | Notes |
|---|---|---|---|
| `leXT-netv2` | 18.6 M | 1× | Full model (paper results) |
| `leXT-netv1` | ~11 M | 1× | No temporal gate, uniform head depth |
| `leXT-netv2s` | ~4.5 M | ~4× | Slim: F=128, L=6, k=7, chest_d=2 |
| **`leXT-netv1s`** | **~3 M** | **~4×** | **Default for inference** |

---

## Experimental Protocol

| Parameter | Value |
|---|---|
| Cross-validation | Sequential 5-fold, patient-wise |
| Sampling rate | 100 Hz |
| Recording length | 10 s (1,000 samples) |
| Normalisation | Per-record, per-lead z-score |
| Evaluation | mAll · mOther · mInd |
| Metrics | R², SSIM, PCC, MSC, MAE, RMSE, NRMSE, SNR |
| Optimiser | AdamW lr=3×10⁻⁴, wd=10⁻⁴ |
| Scheduler | 5-epoch warmup + cosine annealing |
| Loss | 0.70·MAE + 0.25·(1−PCC) + 0.05·Physics + 0.02·Smooth |

---

## Citation

```bibtex
@article{tekin2025lextnet,
  title   = {{LeXT-Net}: A Lead-Adaptive Hierarchical Network for
             3-Lead to 12-Lead {ECG} Reconstruction},
  author  = {Tekin, Ramazan and Tekin, Hazret and Onat, Mine and
             Akku{\c{s}}, Merve and Kaya, Y{\i}lmaz},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.5281/zenodo.23123820},
  note    = {Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
Data and weights are subject to their original licences (PTB-XL: CC-BY 4.0).

## Contact

Ramazan Tekin · Batman University · ramazan.tekin@batman.edu.tr
