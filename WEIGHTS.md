# Pre-trained Model Weights

Pre-trained LeXT-Net weights are published on Zenodo alongside the paper:

> **Zenodo DOI:** https://doi.org/10.5281/zenodo.23123820

---

## Available Checkpoints

| Dataset | Strategy | Model | Folds | Size |
|---|---|---|---|---|
| PTB-XL | EGV1, EGV2, EGV3, QSI | leXT-netv1s | 5 × 4 = 20 | ~400 MB |

Each fold checkpoint is a standard PyTorch state dict (`.pth`).

Expected directory layout after download:

```
results/weights/
└── ptbxl/
    ├── EGV1/leXT-netv1s/fold1.pth … fold5.pth
    ├── EGV2/leXT-netv1s/fold1.pth … fold5.pth
    ├── EGV3/leXT-netv1s/fold1.pth … fold5.pth
    └── QSI/ leXT-netv1s/fold1.pth … fold5.pth
```

---

## Quick Download

```bash
# Download all weights automatically
python scripts/download_weights.py

# Download only EGV3 weights (best strategy, ~100 MB)
python scripts/download_weights.py --strategy EGV3

# List available files without downloading
python scripts/download_weights.py --list
```

---

## Manual Download

If the automatic downloader fails, visit:

**https://doi.org/10.5281/zenodo.23123820**

Download the archive and extract it so that `.pth` files land in
`results/weights/<dataset>/<strategy>/<model>/fold<k>.pth`.

---

## Run Inference Without Training

Once weights are downloaded:

```bash
# Reconstruct 12-lead ECG from a Chapman-format CSV (500 Hz, 12 cols)
python scripts/predict.py --input your_ecg.csv

# Use EGV3 strategy (best), ensemble of all 5 folds, save output
python scripts/predict.py --input your_ecg.csv \
    --strategy EGV3 --ensemble --output reconstructed.csv

# Plot the reconstruction
python scripts/predict.py --input your_ecg.csv --plot
```

---

## Naming Convention on Zenodo

Zenodo filenames use double-underscore as path separator:

```
ptbxl__EGV3__leXT-netv1s__fold1.pth
```

The `download_weights.py` script handles the renaming automatically.
