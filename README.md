# LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23123820-blue)](https://doi.org/10.5281/zenodo.23123820)

Official implementation of the manuscript:

> **LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction**  
> Ramazan Tekin, Hazret Tekin, Mine Onat, Merve Akkuş, Yılmaz Kaya  
> *Scientific Reports* — manuscript under review  
> [Source code](https://github.com/rtekin-lab/LeXT_Net) · [Zenodo archive](https://doi.org/10.5281/zenodo.23123820)

---

## Overview

Standard 12-lead ECG acquisition requires ten electrodes and precise electrode placement, which may be difficult in ambulatory monitoring, wearable sensing, remote-care, and resource-limited settings. **LeXT-Net** is designed to reconstruct a standard 12-lead ECG from only three input leads in a single inference pass.

LeXT-Net combines shared temporal representation learning with lead-specific adaptation and separate modelling of limb and precordial lead characteristics. The repository supports reproducible training, pretrained-model evaluation, single-record inference, external-dataset adaptation, quantitative performance reporting, and reconstruction visualization.

### What this repository supports

1. **Re-train LeXT-Net** on PTB-XL, Chapman-Shaoxing, or CPSC2018.
2. Run either **one selected fold** or the full **5-fold protocol**.
3. Train/evaluate on another compatible 12-lead ECG dataset through a manifest-based interface.
4. **Download pretrained checkpoints** from Zenodo and evaluate them without retraining.
5. Produce aggregate and lead-wise performance reports plus reference-vs-reconstruction plots.
6. Reconstruct a single ECG recording using one pretrained fold or an ensemble.
7. Validate external datasets before training/evaluation.
8. Run a synthetic smoke test for software integration.
9. Build small local sample subsets from source datasets, subject to redistribution permissions.

---

## Architecture

LeXT-Net uses a hierarchical lead-adaptive architecture consisting of:

- **Shared TCN trunk** — dilated residual temporal convolution blocks provide a common temporal representation.
- **Lead-specific temporal attention** — grouped operations independently modulate the shared representation for each target lead while preserving lead-specific parameterization.
- **Lead-wise projection** — transforms the attended representation into lead-adapted latent features.
- **Dual temporal decoders** — separate U-Net-style decoder pathways model limb and precordial lead characteristics.
- **Independent lead heads** — generate the final reconstructed waveforms for the 12 standard ECG leads.
- **Physiologically informed loss terms** — waveform MAE, correlation loss, inter-lead consistency based on Einthoven–Goldberger relationships, and adjacent-precordial smoothness.

The main released model is **`leXT-netv2`**.

---

## Installation

```bash
git clone https://github.com/rtekin-lab/LeXT_Net.git
cd LeXT_Net

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick start with pretrained weights

List the checkpoints currently available from the Zenodo release:

```bash
python scripts/download_weights.py --list
```

Download one pretrained model set:

```bash
python scripts/download_weights.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2
```

The expected local layout is:

```text
results/weights/<dataset>/<strategy>/<model>/fold<k>.pth
```

Example:

```text
results/weights/ptbxl/EGV3/leXT-netv2/fold1.pth
```

See [WEIGHTS.md](WEIGHTS.md) for release details.

---

## Evaluate pretrained models without retraining

Evaluate the released checkpoints on a labelled 12-lead dataset:

```bash
# Strict same-dataset out-of-fold evaluation
python scripts/evaluate_pretrained.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold all

# One selected fold
python scripts/evaluate_pretrained.py \
  --dataset chapman \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold 1

# Full CPSC2018 evaluation
python scripts/evaluate_pretrained.py \
  --dataset cpsc2018 \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold all
```

Outputs are written under `results/evaluation/...` and may include:

- full JSON metrics;
- aggregate normalized and retrospective physical-scale summaries;
- lead-wise CSV reports;
- checkpoint hash/provenance metadata;
- reference-vs-reconstruction plots.

For same-dataset evaluation, each record is evaluated only by its held-out fold model. For cross-database evaluation, fold ensembles may be used because the external records were not used for training.

**Physical-scale note:** training uses per-record, per-lead z-score normalization. Therefore retrospective physical-scale MAE/RMSE can be computed when the reference 12-lead ECG is available. Physical inverse transformation of unseen target leads is not available in a true three-lead-only deployment.

See [docs/PRETRAINED_EVALUATION.md](docs/PRETRAINED_EVALUATION.md).

---

## Single-record inference (`predict.py`)

```bash
# 500-Hz CSV recording; fold 1
python scripts/predict.py \
  --input ecg.csv \
  --fs 500 \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold 1 \
  --plot

# Five-fold ensemble for an external recording
python scripts/predict.py \
  --input ecg.csv \
  --fs 100 \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --ensemble \
  --output reconstructed.csv
```

Input may contain all 12 standard leads or exactly the 3 already-selected strategy channels. When 12 leads are provided, the requested three-lead strategy is extracted automatically.

Because training uses per-record/per-lead z-score normalization, `predict.py` reconstructs unseen target leads in the normalized model-output space. Use `evaluate_pretrained.py` when a labelled 12-lead reference is available and physical-scale evaluation is required.

---

## Training from scratch

Edit `configs/paths.py` so that the built-in loaders point to your local datasets.

### Full five-fold training

```bash
python scripts/train_multidataset.py \
  --dataset ptbxl \
  --model leXT-netv2 \
  --strategy EGV3

python scripts/train_multidataset.py \
  --dataset chapman \
  --model leXT-netv2 \
  --strategy EGV3

python scripts/train_multidataset.py \
  --dataset cpsc2018 \
  --model leXT-netv2 \
  --strategy EGV3
```

### Train only one selected fold

```bash
python scripts/train_multidataset.py \
  --dataset ptbxl \
  --model leXT-netv2 \
  --strategy EGV3 \
  --fold 3
```

`--fold 3` means **fold 3 of the fixed five-fold protocol**. It does not redefine the experiment as one-fold cross-validation.

### Quick development test

```bash
python scripts/train_multidataset.py \
  --dataset ptbxl \
  --model leXT-netv2 \
  --strategy EGV3 \
  --fold 1 \
  --max_rec 200 \
  --epochs 2 \
  --no_excel
```

For PTB-XL model-comparison experiments, `scripts/train_ptbxl.py` is also available.

---

## Datasets

| Dataset | Records | Native sampling | Model sampling | Source |
|---|---:|---:|---:|---|
| **PTB-XL** | 21,837 | 100 / 500 Hz | 100 Hz | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) |
| **Chapman-Shaoxing** | 10,646 | 500 Hz | 100 Hz | [Figshare](https://figshare.com/collections/ChapmanECG/4560497) |
| **CPSC2018** | 6,877 | 500 Hz | 100 Hz | [ICBEB 2018 Challenge](http://2018.icbeb.org/Challenge.html) |

Original datasets are **not redistributed** with this repository or the Zenodo software release. Users should obtain them directly from their original providers and comply with the corresponding licences and terms of use.

### Built-in preprocessing

| Dataset | Repository input | Internal representation |
|---|---|---|
| PTB-XL | WFDB `records100` | 100 Hz, 10 s, 12 leads |
| Chapman-Shaoxing | 12-lead CSV | resampled to 100 Hz, 10 s, source µV converted to mV |
| CPSC2018 | WFDB/MAT records | first 10 s, resampled to 100 Hz, physical mV |

---

## Input strategies

- **`EGV1`**: I, II, V1
- **`EGV2`**: I, II, V2
- **`EGV3`**: I, II, V3
- **`QSI`**: V6, II, −0.5·V2

All models reconstruct the standard 12-lead order:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

---

## Using another compatible ECG dataset

External datasets can be supplied through a manifest file. Records are standardized internally to **100 Hz × 10 s = 1000 samples**, 12 canonical leads, and canonical mV before normalization.

Example manifest:

```csv
path,format,fs,unit,patient_id,record_id
records/r001.csv,csv,500,mV,P001,R001
records/r002.csv,csv,250,uV,P002,R002
records/r003,wfdb,500,,P003,R003
```

Named CSV/WFDB leads are reordered to:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

WFDB signal names and physical units are read from the header. The loader does not silently assume that the first 12 channels are the required ECG leads.

Train on a custom dataset:

```bash
python scripts/train_multidataset.py \
  --dataset custom \
  --manifest /path/to/manifest.csv \
  --data_root /path/to/data \
  --model leXT-netv2 \
  --strategy EGV3 \
  --fold 1
```

If multiple recordings belong to the same patient, provide `patient_id` so that grouped train/test splitting can prevent patient leakage.

See [docs/DATASETS_AND_CUSTOM_DATA.md](docs/DATASETS_AND_CUSTOM_DATA.md).

---

## Validate a custom dataset

Before training or evaluation:

```bash
python scripts/validate_dataset.py \
  --manifest /path/to/manifest.csv \
  --data-root /path/to/data \
  --check-folds
```

This helps identify incompatible lead names, units, sampling metadata, dimensions, and grouping problems before a long training run.

---

## Sample / smoke-test datasets

Run the synthetic software smoke test:

```bash
python scripts/smoke_test.py
```

Authentic source records are not automatically bundled because redistribution permissions differ among datasets. Maintainers can create small standardized subsets from local copies:

```bash
python scripts/build_sample_db.py --dataset ptbxl --n 3
python scripts/build_sample_db.py --dataset chapman --n 3
python scripts/build_sample_db.py --dataset cpsc2018 --n 3
```

See [sample_data/README.md](sample_data/README.md). Verify the original dataset licence before committing source-derived ECG records to GitHub.

---

## Experimental protocol

| Parameter | Value |
|---|---|
| Cross-validation | Five-fold |
| PTB-XL splitting | Patient-grouped when patient IDs are available |
| Chapman/CPSC2018 built-in splitting | Record-wise sequential folds unless patient/group metadata are supplied |
| Sampling rate | 100 Hz |
| Recording length | 10 s (1,000 samples) |
| Normalization | Per-record, per-lead z-score |
| Evaluation scopes | mAll, mOther, mInd |
| Metrics | R², SSIM, PCC, MSC, MAE, RMSE, NRMSE, SNR |
| Optimizer | AdamW, lr = 3×10⁻⁴, weight decay = 10⁻⁴ |
| Scheduler | 5-epoch linear warm-up followed by cosine annealing |
| Epochs | 100 |
| Comparison batch size | 32 |
| LeXT-Net loss | 0.70·MAE + 0.25·(1−PCC) + 0.05·inter-lead consistency + 0.02·precordial smoothness |

For the final model-comparison configuration and manuscript/code alignment notes, see [MANUSCRIPT_CODE_ALIGNMENT.md](MANUSCRIPT_CODE_ALIGNMENT.md).

---

## Repository structure

```text
src/
  models/                 model definitions
  datasets.py             PTB-XL / Chapman / CPSC2018 loaders
  custom_data.py          manifest-based external dataset loader
  transforms.py           EGV1 / EGV2 / EGV3 / QSI
  crossval.py             sequential/grouped fold utilities

scripts/
  train_multidataset.py   main LeXT-Net training entry point
  train_ptbxl.py          PTB-XL model-comparison training
  evaluate_pretrained.py  pretrained evaluation + reports + plots
  predict.py              single-record inference
  download_weights.py     Zenodo checkpoint downloader
  validate_dataset.py     validate compatible external datasets
  smoke_test.py           synthetic integration smoke test
  build_sample_db.py      create small local source-dataset subsets
  package_release_weights.py
  build_weights_manifest.py

configs/
  paths.py                local dataset paths

docs/
  DATASETS_AND_CUSTOM_DATA.md
  PRETRAINED_EVALUATION.md
  RELEASING_WEIGHTS.md
  USAGE_SCENARIOS.md

sample_data/
  README.md
```

---

## Pretrained model release

The Zenodo release contains pretrained **`leXT-netv2`** checkpoints for:

- 3 datasets: PTB-XL, Chapman-Shaoxing, CPSC2018;
- 4 input strategies: EGV1, EGV2, EGV3, QSI;
- 5 folds per dataset/strategy.

This corresponds to **60 pretrained checkpoints** in total.

Checkpoint filenames follow:

```text
<dataset>__<strategy>__<model>__fold<k>.pth
```

A SHA-256 manifest is included for provenance and integrity verification. The downloader discovers the files published in the Zenodo record dynamically.

See [WEIGHTS.md](WEIGHTS.md) and [docs/RELEASING_WEIGHTS.md](docs/RELEASING_WEIGHTS.md).

---

## Reproducibility notes

- Same-dataset pretrained evaluation uses held-out-fold prediction rather than a five-model ensemble over records seen during training.
- Cross-database/custom evaluation may use fold ensembles when the external records were not used to train the selected checkpoints.
- Physical-scale evaluation is retrospective because inverse normalization of reconstructed target leads requires the corresponding reference lead statistics.
- Dataset files are not redistributed automatically.
- Pretrained checkpoint hashes can be recorded and verified through the provided manifest utilities.

---

## Citation

If you use LeXT-Net in your research, please cite the manuscript and the corresponding software release when appropriate.

### Manuscript

```bibtex
@unpublished{tekin2026lextnet,
  title  = {{LeXT-Net}: A Lead-Adaptive Hierarchical Network for
            3-Lead to 12-Lead {ECG} Reconstruction},
  author = {Tekin, Ramazan and Tekin, Hazret and Onat, Mine and
            Akkuş, Merve and Kaya, Yılmaz},
  year   = {2026},
  note   = {Manuscript under review at Scientific Reports}
}
```

### Software and pretrained model weights

```bibtex
@software{tekin2026lextnetsoftware,
  title     = {{LeXT-Net}: Source Code and Pretrained Model Weights
               for 3-Lead to 12-Lead ECG Reconstruction},
  author    = {Tekin, Ramazan and Tekin, Hazret and Onat, Mine and
               Akkuş, Merve and Kaya, Yılmaz},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23123820},
  url       = {https://doi.org/10.5281/zenodo.23123820}
}
```

GitHub also exposes citation metadata through [`CITATION.cff`](CITATION.cff).

**After the Scientific Reports article is formally published, replace the manuscript entry above with the final journal citation and article DOI. The Zenodo DOI must remain associated with the software/model release, not be used as the journal-article DOI.**

---

## License

LeXT-Net source code is released under the [MIT License](LICENSE).

Original ECG datasets are not redistributed and remain subject to the licences and terms of their respective providers. Pretrained model weights and archived release files should be cited using the associated Zenodo record.

---

## Contact

**Ramazan Tekin**  
Batman University  
Email: ramazan.tekin@batman.edu.tr

For reproducibility or software-related questions, users may also open an issue in the [GitHub repository](https://github.com/rtekin-lab/LeXT_Net/issues).
