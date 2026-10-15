# LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23123820-blue)](https://doi.org/10.5281/zenodo.23123820)

Official implementation of **LeXT-Net: A Lead-Adaptive Hierarchical Network for 3-Lead to 12-Lead ECG Reconstruction**.

## What this repository supports

1. **Re-train LeXT-Net** on PTB-XL, Chapman-Shaoxing, or CPSC2018.
2. Run either **one selected fold** or the full **5-fold protocol**.
3. Train/evaluate on another compatible 12-lead ECG dataset through a simple manifest file.
4. **Download pretrained checkpoints** from Zenodo and evaluate them without retraining.
5. Produce machine-readable performance reports and reference-vs-reconstruction plots.
6. Build small local sample subsets for smoke testing, subject to each source dataset's redistribution terms.

## Installation

```bash
git clone https://github.com/rtekin-lab/LeXT_Net.git
cd LeXT_Net
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## A. Train from scratch

Edit `configs/paths.py` to point to your local datasets.

```bash
# Full 5-fold training
python scripts/train_multidataset.py --dataset ptbxl --model leXT-netv2 --strategy EGV3
python scripts/train_multidataset.py --dataset chapman --model leXT-netv2 --strategy EGV3
python scripts/train_multidataset.py --dataset cpsc2018 --model leXT-netv2 --strategy EGV3

# Train only fold 3 from the same fixed 5-fold split
python scripts/train_multidataset.py --dataset ptbxl --model leXT-netv2 --strategy EGV3 --fold 3

# Quick smoke test
python scripts/train_multidataset.py --dataset ptbxl --model leXT-netv2 --strategy EGV3 --fold 1 --max_rec 200 --epochs 2 --no_excel
```

`--fold 3` means “use fold 3 of the 5-fold protocol”; it does not redefine the experiment as one-fold CV.

For PTB-XL baseline comparisons, `scripts/train_ptbxl.py` also supports `--fold`.

## B. Evaluate pretrained models without training

First inspect/download the checkpoints that are actually available on Zenodo:

```bash
python scripts/download_weights.py --list
python scripts/download_weights.py --dataset ptbxl --strategy EGV3 --model leXT-netv2
```

Then evaluate a labelled dataset:

```bash
python scripts/evaluate_pretrained.py --dataset ptbxl --strategy EGV3 --model leXT-netv2 --fold all
python scripts/evaluate_pretrained.py --dataset chapman --strategy EGV3 --model leXT-netv2 --fold 1
python scripts/evaluate_pretrained.py --dataset cpsc2018 --strategy EGV3 --model leXT-netv2 --fold all
```

Outputs are written under `results/evaluation/...` and include full JSON metrics, separate normalized/physical summaries, lead-wise CSV files, checkpoint-hash metadata, and reference-vs-reconstruction plots. Physical MAE/RMSE are reported in canonical mV.

## C. Reconstruct a single recording

```bash
python scripts/predict.py --input ecg.csv --fs 500 \
  --dataset ptbxl --strategy EGV3 --model leXT-netv2 --fold 1 --plot

python scripts/predict.py --input ecg.csv --fs 100 \
  --dataset ptbxl --strategy EGV3 --model leXT-netv2 --ensemble --output reconstructed.csv
```

Input may contain 12 leads (the selected 3-lead strategy is extracted automatically) or exactly 3 already-selected channels.

**Important:** because training uses per-record/per-lead z-score normalization, `predict.py` reconstructs the target leads in normalized units. A physical-amplitude inverse transform for unseen target leads is not available from only three input leads. `evaluate_pretrained.py` can report retrospective physical-scale errors when a 12-lead reference is available.

## D. Try LeXT-Net on another ECG dataset

LeXT-Net expects 12-lead data in this order:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Internally records are converted to **100 Hz × 10 s = 1000 samples** and canonical mV before normalization. For a new dataset, create a manifest:

```csv
path,format,fs,unit,patient_id,record_id
records/r001.csv,csv,500,mV,P001,R001
records/r002.csv,csv,250,uV,P002,R002
records/r003,wfdb,500,,P003,R003
```

Named CSV/WFDB leads are reordered safely to the canonical 12-lead order. WFDB channel names and physical units are read from the header; the loader never silently assumes that the first 12 channels are the required ECG leads.

Then:

```bash
python scripts/train_multidataset.py --dataset custom \
  --manifest /path/to/manifest.csv --data_root /path/to/data \
  --model leXT-netv2 --strategy EGV3 --fold 1
```

See [docs/DATASETS_AND_CUSTOM_DATA.md](docs/DATASETS_AND_CUSTOM_DATA.md).

## E. Validate custom data and run a smoke test

```bash
python scripts/validate_dataset.py --manifest /path/to/manifest.csv --data-root /path/to/data --check-folds
python scripts/smoke_test.py
```

The smoke test is synthetic and checks software integration only. Authentic source records are not automatically bundled because redistribution permissions differ among datasets. Maintainers can create small standardized subsets from local copies:

```bash
python scripts/build_sample_db.py --dataset ptbxl --n 3
python scripts/build_sample_db.py --dataset chapman --n 3
python scripts/build_sample_db.py --dataset cpsc2018 --n 3
```

See [sample_data/README.md](sample_data/README.md). Verify the original dataset licence before committing generated source-derived samples to GitHub.

## Built-in datasets

| Dataset | Repository loader | Native sampling | Model input |
|---|---|---:|---:|
| PTB-XL | WFDB `records100` | 100 Hz | 100 Hz, 10 s |
| Chapman-Shaoxing | 12-lead CSV | 500 Hz | resampled to 100 Hz, 10 s; µV source converted to mV |
| CPSC2018 | WFDB/MAT records | 500 Hz | first 10 s, resampled to 100 Hz; physical mV |

Datasets are downloaded from their original providers and are not automatically redistributed by this repository.

## Input strategies

- `EGV1`: I, II, V1
- `EGV2`: I, II, V2
- `EGV3`: I, II, V3
- `QSI`: V6, II, −0.5·V2

## Experimental protocol

- sampling rate: 100 Hz
- duration: 10 s (1000 samples)
- normalization: per-record, per-lead z-score
- paper CV: 5 folds; PTB-XL is patient-grouped when patient IDs are available
- optimizer: AdamW, lr 3e-4, weight decay 1e-4
- scheduler: 5-epoch linear warm-up followed by cosine annealing
- LeXT-Net loss: 0.70·MAE + 0.25·(1−PCC) + 0.05·inter-lead consistency + 0.02·precordial smoothness

For Chapman/CPSC2018, the current built-in loader does not provide patient identifiers to the splitter; those experiments therefore use record-wise sequential folds unless patient/group metadata are supplied through the custom-manifest interface.

## Repository structure

```text
src/
  models/                 model definitions
  datasets.py             PTB-XL / Chapman / CPSC2018 loaders
  custom_data.py          manifest-based external dataset loader
  transforms.py           EGV1 / EGV2 / EGV3 / QSI
  crossval.py             sequential grouped folds
scripts/
  train_multidataset.py   main LeXT-Net training entry point
  train_ptbxl.py          PTB-XL model-comparison training
  evaluate_pretrained.py  no-training dataset evaluation + plots
  predict.py              single-record inference
  download_weights.py     dynamic Zenodo checkpoint downloader
  build_sample_db.py      build small local test subsets
configs/
  paths.py                 local dataset paths
docs/
  DATASETS_AND_CUSTOM_DATA.md
  PRETRAINED_EVALUATION.md
  RELEASING_WEIGHTS.md
sample_data/
  README.md
```

## Pretrained checkpoint release

Checkpoint filenames on Zenodo should follow:

```text
<dataset>__<strategy>__<model>__fold<k>.pth
```

The downloader discovers published files dynamically. See [WEIGHTS.md](WEIGHTS.md) and [docs/RELEASING_WEIGHTS.md](docs/RELEASING_WEIGHTS.md).

## Citation

Please cite the manuscript and the archived code record when using LeXT-Net. Update the bibliographic status/DOI to the final published article once available.

## License

Code: MIT. Dataset files and pretrained weights remain subject to their applicable source/archive terms.
