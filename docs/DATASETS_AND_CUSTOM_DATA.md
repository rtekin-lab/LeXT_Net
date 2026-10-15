# Datasets, custom data, and reproducible training

LeXT-Net internally uses **12 standard leads**, **100 Hz**, and **10 seconds (1000 samples)**.
The canonical lead order is:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Training uses **per-record, per-lead z-score normalization** after dataset-specific loading.
Before normalization, repository loaders use **mV as the canonical physical amplitude unit**.

## Built-in datasets

Set local paths in `configs/paths.py`, then use `scripts/train_multidataset.py`.

```bash
# Full fixed 5-fold training
python scripts/train_multidataset.py --dataset ptbxl --model leXT-netv2 --strategy EGV3
python scripts/train_multidataset.py --dataset chapman --model leXT-netv2 --strategy EGV3
python scripts/train_multidataset.py --dataset cpsc2018 --model leXT-netv2 --strategy EGV3

# Only fold 3 from the same fixed 5-fold definition
python scripts/train_multidataset.py --dataset ptbxl --model leXT-netv2 --strategy EGV3 --fold 3
```

`--fold 3` means *fold 3 of the 5-fold protocol*; it does not redefine the experiment as one-fold CV.

### Built-in preprocessing and physical units

| Dataset | Source rate | Duration handling | Loader physical unit before normalization |
|---|---:|---|---|
| PTB-XL `records100` | 100 Hz | 10 s | mV |
| Chapman-Shaoxing MUSE CSV | 500 Hz | 10 s, resampled to 100 Hz | source CSV treated as µV, converted to mV |
| CPSC2018 | 500 Hz | first 10 s, then resampled to 100 Hz | WFDB physical mV |

If you use a different distribution/export of Chapman data, verify its amplitude convention before using the built-in CSV loader.

## Trying another 12-lead ECG dataset

Use a manifest rather than editing model code.

### CSV example

```csv
path,format,fs,unit,patient_id,record_id
records/p001.csv,csv,500,mV,P001,R001
records/p002.csv,csv,250,uV,P002,R002
```

Supported CSV units are `mV`, `uV`/`µV`, and `V`. They are converted to mV before resampling/normalization.

Named CSV columns can appear in any order; the loader reorders them by lead name. Extra non-lead columns are ignored. Headerless CSV files are accepted only when they contain exactly 12 columns already ordered as:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

### WFDB example

```csv
path,format,fs,patient_id,record_id
records/p001,wfdb,500,P001,R001
records/p002,wfdb,250,P002,R002
```

For WFDB records, `path` is the record path without `.hea/.dat`. The loader reads `sig_name` and the per-channel physical units from the WFDB header, explicitly finds the 12 required leads, reorders them, and converts supported voltage units to mV. It **does not** silently take the first 12 channels. Missing or duplicated standard leads raise an error.

If `fs` is provided for WFDB, it is checked against the header and a mismatch is treated as an error.

## Duration policy

After conversion to 100 Hz:

- exactly 1000 samples: used as-is;
- more than 1000 samples: first 1000 samples are kept;
- 950–999 samples: right-zero-padded to 1000;
- fewer than 950 samples: rejected.

## Validate a custom dataset before training

```bash
python scripts/validate_dataset.py \
  --manifest /path/to/manifest.csv \
  --data-root /path/to/data \
  --check-folds
```

The validator checks shape, finite values, record-ID uniqueness, canonical mV conversion, and optional patient-group fold leakage.

## Train custom data

```bash
# One selected fold
python scripts/train_multidataset.py --dataset custom \
  --manifest /path/to/manifest.csv --data_root /path/to/data \
  --model leXT-netv2 --strategy EGV3 --fold 1

# Full fixed 5-fold run: omit --fold
python scripts/train_multidataset.py --dataset custom \
  --manifest /path/to/manifest.csv --data_root /path/to/data \
  --model leXT-netv2 --strategy EGV3
```

Provide `patient_id` whenever multiple recordings from one person exist. This allows patient-wise grouping and avoids leakage. If no patient identifier is supplied, each record is treated as its own group.

## Evaluate pretrained weights on custom data

```bash
python scripts/evaluate_pretrained.py --dataset custom \
  --manifest /path/to/manifest.csv --data-root /path/to/data \
  --weights-dataset ptbxl \
  --model leXT-netv2 --strategy EGV3 --fold all
```

Because the custom dataset is external to the PTB-XL training folds, all selected fold checkpoints may be ensembled.

## Comparability checklist for a new dataset

Before reporting results, document:

- source and licensing terms;
- lead names and order;
- physical amplitude unit;
- source sampling frequency;
- recording duration and cropping policy;
- filtering/denoising applied before LeXT-Net;
- patient/group identifier used for splitting;
- exclusions and missing-lead policy;
- whether evaluation is same-dataset OOF or cross-database;
- whether MAE/RMSE are normalized or retrospective physical-scale mV values.
