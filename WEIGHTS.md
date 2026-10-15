# Pretrained model weights

LeXT-Net pretrained checkpoints are distributed through the project Zenodo record rather than being committed to GitHub.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.23123820

## Release format

The pretrained models are published as **one ZIP archive**:

```text
LeXT_Net_release_weights.zip
```

The archive uses a flat layout. Checkpoints are named as:

```text
<dataset>__<strategy>__<model>__fold<k>.pth
```

Examples:

```text
ptbxl__EGV3__leXT-netv2__fold1.pth
chapman__EGV1__leXT-netv2__fold5.pth
cpsc2018__QSI__leXT-netv2__fold3.pth
```

The v1.0.0 pretrained release contains:

- 3 datasets: `ptbxl`, `chapman`, `cpsc2018`
- 4 three-lead input strategies: `EGV1`, `EGV2`, `EGV3`, `QSI`
- 5 trained folds per dataset/strategy
- model: `leXT-netv2`
- **60 checkpoints in total**

A `weights_manifest.csv` file should also be included inside the ZIP. It records checkpoint provenance, file size, code version, and SHA-256 digest for each `.pth` file.

## Why a single archive?

Publishing the weights as one archive keeps the Zenodo record compact and makes the released checkpoint set immutable and easy to preserve. The repository downloader handles the archive automatically: it downloads the ZIP once, caches it locally, and extracts only the dataset/strategy/fold requested by the user.

The first download therefore retrieves the complete weight archive. Subsequent requests reuse the cached ZIP and do not download it again unless `--force` is used.

## List available checkpoints

```bash
python scripts/download_weights.py --list
```

To narrow the listing:

```bash
python scripts/download_weights.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --list
```

## Download / extract pretrained weights

PTB-XL, EGV3, all five folds:

```bash
python scripts/download_weights.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2
```

Chapman-Shaoxing, EGV2, all five folds:

```bash
python scripts/download_weights.py \
  --dataset chapman \
  --strategy EGV2 \
  --model leXT-netv2
```

CPSC2018, QSI, fold 3 only:

```bash
python scripts/download_weights.py \
  --dataset cpsc2018 \
  --strategy QSI \
  --model leXT-netv2 \
  --fold 3
```

Extracted checkpoints are placed in the repository as:

```text
results/weights/<dataset>/<strategy>/<model>/fold<k>.pth
```

For example:

```text
results/weights/ptbxl/EGV3/leXT-netv2/fold1.pth
```

The downloaded Zenodo archive is cached by default at:

```text
results/weights/_cache/LeXT_Net_release_weights.zip
```

You may change these locations with:

```bash
python scripts/download_weights.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --cache-dir /path/to/cache \
  --out-dir /path/to/extracted_weights
```

## Integrity verification

Two verification levels are used by default:

1. the downloaded ZIP archive is checked against the checksum exposed by the Zenodo API;
2. if `weights_manifest.csv` is present inside the archive, each extracted checkpoint is checked against its recorded SHA-256 digest.

Verification can be disabled only when necessary:

```bash
python scripts/download_weights.py --dataset ptbxl --strategy EGV3 --no-verify
```

For reproducible scientific use, leaving verification enabled is strongly recommended.

## Evaluate without training

After extracting the desired weights:

```bash
python scripts/evaluate_pretrained.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold all
```

For a single ECG recording or three-lead input, use:

```bash
python scripts/predict.py --help
```

For a labelled 12-lead dataset and a performance report, use `scripts/evaluate_pretrained.py`.

See also:

- `docs/PRETRAINED_EVALUATION.md`
- `docs/RELEASING_WEIGHTS.md`
- `README.md`

## Maintainer note: building the release manifest

Before creating `LeXT_Net_release_weights.zip`, maintainers can generate the provenance manifest with:

```bash
python scripts/build_weights_manifest.py \
  --weights-dir /path/to/release_checkpoints \
  --out /path/to/release_checkpoints/weights_manifest.csv \
  --code-version v1.0.0
```

The ZIP should then contain the 60 flat `.pth` files plus `weights_manifest.csv`.
