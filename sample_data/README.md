# Sample / smoke-test data

Authentic PTB-XL, Chapman-Shaoxing, and CPSC2018 records are **not bundled automatically** in this repository. Their source licences/redistribution conditions must be checked before committing extracted or resampled records.

## Dataset-free software smoke test

A deterministic synthetic ECG-like dataset is generated in a temporary directory by:

```bash
python scripts/smoke_test.py
```

This tests manifest loading, named-lead reordering, physical-unit handling, resampling, grouped splitting, a LeXT-Net forward pass, metric calculation, and plotting. The model is randomly initialized and the data are synthetic; the resulting metrics are **not scientific performance results**.

## Build local tiny subsets from downloaded datasets

```bash
python scripts/build_sample_db.py --dataset ptbxl --n 3
python scripts/build_sample_db.py --dataset chapman --n 3
python scripts/build_sample_db.py --dataset cpsc2018 --n 3
```

The builder writes standardized 100-Hz, 10-s CSV records and `manifest.csv`. Review the original dataset licence and attribution requirements before publishing any generated subset.

| Dataset | Local builder | Bundled authentic records |
|---|---|---|
| PTB-XL | yes | no |
| Chapman-Shaoxing | yes | no |
| CPSC2018 | yes | no |
