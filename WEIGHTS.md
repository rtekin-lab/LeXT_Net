# Pre-trained model weights

LeXT-Net checkpoints are intended to be distributed through the project Zenodo record rather than GitHub:

**Zenodo DOI:** https://doi.org/10.5281/zenodo.23123820

The downloader queries the record dynamically, so it only advertises files that are actually published there. Downloads are checksum-verified against the checksum exposed by the Zenodo API by default.

## Check availability

```bash
python scripts/download_weights.py --list
```

## Download a specific release subset

```bash
python scripts/download_weights.py --dataset ptbxl --strategy EGV3 --model leXT-netv2
python scripts/download_weights.py --dataset chapman --strategy EGV3 --model leXT-netv2
python scripts/download_weights.py --dataset cpsc2018 --strategy EGV3 --model leXT-netv2
```

Expected local layout:

```text
results/weights/<dataset>/<strategy>/<model>/fold<k>.pth
```

Example:

```text
results/weights/ptbxl/EGV3/leXT-netv2/fold1.pth
```

## Evaluate without training

```bash
python scripts/evaluate_pretrained.py --dataset ptbxl --strategy EGV3 --model leXT-netv2 --fold all
```

For a single recording or 3-lead input, use `scripts/predict.py`. For a labelled 12-lead dataset and performance report, use `scripts/evaluate_pretrained.py`.

See `docs/PRETRAINED_EVALUATION.md` and `docs/RELEASING_WEIGHTS.md`.


## Release provenance

Maintainers can generate `weights_manifest.csv` with SHA-256 digests using:

```bash
python scripts/build_weights_manifest.py --weights-dir /path/to/release_checkpoints --out weights_manifest.csv --code-version v1.0.0
```

See `docs/RELEASING_WEIGHTS.md`.
