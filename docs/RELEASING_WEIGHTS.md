# Publishing pretrained checkpoints

For training-free use, upload the selected checkpoints from the HPC runs to the project Zenodo record using this exact filename convention:

```text
<dataset>__<strategy>__<model>__fold<k>.pth
```

Examples:

```text
ptbxl__EGV3__leXT-netv2__fold1.pth
chapman__EGV3__leXT-netv2__fold1.pth
cpsc2018__EGV3__leXT-netv2__fold1.pth
```

Recommended release set for reproducibility:

- datasets: `ptbxl`, `chapman`, `cpsc2018`
- strategies: `EGV1`, `EGV2`, `EGV3`, `QSI`
- model: `leXT-netv2`
- folds: `1..5`

That is **60 checkpoints** for the full three-dataset release. If storage is a concern, publish EGV3 first and add the other strategies later; `scripts/download_weights.py` discovers whatever is actually present on Zenodo.

Before upload, verify that each checkpoint corresponds to the intended dataset/strategy/fold and archive a checksum list:

```bash
sha256sum *.pth > SHA256SUMS.txt
```

Do not place large `.pth` files in GitHub; the repository `.gitignore` excludes model weights.

## Generate a SHA-256 release manifest

After renaming/copying checkpoints into the release naming convention, generate a provenance table:

```bash
python scripts/build_weights_manifest.py \
  --weights-dir /path/to/release_checkpoints \
  --out weights_manifest.csv \
  --code-version v1.0.0
```

The generated table records dataset, strategy, model, fold, filename, SHA-256, file size, and code version. Publish this CSV alongside the checkpoints on Zenodo.

`download_weights.py` also verifies the checksum exposed by the Zenodo API after download (unless `--no-verify` is explicitly used).

## 60-checkpoint release check

For the complete main v2 release, the expected set is:

```text
3 datasets × 4 strategies × 5 folds = 60 checkpoints
```

Before publishing, verify the count and naming:

```bash
find /path/to/release_checkpoints -maxdepth 1 -name '*.pth' | wc -l
python scripts/build_weights_manifest.py --weights-dir /path/to/release_checkpoints --out weights_manifest.csv
```

After Zenodo publication:

```bash
python scripts/download_weights.py --list
```

and confirm that all intended files are discoverable.

## Package checkpoints from the training-results tree

If your training results follow the repository layout, collect the complete 60-file v2 release without renaming files manually:

```bash
python scripts/package_release_weights.py \
  --results-root /path/to/results \
  --out-dir /path/to/zenodo_release \
  --model leXT-netv2 \
  --strict

python scripts/build_weights_manifest.py \
  --weights-dir /path/to/zenodo_release \
  --out /path/to/zenodo_release/weights_manifest.csv \
  --code-version v1.0.0
```
