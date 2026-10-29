# Publishing pretrained checkpoints

LeXT-Net pretrained checkpoints are distributed through the project Zenodo record as a **single ZIP archive** rather than as individual `.pth` files.

The release archive must use the exact filename:

```text
LeXT_Net_release_weights.zip
```

The archive should contain the complete pretrained `leXT-netv2` checkpoint set together with a SHA-256 provenance manifest.

---

## Release contents

The recommended v1.0.0 release contains:

- datasets: `ptbxl`, `chapman`, `cpsc2018`
- input configurations: `EGV1`, `EGV2`, `EGV3`, `QSI`
- model: `leXT-netv2`
- folds: `1..5`

This corresponds to:

```text
3 datasets × 4 input configurations × 5 folds = 60 checkpoints
```

The ZIP archive should use a flat file layout:

```text
ptbxl__EGV1__leXT-netv2__fold1.pth
ptbxl__EGV1__leXT-netv2__fold2.pth
...
ptbxl__EGV3__leXT-netv2__fold5.pth
...
chapman__QSI__leXT-netv2__fold5.pth
...
cpsc2018__QSI__leXT-netv2__fold5.pth
weights_manifest.csv
```

Checkpoint filenames must follow:

```text
<dataset>__<strategy>__<model>__fold<k>.pth
```

Examples:

```text
ptbxl__EGV3__leXT-netv2__fold1.pth
chapman__EGV1__leXT-netv2__fold5.pth
cpsc2018__QSI__leXT-netv2__fold3.pth
```

---

## Why a single ZIP archive?

Publishing the complete pretrained checkpoint set as one archive provides a compact and immutable release artifact and keeps the Zenodo record easy to manage.

The repository downloader is designed specifically for this release format. `scripts/download_weights.py`:

1. queries the project Zenodo record;
2. locates `LeXT_Net_release_weights.zip`;
3. downloads the archive once and stores it in a local cache;
4. verifies the Zenodo archive checksum;
5. reads `weights_manifest.csv` from the archive;
6. extracts only the dataset, input configuration, model, and fold requested by the user;
7. verifies extracted checkpoints against the manifest SHA-256 values when available.

The archive is cached locally at:

```text
results/weights/_cache/LeXT_Net_release_weights.zip
```

Extracted checkpoints are written to:

```text
results/weights/<dataset>/<strategy>/<model>/fold<k>.pth
```

For example:

```text
results/weights/ptbxl/EGV3/leXT-netv2/fold1.pth
```

Because the full archive is downloaded on the first request, subsequent checkpoint requests reuse the cached ZIP unless the user specifies `--force`.

---

## Package checkpoints from training results

If the training outputs follow the standard repository structure:

```text
<results-root>/<dataset>/leXT-netv2/<strategy>_fold<k>/fold<k>.pth
```

the complete release set can be collected automatically using:

```bash
python scripts/package_release_weights.py \
  --results-root /path/to/results \
  --out-dir /path/to/zenodo_release \
  --model leXT-netv2 \
  --strict
```

The `--strict` option causes the script to fail if any of the expected 60 checkpoints is missing.

---

## Generate the SHA-256 manifest

After collecting the checkpoints, generate the release manifest with:

```bash
python scripts/build_weights_manifest.py \
  --weights-dir /path/to/zenodo_release \
  --out /path/to/zenodo_release/weights_manifest.csv \
  --code-version v1.0.0
```

The manifest records, for each checkpoint:

- dataset;
- input configuration;
- model;
- fold;
- filename;
- SHA-256 digest;
- file size;
- code version.

For reproducible scientific use, `weights_manifest.csv` should be included inside the final release ZIP.

It may also be uploaded separately to Zenodo so that users can inspect the checkpoint provenance without downloading the full archive.

---

## Verify the complete release before archiving

Before creating the ZIP archive, confirm that exactly 60 checkpoint files are present:

```bash
find /path/to/zenodo_release -maxdepth 1 -name '*.pth' | wc -l
```

The expected result is:

```text
60
```

Verify that `weights_manifest.csv` is also present.

---

## Create the final ZIP archive

Create a single archive named exactly:

```text
LeXT_Net_release_weights.zip
```

The ZIP should contain the 60 checkpoint files and `weights_manifest.csv` in a flat layout.

Before uploading, inspect the archive contents and confirm that:

- all 60 expected checkpoints are present;
- `weights_manifest.csv` is present;
- filenames follow the required release convention;
- no temporary files, logs, datasets, or unrelated training artifacts are included.

---

## Zenodo release layout

The Zenodo record should contain:

```text
LeXT-Net_Repo_v1.0.0.zip
LeXT_Net_release_weights.zip
weights_manifest.csv
```

`LeXT-Net_Repo_v1.0.0.zip` is a frozen source-code snapshot corresponding to the software release.

`LeXT_Net_release_weights.zip` is the primary pretrained-model artifact.

The standalone `weights_manifest.csv` is optional but recommended for transparency and provenance inspection.

Individual `.pth` files do not need to be uploaded separately because the downloader uses the single ZIP archive.

Original PTB-XL, Chapman-Shaoxing, and CPSC2018 datasets must not be redistributed through this release.

---

## Zenodo DOI

The project software and pretrained-model release is associated with:

```text
10.5281/zenodo.23123820
```

The repository downloader uses this DOI-derived record identifier to query the Zenodo API.

---

## Test the Zenodo release after publication

After the Zenodo record has been published, verify that the archive is discoverable:

```bash
python scripts/download_weights.py --list
```

The command should report all 60 pretrained checkpoints.

Then test extraction of one model set:

```bash
python scripts/download_weights.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2
```

The expected extracted files are:

```text
results/weights/ptbxl/EGV3/leXT-netv2/fold1.pth
results/weights/ptbxl/EGV3/leXT-netv2/fold2.pth
results/weights/ptbxl/EGV3/leXT-netv2/fold3.pth
results/weights/ptbxl/EGV3/leXT-netv2/fold4.pth
results/weights/ptbxl/EGV3/leXT-netv2/fold5.pth
```

Evaluate the downloaded checkpoints with:

```bash
python scripts/evaluate_pretrained.py \
  --dataset ptbxl \
  --strategy EGV3 \
  --model leXT-netv2 \
  --fold all
```

For single-record inference, use:

```bash
python scripts/predict.py --help
```

---

## Integrity verification

By default, two integrity checks are performed by `download_weights.py`:

1. the downloaded ZIP archive is checked against the checksum exposed by the Zenodo API;
2. extracted checkpoints are checked against the SHA-256 values recorded in `weights_manifest.csv`.

For reproducible use, checksum verification should remain enabled.

---

## Versioning

The initial software and pretrained-model release is:

```text
v1.0.0
```

If pretrained checkpoints or release files need to be changed after publication, create a new Zenodo version rather than modifying the scientific meaning of the archived v1.0.0 release.

The repository source snapshot, release archive, manifest, and Zenodo metadata should all identify the same software version.

---

## Final maintainer checklist

Before publishing the Zenodo record, confirm that:

- [ ] the release contains exactly 60 `leXT-netv2` checkpoints;
- [ ] all checkpoint filenames follow the required naming convention;
- [ ] `weights_manifest.csv` was generated from the final checkpoint files;
- [ ] all SHA-256 values correspond to the archived files;
- [ ] `LeXT_Net_release_weights.zip` contains the 60 checkpoints and manifest;
- [ ] the source-code snapshot corresponds to repository version `v1.0.0`;
- [ ] the Zenodo record uses DOI `10.5281/zenodo.23123820`;
- [ ] the Zenodo license matches the repository MIT License;
- [ ] the original ECG datasets are not included;
- [ ] `python scripts/download_weights.py --list` succeeds after publication;
- [ ] at least one pretrained evaluation is successfully completed from the published Zenodo archive.

See also:

- `README.md`
- `WEIGHTS.md`
- `docs/PRETRAINED_EVALUATION.md`
