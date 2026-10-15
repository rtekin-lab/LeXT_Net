# Pretrained evaluation without retraining

`scripts/evaluate_pretrained.py` evaluates published LeXT-Net checkpoints on a labelled 12-lead dataset and produces machine-readable reports plus reconstruction plots.

## Same-dataset evaluation: strict out-of-fold (OOF)

When the evaluation dataset and weight-training dataset are the same, every record is evaluated only by the checkpoint corresponding to the fold in which that record was held out.

```bash
python scripts/evaluate_pretrained.py \
  --dataset ptbxl --strategy EGV3 --model leXT-netv2 --fold all
```

This prevents a record from being scored by a model that trained on that record.

## External / cross-database evaluation

For a dataset that was not used to train the checkpoints, all selected checkpoints may evaluate every record. `--fold all` averages the five checkpoint predictions.

```bash
python scripts/evaluate_pretrained.py \
  --dataset chapman --weights-dataset ptbxl \
  --strategy EGV3 --model leXT-netv2 --fold all
```

## Custom dataset

```bash
python scripts/evaluate_pretrained.py --dataset custom \
  --manifest /data/site/manifest.csv --data-root /data/site \
  --weights-dataset ptbxl \
  --strategy EGV3 --model leXT-netv2 --fold all
```

Validate first with `scripts/validate_dataset.py`.

## Normalized vs physical metrics

Training uses per-record, per-lead z-score normalization. Evaluation therefore reports two metric spaces:

- **normalized**: metrics computed directly in z-score space;
- **physical**: predictions are retrospectively inverse-transformed with the labelled reference record's per-lead mean/std and reported in canonical **mV** for absolute errors such as MAE/RMSE.

The physical inverse transform is an evaluation calculation. In a real three-lead-only deployment, the unknown target leads' reference mean/std are not available, so this procedure should not be interpreted as a deployment-time amplitude-calibration method.

## Output files

Outputs are stored under a path that records both the evaluation dataset and the checkpoint source, for example:

```text
results/evaluation/chapman__weights-ptbxl/EGV3/leXT-netv2/
```

Files include:

- `metrics.json` — full normalized and physical lead-wise metrics;
- `summary.csv` — compact backward-compatible normalized/physical summary;
- `summary_normalized.csv` — mAll normalized metrics;
- `summary_physical.csv` — mAll physical metrics (MAE/RMSE in mV);
- `leadwise_metrics_normalized.csv`;
- `leadwise_metrics_physical_mV.csv`;
- `run_metadata.json` — dataset/model/protocol/checkpoint hashes and preprocessing metadata;
- `overlay_*_physical_mV.png` or `overlay_*_normalized.png`.

Choose plot space with:

```bash
--plot-space physical
--plot-space normalized
```

## Checkpoint provenance

`run_metadata.json` stores the path and SHA-256 digest of every checkpoint used for that evaluation run. This makes a reported result traceable to exact weight files.
