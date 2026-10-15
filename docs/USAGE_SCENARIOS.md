# Practical usage scenarios

## Scenario 1 — Exact paper-style retraining
Train `leXT-netv2` on one of the three built-in datasets with all five folds and one or all input strategies.

## Scenario 2 — Fast development run
Select one fold with `--fold N`, optionally reduce records/epochs while debugging, and use `--no_excel`. A selected fold is still defined from the normal multi-fold split.

## Scenario 3 — Reproduce published checkpoint performance
Download same-dataset checkpoints and run `evaluate_pretrained.py`. For built-in datasets the script uses strict out-of-fold prediction, so a record is not evaluated by a model that trained on that record.

## Scenario 4 — Cross-database generalization
Evaluate weights trained on one dataset against a different labelled dataset by specifying `--weights-dataset` or `--weights-dir`. Here every external record is unseen by construction, so a five-checkpoint ensemble is allowed.

Example:

```bash
python scripts/evaluate_pretrained.py --dataset chapman \
  --weights-dataset ptbxl --strategy EGV3 --model leXT-netv2 --fold all
```

## Scenario 5 — New institutional / private dataset
Describe 12-lead records with a manifest, preserve patient/group IDs, then either train from scratch (`train_multidataset.py --dataset custom`) or evaluate existing weights (`evaluate_pretrained.py --dataset custom`).

## Scenario 6 — Single-record demonstration
Use `predict.py` for a 12-lead or 3-lead input. If only three leads are supplied, reconstruction is possible but supervised performance metrics require a time-aligned 12-lead reference.

## Scenario 7 — CI / smoke testing
Use a tiny, legally redistributable sample subset (or synthetic 12-lead records) to verify installation, loading, checkpoint loading, inference, report generation, and plotting. Do not interpret smoke-test metrics as scientific results.
