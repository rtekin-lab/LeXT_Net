# ECG-Transformer integration

The existing `ECGTransformer` class is exposed through the normal training pipeline.
The configuration is taken from the original Transformer experiment script `try7TransOnemli.py`:

- `d_model=64`
- `nhead=4`
- `num_layers=3`
- `dim_feedforward=256` (the existing baseline class setting)
- batch size `16`
- loss `MSE`
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- 100 epochs
- no LR scheduler (constant learning rate)

The current repository cross-validation/data strategy pipeline is retained; only Transformer model registration and its original model-specific optimization settings are added.
