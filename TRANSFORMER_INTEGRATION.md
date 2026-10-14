# ECG-Transformer integration

The existing `ECGTransformer` class is exposed through the normal training pipeline.
The configuration is taken from the original Transformer experiment script `try7TransOnemli.py`:

- `d_model=64`
- `nhead=4`
- `num_layers=3`
- `dim_feedforward=256` (the existing baseline class setting)
- batch size `32`
- loss `MSE`
- AdamW, learning rate `3e-4`, weight decay `1e-4`
- 100 epochs
- 5-epoch linear warmup followed by cosine annealing

The current repository cross-validation/data strategy pipeline is retained. The Transformer is trained with the same manuscript-specified optimizer, scheduler, batch size, preprocessing, and training duration as the other comparison models.
