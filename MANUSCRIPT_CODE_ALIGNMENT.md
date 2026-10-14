# Manuscript–code alignment

This repository revision aligns the comparative-training implementation with the submitted manuscript.

- All comparison models use AdamW with `lr=3e-4` and `weight_decay=1e-4`.
- All comparison models use a 5-epoch linear warmup followed by cosine annealing.
- All comparison models use gradient clipping at norm 1.0.
- Default training duration remains 100 epochs.
- Comparison batch size is unified at 32.
- The TCN baseline uses 6 causal dilated residual blocks with 128 feature channels and kernel size 7.
- The BiLSTM output projection is a 2-layer MLP, matching the manuscript description.
- ECG-Transformer remains available from the standard CLI and follows the same optimization schedule.

Note: changing architecture or optimization settings changes the model being trained. Any numerical results claimed for these exact settings should be produced (or reproduced) with this aligned revision.
