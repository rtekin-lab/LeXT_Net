# Changelog

## 1.0.0-pre — reproducibility upgrade

- Added fixed-fold and full 5-fold training workflows for PTB-XL, Chapman, CPSC2018, and custom datasets.
- Added pretrained OOF/cross-database evaluation and richer CSV/JSON/plot reports.
- Standardized built-in physical evaluation to mV before normalization.
- Hardened custom CSV/WFDB lead-name mapping and physical-unit conversion.
- Added custom dataset validator and synthetic integration smoke test.
- Added checkpoint SHA-256 manifest generator and checkpoint provenance in evaluation metadata.
- Added GitHub Actions basic CI, `CITATION.cff`, and expanded reproducibility documentation.
