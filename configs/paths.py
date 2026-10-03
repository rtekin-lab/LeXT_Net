# -*- coding: utf-8 -*-
"""
configs/paths.py  —  Dataset path configuration
================================================
Edit the BASE variable below to point to the folder that contains
your downloaded datasets, then run any training or evaluation script.

Dataset download links
----------------------
PTB-XL     : https://physionet.org/content/ptb-xl/1.0.3/
             DOI: 10.13026/kfzx-aw45
CPSC 2018  : http://2018.icbeb.org/Challenge.html
Chapman    : https://figshare.com/collections/ChapmanECG/4560497
             (use the ECGDataDenoised folder for cleaner signals)

Expected directory layout
--------------------------
<BASE>/
├── ptb_xl/
│   └── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
│       ├── ptbxl_database.csv
│       └── records100/
│           ├── 00000/  00001_lr.dat  00001_lr.hea  ...
│           └── ...
├── cpsc_2018/
│   ├── A0001.hea
│   ├── A0001.mat
│   └── ...
└── ECGData_Chapman/
    └── ECGData/
        ├── MUSE_20180111_155115_19000.csv
        └── ...
"""

from pathlib import Path

# ── Set this to your datasets root folder ────────────────────────────────────
BASE = Path(r"/path/to/your/datasets")
# Examples:
#   Windows : Path(r"D:\datasets")
#   Linux   : Path("/home/user/datasets")
#   macOS   : Path("/Users/user/datasets")
# ─────────────────────────────────────────────────────────────────────────────

PATHS = {
    # PTB-XL: pass the records100 subfolder directly.
    # ptbxl_database.csv is expected one level up (inside ptb-xl-...-1.0.3/).
    "ptbxl": BASE / "ptb_xl"
             / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
             / "records100",

    # CPSC2018: flat folder containing A*.hea and A*.mat files.
    "cpsc2018": BASE / "cpsc_2018",

    # Chapman-Shaoxing: folder containing MUSE_*.csv files.
    "chapman":  BASE / "ECGData_Chapman" / "ECGData",
}
