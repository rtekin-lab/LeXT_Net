# -*- coding: utf-8 -*-
"""Manifest-based loader for external 12-lead ECG datasets.

The loader converts supported physical amplitude units to **mV**, reorders named
channels to the canonical LeXT-Net lead order, resamples to 100 Hz, and enforces
10-second records before optional per-record/per-lead z-score normalization.

Manifest columns
----------------
path        required. CSV path or WFDB record path (without .hea/.dat)
format      required. ``csv`` or ``wfdb``
fs          required for CSV. For WFDB it is checked against the header when given.
unit        recommended for CSV: ``mV``, ``uV``/``µV``, or ``V``. Default: mV.
patient_id  strongly recommended when a patient has multiple records
record_id   optional human-readable identifier

CSV rules
---------
Named CSV columns may appear in any order and are reordered by lead name.
Headerless CSV files are accepted only under the explicit assumption that their
12 columns already follow the canonical order:
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

WFDB rules
----------
WFDB channel names are read from ``sig_name`` and explicitly reordered. Missing
or duplicated required leads are fatal errors; the loader never silently takes
the first 12 channels. WFDB physical units are read per channel and converted to
mV when they are one of the supported voltage units.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from src.datasets import LEAD_ORDER, _resample_to_target, _ensure_10s, _zscore_normalize


_LEAD_ALIASES = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
}


def canonical_lead_name(name: str) -> str:
    key = str(name).strip().replace(" ", "").upper()
    if key.startswith("ECG"):
        key = key[3:]
    if key in _LEAD_ALIASES:
        return _LEAD_ALIASES[key]
    raise ValueError(f"Unsupported ECG lead name: {name!r}")


def _unit_factor_to_mV(unit: str) -> float:
    u = str(unit).strip().replace("μ", "µ").lower()
    aliases = {
        "mv": 1.0, "millivolt": 1.0, "millivolts": 1.0,
        "uv": 1e-3, "µv": 1e-3, "microvolt": 1e-3, "microvolts": 1e-3,
        "v": 1e3, "volt": 1e3, "volts": 1e3,
    }
    if u not in aliases:
        raise ValueError(
            f"Unsupported physical unit {unit!r}. Use mV, uV/µV, or V, "
            "or convert the source data before loading."
        )
    return aliases[u]


def _validated_lead_indices(names):
    # Ignore non-ECG metadata/extra channels, but never ignore ambiguity among
    # the 12 required standard leads.
    recognized = []
    for i, name in enumerate(names):
        try:
            recognized.append((i, canonical_lead_name(name)))
        except ValueError:
            continue
    labels = [x[1] for x in recognized]
    duplicated = sorted({x for x in labels if labels.count(x) > 1})
    if duplicated:
        raise ValueError(f"Duplicated required lead name(s): {duplicated}")
    missing = [lead for lead in LEAD_ORDER if lead not in labels]
    if missing:
        raise ValueError(f"Missing required lead(s): {missing}. Recognized: {labels}")
    pos = {lead: idx for idx, lead in recognized}
    return [pos[lead] for lead in LEAD_ORDER]


def _read_csv_12(path: Path, unit: str) -> np.ndarray:
    # First try a named-column CSV. Extra columns are allowed and ignored.
    df = pd.read_csv(path, encoding="utf-8-sig")
    try:
        idx = _validated_lead_indices(df.columns)
        sig = df.iloc[:, idx].to_numpy(dtype=np.float32)
    except ValueError:
        # Retry as headerless. Headerless input must contain exactly 12 columns
        # already in the canonical order.
        df = pd.read_csv(path, header=None, encoding="utf-8-sig")
        if df.shape[1] != 12:
            raise ValueError(
                f"CSV {path} has no usable 12-lead header and contains "
                f"{df.shape[1]} columns."
            )
        sig = df.to_numpy(dtype=np.float32)
    return (sig * _unit_factor_to_mV(unit)).astype(np.float32)


def _read_wfdb_12(path: Path, manifest_fs=None) -> tuple[np.ndarray, float]:
    try:
        import wfdb
    except ImportError as e:
        raise ImportError("Install wfdb: pip install wfdb") from e

    data, header = wfdb.rdsamp(str(path))
    names = header.get("sig_name") or []
    if len(names) != data.shape[1]:
        raise ValueError(f"WFDB sig_name metadata missing/incomplete for {path}")
    idx = _validated_lead_indices(names)
    sig = data[:, idx].astype(np.float32)

    units = header.get("units") or []
    if len(units) != data.shape[1]:
        raise ValueError(f"WFDB physical-unit metadata missing/incomplete for {path}")
    ordered_units = [units[i] for i in idx]
    for j, u in enumerate(ordered_units):
        sig[:, j] *= _unit_factor_to_mV(u)

    fs = float(header["fs"])
    if manifest_fs is not None and np.isfinite(float(manifest_fs)):
        if abs(float(manifest_fs) - fs) > 1e-6:
            raise ValueError(
                f"Manifest fs={manifest_fs} disagrees with WFDB header fs={fs} for {path}"
            )
    return sig.astype(np.float32), fs


def load_custom_manifest(manifest_path, data_root=None, normalize=True,
                         max_records=None, verbose=True):
    """Load a manifest-described 12-lead dataset into ``(N, 1000, 12)``.

    Before optional normalization, all supported input formats are converted to
    canonical **mV**. Returns ``signals, record_ids, patient_ids``.
    """
    manifest_path = Path(manifest_path)
    root = Path(data_root) if data_root else manifest_path.parent
    tab = pd.read_csv(manifest_path)
    required = {"path", "format"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if max_records is not None:
        tab = tab.iloc[:max_records].copy()

    signals, record_ids, patient_ids = [], [], []
    warned_default_unit = False
    for _, row in tab.iterrows():
        p = Path(str(row["path"]))
        if not p.is_absolute():
            p = root / p
        fmt = str(row["format"]).strip().lower()

        if fmt == "csv":
            if "fs" not in tab.columns or pd.isna(row.get("fs")):
                raise ValueError(f"CSV record {p} requires an 'fs' value in the manifest")
            fs = float(row["fs"])
            if "unit" in tab.columns and pd.notna(row.get("unit")):
                unit = str(row["unit"])
            else:
                unit = "mV"
                if not warned_default_unit:
                    warnings.warn(
                        "Custom CSV manifest has no 'unit' value; assuming mV. "
                        "Add unit=mV/uV/V explicitly for reproducibility."
                    )
                    warned_default_unit = True
            sig = _read_csv_12(p, unit)
        elif fmt == "wfdb":
            fs_manifest = row.get("fs") if "fs" in tab.columns else None
            sig, fs = _read_wfdb_12(p, fs_manifest)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for {p}")

        sig = _resample_to_target(sig, fs_in=fs)
        sig = _ensure_10s(sig, source_label=str(p))
        if sig is None:
            continue
        if normalize:
            sig = _zscore_normalize(sig)

        rid = (str(row["record_id"]) if "record_id" in tab.columns
               and pd.notna(row.get("record_id")) else p.stem)
        pid = (str(row["patient_id"]) if "patient_id" in tab.columns
               and pd.notna(row.get("patient_id")) else rid)
        signals.append(sig.astype(np.float32))
        record_ids.append(rid)
        patient_ids.append(pid)
        if verbose and (len(signals) % 100 == 0):
            print(f"\r  Custom  {len(signals):>6}/{len(tab)}", end="", flush=True)

    if not signals:
        raise RuntimeError("No valid custom ECG records were loaded.")
    if verbose:
        print(f"\n  Custom: loaded {len(signals):,} records from {manifest_path}")
    return np.stack(signals), record_ids, patient_ids
