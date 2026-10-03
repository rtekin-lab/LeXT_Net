#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_weights.py  —  Download pre-trained LeXT-Net weights from Zenodo
=========================================================================
Downloads the pre-trained model weights published alongside the paper:

    LeXT-Net: A Lead-Adaptive Hierarchical Network for
    3-Lead to 12-Lead ECG Reconstruction
    Tekin et al., Scientific Reports (2025)
    Zenodo: https://doi.org/10.5281/zenodo.23123820

Weights are stored under:
    results/weights/<dataset>/<strategy>/<model>/fold<k>.pth

Usage:
    # Download all available weights (~391 MB)
    python scripts/download_weights.py

    # Download only a specific dataset / strategy / model
    python scripts/download_weights.py --dataset ptbxl --strategy EGV3 --model leXT-netv1s

    # List available weight files without downloading
    python scripts/download_weights.py --list
"""

import os
import sys
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
ZENODO_DOI = "10.5281/zenodo.23123820"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_DOI.split('zenodo.')[1]}"

# ── Weight file registry ───────────────────────────────────────────────────────
# Each entry: relative path inside results/weights/ → Zenodo filename
# (Populated automatically from the Zenodo record; listed here as fallback)
WEIGHT_FILES = []
for strategy in ["EGV1", "EGV2", "EGV3", "QSI"]:
    for fold in range(1, 6):
        rel = f"ptbxl/{strategy}/leXT-netv1s/fold{fold}.pth"
        WEIGHT_FILES.append(rel)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download LeXT-Net pre-trained weights")
    p.add_argument("--dataset",  default="all",
                   choices=["all", "ptbxl", "chapman", "cpsc2018"])
    p.add_argument("--strategy", default="all",
                   choices=["all", "EGV1", "EGV2", "EGV3", "QSI"])
    p.add_argument("--model",    default="all",
                   choices=["all", "leXT-netv1s", "leXT-netv2s",
                            "leXT-netv1", "leXT-netv2"])
    p.add_argument("--list",     action="store_true",
                   help="List available files without downloading")
    p.add_argument("--force",    action="store_true",
                   help="Re-download even if file already exists")
    return p.parse_args()


def filter_files(files, dataset, strategy, model):
    result = []
    for f in files:
        parts = Path(f).parts   # e.g. ('ptbxl', 'EGV1', 'leXT-netv1s', 'fold1.pth')
        if len(parts) < 4:
            continue
        ds, strat, mdl, _ = parts[0], parts[1], parts[2], parts[3]
        if dataset  != "all" and ds    != dataset:  continue
        if strategy != "all" and strat != strategy: continue
        if model    != "all" and mdl   != model:    continue
        result.append(f)
    return result


def fetch_zenodo_files():
    """Query Zenodo API to get actual download URLs."""
    try:
        with urllib.request.urlopen(ZENODO_API, timeout=10) as resp:
            data = json.loads(resp.read())
        files = {}
        for f in data.get("files", []):
            name = f["key"]
            url  = f["links"]["self"]
            files[name] = url
        return files
    except Exception as e:
        print(f"  Warning: Could not reach Zenodo API ({e})")
        print(f"  Visit https://doi.org/{ZENODO_DOI} to download manually.")
        return {}


def download_file(url, dest_path, force=False):
    if dest_path.exists() and not force:
        print(f"  Already exists: {dest_path.name}")
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(".tmp")
    try:
        print(f"  Downloading {dest_path.name} ...", end="", flush=True)
        def _progress(count, block, total):
            if total > 0:
                pct = min(count * block / total * 100, 100)
                print(f"\r  Downloading {dest_path.name} ... {pct:.0f}%",
                      end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest_path)
        print(f"\r  ✓ {dest_path.name} ({dest_path.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"\r  ✗ Failed: {dest_path.name} — {e}")
        if tmp.exists(): tmp.unlink()
        return False


def main():
    args = parse_args()

    files = filter_files(WEIGHT_FILES, args.dataset, args.strategy, args.model)

    if args.list:
        print(f"\nAvailable weight files ({len(files)} total):")
        for f in files:
            dest = ROOT / "results" / "weights" / f
            status = "✓ present" if dest.exists() else "  missing"
            print(f"  [{status}]  {f}")
        print(f"\nZenodo: https://doi.org/{ZENODO_DOI}")
        return

    if not files:
        print("No matching weight files found.")
        return

    print(f"\nLeXT-Net weight downloader")
    print(f"Zenodo: https://doi.org/{ZENODO_DOI}")
    print(f"Files to download: {len(files)}")
    print(f"Destination: {ROOT / 'results' / 'weights'}\n")

    # Fetch actual URLs from Zenodo
    zenodo_files = fetch_zenodo_files()
    if not zenodo_files:
        print(f"\nCould not connect to Zenodo automatically.")
        print(f"Please download manually from:")
        print(f"  https://doi.org/{ZENODO_DOI}")
        print(f"\nAfter downloading, place .pth files under:")
        print(f"  results/weights/<dataset>/<strategy>/<model>/fold<k>.pth")
        return

    ok, fail = 0, 0
    for rel_path in files:
        # Zenodo filename format: dataset__strategy__model__fold<k>.pth
        zenodo_name = rel_path.replace("/", "__")
        if zenodo_name not in zenodo_files:
            print(f"  ? Not found on Zenodo: {zenodo_name}")
            fail += 1
            continue
        dest = ROOT / "results" / "weights" / rel_path
        if download_file(zenodo_files[zenodo_name], dest, force=args.force):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"  Downloaded: {ok}   Failed: {fail}")
    if ok > 0:
        print(f"\nNow run inference without training:")
        print(f"  python scripts/predict.py --input your_ecg.csv \\")
        print(f"      --dataset ptbxl --strategy EGV3")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
