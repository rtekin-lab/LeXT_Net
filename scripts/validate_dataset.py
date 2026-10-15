#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate a custom ECG manifest before training/evaluation."""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.custom_data import load_custom_manifest
from src.datasets import LEAD_ORDER, TARGET_FS, TARGET_LEN
from src.crossval import sequential_patient_folds


def main():
    p=argparse.ArgumentParser(description="Validate LeXT-Net custom 12-lead ECG data")
    p.add_argument('--manifest',required=True)
    p.add_argument('--data-root',default=None)
    p.add_argument('--max-rec',type=int,default=None)
    p.add_argument('--check-folds',action='store_true',help='Also verify patient/group 5-fold splitting')
    a=p.parse_args()
    x,ids,pids=load_custom_manifest(a.manifest,a.data_root,normalize=False,max_records=a.max_rec,verbose=True)
    problems=[]
    if x.ndim!=3 or x.shape[1:]!=(TARGET_LEN,12): problems.append(f"unexpected shape {x.shape}")
    if not np.isfinite(x).all(): problems.append('NaN/Inf found')
    if len(ids)!=len(set(ids)): problems.append('duplicate record_id values')
    if np.max(np.abs(x))>100:
        print('WARNING: absolute amplitude exceeds 100 mV; verify the manifest unit field.')
    print(f"Shape          : {x.shape}")
    print(f"Internal fs    : {TARGET_FS} Hz")
    print(f"Duration       : {TARGET_LEN/TARGET_FS:.1f} s")
    print(f"Lead order     : {', '.join(LEAD_ORDER)}")
    print('Physical unit  : mV (before optional z-score normalization)')
    print(f"Records        : {len(ids)}")
    print(f"Unique patients: {len(set(pids))}")
    print(f"Amplitude range: [{float(x.min()):.6g}, {float(x.max()):.6g}] mV")
    if a.check_folds:
        folds=sequential_patient_folds(ids,pids,n_folds=5)
        for i,(tr,te) in enumerate(folds,1):
            trp={pids[j] for j in tr}; tep={pids[j] for j in te}
            overlap=trp & tep
            if overlap: problems.append(f'fold {i}: patient leakage ({len(overlap)} IDs)')
            print(f"Fold {i}: train={len(tr):5d} test={len(te):5d} patient_overlap={len(overlap)}")
    if problems:
        print('\nFAILED:')
        for q in problems: print(' -',q)
        raise SystemExit(2)
    print('\nVALIDATION PASSED')

if __name__=='__main__': main()
