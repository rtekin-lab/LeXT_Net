#!/usr/bin/env python3
"""Collect trained LeXT-Net checkpoints into the Zenodo release naming scheme.

Expected training-result layout:
  <results-root>/<dataset>/leXT-netv2/<strategy>_fold<k>/fold<k>.pth

The script copies (never moves) checkpoints into --out-dir and can then be
followed by build_weights_manifest.py.
"""
import argparse, shutil
from pathlib import Path

DATASETS=('ptbxl','chapman','cpsc2018')
STRATEGIES=('EGV1','EGV2','EGV3','QSI')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-root',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--model',default='leXT-netv2')
    p.add_argument('--strict',action='store_true',help='Fail if any expected checkpoint is missing')
    a=p.parse_args()
    root=Path(a.results_root); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    copied=[]; missing=[]
    for ds in DATASETS:
        for st in STRATEGIES:
            for fold in range(1,6):
                src=root/ds/a.model/f'{st}_fold{fold}'/f'fold{fold}.pth'
                dst=out/f'{ds}__{st}__{a.model}__fold{fold}.pth'
                if not src.exists():
                    missing.append(src); continue
                shutil.copy2(src,dst); copied.append(dst)
                print('COPY',src,'->',dst)
    print(f'\nCopied: {len(copied)}/60')
    if missing:
        print(f'Missing: {len(missing)}')
        for x in missing: print('  ',x)
        if a.strict: raise SystemExit(2)

if __name__=='__main__': main()
