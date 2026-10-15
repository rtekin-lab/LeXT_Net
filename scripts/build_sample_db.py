#!/usr/bin/env python
"""Create small standardized 100-Hz/10-s sample subsets from locally downloaded datasets.

This script does not download or redistribute source datasets. It exports selected
records as 12-lead CSV plus manifest.csv so maintainers can review licensing before
committing sample files to GitHub.
"""
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from src.datasets import load_ptbxl, load_chapman, load_cpsc2018, LEAD_ORDER
from configs.paths import PATHS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=['ptbxl','chapman','cpsc2018']); p.add_argument('--n',type=int,default=5); p.add_argument('--out',default=None); a=p.parse_args()
    if a.dataset=='ptbxl':
        sig,ids,meta=load_ptbxl(PATHS['ptbxl'],normalize=False,max_records=a.n,verbose=True); unit='mV'
        patient_map={str(i): str(int(meta.loc[int(i),'patient_id'])) for i in ids}
    elif a.dataset=='chapman': sig,ids=load_chapman(PATHS['chapman'],normalize=False,max_records=a.n,verbose=True); unit='mV'
    else: sig,ids=load_cpsc2018(PATHS['cpsc2018'],normalize=False,max_records=a.n,verbose=True); unit='mV'
    out=Path(a.out or ROOT/'sample_data'/a.dataset); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(x,rid) in enumerate(zip(sig,ids),1):
        name=f'{a.dataset}_{i:03d}.csv'; pd.DataFrame(x,columns=LEAD_ORDER).to_csv(out/name,index=False)
        pid=patient_map.get(str(rid),str(rid)) if a.dataset=='ptbxl' else str(rid)
        rows.append({'path':name,'format':'csv','fs':100,'record_id':rid,'patient_id':pid,'unit':unit})
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    print(f'Wrote {len(rows)} records to {out}')
    print('Review the original dataset licence/redistribution terms before committing these files.')
if __name__=='__main__': main()
