#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, dataset-free LeXT-Net smoke test using synthetic 12-lead ECG records.

This is a software/integration test, not a scientific benchmark. It validates the
custom manifest loader, unit conversion, resampling, grouped splitting, model
forward pass, metric computation, and plotting without downloading a real ECG
corpus or pretrained checkpoint.
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.datasets import LEAD_ORDER
from src.custom_data import load_custom_manifest
from src.crossval import sequential_patient_folds
from src.transforms import apply_strategy
from src.utils import compute_lead_metrics_batch
from src.models.leXT_net import TCNTrunk12UNetHeadsExtendedFast
from scripts.train_multidataset import MODEL_CFG


def synth(fs=250, seconds=10, n=6):
    t=np.arange(int(fs*seconds))/fs
    arr=[]
    for r in range(n):
        leads=[]
        for k in range(12):
            # Simple deterministic ECG-like test signal in mV, not physiological truth.
            y=0.25*np.sin(2*np.pi*(1.0+0.02*k)*t+0.1*k)
            y+=0.04*np.sin(2*np.pi*3*t+0.2*r)
            leads.append(y)
        arr.append(np.stack(leads,axis=1).astype(np.float32))
    return arr


def main():
    out=ROOT/'results'/'smoke_test'; out.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='lext_smoke_'))
    try:
        rows=[]
        for i,x in enumerate(synth(),1):
            fn=f'r{i:02d}.csv'
            # Write in deliberately shuffled named-column order to test reordering.
            order=[6,0,11,1,7,2,8,3,9,4,10,5]
            pd.DataFrame(x[:,order],columns=[LEAD_ORDER[j] for j in order]).to_csv(tmp/fn,index=False)
            rows.append({'path':fn,'format':'csv','fs':250,'unit':'mV','patient_id':f'P{(i-1)//2:02d}','record_id':f'R{i:02d}'})
        man=tmp/'manifest.csv'; pd.DataFrame(rows).to_csv(man,index=False)
        raw,ids,pids=load_custom_manifest(man,normalize=False,verbose=False)
        norm,_,_=load_custom_manifest(man,normalize=True,verbose=False)
        folds=sequential_patient_folds(ids,pids,n_folds=3)
        for tr,te in folds:
            assert not ({pids[j] for j in tr} & {pids[j] for j in te})
        x3=apply_strategy(norm,'EGV3')
        model=TCNTrunk12UNetHeadsExtendedFast(input_size=3,output_size=12,**MODEL_CFG['leXT-netv2'])
        model.eval()
        with torch.no_grad(): pred=model(torch.from_numpy(x3[:1])).cpu().numpy()
        metrics=compute_lead_metrics_batch(norm[:1],pred,verbose=False)
        (out/'metrics_smoke.json').write_text(json.dumps(metrics,indent=2,allow_nan=True))
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,3)); ax.plot(norm[0,:,1],label='Reference II'); ax.plot(pred[0,:,1],label='Random-model output II'); ax.legend(); ax.set_xlabel('sample'); fig.tight_layout(); fig.savefig(out/'smoke_overlay.png',dpi=120); plt.close(fig)
        print('SMOKE TEST PASSED')
        print(' raw shape :',raw.shape,'(canonical mV)')
        print(' norm shape:',norm.shape)
        print(' model out :',pred.shape)
        print(' outputs   :',out)
        print('NOTE: random model + synthetic data; metrics are not scientific results.')
    finally:
        shutil.rmtree(tmp,ignore_errors=True)

if __name__=='__main__': main()
