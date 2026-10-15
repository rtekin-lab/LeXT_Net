#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate pretrained LeXT-Net checkpoints without retraining.

Built-in dataset + same-dataset weights:
  Uses the original fold definition and produces out-of-fold (OOF) predictions.
  This avoids evaluating a record with a fold model that was trained on it.

External/custom evaluation (or --weights-dir from another dataset):
  Evaluates every record with one selected fold or a 5-checkpoint ensemble.

Reports normalized-space metrics and, when a 12-lead reference is available,
retrospective physical-scale metrics using each target lead's per-record mean/std.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_ptbxl, load_chapman, load_cpsc2018, LEAD_ORDER
from src.custom_data import load_custom_manifest
from src.transforms import apply_strategy
from src.crossval import sequential_patient_folds
from src.utils import normalize_per_record_leadwise, compute_lead_metrics_batch, METRIC_KEYS
from src.models.leXT_net import TCNTrunk12UNetHeadsExtendedFast
from scripts.train_multidataset import MODEL_CFG
from configs.paths import PATHS


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['ptbxl','chapman','cpsc2018','custom'],
                   help='Dataset to evaluate')
    p.add_argument('--weights-dataset', default=None, choices=['ptbxl','chapman','cpsc2018'],
                   help='Dataset that produced the checkpoints. Default: same as --dataset for built-ins.')
    p.add_argument('--strategy', default='EGV3', choices=['EGV1','EGV2','EGV3','QSI'])
    p.add_argument('--model', default='leXT-netv2', choices=list(MODEL_CFG.keys()))
    p.add_argument('--fold', default='all', choices=['all','1','2','3','4','5'],
                   help="For same-dataset evaluation: OOF fold(s). For external data: one checkpoint or all-fold ensemble.")
    p.add_argument('--weights-dir', default=None,
                   help='Override checkpoint directory containing fold1.pth ... fold5.pth. Also set --weights-dataset to declare their training dataset when known.')
    p.add_argument('--manifest', default=None, help='Required for --dataset custom')
    p.add_argument('--data-root', default=None)
    p.add_argument('--max-rec', type=int, default=None)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--out-dir', default='results/evaluation')
    p.add_argument('--plot-records', type=int, default=3)
    p.add_argument('--plot-space', default='physical', choices=['physical','normalized'],
                   help='Amplitude space used for overlay plots (default: physical mV).')
    p.add_argument('--no-cuda', action='store_true')
    return p.parse_args()


def load_raw(a):
    patient_ids=None
    if a.dataset=='ptbxl':
        x, ids, meta = load_ptbxl(PATHS['ptbxl'], normalize=False, max_records=a.max_rec, verbose=True)
        try:
            patient_ids=[str(int(meta.loc[int(r),'patient_id'])) for r in ids]
        except Exception:
            patient_ids=None
        unit='mV'
    elif a.dataset=='chapman':
        x, ids = load_chapman(PATHS['chapman'], normalize=False, max_records=a.max_rec, verbose=True); unit='mV'
    elif a.dataset=='cpsc2018':
        x, ids = load_cpsc2018(PATHS['cpsc2018'], normalize=False, max_records=a.max_rec, verbose=True); unit='mV'
    else:
        if not a.manifest: raise ValueError('--manifest is required for custom data')
        x, ids, patient_ids = load_custom_manifest(a.manifest, a.data_root, normalize=False, max_records=a.max_rec, verbose=True); unit='mV'
    return x, ids, patient_ids, unit


def make_model(name, device):
    return TCNTrunk12UNetHeadsExtendedFast(input_size=3, output_size=12, **MODEL_CFG[name]).to(device)


def checkpoint_path(a, fold, weights_dataset):
    if a.weights_dir:
        return Path(a.weights_dir)/f'fold{fold}.pth'
    return ROOT/'results'/'weights'/weights_dataset/a.strategy/a.model/f'fold{fold}.pth'


def load_checkpoint(a, fold, weights_dataset, device):
    ck=checkpoint_path(a,fold,weights_dataset)
    if not ck.exists(): raise FileNotFoundError(f'Missing checkpoint: {ck}')
    m=make_model(a.model,device)
    state=torch.load(ck,map_location=device,weights_only=True)
    m.load_state_dict(state,strict=True); m.eval()
    return m,ck


def infer(model, x, device, batch):
    out=[]
    with torch.no_grad():
        for s in range(0,len(x),batch):
            out.append(model(torch.from_numpy(x[s:s+batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def summarize(md):
    return {m: float(np.nanmean([md[l][m] for l in LEAD_ORDER])) for m in METRIC_KEYS}


def plot_overlay(y,p,rid,out,space='physical'):
    import matplotlib.pyplot as plt
    t=np.arange(y.shape[0])/100.0
    fig,axes=plt.subplots(12,1,figsize=(13,16),sharex=True)
    for i,(ax,lead) in enumerate(zip(axes,LEAD_ORDER)):
        ax.plot(t,y[:,i],lw=.8,label='Reference'); ax.plot(t,p[:,i],lw=.8,alpha=.8,label='Reconstruction')
        ax.set_ylabel(lead,rotation=0,labelpad=20,fontsize=8)
        if i==0: ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)'); fig.suptitle(f'{rid} | {space} evaluation' + (' (mV)' if space=='physical' else ' (z-score)'))
    fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)




def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def leadwise_frame(md):
    rows=[]
    for lead in LEAD_ORDER:
        row={'lead':lead}
        row.update({m:md[lead].get(m,np.nan) for m in METRIC_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a=args_parser(); device=torch.device('cpu' if a.no_cuda or not torch.cuda.is_available() else 'cuda')
    weights_dataset=a.weights_dataset or (a.dataset if a.dataset!='custom' else None)
    if weights_dataset is None and not a.weights_dir:
        raise ValueError('For custom evaluation, provide --weights-dir or --weights-dataset.')

    raw,ids,patient_ids,unit=load_raw(a)
    norm,stats=normalize_per_record_leadwise(raw,'zscore'); x3=apply_strategy(norm,a.strategy)
    selected_folds=[1,2,3,4,5] if a.fold=='all' else [int(a.fold)]

    same_dataset=(a.dataset!='custom' and weights_dataset==a.dataset)
    if same_dataset and a.max_rec is not None:
        raise ValueError('--max-rec cannot be used for same-dataset OOF evaluation because it changes the fold definition. Use the full dataset, or evaluate a curated subset as --dataset custom with --weights-dir.')
    evaluated_indices=None
    checkpoints=[]

    if same_dataset:
        # Strict OOF evaluation: each record is predicted only by the fold model for which it was held out.
        folds=sequential_patient_folds(ids,patient_ids,n_folds=5)
        idx_parts=[]; pred_parts=[]
        for f in selected_folds:
            _,te=folds[f-1]
            m,ck=load_checkpoint(a,f,weights_dataset,device); checkpoints.append(str(ck))
            pred_parts.append(infer(m,x3[te],device,a.batch)); idx_parts.append(te)
            print(f'OOF fold {f}: {len(te)} records')
        evaluated_indices=np.concatenate(idx_parts)
        pred_norm=np.concatenate(pred_parts)
        # Keep references in the exact concatenated order.
        norm_eval=norm[evaluated_indices]; raw_eval=raw[evaluated_indices]
        mu_eval=stats['mu'][evaluated_indices]; sd_eval=stats['sd'][evaluated_indices]
        ids_eval=[ids[i] for i in evaluated_indices]
        protocol='out-of-fold'
    else:
        # External/custom test set: every selected checkpoint can evaluate every record.
        pred_folds=[]
        for f in selected_folds:
            m,ck=load_checkpoint(a,f,weights_dataset,device); checkpoints.append(str(ck))
            pred_folds.append(infer(m,x3,device,a.batch)); print(f'External evaluation fold {f} complete')
        pred_norm=np.mean(pred_folds,axis=0)
        norm_eval=norm; raw_eval=raw; mu_eval=stats['mu']; sd_eval=stats['sd']; ids_eval=ids
        protocol='external-ensemble' if len(selected_folds)>1 else 'external-single-checkpoint'

    pred_phys=(pred_norm*sd_eval+mu_eval).astype(np.float32)
    mn=compute_lead_metrics_batch(norm_eval,pred_norm,verbose=True)
    mp=compute_lead_metrics_batch(raw_eval,pred_phys,verbose=True)

    weights_tag=weights_dataset or 'custom_weights'
    out=Path(a.out_dir)/f'{a.dataset}__weights-{weights_tag}'/a.strategy/a.model
    out.mkdir(parents=True,exist_ok=True)
    ck_meta=[{'path':c,'sha256':sha256_file(c)} for c in checkpoints]
    result={'dataset':a.dataset,'weights_dataset':weights_dataset,'strategy':a.strategy,'model':a.model,
            'folds':selected_folds,'protocol':protocol,'n_records':len(norm_eval),'physical_unit':'mV',
            'sampling_rate_hz':100,'duration_s':10,'normalization':'per-record per-lead z-score',
            'checkpoints':ck_meta,'summary_normalized_mAll':summarize(mn),'summary_physical_mAll':summarize(mp),
            'metrics_normalized':mn,'metrics_physical':mp,
            'note_physical':'Retrospective evaluation in mV: inverse transform uses each labelled 12-lead reference record\'s per-lead mean/std. These target-lead statistics are not available in a three-lead-only deployment.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=True),encoding='utf-8')
    (out/'run_metadata.json').write_text(json.dumps({k:result[k] for k in ['dataset','weights_dataset','strategy','model','folds','protocol','n_records','physical_unit','sampling_rate_hz','duration_s','normalization','checkpoints','note_physical']},indent=2),encoding='utf-8')
    pd.DataFrame([result['summary_normalized_mAll']]).to_csv(out/'summary_normalized.csv',index=False)
    pd.DataFrame([result['summary_physical_mAll']]).to_csv(out/'summary_physical.csv',index=False)
    leadwise_frame(mn).to_csv(out/'leadwise_metrics_normalized.csv',index=False)
    leadwise_frame(mp).to_csv(out/'leadwise_metrics_physical_mV.csv',index=False)
    # Backward-compatible compact summary.
    pd.DataFrame([result['summary_normalized_mAll'],result['summary_physical_mAll']],index=['normalized','physical_mV']).to_csv(out/'summary.csv')
    for i in range(min(a.plot_records,len(norm_eval))):
        if a.plot_space=='physical':
            plot_overlay(raw_eval[i],pred_phys[i],ids_eval[i],out/f'overlay_{i+1:03d}_physical_mV.png','physical')
        else:
            plot_overlay(norm_eval[i],pred_norm[i],ids_eval[i],out/f'overlay_{i+1:03d}_normalized.png','normalized')
    print('\nSaved:',out); print('Protocol:',protocol); print('Normalized mAll:',result['summary_normalized_mAll']); print('Physical mAll:',result['summary_physical_mAll'],'[mV]')

if __name__=='__main__': main()
