#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download pretrained LeXT-Net checkpoints from the project Zenodo record.

Expected Zenodo filename convention:
  <dataset>__<strategy>__<model>__fold<k>.pth
"""
import argparse, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZENODO_DOI = "10.5281/zenodo.23123820"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_DOI.split('zenodo.')[1]}"


def parse_args():
    p=argparse.ArgumentParser(description='Download LeXT-Net pretrained weights')
    p.add_argument('--dataset',default='all',choices=['all','ptbxl','chapman','cpsc2018'])
    p.add_argument('--strategy',default='all',choices=['all','EGV1','EGV2','EGV3','QSI'])
    p.add_argument('--model',default='all',choices=['all','leXT-netv1','leXT-netv1s','leXT-netv2','leXT-netv2s'])
    p.add_argument('--fold',default='all',choices=['all','1','2','3','4','5'])
    p.add_argument('--list',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--no-verify',action='store_true',help='Skip checksum verification supplied by Zenodo')
    return p.parse_args()


def fetch_files():
    try:
        with urllib.request.urlopen(ZENODO_API,timeout=20) as r:
            data=json.loads(r.read())
    except Exception as e:
        raise RuntimeError(f"Could not query Zenodo record {ZENODO_DOI}: {e}\nVisit https://doi.org/{ZENODO_DOI} manually.")
    out=[]
    for f in data.get('files',[]):
        name=f.get('key','')
        if not name.endswith('.pth') or '__' not in name: continue
        parts=name[:-4].split('__')
        if len(parts)!=4: continue
        ds,st,mo,foldname=parts
        if not foldname.startswith('fold'): continue
        out.append({'name':name,'dataset':ds,'strategy':st,'model':mo,'fold':foldname[4:],
                    'url':f['links']['self'],'checksum':f.get('checksum','')})
    return out


def filt(files,a):
    return [f for f in files if (a.dataset=='all' or f['dataset']==a.dataset)
            and (a.strategy=='all' or f['strategy']==a.strategy)
            and (a.model=='all' or f['model']==a.model)
            and (a.fold=='all' or f['fold']==a.fold)]


def digest(path,algorithm):
    h=hashlib.new(algorithm)
    with open(path,'rb') as fh:
        for b in iter(lambda:fh.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def verify(path,checksum):
    if not checksum: return True, 'no remote checksum'
    if ':' not in checksum: return True, f'unrecognized checksum format: {checksum}'
    alg,expected=checksum.split(':',1)
    alg=alg.lower()
    try: actual=digest(path,alg)
    except ValueError: return True, f'unsupported checksum algorithm: {alg}'
    return actual.lower()==expected.lower(), f'{alg}:{actual}'


def download(item,dest,force=False,do_verify=True):
    if dest.exists() and not force:
        if do_verify:
            ok,msg=verify(dest,item.get('checksum',''))
            if not ok:
                print('  checksum mismatch; re-download',dest)
            else:
                print('  present/verified',dest); return True
        else:
            print('  present ',dest); return True
    dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix('.tmp')
    try:
        print('  download',dest)
        urllib.request.urlretrieve(item['url'],tmp)
        if do_verify:
            ok,msg=verify(tmp,item.get('checksum',''))
            if not ok: raise RuntimeError(f'checksum verification failed ({msg})')
        tmp.replace(dest); return True
    except Exception as e:
        print('  FAILED  ',dest,e)
        if tmp.exists(): tmp.unlink()
        return False


def main():
    a=parse_args(); files=filt(fetch_files(),a)
    if not files:
        print('No matching .pth checkpoint files are currently exposed by the Zenodo record.')
        print(f'Zenodo: https://doi.org/{ZENODO_DOI}')
        return
    files=sorted(files,key=lambda f:(f['dataset'],f['strategy'],f['model'],int(f['fold'])))
    if a.list:
        print(f'Available checkpoints ({len(files)}):')
        for f in files:
            ck=f" [{f['checksum']}]" if f.get('checksum') else ''
            print(f"  {f['dataset']}/{f['strategy']}/{f['model']}/fold{f['fold']}.pth{ck}")
        return
    ok=0
    for f in files:
        dest=ROOT/'results'/'weights'/f['dataset']/f['strategy']/f['model']/f"fold{f['fold']}.pth"
        ok += int(download(f,dest,a.force,not a.no_verify))
    print(f'\nDownloaded/present: {ok}/{len(files)}')
    if ok != len(files): raise SystemExit(2)

if __name__=='__main__': main()
