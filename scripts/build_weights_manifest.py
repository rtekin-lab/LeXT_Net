#!/usr/bin/env python3
"""Create a SHA-256 provenance manifest for released checkpoint files."""
import argparse, csv, hashlib, re
from pathlib import Path

PAT=re.compile(r'^(ptbxl|chapman|cpsc2018)__(EGV1|EGV2|EGV3|QSI)__([^_]+)__fold([1-5])\.pth$')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights-dir',required=True,help='Directory containing release-named .pth files')
    p.add_argument('--out',default='weights_manifest.csv')
    p.add_argument('--code-version',default='')
    a=p.parse_args()
    rows=[]
    for fp in sorted(Path(a.weights_dir).glob('*.pth')):
        m=PAT.match(fp.name)
        if not m:
            print('SKIP non-release filename:',fp.name); continue
        ds,st,model,fold=m.groups()
        rows.append({'dataset':ds,'strategy':st,'model':model,'fold':fold,'filename':fp.name,
                     'sha256':sha256(fp),'size_bytes':fp.stat().st_size,'code_version':a.code_version})
    if not rows: raise SystemExit('No release-named checkpoints found.')
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} checkpoint entries -> {a.out}')

if __name__=='__main__': main()
