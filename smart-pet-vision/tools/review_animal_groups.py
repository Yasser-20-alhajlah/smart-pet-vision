"""Create a NEW reviewed-group split, never overwrite the historical split.

Input embeddings must cover ALL retained images in metadata order (not nearest
test-to-train pairs). Proposals are image-similarity candidates, not identities.
Use proposals, review every pair, then use split. Training is a separate step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from spv.data import group_stratified_split, leakage_audit


def propose(meta, features, threshold):
    features=np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or len(features) != len(meta) or not np.isfinite(features).all()
            or not 0 < threshold <= 1 or (np.linalg.norm(features,axis=1)==0).any()):
        raise ValueError('Need finite nonzero full-dataset embeddings and threshold in (0,1].')
    features=features/np.linalg.norm(features,axis=1,keepdims=True)
    rows=[]
    # Every unordered pair is considered, including train/train and val/test.
    for start in range(0,len(meta),256):
        sims=features[start:start+256] @ features.T
        for local,scores in enumerate(sims):
            i=start+local
            for j in np.flatnonzero(scores >= threshold):
                if j <= i:continue
                rows.append({'image_a':meta.iloc[i].image,'image_b':meta.iloc[j].image,
                             'cosine':float(scores[j]),'decision':'pending','reviewer':'','evidence':''})
    return pd.DataFrame(rows,columns=['image_a','image_b','cosine','decision','reviewer','evidence'])


def reviewed_split(meta, proposals, reviews, seed=42):
    if meta.image.isna().any() or meta.image.duplicated().any():raise ValueError('Invalid image IDs.')
    names=set(meta.image);index={v:i for i,v in enumerate(meta.image)};parent=list(range(len(meta)))
    def find(i):
        while parent[i] != i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        x,y=find(index[a]),find(index[b]);parent[max(x,y)]=min(x,y)
    def pair_keys(frame):
        if not {'image_a','image_b'}.issubset(frame):raise ValueError('Missing pair columns.')
        keys=[]
        for a,b in zip(frame.image_a,frame.image_b):
            if a not in names or b not in names or a==b:raise ValueError('Unknown or self image pair.')
            keys.append(tuple(sorted((a,b))))
        if len(set(keys)) != len(keys):raise ValueError('Duplicate review pairs.')
        return set(keys)
    if pair_keys(proposals) != pair_keys(reviews):raise ValueError('Review coverage must match proposals exactly.')
    if not {'decision','reviewer','evidence'}.issubset(reviews):raise ValueError('Review sign-off is missing.')
    if not reviews.decision.isin(['same','different','uncertain']).all():raise ValueError('Pending identity reviews block split creation.')
    if reviews[['reviewer','evidence']].fillna('').apply(lambda c:c.astype(str).str.strip().eq('')).any().any():
        raise ValueError('Every decision requires reviewer and evidence.')
    for key in ['group','md5']:
        if key not in meta or meta[key].isna().any():raise ValueError('Existing duplicate records must be complete.')
        for _,f in meta.groupby(key):
            for b in f.image.iloc[1:]:union(f.image.iloc[0],b)
    # Uncertain relations are grouped conservatively, never treated as independent.
    for row in reviews.itertuples():
        if row.decision in ['same','uncertain']:union(row.image_a,row.image_b)
    for row in reviews.itertuples():
        if row.decision=='different' and find(index[row.image_a])==find(index[row.image_b]):
            raise ValueError('Transitive grouping contradicts a reviewed different-animal decision.')
    out=meta.copy();out['historical_split']=out.split;out['historical_duplicate_group']=out.group
    out['animal_group']=[f'candidate_{find(i):05d}' for i in range(len(out))]
    out['group']=out.animal_group
    if (out.groupby('group').species.nunique()>1).any():raise ValueError('A proposed animal group contains cat and dog labels.')
    out['split']=group_stratified_split(out,{'train':.70,'val':.15,'test':.15},seed)
    audit=leakage_audit(out)
    return out,audit


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('proposals');p.add_argument('--metadata',type=Path,required=True);p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--threshold',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('split');p.add_argument('--metadata',type=Path,required=True);p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('Choose a NEW output path.')
    meta=pd.read_csv(a.metadata)
    if a.action=='proposals':
        with np.load(a.embeddings,allow_pickle=False) as z:
            if not np.array_equal(z['image'].astype(str),meta.image.astype(str).to_numpy()):raise ValueError('Embedding IDs/order differ from metadata.')
            proposals=propose(meta,z['features'],a.threshold)
        a.output.parent.mkdir(parents=True,exist_ok=True);proposals.to_csv(a.output,index=False)
    else:
        result,audit=reviewed_split(meta,pd.read_csv(a.proposals),pd.read_csv(a.reviews))
        a.output.mkdir(parents=True);result.to_csv(a.output/'metadata.csv',index=False)
        audit.update(scope='Separation of reviewed candidate groups, not proof of all animal identities',
                     input_sha256={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ['metadata','proposals','reviews']},
                     split_sha256=hashlib.sha256((a.output/'metadata.csv').read_bytes()).hexdigest(),
                     new_training_completed=False,independent_reproduction_completed=False)
        (a.output/'group_split_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
