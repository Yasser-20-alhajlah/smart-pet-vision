#!/usr/bin/env python3
"""Paired baseline/candidate regression on one frozen cohort using trained models.

The previously-inspected 401 team photos are REGRESSION DATA, not untouched holdout.
This tool does not tune thresholds, train, change weights, or overwrite past runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_cohort(images: Path, labels: pd.DataFrame):
    required={'image','n_animals','species'}
    if not required.issubset(labels.columns):
        raise ValueError(f'Labels require columns: {sorted(required)}.')
    if labels.empty or labels.image.isna().any() or labels.image.duplicated().any():
        raise ValueError('Missing/duplicate image identifiers in labels.')
    if not labels.species.isin(['cat','dog','mixed']).all():
        raise ValueError('Invalid species truth.')
    counts=pd.to_numeric(labels.n_animals, errors='coerce')
    if not np.isfinite(counts).all() or (counts < 1).any() or (counts % 1 != 0).any():
        raise ValueError('Animal count must be a positive integer.')
    paths={}
    for p in images.iterdir():
        if p.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        if p.stem in paths:
            raise ValueError('Duplicate image stem with different extensions.')
        paths[p.stem]=p
    if set(paths)!=set(labels.image.astype(str)):
        missing=sorted(set(labels.image.astype(str))-set(paths))
        extra=sorted(set(paths)-set(labels.image.astype(str)))
        raise ValueError(f'Cohort mismatch: missing={len(missing)},extra={len(extra)}')
    return paths


def summarize(df: pd.DataFrame, column: str) -> dict:
    ready=df[column].eq('ready')
    multi=df.n_animals.astype(int)>1
    single=~multi
    catdog=single & df.species.isin(['cat','dog'])
    m=int(multi.sum()); s=int(single.sum())
    out={
        'all_images':int(len(df)), 'all_referrals':int((~ready).sum()),
        'single_animal_images':s, 'single_animal_referrals':int((single & ~ready).sum()),
        'single_animal_referral_rate':float((single & ~ready).sum()/s) if s else None,
        'multi_animal_images':m, 'multi_animal_ready':int((multi & ready).sum()),
        'multi_animal_false_ready_rate':float((multi & ready).sum()/m) if m else None,
        'single_species_ready_correct':int((catdog & ready & df.pred.eq(df.species)).sum()),
        'single_species_ready_total':int((catdog & ready).sum()),
    }
    return out


def evaluate_paired(df: pd.DataFrame) -> dict:
    if df.image.duplicated().any():
        raise ValueError('Duplicate prediction rows.')
    before=summarize(df, 'baseline_verdict'); after=summarize(df,'candidate_verdict')
    unexpected=((df.baseline_verdict == 'refer_to_staff') & (df.candidate_verdict == 'ready')).sum()
    if unexpected:
        raise ValueError('Safety regression: review promoted previously-referred images to READY.')
    return {'baseline': before, 'candidate': after,
      'new_referrals':int(((df.baseline_verdict=='ready')&(df.candidate_verdict!='ready')).sum()),
      'multi_ready_delta':after['multi_animal_ready']-before['multi_animal_ready'],
      'single_referral_delta':after['single_animal_referrals']-before['single_animal_referrals'],
      'generalization_not_established': True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        ap.error('Output directory exists; do not overwrite prior evidence.')
    df=pd.read_csv(args.labels)
    paths=validate_cohort(args.images,df)
    from spv.pipeline import Manifest, PetPipeline
    m=Manifest.load(args.manifest)
    if m.review_mode!='enforce':
        ap.error('This paired evaluator requires review_mode=enforce.')
    model_paths={'detector':Path(m.detector), 'segmenter':Path(m.segmenter),
                 'classifier':Path(m.classifier)}
    if any(not p.is_file() for p in model_paths.values()):
        ap.error('Trained checkpoint files for all three models are required.')
    from spv.runtime import environment_record
    from PIL import Image
    model=PetPipeline(m)
    rows=[]
    t0=time.perf_counter()
    for item in df.sort_values('image').itertuples(index=False):
        with Image.open(paths[item.image]) as im:
            result=model.analyze(im,with_images=False)
        if not result['staff_approval_required'] or result['automatic_publication_allowed']:
            raise RuntimeError('Human-approval invariant failed.')
        codes=[x['code'] for x in result['reasons']]
        new_codes={'secondary_animal_candidate','secondary_review_failed'}
        baseline='refer_to_staff' if any(x not in new_codes for x in codes) else 'ready'
        rows.append({'image':item.image, 'species':item.species,'n_animals':int(item.n_animals),
                     'pred':result['species'],'baseline_verdict':baseline,
                     'candidate_verdict':result['verdict'],'reason_codes':';'.join(codes),
                     'review_status':result['secondary_review']['status'],
                     'secondary_candidates':result['secondary_review']['candidate_count'],
                     'secondary_review_ms':result['timings'].get('secondary_review_ms',0.),
                     'total_ms':result['timings']['total_ms']})
    out=pd.DataFrame(rows)
    if set(out.image)!=set(df.image) or len(out)!=len(df):
        raise RuntimeError('Prediction cohort not complete.')
    metrics=evaluate_paired(out)
    metrics.update({'dataset':'previously-inspected team set (regression only)',
                    'independent_holdout':False,'thresholds_selected_using_this_set':False,
                    'model_sha256':{k:sha(v) for k,v in model_paths.items()},
                    'manifest_sha256':sha(args.manifest),'labels_sha256':sha(args.labels),
                    'source_sha256':{'pipeline.py':sha(ROOT/'src/spv/pipeline.py'),
                                     'multi_animal_review.py':sha(ROOT/'src/spv/multi_animal_review.py')},
                    'evaluation_runtime_s':time.perf_counter()-t0,
                    'environment':environment_record()})
    args.output.mkdir(parents=True)
    out.to_csv(args.output/'per_image_regression.csv',index=False)
    (args.output/'paired_summary.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
