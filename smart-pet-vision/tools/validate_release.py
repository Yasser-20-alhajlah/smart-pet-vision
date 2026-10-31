"""Portable real-weight CPU regression. Never trains or overwrites baseline runs."""
import argparse,json,sys,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def validate_label_coverage(paths, labels):
    """Reject any silent cohort reduction before inference or metric merging."""
    required = {'image', 'species'}
    if not required.issubset(labels.columns):
        raise ValueError('Labels require image and species columns.')
    ids = labels.image
    image_ids = [path.stem for path in paths]
    if (not image_ids or labels.empty or ids.isna().any() or ids.duplicated().any()
            or len(set(image_ids)) != len(image_ids)):
        raise ValueError('Image and label IDs must be nonempty, unique and complete.')
    if set(ids.astype(str)) != set(image_ids):
        raise ValueError('Image and label IDs must cover exactly the same cohort.')
    if not labels.species.isin(['cat', 'dog', 'mixed']).all():
        raise ValueError('Species labels must be cat, dog or mixed.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--http-url',help='Optional already-running service URL')
    parser.add_argument('--http-requests',type=int,default=300)
    parser.add_argument('--benchmark',action='store_true')
    a=parser.parse_args()
    if a.output.exists():parser.error('Choose a new output directory; existing results are not overwritten.')
    paths=sorted(a.images.glob('*.jpg'))
    if not paths or a.threads<1 or a.http_requests<1:parser.error('Need JPEG images, positive threads and request count.')
    import pandas as pd
    labels = None
    if a.labels:
        labels = pd.read_csv(a.labels)
        validate_label_coverage(paths, labels)
    # This command promises a CPU regression even on a machine with CUDA.
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,pandas as pd,torch
    from PIL import Image
    from spv.pipeline import PetPipeline,Manifest
    from spv.deploy import benchmark
    from spv.runtime import environment_record
    torch.set_num_threads(a.threads);a.output.mkdir(parents=True)
    pipeline=PetPipeline(Manifest.load(a.manifest));rows=[]
    for path in paths:
        start=time.perf_counter();r=pipeline.analyze(Image.open(path),False)
        assert r['staff_approval_required'] and not r['automatic_publication_allowed']
        rows.append(dict(image=path.stem,pred=r['species'],verdict=r['verdict'],confidence=r['confidence'],reasons=';'.join(x['code'] for x in r['reasons']),n_detected=len(r['detections']),n_segmented=r.get('segmentation_animal_count'),segmentation_species=r.get('segmentation_species'),mask_fraction=r['mask_fraction'],wall_ms=(time.perf_counter()-start)*1000))
    df=pd.DataFrame(rows);summary=dict(runtime=platform.platform(),environment=environment_record(),device='CPU',n=len(df),referrals=int(sum(df.verdict!='ready')),all_publication_guards_passed=True,independent_new_test=False)
    if labels is not None:
        labels=labels.rename(columns={'species':'true'})
        df=df.merge(labels,on='image',validate='one_to_one');v=df[df.true.isin(['cat','dog'])]
        summary.update(classification_n=len(v),accuracy=float((v.true==v.pred).mean()))
        if 'n_animals' in df:summary['multi_animal_ready']=int(sum((df.n_animals>1)&(df.verdict=='ready')))
    df.to_csv(a.output/'predictions.csv',index=False)
    if a.benchmark:summary['benchmark']=benchmark(pipeline,paths[:50],warmup=10,runs=200)
    if a.http_url:
        import requests
        times=[];succeeded=0
        for i in range(a.http_requests+5):
            path=paths[i%len(paths)];start=time.perf_counter()
            with path.open('rb') as f:r=requests.post(a.http_url.rstrip('/')+'/analyze?images=false',files={'file':(path.name,f,'image/jpeg')},timeout=120)
            if i>=5:
                times.append((time.perf_counter()-start)*1000)
                succeeded+=int(r.status_code==200 and r.json().get('staff_approval_required') is True and r.json().get('automatic_publication_allowed') is False)
        summary['http']={'requests':a.http_requests,'successful_with_policy':succeeded,'warmup':5,'p50_p95_p99_ms':[float(x) for x in np.percentile(times,[50,95,99])]}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
