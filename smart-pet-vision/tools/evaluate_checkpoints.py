"""Fresh inference verification; preserves original runs and exports sample-level evidence.
Run only after restoring submitted weights and original JPEGs. No training or model selection.
"""
from pathlib import Path
import argparse,hashlib,json,platform,sys,time,importlib.metadata,subprocess
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spv import classify
from spv.common import load_config

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--images',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');a.add_argument('--batch-size',type=int,default=32);args=a.parse_args()
 root=Path(__file__).resolve().parents[1];out=args.output.resolve();runs=(root/'runs').resolve()
 if out==runs/'classification' or (runs/'classification').is_relative_to(out) or out.is_relative_to(runs/'classification'):raise ValueError('Use a separate verification output directory')
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new or empty')
 out.mkdir(parents=True,exist_ok=True);cfg=load_config();cfg['classification']['num_workers']=0;cfg['classification']['batch_size']=args.batch_size;cfg['wandb']['enabled']=False
 meta=pd.read_csv(runs/'data/metadata.csv');sub=meta[meta.split.isin(['val','test'])]
 for r in sub.itertuples():
  p=args.images/(r.image+'.jpg')
  if hashlib.md5(p.read_bytes()).hexdigest()!=r.md5:raise ValueError('Image fingerprint mismatch: '+r.image)
 dev=torch.device(args.device);classify.device=lambda:dev
 data=classify.loaders(meta,args.images,cfg);environment={'evaluator_sha256':sha(__file__),'source_sha256':{k:sha(root/'src/spv'/k)for k in ['classify.py','common.py']},'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'device':str(dev),'inference_precision':'float32','metadata_sha256':sha(runs/'data/metadata.csv'),'config_sha256':sha(root/'configs/config.yaml'),'kind':'fresh-checkpoint-inference; not training reproduction','model_selection':'none; all three submitted checkpoints preselected','packages':{k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','pandas','pillow','scikit-learn']},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};results={}
 for kind in ['scratch','frozen','finetune']:
  print('START',kind,flush=True);model=classify.load_model(kind,cfg,runs/'classification'/f'{kind}.pt');lg,y=classify.predict_logits(model,data['test']);saved=json.loads((runs/'classification'/f'{kind}_test_metrics.json').read_text());t=saved['temperature'];p=classify.softmax(lg);pc=classify.softmax(lg,t);tm=meta[meta.split=='test'].reset_index(drop=True)
  np.savez_compressed(out/f'{kind}_test_logits.npz',logits=lg,labels=y,images=tm.image.to_numpy(dtype=str))
  raw=classify.evaluate(p,y,tm);cal=classify.evaluate(pc,y,tm)
  pd.DataFrame({'image':tm.image,'label':y,'pred':p.argmax(1),'logit_cat':lg[:,0],'logit_dog':lg[:,1],'p_cat_raw':p[:,0],'p_dog_raw':p[:,1],'p_cat_calibrated':pc[:,0],'p_dog_calibrated':pc[:,1]}).to_csv(out/f'{kind}_predictions.csv',index=False)
  vl,vy=classify.predict_logits(model,data['val']);np.savez_compressed(out/f'{kind}_validation_logits.npz',logits=vl,labels=vy,images=meta[meta.split=='val'].image.to_numpy(dtype=str));vt=classify.fit_temperature(vl,vy)
  result={'checkpoint_sha256':sha(runs/'classification'/f'{kind}.pt'),'saved_temperature':t,'refitted_validation_temperature':vt,'raw':raw,'calibrated_using_saved_temperature':cal,'saved_comparison':{'raw_accuracy_delta':raw['accuracy']-saved['test_raw']['accuracy'],'calibrated_ece_delta':cal['ece']-saved['test_calibrated']['ece'],'confusion_matrix_matches':cal['confusion_matrix']==saved['test_calibrated']['confusion_matrix']},'prediction_agreement_with_original':'unverifiable: original per-image predictions were not saved'}
  (out/f'{kind}_metrics.json').write_text(json.dumps(result,indent=2));results[kind]=result;print('DONE',kind,result['saved_comparison'],flush=True);del model
  if args.device=='mps':torch.mps.empty_cache()
 environment['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(out/'environment.json').write_text(json.dumps(environment,indent=2));(out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
