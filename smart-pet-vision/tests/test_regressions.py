import ast,csv,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from spv import data,classify

def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
v=module('verify_predictions',ROOT/'tools/verify_prediction_exports.py');ev=module('evaluate_checkpoints',ROOT/'tools/evaluate_checkpoints.py')
class ReviewChecks(unittest.TestCase):
 def test_invalid_label_is_rejected(self):
  r={'label':'-1','p_cat_raw':'.8','p_dog_raw':'.2'}
  with self.assertRaises(ValueError):v.recompute([r],False)
 def test_empty_predictions_are_rejected(self):
  with self.assertRaises(ValueError):v.recompute([],False)
 def test_nan_probability_is_rejected(self):
  with self.assertRaises(ValueError):v.recompute([{'label':'0','p_cat_raw':'nan','p_dog_raw':'.2'}],False)
 def test_known_metrics(self):
  r=[{'label':'0','p_cat_raw':'.8','p_dog_raw':'.2'},{'label':'1','p_cat_raw':'.3','p_dog_raw':'.7'}];a=v.recompute(r,False)
  self.assertEqual(a['confusion_matrix'],[[1,0],[0,1]]);self.assertAlmostEqual(a['ece'],.25);self.assertEqual(a['macro_f1'],1)
 @unittest.skipUnless((ROOT/"runs/verification_2026-10-28").is_dir(), "restore release evidence first")
 def test_label_alignment_is_checked(self):
  import shutil
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)
   for k in ['scratch','frozen','finetune']:
    for suffix in ['predictions.csv','metrics.json']:shutil.copy2(ROOT/'runs/verification_2026-10-28'/f'{k}_{suffix}',d/f'{k}_{suffix}')
   p=d/'frozen_predictions.csv'
   with p.open() as stream:rows=list(csv.DictReader(stream))
   rows[0]['label']=str(1-int(rows[0]['label']))
   with p.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
   with self.assertRaisesRegex(ValueError,'order or labels'):v.verify(d)
 def test_model_typo_is_rejected(self):
  with self.assertRaises(ValueError):classify.build_model('finetun',{})
 def test_scratch_parameter_count(self):self.assertEqual(classify.count_params(classify.ScratchCNN())['total'],2354402)
 def test_zip_source_without_git(self):
  with tempfile.TemporaryDirectory() as t:self.assertIn('unavailable',ev.source_revision(Path(t)))
 def test_missing_detector_export_is_rebuilt(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);raw=p/'raw/oxford-iiit-pet';(raw/'annotations').mkdir(parents=True);(raw/'annotations/list.txt').touch();out=p/'processed';(out/'yolo_segment').mkdir(parents=True);(out/'yolo_segment/data.yaml').touch();(out/'masks').mkdir();(out/'masks/pet.png').touch();saved=p/'runs/data';saved.mkdir(parents=True);pd.DataFrame({'image':['pet'],'split':['train']}).to_csv(saved/'metadata.csv',index=False)
   cfg={'paths':{'raw':p/'raw','processed':out,'runs':p/'runs'}}
   with patch.object(data,'build_annotations') as build,patch.object(data,'export_yolo') as export:
    data.ensure_data(cfg);self.assertEqual(build.call_count,1);self.assertEqual(export.call_count,2)
 def test_half_open_body_box(self):self.assertEqual(data.mask_to_box(np.array([[0,1,0],[0,1,0]])),(1,0,2,2))
 def test_trimap_border_protocol(self):
  body,ignore=data.trimap_to_masks(np.array([[1,2,3]]));self.assertEqual(body.tolist(),[[True,False,True]]);self.assertEqual(ignore.tolist(),[[False,False,True]])
 @unittest.skipUnless((ROOT/"runs/data/metadata.csv").is_file(), "restore release evidence first")
 def test_actual_split_and_crossings(self):
  df=pd.read_csv(ROOT/'runs/data/metadata.csv');self.assertEqual(len(df),7335);self.assertEqual(df.split.value_counts().to_dict(),{'train':5123,'val':1106,'test':1106});self.assertTrue(data.leakage_audit(df)['passed']);self.assertEqual(df.groupby('group').ngroups,7292)

if __name__ == '__main__':
 unittest.main(verbosity=2)
