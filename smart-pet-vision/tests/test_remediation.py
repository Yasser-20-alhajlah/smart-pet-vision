import importlib.util
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np,pandas as pd
from PIL import Image
from spv import data,common,classify

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('animal_groups',ROOT/'tools/review_animal_groups.py')
groups=importlib.util.module_from_spec(spec);spec.loader.exec_module(groups)

class Remediation(unittest.TestCase):
    def test_real_mask_area_is_not_box_area(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'annotations/trimaps';folder.mkdir(parents=True)
            Image.fromarray(np.array([[1,1,2],[1,2,2],[2,2,2]],np.uint8)).save(folder/'x.png')
            out=data.build_annotations(root,pd.DataFrame([dict(image='x',width=3,height=3)]),root/'processed')
            self.assertAlmostEqual(out.mask_fraction.iloc[0],3/9);self.assertAlmostEqual(out.box_fraction.iloc[0],4/9)

    def test_missing_single_yolo_label_or_border_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=root/'raw/oxford-iiit-pet';(raw/'annotations').mkdir(parents=True);(raw/'annotations/list.txt').touch()
            processed=root/'processed';saved=root/'runs/data';saved.mkdir(parents=True)
            pd.DataFrame([dict(image='pet',split='train')]).to_csv(saved/'metadata.csv',index=False)
            for task in ['detect','segment']:
                f=processed/f'yolo_{task}';f.mkdir(parents=True);(f/'data.yaml').touch()
                l=f/'labels/train';l.mkdir(parents=True);(l/'pet.txt').touch()
            for name in ['masks','borders']:
                f=processed/name;f.mkdir();(f/'pet.png').touch()
            for missing in [processed/'yolo_detect/labels/train/pet.txt',processed/'borders/pet.png']:
                missing.unlink()
                with patch.object(data,'build_annotations') as build,patch.object(data,'export_yolo') as export:
                    data.ensure_data({'paths':{'raw':root/'raw','processed':processed,'runs':root/'runs'}})
                    self.assertEqual(build.call_count,1);self.assertEqual(export.call_count,2)
                missing.touch()

    def test_source_fingerprint_covers_tools(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'tools').mkdir();p=root/'tools/run.py';p.write_text('x=1')
            a=common.source_provenance(root);p.write_text('x=2');b=common.source_provenance(root)
            self.assertNotEqual(a['source_sha256'],b['source_sha256']);self.assertIsNone(a['git_commit'])

    def test_wandb_entity_and_source_are_forwarded_without_mutating_config(self):
        import sys,types
        calls=[];fake=types.SimpleNamespace(init=lambda **kw:calls.append(kw))
        config={'seed':42}
        with patch.dict(sys.modules,{'wandb':fake}):
            common.Tracker({'wandb':{'enabled':True,'project':'test','entity':'authorized-team'}},'synthetic',config)
        self.assertEqual(calls[0]['entity'],'authorized-team');self.assertIn('source_provenance',calls[0]['config']);self.assertEqual(config,{'seed':42})

    def test_nonfinite_calibration_fails_before_optimization(self):
        for logits,labels in [(np.array([[np.nan,0]]),[0]),(np.ones((2,2)),[0]),(np.ones((1,2)),[2])]:
            with self.assertRaises(ValueError):classify.fit_temperature(logits,np.array(labels))

    def test_all_pair_proposals_include_original_same_partition(self):
        meta=pd.DataFrame({'image':['a','b','c']})
        out=groups.propose(meta,np.array([[1,0],[1,0],[1,0]]),.99)
        self.assertEqual(set(zip(out.image_a,out.image_b)),{('a','b'),('a','c'),('b','c')})

    def test_pending_or_incomplete_identity_review_cannot_create_split(self):
        meta=pd.DataFrame(dict(image=['a','b'],group=[0,1],md5=['a','b'],species=['cat','cat'],breed=['x','x'],split=['train','test'],label=[0,0]))
        proposals=pd.DataFrame([dict(image_a='a',image_b='b',cosine=.99,decision='pending',reviewer='',evidence='')])
        with self.assertRaisesRegex(ValueError,'Pending'):groups.reviewed_split(meta,proposals,proposals)
        with self.assertRaisesRegex(ValueError,'coverage'):groups.reviewed_split(meta,proposals,proposals.iloc[:0])

    def test_contradictory_transitive_identity_decisions_fail(self):
        meta=pd.DataFrame(dict(image=['a','b','c'],group=[0,1,2],md5=['a','b','c'],species=['cat']*3,breed=['x']*3,split=['train','val','test'],label=[0]*3))
        reviews=pd.DataFrame([dict(image_a=a,image_b=b,decision=d,reviewer='human',evidence='review note') for a,b,d in [('a','b','same'),('b','c','same'),('a','c','different')]])
        with self.assertRaisesRegex(ValueError,'Transitive'):groups.reviewed_split(meta,reviews,reviews)

if __name__=='__main__':unittest.main()
