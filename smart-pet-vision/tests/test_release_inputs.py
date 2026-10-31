import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spv.pipeline import Manifest
from spv.instance_alignment import best_mask_index
import importlib.util

class ReleaseInputs(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; classifier construction not executed")
    def test_invalid_classifier_is_rejected(self):
        from spv.classify import build_model
        with self.assertRaises(ValueError):build_model('not-a-model',{})
    def manifest(self,root,**extra):
        d=dict(detector='runs/d.pt',segmenter='runs/s.pt',classifier='runs/c.pt',classifier_kind='finetune',temperature=.05,detector_conf=.4,detector_iou=.5,verify_threshold=.9,min_mask_fraction=.05);d.update(extra)
        p=root/'runs/manifest.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(d));return p
    def test_manifest_independent_of_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();m=Manifest.load(self.manifest(root));self.assertEqual(m.detector,str(root/'runs/d.pt'))
    def test_bad_temperature_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for params in [dict(temperature=0),dict(temperature=float('nan')),dict(detector_conf=1.2)]:
                with self.assertRaises(ValueError):Manifest.load(self.manifest(root,**params))
    def test_invalid_boxes_do_not_match(self):
        self.assertIsNone(best_mask_index([[0,0,10,10]],[0,0,float('nan'),10]))
        self.assertIsNone(best_mask_index([[10,10,0,0]],[0,0,10,10]))
    def test_class_count_and_iou_validation(self):
        with self.assertRaises(ValueError):best_mask_index([[0,0,10,10]],[0,0,10,10],[],0)
        with self.assertRaises(ValueError):best_mask_index([[0,0,10,10]],[0,0,10,10],min_iou=0)

if __name__=='__main__':unittest.main()
