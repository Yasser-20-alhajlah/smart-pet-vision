"""Standalone instance association checks using synthetic boxes and masks.

These tests do NOT imply GPU integration or improved real-image accuracy.
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spv.instance_alignment import best_mask_index
from spv.pipeline import PetPipeline

class TensorStub:
    def __init__(self, value): self.value=np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value

class InstanceAlignment(unittest.TestCase):
    def setUp(self):
        self.p = PetPipeline.__new__(PetPipeline)
        self.p.m = MagicMock(imgsz=640, verify_threshold=.9,min_mask_fraction=.05, segmenter_conf=.25, segmenter_iou=.7, mask_match_iou=.1)
        self.p.detect=MagicMock(return_value=(np.array([[2,2,18,18.]],dtype=float),np.array([0]),np.array([.99])))
        self.p.classify=MagicMock(return_value=np.array([.99,.01]))
        self.img=Image.new('RGB',(32,32))
        self.m0=np.zeros((32,32),bool);self.m0[2:18,2:18]=True
        self.m1=np.zeros((32,32),bool);self.m1[18:31,18:31]=True
        self.set_segmenter(np.stack([self.m0,self.m1]),np.array([[2,2,18,18],[18,18,31,31]]),np.array([0,1]))
    def set_segmenter(self,masks,boxes,classes):
        r=SimpleNamespace(orig_shape=(32,32),masks=SimpleNamespace(data=TensorStub(masks)),boxes=SimpleNamespace(xyxy=TensorStub(boxes),cls=TensorStub(classes)))
        self.p.segmenter=MagicMock();self.p.segmenter.predict.return_value=[r]
    def test_selects_correct_instance_not_union(self):
        out=self.p.analyze(self.img,with_images=False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertAlmostEqual(out['mask_fraction'], self.m0.mean())
        self.assertTrue(out['segmentation_matched'])
        self.assertTrue(out['staff_approval_required'])
        self.assertFalse(out['automatic_publication_allowed'])
        self.assertEqual(out['segmentation_animal_count'], 2)
        self.assertIn('multiple_animals', [r['code'] for r in out['reasons']])
    def test_single_matched_instance_is_provisionally_ready(self):
        self.set_segmenter(np.stack([self.m0]),np.array([[2,2,18,18]]),np.array([0]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertEqual(out['verdict'],'ready')
        self.assertEqual(out['segmentation_animal_count'],1)
        self.assertFalse(self.p.segmenter.predict.call_args.kwargs['rect'])
    def test_same_species_second_animal_is_also_referred(self):
        self.set_segmenter(np.stack([self.m0,self.m1]),np.array([[2,2,18,18],[18,18,31,31]]),np.array([0,0]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertIn('multiple_animals',[r['code'] for r in out['reasons']])
    def test_matched_mask_species_conflict_is_referred(self):
        self.set_segmenter(np.stack([self.m0]),np.array([[2,2,18,18]]),np.array([1]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertIn('segmentation_disagreement',[r['code'] for r in out['reasons']])
    def test_mask_and_box_instance_count_mismatch_is_referred(self):
        self.set_segmenter(np.stack([self.m0]),np.array([[2,2,18,18],[18,18,31,31]]),np.array([0,1]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertIn('segmentation_unmatched',[r['code'] for r in out['reasons']])
    def test_missing_instance_referred(self):
        self.set_segmenter(np.stack([self.m1]),np.array([[18,18,31,31]]),np.array([1]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertIn('segmentation_unmatched',[r['code'] for r in out['reasons']])
    def test_mismatched_mask_shape_referred(self):
        self.set_segmenter(np.ones((1,20,20),bool),np.array([[2,2,18,18]]),np.array([0]))
        out=self.p.analyze(self.img,with_images=False)
        self.assertEqual(out['verdict'],'refer_to_staff')
    def test_no_animal_returns_no_animal(self):
        self.p.detect.return_value=(np.empty((0,4)),np.empty((0,),int),np.empty((0,)))
        out=self.p.analyze(self.img,with_images=False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertIn('no_animal',[r['code'] for r in out['reasons']])
    def test_same_class_preferred_with_overlap(self):
        idx=best_mask_index(np.array([[0,0,18,18],[0,0,20,20]]),np.array([0,0,20,20]),np.array([0,1]),0)
        self.assertEqual(idx,0)
    def test_fallback_different_class_when_only_overlap(self):
        idx=best_mask_index(np.array([[0,0,20,20]]),np.array([0,0,20,20]),np.array([1]),0)
        self.assertEqual(idx,0)
    def test_nonoverlap_none(self):
        idx=best_mask_index(np.array([[40,40,50,50]]),np.array([0,0,20,20]))
        self.assertIsNone(idx)
    def test_empty_masks_none(self):
        self.assertIsNone(best_mask_index(np.empty((0,4)),np.array([0,0,20,20])))

if __name__=='__main__': unittest.main()
