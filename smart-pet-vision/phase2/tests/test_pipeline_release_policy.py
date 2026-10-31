"""CPU-only release metadata regression tests. Requires project Python deps (except trained weights)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spv.pipeline import PetPipeline

class SafetyContract(unittest.TestCase):
    def setUp(self):
        self.p=PetPipeline.__new__(PetPipeline)
        self.p.m=MagicMock(verify_threshold=.9,min_mask_fraction=.05)
        self.p.detect=MagicMock(return_value=(np.array([[5,5,45,45]],dtype=float),np.array([0]),np.array([.99])))
        self.p.classify=MagicMock(return_value=np.array([.99,.01]))
        self.p.segment=MagicMock(return_value=(np.ones((50,50),dtype=bool), True))
        self.img=Image.new('RGB',(50,50))
    def test_ready_never_auto_publishes(self):
        r=self.p.analyze(self.img,with_images=False)
        self.assertEqual(r['verdict'],'ready')
        self.assertTrue(r['staff_approval_required'])
        self.assertFalse(r['automatic_publication_allowed'])
    def test_referred_never_auto_publishes(self):
        self.p.detect.return_value=(np.zeros((0,4)),np.zeros((0,),dtype=int),np.zeros(0))
        r=self.p.analyze(self.img,with_images=False)
        self.assertEqual(r['verdict'],'refer_to_staff')
        self.assertTrue(r['staff_approval_required'])
        self.assertFalse(r['automatic_publication_allowed'])
    def test_multi_detected_refers(self):
        self.p.detect.return_value=(np.array([[5,5,30,30],[9,8,40,40]],dtype=float),np.array([0,0]),np.array([.9,.7]))
        r=self.p.analyze(self.img,with_images=False)
        self.assertEqual(r['verdict'],'refer_to_staff')
        self.assertTrue(any(x['code']=='multiple_animals' for x in r['reasons']))
        self.assertFalse(r['automatic_publication_allowed'])

if __name__=='__main__': unittest.main()
