"""Synthetic decision/regression tests only; no trained-weight performance claims."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from spv.pipeline import Manifest, PetPipeline, SegmentationEvidence
from spv.multi_animal_review import (ReviewProposal, additional_animal_cues,
                                     map_tile_boxes, review_tiles)


class TensorStub:
    def __init__(self, arr): self.value = np.asarray(arr)
    def cpu(self): return self
    def numpy(self): return self.value


def result(boxes, scores, size=(100, 100)):
    return SimpleNamespace(orig_shape=(size[1], size[0]),
                           boxes=SimpleNamespace(xyxy=TensorStub(boxes), conf=TensorStub(scores)))


class ProposalGeometry(unittest.TestCase):
    def test_duplicate_from_second_model_is_not_extra_animal(self):
        cues = additional_animal_cues([[10, 10, 50, 50]],
            [ReviewProposal('high_res', np.array([[11, 11, 49, 49]]), np.array([.99]))],
            confidence_floor=.1, same_instance_iou=.55, image_size=(100, 100))
        self.assertEqual(cues, [])

    def test_clear_second_animal_is_cue_independent_of_species(self):
        p = [ReviewProposal('high_res', np.array([[10, 10, 50, 50], [65, 60, 95, 90]]), np.array([.98,.19]))]
        cues = additional_animal_cues([[10, 10, 50, 50]], p, confidence_floor=.1,
            same_instance_iou=.55, image_size=(100,100))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]['sources'], ['high_res'])

    def test_secondary_dedup_but_cross_source_evidence_preserved(self):
        p = [ReviewProposal('high_res', np.array([[60, 60, 88, 88]]), np.array([.3])),
             ReviewProposal('segmenter', np.array([[61, 61, 87, 87]]), np.array([.5]))]
        cues = additional_animal_cues([[10,10,50,50]],p,confidence_floor=.1,
            same_instance_iou=.55,image_size=(100,100))
        self.assertEqual(len(cues),1)
        self.assertEqual(set(cues[0]['sources']), {'segmenter','high_res'})

    def test_confidence_floor(self):
        cues=additional_animal_cues([[10,10,50,50]],
            [ReviewProposal('high_res',np.array([[60,60,90,90]]),np.array([.09]))],
            confidence_floor=.1,same_instance_iou=.55,image_size=(100,100))
        self.assertEqual(cues,[])

    def test_invalid_coordinate_and_scores_are_fail_closed(self):
        for box,score in [([[-1,0,2,2]],[.2]), ([[1,1,999,999]],[.2]),
                          ([[1,1,2,2]],[float('nan')]), ([[1,1,1,2]],[.2])]:
            with self.subTest(box=box,score=score):
                with self.assertRaises(ValueError):
                    additional_animal_cues([[10,10,50,50]],
                        [ReviewProposal('high_res',np.asarray(box),np.asarray(score))],
                        confidence_floor=.1,same_instance_iou=.55,image_size=(100,100))

    def test_proposal_overflow_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'Excessive'):
            additional_animal_cues([[10,10,50,50]],
               [ReviewProposal('many',np.tile([[60,60,70,70]],(301,1)),np.ones(301))],
               confidence_floor=.1,same_instance_iou=.55,image_size=(100,100))

    def test_tiled_coordinates_and_overlap_are_deterministic(self):
        tiles=review_tiles(1000,800)
        self.assertEqual(len(tiles),4)
        tile=tiles[-1]
        mapped=map_tile_boxes([[0,0,20,30]],tile,(1000,800))
        self.assertEqual(mapped[0,0],tile[0])
        self.assertEqual(mapped[0,1],tile[1])
        self.assertEqual(review_tiles(50,50),[])


class ReviewFlow(unittest.TestCase):
    def setUp(self):
        self.pipe=PetPipeline.__new__(PetPipeline)
        self.pipe.m=SimpleNamespace(review_mode='enforce', review_conf=.1,
                review_imgsz=960, review_match_iou=.55, review_include_segmenter=True,
                review_tiles_enabled=False, review_tiles_conf=.1, imgsz=640,
                detector_iou=.5, verify_threshold=.9, min_mask_fraction=.05)
        self.pipe.detect=MagicMock(return_value=(np.array([[10,10,50,50]],float),
                                                 np.array([0]),np.array([.99])))
        self.pipe.classify=MagicMock(return_value=np.array([.99,.01]))
        mask=np.zeros((100,100),bool);mask[10:50,10:50]=1
        self.pipe.segment=MagicMock(return_value=SegmentationEvidence(mask,True,1,'cat'))
        self.pipe.detector=MagicMock()
        self.pipe.segmenter=MagicMock()
        self.pipe.detector.predict.return_value=[result([[10,10,50,50]], [.99])]
        self.pipe.segmenter.predict.return_value=[result([[10,10,50,50]], [.99])]
        self.image=Image.new('RGB',(100,100))

    def test_single_animal_remains_ready(self):
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'ready')
        self.assertEqual(out['secondary_review']['candidate_count'],0)
        self.assertFalse(out['automatic_publication_allowed'])
        self.assertTrue(out['staff_approval_required'])

    def test_detector_proposal_refers_without_changing_species(self):
        self.pipe.detector.predict.return_value=[result([[10,10,50,50],[60,60,90,90]], [.99,.2])]
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertEqual(out['species'],'cat')
        self.assertIn('secondary_animal_candidate',[x['code'] for x in out['reasons']])
        self.assertEqual(len(out['detections']),1)

    def test_segmenter_only_secondary_proposal_refers(self):
        self.pipe.segmenter.predict.return_value=[result([[12,12,50,50],[65,60,95,90]],[.99,.2])]
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertEqual(out['secondary_review']['candidate_count'],1)

    def test_audit_mode_records_but_does_not_change_verdict(self):
        self.pipe.m.review_mode='audit'
        self.pipe.detector.predict.return_value=[result([[10,10,50,50],[60,60,90,90]],[.99,.2])]
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'ready')
        self.assertEqual(out['secondary_review']['candidate_count'],1)

    def test_probe_failure_refers_conservatively_in_enforce(self):
        self.pipe.detector.predict.side_effect=RuntimeError('unsafe low-level details')
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertEqual(out['secondary_review']['error_type'],'RuntimeError')
        self.assertNotIn('unsafe low-level details',str(out))

    def test_audit_failure_reported_without_changing_decision(self):
        self.pipe.m.review_mode='audit'
        self.pipe.detector.predict.side_effect=RuntimeError('offline')
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'ready')
        self.assertEqual(out['secondary_review']['status'],'failed')

    def test_no_repeat_inference_when_previously_referred(self):
        self.pipe.detect.return_value=(np.array([[10,10,50,50],[60,60,90,90]]),
                                      np.array([0,0]),np.array([.99,.7]))
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['verdict'],'refer_to_staff')
        self.assertEqual(out['secondary_review']['status'],'skipped_already_referred')
        self.pipe.detector.predict.assert_not_called()

    def test_tile_secondary_is_translated_and_referred(self):
        self.pipe.m.review_tiles_enabled=True
        self.pipe.m.review_include_segmenter=False
        # Use 200x200 pixels so the deterministic 2x2 tile policy is active.
        self.image=Image.new('RGB',(200,200))
        mask=np.zeros((200,200),bool);mask[10:60,10:60]=1
        self.pipe.segment.return_value=SegmentationEvidence(mask,True,1,'cat')
        primary=result([[10,10,50,50]], [.99],size=(200,200))
        tiled=result([[42,42,59,59]],[.25],size=(130,130))
        self.pipe.detector.predict.side_effect=[[primary]]+[[result([],[],size=(130,130))] for _ in range(3)]+[[tiled]]
        out=self.pipe.analyze(self.image,False)
        self.assertEqual(out['secondary_review']['pass_count'],5)
        self.assertEqual(out['verdict'],'refer_to_staff')


class ReviewManifest(unittest.TestCase):
    def test_candidate_load_and_invalid_controls(self):
        base=dict(detector='runs/d.pt',segmenter='runs/s.pt',classifier='runs/c.pt',
            classifier_kind='finetune',temperature=.05,detector_conf=.4,
            detector_iou=.5,verify_threshold=.9,min_mask_fraction=.05)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'runs'/'manifest.json';f.parent.mkdir()
            f.write_text(json.dumps({**base,'review_mode':'enforce'}))
            self.assertEqual(Manifest.load(f).review_mode,'enforce')
            for bad in [{'review_mode':'pass'}, {'review_conf':.7},
                        {'review_imgsz':320},{'review_tiles_enabled':'yes'}]:
                f.write_text(json.dumps({**base,**bad}))
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):Manifest.load(f)


if __name__=='__main__': unittest.main()
