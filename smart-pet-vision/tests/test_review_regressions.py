"""CPU regressions; real model/GPU integration is intentionally separate."""
import asyncio
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastapi import HTTPException
from spv import serve, deploy
from spv.segment import mask_metrics

class MaskEvaluation(unittest.TestCase):
    def test_known_overlap_and_ignore(self):
        p = np.array([[1, 1], [0, 0]], bool)
        g = np.array([[1, 0], [1, 0]], bool)
        self.assertAlmostEqual(mask_metrics(p,g)['iou'],1/3)
        self.assertAlmostEqual(mask_metrics(p,g)['dice'],.5)
        self.assertEqual(mask_metrics(p,g,np.array([[0,1],[1,0]],bool))['iou'],1)
    def test_invalid_alignment_cannot_broadcast(self):
        with self.assertRaises(ValueError): mask_metrics(np.ones((2,1)), np.ones((2,2)))
    def test_ignored_image_cannot_score_perfect(self):
        with self.assertRaises(ValueError): mask_metrics(np.ones((2,2)),np.ones((2,2)),np.ones((2,2)))
    def test_ignore_alignment(self):
        with self.assertRaises(ValueError): mask_metrics(np.ones((2,2)),np.ones((2,2)),np.ones((2,1)))

class Upload:
    def __init__(self,data): self.data=data; self.requested=None
    async def read(self,size=-1):
        self.requested=size
        return self.data if size<0 else self.data[:size]

class ServiceInputs(unittest.TestCase):
    def test_upload_limit_is_enforced_before_decode(self):
        f=Upload(b'x'*17)
        with patch.object(serve,'MAX_BYTES',16), patch.object(serve,'pipeline') as model:
            with self.assertRaises(HTTPException) as e: asyncio.run(serve.analyze(f,False))
            self.assertEqual(e.exception.status_code,413)
            self.assertEqual(f.requested,17)
            model.assert_not_called()
    def test_empty_and_unreadable(self):
        for data,status in [(b'',400),(b'not an image',415)]:
            with self.assertRaises(HTTPException) as e: asyncio.run(serve.analyze(Upload(data),False))
            self.assertEqual(e.exception.status_code,status)
    def test_valid_image_reaches_pipeline(self):
        b=io.BytesIO();Image.new('RGB',(8,8)).save(b,format='PNG')
        with patch.object(serve,'pipeline') as model:
            model.return_value.analyze.return_value={'verdict':'ready','automatic_publication_allowed':False}
            r=asyncio.run(serve.analyze(Upload(b.getvalue()),False))
            self.assertFalse(r['automatic_publication_allowed'])
            model.return_value.analyze.assert_called_once()

class BenchmarkInputs(unittest.TestCase):
    def test_invalid_run_configuration(self):
        with self.assertRaises(ValueError):deploy.benchmark(None,[])
        with self.assertRaises(ValueError):deploy.benchmark(None,[Path('missing')],runs=0)
    def test_rss_units_are_portable(self):
        class Usage: ru_maxrss=1024*1024
        with patch.object(deploy.resource,'getrusage',return_value=Usage()):
            with patch.object(deploy.sys,'platform','darwin'):self.assertEqual(deploy._peak_rss_mb(),1)
            with patch.object(deploy.sys,'platform','linux'):self.assertEqual(deploy._peak_rss_mb(),1024)

if __name__=='__main__':unittest.main()
