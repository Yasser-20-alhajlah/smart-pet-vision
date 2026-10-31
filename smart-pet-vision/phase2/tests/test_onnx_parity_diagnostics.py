"""Synthetic tests of diagnostics, hash barriers, and frozen cohort handling.

These do NOT certify trained-weight parity and are not counted as M8 real-model tests.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from onnx_parity_diagnostics import (TOLERANCE, sha256_file, tensor_diagnostics,
                                     verify_frozen_snapshot, verify_cohort_manifest,
                                     generate_cohort_manifest)
from run_onnx_stage2 import parse_args, run


class ParityDiagnostics(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((1,6,8), np.float32)

    def test_per_channel_box_vs_class_error(self):
        b = self.a.copy(); b[0, 2, 3] = .002; b[0, 5, 5] = .0001
        r = tensor_diagnostics(self.a, b)
        self.assertFalse(r['raw_passed'])
        self.assertAlmostEqual(r['box_max_abs'], .002, places=6)
        self.assertAlmostEqual(r['class_max_abs'], .0001, places=6)
        self.assertEqual(r['channels'][2]['role'], 'box_width')
        self.assertEqual(r['overall']['elements_above_tolerance'], 1)

    def test_pass_and_same_size(self):
        b = self.a.copy(); b[0,5,0]=.0009
        self.assertTrue(tensor_diagnostics(self.a, b)['raw_passed'])

    def test_reject_nonfinite_or_bad_shapes(self):
        for b in (np.zeros((1,6,7), np.float32), np.full_like(self.a, np.nan)):
            with self.assertRaises(ValueError):tensor_diagnostics(self.a,b)

    def test_original_tolerance_unchanged(self):
        with self.assertRaises(ValueError):tensor_diagnostics(self.a, self.a, tolerance=.005)

    def test_frozen_source_blocks_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'a.py';p.write_text('v1')
            baseline={'a.py':sha256_file(p)}
            self.assertTrue(verify_frozen_snapshot(root, baseline)['passed'])
            p.write_text('v2')
            self.assertFalse(verify_frozen_snapshot(root, baseline)['passed'])

    def test_cohort_rejects_duplicates_and_changed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'a.jpg'; Image.new('RGB',(8,8)).save(p)
            manifest=root/'cohort.json'; entry={'file':'a.jpg','sha256':sha256_file(p)}
            manifest.write_text(json.dumps([entry,entry]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                verify_cohort_manifest(root, manifest, min_images=2)
            manifest.write_text(json.dumps([entry]));self.assertEqual(len(verify_cohort_manifest(root,manifest,1)),1)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Missing/modified'):
                verify_cohort_manifest(root,manifest,1)

    def test_cohort_refuses_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(50):Image.new('RGB',(2,2)).save(root/f'{i:02}.png')
            out=root/'cohort.json';rows=generate_cohort_manifest(root,out,50)
            self.assertEqual(len(rows), 50)
            with self.assertRaises(FileExistsError):generate_cohort_manifest(root,out,50)

    def test_gate_fails_closed_without_dependencies_without_mutating_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';repo.mkdir()
            source=repo/'a.py';source.write_text('unchanged')
            weight=root/'model.pt';weight.write_bytes(b'trained-pt-original')
            onnx=root/'model.onnx';onnx.write_bytes(b'onnx-original')
            images=root/'images';images.mkdir()
            for i in range(50):Image.new('RGB',(2,2)).save(images/f'{i:02}.png')
            cohort=root/'cohort.json';generate_cohort_manifest(images,cohort,50)
            baseline=root/'approved.json';baseline.write_text(json.dumps({
              'source_snapshot':{'a.py':sha256_file(source)},
              'weights':{'best.pt':{'sha256':sha256_file(weight)},
                         'best.onnx.bin':{'sha256':sha256_file(onnx)}}}))
            # Missing model dependencies or invalid binary must block, never be marked PASS.
            out=root/'result';args=['--repo',str(repo),'--weights',str(weight),'--onnx',str(onnx),
                '--image-root',str(images),'--cohort',str(cohort),'--baseline',str(baseline),
                '--output',str(out)]
            self.assertEqual(run(args),2)
            record=json.loads((out/'onnx_stage2_gate.json').read_text())
            self.assertEqual(record['status'],'BLOCKED')
            self.assertTrue(record['source_snapshot']['passed'])
            self.assertEqual(record['weights_sha256'], sha256_file(weight))
            self.assertEqual(record['onnx_sha256'], sha256_file(onnx))
            self.assertFalse(record['onnx_deployment_allowed'])
            self.assertEqual(weight.read_bytes(),b'trained-pt-original')
            with self.assertRaises(SystemExit):parse_args(args)


if __name__ == '__main__':unittest.main()
