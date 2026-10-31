import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('gallery_redaction', Path(__file__).resolve().parents[1]/'tools/redact_evidence_gallery.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class EvidenceRedaction(unittest.TestCase):
    def test_only_requested_pixels_change_and_prior_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'original.png';out=Path(folder)/'redacted.png'
            original=np.full((10,12,3),200,np.uint8);Image.fromarray(original).save(source)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            record=module.redact(source,out,sha,[[1,2,4,5]])
            actual=np.asarray(Image.open(out));mask=np.zeros((10,12),bool);mask[2:5,1:4]=True
            np.testing.assert_array_equal(actual[~mask],original[~mask])
            self.assertTrue((actual[mask]==0).all())
            self.assertTrue(record['pixels_outside_rectangles_identical'])
            output_sha=record['output_sha256']
            with self.assertRaises(ValueError):module.redact(source,out,sha,[[0,0,1,1]])
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(),output_sha)

    def test_changed_input_or_invalid_rectangle_requires_review(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'original.png';out=Path(folder)/'redacted.png'
            Image.new('RGB',(12,10)).save(source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
            for expected,rect in [('wrong',[[0,0,1,1]]),(sha,[[0,0,13,2]])]:
                with self.assertRaises(ValueError):module.redact(source,out,expected,rect)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
