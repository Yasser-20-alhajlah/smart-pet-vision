"""Detect a corrupted source cohort or saved metric instead of silently passing it."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import verify_v2_saved_results as replay  # noqa: E402
import reproduce_v2_eval as reproduction  # noqa: E402


class V2ReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / 'report/evidence/v2_saved_review_20261114',
                        self.root / 'report/evidence/v2_saved_review_20261114')
        self.head = self.root / 'report/evidence/P2_v2_headline_three_seeds_20261113.json'
        shutil.copy2(ROOT / 'report/evidence/P2_v2_headline_three_seeds_20261113.json', self.head)

    def test_complete_saved_evidence(self):
        self.assertTrue(all(c['passed'] for c in replay.verify(self.root)))

    def test_corrupted_mask_score_is_detected(self):
        p = self.root / 'report/evidence/v2_saved_review_20261114/seed44_seg_per_image.csv'
        with p.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['iou'] = '1.5'
        with p.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        self.assertFalse(all(c['passed'] for c in replay.verify(self.root)))

    def test_wrong_cohort_is_detected(self):
        p = self.root / 'report/evidence/v2_saved_review_20261114/seed43_seg_per_image.csv'
        p.write_text(p.read_text().replace('Abyssinian_107', 'Abyssinian_unrelated'))
        self.assertFalse(all(c['passed'] for c in replay.verify(self.root)))

    def test_incorrect_headline_mean_is_detected(self):
        head = json.loads(self.head.read_text());head['summary_over_seeds']['mIoU']['mean'] = .99
        self.head.write_text(json.dumps(head))
        self.assertFalse(all(c['passed'] for c in replay.verify(self.root)))

    def test_reproduction_rejects_wrong_metadata_before_importing_models(self):
        p = self.root / 'wrong.csv';p.write_text('wrong metadata')
        with self.assertRaisesRegex(ValueError, 'Metadata SHA-256'):
            reproduction.validate_inputs({'split_v2_metadata_sha256': '0'*64}, p, {})


if __name__ == '__main__':
    unittest.main()
