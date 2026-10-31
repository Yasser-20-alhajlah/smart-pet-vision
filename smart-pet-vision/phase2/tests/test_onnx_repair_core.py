"""Pure gate regression tests; not trained-model parity results."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from onnx_repair_core import ParityGate, classify_variant_result

class ParityGateTests(unittest.TestCase):
    def make(self,**updates):
        values=dict(raw_random_max=.0005,raw_real_max=.0006,same_counts=50,same_classes=50,
                    images=50, final_box_max_px=.0008,images_with_detections=30)
        values.update(updates)
        return ParityGate(**values)
    def test_pass_still_never_allows_auto_deploy(self):
        d=self.make().check();self.assertEqual(d['status'],'PASS');self.assertFalse(d['deploy_allowed'])
    def test_original_raw_failure_not_overridden_by_final_parity(self):
        d=self.make(raw_random_max=.00201416,final_box_max_px=.0001).check()
        self.assertEqual(d['status'],'FAIL');self.assertIn('RAW_HEAD_PARITY_FAILED',d['reasons'])
    def test_missing_detections_rejected_even_if_zero_displacement(self):
        with self.assertRaisesRegex(ValueError,'actual predicted detection'):
            self.make(images_with_detections=0).check()
    def test_count_or_class_mismatch_fails(self):
        for kw in [dict(same_counts=49,same_classes=49),dict(same_classes=49)]:
            self.assertEqual(self.make(**kw).check()['status'],'FAIL')
    def test_invalid_counter_fails(self):
        with self.assertRaises(ValueError):self.make(same_counts=49,same_classes=50).check()
    def test_no_posthoc_tolerance_inflation(self):
        self.assertEqual(self.make(raw_random_max=.001001).check()['status'],'FAIL')
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.make(raw_random_max=float('nan')).check()
    def test_no_partial_or_blocked_claim_as_pass(self):
        self.assertEqual(classify_variant_result({'status':'PASS','raw_passed':True}), 'NO_GO')
        self.assertEqual(classify_variant_result({'status':'BLOCKED'}),'NO_GO')
        self.assertEqual(classify_variant_result(self.make().check()),'ELIGIBLE_FOR_INDEPENDENT_REVIEW')

if __name__=='__main__':unittest.main()
