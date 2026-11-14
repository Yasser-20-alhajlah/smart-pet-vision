"""An explicit CPU probe must not weaken the recorded-environment verifier."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from reproduce_v2_eval import verify_versions


class EnvironmentContract(unittest.TestCase):
    def setUp(self):
        self.expected={'numpy':'2.1.3','torch':'2.11.0'}

    def test_default_still_rejects_numpy_difference(self):
        with self.assertRaises(ValueError):
            verify_versions(self.expected,{'numpy':'2.3.5','torch':'2.11.0'},host_platform='darwin')

    def test_explicit_macos_cpu_probe_records_actual_difference(self):
        d=verify_versions(self.expected,{'numpy':'2.3.5','torch':'2.11.0'},True,'darwin')
        self.assertEqual(d['numpy']['actual'],'2.3.5')
        self.assertEqual(d['numpy']['original'],'2.1.3')

    def test_numpy_exception_does_not_extend_to_linux(self):
        with self.assertRaises(ValueError):
            verify_versions(self.expected,{'numpy':'2.3.5','torch':'2.11.0'},True,'linux')

    def test_cpu_probe_rejects_other_model_library_difference(self):
        with self.assertRaises(ValueError):
            verify_versions(self.expected,{'numpy':'2.3.5','torch':'2.10.0'},True,'darwin')

    def test_cpu_probe_rejects_unreviewed_numpy_version(self):
        with self.assertRaises(ValueError):
            verify_versions(self.expected,{'numpy':'2.4.0','torch':'2.11.0'},True,'darwin')


if __name__=='__main__':unittest.main()
