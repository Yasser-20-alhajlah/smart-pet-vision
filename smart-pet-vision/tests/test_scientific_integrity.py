import unittest
import numpy as np
import pandas as pd
from spv.integrity import validate_metadata, validate_predictions, training_class_weights


def frame():
    return pd.DataFrame(dict(image=['a','b','c','d','e','f'],label=[0,1,0,1,0,1],
        species=['cat','dog']*3,breed=['catbreed','dogbreed']*3,
        split=['train','train','val','val','test','test'],group=list(range(6)),md5=list('abcdef')))


class ScientificIntegrity(unittest.TestCase):
    def test_valid_group_split(self):
        validate_metadata(frame())

    def test_missing_ids_hashes_groups_cannot_pass(self):
        for key in ['image','md5','group','split','label','species']:
            f=frame();f.loc[0,key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):validate_metadata(f)

    def test_duplicate_and_identity_leaks_are_rejected(self):
        for key in ['md5','group','animal_group','source_group']:
            f=frame()
            if key not in f:f[key]=f.image
            f.loc[4,key]=f.loc[0,key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_metadata(f)

    def test_empty_duplicate_unknown_or_mislabeled_rows(self):
        for change in [dict(image='a'),dict(split='calibration'),dict(species='dog'),dict(label=2)]:
            f=frame()
            for k,v in change.items():f.loc[4,k]=v
            with self.assertRaises(ValueError):validate_metadata(f)
        with self.assertRaises(ValueError):validate_metadata(frame().iloc[:0])

    def test_prediction_truth_order_must_match(self):
        f=frame().iloc[:2];probs=np.array([[.8,.2],[.1,.9]])
        validate_predictions(probs,np.array([0,1]),f)
        with self.assertRaises(ValueError):validate_predictions(probs,np.array([1,0]),f)

    def test_bad_probabilities_never_generate_metrics(self):
        f=frame().iloc[:2]
        for p in [np.array([[np.nan,0],[0,1]]),np.ones((2,2)),np.ones((1,2)),np.array([[-1,2],[0,1]])]:
            with self.assertRaises(ValueError):validate_predictions(p,np.array([0,1]),f)

    def test_balancing_uses_training_labels_only_and_default_is_unchanged(self):
        self.assertIsNone(training_class_weights([0,1,1,1]))
        np.testing.assert_allclose(training_class_weights([0,1,1,1],'balanced'),[2,2/3])
        for labels,strategy in [([], 'balanced'),([1,1],'balanced'),([0,2],'none'),([0,1],'invalid')]:
            with self.assertRaises(ValueError):training_class_weights(labels,strategy)

if __name__=='__main__':unittest.main()
