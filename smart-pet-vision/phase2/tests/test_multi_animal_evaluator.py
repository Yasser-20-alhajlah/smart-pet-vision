"""Cohort/metric integrity checks without trained-weight inference."""
from __future__ import annotations
import tempfile,unittest,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_multi_animal_candidate import validate_cohort,evaluate_paired

class MultiAnimalEvaluator(unittest.TestCase):
    def test_complete_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['a','b']:
                Image.new('RGB',(5,5)).save(root/f'{name}.jpg')
            ok=pd.DataFrame([{'image':'a','species':'cat','n_animals':1},
                             {'image':'b','species':'mixed','n_animals':2}])
            self.assertEqual(set(validate_cohort(root,ok)),{'a','b'})
            for changed in [ok.iloc[:1],ok.assign(n_animals=[1,0]),pd.concat([ok,ok.iloc[:1]]),
                            ok.assign(species=['cat','horse'])]:
                with self.assertRaises(ValueError): validate_cohort(root,changed)
    def test_paired_safety_and_referral_cost(self):
        data=pd.DataFrame([
           {'image':'a','species':'cat','n_animals':1,'pred':'cat',
            'baseline_verdict':'ready','candidate_verdict':'ready'},
           {'image':'b','species':'mixed','n_animals':2,'pred':'cat',
            'baseline_verdict':'ready','candidate_verdict':'refer_to_staff'},
           {'image':'c','species':'dog','n_animals':1,'pred':'dog',
            'baseline_verdict':'ready','candidate_verdict':'refer_to_staff'},
        ])
        x=evaluate_paired(data)
        self.assertEqual(x['multi_ready_delta'],-1)
        self.assertEqual(x['single_referral_delta'],1)
        self.assertEqual(x['new_referrals'],2)
        data.loc[1,'baseline_verdict']='refer_to_staff';data.loc[1,'candidate_verdict']='ready'
        with self.assertRaises(ValueError):evaluate_paired(data)

if __name__=='__main__': unittest.main()
