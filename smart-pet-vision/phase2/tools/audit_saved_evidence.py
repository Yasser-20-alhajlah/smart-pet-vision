"""Recompute saved-record arithmetic; never infer models or treat this as a new test.

Usage: python tools/audit_saved_evidence.py --reference PATH --archive PATH --output PATH
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def audit(reference: Path, archive: Path) -> dict:
    meta = pd.read_csv(reference/'data/metadata.csv')
    if meta.image.duplicated().any():
        raise ValueError('Duplicate Oxford image IDs')
    split_leaks = int((meta.groupby('group').split.nunique() > 1).sum())
    md5_leaks = int((meta.groupby('md5').split.nunique() > 1).sum())
    labels = pd.read_csv(reference/'team/labels.csv')
    historical = pd.read_csv(archive/'evidence/team_predictions.csv')
    current = pd.read_csv(archive/'new_evidence/candidate_team_predictions.csv')
    out = {'scope': 'Saved records, arithmetic and ID consistency; no new model inference',
           'oxford_rows': len(meta), 'oxford_splits': meta.split.value_counts().to_dict(),
           'groups_crossing_splits': split_leaks, 'md5_crossing_splits': md5_leaks}
    if split_leaks or md5_leaks:
        raise ValueError('Cross-split duplicate group or MD5')
    for name, records in [('historical', historical), ('reported_cpu', current)]:
        if set(records.image) != set(labels.image) or records.image.duplicated().any():
            raise ValueError('Prediction IDs do not cover labels one-to-one')
        original_truth = records[['image', 'true']].merge(
            labels[['image', 'species']], on='image', validate='one_to_one')
        if (original_truth.true != original_truth.species).any():
            raise ValueError('Stored prediction truths differ from current label version')
        merged = records.drop(columns=['true', 'n_animals', 'part'], errors='ignore').merge(
            labels.rename(columns={'species': 'true'}), on='image', validate='one_to_one')
        valid = merged[merged.true.isin(['cat', 'dog'])]
        ready = valid[valid.verdict == 'ready']
        out[name] = dict(images=len(merged), species_n=len(valid),
                         correct=int((valid.true == valid.pred).sum()),
                         referrals=int((merged.verdict != 'ready').sum()),
                         ready_total=int((merged.verdict == 'ready').sum()),
                         ready_species_n=len(ready), ready_correct=int((ready.true == ready.pred).sum()),
                         multi_animal_ready=int(((merged.n_animals > 1) & (merged.verdict == 'ready')).sum()),
                         mixed_species_ready=int(((~merged.true.isin(['cat', 'dog'])) & (merged.verdict == 'ready')).sum()))
    sam = pd.read_csv(archive/'evidence/sam_baseline_per_image.csv')
    yolo = pd.read_csv(archive/'evidence/yolo_seg_per_image.csv')
    test_ids = set(meta.loc[meta.split == 'test', 'image'])
    columns = ['iou', 'dice', 'iou_no_border', 'dice_no_border']
    for name, frame in [('sam', sam), ('yolo', yolo)]:
        if frame.image.duplicated().any() or not set(frame.image).issubset(test_ids):
            raise ValueError('Duplicate or non-test segmentation image IDs')
        if not frame[columns].apply(lambda x: x.between(0, 1)).all().all():
            raise ValueError('Invalid per-image scores')
        out[name] = {'n': len(frame), 'means': frame[columns].mean().to_dict()}
    paired = sam.merge(yolo, on='image', suffixes=('_sam', '_yolo'), validate='one_to_one')
    if len(paired) != len(sam):
        raise ValueError('SAM subset is not fully covered by YOLO')
    out['matched_subset'] = {'n': len(paired), 'means': paired[
        [c+'_'+model for model in ['sam', 'yolo'] for c in columns]].mean().to_dict()}
    for kind in ['labels', 'boxes']:
        frame = pd.read_csv(reference/f'review_{kind}/review.csv')
        out[f'{kind}_review'] = {'n': len(frame), 'accepted': int((frame.ok == 1).sum()),
                                'flagged': int((frame.ok == 0).sum()),
                                'blank': int(frame.ok.isna().sum())}
    # One-sided exact 95% upper bound for 0 observed errors in the label audit.
    out['label_error_upper_95_one_sided'] = 1 - .05 ** (1 / out['labels_review']['n'])
    out['reported_cpu_policy_flags'] = dict(
        staff_approval_required=int(current.staff_approval_required.eq(True).sum()),
        automatic_publication_allowed=int(current.automatic_publication_allowed.eq(True).sum()))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from spv.detect import choose_operating_point
    sweep = pd.read_csv(archive/'evidence/M4_threshold_sweep.csv')
    out['recomputed_operating_point'] = choose_operating_point(sweep)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = audit(a.reference, a.archive)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
