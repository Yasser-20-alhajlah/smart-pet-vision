#!/usr/bin/env python3
"""Replay saved split-v2 arithmetic. No inference, training, network or data-split edits.

This checks retained summaries against per-image masks, validation sweeps and
confusion matrices. It does not independently reproduce mAP from checkpoints.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))



def verify(root):
    evidence = root / 'report' / 'evidence'
    saved = evidence / 'v2_saved_review_20261114'
    headline = json.loads((evidence / 'P2_v2_headline_three_seeds_20261113.json').read_text())
    checks = []

    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})

    cohorts = []
    records = {}
    for seed in ('42', '43', '44'):
        r = json.loads((saved / f'seed{seed}_record.json').read_text())
        s = headline['seeds'][seed]
        records[seed] = r
        check(f'{seed}: recorded seed and split counts', r['seed'] == int(seed) and r['split_v2_counts']['test'] == 1050)
        bindings = [
            (s['classifier']['accuracy'], r['classifier']['test']['accuracy']),
            (s['classifier']['macro_f1'], r['classifier']['test']['macro_f1']),
            (s['classifier']['errors'], 1050 * (1-r['classifier']['test']['accuracy'])),
            *[(s['detector'][k], r['detector']['coco'][k]) for k in ('mAP50', 'mAP50_95')],
            *[(s['detector']['test_at_operating_point'][k], r['detector']['test_at_operating_point'][k])
              for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')],
            *[(s['detector']['operating_point'][k], r['detector']['operating_point'][k])
              for k in ('conf', 'nms_iou', 'val_f1')],
            (s['segmenter']['mask_mAP50_95'], r['segmenter']['coco']['mask_mAP50_95']),
            (s['segmenter']['mIoU'], r['segmenter']['test_metrics']['mIoU']),
            (s['segmenter']['dice'], r['segmenter']['test_metrics']['mean_dice']),
        ]
        check(f'{seed}: headline bound to retained seed record', all(abs(a-b) < 1e-12 for a,b in bindings))
        check(f'{seed}: headline detector checkpoint hash', s['detector']['weights_sha256'] == r['weights_sha256']['detector'])
        check(f'{seed}: headline W&B identifiers', all(s['wandb_run_ids'][k] == r['wandb'][k]['run_id'] for k in ('classifier', 'detector', 'segmenter')))
        masks = read_csv(saved / f'seed{seed}_seg_per_image.csv')
        ids = [v['image'] for v in masks]
        check(f'{seed}: 1050 unique test images', len(ids) == len(set(ids)) == 1050)
        cohorts.append(set(ids))
        for column, key in [('iou', 'mIoU'), ('dice', 'mean_dice'),
                            ('iou_no_border', 'mIoU_no_border'), ('dice_no_border', 'mean_dice_no_border')]:
            values = [float(v[column]) for v in masks]
            check(f'{seed}: {column} values finite and within bounds',
                  all(math.isfinite(v) and 0 <= v <= 1 for v in values))
            check(f'{seed}: recompute {key}', abs(statistics.mean(values) - r['segmenter']['test_metrics'][key]) < 1e-12)
        cm = r['classifier']['test']['confusion_matrix']
        total = sum(sum(v) for v in cm)
        check(f'{seed}: classification accuracy from confusion matrix', total == 1050 and
              abs((cm[0][0] + cm[1][1]) / total - r['classifier']['test']['accuracy']) < 1e-12)
        f1 = [2*cm[i][i] / (sum(cm[i]) + sum(row[i] for row in cm)) for i in range(2)]
        check(f'{seed}: classification macro-F1 from confusion matrix',
              abs(statistics.mean(f1) - r['classifier']['test']['macro_f1']) < 1e-12)
        counts = r['detector']['test_at_operating_point']
        tp, fp, fn = (counts[k] for k in ['tp', 'fp', 'fn'])
        for key, val in [('precision', tp/(tp+fp)), ('recall', tp/(tp+fn)), ('f1', 2*tp/(2*tp+fp+fn))]:
            check(f'{seed}: detector {key} from counts', abs(val-counts[key]) < 1e-12)
        sweep = read_csv(saved / f'seed{seed}_det_val_sweep.csv')
        candidates = [v for v in sweep if float(v['recall']) >= .80]
        best = max(candidates or sweep, key=lambda v: (float(v['f1']), float(v['conf'])))
        op = r['detector']['operating_point']
        check(f'{seed}: operating point from validation only',
              all(abs(float(best[k])-op[k]) < 1e-12 for k in ['conf', 'nms_iou']) and
              abs(float(best['f1'])-op['val_f1']) < 1e-12 and
              all(abs(float(best['val_'+k] if 'val_'+k in best else best[k])-op['val_'+k]) < 1e-12 for k in ('precision', 'recall')) and
              op['recall_floor_met'] == bool(candidates))
        curve = read_csv(saved / f'seed{seed}_det_train_val.csv')
        best_val = max(float(v['metrics/mAP50-95(B)']) for v in curve)
        check(f'{seed}: detector selection score from training CSV', abs(best_val-s['detector']['best_val_mAP50_95']) < 1e-12)
    check('all seeds use the same recorded test cohort', all(v == cohorts[0] for v in cohorts))
    getters = {
        'classifier_accuracy': lambda s: s['classifier']['accuracy'],
        'detector_mAP50': lambda s: s['detector']['mAP50'],
        'detector_mAP50_95': lambda s: s['detector']['mAP50_95'],
        'detector_f1_at_operating_point': lambda s: s['detector']['test_at_operating_point']['f1'],
        'mask_mAP50_95': lambda s: s['segmenter']['mask_mAP50_95'],
        'mIoU': lambda s: s['segmenter']['mIoU'],
        'dice': lambda s: s['segmenter']['dice'],
    }
    for key, get in getters.items():
        values = [get(s) for s in headline['seeds'].values()]
        recorded = headline['summary_over_seeds'][key]
        check(key + ': mean and sample SD', abs(statistics.mean(values)-recorded['mean']) < 1e-12 and
              abs(statistics.stdev(values)-recorded['sd']) < 1e-12 and
              abs(min(values)-recorded['min']) < 1e-12 and abs(max(values)-recorded['max']) < 1e-12)
    selected = max(headline['seeds'], key=lambda k: headline['seeds'][k]['detector']['best_val_mAP50_95'])
    check('seed selected on validation, not ONNX/test', int(selected) == headline['served_checkpoint_by_rule']['seed'] == 44)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=PHASE2)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    checks = verify(args.root)
    out = {'scope': __doc__.strip(), 'checks': checks, 'passed': all(v['passed'] for v in checks)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2)+'\n')
    for c in checks:
        print(('PASS' if c['passed'] else 'FAIL') + ': ' + c['check'])
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks")
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
