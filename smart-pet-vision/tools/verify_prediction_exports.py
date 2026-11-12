"""Recompute accuracy, macro-F1, confusion matrices and 15-bin ECE from saved CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path


def recompute(rows, calibrated):
    if not rows:
        raise ValueError('Prediction export is empty')
    suffix = 'calibrated' if calibrated else 'raw'
    cm = [[0, 0], [0, 0]]
    bins = [[] for _ in range(15)]
    for row in rows:
        label = int(row['label'])
        if label not in (0, 1):
            raise ValueError('Species labels must be 0 (cat) or 1 (dog)')
        probs = [float(row['p_cat_' + suffix]), float(row['p_dog_' + suffix])]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in probs):
            raise ValueError('Invalid probabilities')
        if abs(sum(probs) - 1) > 1e-6:
            raise ValueError('Probabilities do not sum to one')
        pred = 0 if probs[0] >= probs[1] else 1
        cm[label][pred] += 1
        confidence = max(probs)
        index = max(0, min(14, math.ceil(confidence * 15) - 1))
        bins[index].append((confidence, int(pred == label)))
    n = len(rows)
    f1s = []
    for i in range(2):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(2) if j != i)
        fn = sum(cm[i][j] for j in range(2) if j != i)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0)
    ece = sum(abs(sum(c for c, _ in b) - sum(a for _, a in b)) / n
              for b in bins if b)
    return {'n': n, 'confusion_matrix': cm,
            'accuracy': sum(cm[i][i] for i in range(2)) / n,
            'macro_f1': sum(f1s) / 2, 'ece': ece}


def verify(directory):
    reports, reference = {}, None
    for kind in ('scratch', 'frozen', 'finetune'):
        with (directory / f'{kind}_predictions.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        identities = [(r['image'], int(r['label'])) for r in rows]
        if len(identities) != 1106 or len({x[0] for x in identities}) != 1106:
            raise ValueError('Expected 1106 unique test images')
        if reference is not None and identities != reference:
            raise ValueError('Model exports have different image order or labels')
        reference = identities
        saved = json.loads((directory / f'{kind}_metrics.json').read_text())
        result = {}
        for calibrated, key in ((False, 'raw'), (True, 'calibrated_using_saved_temperature')):
            actual = recompute(rows, calibrated)
            expected = saved[key]
            if actual['confusion_matrix'] != expected['confusion_matrix'] or actual['n'] != expected['n']:
                raise ValueError(f'{kind} {key}: count or confusion matrix differs')
            for metric in ('accuracy', 'macro_f1', 'ece'):
                if not math.isfinite(expected[metric]) or abs(actual[metric] - expected[metric]) >= 1e-6:
                    raise ValueError(f'{kind} {key}: {metric} differs')
            result[key] = {'recomputed': actual, 'passed': True}
        reports[kind] = result
    return {'implementation': 'independent Python standard library; no model execution',
            'models': reports, 'sample_order_labels_and_unique_1106': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, help='Optional new JSON check record')
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print('All 3 models: accuracy, macro-F1, confusion matrices and ECE passed; image order and labels agree')


if __name__ == '__main__':
    main()
