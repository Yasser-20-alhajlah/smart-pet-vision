#!/usr/bin/env python3
"""Evaluate the frozen split-v2 checkpoints; never train, select thresholds or refit temperature.

Use a fresh environment and an actual teammate's operator name. Output is a new
receipt, not an edit to the original records. Six detector/segmenter metrics use
the existing reproduction tolerance (0.005). Other metrics/counts are reported
side by side without inventing additional instructor acceptance criteria.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE2 / 'src'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_inputs(record, metadata, weights):
    if sha(metadata) != record['split_v2_metadata_sha256']:
        raise ValueError('Metadata SHA-256 differs from the original split-v2 record')
    with Path(metadata).open() as stream:
        rows = list(csv.DictReader(stream))
    counts = {k: sum(r['split'] == k for r in rows) for k in ('train', 'val', 'test')}
    if counts != record['split_v2_counts'] or len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Metadata partition counts or unique image IDs differ')
    for kind, path in weights.items():
        if sha(path) != record['weights_sha256'][kind]:
            raise ValueError(f'{kind} checkpoint SHA-256 differs from the selected seed')
    return {r['image'] for r in rows if r['split'] == 'test'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--operator', required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--det-weights', type=Path, required=True)
    parser.add_argument('--seg-weights', type=Path, required=True)
    parser.add_argument('--cls-weights', type=Path, required=True)
    parser.add_argument('--det-yaml', type=Path, required=True)
    parser.add_argument('--seg-yaml', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; choose a new receipt path')
    original = json.loads(args.record.read_text())
    if original.get('seed') not in (42, 43, 44):
        parser.error('Expected a three-seed split-v2 original record')
    weights = {'classifier': args.cls_weights, 'detector': args.det_weights, 'segmenter': args.seg_weights}
    ids = validate_inputs(original, args.metadata, weights)

    import importlib.metadata as md
    import yaml
    for package, expected in original['versions'].items():
        actual = md.version(package).split('+')[0]
        if actual != expected:
            raise ValueError(f'{package}: expected {expected}, found {actual}; use the recorded environment')
    for path in (args.det_yaml, args.seg_yaml):
        spec = yaml.safe_load(path.read_text())
        base = Path(spec.get('path', path.parent))
        base = base if base.is_absolute() else path.parent / base
        part = base / spec['test']
        found = {p.stem for p in part.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        if found != ids:
            raise ValueError(f'{path.name}: YOLO test cohort differs from the frozen metadata')

    import pandas as pd
    from spv import classify, detect, segment
    from spv.common import load_config, seed_everything
    from spv.provenance import record
    from reproduce_phase2_eval import side_by_side
    cfg = load_config()
    cfg['seed'] = original['seed']
    cfg['classification'] = original['recipes']['classification']
    cfg['wandb']['enabled'] = False
    seed_everything(cfg['seed'])
    frame = pd.read_csv(args.metadata)
    test = frame[frame.split == 'test'].reset_index(drop=True)
    inputs = [args.image_dir / f'{im}.jpg' for im in test.image]
    if not all(p.is_file() for p in inputs):
        raise ValueError('Missing original Oxford test images')
    box = detect.test_metrics(args.det_weights, args.det_yaml, 640)
    op = original['detector']['operating_point']
    preds = detect.predict_all(args.det_weights, inputs, conf=op['conf'], iou=op['nms_iou'], imgsz=640)
    at_op = detect.score(preds, detect.gt_for(test), op['conf'])
    mask_ap = segment.yolo_coco_metrics(args.seg_weights, args.seg_yaml, 640)
    masks = segment.yolo_masks(args.seg_weights, inputs, conf=.25, imgsz=640)
    mask, per_image = segment.evaluate(masks, test, args.processed)
    loaders = classify.loaders(frame, args.image_dir, cfg)
    classifier = classify.load_model('finetune', cfg, args.cls_weights)
    logits, labels = classify.predict_logits(classifier, loaders['test'])
    cls = classify.evaluate(classify.softmax(logits, original['classifier']['temperature']), labels, test)
    expected = {'mAP50': original['detector']['coco']['mAP50'],
                'mAP50_95': original['detector']['coco']['mAP50_95'],
                'mask_mAP50': original['segmenter']['coco']['mask_mAP50'],
                'mask_mAP50_95': original['segmenter']['coco']['mask_mAP50_95'],
                'mIoU': original['segmenter']['test_metrics']['mIoU'],
                'mean_dice': original['segmenter']['test_metrics']['mean_dice']}
    measured = {'mAP50': box['mAP50'], 'mAP50_95': box['mAP50_95'],
                'mask_mAP50': mask_ap['mask_mAP50'], 'mask_mAP50_95': mask_ap['mask_mAP50_95'],
                'mIoU': mask['mIoU'], 'mean_dice': mask['mean_dice']}
    compared = side_by_side(expected, measured)
    additional = {}
    for kind, old, new, keys in [
        ('classifier', original['classifier']['test_calibrated'], cls, ['accuracy', 'macro_f1']),
        ('detector_operating_point', original['detector']['test_at_operating_point'], at_op,
         ['tp', 'fp', 'fn', 'precision', 'recall', 'f1'])]:
        additional[kind] = {k: {'original': old[k], 'reproduced': new[k], 'difference': new[k]-old[k]} for k in keys}
    result = {'operator': args.operator, 'executed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Checkpoint evaluation on previously evaluated split-v2 test images; no training or model/threshold selection.',
              'seed': original['seed'], 'source_record_sha256': sha(args.record),
              'metadata_sha256': sha(args.metadata), 'test_images': len(test),
              'weights_sha256': {k: sha(v) for k, v in weights.items()},
              'original_temperature': original['classifier']['temperature'], 'operating_point': op,
              'provenance': record(PHASE2), 'versions': {p: md.version(p) for p in original['versions']},
              'originals': expected, 'reproduced': measured, 'comparison': compared,
              'additional_metrics_no_new_acceptance_thresholds': additional}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    per_image.to_csv(args.out.with_suffix('.per_image.csv'), index=False)
    print(json.dumps(compared, indent=2))
    return 0 if compared['all_within_tolerance'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
