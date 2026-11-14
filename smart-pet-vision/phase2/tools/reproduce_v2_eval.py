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
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE2 / 'src'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_versions(expected, actual, cpu_reference=False, host_platform=None):
    """Keep strict reproduction; permit only the documented macOS NumPy CPU probe."""
    differences = {}
    for package, wanted in expected.items():
        found = actual[package].split('+')[0]
        if found == wanted:
            continue
        permitted = (cpu_reference and (host_platform or sys.platform) == 'darwin'
                     and package == 'numpy' and wanted == '2.1.3' and found == '2.3.5')
        if not permitted:
            raise ValueError(f'{package}: expected {wanted}, found {found}; use the recorded environment')
        differences[package] = {'original': wanted, 'actual': found,
                                'reason': 'Ultralytics 8.4.182 excludes NumPy 2.1.x on Darwin; CPU reference only'}
    return differences


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



def validate_yolo_test(spec, yaml_path, ids, metadata_rows, check_boxes=False, processed=None):
    """Reject duplicate IDs, changed class mapping or missing/changed test labels."""
    names = spec.get('names')
    if names not in ({0: 'cat', 1: 'dog'}, ['cat', 'dog']):
        raise ValueError(f'{yaml_path.name}: expected class order cat, dog')
    base = Path(spec.get('path', yaml_path.parent))
    base = base if base.is_absolute() else yaml_path.parent / base
    part = base / spec['test']
    images = [p for p in part.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    found = [p.stem for p in images]
    if len(found) != len(ids) or len(found) != len(set(found)) or set(found) != ids:
        raise ValueError(f'{yaml_path.name}: YOLO test cohort differs from frozen metadata')
    rows = {r['image']: r for r in metadata_rows if r['split'] == 'test'}
    for image in images:
        # The prepared export uses images/test and labels/test, as Ultralytics does.
        parts = list(image.parts)
        positions = [i for i, v in enumerate(parts) if v == 'images']
        if not positions:
            raise ValueError('Prepared YOLO layout must contain an images directory')
        parts[positions[-1]] = 'labels'
        label = Path(*parts).with_suffix('.txt')
        if not label.is_file():
            raise ValueError(f'Missing test label for {image.stem}')
        if check_boxes:
            row = rows[image.stem]
            x1, y1, x2, y2, width, height = [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2', 'width', 'height')]
            expected = f"{int(row['label'])} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
            if label.read_text().strip() != expected:
                raise ValueError(f'Detector test label differs from frozen metadata: {image.stem}')
        elif processed is not None:
            import numpy as np
            from PIL import Image
            from spv.data import mask_to_polygon
            row = rows[image.stem]
            mask = np.array(Image.open(processed/'masks'/f'{image.stem}.png')) > 0
            if mask.shape != (int(row['height']), int(row['width'])):
                raise ValueError(f'Test mask dimensions differ: {image.stem}')
            poly = mask_to_polygon(mask)
            if poly is None:
                raise ValueError(f'Test mask has no polygon: {image.stem}')
            xy = np.asarray(poly).reshape(-1,2) / [float(row['width']), float(row['height'])]
            expected = f"{int(row['label'])} " + ' '.join(f'{v:.6f}' for v in xy.flatten())
            if label.read_text().strip() != expected:
                raise ValueError(f'Segmenter test label differs from prepared mask: {image.stem}')

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--operator', required=True)
    parser.add_argument('--cpu-reference', action='store_true',
                        help='Automated CPU reference, not a human second-member or original-T4 reproduction claim')
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
    if args.out.exists() or args.out.with_suffix('.per_image.csv').exists():
        parser.error('Output already exists; choose a new receipt path')
    original = json.loads(args.record.read_text())
    if original.get('seed') not in (42, 43, 44):
        parser.error('Expected a three-seed split-v2 original record')
    weights = {'classifier': args.cls_weights, 'detector': args.det_weights, 'segmenter': args.seg_weights}
    ids = validate_inputs(original, args.metadata, weights)

    import importlib.metadata as md
    import yaml
    actual_versions = {p: md.version(p) for p in original['versions']}
    differences = verify_versions(original['versions'], actual_versions, args.cpu_reference)
    if args.cpu_reference:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['WANDB_MODE'] = 'disabled'
        import torch
        if torch.cuda.is_available():
            raise RuntimeError('CPU reference must not use CUDA')
        torch.set_num_threads(2)
    with args.metadata.open() as stream:
        metadata_rows = list(csv.DictReader(stream))
    for path, check_boxes in ((args.det_yaml, True), (args.seg_yaml, False)):
        spec = yaml.safe_load(path.read_text())
        validate_yolo_test(spec, path, ids, metadata_rows, check_boxes=check_boxes, processed=args.processed)

    import pandas as pd
    from spv import classify, detect, segment
    from spv.common import load_config, seed_everything
    from spv.provenance import record
    from reproduce_phase2_eval import side_by_side
    cfg = load_config()
    cfg['seed'] = original['seed']
    cfg['classification'] = original['recipes']['classification']
    if args.cpu_reference:
        cfg['classification'] = {**cfg['classification'], 'num_workers': 0}
    cfg['wandb']['enabled'] = False
    seed_everything(cfg['seed'])
    frame = pd.read_csv(args.metadata)
    test = frame[frame.split == 'test'].reset_index(drop=True)
    inputs = [args.image_dir / f'{im}.jpg' for im in test.image]
    if not all(p.is_file() for p in inputs):
        raise ValueError('Missing original Oxford test images')
    for path, row in zip(inputs, test.itertuples()):
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'md5').hexdigest() != row.md5:
                raise ValueError(f'Original Oxford image bytes differ: {path.name}')
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
    per_class = {}
    for index, species in enumerate(('cat', 'dog')):
        filtered_preds = {name: (b[c == index], c[c == index], s[c == index])
                          for name, (b, c, s) in preds.items()}
        filtered_gts = {name: (b[c == index], c[c == index])
                        for name, (b, c) in detect.gt_for(test).items()}
        per_class[species] = {'images': int((test.species == species).sum()),
                             'detector_at_operating_point': detect.score(filtered_preds, filtered_gts, op['conf']),
                             'box_AP50': box['AP50_per_class'][species],
                             'box_AP50_95': box['AP50_95_per_class'][species],
                             'mask_AP50': mask_ap['mask_AP50_per_class'][species],
                             'mask_AP50_95': mask_ap['mask_AP50_95_per_class'][species],
                             'mask_overlap': mask['by_species'][species]}
    result = {'operator': args.operator, 'executed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Checkpoint evaluation on previously evaluated split-v2 test images; no training or model/threshold selection.',
              'seed': original['seed'], 'source_record_sha256': sha(args.record),
              'metadata_sha256': sha(args.metadata), 'test_images': len(test),
              'weights_sha256': {k: sha(v) for k, v in weights.items()},
              'original_temperature': original['classifier']['temperature'], 'operating_point': op,
              'provenance': record(PHASE2), 'versions': {p: md.version(p) for p in original['versions']},
              'verification_kind': 'automated_cpu_reference' if args.cpu_reference else 'recorded_environment_checkpoint_evaluation',
              'human_second_member_claimed': False,
              'platform': platform.platform(), 'python': platform.python_version(),
              'environment_differences': differences,
              'cpu_reference_inference_workers': 0 if args.cpu_reference else None,
              'originals': expected, 'reproduced': measured, 'comparison': compared,
              'per_class_from_this_evaluation': per_class,
              'additional_metrics_no_new_acceptance_thresholds': additional}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    per_image.to_csv(args.out.with_suffix('.per_image.csv'), index=False)
    print(json.dumps(compared, indent=2))
    return 0 if compared['all_within_tolerance'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
