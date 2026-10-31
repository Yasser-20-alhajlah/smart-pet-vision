"""Conservative instance-aware mask matching (CPU-unit-tested; GPU regression pending).

Avoid applying a union mask to the crop of only the highest-scoring detection.
Does not solve missed animals, especially on images with multiple animals.
"""
from __future__ import annotations

import numpy as np


def pair_detection_boxes(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Greedy one-to-one IoU association for equal-sized final detection sets.

    Class labels are compared AFTER geometry matching. Independent row argmax
    can reuse one candidate several times and hide a displaced/duplicate output.
    Returned index j[i] assigns each reference box one distinct candidate.
    """
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.ndim != 2 or a.shape[1:] != (4,) or b.shape != a.shape:
        raise ValueError("Parity boxes must have equal (N, 4) shapes.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Parity boxes must be finite.")
    if np.any(a[:, 2:] <= a[:, :2]) or np.any(b[:, 2:] <= b[:, :2]):
        raise ValueError("Parity boxes must have positive area.")
    n = len(a)
    indices = np.empty(n, dtype=int)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(0, right - left).prod(2)
    areas_a = (a[:, 2:] - a[:, :2]).prod(1)
    areas_b = (b[:, 2:] - b[:, :2]).prod(1)
    union = areas_a[:, None] + areas_b[None, :] - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        indices[i] = j
        ious[i, :] = -1
        ious[:, j] = -1
    return indices


def best_mask_index(seg_boxes: np.ndarray, det_box: np.ndarray,
                    seg_classes: np.ndarray | None = None,
                    det_class: int | None = None,
                    min_iou: float = 0.10) -> int | None:
    """Return index of segment matching selected detected box; None if unmatched.

    Shape: seg_boxes (N,4), det_box (4,), xyxy. Same-class candidates are
    preferred ONLY if they overlap >= min_iou. Falls back to any overlapping
    candidate to handle classifier/segmenter category disagreements.
    """
    boxes = np.asarray(seg_boxes, dtype=float).reshape((-1, 4))
    target = np.asarray(det_box, dtype=float).reshape((4,))
    if len(boxes) == 0:
        return None
    if not np.isfinite(target).all() or np.any(target[2:] <= target[:2]):
        return None
    if not 0 < min_iou <= 1:
        raise ValueError("Matching IoU must be in (0, 1].")
    if seg_classes is not None and len(seg_classes) != len(boxes):
        raise ValueError("Segmentation class count must match box count.")
    a = np.maximum(boxes[:, :2], target[:2])
    b = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(0.0, b - a).prod(axis=1)
    area = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    target_area = np.maximum(0.0, target[2:] - target[:2]).prod()
    union = area + target_area - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    eligible = np.where(valid & (ious >= min_iou))[0]
    if len(eligible) == 0:
        return None
    if seg_classes is not None and det_class is not None:
        class_indices = eligible[np.asarray(seg_classes)[eligible] == det_class]
        if len(class_indices):
            eligible = class_indices
    return int(eligible[np.argmax(ious[eligible])])
