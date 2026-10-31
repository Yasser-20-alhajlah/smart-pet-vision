"""Conservative secondary-animal cues for *human review*, not animal counting.

Boxes from different passes use the original photograph's pixel coordinate frame.
These helpers do not access ground-truth labels or choose thresholds from a test set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReviewProposal:
    source: str
    boxes_xyxy: np.ndarray
    confidence: np.ndarray


def validate_boxes(boxes, scores=None, *, bounds: tuple[int, int] | None = None):
    b = np.asarray(boxes, dtype=float)
    if b.size == 0:
        b = np.empty((0, 4), dtype=float)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError('Secondary-animal boxes must be finite (N,4) xyxy coordinates.')
    if np.any(b[:, 2:] <= b[:, :2]):
        raise ValueError('Secondary-animal boxes must have positive area.')
    if bounds is not None and len(b):
        w, h = bounds
        if w <= 0 or h <= 0 or np.any(b[:, :2] < -0.5) or np.any(b[:, 2] > w + 0.5) or np.any(b[:, 3] > h + 0.5):
            raise ValueError('Secondary-animal box is outside original photo.')
    if scores is None:
        return b
    c = np.asarray(scores, dtype=float)
    if c.shape != (len(b),) or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError('Secondary-animal scores must be finite [0,1] and align with boxes.')
    return b, c


def iou(reference, candidate) -> float:
    a = validate_boxes(np.asarray(reference).reshape(1, 4))[0]
    b = validate_boxes(np.asarray(candidate).reshape(1, 4))[0]
    overlap = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union else 0.


def additional_animal_cues(primary_boxes, proposals: Iterable[ReviewProposal], *,
                           confidence_floor: float, same_instance_iou: float,
                           image_size: tuple[int, int], max_proposals: int = 300) -> list[dict]:
    """Return candidates not associated with a *primary* detection.

    Geometric association is class-independent. High overlap is interpreted as
    the same instance; ambiguous boxes are referred rather than silently ignored.
    This is not a proof that a second animal exists.
    """
    if not 0 < confidence_floor <= 1 or not 0 < same_instance_iou <= 1 or max_proposals < 1:
        raise ValueError('Invalid proposal matching parameters.')
    primary = validate_boxes(primary_boxes, bounds=image_size)
    results: list[dict] = []
    total = 0
    for item in proposals:
        boxes, scores = validate_boxes(item.boxes_xyxy, item.confidence, bounds=image_size)
        if not isinstance(item.source, str) or not item.source:
            raise ValueError('Proposal source is required.')
        total += len(boxes)
        if total > max_proposals:
            raise ValueError('Excessive secondary proposals; conservatively refer to staff.')
        for box, score in zip(boxes, scores):
            if score < confidence_floor:
                continue
            best_primary_iou = max((iou(box, p) for p in primary), default=0.)
            if best_primary_iou >= same_instance_iou:
                continue
            if any(iou(box, seen['box_xyxy']) >= same_instance_iou for seen in results):
                seen = next(v for v in results if iou(box, v['box_xyxy']) >= same_instance_iou)
                seen['sources'] = sorted(set(seen['sources'] + [item.source]))
                seen['confidence'] = max(seen['confidence'], float(score))
                continue
            results.append({'box_xyxy': [float(v) for v in box],
                            'confidence': float(score), 'sources': [item.source],
                            'max_primary_iou': best_primary_iou,
                            'interpretation': 'additional_or_ambiguous_animal_candidate'})
    return results


def review_tiles(width: int, height: int, *, fraction: float = 0.65,
                 min_side: int = 128) -> list[tuple[int, int, int, int]]:
    """Deterministic 2x2 overlapping tiles; no ground-truth or tuning involved."""
    if not (0.5 <= fraction < 1.0) or min(width, height) < min_side:
        return []
    tile_w, tile_h = math.ceil(width * fraction), math.ceil(height * fraction)
    return [(x, y, x + tile_w, y + tile_h) for y in (0, height - tile_h)
            for x in (0, width - tile_w)]


def map_tile_boxes(boxes, tile: tuple[int, int, int, int], image_size: tuple[int, int]) -> np.ndarray:
    b = validate_boxes(boxes, bounds=(tile[2] - tile[0], tile[3] - tile[1])).copy()
    if len(b):
        b += np.array([tile[0], tile[1], tile[0], tile[1]], dtype=float)
        validate_boxes(b, bounds=image_size)
    return b
