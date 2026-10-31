"""Pure, deterministic ONNX parity acceptance logic for Smart Pet Vision.

No model libraries, images, network, thresholds changes, or I/O side effects.
The project rule is max absolute raw-head error <= 0.001, not a relative score.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


PROJECT_TOLERANCE = 0.001


@dataclass(frozen=True)
class ParityGate:
    raw_random_max: float
    raw_real_max: float
    same_counts: int
    same_classes: int
    images: int
    final_box_max_px: float
    images_with_detections: int

    def check(self) -> dict:
        vals = (self.raw_random_max, self.raw_real_max, self.final_box_max_px)
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in vals):
            raise ValueError('Parity errors must be nonnegative finite numbers')
        if self.images < 50 or self.images_with_detections < 1:
            raise ValueError('Need >=50 real images and >=1 actual predicted detection')
        if not all(isinstance(v, int) and 0 <= v <= self.images
                   for v in (self.same_counts, self.same_classes, self.images_with_detections)):
            raise ValueError('Invalid denominator or detection agreement counters')
        if self.same_classes > self.same_counts:
            raise ValueError('Class agreement cannot exceed count agreement')
        raw_ok = max(self.raw_random_max, self.raw_real_max) <= PROJECT_TOLERANCE
        final_ok = (self.same_counts == self.images and self.same_classes == self.images and
                    self.final_box_max_px <= PROJECT_TOLERANCE)
        return {'status': 'PASS' if raw_ok and final_ok else 'FAIL',
                'raw_passed': raw_ok, 'final_passed': final_ok,
                'tolerance': PROJECT_TOLERANCE,
                'deploy_allowed': False,  # independent multi-animal and T4 safety gates
                'reasons': ([] if raw_ok and final_ok else
                            (['RAW_HEAD_PARITY_FAILED'] if not raw_ok else []) +
                            (['POST_NMS_PARITY_FAILED'] if not final_ok else []))}


def classify_variant_result(result: dict) -> str:
    """Keep PASS distinct from incomplete/blocked/unknown; never mask a failure."""
    if result.get('status') != 'PASS':
        return 'NO_GO'
    if (result.get('raw_passed') is not True or
        result.get('final_passed') is not True):
        return 'NO_GO'
    return 'ELIGIBLE_FOR_INDEPENDENT_REVIEW'
