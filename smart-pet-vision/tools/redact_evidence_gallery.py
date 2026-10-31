"""Lossless display-only redaction bound to the exact original gallery SHA256.

Does not run a model or change any metric or source dataset image.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def redact(source: Path, output: Path, expected_sha256: str, rectangles: list) -> dict:
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if original_sha != expected_sha256:
        raise ValueError('Gallery changed; manually review coordinates before redaction.')
    if output.exists() or output.suffix.lower() != '.png':
        raise ValueError('Choose a new PNG output; source and prior output are preserved.')
    original = np.asarray(Image.open(source).convert('RGB')).copy()
    redacted = original.copy(); mask = np.zeros(original.shape[:2], dtype=bool)
    h, w = mask.shape
    for x1, y1, x2, y2 in rectangles:
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('Redaction rectangle is outside the gallery.')
        mask[y1:y2, x1:x2] = True
    if not mask.any():
        raise ValueError('At least one redaction rectangle is required.')
    redacted[mask] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(redacted).save(output, format='PNG')
    decoded = np.asarray(Image.open(output).convert('RGB'))
    unchanged = bool(np.array_equal(original[~mask], decoded[~mask]))
    if not unchanged:
        raise RuntimeError('Pixels outside redaction changed.')
    return dict(source=source.name, source_sha256=original_sha, output=output.name,
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                rectangles_xyxy=rectangles, pixels_outside_rectangles_identical=unchanged,
                purpose='Hide visible people above the pet in the basset_hound failure tile',
                metric_files_modified=False, source_dataset_images_modified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--rectangles', required=True, help='JSON list of xyxy pixel rectangles')
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    record = redact(a.source, a.output, a.expected_sha256, json.loads(a.rectangles))
    a.record.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
