#!/usr/bin/env python3
"""Freeze exactly 50+ labeled/representative images ONCE before parity debugging."""
import argparse
from pathlib import Path
from onnx_parity_diagnostics import generate_cohort_manifest

p = argparse.ArgumentParser()
p.add_argument('--image-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--count', type=int, default=50)
a = p.parse_args()
records = generate_cohort_manifest(a.image_root, a.output, n=a.count)
print(f"Wrote {len(records)} frozen IDs to {a.output}")
