"""Model-independent numeric diagnostics for SPV detector raw-head ONNX parity.

The project release tolerance remains an absolute 0.001 in native raw-tensor units.
This module DOES NOT infer, export, alter checkpoints, or change any acceptance rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TOLERANCE = 0.001


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_snapshot(repo: Path, snapshot: dict[str, str]) -> dict:
    """Fail closed if any approved original source byte differs from saved SHA256."""
    repo = Path(repo).resolve()
    if not snapshot:
        raise ValueError("Frozen source manifest must not be empty")
    mismatches = []
    for rel, expected in sorted(snapshot.items()):
        p = repo / rel
        if p.resolve() != repo and repo not in p.resolve().parents:
            raise ValueError(f"Unsafe path in source snapshot: {rel}")
        actual = sha256_file(p) if p.is_file() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"approved_file_count": len(snapshot), "matched": len(snapshot) - len(mismatches),
            "mismatches": mismatches, "passed": not mismatches}


def tensor_diagnostics(reference: np.ndarray, candidate: np.ndarray,
                       tolerance: float = TOLERANCE, top_k: int = 12) -> dict:
    """Compare numerically aligned raw outputs; annotate channels, never alter tensors.

    Expected YOLO11 detect shape is (batch, 4+class_count, positions). The gate
    also validates class/shape in model execution; this function does not infer
    semantic correctness from a small difference.
    """
    if tolerance != TOLERANCE:
        raise ValueError("Original project tolerance must remain 0.001")
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != 1 or a.shape[1] < 5:
        raise ValueError(f"Expected aligned (1, C>=5, N) tensors; got {a.shape} and {b.shape}")
    if a.dtype.kind not in "f" or b.dtype.kind not in "f":
        raise ValueError("Raw outputs must be floating-point")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Non-finite raw outputs are not valid parity")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    desc = lambda x: {"max": float(np.max(x)), "mean": float(np.mean(x)),
                      "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                      "p99": float(np.percentile(x, 99)),
                      "elements_above_tolerance": int(np.count_nonzero(x > tolerance)),
                      "total_elements": int(x.size)}
    channels = []
    for c in range(a.shape[1]):
        role = ("box_" + ("cx", "cy", "width", "height")[c]) if c < 4 else f"class_{c-4}"
        channels.append({"index": c, "role": role, **desc(delta[:, c, :])})
    flat = delta.ravel()
    inds = np.argsort(flat)[-min(top_k, flat.size):][::-1]
    largest = []
    for idx in inds:
        batch, channel, position = np.unravel_index(int(idx), delta.shape)
        largest.append({"channel": int(channel), "position": int(position),
                        "reference": float(a[batch, channel, position]),
                        "onnx": float(b[batch, channel, position]),
                        "absolute_error": float(flat[idx])})
    box_max = float(np.max(delta[:, :4, :]))
    class_max = float(np.max(delta[:, 4:, :]))
    return {"tensor_shape": list(a.shape), "reference_dtype": str(a.dtype),
            "onnx_dtype": str(b.dtype), "original_absolute_tolerance": tolerance,
            "overall": desc(delta), "box_max_abs": box_max,
            "class_max_abs": class_max, "channels": channels, "largest_errors": largest,
            "raw_passed": bool(delta.max() <= tolerance)}


def verify_cohort_manifest(image_root: Path, cohort_file: Path, min_images: int = 50) -> list[Path]:
    """A frozen JSON list [{file: RELATIVE_PATH, sha256: HEX}] prevents selection drift."""
    image_root = Path(image_root).resolve()
    cohorts = json.loads(Path(cohort_file).read_text(encoding="utf-8"))
    if not isinstance(cohorts, list) or len(cohorts) < min_images:
        raise ValueError(f"Frozen parity cohort must include at least {min_images} images")
    paths, seen = [], set()
    for item in cohorts:
        if not isinstance(item, dict) or set(item) != {"file", "sha256"}:
            raise ValueError("Each cohort entry requires only file and sha256")
        rel = Path(item["file"])
        p = (image_root / rel).resolve()
        if rel.is_absolute() or (p != image_root and image_root not in p.parents):
            raise ValueError("Cohort image path escapes the image root")
        if rel.as_posix() in seen:
            raise ValueError(f"Duplicate cohort image: {rel}")
        seen.add(rel.as_posix())
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            raise ValueError(f"Invalid image type for cohort: {rel}")
        if not p.is_file() or sha256_file(p) != item["sha256"]:
            raise ValueError(f"Missing/modified frozen cohort image: {rel}")
        paths.append(p)
    return paths


def generate_cohort_manifest(image_root: Path, output: Path, n: int = 50) -> list[dict]:
    """Create a manifest once, not silently during an acceptance run."""
    root = Path(image_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace historical cohort: {output}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if n < 50 or len(paths) < n:
        raise ValueError("Need at least 50 representative images to freeze a parity cohort")
    records = [{"file": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
               for p in paths[:n]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
        fp.write("\n")
    return records
