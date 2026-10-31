"""Fail-fast checks run before any real-weight validation or HTTP test.

Does not import torch or load models; it only reads files, so it finishes in seconds.
Raises PreflightError with a specific message on the first problem found.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

# Weights evaluated in the Phase 2 report (team record; W&B f8manl0i config)
EXPECTED_SHA256 = {
    "detector": "f48e4d8ea5c925c18a4b43744c2f697bfc01a0a473492275f000848125da1cd4",
    "segmenter": "868f3842bf0e05c8b936d3e51e6e81b7f44fdcf12f7b666e5041bf1d47a5d8ee",
    "classifier": "22dd58d471f2dbb39eef10331f208d668a569b1feae378171df8074fbe2081ba",
}
OPERATING_POINT = {"detector_conf": 0.4, "detector_iou": 0.5}


class PreflightError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_requirements(requirements):
    """Compare installed versions with the pinned file; returns a list of mismatches."""
    bad = []
    for line in Path(requirements).read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, want = line.split("==")
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            bad.append(f"{name}: not installed (pinned {want})")
            continue
        if have.split("+")[0] != want:
            bad.append(f"{name}: {have} installed, pinned {want}")
    return bad


def run(manifest, images=None, labels=None, output=None, expected=EXPECTED_SHA256,
        requirements=None, strict_versions=False):
    from spv.pipeline import Manifest

    manifest = Path(manifest)
    if not manifest.is_file():
        raise PreflightError(f"Manifest not found: {manifest}")
    try:
        m = Manifest.load(manifest)
    except (ValueError, KeyError, TypeError) as e:
        raise PreflightError(f"Manifest is invalid: {e}") from e
    if not m.detector.endswith(".pt"):
        raise PreflightError("The service must use the PyTorch detector (.pt), not ONNX.")
    for key, want in OPERATING_POINT.items():
        if getattr(m, key) != want:
            raise PreflightError(f"{key}={getattr(m, key)} differs from the reported operating point {want}.")
    if m.review_mode != "off":
        raise PreflightError("Baseline validation expects review_mode 'off'; test other modes separately.")
    hashes = {}
    for key in ("detector", "segmenter", "classifier"):
        path = Path(getattr(m, key))
        if not path.is_file():
            raise PreflightError(f"{key} weights not found: {path}")
        hashes[key] = sha256(path)
        if expected and expected.get(key) and hashes[key] != expected[key]:
            raise PreflightError(f"{key} weights differ from the evaluated file (sha256 {hashes[key][:12]}...).")
    if output is not None and Path(output).exists():
        raise PreflightError(f"Output folder already exists, choose a new one: {output}")
    n_images = None
    if images is not None:
        paths = sorted(Path(images).glob("*.jpg"))
        if not paths:
            raise PreflightError(f"No JPEG images in {images}")
        n_images = len(paths)
        if labels is not None:
            import pandas as pd
            from validate_release import validate_label_coverage
            try:
                validate_label_coverage(paths, pd.read_csv(labels))
            except ValueError as e:
                raise PreflightError(f"Labels do not match images: {e}") from e
    mismatches = check_requirements(requirements) if requirements else []
    if mismatches and strict_versions:
        raise PreflightError("Pinned versions differ: " + "; ".join(mismatches))
    return {"weights_sha256": hashes, "n_images": n_images, "version_mismatches": mismatches}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--output")
    ap.add_argument("--requirements", default=str(ROOT.parent / "requirements.txt"))
    ap.add_argument("--strict-versions", action="store_true")
    a = ap.parse_args()
    try:
        print(json.dumps(run(a.manifest, a.images, a.labels, a.output,
                             requirements=a.requirements, strict_versions=a.strict_versions), indent=2))
    except PreflightError as e:
        print("PREFLIGHT FAILED:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
