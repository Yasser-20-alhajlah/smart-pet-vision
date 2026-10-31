#!/usr/bin/env python3
"""Fail-closed Stage 2 ONNX gate against the approved frozen Drive source.

Command runs with model dependencies + 50+ representative frozen image files.
No original PT/ONNX or old outputs are modified; failure always leaves evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from onnx_parity_diagnostics import (TOLERANCE, sha256_file, tensor_diagnostics,
                                     verify_cohort_manifest, verify_frozen_snapshot)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True, help="Frozen JSON file+sha256 manifest")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, absent output folder")
    parser.add_argument("--random-inputs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-images", type=int, default=50)
    a = parser.parse_args(argv)
    if a.output.exists():
        parser.error("Output directory exists. Previous evidence must never be overwritten")
    if a.min_images < 50 or a.random_inputs < 4:
        parser.error("Require at least 50 frozen real images and four random inputs")
    for attr in ("repo", "weights", "onnx", "image_root", "cohort", "baseline"):
        if not getattr(a, attr).exists():
            parser.error(f"Missing {attr}: {getattr(a, attr)}")
    return a


def run(argv=None):
    a = parse_args(argv)
    out = a.output
    record = {"status": "INCOMPLETE", "original_tolerance": TOLERANCE,
              "autonomous_publication_allowed": False,
              "source_is_original_phase1": False, "runtime": {"python": platform.python_version()}}
    # Reserve output immediately so even import/runtime errors are written as evidence.
    out.mkdir(parents=True, exist_ok=False)
    result_file = out / "onnx_stage2_gate.json"
    try:
        baseline = json.loads(a.baseline.read_text(encoding="utf-8"))
        checks = verify_frozen_snapshot(a.repo, baseline["source_snapshot"])
        record["source_snapshot"] = checks
        if not checks["passed"]:
            raise ValueError("Approved source SHA mismatch; gate blocked")
        images = verify_cohort_manifest(a.image_root, a.cohort, a.min_images)
        record["cohort"] = {"n": len(images), "sha256": sha256_file(a.cohort),
                            "images": [p.relative_to(a.image_root.resolve()).as_posix() for p in images]}
        expected = baseline["weights"]
        record["weights_sha256"] = sha256_file(a.weights)
        record["onnx_sha256"] = sha256_file(a.onnx)
        # The approved model hashes must match the trusted Drive assets.
        for k, actual in (("best.pt", record["weights_sha256"]),
                          ("best.onnx.bin", record["onnx_sha256"])):
            info = expected.get(k)
            e = info.get("sha256") if isinstance(info, dict) else None
            if not e or actual != e:
                raise ValueError(f"{k} changed versus approved Drive asset: {e} != {actual}")
        # Always evaluate the exact source under review; never import a different spv installation.
        sys.path.insert(0, str(a.repo.resolve() / "src"))
        import numpy as np
        import torch
        import onnxruntime as ort
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        from PIL import Image
        from spv.deploy import yolo_parity_real
        record["runtime"].update(torch=torch.__version__, onnxruntime=ort.__version__,
                                  python=platform.python_version(), cpu_only=True)
        torch.manual_seed(a.seed)
        net = YOLO(str(a.weights)).model.float().cpu().eval()
        ort_session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
        ort_input = ort_session.get_inputs()[0]
        record["model_contract"] = {"onnx_input": ort_input.name, "shape": ort_input.shape,
                                    "onnx_output_shape": ort_session.get_outputs()[0].shape}
        input_name = ort_input.name
        size = 640
        random_runs = []
        for _ in range(a.random_inputs):
            x = torch.rand((1,3,size,size), dtype=torch.float32)
            with torch.no_grad():
                ref = net(x)
            ref = (ref[0] if isinstance(ref, (list,tuple)) else ref).numpy()
            candidate = ort_session.run(None, {input_name: x.numpy()})[0]
            random_runs.append(tensor_diagnostics(ref, candidate))
        record["random"] = {"seed": a.seed, "n": a.random_inputs,
                            "max_abs": max(x["overall"]["max"] for x in random_runs),
                            "per_run": random_runs}
        lb = LetterBox((size, size), auto=False)
        real_runs = []
        for path in images:
            rgb = np.array(Image.open(path).convert("RGB"))
            arr = lb(image=rgb)
            x = torch.from_numpy(arr.transpose(2,0,1).copy()).float().unsqueeze(0)/255
            with torch.no_grad():
                ref = net(x)
            ref = (ref[0] if isinstance(ref, (list,tuple)) else ref).numpy()
            candidate = ort_session.run(None, {input_name: x.numpy()})[0]
            summary = tensor_diagnostics(ref, candidate)
            # Report numeric aggregates without leaking private image pixels.
            real_runs.append({"image": path.relative_to(a.image_root.resolve()).as_posix(),
                              "image_sha256": sha256_file(path),
                              "overall": summary["overall"], "box_max_abs": summary["box_max_abs"],
                              "class_max_abs": summary["class_max_abs"],
                              "raw_passed": summary["raw_passed"],
                              "top_errors": summary["largest_errors"][:3]})
        record["real"] = {"n": len(real_runs), "max_abs": max(x["overall"]["max"] for x in real_runs),
                          "per_image": real_runs}
        # Final predicted boxes are a separate parity condition, not a replacement for raw parity.
        final = yolo_parity_real(a.weights, a.onnx, images, size, conf=.40, iou=.50)
        record["post_nms"] = final
        record["raw_passed"] = (record["random"]["max_abs"] <= TOLERANCE and
                                record["real"]["max_abs"] <= TOLERANCE)
        record["final_passed"] = (final["same_number_of_detections"] == 1.0 and
                                  final["same_classes"] == 1.0 and
                                  final["final_box_max_abs_diff_px"] <= TOLERANCE)
        record["status"] = "PASS" if record["raw_passed"] and record["final_passed"] else "FAIL"
        record["reason"] = ("RAW_AND_FINAL_PARITY_PASS" if record["status"] == "PASS"
                            else "RAW_OR_POST_NMS_PARITY_FAILURE")
    except Exception as exc:
        record["status"] = "BLOCKED"
        record["error"] = {"type": type(exc).__name__, "detail": str(exc)}
    finally:
        record["onnx_candidate_parity_passed"] = record["status"] == "PASS"
        record["onnx_deployment_allowed"] = False  # Deployment also requires independent count-policy and T4 gates.
        result_file.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{record['status']}: evidence at {result_file}")
    return 0 if record["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run())
