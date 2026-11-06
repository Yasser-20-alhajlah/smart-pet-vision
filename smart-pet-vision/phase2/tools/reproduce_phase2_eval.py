#!/usr/bin/env python3
"""Independent reproduction of the Phase 2 test metrics from the delivered checkpoints.

A teammate other than the author of the runs executes this on a clean clone with the pinned
environment, the delivered weights and the Oxford-IIIT Pet test split (Phase 1 feedback, Part 13).
It re-evaluates the detector and the segmenter exactly as the reported numbers were produced
(Ultralytics ``val`` on ``split=test`` at 640 px: ``spv.detect.test_metrics`` and
``spv.segment.yolo_coco_metrics``; mIoU/Dice through ``spv.segment.yolo_masks`` + ``evaluate``
at the served mask confidence) and writes the reproduced numbers **side by side** with the
originals from ``report/evidence/``, the absolute differences and the predefined tolerances, plus
the operator's name, the code provenance (commit, dirty state), weight hashes and environment.

Tolerances were fixed before any reproduction: 0.005 absolute for mAP50 / mAP50-95 / mask mAP
(Ultralytics evaluation is deterministic up to GPU kernel differences), 0.005 for mIoU / Dice.
"Within tolerance" is reported per metric; the script never edits the originals.

This script does not train anything. A training reproduction (notebooks 03 and 04 on a T4,
about one hour each) is a separate step described in docs/INDEPENDENT_REPRODUCTION_PROCEDURE.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2 / "src"))

TOLERANCE = {"mAP50": 0.005, "mAP50_95": 0.005, "mask_mAP50": 0.005, "mask_mAP50_95": 0.005,
             "mIoU": 0.005, "mean_dice": 0.005}


def side_by_side(original: dict, reproduced: dict, tolerance: dict = TOLERANCE) -> dict:
    """Per-metric original, reproduced, difference and tolerance verdict."""
    rows = {}
    for k, tol in tolerance.items():
        if k in original and k in reproduced and original[k] is not None and reproduced[k] is not None:
            diff = float(reproduced[k]) - float(original[k])
            rows[k] = {"original": float(original[k]), "reproduced": float(reproduced[k]),
                       "difference": diff, "tolerance": tol, "within_tolerance": abs(diff) <= tol}
    rows["all_within_tolerance"] = all(r["within_tolerance"] for r in rows.values()) if rows else None
    return rows


def load_originals(evidence: Path) -> dict:
    det = json.loads((evidence / "M4_detection_test_metrics.json").read_text())["coco"]
    seg_coco = json.loads((evidence / "M5_yolo_seg_coco.json").read_text())
    seg = json.loads((evidence / "M5_yolo_seg_test_metrics.json").read_text())
    return {"mAP50": det["mAP50"], "mAP50_95": det["mAP50_95"],
            "mask_mAP50": seg_coco["mask_mAP50"], "mask_mAP50_95": seg_coco["mask_mAP50_95"],
            "mIoU": seg["mIoU"], "mean_dice": seg["mean_dice"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--operator", required=True, help="full name of the person executing this run")
    ap.add_argument("--det-weights", required=True, type=Path)
    ap.add_argument("--seg-weights", required=True, type=Path)
    ap.add_argument("--det-yaml", required=True, type=Path)
    ap.add_argument("--seg-yaml", required=True, type=Path)
    ap.add_argument("--processed", type=Path, help="processed data root with masks (for mIoU/Dice)")
    ap.add_argument("--metadata", type=Path, help="runs/data/metadata.csv (split v1) for mIoU/Dice")
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--wandb-run", default=None, help="W&B run id if the run was logged")
    ap.add_argument("--evidence", type=Path, default=PHASE2 / "report" / "evidence")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from spv import detect, segment
    from spv.provenance import file_sha256, record

    originals = load_originals(a.evidence)
    reproduced: dict = {}
    det = detect.test_metrics(a.det_weights, a.det_yaml, a.imgsz)
    reproduced.update({"mAP50": det["mAP50"], "mAP50_95": det["mAP50_95"], "AP50_per_class": det["AP50_per_class"]})
    seg_coco = segment.yolo_coco_metrics(a.seg_weights, a.seg_yaml, a.imgsz)
    reproduced.update({"mask_mAP50": seg_coco["mask_mAP50"], "mask_mAP50_95": seg_coco["mask_mAP50_95"]})
    if a.processed and a.metadata and a.image_dir:
        import pandas as pd
        df = pd.read_csv(a.metadata)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        masks = segment.yolo_masks(a.seg_weights, [a.image_dir / f"{im}.jpg" for im in test_df["image"]],
                                   a.seg_conf, a.imgsz)
        summary, _ = segment.evaluate(masks, test_df, a.processed)
        reproduced.update({"mIoU": summary["mIoU"], "mean_dice": summary["mean_dice"]})
    comparison = side_by_side(originals, reproduced)
    out = {"operator": a.operator, "executed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": " ".join(sys.argv), "wandb_run": a.wandb_run,
           "provenance": record(PHASE2, extra={"split_version": "v1 (runs/data/metadata.csv)",
                                                "det_weights_sha256": file_sha256(a.det_weights),
                                                "seg_weights_sha256": file_sha256(a.seg_weights),
                                                "imgsz": a.imgsz, "seg_conf": a.seg_conf}),
           "originals": originals, "reproduced": reproduced, "comparison": comparison,
           "scope": "evaluation of the delivered checkpoints on the Oxford test split; not a training reproduction"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(comparison, indent=1))
    return 0 if comparison.get("all_within_tolerance") else 1


if __name__ == "__main__":
    sys.exit(main())
