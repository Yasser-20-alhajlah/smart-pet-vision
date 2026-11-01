"""Phase 2 follow-up experiments (validation selection, baselines, per-stage timing).

Used by notebooks/11_phase2_gap_experiments.ipynb. Every choice is made on the validation split;
the test split is run once with the settings chosen on validation. The only training is the
optional second detector recipe in the notebook.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# COCO class ids of the two animals in the pretrained (not fine-tuned) Ultralytics models.
COCO_TO_OURS = {15: 0, 16: 1}  # cat, dog
OURS_TO_COCO = {v: k for k, v in COCO_TO_OURS.items()}


# ----------------------------------------------------------------- datasets
def remap_yolo_dataset(src_base: Path, out_base: Path, names: dict, class_map: dict = OURS_TO_COCO) -> Path:
    """Copy a YOLO dataset (labels rewritten, images linked) with class ids mapped to COCO ids.

    A COCO-pretrained model can then be scored with the standard Ultralytics metrics on our
    labels; Ultralytics reports only classes present in the labels (cat and dog).
    """
    import yaml

    src_base, out_base = Path(src_base), Path(out_base)
    if out_base.exists():
        raise FileExistsError(f"Choose a new output folder: {out_base}")
    found = False
    for split in ("train", "val", "test"):
        src_lab, src_img = src_base / "labels" / split, src_base / "images" / split
        if not src_lab.is_dir():
            continue
        found = True
        (out_base / "labels" / split).mkdir(parents=True)
        (out_base / "images" / split).mkdir(parents=True)
        # Per-file hard links (copy as fallback), not a linked folder: Ultralytics resolves a linked
        # folder to its real path and would then read the original, unmapped labels.
        for img in src_img.iterdir():
            dst = out_base / "images" / split / img.name
            try:
                os.link(img, dst)
            except OSError:
                shutil.copy2(img, dst)
        for f in src_lab.glob("*.txt"):
            lines = []
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                cls, rest = line.split(maxsplit=1)
                if int(cls) not in class_map:
                    raise ValueError(f"Unexpected class {cls} in {f}")
                lines.append(f"{class_map[int(cls)]} {rest}")
            (out_base / "labels" / split / f.name).write_text("\n".join(lines) + "\n")
    if not found:
        raise FileNotFoundError(f"No YOLO label folders under {src_base}")
    data = {"path": str(out_base.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {int(k): v for k, v in names.items()}}
    with open(out_base / "data.yaml", "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out_base / "data.yaml"


def ultralytics_val(model_path, data_yaml: Path, split: str, imgsz: int, classes=None) -> dict:
    """Standard Ultralytics validation (evaluation conf 0.001): box and, if present, mask mAP."""
    from ultralytics import YOLO

    m = YOLO(str(model_path)).val(data=str(data_yaml), split=split, imgsz=imgsz, classes=classes,
                                  verbose=False, plots=False)
    out = {"box_mAP50": float(m.box.map50), "box_mAP50_95": float(m.box.map)}
    seg = getattr(m, "seg", None)
    if seg is not None:
        out.update(mask_mAP50=float(seg.map50), mask_mAP50_95=float(seg.map))
    return out


def predict_mapped(model_path, image_paths: list, conf: float, iou: float, imgsz: int,
                   class_map: dict | None = None) -> dict:
    """Predictions in the format of spv.detect.predict_all; COCO ids mapped to ours if given."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    keep = list(class_map) if class_map else None
    out = {}
    for start in range(0, len(image_paths), 32):
        batch = [str(p) for p in image_paths[start:start + 32]]
        for p, r in zip(batch, model.predict(batch, conf=conf, iou=iou, imgsz=imgsz, classes=keep,
                                             verbose=False)):
            b = r.boxes
            cls = b.cls.cpu().numpy().astype(int)
            if class_map:
                cls = np.array([class_map[int(c)] for c in cls], dtype=int)
            out[Path(p).stem] = (b.xyxy.cpu().numpy(), cls, b.conf.cpu().numpy())
    return out


def sweep_from_predictions(preds: dict, gts: dict, confs, nms_iou: float) -> pd.DataFrame:
    from spv.detect import score

    return pd.DataFrame([{"nms_iou": nms_iou, "conf": c, **score(preds, gts, c)} for c in confs])


# ------------------------------------------------------------ segmentation
def yolo_masks_classes(model_path, image_paths: list, conf: float, imgsz: int, classes=None) -> dict:
    """Union of predicted instance masks per image, optionally keeping only some classes.

    Same output as spv.segment.yolo_masks; the class filter is needed for COCO-pretrained models,
    which also segment people and objects.
    """
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    out = {}
    for start in range(0, len(image_paths), 16):
        batch = [str(p) for p in image_paths[start:start + 16]]
        results = list(model.predict(batch, conf=conf, imgsz=imgsz, classes=classes, retina_masks=True,
                                     verbose=False))
        for p, r in zip(batch, results):
            h, w = r.orig_shape
            with Image.open(p) as image:
                if (image.height, image.width) != (h, w):
                    raise ValueError(f"Unaligned original image coordinates for {p}.")
            m = np.zeros((h, w), bool)
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy().astype(bool)
                if masks.ndim != 3 or masks.shape[1:] != (h, w):
                    raise ValueError(f"Unaligned native-resolution masks for {p}.")
                m = masks.any(0)
            out[Path(p).stem] = m
    return out


def seg_conf_sweep(weights, df: pd.DataFrame, image_dir: Path, processed: Path, confs, imgsz: int,
                   classes=None) -> pd.DataFrame:
    from spv.segment import evaluate

    paths = [Path(image_dir) / f"{n}.jpg" for n in df.image]
    rows = []
    for c in confs:
        masks = yolo_masks_classes(weights, paths, conf=c, imgsz=imgsz, classes=classes)
        summary, _ = evaluate(masks, df, Path(processed))
        rows.append({"conf": c, "mIoU": summary["mIoU"], "mean_dice": summary["mean_dice"],
                     "mIoU_no_border": summary["mIoU_no_border"], "n": summary["n"]})
    return pd.DataFrame(rows)


def choose_seg_conf(sweep: pd.DataFrame) -> float:
    """Highest validation mean IoU; ties go to the higher confidence."""
    if sweep.empty:
        raise ValueError("Empty sweep.")
    return float(sweep.sort_values(["mIoU", "conf"], ascending=False).iloc[0].conf)


# ----------------------------------------------------------------- pipeline
def run_pipeline(manifest_path: Path, image_paths: list, overrides: dict | None = None, warmup: int = 5) -> pd.DataFrame:
    """Run the full pipeline as the service does; keep verdict, reasons and stage timings."""
    import dataclasses
    from PIL import Image
    from spv.pipeline import Manifest, PetPipeline

    m = Manifest.load(manifest_path)
    if overrides:
        m = dataclasses.replace(m, **overrides)
    pipe = PetPipeline(m)
    for p in image_paths[:warmup]:
        pipe.analyze(Image.open(p), with_images=False)
    rows = []
    for p in image_paths:
        r = pipe.analyze(Image.open(p), with_images=False)
        rows.append({"image": Path(p).stem, "verdict": r["verdict"],
                     "reasons": ";".join(x["code"] for x in r["reasons"]),
                     "n_detected": len(r["detections"]), "species": r["species"],
                     "mask_fraction": r["mask_fraction"],
                     "segmentation_matched": r["segmentation_matched"], **r["timings"]})
    return pd.DataFrame(rows)


def stage_percentiles(records: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in records.columns if c.endswith("_ms")]
    if not cols:
        raise ValueError("No timing columns found.")
    out = []
    for c in cols:
        v = records[c].dropna().astype(float)
        out.append({"stage": c, "p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)),
                    "mean_ms": float(v.mean()), "n": int(len(v))})
    return pd.DataFrame(out)


def min_mask_fraction_sweep(records: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Images that would be referred as 'animal too small' at each threshold (matched masks only).

    Every Oxford validation image holds one large animal, so each such referral is a false alarm.
    """
    frac = records.mask_fraction.astype(float)
    matched = records.segmentation_matched.astype(bool)
    return pd.DataFrame([{"min_mask_fraction": t, "flagged_too_small": int(((frac < t) & matched).sum()),
                          "share": float(((frac < t) & matched).mean())} for t in thresholds])


def referral_summary(records: pd.DataFrame) -> dict:
    reasons = records.reasons.fillna("").str.split(";").explode()
    reasons = reasons[reasons != ""]
    return {"n": int(len(records)), "referred": int((records.verdict != "ready").sum()),
            "reason_counts": {k: int(v) for k, v in reasons.value_counts().items()},
            "unmatched": int((~records.segmentation_matched.astype(bool)).sum())}


def largest_within(table: pd.DataFrame, column: str, value_col: str, limit: float) -> float:
    """Largest setting whose rate stays within the limit (pre-stated selection rule)."""
    ok = table[table[value_col] <= limit]
    if ok.empty:
        raise ValueError(f"No {column} keeps {value_col} within {limit}.")
    return float(ok[column].max())
