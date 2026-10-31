"""M5 - body segmentation: SAM baseline vs. fine-tuned YOLO11s-seg; mIoU and Dice."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import save_json


def train(cfg: dict, data_yaml: Path, name: str = "yolo11s_seg") -> Path:
    from ultralytics import YOLO

    sc = cfg["segmentation"]
    model = YOLO(sc["model"])
    model.train(data=str(data_yaml), epochs=sc["epochs"], imgsz=sc["imgsz"], batch=sc["batch"],
                seed=cfg["seed"], deterministic=True,
                project=str(Path(cfg["paths"]["runs"]) / "segment"), name=name, exist_ok=True,
                plots=True, verbose=False)
    return Path(model.trainer.best)


def mask_metrics(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> dict:
    """Binary IoU and Dice. `ignore` = trimap border pixels, excluded when given."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Prediction and ground truth must be aligned 2D masks.")
    if ignore is not None:
        ignore = np.asarray(ignore, dtype=bool)
        if ignore.shape != gt.shape:
            raise ValueError("Ignore mask must match ground-truth dimensions.")
        keep = ~ignore
        if not keep.any():
            raise ValueError("No evaluable pixels remain after ignoring the border.")
        pred, gt = pred[keep], gt[keep]
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return {"iou": float(inter / union) if union else 1.0,
            "dice": float(2 * inter / total) if total else 1.0}


def load_gt(processed: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    gt = np.array(Image.open(processed / "masks" / f"{name}.png")) > 0
    border = np.array(Image.open(processed / "borders" / f"{name}.png")) > 0
    return gt, border


def yolo_masks(weights: Path, image_paths: list[Path], conf: float, imgsz: int) -> dict:
    """Union of predicted instance masks per image at the original resolution."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for start in range(0, len(image_paths), 16):
        batch = [str(p) for p in image_paths[start:start + 16]]
        results = list(model.predict(batch, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False))
        if len(results) != len(batch):
            raise ValueError("Segmentation results do not cover the entire image batch.")
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


def sam_masks(df: pd.DataFrame, image_dir: Path, sam_weights: str = "sam2.1_b.pt") -> dict:
    """SAM prompted with the trimap-derived body box (baseline and annotation helper)."""
    from ultralytics import SAM

    model = SAM(sam_weights)
    out = {}
    for r in df.itertuples():
        res = model(str(image_dir / f"{r.image}.jpg"), bboxes=[[r.x1, r.y1, r.x2, r.y2]],
                    verbose=False)[0]
        h, w = res.orig_shape
        with Image.open(image_dir / f"{r.image}.jpg") as image:
            if (image.height, image.width) != (h, w):
                raise ValueError(f"Unaligned SAM image coordinates for {r.image}.")
        if res.masks is None:
            out[r.image] = np.zeros((h, w), bool)
        else:
            masks = res.masks.data.cpu().numpy().astype(bool)
            if masks.ndim != 3 or masks.shape[1:] != (h, w):
                raise ValueError(f"Unaligned SAM masks for {r.image}.")
            out[r.image] = masks.any(0)
    return out


def evaluate(masks: dict, df: pd.DataFrame, processed: Path) -> tuple[dict, pd.DataFrame]:
    if df.empty or df.image.duplicated().any():
        raise ValueError("Mask evaluation requires nonempty, unique image IDs.")
    rows = []
    for r in df.itertuples():
        gt, border = load_gt(processed, r.image)
        if r.image not in masks:
            raise ValueError(f"Missing prediction record for {r.image}.")
        pred = masks[r.image]
        # A size mismatch may be letterboxing/coordinate failure, not a scale change.
        # Preserve the failure rather than silently resizing it into a valid score.
        if pred.shape != gt.shape:
            raise ValueError(f"Unaligned prediction for {r.image}: {pred.shape} != {gt.shape}.")
        full = mask_metrics(pred, gt)
        core = mask_metrics(pred, gt, border)
        rows.append({"image": r.image, "species": r.species, "breed": r.breed,
                     "iou": full["iou"], "dice": full["dice"],
                     "iou_no_border": core["iou"], "dice_no_border": core["dice"],
                     "pred_fraction": float(pred.mean()), "gt_fraction": float(gt.mean())})
    per_image = pd.DataFrame(rows)
    summary = {
        "mIoU": float(per_image.iou.mean()), "mean_dice": float(per_image.dice.mean()),
        "mIoU_no_border": float(per_image.iou_no_border.mean()),
        "mean_dice_no_border": float(per_image.dice_no_border.mean()),
        "by_species": per_image.groupby("species")[["iou", "dice"]].mean().to_dict("index"),
        "worst_breeds": (per_image.groupby("breed").iou.mean().sort_values().head(5).to_dict()),
        "n": int(len(per_image)),
    }
    return summary, per_image


def failure_gallery(per_image: pd.DataFrame, masks: dict, image_dir: Path, processed: Path,
                    path: Path, k: int = 18) -> None:
    """Lowest-IoU images: red = prediction only, green = GT only, yellow = agreement."""
    from PIL import ImageDraw

    worst = per_image.sort_values("iou").head(k)
    tile, cols = 220, 6
    sheet = Image.new("RGB", (cols * tile, ((len(worst) + cols - 1) // cols) * (tile + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(worst.itertuples()):
        im = np.array(Image.open(image_dir / f"{r.image}.jpg").convert("RGB")).astype(float)
        gt, _ = load_gt(processed, r.image)
        pred = masks[r.image]
        overlay = im.copy()
        overlay[pred & ~gt] = [255, 0, 0]
        overlay[gt & ~pred] = [0, 200, 0]
        overlay[gt & pred] = [255, 220, 0]
        blend = Image.fromarray((0.55 * im + 0.45 * overlay).astype(np.uint8))
        blend.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 16)
        sheet.paste(blend, (x, y))
        d.text((x + 2, y + tile + 2), f"{r.breed[:16]} IoU {r.iou:.2f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def run_test(name: str, masks: dict, test_df: pd.DataFrame, image_dir: Path, processed: Path,
             out_dir: Path) -> dict:
    summary, per_image = evaluate(masks, test_df, processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out_dir / f"{name}_per_image.csv", index=False)
    failure_gallery(per_image, masks, image_dir, processed, out_dir / "figures" / f"{name}_failures.jpg")
    save_json(summary, out_dir / f"{name}_test_metrics.json")
    return summary


def yolo_coco_metrics(weights: Path, data_yaml: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=imgsz, verbose=False,
                               plots=False)
    return {"mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map),
            "box_mAP50": float(m.box.map50)}


def gt_overlay_sheet(df: pd.DataFrame, image_dir: Path, processed: Path, path: Path, n: int = 24,
                     seed: int = 42) -> None:
    """Visual check of the ground-truth masks (added 24 Oct 2026).

    Green = trimap foreground, yellow = ambiguous border band; the label under each tile is the breed.
    """
    from PIL import ImageDraw

    sample = df.sample(n=min(n, len(df)), random_state=seed)
    tile, cols = 220, 6
    sheet = Image.new("RGB", (cols * tile, ((len(sample) + cols - 1) // cols) * (tile + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(sample.itertuples()):
        im = np.array(Image.open(image_dir / f"{r.image}.jpg").convert("RGB")).astype(float)
        gt, border = load_gt(processed, r.image)
        overlay = im.copy()
        overlay[gt & ~border] = [0, 200, 0]
        overlay[border] = [255, 220, 0]
        blend = Image.fromarray((0.55 * im + 0.45 * overlay).astype(np.uint8))
        blend.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 16)
        sheet.paste(blend, (x, y))
        d.text((x + 2, y + tile + 2), f"{r.split}: {r.breed[:20]}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)
