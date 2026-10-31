"""M4 - full-body detection with YOLO11s: training, test mAP, threshold sweep, galleries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import save_json

NAMES = ["cat", "dog"]


def train(cfg: dict, data_yaml: Path, name: str = "yolo11s_det") -> Path:
    from ultralytics import YOLO

    dc = cfg["detection"]
    model = YOLO(dc["model"])
    model.train(data=str(data_yaml), epochs=dc["epochs"], imgsz=dc["imgsz"], batch=dc["batch"],
                seed=cfg["seed"], deterministic=True, project=str(Path(cfg["paths"]["runs"]) / "detect"),
                name=name, exist_ok=True, plots=True, verbose=False)
    return Path(model.trainer.best)


def test_metrics(weights: Path, data_yaml: Path, imgsz: int) -> dict:
    """Ultralytics COCO-style metrics on the held-out test split."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=imgsz, verbose=False,
                               plots=False)
    per_class = {NAMES[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr), "AP50_per_class": per_class}


# ------------------------------------------------------------- matching
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, iou_thr: float = 0.5):
    """Greedy class-aware matching by descending confidence. Returns TP flags and matched GT."""
    order = np.argsort(-pred_conf)
    ious = box_iou(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(pred_boxes), bool)
    for i in order:
        cand = np.where((gt_cls == pred_cls[i]) & ~used)[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= iou_thr:
            tp[i], used[j] = True, True
    return tp, used


def gt_for(df: pd.DataFrame) -> dict:
    return {r.image: (np.array([[r.x1, r.y1, r.x2, r.y2]]), np.array([r.label])) for r in df.itertuples()}


def predict_all(weights: Path, image_paths: list[Path], conf: float, iou: float, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for start in range(0, len(image_paths), 32):
        batch = [str(p) for p in image_paths[start:start + 32]]
        for p, r in zip(batch, model.predict(batch, conf=conf, iou=iou, imgsz=imgsz, verbose=False)):
            b = r.boxes
            out[Path(p).stem] = (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                 b.conf.cpu().numpy())
    return out


def score(preds: dict, gts: dict, conf: float) -> dict:
    tp = fp = fn = 0
    for name, (gb, gc) in gts.items():
        pb, pc, ps = preds.get(name, (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        keep = ps >= conf
        t, used = match(pb[keep], pc[keep], ps[keep], gb, gc)
        tp += int(t.sum())
        fp += int((~t).sum())
        fn += int((~used).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9)}


def threshold_sweep(cfg: dict, weights: Path, val_df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Sweep confidence x NMS IoU on validation only; the test split is never used to choose."""
    dc = cfg["detection"]
    gts = gt_for(val_df)
    paths = [image_dir / f"{n}.jpg" for n in gts]
    rows = []
    for iou in dc["iou_grid"]:
        preds = predict_all(weights, paths, conf=min(dc["conf_grid"]), iou=iou, imgsz=dc["imgsz"])
        for conf in dc["conf_grid"]:
            rows.append({"nms_iou": iou, "conf": conf, **score(preds, gts, conf)})
    return pd.DataFrame(rows)


def choose_operating_point(sweep: pd.DataFrame, min_recall: float = 0.80) -> dict:
    """Highest F1 among points meeting the recall floor (criterion A4); ties -> higher conf."""
    ok = sweep[sweep.recall >= min_recall]
    pool = ok if len(ok) else sweep
    best = pool.sort_values(["f1", "conf"], ascending=False).iloc[0]
    return {"conf": float(best.conf), "nms_iou": float(best.nms_iou),
            "val_precision": float(best.precision), "val_recall": float(best.recall),
            "val_f1": float(best.f1), "recall_floor_met": bool(len(ok))}


def recall_by_size(preds: dict, df: pd.DataFrame, conf: float) -> pd.DataFrame:
    """COCO size buckets on the GT box area (small < 32^2 <= medium < 96^2 <= large)."""
    rows = []
    for r in df.itertuples():
        area = (r.x2 - r.x1) * (r.y2 - r.y1)
        bucket = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
        pb, pc, ps = preds.get(r.image, (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        keep = ps >= conf
        _, used = match(pb[keep], pc[keep], ps[keep], np.array([[r.x1, r.y1, r.x2, r.y2]]),
                        np.array([r.label]))
        rows.append({"bucket": bucket, "found": bool(used[0])})
    return (pd.DataFrame(rows).groupby("bucket")["found"].agg(["size", "mean"])
            .rename(columns={"size": "n", "mean": "recall"}).reset_index())


def galleries(preds: dict, df: pd.DataFrame, image_dir: Path, conf: float, out_dir: Path,
              k: int = 24) -> dict:
    """False positives (red) and missed animals (green GT) at the operating point."""
    from PIL import ImageDraw

    fps, fns = [], []
    for r in df.itertuples():
        gb, gc = np.array([[r.x1, r.y1, r.x2, r.y2]]), np.array([r.label])
        pb, pc, ps = preds.get(r.image, (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        keep = ps >= conf
        t, used = match(pb[keep], pc[keep], ps[keep], gb, gc)
        for box, c, s in zip(pb[keep][~t], pc[keep][~t], ps[keep][~t]):
            fps.append((r.image, box, NAMES[c], float(s), gb[0]))
        if not used[0]:
            fns.append((r.image, None, r.species, 0.0, gb[0]))
    fps.sort(key=lambda x: -x[3])

    def sheet(items, path, title):
        tile, cols = 220, 6
        visible = items[:k]
        img = Image.new("RGB", (cols * tile, max(1, (len(visible) + cols - 1) // cols) * (tile + 16)), "white")
        d = ImageDraw.Draw(img)
        for i, (name, box, cls, s, gt) in enumerate(visible):
            im = Image.open(image_dir / f"{name}.jpg").convert("RGB")
            di = ImageDraw.Draw(im)
            w = max(2, im.width // 150)
            di.rectangle(list(gt), outline=(0, 200, 0), width=w)
            if box is not None:
                di.rectangle(list(box), outline=(255, 0, 0), width=w)
            im.thumbnail((tile, tile))
            x, y = (i % cols) * tile, (i // cols) * (tile + 16)
            img.paste(im, (x, y))
            d.text((x + 2, y + tile + 2), f"{title}: {cls} {s:.2f}", fill="black")
        img.save(path, quality=85)

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet(fps, out_dir / "det_false_positives.jpg", "FP")
    sheet(fns, out_dir / "det_false_negatives.jpg", "FN")
    return {"false_positives": len(fps), "false_negatives": len(fns)}


def run_test(cfg: dict, weights: Path, data_yaml: Path, df: pd.DataFrame, image_dir: Path,
             op: dict, out_dir: Path) -> dict:
    test_df = df[df.split == "test"]
    preds = predict_all(weights, [image_dir / f"{n}.jpg" for n in test_df.image],
                        conf=op["conf"], iou=op["nms_iou"], imgsz=cfg["detection"]["imgsz"])
    result = {
        "coco": test_metrics(weights, data_yaml, cfg["detection"]["imgsz"]),
        "operating_point": op,
        "test_at_operating_point": score(preds, gt_for(test_df), op["conf"]),
        "recall_by_size": recall_by_size(preds, test_df, op["conf"]).to_dict("records"),
        "galleries": galleries(preds, test_df, image_dir, op["conf"], out_dir / "figures"),
    }
    save_json(result, out_dir / "detection_test_metrics.json")
    return result


# ------------------------------------------------ additional analysis (added 24 Oct 2026)
# These functions read the trained weights and saved splits; none of them retrains a model.
def per_class_ap(weights: Path, data_yaml: Path, imgsz: int, split: str = "test") -> dict:
    """AP50 and AP50-95 per class from the Ultralytics COCO-style evaluation."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, verbose=False,
                               plots=False)
    out = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
    for i, c in enumerate(m.box.ap_class_index):
        out[NAMES[int(c)]] = {"AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])}
    return out


def _empty():
    return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)


def per_class_at_operating_point(preds: dict, gts: dict, conf: float) -> dict:
    """Precision, recall and F1 per class with the same greedy matching as `score`."""
    stats = {n: {"tp": 0, "fp": 0, "fn": 0} for n in NAMES}
    for name, (gb, gc) in gts.items():
        pb, pc, ps = preds.get(name, _empty())
        keep = ps >= conf
        t, used = match(pb[keep], pc[keep], ps[keep], gb, gc)
        for c, ok in zip(pc[keep], t):
            stats[NAMES[int(c)]]["tp" if ok else "fp"] += 1
        for c, u in zip(gc, used):
            if not u:
                stats[NAMES[int(c)]]["fn"] += 1
    for s in stats.values():
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        s.update(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))
    return stats


def error_breakdown(preds: dict, gts: dict, conf: float, loc_low: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every false positive and false negative at the operating point.

    FP kinds: class_confusion (IoU >= 0.5 with a GT of the other class), duplicate (IoU >= 0.5 with
    an already matched GT of the same class), localisation (0.1 <= IoU < 0.5, same class),
    localisation_and_class (0.1 <= IoU < 0.5, other class), background (IoU < 0.1).
    FN kinds: class_confusion, localisation (a kept box overlaps 0.1-0.5), missed (no kept box).
    """
    fp_rows, fn_rows = [], []
    for name, (gb, gc) in gts.items():
        pb, pc, ps = preds.get(name, _empty())
        keep = ps >= conf
        pb, pc, ps = pb[keep], pc[keep], ps[keep]
        t, used = match(pb, pc, ps, gb, gc)
        ious = box_iou(pb, gb)
        for i in np.where(~t)[0]:
            j = int(ious[i].argmax()) if len(gb) else -1
            best = float(ious[i, j]) if j >= 0 else 0.0
            same = j >= 0 and gc[j] == pc[i]
            if best >= 0.5:
                kind = "duplicate" if same else "class_confusion"
            elif best >= loc_low:
                kind = "localisation" if same else "localisation_and_class"
            else:
                kind = "background"
            fp_rows.append({"image": name, "pred": NAMES[int(pc[i])], "conf": float(ps[i]),
                            "gt": NAMES[int(gc[j])] if j >= 0 else None, "best_iou": best,
                            "kind": kind, "image_also_has_fn": bool((~used).any())})
        for j in np.where(~used)[0]:
            col = ious[:, j] if len(pb) else np.zeros(0)
            best = float(col.max()) if len(col) else 0.0
            i = int(col.argmax()) if len(col) else -1
            if best >= 0.5 and pc[i] != gc[j]:
                kind = "class_confusion"
            elif best >= loc_low:
                kind = "localisation"
            else:
                kind = "missed"
            fn_rows.append({"image": name, "gt": NAMES[int(gc[j])], "best_iou": best,
                            "pred_at_best": NAMES[int(pc[i])] if i >= 0 else None, "kind": kind,
                            "image_also_has_fp": bool((~t).any())})
    return pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)


def label_roundtrip(df: pd.DataFrame, labels_root: Path) -> pd.DataFrame:
    """Decode every exported YOLO label back to pixels and compare it with the source box."""
    rows = []
    for r in df.itertuples():
        f = labels_root / r.split / f"{r.image}.txt"
        if not f.exists():
            rows.append({"image": r.image, "split": r.split, "exists": False, "class_ok": False,
                         "max_err_px": np.nan})
            continue
        c, cx, cy, bw, bh = (float(v) for v in f.read_text().split()[:5])
        box = np.array([(cx - bw / 2) * r.width, (cy - bh / 2) * r.height,
                        (cx + bw / 2) * r.width, (cy + bh / 2) * r.height])
        err = float(np.abs(box - np.array([r.x1, r.y1, r.x2, r.y2])).max())
        rows.append({"image": r.image, "split": r.split, "exists": True,
                     "class_ok": int(c) == int(r.label), "max_err_px": err})
    return pd.DataFrame(rows)


def label_sheet(df: pd.DataFrame, labels_root: Path, image_dir: Path, path: Path, n: int = 24,
                seed: int = 42) -> None:
    """Contact sheet of exported YOLO labels drawn back on the images (visual verification)."""
    from PIL import ImageDraw

    sample = df.sample(n=min(n, len(df)), random_state=seed)
    tile, cols = 220, 6
    sheet = Image.new("RGB", (cols * tile, ((len(sample) + cols - 1) // cols) * (tile + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(sample.itertuples()):
        c, cx, cy, bw, bh = (float(v) for v in (labels_root / r.split / f"{r.image}.txt").read_text().split()[:5])
        im = Image.open(image_dir / f"{r.image}.jpg").convert("RGB")
        w, h = im.size
        ImageDraw.Draw(im).rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                                      (cy + bh / 2) * h], outline=(255, 0, 0), width=max(2, w // 150))
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 16)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + tile + 2), f"{r.split}: {NAMES[int(c)]} ({r.breed[:14]})", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def export_team_yolo(team_boxes: pd.DataFrame, team_images: Path, out: Path) -> Path:
    """YOLO-format copy of the team set (test split only) for Ultralytics evaluation."""
    import shutil

    import yaml

    (out / "images" / "test").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "test").mkdir(parents=True, exist_ok=True)
    cls = {n: i for i, n in enumerate(NAMES)}
    for name, g in team_boxes.groupby("image"):
        src = team_images / f"{name}.jpg"
        shutil.copy(src, out / "images" / "test" / src.name)
        w, h = Image.open(src).size
        lines = []
        for r in g.itertuples():
            cx, cy = (r.xtl + r.xbr) / 2 / w, (r.ytl + r.ybr) / 2 / h
            lines.append(f"{cls[r.label]} {cx:.6f} {cy:.6f} {(r.xbr - r.xtl) / w:.6f} {(r.ybr - r.ytl) / h:.6f}")
        (out / "labels" / "test" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    data = {"path": str(out.resolve()), "train": "images/test", "val": "images/test",
            "test": "images/test", "names": {i: n for i, n in enumerate(NAMES)}}
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out / "data.yaml"


def team_gts(team_boxes: pd.DataFrame) -> dict:
    cls = {n: i for i, n in enumerate(NAMES)}
    return {name: (g[["xtl", "ytl", "xbr", "ybr"]].to_numpy(float), g.label.map(cls).to_numpy())
            for name, g in team_boxes.groupby("image")}


def recall_by_size_multi(preds: dict, gts: dict, conf: float) -> pd.DataFrame:
    """Recall per COCO size bucket for images with several ground-truth boxes."""
    rows = []
    for name, (gb, gc) in gts.items():
        pb, pc, ps = preds.get(name, _empty())
        keep = ps >= conf
        _, used = match(pb[keep], pc[keep], ps[keep], gb, gc)
        for box, u in zip(gb, used):
            area = (box[2] - box[0]) * (box[3] - box[1])
            bucket = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
            rows.append({"bucket": bucket, "found": bool(u)})
    return (pd.DataFrame(rows).groupby("bucket")["found"].agg(["size", "mean"])
            .rename(columns={"size": "n", "mean": "recall"}).reset_index())


def video_inference(weights: Path, video: Path, out_mp4: Path, conf: float, iou: float, imgsz: int,
                    sheet_path: Path | None = None, sheet_frames: int = 8) -> dict:
    """Run the detector frame by frame on a video, write an annotated MP4 and a frame sheet."""
    import subprocess
    import time

    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_mp4.with_suffix(".raw.mp4")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    counts, species, times, frames = [], [], [], []
    for _ in range(3):  # warm-up on the first frame
        ok, first = cap.read()
        if ok:
            model.predict(first, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
        annotated = r.plot()
        writer.write(annotated)
        frames.append(annotated)
        counts.append(len(r.boxes))
        species += [NAMES[int(c)] for c in r.boxes.cls]
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "28", str(out_mp4)], check=True)
    raw.unlink()
    if sheet_path is not None and frames:
        idx = np.linspace(0, len(frames) - 1, sheet_frames).astype(int)
        tiles = [Image.fromarray(frames[i][..., ::-1]) for i in idx]
        for t_ in tiles:
            t_.thumbnail((320, 320))
        tw, th = max(t_.width for t_ in tiles), max(t_.height for t_ in tiles)
        sheet = Image.new("RGB", (4 * tw, ((len(tiles) + 3) // 4) * th), "white")
        for k, t_ in enumerate(tiles):
            sheet.paste(t_, ((k % 4) * tw, (k // 4) * th))
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path, quality=85)
    c = np.array(counts)
    return {"video": video.name, "frames": int(len(c)), "video_fps": float(fps), "size": [w, h],
            "conf": conf, "nms_iou": iou, "imgsz": imgsz,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "inference_ms": {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95))},
            "processing_fps": float(1000 / np.mean(times)),
            "frames_with_0": int((c == 0).sum()), "frames_with_1": int((c == 1).sum()),
            "frames_with_2_or_more": int((c >= 2).sum()),
            "detections_by_species": {n: species.count(n) for n in NAMES}}


def recall_by_relative_size(preds: dict, gts: dict, sizes: dict, conf: float) -> pd.DataFrame:
    """Recall by box area as a share of the image area: small < 1%, medium 1-10%, large >= 10%.

    The COCO pixel buckets put every box of a 2048 px phone photo in 'large', so the share of the
    frame is the informative size measure for the team set.
    """
    rows = []
    for name, (gb, gc) in gts.items():
        pb, pc, ps = preds.get(name, _empty())
        keep = ps >= conf
        _, used = match(pb[keep], pc[keep], ps[keep], gb, gc)
        w, h = sizes[name]
        for box, u in zip(gb, used):
            share = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            bucket = "small (<1%)" if share < 0.01 else "medium (1-10%)" if share < 0.10 else "large (>=10%)"
            rows.append({"bucket": bucket, "found": bool(u)})
    return (pd.DataFrame(rows).groupby("bucket")["found"].agg(["size", "mean"])
            .rename(columns={"size": "n", "mean": "recall"}).reset_index())
