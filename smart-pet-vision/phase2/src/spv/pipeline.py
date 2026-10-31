"""M8 - the integrated system: detect -> crop -> classify -> segment -> verdict."""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .constants import CLASSES
from .common import load_json, save_json
from .instance_alignment import best_mask_index

REASONS = {
    "no_animal": "No cat or dog was detected.",
    "multiple_animals": "More than one animal was detected.",
    "low_confidence": "The species prediction is below the verification threshold.",
    "disagreement": "The detector and the classifier disagree on the species.",
    "animal_too_small": "The animal covers too little of the frame; retake the photo.",
    "segmentation_unmatched": "The segmenter could not match a reliable animal mask to the selected detection.",
    "segmentation_disagreement": "The segmenter and the selected detector disagree on the animal species.",
}


@dataclass
class SegmentationEvidence:
    """One selected mask and independent animal evidence from the segmenter.

    Iteration preserves the public (mask, matched) unpacking contract.
    """
    mask: np.ndarray
    matched: bool
    animal_count: int
    selected_species: str | None = None

    def __iter__(self):
        yield self.mask
        yield self.matched


@dataclass
class Manifest:
    """Everything the service needs, written once by notebook 05."""
    detector: str
    segmenter: str
    classifier: str
    classifier_kind: str
    temperature: float
    detector_conf: float
    detector_iou: float
    verify_threshold: float
    min_mask_fraction: float
    image_size: int = 224
    imgsz: int = 640
    segmenter_conf: float = 0.25
    segmenter_iou: float = 0.70
    mask_match_iou: float = 0.10

    @classmethod
    def load(cls, path: str | Path, project_root: str | Path | None = None) -> "Manifest":
        path = Path(path).expanduser().resolve()
        data = load_json(path)
        root = Path(project_root).resolve() if project_root else (
            path.parent.parent if path.parent.name == "runs" else path.parent)
        for key in ("detector", "segmenter", "classifier"):
            value = Path(data[key]).expanduser()
            data[key] = str(value if value.is_absolute() else (root / value).resolve())
        manifest = cls(**data)
        if not np.isfinite(manifest.temperature) or manifest.temperature <= 0:
            raise ValueError("Manifest temperature must be finite and positive.")
        for name in ("detector_conf", "detector_iou", "verify_threshold", "min_mask_fraction",
                     "segmenter_conf", "segmenter_iou", "mask_match_iou"):
            value = getattr(manifest, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Manifest {name} must be between zero and one.")
        if manifest.mask_match_iou == 0:
            raise ValueError("Manifest mask_match_iou must be positive.")
        for name in ("image_size", "imgsz"):
            value = getattr(manifest, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Manifest {name} must be a positive integer.")
        if manifest.classifier_kind not in {"scratch", "frozen", "finetune"}:
            raise ValueError("Manifest classifier_kind is not supported.")
        return manifest

    def save(self, path: str | Path) -> None:
        save_json(self.__dict__, path)


class PetPipeline:
    def __init__(self, m: Manifest, cfg: dict | None = None):
        from ultralytics import YOLO
        from .classify import load_model, transforms

        self.m = m
        self.detector = YOLO(m.detector, task="detect")
        self.segmenter = YOLO(m.segmenter, task="segment")
        cfg = cfg or {"classification": {"backbone": "resnet50", "pretrained": False}}
        self.classifier = load_model(m.classifier_kind, cfg, Path(m.classifier))
        self.tf = transforms(m.image_size, train=False)
        self.device = next(self.classifier.parameters()).device

    # ---------------------------------------------------------- stages
    def detect(self, im: Image.Image):
        r = self.detector.predict(im, conf=self.m.detector_conf, iou=self.m.detector_iou,
                                  imgsz=self.m.imgsz, rect=False, verbose=False)[0]
        return (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy())

    def classify(self, crop: Image.Image) -> np.ndarray:
        import torch
        from .classify import softmax

        x = self.tf(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return softmax(self.classifier(x).float().cpu().numpy(), self.m.temperature)[0]

    def segment(self, im: Image.Image, selected_box: np.ndarray | None = None,
                selected_class: int | None = None) -> SegmentationEvidence:
        """Match one segmenter instance to the classified detection (never union all masks).

        Preserve the segmenter's independent animal count: selecting a single
        mask must not discard evidence of another animal missed by the detector.
        """
        r = self.segmenter.predict(im, imgsz=self.m.imgsz, retina_masks=True,
                                   conf=self.m.segmenter_conf, iou=self.m.segmenter_iou,
                                   rect=False, verbose=False)[0]
        h, w = im.height, im.width
        empty = np.zeros((h, w), dtype=bool)
        if r.boxes is None:
            return SegmentationEvidence(empty, False, 0)
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or len(classes) != len(boxes):
            return SegmentationEvidence(empty, False, 0)
        valid = (np.isfinite(boxes).all(axis=1)
                 & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
                 & np.isin(classes, range(len(CLASSES))))
        count = int(valid.sum())
        if tuple(r.orig_shape) != (h, w) or selected_box is None or r.masks is None or r.boxes is None:
            return SegmentationEvidence(empty, False, count)
        masks = r.masks.data.cpu().numpy().astype(bool)
        if len(masks) != len(boxes) or not valid.all():
            return SegmentationEvidence(empty, False, count)
        index = best_mask_index(boxes, selected_box, classes, selected_class,
                                min_iou=self.m.mask_match_iou)
        if index is None or index >= len(masks):
            return SegmentationEvidence(empty, False, count)
        mask = masks[index]
        # Don't silently remap pixel coordinates if the runtime changes mask size.
        if mask.shape != (h, w):
            return SegmentationEvidence(empty, False, count)
        return SegmentationEvidence(mask, True, count, CLASSES[int(classes[index])])

    # ------------------------------------------------------------ main
    def analyze(self, im: Image.Image, with_images: bool = True) -> dict:
        im = ImageOps.exif_transpose(im).convert("RGB")
        timings, t = {}, time.perf_counter()

        boxes, classes, confs = self.detect(im)
        timings["detect_ms"] = (time.perf_counter() - t) * 1e3
        reasons = []
        if len(boxes) == 0:
            reasons.append("no_animal")
            box = np.array([0, 0, im.width, im.height], float)
            det_species = None
            det_class = None
        else:
            if len(boxes) > 1:
                reasons.append("multiple_animals")
            k = int(confs.argmax())
            box, det_species = boxes[k], CLASSES[classes[k]]
            det_class = int(classes[k])

        t = time.perf_counter()
        crop = im.crop(tuple(int(round(v)) for v in box))
        probs = self.classify(crop)
        species = CLASSES[int(probs.argmax())]
        timings["classify_ms"] = (time.perf_counter() - t) * 1e3
        if probs.max() < self.m.verify_threshold:
            reasons.append("low_confidence")
        if det_species and det_species != species:
            reasons.append("disagreement")

        t = time.perf_counter()
        segmentation = self.segment(im,
                                         selected_box=box if det_class is not None else None,
                                         selected_class=det_class)
        mask, seg_matched = segmentation
        segmentation_count = getattr(segmentation, "animal_count", None)
        segmentation_species = getattr(segmentation, "selected_species", None)
        if segmentation_count is not None and segmentation_count > 1 and "multiple_animals" not in reasons:
            reasons.append("multiple_animals")
        if seg_matched and segmentation_species is not None and segmentation_species != det_species:
            reasons.append("segmentation_disagreement")
        if "no_animal" not in reasons and not seg_matched:
            reasons.append("segmentation_unmatched")
        timings["segment_ms"] = (time.perf_counter() - t) * 1e3
        fraction = float(mask.mean())
        if "no_animal" not in reasons and fraction < self.m.min_mask_fraction:
            reasons.append("animal_too_small")
        timings["total_ms"] = sum(timings.values())

        result = {
            "verdict": "ready" if not reasons else "refer_to_staff",
            # A readiness candidate is NOT permission to publish.
            # The current detector missed animals in multi-animal external photos.
            # Downstream clients must not bypass explicit staff approval.
            "staff_approval_required": True,
            "automatic_publication_allowed": False,
            "reasons": [{"code": r, "message": REASONS[r]} for r in reasons],
            "species": species,
            "confidence": float(probs.max()),
            "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
            "detections": [{"box_xyxy": [float(v) for v in b], "species": CLASSES[c],
                            "confidence": float(s)} for b, c, s in zip(boxes, classes, confs)],
            "mask_fraction": fraction,
            "segmentation_matched": bool(seg_matched),
            "segmentation_animal_count": segmentation_count,
            "segmentation_species": segmentation_species,
            "image_size": [im.width, im.height],
            "timings": timings,
        }
        if with_images:
            result["annotated_png"] = _b64(annotate(im, result, mask))
            result["cutout_png"] = _b64(cutout(im, mask, box))
            result["mask_png"] = _b64(Image.fromarray((mask * 255).astype(np.uint8)))
        return result


def annotate(im: Image.Image, res: dict, mask: np.ndarray) -> Image.Image:
    arr = np.array(im).astype(float)
    arr[mask] = 0.6 * arr[mask] + 0.4 * np.array([0, 160, 255])
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    w = max(2, im.width // 200)
    for det in res["detections"]:
        d.rectangle(det["box_xyxy"], outline=(255, 80, 0), width=w)
    colour = (0, 150, 0) if res["verdict"] == "ready" else (200, 0, 0)
    d.rectangle([0, 0, im.width, 22], fill=(255, 255, 255))
    d.text((4, 4), f"{res['verdict']} | {res['species']} {res['confidence']:.2f} | "
                   f"mask {res['mask_fraction']:.0%}", fill=colour)
    return out


def cutout(im: Image.Image, mask: np.ndarray, box: np.ndarray) -> Image.Image:
    rgba = im.convert("RGBA")
    rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    return rgba.crop(tuple(int(round(v)) for v in box))


def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
