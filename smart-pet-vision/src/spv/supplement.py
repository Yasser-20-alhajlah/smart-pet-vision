"""Phase 1 supplement: checkpoint check, full learning curves, similarity sensitivity, team-set audit.

Run from notebooks/02b_phase1_supplement.ipynb. Nothing here retrains or replaces the submitted
models in runs/classification; numeric outputs go to runs/supplement/. Private team review images are written outside the repository.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from .classify import PetDataset, device, transforms


# ---------------------------------------------------------------- helpers
def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def subset_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    n, k = len(y), int((pred == y).sum())
    lo, hi = wilson(k, n)
    f1 = float(f1_score(y, pred, average="macro")) if n and len(np.unique(y)) > 1 else float("nan")
    return {"n": n, "correct": k, "accuracy": k / n if n else float("nan"),
            "accuracy_ci95": [lo, hi], "macro_f1": f1}


# ----------------------------------------------------------- learning curves
def plot_full_curves(histories: dict[str, pd.DataFrame], path: Path, note: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, (kind, h) in enumerate(histories.items()):
        c = f"C{i}"  # one colour per model: solid = training, dashed = validation
        ax[0].plot(h.epoch, h.train_loss, "-", color=c, label=f"{kind} train")
        ax[0].plot(h.epoch, h.val_loss, "--", color=c, label=f"{kind} val")
        ax[1].plot(h.epoch, h.train_acc, "-", color=c, label=f"{kind} train")
        ax[1].plot(h.epoch, h.val_acc, "--", color=c, label=f"{kind} val")
        ax[2].plot(h.epoch, h.val_macro_f1, "-", color=c, label=kind)
    ax[0].set(title="Loss (train vs. validation)", xlabel="epoch", ylabel="cross-entropy")
    ax[1].set(title="Accuracy (train vs. validation)", xlabel="epoch")
    ax[2].set(title="Validation macro-F1", xlabel="epoch")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(note, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- identity leakage
@torch.no_grad()
def embed(df: pd.DataFrame, image_dir: Path, cfg: dict, pretrained: bool = True) -> np.ndarray:
    """L2-normalised 2048-d features of an ImageNet ResNet-50 (not trained on this data)."""
    from torchvision import models

    net = models.resnet50(weights="DEFAULT" if pretrained else None)
    net.fc = nn.Identity()
    net = net.to(device()).eval()
    ds = PetDataset(df, image_dir, transforms(cfg["classification"]["image_size"], False))
    dl = DataLoader(ds, batch_size=cfg["classification"]["batch_size"], shuffle=False,
                    num_workers=cfg["classification"]["num_workers"])
    feats = []
    for x, _ in dl:
        feats.append(nn.functional.normalize(net(x.to(device())).float(), dim=1).cpu())
    return torch.cat(feats).numpy()


def nearest_train_neighbours(df: pd.DataFrame, feats: np.ndarray) -> pd.DataFrame:
    """For every test image, the most similar training image of the same breed."""
    is_tr, is_te = (df.split == "train").values, (df.split == "test").values
    tr, te = df[is_tr].reset_index(drop=True), df[is_te].reset_index(drop=True)
    sims = feats[is_te] @ feats[is_tr].T
    same = te.breed.to_numpy(dtype=object)[:, None] == tr.breed.to_numpy(dtype=object)[None, :]
    sims = np.where(same, sims, -1.0)
    j = sims.argmax(1)
    out = te[["image", "breed", "species", "label"]].copy()
    out["nn_train_image"] = tr.image.values[j]
    out["nn_sim"] = sims[np.arange(len(te)), j]
    return out


def identity_effect(nn_df: pd.DataFrame, preds: dict[str, np.ndarray],
                    thresholds=(0.95, 0.90, 0.85, 0.80, 0.75)) -> pd.DataFrame:
    """Test metrics on the test images whose closest same-breed training image is below t.

    This is sensitivity analysis, not an identity-disjoint evaluation. Filtering changes
    the evaluated population and cannot establish or quantify a causal leakage effect.
    """
    rows = []
    y = nn_df.label.values
    for t in (None, *thresholds):
        keep = np.ones(len(y), bool) if t is None else nn_df.nn_sim.values < t
        k_close = int((~keep).sum())
        lo, hi = wilson(k_close, len(y))
        for kind, p in preds.items():
            m = subset_metrics(p[keep], y[keep])
            rows.append({"threshold": "all" if t is None else t, "model": kind,
                         "test_with_close_train_match": k_close,
                         "share_close": k_close / len(y), "share_close_ci95": [lo, hi], **m})
    return pd.DataFrame(rows)


def pair_sheet(nn_df: pd.DataFrame, image_dir: Path, path: Path, n: int = 40,
               thumb: int = 180) -> None:
    """Top-n most similar test/train pairs side by side, for visual review."""
    top = nn_df.sort_values("nn_sim", ascending=False).head(n).reset_index(drop=True)
    cols = 4
    rows = math.ceil(len(top) / cols)
    W, H = cols * (2 * thumb + 20), rows * (thumb + 34)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in top.iterrows():
        x0, y0 = (i % cols) * (2 * thumb + 20), (i // cols) * (thumb + 34)
        for k, name in enumerate((r.image, r.nn_train_image)):
            im = ImageOps.fit(Image.open(image_dir / f"{name}.jpg").convert("RGB"), (thumb, thumb))
            sheet.paste(im, (x0 + k * thumb, y0))
        d.text((x0 + 2, y0 + thumb + 2), f"{i + 1}. test {r.image}", fill="black")
        d.text((x0 + 2, y0 + thumb + 16), f"   train {r.nn_train_image}  sim {r.nn_sim:.3f}",
               fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


# ----------------------------------------------------------- team-set audit
def _stem(s: str) -> str:
    return Path(str(s)).stem


def audit_team(team_dir: Path, out_dir: Path, per_sheet: int = 48, thumb: int = 200) -> dict:
    """Check every team photo, mask, label and box, and render all photos with their boxes."""
    team_dir, out_dir = Path(team_dir), Path(out_dir)
    images = sorted((team_dir / "images").glob("*.jpg"))
    masks = {p.stem: p for p in (team_dir / "masks").glob("*.png")}
    labels = pd.read_csv(team_dir / "labels.csv")
    labels["stem"] = labels["image"].map(_stem)
    boxes = pd.read_csv(team_dir / "team_boxes.csv")
    boxes["stem"] = boxes["image"].map(_stem)
    stems = [p.stem for p in images]
    md5s, rows = {}, []
    for p in images:
        r = {"image": p.stem, "readable": True}
        try:
            im = Image.open(p)
            im.load()
            r.update(width=im.width, height=im.height, max_side_ok=max(im.size) <= 2048,
                     exif_tags=len(im.getexif()))
            mp = masks.get(p.stem)
            if mp is not None:
                m = np.array(Image.open(mp).convert("L")) > 0
                r.update(mask_same_size=m.shape == (im.height, im.width),
                         mask_fraction=float(m.mean()))
            else:
                r.update(mask_same_size=False, mask_fraction=0.0)
        except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
            r.update(readable=False, error=str(e))
        md5s.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(p.stem)
        rows.append(r)
    per = pd.DataFrame(rows)
    nbox = boxes.groupby("stem").size()
    lab = labels.set_index("stem")
    per["species"] = per.image.map(lab["species"]) if "species" in lab else None
    per["n_animals"] = per.image.map(lab["n_animals"]) if "n_animals" in lab else None
    per["n_boxes"] = per.image.map(nbox).fillna(0).astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / "team_audit_per_image.csv", index=False)

    summary = {
        "images": len(images), "masks": len(masks), "label_rows": len(labels),
        "box_rows": len(boxes),
        "images_without_mask": sorted(set(stems) - set(masks)),
        "masks_without_image": sorted(set(masks) - set(stems)),
        "images_without_label": sorted(set(stems) - set(labels.stem)),
        "labels_without_image": sorted(set(labels.stem) - set(stems)),
        "unreadable": per.loc[~per.readable, "image"].tolist(),
        "with_exif": per.loc[per.get("exif_tags", 0).fillna(0) > 0, "image"].tolist(),
        "over_2048px": per.loc[~per.get("max_side_ok", True).fillna(True).astype(bool), "image"].tolist(),
        "mask_size_mismatch": per.loc[~per.mask_same_size.fillna(False).astype(bool), "image"].tolist(),
        "empty_masks": per.loc[per.mask_fraction.fillna(0) == 0, "image"].tolist(),
        "box_count_ne_n_animals": per.loc[per.n_boxes != per.n_animals, "image"].tolist(),
        "exact_duplicates_within_set": [v for v in md5s.values() if len(v) > 1],
        "species_counts": labels["species"].value_counts().to_dict() if "species" in labels else {},
        "mask_fraction_median": float(per.mask_fraction.median()),
    }

    # every photo with its boxes and label, so the whole set can be checked by eye
    bx = boxes.groupby("stem")
    for s in range(0, len(images), per_sheet):
        chunk = images[s:s + per_sheet]
        cols = 8
        sheet = Image.new("RGB", (cols * thumb, math.ceil(len(chunk) / cols) * (thumb + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for i, p in enumerate(chunk):
            im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
            sc = thumb / max(im.size)
            im_s = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            dd = ImageDraw.Draw(im_s)
            if p.stem in bx.groups:
                for _, b in bx.get_group(p.stem).iterrows():
                    dd.rectangle([b.xtl * sc, b.ytl * sc, b.xbr * sc, b.ybr * sc],
                                 outline=(255, 0, 0) if b.label == "dog" else (0, 120, 255), width=2)
            x0, y0 = (i % cols) * thumb, (i // cols) * (thumb + 16)
            sheet.paste(im_s, (x0, y0))
            sp = lab["species"].get(p.stem, "?") if "species" in lab else "?"
            d.text((x0 + 2, y0 + thumb + 1), f"{p.stem} {sp}", fill="black")
        sheet.save(out_dir / f"team_sheet_{s // per_sheet:02d}.jpg", quality=85)
    return summary
