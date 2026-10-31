"""M1 - data preparation for Oxford-IIIT Pet.

Steps: download -> parse labels -> clean (corrupt, tiny, missing trimap)
-> exact and near-duplicate grouping -> group-aware split stratified by breed
-> body masks and boxes from trimaps -> YOLO datasets -> review sheets.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPECIES = {1: "cat", 2: "dog"}
SPECIES_TO_ID = {"cat": 0, "dog": 1}  # class index used by every model
TRIMAP_FG, TRIMAP_BG, TRIMAP_BORDER = 1, 2, 3


# ----------------------------------------------------------------- download
def download(cfg: dict) -> Path:
    """Download via torchvision (works on Colab). Returns the dataset folder."""
    from torchvision.datasets import OxfordIIITPet

    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("trainval", "test"):
        OxfordIIITPet(root=str(raw), split=split, target_types="category", download=True)
    return dataset_dir(cfg)


def dataset_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw"]) / "oxford-iiit-pet"


# ------------------------------------------------------------------- labels
def parse_labels(root: Path) -> pd.DataFrame:
    """Read annotations/list.txt: `name class_id species breed_id`."""
    rows = []
    with open(root / "annotations" / "list.txt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, class_id, species, breed_id = line.split()
            rows.append(
                {
                    "image": name,
                    "class_id": int(class_id),
                    "breed": name.rsplit("_", 1)[0].lower(),
                    "species": SPECIES[int(species)],
                    "label": SPECIES_TO_ID[SPECIES[int(species)]],
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- clean
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect_images(root: Path, df: pd.DataFrame, min_side: int) -> pd.DataFrame:
    """Open every image and trimap; record size, hashes and drop reasons."""
    import imagehash

    records = []
    for row in df.itertuples():
        img_path = root / "images" / f"{row.image}.jpg"
        tri_path = root / "annotations" / "trimaps" / f"{row.image}.png"
        rec = {"image": row.image, "drop_reason": "", "width": 0, "height": 0,
               "mode": "", "md5": "", "phash": ""}
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                rec["mode"] = im.mode
                im = im.convert("RGB")
                rec["width"], rec["height"] = im.size
                rec["phash"] = str(imagehash.phash(im))
            rec["md5"] = _md5(img_path)
        except Exception as exc:  # corrupt or unreadable
            rec["drop_reason"] = f"unreadable: {type(exc).__name__}"
        if not rec["drop_reason"] and min(rec["width"], rec["height"]) < min_side:
            rec["drop_reason"] = "too_small"
        if not rec["drop_reason"]:
            if not tri_path.exists():
                rec["drop_reason"] = "missing_trimap"
            else:
                try:  # a corrupt trimap is dropped with its reason instead of stopping the run
                    with Image.open(tri_path) as t:
                        t.verify()
                    with Image.open(tri_path) as t:
                        if t.size != (rec["width"], rec["height"]):
                            rec["drop_reason"] = "trimap_size_mismatch"
                except Exception as exc:
                    rec["drop_reason"] = f"unreadable_trimap: {type(exc).__name__}"
        records.append(rec)
    return df.merge(pd.DataFrame(records), on="image")


# --------------------------------------------------------------- duplicates
def _popcount_table() -> np.ndarray:
    return np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def duplicate_groups(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Union-find over exact (MD5) and near (pHash Hamming <= threshold) duplicates."""
    df = df.reset_index(drop=True).copy()
    n = len(df)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for _, idx in df.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)

    hashes = np.array(
        [np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in df["phash"]]
    )  # (n, 8)
    table = _popcount_table()
    near_pairs = []
    for i in range(n - 1):
        dist = table[np.bitwise_xor(hashes[i + 1:], hashes[i])].sum(axis=1)
        for k in np.nonzero(dist <= threshold)[0]:
            j = i + 1 + int(k)
            union(i, j)
            near_pairs.append((df.at[i, "image"], df.at[j, "image"], int(dist[k])))

    df["group"] = [find(i) for i in range(n)]
    df["exact_dup"] = df.duplicated("md5", keep="first")
    df.attrs["near_pairs"] = near_pairs
    return df


# -------------------------------------------------------------------- split
def group_stratified_split(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Assign whole duplicate groups to splits, balancing image counts per breed."""
    rng = np.random.default_rng(seed)
    names = list(fractions)
    split = pd.Series(index=df.index, dtype=object)
    group_breed = df.groupby("group")["breed"].agg(lambda s: s.mode().iat[0])
    group_size = df.groupby("group").size()
    for breed in sorted(group_breed.unique()):
        groups = group_breed.index[group_breed == breed].to_numpy().copy()
        rng.shuffle(groups)
        total = group_size[groups].sum()
        target = {k: fractions[k] * total for k in names}
        filled = {k: 0 for k in names}
        for g in groups:
            # put the group where it closes the largest relative gap
            k = max(names, key=lambda s: (target[s] - filled[s]) / max(target[s], 1e-9))
            split[df.index[df["group"] == g]] = k
            filled[k] += group_size[g]
    return split


# -------------------------------------------------------- masks and boxes
def trimap_to_masks(trimap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (body mask incl. border, ignore mask = border pixels)."""
    body = np.isin(trimap, (TRIMAP_FG, TRIMAP_BORDER))
    border = trimap == TRIMAP_BORDER
    return body, border


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0) -> list[float] | None:
    """Largest external contour as a flat [x1, y1, x2, y2, ...] list in pixels."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return c.astype(float).flatten().tolist()


def build_annotations(root: Path, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write binary masks (0/255) and border masks; add body boxes to df."""
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "borders").mkdir(parents=True, exist_ok=True)
    boxes, mask_fractions = [], []
    for name in df["image"]:
        tri = np.array(Image.open(root / "annotations" / "trimaps" / f"{name}.png"))
        body, border = trimap_to_masks(tri)
        mask_fractions.append(float(body.mean()))
        Image.fromarray((body * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
        Image.fromarray((border * 255).astype(np.uint8)).save(out / "borders" / f"{name}.png")
        box = mask_to_box(body)
        boxes.append(box if box else (np.nan,) * 4)
    df = df.copy()
    df[["x1", "y1", "x2", "y2"]] = np.array(boxes, dtype=float)
    df["box_fraction"] = ((df.x2 - df.x1) * (df.y2 - df.y1)) / (df.width * df.height)
    df["mask_fraction"] = mask_fractions
    return df


# --------------------------------------------------------------- YOLO export
def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _rgb_copy(src: Path, dst: Path) -> None:
    """YOLO needs readable RGB JPEGs; a few dataset files are PNG/CMYK inside .jpg."""
    with Image.open(src) as im:
        if im.format == "JPEG" and im.mode == "RGB":
            _link(src, dst)
        else:
            im.convert("RGB").save(dst, quality=95)


def export_yolo(root: Path, df: pd.DataFrame, out: Path, task: str) -> Path:
    """task = 'detect' or 'segment'. Writes images/, labels/ and data.yaml."""
    import yaml

    base = out / f"yolo_{task}"
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        _rgb_copy(root / "images" / f"{row.image}.jpg",
                  base / "images" / row.split / f"{row.image}.jpg")
        w, h = row.width, row.height
        if task == "detect":
            cx, cy = (row.x1 + row.x2) / 2 / w, (row.y1 + row.y2) / 2 / h
            bw, bh = (row.x2 - row.x1) / w, (row.y2 - row.y1) / h
            line = f"{row.label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        else:
            mask = np.array(Image.open(out / "masks" / f"{row.image}.png")) > 0
            poly = mask_to_polygon(mask)
            if poly is None:
                continue
            xy = np.array(poly).reshape(-1, 2) / [w, h]
            line = f"{row.label} " + " ".join(f"{v:.6f}" for v in xy.flatten())
        (base / "labels" / row.split / f"{row.image}.txt").write_text(line + "\n")
    data = {"path": str(base.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {0: "cat", 1: "dog"}}
    with open(base / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return base / "data.yaml"


# ------------------------------------------------------------- review sheets
def contact_sheet(root: Path, out_masks: Path, rows: pd.DataFrame, path: Path,
                  show_box: bool, cols: int = 6, tile: int = 220) -> None:
    """Grid of thumbnails with the label (and box) drawn, for manual review."""
    from PIL import ImageDraw

    n = len(rows)
    sheet = Image.new("RGB", (cols * tile, ((n + cols - 1) // cols) * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows.itertuples()):
        im = Image.open(root / "images" / f"{row.image}.jpg").convert("RGB")
        if show_box:
            d = ImageDraw.Draw(im)
            d.rectangle([row.x1, row.y1, row.x2, row.y2], outline=(255, 0, 0),
                        width=max(2, im.width // 150))
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 18)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile + 2), f"{i}:{row.species}/{row.breed}"[:34], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def review_sheets(root: Path, df: pd.DataFrame, out: Path, n: int, kind: str,
                  seed: int, per_sheet: int = 60) -> Path:
    """Writes sheets plus a CSV where the reviewer records the verdict per image."""
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    folder = out / "review" / kind
    for start in range(0, len(sample), per_sheet):
        contact_sheet(root, out / "masks", sample.iloc[start:start + per_sheet],
                      folder / f"sheet_{start // per_sheet:02d}.jpg", show_box=(kind == "boxes"))
    sample.assign(sheet=sample.index // per_sheet, ok="", note="")[
        ["sheet", "image", "species", "breed", "ok", "note"]
    ].to_csv(folder / "review.csv", index_label="idx")
    return folder


# ------------------------------------------------------------------ driver
def prepare(cfg: dict, root: Path | None = None) -> pd.DataFrame:
    root = root or dataset_dir(cfg)
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    dc = cfg["data"]

    df = parse_labels(root)
    df = inspect_images(root, df, dc["min_side"])
    dropped = df[df.drop_reason != ""]
    df = df[df.drop_reason == ""]
    df = duplicate_groups(df, dc["phash_threshold"])
    near_pairs = df.attrs.get("near_pairs", [])
    exact = df[df.exact_dup]
    df = df[~df.exact_dup].copy()  # keep one copy of byte-identical files
    df["split"] = group_stratified_split(df, dc["splits"], cfg["seed"])
    df = build_annotations(root, df, out)

    no_mask = df[df.x1.isna()]
    df = df[df.x1.notna()].copy()

    df.to_csv(out / "metadata.csv", index=False)
    pd.concat([dropped.assign(stage="clean"),
               exact.assign(drop_reason="exact_duplicate", stage="dedup"),
               no_mask.assign(drop_reason="empty_trimap", stage="masks")]
              )[["image", "drop_reason", "stage"]].to_csv(out / "dropped.csv", index=False)
    pd.DataFrame(near_pairs, columns=["image_a", "image_b", "hamming"]).to_csv(
        out / "near_duplicates.csv", index=False)
    return df


def ensure_data(cfg: dict) -> pd.DataFrame:
    """Entry point for every notebook.

    The split is decided once (notebook 01) and saved to runs/data/metadata.csv, which lives on
    Google Drive. Later sessions re-download the images to fast local disk and rebuild masks
    and YOLO folders from that saved split, so every stage sees exactly the same split.
    """
    root = dataset_dir(cfg)
    if not (root / "annotations" / "list.txt").exists():
        download(cfg)
    out = Path(cfg["paths"]["processed"])
    saved = Path(cfg["paths"]["runs"]) / "data"
    if (saved / "metadata.csv").exists():
        df = pd.read_csv(saved / "metadata.csv")
        if not (out / "yolo_segment" / "data.yaml").exists():
            build_annotations(root, df, out)
            for task in ("detect", "segment"):
                export_yolo(root, df, out, task)
        return df
    df = prepare(cfg, root)
    for task in ("detect", "segment"):
        export_yolo(root, df, out, task)
    saved.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.csv", "dropped.csv", "near_duplicates.csv"):
        shutil.copy2(out / name, saved / name)
    return df


def leakage_audit(df: pd.DataFrame) -> dict:
    """Every duplicate group must live in exactly one split; no MD5 across splits."""
    groups_per_split = df.groupby("group")["split"].nunique()
    md5_per_split = df.groupby("md5")["split"].nunique()
    return {
        "groups_crossing_splits": int((groups_per_split > 1).sum()),
        "md5_crossing_splits": int((md5_per_split > 1).sum()),
        "passed": bool((groups_per_split <= 1).all() and (md5_per_split <= 1).all()),
    }


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(df["split"], df["species"], margins=True, margins_name="total")
    t["share"] = (t["total"] / len(df)).round(3)
    return t
