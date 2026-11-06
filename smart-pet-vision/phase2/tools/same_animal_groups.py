#!/usr/bin/env python3
"""Same-animal grouping within breed before splitting (Phase 1 feedback, Part 2).

The Phase 1 split grouped only exact (MD5) and near-identical (pHash) images, so different
photographs of the same pet could cross train and test. This tool builds *identity groups*:
within each breed, two images are linked when the cosine similarity of their ImageNet
ResNet-50 features is at least ``--threshold`` (frozen at 0.90 before any run; the cutoff at
which the instructor's own check counted 133 test images), and connected components of these
links are merged with the MD5/pHash groups. A group is evidence for review, not a verified
animal identity: every linked pair and every pair in the review band
[threshold - 0.05, threshold) is written for visual review, and a review-decisions file can
override links.

Outputs (under --out): ``identity_groups.csv`` (image, breed, ..., identity_group, nn_image,
nn_sim), ``pairs_for_review.csv``, ``grouping_params.json`` (feature-model identity, threshold,
counts, input hashes) and, with --split, ``cross_partition_check.json`` counting the groups
that span two partitions of that split. With --new-split the Phase 1 group-aware splitter is
rerun on the identity groups (seed 42) and written as ``split_v2.csv``. Nothing is trained.

Features: ``--features feats.npy`` (rows aligned with the CSV) or computed with torchvision
ResNet-50 (``--weights DEFAULT`` needs the ImageNet weights; ``--weights none`` is for tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_THRESHOLD = 0.90
REVIEW_BAND = 0.05
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def identity_groups(df: pd.DataFrame, feats: np.ndarray, threshold: float = FROZEN_THRESHOLD,
                    overrides: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union of duplicate groups and within-breed feature links with similarity >= threshold.

    ``df`` needs columns image and breed; ``dup_group`` (or ``group``) is used when present.
    ``feats`` has one row per df row. ``overrides`` rows are (image_a, image_b, decision) with
    decision in {link, unlink}, applied after the automatic links.
    Returns (groups, pairs); pairs holds every pair with similarity >= threshold - REVIEW_BAND.
    """
    if len(df) != len(feats):
        raise ValueError(f"{len(df)} rows but {len(feats)} feature vectors")
    df = df.reset_index(drop=True).copy()
    n = len(df)
    uf = UnionFind(n)
    dup_col = "dup_group" if "dup_group" in df else ("group" if "group" in df else None)
    if dup_col is not None:
        for idx in df.groupby(dup_col).indices.values():
            for j in idx[1:]:
                uf.union(int(idx[0]), int(j))
    f = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    images = df["image"].to_numpy(dtype=object)
    nn_image = np.array([""] * n, dtype=object)
    nn_sim = np.full(n, -1.0)
    unlink = set()
    if overrides is not None and len(overrides):
        for r in overrides.itertuples(index=False):
            if str(r.decision).lower() == "unlink":
                unlink.add(frozenset((r.image_a, r.image_b)))
    pairs = []
    for breed, idx in df.groupby("breed").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        s = f[idx] @ f[idx].T
        np.fill_diagonal(s, -1.0)
        j = s.argmax(1)
        nn_image[idx] = images[idx[j]]
        nn_sim[idx] = s[np.arange(len(idx)), j]
        a, b = np.where(np.triu(s >= threshold - REVIEW_BAND, 1))
        for x, y in zip(a, b):
            ia, ib = int(idx[x]), int(idx[y])
            sim = float(s[x, y])
            linked = sim >= threshold and frozenset((images[ia], images[ib])) not in unlink
            pairs.append({"breed": breed, "image_a": images[ia], "image_b": images[ib], "sim": sim,
                          "auto_link": linked, "band": "linked" if sim >= threshold else "review"})
            if linked:
                uf.union(ia, ib)
    if overrides is not None and len(overrides):
        pos = {im: i for i, im in enumerate(images)}
        for r in overrides.itertuples(index=False):
            if str(r.decision).lower() == "link" and r.image_a in pos and r.image_b in pos:
                uf.union(pos[r.image_a], pos[r.image_b])
    roots = np.array([uf.find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    out = df.copy()
    out["identity_group"] = gid
    out["nn_image"] = nn_image
    out["nn_sim"] = nn_sim
    pairs_df = pd.DataFrame(pairs, columns=["breed", "image_a", "image_b", "sim", "auto_link", "band"])
    return out, pairs_df.sort_values("sim", ascending=False).reset_index(drop=True)


def cross_partition_check(groups: pd.DataFrame, split: pd.Series) -> dict:
    """Count identity groups whose members fall in more than one partition of ``split``."""
    g = groups.assign(split=np.asarray(split))
    span = g.groupby("identity_group")["split"].nunique()
    crossing = span.index[span > 1]
    members = g[g["identity_group"].isin(crossing)]
    by_pair: dict[str, int] = {}
    for _, grp in members.groupby("identity_group"):
        key = "+".join(sorted(grp["split"].unique()))
        by_pair[key] = by_pair.get(key, 0) + 1
    return {"groups": int(groups["identity_group"].nunique()),
            "groups_crossing_partitions": int(len(crossing)),
            "images_in_crossing_groups": int(len(members)),
            "images_in_crossing_groups_by_partition": {k: int(v) for k, v in members["split"].value_counts().items()},
            "crossing_groups_by_partition_pair": by_pair,
            "leakage_free": bool(len(crossing) == 0)}


def resnet50_features(df: pd.DataFrame, image_dir: Path, weights: str, image_size: int = 224,
                      batch_size: int = 64) -> tuple[np.ndarray, dict]:
    """L2-normalised 2048-d ResNet-50 features; returns (features, model identity)."""
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision import models, transforms

    w = None if weights.lower() == "none" else getattr(models.ResNet50_Weights, weights)
    net = models.resnet50(weights=w)
    net.fc = nn.Identity()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    tf = transforms.Compose([transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            names = df["image"].iloc[i:i + batch_size]
            x = torch.stack([tf(Image.open(image_dir / f"{im}.jpg").convert("RGB")) for im in names]).to(dev)
            feats.append(nn.functional.normalize(net(x).float(), dim=1).cpu().numpy())
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd):
        h.update(sd[k].detach().cpu().numpy().tobytes())
    ident = {"model": "torchvision resnet50 (fc removed)", "weights": weights,
             "torchvision": torchvision.__version__, "torch": torch.__version__,
             "state_dict_sha256": h.hexdigest(), "image_size": image_size, "device": dev}
    return np.concatenate(feats), ident


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images-csv", required=True, type=Path,
                    help="columns image, breed[, species, md5, phash, dup_group]")
    ap.add_argument("--image-dir", type=Path, help="folder with <image>.jpg (unless --features)")
    ap.add_argument("--features", type=Path, help=".npy aligned with --images-csv")
    ap.add_argument("--weights", default="DEFAULT", help="torchvision ResNet50_Weights name or 'none'")
    ap.add_argument("--threshold", type=float, default=FROZEN_THRESHOLD)
    ap.add_argument("--overrides", type=Path, help="CSV image_a,image_b,decision(link|unlink)")
    ap.add_argument("--split", type=Path, help="existing split CSV (image, split) to check")
    ap.add_argument("--new-split", action="store_true", help="write split_v2.csv (Phase 1 splitter)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    df = pd.read_csv(a.images_csv)
    if a.features:
        feats = np.load(a.features)
        ident = {"model": "precomputed", "features_file": str(a.features),
                 "features_sha256": hashlib.sha256(a.features.read_bytes()).hexdigest()}
    elif a.image_dir:
        feats, ident = resnet50_features(df, a.image_dir, a.weights)
    else:
        print("--image-dir or --features is required", file=sys.stderr)
        return 2
    overrides = pd.read_csv(a.overrides) if a.overrides else None
    groups, pairs = identity_groups(df, feats, a.threshold, overrides)
    a.out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(a.out / "identity_groups.csv", index=False)
    pairs.to_csv(a.out / "pairs_for_review.csv", index=False)
    dup_col = "dup_group" if "dup_group" in df else ("group" if "group" in df else None)
    sizes = groups.groupby("identity_group")["image"].transform("size")
    params = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "threshold": a.threshold, "review_band": REVIEW_BAND, "frozen_threshold": FROZEN_THRESHOLD,
              "feature_model": ident, "images": int(len(df)), "breeds": int(df["breed"].nunique()),
              "dup_groups": int(df[dup_col].nunique()) if dup_col else None,
              "identity_groups": int(groups["identity_group"].nunique()),
              "images_in_multi_image_groups": int((sizes > 1).sum()),
              "linked_pairs": int(pairs["auto_link"].sum()),
              "review_band_pairs": int((pairs["band"] == "review").sum()),
              "overrides_applied": int(len(overrides)) if overrides is not None else 0,
              "images_csv_sha256": hashlib.sha256(a.images_csv.read_bytes()).hexdigest(),
              "note": "Similarity is evidence for review, not verified animal identity."}
    if a.split:
        sp = pd.read_csv(a.split).set_index("image")["split"].reindex(groups["image"])
        if sp.isna().any():
            print(f"{int(sp.isna().sum())} images missing from --split", file=sys.stderr)
        check = cross_partition_check(groups, sp.fillna("missing"))
        check["split_file"] = str(a.split)
        check["split_sha256"] = hashlib.sha256(a.split.read_bytes()).hexdigest()
        (a.out / "cross_partition_check.json").write_text(json.dumps(check, indent=2) + "\n")
        params["cross_partition_check"] = check
    if a.new_split:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from spv.data import group_stratified_split  # Phase 1 splitter, unchanged

        g2 = groups.drop(columns=[c for c in ("group", "split") if c in groups]).rename(columns={"identity_group": "group"})
        g2["split"] = group_stratified_split(g2, FRACTIONS, a.seed).to_numpy()
        g2[["image", "breed", "group", "split"]].to_csv(a.out / "split_v2.csv", index=False)
        params["split_v2"] = {"seed": a.seed, "fractions": FRACTIONS,
                              "counts": {k: int(v) for k, v in g2["split"].value_counts().items()},
                              "crossing_groups": cross_partition_check(groups, g2["split"])["groups_crossing_partitions"]}
    (a.out / "grouping_params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps({k: v for k, v in params.items() if k != "feature_model"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
