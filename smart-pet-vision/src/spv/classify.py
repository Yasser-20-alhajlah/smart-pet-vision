"""M2-M3 - species classification: CNN from scratch vs. ResNet frozen vs. fine-tuned."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import Tracker, save_json

CLASSES = ["cat", "dog"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ data
def transforms(size: int, train: bool):
    from torchvision import transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.5, 1.0)),  # also covers detector crops
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class PetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, tf, crop_to_box: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tf = tf
        self.crop_to_box = crop_to_box

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        im = Image.open(self.image_dir / f"{row.image}.jpg").convert("RGB")
        if self.crop_to_box:
            im = im.crop((row.x1, row.y1, row.x2, row.y2))
        return self.tf(im), int(row.label)


def loaders(df: pd.DataFrame, image_dir: Path, cfg: dict) -> dict:
    cc = cfg["classification"]
    out = {}
    for split in ("train", "val", "test"):
        ds = PetDataset(df[df.split == split], image_dir, transforms(cc["image_size"], split == "train"))
        g = torch.Generator().manual_seed(cfg["seed"])
        out[split] = DataLoader(ds, batch_size=cc["batch_size"], shuffle=(split == "train"),
                                num_workers=cc["num_workers"], generator=g,
                                pin_memory=torch.cuda.is_available())
    return out


# ---------------------------------------------------------------- models
class ScratchCNN(nn.Module):
    """Five conv blocks (BN + ReLU + max-pool), global average pooling, dropout."""

    def __init__(self, num_classes: int = 2, width: int = 32):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (width, width * 2, width * 4, width * 8, width * 8):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(kind: str, cfg: dict) -> nn.Module:
    """kind: 'scratch' | 'frozen' | 'finetune'."""
    if kind == "scratch":
        return ScratchCNN(len(CLASSES))
    from torchvision import models

    cc = cfg["classification"]
    weights = "DEFAULT" if cc["pretrained"] else None
    net = getattr(models, cc["backbone"])(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    if kind == "frozen":
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return net


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# -------------------------------------------------------------- training
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dev = device()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(dev)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train(kind: str, data: dict, cfg: dict, out_dir: Path, max_batches: int | None = None,
          run_name: str | None = None) -> dict:
    """Train one variant; keeps the checkpoint with the best validation macro-F1.

    Logs training loss and accuracy (running, on augmented batches in train mode) and
    validation loss, accuracy and macro-F1 per epoch. The extra metrics reuse quantities that
    are computed anyway, so they draw no random numbers and do not change training.
    """
    from sklearn.metrics import f1_score

    hp = cfg["classification"][kind]
    dev = device()
    model = build_model(kind, cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    tracker = Tracker(cfg, run_name or f"cls-{kind}", {"kind": kind, **hp, **count_params(model)})

    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(hp["epochs"]):
        model.train()
        if kind == "frozen":  # frozen backbone: BatchNorm statistics must not adapt either
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        t0, run_loss, seen, correct = time.time(), 0.0, 0, 0
        for b, (x, y) in enumerate(data["train"]):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
            correct += int((out.argmax(1) == y).sum().item())
        sched.step()
        val_logits, val_y = predict_logits(model, data["val"])
        val_pred = val_logits.argmax(1)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(val_logits),
                                                     torch.from_numpy(val_y).long()))
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(seen, 1),
               "train_acc": correct / max(seen, 1), "val_loss": val_loss,
               "val_acc": float((val_pred == val_y).mean()),
               "val_macro_f1": float(f1_score(val_y, val_pred, average="macro")),
               "epoch_sec": time.time() - t0}
        history.append(rec)
        tracker.log(rec, step=epoch + 1)
        print(f"[{kind}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in rec.items()))
        if rec["val_macro_f1"] > best:
            best = rec["val_macro_f1"]
            torch.save({"kind": kind, "state_dict": model.state_dict()}, out_dir / f"{kind}.pt")
    tracker.finish()
    pd.DataFrame(history).to_csv(out_dir / f"{kind}_history.csv", index=False)
    return {"kind": kind, "best_val_macro_f1": best, **count_params(model)}


def load_model(kind: str, cfg: dict, ckpt: Path) -> nn.Module:
    cfg = {**cfg, "classification": {**cfg["classification"], "pretrained": False}}
    model = build_model(kind, cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    return model.to(device()).eval()


# ----------------------------------------------------------- calibration
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling on validation logits (Guo et al., 2017)."""
    lg, y = torch.tensor(logits), torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    # Perfectly separated validation logits push T towards 0; keep it in a sane range.
    return float(np.clip(log_t.detach().exp().item(), 0.05, 20.0))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> tuple[float, pd.DataFrame]:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    rows, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = (pred[m] == labels[m]).mean(), conf[m].mean()
        total += m.mean() * abs(acc - c)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": int(m.sum()), "accuracy": acc,
                     "confidence": c})
    return float(total), pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation
def evaluate(probs: np.ndarray, labels: np.ndarray, meta: pd.DataFrame) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, pred, labels=[0, 1], zero_division=0)
    err, _ = ece(probs, labels)
    meta = meta.reset_index(drop=True).assign(pred=pred, conf=probs.max(1),
                                              correct=pred == labels)
    by_breed = (meta.groupby(["species", "breed"])
                .agg(n=("correct", "size"), errors=("correct", lambda c: int((~c).sum())))
                .assign(error_rate=lambda d: d.errors / d.n)
                .sort_values("error_rate", ascending=False).reset_index())
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "ece": err,
        "by_breed": by_breed.to_dict("records"),
        "n": int(len(labels)),
    }


def plot_reliability(probs: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    err, table = ece(probs, labels)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(table.confidence, table.accuracy, "o-", lw=1.5)
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=f"{title}\nECE = {err:.4f}",
           xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_gallery(meta: pd.DataFrame, probs: np.ndarray, labels: np.ndarray, image_dir: Path,
                  path: Path, k: int = 24) -> pd.DataFrame:
    """Most confident mistakes first - these drive the data-improvement plan."""
    from PIL import ImageDraw

    meta = meta.reset_index(drop=True).assign(pred=probs.argmax(1), conf=probs.max(1))
    wrong = meta[meta.pred != labels].sort_values("conf", ascending=False).head(k)
    if wrong.empty:
        return wrong
    tile, cols = 200, 6
    sheet = Image.new("RGB", (cols * tile, ((len(wrong) + cols - 1) // cols) * (tile + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for i, row in enumerate(wrong.itertuples()):
        im = Image.open(Path(image_dir) / f"{row.image}.jpg").convert("RGB")
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 30)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + tile + 2), f"true {row.species} / pred {CLASSES[row.pred]}", fill="black")
        d.text((x + 2, y + tile + 15), f"{row.breed[:18]} conf {row.conf:.2f}", fill="black")
    sheet.save(path, quality=85)
    return wrong[["image", "species", "breed", "pred", "conf"]]


def run_evaluation(kind: str, model: nn.Module, data: dict, df: pd.DataFrame, image_dir: Path,
                   out_dir: Path) -> dict:
    """Fit temperature on val, report raw and calibrated metrics on the test split."""
    val_logits, val_y = predict_logits(model, data["val"])
    test_logits, test_y = predict_logits(model, data["test"])
    t = fit_temperature(val_logits, val_y)
    test_meta = df[df.split == "test"]
    raw = evaluate(softmax(test_logits), test_y, test_meta)
    cal = evaluate(softmax(test_logits, t), test_y, test_meta)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(softmax(test_logits), test_y, fig_dir / f"{kind}_reliability_raw.png",
                     f"{kind} (raw)")
    plot_reliability(softmax(test_logits, t), test_y, fig_dir / f"{kind}_reliability_cal.png",
                     f"{kind} (T={t:.2f})")
    plot_confusion(cal["confusion_matrix"], fig_dir / f"{kind}_confusion.png", kind)
    error_gallery(test_meta, softmax(test_logits, t), test_y, image_dir,
                  fig_dir / f"{kind}_errors.jpg").to_csv(out_dir / f"{kind}_errors.csv", index=False)
    result = {"kind": kind, "temperature": t, "test_raw": raw, "test_calibrated": cal}
    save_json(result, out_dir / f"{kind}_test_metrics.json")
    return result


def comparison_table(results: list[dict], train_info: list[dict]) -> pd.DataFrame:
    info = {r["kind"]: r for r in train_info}
    rows = []
    for r in results:
        c = r["test_calibrated"]
        rows.append({"model": r["kind"], "trainable_params": info[r["kind"]]["trainable"],
                     "val_macro_f1": info[r["kind"]]["best_val_macro_f1"],
                     "test_accuracy": c["accuracy"], "test_macro_f1": c["macro_f1"],
                     "ece_raw": r["test_raw"]["ece"], "ece_calibrated": c["ece"],
                     "temperature": r["temperature"]})
    return pd.DataFrame(rows)
