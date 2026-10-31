"""Shared helpers: configuration, seeding, paths, experiment logging."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Tracker:
    """Thin wrapper so every stage logs the same way with or without W&B."""

    def __init__(self, cfg: dict, name: str, config: dict | None = None):
        self.run = None
        if cfg["wandb"]["enabled"]:
            import wandb

            self.run = wandb.init(
                entity=cfg["wandb"].get("entity"), project=cfg["wandb"]["project"],
                name=name, config=config or {}, reinit=True
            )

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(data, step=step)

    def image(self, key: str, path: str | Path) -> None:
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Image(str(path))})

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
