"""Publish the standalone Phase 1 report to Weights & Biases as a new draft report.

The text comes from docs/PHASE1_REPORT_STANDALONE.md. Each ``<!-- PANELS: name -->`` marker in that
file is replaced by a panel grid whose run set is pinned to explicit run ids, so every curve has one
stated source. The earlier report is never opened or modified; it stays as the historical record.

    python tools/publish_phase1_wandb_report.py              # dry run: print the outline, no network
    python tools/publish_phase1_wandb_report.py --publish    # create the draft (needs `wandb login`)

Requires ``pip install wandb wandb-workspaces`` for --publish. No training and no run is changed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ENTITY = "yasser-alhajlah-kaust"
PROJECT = "smart-pet-vision"
TITLE = "Smart Pet Vision — Phase 1 report (M1–M3)"
DESCRIPTION = "Data foundations and cat/dog classification. Current results; earlier values are in Appendix H."
SOURCE = Path(__file__).resolve().parents[1] / "docs" / "PHASE1_REPORT_STANDALONE.md"
MARKER = re.compile(r"^<!-- PANELS: ([a-z0-9-]+) -->$")

FULL = [  # (panel title, logged keys, log-scale y)
    ("Loss (train and validation)", ["train_loss", "val_loss"], False),
    ("Accuracy (train and validation)", ["train_acc", "val_acc"], False),
    ("Validation macro-F1", ["val_macro_f1"], False),
]
ORIGINAL = [  # the submitted runs logged these keys only
    ("Training loss", ["train_loss"], False),
    ("Validation accuracy", ["val_acc"], False),
    ("Validation macro-F1", ["val_macro_f1"], False),
]
# marker name -> (run-set name, run ids, panels)
GRIDS = {
    "curves-scratch": ("Scratch CNN — full-logging rerun bvns5vir", ["bvns5vir"], FULL),
    "curves-frozen": ("Frozen ResNet-50 — full-logging rerun 4ixtdkiq", ["4ixtdkiq"], FULL),
    "curves-finetuned": (
        "Fine-tuned ResNet-50 — full-logging rerun b3t0emrs",
        ["b3t0emrs"],
        [(FULL[0][0] + ", log axis", FULL[0][1], True), FULL[1], FULL[2]],
    ),
    "historical-original-runs": (
        "Historical — submitted runs gh7ayx6h, xsz5afr9, 8nvxbsn2",
        ["gh7ayx6h", "xsz5afr9", "8nvxbsn2"],
        ORIGINAL,
    ),
}


def parse(text: str) -> list[tuple[str, str]]:
    """Split the document into ("markdown", text) and ("panels", marker name) parts, in order."""
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    for line in text.splitlines():
        m = MARKER.match(line.strip())
        if not m:
            buf.append(line)
            continue
        if m.group(1) not in GRIDS:
            raise SystemExit(f"unknown panel marker: {m.group(1)}")
        if "".join(buf).strip():
            parts.append(("markdown", "\n".join(buf).strip()))
        buf = []
        parts.append(("panels", m.group(1)))
    if "".join(buf).strip():
        parts.append(("markdown", "\n".join(buf).strip()))
    used = [name for kind, name in parts if kind == "panels"]
    if sorted(used) != sorted(GRIDS):
        raise SystemExit(f"markers in the document {sorted(used)} differ from the grids {sorted(GRIDS)}")
    return parts


def run_filter(run_ids: list[str]) -> str:
    return "ID in [" + ", ".join(f"'{r}'" for r in run_ids) + "]"


def outline(parts: list[tuple[str, str]]) -> None:
    for kind, value in parts:
        if kind == "markdown":
            heads = [ln for ln in value.splitlines() if ln.startswith("#")]
            print(f"text   {len(value):>5} chars  {' | '.join(heads) or '(continuation)'}")
        else:
            name, run_ids, panels = GRIDS[value]
            print(f"panels {name}")
            print(f"       run set filter: {run_filter(run_ids)}")
            for title, keys, log_y in panels:
                print(f"       - {title}: x=epoch, y={keys}{' (log y)' if log_y else ''}")


def publish(parts: list[tuple[str, str]]) -> str:
    import wandb_workspaces.reports.v2 as wr

    blocks = []
    for kind, value in parts:
        if kind == "markdown":
            blocks.append(wr.MarkdownBlock(text=value))
            continue
        name, run_ids, panels = GRIDS[value]
        blocks.append(
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=run_filter(run_ids))],
                panels=[
                    wr.LinePlot(title=title, x="epoch", y=keys, log_y=log_y or None, smoothing_type="none")
                    for title, keys, log_y in panels
                ],
            )
        )
    report = wr.Report(project=PROJECT, entity=ENTITY, title=TITLE, description=DESCRIPTION, blocks=blocks)
    report.save(draft=True)
    return report.url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publish", action="store_true", help="create the draft report on W&B")
    args = ap.parse_args()
    parts = parse(SOURCE.read_text(encoding="utf-8"))
    outline(parts)
    if args.publish:
        print("draft report:", publish(parts))
    else:
        print("dry run only; nothing was sent to W&B")


if __name__ == "__main__":
    main()
