#!/usr/bin/env python3
"""Static execution check of the Phase 2 notebooks.

For every notebook under ``notebooks/`` the script verifies that the file is
valid nbformat JSON, that every code cell parses as Python once IPython magics
and shell escapes are removed, and it records what each notebook needs to run
(Google Drive mount, CUDA, training calls, W&B login, Oxford images, team
photos). The result is written as JSON so that the run can be cited.

The script does not execute any notebook. Notebooks that need Drive, a GPU,
the Oxford images, the team photos or a W&B key can only be run in Colab or
Kaggle with the team's private assets; see ``docs/NOTEBOOK_EXECUTION_PLAN.md``.

Usage: python tools/check_notebooks.py [--notebooks DIR] [--output FILE]
Exit status 1 if any notebook is unreadable or a code cell does not parse.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

NEEDS = {
    "google_drive": re.compile(r"drive\.mount\(|/content/drive"),
    "cuda_or_gpu": re.compile(r"\bcuda\b|nvidia-smi|device\s*=\s*['\"]?0", re.I),
    "training_call": re.compile(r"\.train\(|\.fit\("),
    "wandb_login": re.compile(r"wandb\.(login|init)\("),
    "oxford_images": re.compile(r"oxford|data\.yaml", re.I),
    "team_photos": re.compile(r"team_photos|TeamData|team_images|\[\"team\"\]", re.I),
    "pip_install": re.compile(r"pip\s+(-q\s+)?install"),
    "git_clone": re.compile(r"git clone"),
}
PRIVATE_NEEDS = ("google_drive", "cuda_or_gpu", "training_call", "wandb_login",
                 "oxford_images", "team_photos")

MAGIC = re.compile(r"^([ \t]*)(%%?\w+|!).*$", re.M)
HELP = re.compile(r"\?\s*$", re.M)


def strip_ipython(src: str) -> str:
    """Replace IPython magics and shell escapes by ``pass`` at the same indent."""
    src = MAGIC.sub(lambda m: m.group(1) + "pass", src)
    return HELP.sub("", src)


def check_notebook(path: Path) -> dict:
    rec = {"notebook": path.name, "valid_json": False, "nbformat": None,
           "code_cells": 0, "markdown_cells": 0, "executed_cells": 0,
           "syntax_errors": [], "needs": {}, "imports": [],
           "runnable_in_cpu_container_without_private_assets": False}
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec["syntax_errors"].append(f"not valid JSON: {exc}")
        return rec
    rec["valid_json"] = True
    rec["nbformat"] = f"{nb.get('nbformat')}.{nb.get('nbformat_minor')}"
    joined = []
    imports: set[str] = set()
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            rec["markdown_cells"] += 1
            continue
        if cell.get("cell_type") != "code":
            continue
        rec["code_cells"] += 1
        if cell.get("execution_count"):
            rec["executed_cells"] += 1
        joined.append(src)
        try:
            tree = ast.parse(strip_ipython(src))
        except SyntaxError as exc:
            rec["syntax_errors"].append(f"cell {i}: {exc.msg} (line {exc.lineno})")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    text = "\n".join(joined)
    rec["needs"] = {k: bool(p.search(text)) for k, p in NEEDS.items()}
    rec["imports"] = sorted(imports)
    rec["runnable_in_cpu_container_without_private_assets"] = not any(
        rec["needs"][k] for k in PRIVATE_NEEDS)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--notebooks", type=Path, default=HERE / "notebooks")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    paths = sorted(args.notebooks.glob("*.ipynb"))
    if not paths:
        print(f"no notebooks in {args.notebooks}", file=sys.stderr)
        return 1
    records = [check_notebook(p) for p in paths]
    failed = [r for r in records if not r["valid_json"] or r["syntax_errors"]]
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notebooks_dir": (str(args.notebooks.relative_to(HERE)) if args.notebooks.is_relative_to(HERE)
                          else args.notebooks.name),
        "python": sys.version.split()[0],
        "what_was_checked": ("nbformat JSON validity and Python syntax of every code cell "
                             "(magics and shell escapes removed); resource needs by pattern. "
                             "No notebook was executed."),
        "notebooks": len(records),
        "parse_ok": len(records) - len(failed),
        "parse_failed": [r["notebook"] for r in failed],
        "records": records,
    }
    for r in records:
        needs = ", ".join(k for k, v in r["needs"].items() if v) or "none"
        status = "FAIL" if (not r["valid_json"] or r["syntax_errors"]) else "ok"
        print(f"{status:4} {r['notebook']:45} code {r['code_cells']:2}  "
              f"executed {r['executed_cells']:2}  needs: {needs}")
        for e in r["syntax_errors"]:
            print("      ", e)
    print(f"{summary['parse_ok']}/{summary['notebooks']} notebooks parse")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
