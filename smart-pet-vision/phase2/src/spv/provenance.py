"""Run provenance: code commit, dirty state or source fingerprint, environment, inputs.

The instructor asked that every run logs its commit hash. ``record()`` returns a JSON-ready dict
to store next to a run's outputs and in its W&B config:

    from spv.provenance import record
    prov = record(extra={"split_version": "v1", "checkpoint_sha256": ..., "config": cfg})

Inside a Git checkout it records the commit, branch and whether the tree is dirty (uncommitted
changes under the repository); outside one (a Colab copy, an extracted archive) it reads
``CODE_ARCHIVE_INDEX.json`` when present and otherwise fingerprints the source files, so a run
can always be matched to the code that produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[2]
SOURCE_GLOBS = ("src/**/*.py", "tools/*.py", "notebooks/*.ipynb", "../configs/*.yaml", "../requirements.txt")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_fingerprint(root: Path = PHASE2) -> dict:
    """SHA-256 over (relative path, file SHA-256) of the Phase 2 source files, sorted."""
    files = sorted({p.resolve() for g in SOURCE_GLOBS for p in root.glob(g) if p.is_file()})
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root.parent) if root.parent in p.parents else p.name
        h.update(str(rel).encode())
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return {"sha256": h.hexdigest(), "files": len(files)}


def record(root: Path = PHASE2, extra: dict | None = None) -> dict:
    """Provenance of the code and environment at call time (see module docstring)."""
    out: dict = {"recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "python": sys.version.split()[0], "platform": platform.platform()}
    commit = _git(["rev-parse", "HEAD"], root)
    if commit:
        status = _git(["status", "--porcelain", "--", "."], root)
        out.update({"code_source": "git", "commit": commit,
                    "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
                    "dirty": bool(status), "dirty_files": status.splitlines()[:50] if status else []})
    else:
        candidates = (root / "CODE_ARCHIVE_INDEX.json", root.parent / "CODE_ARCHIVE_INDEX.json")
        index = next((p for p in candidates if p.exists()), None)
        if index is not None:
            d = json.loads(index.read_text())
            out.update({"code_source": "archive", "commit": d.get("source_commit"),
                        "tree": d.get("source_tree"), "archive_index": str(index)})
        else:
            out["code_source"] = "unversioned copy"
    out["source_fingerprint"] = source_fingerprint(root)
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch is optional for provenance
        pass
    if extra:
        out.update(extra)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    print(json.dumps(record(), indent=2))
