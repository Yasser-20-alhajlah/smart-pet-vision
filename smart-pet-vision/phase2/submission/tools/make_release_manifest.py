"""Write RELEASE_MANIFEST.json: SHA-256 of every file that makes up the Phase 2 submission.

Run by build.sh after the PDF and ZIP are built. Standard library only.
Model weights are private and not in this workspace; their hashes are the recorded release values
(tools/preflight_release.py EXPECTED_SHA256), which preflight checks wherever the weights are present.
"""
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
PHASE2 = SUB.parent
sys.path.insert(0, str(PHASE2 / "tools"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PHASE2, capture_output=True, text=True).stdout.strip()


def run_checks():
    """Run the unit tests and the evidence checker on this revision; keep only the summary lines."""
    env = {**__import__("os").environ, "PYTHONPATH": str(PHASE2 / "src")}
    t = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=PHASE2,
                       capture_output=True, text=True, env=env)
    ran = [l for l in t.stderr.splitlines() if l.startswith("Ran ")]
    v = subprocess.run([sys.executable, "tools/verify_submission_evidence.py"], cwd=PHASE2,
                       capture_output=True, text=True)
    return {"unit_tests": {"command": "PYTHONPATH=src python -m unittest discover -s tests",
                           "summary": (ran[-1] if ran else "") + (" OK" if t.returncode == 0 else " FAILED"),
                           "returncode": t.returncode},
            "evidence_checks": {"command": "python tools/verify_submission_evidence.py",
                                "summary": v.stdout.strip().splitlines()[-1] if v.stdout.strip() else "",
                                "returncode": v.returncode},
            "scope": "CPU, this build host; synthetic/unit tests and report-vs-evidence checks only, no real-weight inference"}


def main():
    from preflight_release import EXPECTED_SHA256  # recorded release weights

    extra = [Path(a) for a in sys.argv[1:]]  # optional delivered files outside Git (e.g. the deck)

    zpath = SUB / "SmartPetVision_Phase2_source.zip"
    with zipfile.ZipFile(zpath) as z:
        members = {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist())}
    dirty = git("status", "--porcelain", "--", ".")
    out = {
        "release": "Smart Pet Vision Phase 2 submission",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": git("rev-parse", "HEAD"),
        "source_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes_in_phase2": dirty.splitlines(),
        "build_environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "pdflatex": subprocess.run(["pdflatex", "--version"], capture_output=True,
                                                         text=True).stdout.splitlines()[0]},
        "submission_files": {p.name: sha(p) for p in (SUB / "SmartPetVision_Phase2.pdf", zpath)},
        "source_zip_members": members,
        "other_delivered_files": {p.name: sha(p) for p in extra},
        "checks_at_build": run_checks(),
        "model_weights_sha256_recorded": EXPECTED_SHA256,
        "service_manifest": {"path": "phase2/runs/manifest.json", "sha256": sha(PHASE2 / "runs" / "manifest.json")},
        "evidence_files": {p.name: sha(p) for p in sorted((PHASE2 / "report" / "evidence").glob("*")) if p.is_file()},
        "documents": {str(p.relative_to(PHASE2)): sha(p) for p in (
            PHASE2 / "docs" / "model_card.md", PHASE2 / "docs" / "DEMO_AND_REHEARSAL_CHECKLIST.md",
            PHASE2 / "README.md", PHASE2 / "tools" / "verify_submission_evidence.py")},
        "acceptance": {"A1-A8": "met", "A9a": "met", "A9b": "not met (0.0018-0.0020 vs 0.001)",
                       "serving_backend": "PyTorch"},
        "external_assets_required": "Three private weight files from Drive MyDrive/capstone /smart-pet-vision/runs/ at the paths in runs/manifest.json; their SHA-256 must equal model_weights_sha256_recorded (checked by tools/preflight_release.py).",
        "known_limitations": ["A9b detector ONNX raw parity not met; PyTorch serves.",
                              "17 of 52 previously inspected multi-animal team photos were marked ready.",
                              "One training recipe per model; the four corrected training boxes were not used for retraining.",
                              "Timings and team-photo results were measured with code 879b0e8, not re-measured on this candidate."],
        "open_release_gates": ["Team decision on which report is submitted (template report here vs Drive 01_Report version)",
                               "GPU revalidation of the candidate code", "Live demo on a clean runtime and backup recording",
                               "Four-person rehearsal", "Owner revokes the previously exposed W&B key", "Merge to main (approval)"],
        "notes": ["PDF bytes include a build timestamp, so the PDF hash changes on every rebuild.",
                  "Weights, team photos and private_ids.tex are private and not in Git."],
    }
    (SUB / "RELEASE_MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote RELEASE_MANIFEST.json" + (" (WARNING: uncommitted changes)" if dirty else ""))


if __name__ == "__main__":
    main()
