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
sys.path.insert(0, str(SUB / "tools"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PHASE2, capture_output=True, text=True).stdout.strip()


def run_checks():
    """Run the unit tests and the evidence checker on this revision; keep only the summary lines."""
    env = {**__import__("os").environ, "PYTHONPATH": str(PHASE2 / "src")}
    t = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=PHASE2,
                       capture_output=True, text=True, env=env)
    ran = [line for line in t.stderr.splitlines() if line.startswith("Ran ")]
    verdict = [line for line in t.stderr.splitlines() if line.startswith(("OK", "FAILED"))]  # e.g. "OK (skipped=15)"
    v = subprocess.run([sys.executable, "tools/verify_submission_evidence.py"], cwd=PHASE2,
                       capture_output=True, text=True)
    return {"unit_tests": {"command": "PYTHONPATH=src python -m unittest discover -s tests",
                           "summary": (ran[-1] if ran else "") + " " + (verdict[-1] if verdict else ("OK" if t.returncode == 0 else "FAILED")),
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
    from make_code_archive import INDEX_NAME, code_members_identical
    cpath = SUB / "reviewed_source.zip"
    with zipfile.ZipFile(cpath) as z:
        cidx = json.loads(z.read(INDEX_NAME))
    code_archive = {"file": cpath.name, "sha256": sha(cpath), "source_commit": cidx["source_commit"],
                    "source_tree": cidx["source_tree"], "n_members": len(cidx["members"]),
                    "excluded": cidx["excluded"],
                    "identical_to_source_zip_code_folder": code_members_identical(zpath, cpath),
                    "relationship": "SmartPetVision_Phase2_source.zip = main.tex + figures/ + private_ids.tex "
                                    "(instructor submission only) + code/ holding exactly the members of "
                                    "reviewed_source.zip with equal bytes. reviewed_source.zip has no private IDs."}
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
        "code_archive": code_archive,
        "other_delivered_files": {p.name: sha(p) for p in extra},
        "checks_at_build": run_checks(),
        "model_weights_sha256_recorded": EXPECTED_SHA256,
        "service_manifest": {"path": "phase2/runs/manifest.json", "sha256": sha(PHASE2 / "runs" / "manifest.json")},
        "evidence_files": {p.name: sha(p) for p in sorted((PHASE2 / "report" / "evidence").glob("*")) if p.is_file()},
        "documents": {str(p.relative_to(PHASE2)): sha(p) for p in (
            PHASE2 / "docs" / "model_card.md", PHASE2 / "docs" / "DEMO_AND_REHEARSAL_CHECKLIST.md",
            PHASE2 / "docs" / "AUDIT_TRACKER_2026-11-04.md", PHASE2 / "tools" / "a9b_root_cause.py",
            PHASE2 / "docs" / "PRESENTATION_QA.md", PHASE2 / "docs" / "QUICKSTART.md", PHASE2 / "docs" / "TROUBLESHOOTING.md",
            PHASE2 / "docs" / "ONNX_EVIDENCE_INDEX.md", PHASE2 / "docs" / "REHEARSAL_CHECKLIST_ONE_PAGE.md",
            PHASE2 / "docs" / "TEAM_READTHROUGH_SIGNOFF.md", PHASE2 / "tools" / "run_service.py",
            PHASE2 / "README.md", PHASE2 / "tools" / "verify_submission_evidence.py",
            PHASE2 / "tools" / "smoke_test_http.py")},
        "acceptance": {"A1-A8": "met", "A9a": "met", "A9b": "not met (0.0018-0.0023 vs 0.001; re-run 5 Nov)",
                       "serving_backend": "PyTorch"},
        "external_assets_required": "Three private weight files from Drive MyDrive/capstone /smart-pet-vision/runs/ at the paths in runs/manifest.json; their SHA-256 must equal model_weights_sha256_recorded (checked by tools/preflight_release.py).",
        "known_limitations": ["A9b detector ONNX raw parity not met; PyTorch serves.",
                              "17 of 52 previously inspected multi-animal team photos were marked ready.",
                              "One training recipe per model; the four corrected training boxes were not used for retraining.",
                              "Timings and team-photo results were measured with src/spv as at 879b0e8 (identical from a74784b), not re-measured on this candidate."],
        # Three groups, so optional work is not read as a condition for submission.
        "decisions_recorded": ["Owner chose this package and the template report as the one Phase 2 report (5 Nov 2026)",
                               "Team members table follows the Phase 1 order and workstreams (5 Nov 2026)"],
        "submission_gates": ["Each member reads the report and confirms their line in the Team members table",
                             "Owner confirms the previously exposed W&B key was revoked",
                             "One member uploads SmartPetVision_Phase2.pdf and SmartPetVision_Phase2_source.zip by 15 Nov 2026"],
        "presentation_gates": ["Four-person rehearsal with a timer (15-20 min)",
                               "Live demo on the presentation host and a backup recording made there",
                               "The existing recordings are cloud-container runs (CPU smoke test of an earlier candidate; demo page of earlier candidates and of this one); they are fallbacks, not the presentation-host demo"],
        "optional_improvements": ["GPU revalidation of the candidate code (output validation added after 879b0e8)",
                                  "New independent multi-animal data to re-measure the 17/52 limitation",
                                  "Retraining with the four corrected training boxes (validation-only selection, one final test)",
                                  "Browser tests in CI (Playwright and Chromium are not in requirements.txt; CI reports the 15 browser tests as skipped)"],
        "notes": ["build.sh dates the PDF by the commit time (SOURCE_DATE_EPOCH), so rebuilding the same commit with the same TeX installation gives the same PDF bytes; the ZIPs are byte-reproducible.",
                  "evidence_checks counts checks of two reports: the superseded free-form report (report/main.tex) and the submission report (submission/main.tex); see the summary line for the split.",
                  "Weights, team photos and private_ids.tex are private and not in Git."],
    }
    (SUB / "RELEASE_MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote RELEASE_MANIFEST.json" + (" (WARNING: uncommitted changes)" if dirty else ""))


if __name__ == "__main__":
    main()
