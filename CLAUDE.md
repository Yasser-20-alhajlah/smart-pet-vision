# Smart Pet Vision — capstone project (AI251P Applied Computer Vision, KAUST)

Team of 4: Abdulrahman Mastour Alharthi (WS1 data/classification), Yasser Mansour Alhajlah (WS2 detection), Mohammed Adeeb Alrashdi (WS3 segmentation), Musaad Obaied Saad Alqahtani (WS4 integration). Original topic approved by the instructor on 19 Oct 2026.

## How to work with the owner (Yasser)
- Reply in Arabic, short, result first. Use the format: Clear Answer / Confidence Level (0.0–1.0) / Key Caveats.
- Every claim of progress rests on a command or file read in the same session. Say what was not verified.
- One recommendation, not a menu. Ask only for: deletions, pushes/merges to `main`, anything reaching people, secrets, or changing a fixed decision.
- No deletion of files: move or rename to `*_superseded` instead.
- Commits and PRs are attributed to the team ("Smart Pet Vision Team"); do not add AI signatures or co-author lines.
- Work on a branch and open a PR; do not commit to `main` directly.
- Write reports in a neutral academic voice; never invent results, names, roles or human reviews.

## Fixed facts (do not change without evidence)
- Splits: Oxford-IIIT Pet, 7,335 images, 5,123 / 1,106 / 1,106, seed 42, group-aware. Test split is used once.
- Detector YOLO11s 640 px, operating point conf 0.40, NMS 0.50 (chosen on validation). Test mAP50 0.984, mAP50–95 0.850, P/R/F1 0.958/0.959/0.959.
- Segmenter YOLO11s-seg 640 px, conf 0.25. Test mIoU 0.923, Dice 0.959, mask mAP50–95 0.875. SAM 2.1-B oracle-box baseline (300 test images) mIoU 0.812.
- Service p95 61.4 ms (T4, in-process, A8). CPU HTTP (W&B mu2tolkg) p50/p95/p99 383.8/430.8/690.1 ms — never mix CPU and T4 numbers.
- Acceptance: A1–A8 and A9a met. **A9b (detector ONNX parity ≤ 0.001) is NOT met** (0.0018–0.0020; six settings 0.00198–0.00201). Never relax the threshold or claim ONNX passed. The service uses PyTorch.
- Team photos: 401 photos, 503 animals, evaluation only, private (never commit, never upload to W&B or elsewhere).
- Key W&B runs: detector 2seovilv, segmenter 3ztpz2tc, SAM 1jb5y453, T4 service fqow5ly7, CPU regression f8manl0i, CPU HTTP mu2tolkg.

## Layout
- `smart-pet-vision/` Phase 1 code, report and `SmartPetVision_Phase1.pdf`.
- `smart-pet-vision/phase2/` Phase 2 code (`src/spv`), notebooks, tests, tools, `report/evidence/*.json` (source of every reported number).
- `smart-pet-vision/phase2/submission/` **final Phase 2 report on the instructor's template** (`main.tex`, `figures/`, `build.sh`, `tools/make_figs.py`). Build: `cd smart-pet-vision/phase2/submission && ./build.sh` → `SmartPetVision_Phase2.pdf` + `SmartPetVision_Phase2_source.zip`. Student IDs come from an untracked `private_ids.tex` (kept in Drive); without it the PDF prints "(ID)".
- `smart-pet-vision/phase2/report/` earlier free-form report (superseded for submission by `submission/`).
- `smart-pet-vision/phase2/docs/model_card.md`, `docs/SUBMISSION_HARDENING_2026-10-31.md`.

## Checks to run after any change
```bash
cd smart-pet-vision/phase2
pip install -r ../requirements.txt
PYTHONPATH=src python -m unittest discover -s tests      # 117 tests expected to pass (candidate; 111 at 879b0e8, 108 in the reviewed archive b1aff11)
python tools/verify_submission_evidence.py               # report/main.tex vs evidence
```
CI (`.github/workflows/phase2-tests.yml`) runs the same on every PR.

## Private assets (not in Git)
Google Drive (owner's account): `MyDrive/capstone /smart-pet-vision/runs/` (weights, `manifest.json`), `MyDrive/capstone /SmartPetVision_TeamData/` (team photos, labels), `MyDrive/capstone /Submission/FINAL PHASE_2/` (submission folders 01–06, 99_Review). Real-weight runs happen in Colab with `notebooks/06_release_validation_v2.ipynb`.

## Open items (status 4 Nov 2026)
1. Upload `SmartPetVision_Phase2.pdf` and `_source.zip` (one team member, due 15 Nov 2026).
2. Each member confirms their line in the "Team members" table of `submission/main.tex`.
3. Record the demo backup video (script: Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md`); export the deck (claude.ai Slides artifact) to `04_Presentation`.
4. Privacy: student IDs were removed from current tracked text and from `phase2/report/SmartPetVision_Phase2_Report.pdf`; they remain in `SmartPetVision_Phase1.pdf` (the submitted Phase 1 record) and in Git history. Removing them from history needs the owner's explicit approval.
5. Notebook 11 ran on 2 Nov (T4, code 879b0e8); results in `report/evidence/P2_gap_experiments_20261102.json` and in `submission/main.tex` (baselines, validation sweeps, per-stage p50/p95, ONNX test mAP 0.983/0.852, channel diagnosis). Remaining gaps (self-assessment 41/45): one training recipe per model (optional YOLO11n Part F not run; needs compute approval); A9b fails; notebook 11 results not logged to W&B (Part G).
6. Two Phase 2 reports exist: the template report here (`submission/`) and a Phase 1-layout report in Drive `01_Report` (built 4 Nov by teammates). Team must choose one. Audit and change log: `phase2/docs/AUDIT_TRACKER_2026-11-04.md`. Corrected deck copy: Drive `04_Presentation/SmartPetVision_Phase2_Presentation_v2_20261104` (notes rewritten).
7. Training optimiser is AdamW via `optimizer=auto` (lr0 0.001667), not SGD; evidence `report/evidence/P2_training_optimizer_provenance_20261104.json`.
8. Release: `submission/build.sh` also writes `RELEASE_MANIFEST.json` (SHA-256 of PDF, ZIP members, evidence, docs; recorded weight hashes). Demo and rehearsal: `phase2/docs/DEMO_AND_REHEARSAL_CHECKLIST.md`.
9. Owner to confirm the old W&B API key was revoked. AI-use: guidelines contain no policy; draft `phase2/docs/AI_ASSISTANCE_DISCLOSURE_DRAFT.md` awaits team approval.
10. 4 Nov evening: clean-install real-weight CPU smoke test passed at `1cab83f` (weights SHA-256 recomputed); deck v3 (`SmartPetVision_Phase2_Presentation_v3_20261104`) replaces v2; repo is public and `SmartPetVision_Phase1.pdf` plus history still contain student IDs (owner decision). Details: `phase2/docs/AUDIT_TRACKER_2026-11-04.md` section 5.
11. Drive is authoritative for the release (owner, 4 Nov 22:27): proposed package in `MyDrive/capstone /Submission/FINAL PHASE_2/REVIEW_CANDIDATE_<UTC>` (00_RELEASE … 05_Demo). Do not modify GitHub, merge, rename/move/overwrite existing Drive files or change sharing without approval. Optimiser now has a direct log record (`P2_training_optimizer_direct_log_20261104.json`).
