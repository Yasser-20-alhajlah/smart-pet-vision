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
PYTHONPATH=src python -m unittest discover -s tests      # 102 tests expected to pass
python tools/verify_submission_evidence.py               # report/main.tex vs evidence
```
CI (`.github/workflows/phase2-tests.yml`) runs the same on every PR.

## Private assets (not in Git)
Google Drive (owner's account): `MyDrive/capstone /smart-pet-vision/runs/` (weights, `manifest.json`), `MyDrive/capstone /SmartPetVision_TeamData/` (team photos, labels), `MyDrive/capstone /Submission/FINAL PHASE_2/` (submission folders 01–06, 99_Review). Real-weight runs happen in Colab with `notebooks/06_release_validation_v2.ipynb`.

## Open items (status 1 Nov 2026)
1. Upload `SmartPetVision_Phase2.pdf` and `_source.zip` (one team member, due 15 Nov 2026).
2. Each member confirms their line in the "Team members" table of `submission/main.tex`.
3. Record the demo backup video (script: Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md`); export the deck (claude.ai Slides artifact) to `04_Presentation`.
4. Privacy: the repo is public and still contains student IDs in `README_SUBMISSION.md`, `phase2/report/main.tex`, the committed PDFs, and Git history. Removing them from history needs the owner's explicit approval.
5. Known report gaps (stated in the self-assessment, 40/45): one training recipe per model; mask thresholds not swept on validation; A9b fails; no public deployment; per-stage latency percentiles not recorded.
6. Optional: run `notebooks/09_onnx_channel_diagnosis_v2.ipynb` (CPU, ~5 min) to test whether the ONNX gap sits in the box channels.
7. Owner to confirm the old W&B API key was revoked, and decide on a short AI-assistance disclosure (course guidelines contain no policy).
