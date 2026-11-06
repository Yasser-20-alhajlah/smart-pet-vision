# Release-candidate audit, change log and verification log (4 November 2026)

Candidate: branch `phase2-gap-closure` (PR #5), not merged into `main`. Code that produced every reported result: `879b0e8`. Candidate code (src and tests unchanged since): `ee4a73f`. Earlier record: `docs/SUBMISSION_HARDENING_2026-10-31.md` (kept as history).

States: **V** verified in this execution · **H** verified from inspected historical evidence · **I** implemented, not verified · **P** prepared, not executed · **B** blocked by access/environment · **A** awaiting human approval/decision · **S** superseded record.

## 1. Sources of truth found (inventory)

| Item | Location and identifier | Status |
|---|---|---|
| Instructor-template report | `phase2/submission/main.tex` on this branch; build `./build.sh` | Candidate report |
| Second Phase 2 report (Phase 1 layout, Tectonic, 22 pages, Appendix C) | Drive `FINAL PHASE_2/01_Report/SmartPetVision_Phase2.pdf` sha256 `d7bc050b…`, `_source.zip` sha256 `06c056e2…` (built 4 Nov 20:23) | Conflicts with the template report; team decision needed (A) |
| Reviewed code archive | branch `phase2-reviewed-release-20261104` commit `b1aff11`; `SmartPetVision_Phase2_Reviewed_Code_20261104.zip` sha256 `96bbf0ef…` (matches its README) | Its code changes are merged here; archive kept as history |
| Team deck | Drive `04_Presentation/SmartPetVision_Phase2_Presentation.pptx/.pdf` (20 slides) | Corrected copy `_v2_20261104` prepared; original untouched |
| Earlier online slide draft | (superseded) | Not the presentation of record (S) |

## 2. Issues

| # | Issue | Evidence | Sev. | Version | Correction | Verification | State | Owner / next action |
|---|---|---|---|---|---|---|---|---|
| 1 | Report said the detector and segmenter were trained with "SGD (auto), lr0 0.01" | Training call passes no optimiser; Ultralytics `optimizer=auto` rule (3,240 iterations < 10,000 → AdamW, lr 0.002·5/6); W&B `lr/pg0` of 2seovilv and 3ztpz2tc fits lr0 0.001667 with max relative error 4e-6 (epochs 4–40); checkpoint `train_args` record AdamW/0.001667 | High (factual) | main.tex up to 2f0b202 | Rows corrected; provenance paragraph; `P2_training_optimizer_provenance_20261104.json`; checker "never SGD" | V | — |
| 2 | Four loose training boxes described as "not re-derived" | Team records `M4_training_annotation_manifest/validation_20261103.json`: 4 train labels changed, val/test 0, structural PASS, no training | Medium | main.tex | Phase 1 plan row separates proposal, attested review, applied label set, validation, no retraining, no metric claim | V (record), H (team review attestation; no individual sign-off) | — |
| 3 | Two test totals (111, 108) in circulation | Repo 111 = 102 shared + 9 analysis tests; archive 108 = 102 shared + 6 output-safety | Medium | 879b0e8 / b1aff11 | Output-safety code and tests merged; one revision now runs 117 | V: 108/108 archive and 117/117 candidate run here on CPU | — |
| 4 | Pipeline lacked output validation (NaN/out-of-range boxes, probabilities, masks) | Reviewed archive diff (pipeline, serve, deploy) | Medium | ≤ 2f0b202 | Merged; invalid output → HTTP 503, never `ready`; `total_ms` becomes wall time, `stages_sum_ms` keeps the sum; crops use floor/ceil | V (6 fault-injection tests); not re-measured with real weights (B) | GPU revalidation (Musaad) |
| 5 | `runs/manifest.json` not in Git (clean clone had no manifest) | `git ls-files` empty; root `.gitignore runs/` | Medium | all | Tracked (and the two candidate manifests); config only | V | — |
| 6 | 3 Nov T4 repeat (p50/p95/p99 51.1/70.9/84.8 ms) in deck but not report | Drive `phase2_closure_v3_20261103T081323Z_4820f5/closure_status.json` (879b0e8 + local timing patch) | Low | main.tex | Separate sentence under the benchmark table; extract `P2_closure_v3_20261103_extract.json` | V (checker) | — |
| 7 | ONNX final-detection agreement quoted from the confounded run only | `M8_onnx_parity_recheck_A9b_20261103.json` (Oxford 50: 100% same detections/classes, 0.00027 px); `M8_onnx_final_agreement_team50_20261103.json` (team 50: 0.00055 px) | Medium | main.tex | Row states both populations; 28.7 px kept as confounded; A9b still Fail | V (checker) | — |
| 8 | Team deck speaker notes wrong or stale | Notes cited mAP 0.9855, "YOLO11n-seg", "SAM-ViT-B 89.9x lower latency", "28.67 px … root cause not established", "100 tests, 99 passed", presenters differing from slide headers | High (said aloud) | Drive 04_Presentation | Corrected copy `SmartPetVision_Phase2_Presentation_v2_20261104.pptx/.pdf`: all 20 notes rewritten from report/evidence; slide 15 claim "FileNotFoundError on missing weights" (not in code) replaced by the preflight check; test counts by revision | V (PDF rendered and inspected) | Team adopts v2 (A) |
| 9 | Credential exposure | Pattern scan of 198 tracked files and 272 blobs from all branches, tags and PR heads (incl. ZIP members), positive control passed | High until revoked | all | `P2_security_scan_20261104.json`; no values printed | V: 0 findings; does not cover Drive, Colab outputs or W&B | Yasser: revoke old W&B key (A) |
| 10 | Gradio interface candidate | Drive `05_Demo/gradio_app_CANDIDATE_20261104.py`: calls `/analyze`, local-only URL, `share=False`, fails closed, shows both policy flags; field names match the API | Low | Drive only | None needed; optional demo aid | Static review only; not executed (gradio not installed) (P) | Optional |
| 11 | Presenter allocation vs "each member presents own work" | Deck slides 3–4 (detection) presented by the data/classification member | Low | deck | Suggestion in the rehearsal plan | — | Team (A) |
| 12 | Old status documents | `REQUIREMENTS_CHECKLIST_AR.md` (31 Oct) | Low | — | Marked superseded | V | — |
| 13 | A9b raw parity | 0.0018 (six settings 0.00198–0.00201) > 0.001 | Fixed criterion | all | Unchanged; PyTorch serves | V (checker) | No action; never relax |
| 14 | 17/52 multi-animal false-ready | f8manl0i | Known limitation | all | Unchanged; secondary review stays opt-in | H | Needs new independent data |

Phase 1 (checked, unchanged): binary Cat/Dog; scratch 0.906 vs fine-tuned 0.999 test accuracy; temperature fitted on validation only, fine-tuned temperature at its lower bound (disclosed). No change needed.

## 3. Change log (this branch, 4 November)

- `20fce7e` notebook 11 results in the report; evidence JSON; 13 submission checks.
- `982567d` release manifest generator; demo and rehearsal checklist; model card update; old checklist superseded.
- `2f0b202` report cites code commit `879b0e8`.
- `ee4a73f` optimiser corrected with provenance; four-box label set recorded; output-safety code and 6 tests merged; service manifests tracked.
- `3d725cb` 3 Nov T4 repeat; matched-preprocessing ONNX agreement with populations; candidate code changes stated in the report; security scan record.
- next commit: this tracker; demo checklist (multi-animal case, recovery steps, deck timings); manifest records checks, limitations and gates.

## 4. Verification log (this execution, CPU container, Python 3.13)

| Check | Command | Result |
|---|---|---|
| Unit tests, candidate | `PYTHONPATH=src python -m unittest discover -s tests` | 117 run, OK |
| Unit tests, reviewed archive `b1aff11` with repo `report/` restored | same, in the extracted archive | 108 run, OK |
| Report vs evidence | `python tools/verify_submission_evidence.py` | 33/33 |
| Report build | `submission/build.sh` (pdflatex) | see `RELEASE_MANIFEST.json` |
| Optimiser schedule fit | W&B GraphQL `lr/pg0` vs linear schedule | max rel. error 4e-6 |
| Credential scan | pattern scan, tree + history | 0 findings |
| CI | GitHub Actions on PR #5 | reported in PR |

Not run here: real-weight inference, T4 benchmarks, HTTP with real weights, Gradio, live demo, rehearsal (weights and GPU are not in this environment).


## 5. Second pass (4 Nov, evening): evidence for checks, versions, claims, demo

### 5.1 What the checks and tests cover (and do not)
| Revision | Unit tests (this container) | Checker | CI (GitHub Actions, pinned deps, torch 2.11.0+cpu) |
|---|---|---|---|
| `2f0b2026c6de11bd501556f39de73fc020573cd2` | 111 run, 0 failures, 0 errors, 0 skipped (isolated worktree; local torch 2.14.1/ultralytics 8.4.194, not the pins) | 29/29 | job 126774729468: success; tested merge commit `536852f` whose tree equals 2f0b202 (main `821fac1` is an ancestor) |
| `6011329…` | 117 run, 0/0/0 | 33/33 | job 126792173954: success; merge `597bf1a`, tree `bdb33a0` = 6011329 |
| `1cab83f501e872b840cafc08793c049061b71b40` (fresh clone, clean venv with the exact pins from PyPI) | 117 run, 0/0/0 | 35/35 | see PR #5 |

- Of the 29 checks at 2f0b202, 16 test the superseded free-form report (`report/main.tex`) and 13 test the submission report; at the current head 16 + 19.
- The checks compare selected numbers with evidence files and guard specific wordings (A9b Fail, 28.7 px confounded, optimiser, close-not-identical ONNX mAP, sweep limitations). They do not validate every number or claim.
- A separate number trace of the submission report found 243 distinct numbers; 215 matched an evidence value directly; the rest were section numbers, notebook names, computed values (Wilson intervals 0.945–0.969, 0.946–0.969, 0.691–0.768 recomputed and confirmed; percentages from counts) or Phase 1 dataset counts (confirmed in the Phase 1 report). Three items had no saved source (355/401, 372/378 and the 13/10/7 referral split on team photos) and were removed from the report.
- Unit tests are synthetic (no weights); they test code paths, not model quality.

### 5.2 Source versions and which measurements apply
| Version | What it is | Measurements that belong to it |
|---|---|---|
| 30 Oct snapshot = `a74784b` (src 30/30 SHA-256 match) | Code of the October training/evaluation notebooks and A8 benchmark | Training (2seovilv, 3ztpz2tc), Oxford test metrics, sweeps, A8 T4 benchmark, ONNX re-checks |
| `051bd2c` | Same `src/spv` as a74784b | CPU regression f8manl0i, CPU HTTP mu2tolkg, team rerun 2 Nov and 4 Nov 22:19 (401/81/386/17 reproduced) |
| `879b0e8` | Same `src/spv`; adds tools and tests | Notebook 11 (baselines, sweeps, per-stage p50/p95, ONNX test mAP, channel diagnosis); 3 Nov T4 repeat (with a local timing patch) |
| `2f0b202` | Same `src/spv`; report and checker only | No new measurement |
| `ee4a73f` … `1cab83f` | Adds output validation in `pipeline.py`/`serve.py` (+6 tests) | 4 Nov CPU smoke test only (functional, not a benchmark) |
| Source ZIP | `main.tex`, 16 figures, `private_ids.tex`; no code | Code is referenced by repository URL and commit `879b0e8` in the report |

`src/spv` is byte-identical from a74784b to 2f0b202 (git diff empty), so all historical measurements were produced by the same service and model code. Merging is not a submission requirement; the cited commits stay reachable through the branch and PR #5.

### 5.3 Claims re-verified in this pass
- ONNX: saved per-channel analysis (4 random inputs, seed 0, CPU): box channels 0.0018311, score channels 1.46e-7 (ONNX vs float32 PyTorch); vs float64: PyTorch 0.0018189, ONNX 0.0009632. Task level on Oxford test: mAP50 0.98252 vs 0.98357, mAP50–95 0.85176 vs 0.85000 — reported as close, not identical. A9b ≤ 0.001 unchanged; PyTorch served.
- Served vs validated settings (1,106 validation images, T4): match IoU 0.05/0.10/0.25 give the same counts (7 unmatched, 35 referred; per-image identity not saved); mask confidence compared only on mask mIoU (0.92240 at 0.10 vs 0.92216 at 0.25); referrals at 0.10 not evaluated.

### 5.4 Clean install and real-weight smoke test (executed)
- Fresh clone of `1cab83f`, Python 3.13.16 venv, `pip install -r requirements.txt` from PyPI (exact pins; torch 2.11.0 is the CUDA build running on CPU because download.pytorch.org is blocked here).
- Weights downloaded from the owner's Drive with approval; SHA-256 recomputed from the files and equal to the recorded values (detector f48e4d8e…, segmenter 868f3842…, classifier 22dd58d4…). Preflight `--strict-versions`: exit 0, no version mismatches.
- uvicorn 1 worker on 127.0.0.1: /health 200; CC0 cat image → 200, ready, cat, policy flags true/false, PNGs decoded; blank image → refer_to_staff (no_animal); text 415, empty 400, >15 MB 413. Evidence: `report/evidence/P2_cpu_smoke_test_20261104.json`. Recording of this run: `SmartPetVision_backup_recording_CPU_smoke_20261104.mp4` (container run, not the presentation host).

### 5.5 Presentation
- Files found: Drive `04_Presentation/SmartPetVision_Phase2_Presentation.pptx/.pdf` (team, 17:24); `06_Final_Submission/00_Originals_Before_Review_20261104/…Updated_20261104_20Slides.pptx` (earlier); an earlier online slide draft (superseded; stage-time label and 'same mAP' wording corrected in its local source, not republished).
- `SmartPetVision_Phase2_Presentation_v3_20261104.pptx/.pdf`: from the team deck; rounded-frame radius reduced so pictures no longer overlap frame borders (slides 3–9, 12); slide 14 now shows only sourced latency figures (removed untraced 'In-process CPU 100 img' 395.30/596.98/662.21 ms, mean 396.32, max 848.72, RSS 982/974 MB) and labels the stage means 14.2/9.3/17.6 ms as A8 benchmark means; slide 11 flags reason counts without saved per-photo records; slides 15–17 updated; notes say 'close, not identical'. All 20 pages rendered and inspected.

### 5.6 Privacy and public/private split
- Repository is public. Current tracked text: 0 student-ID matches; `SmartPetVision_Phase1.pdf` (tracked) contains the four IDs; history contains them in earlier `README_SUBMISSION.md`, `phase2/report/main.tex` and the earlier Phase 2 report PDF (counts only; values not printed). Removing them needs the owner's decision (file removal and/or history rewrite).
- Private instructor submission: report PDF and source ZIP (includes `private_ids.tex`), deck, recording. Public code release: the repository branch (no weights, no team photos, no IDs in current text except the Phase 1 PDF).

## 6. Drive release candidate (4 Nov, late evening)
- Target confirmed: `MyDrive/capstone /Submission/FINAL PHASE_2` (folder id `1FVTCMi8HK3EqSMURGccTiOHBABO4Iful`). The similarly named `capstone /smart-pet-vision/FINAL_QA_2026-10-25/Phase2_FINAL` (another owner, 25 Oct) is a different, older tree and was not touched.
- The proposed package is a new subfolder `REVIEW_CANDIDATE_<UTC timestamp>` there; existing files are left unchanged and are listed as current or historical in its `00_RELEASE/START_HERE.md`. Replacing any existing file needs the owner's approval.
- Optimiser provenance upgraded from inferred to direct: the executed training notebooks in Drive `02_Notebooks` print `AdamW(lr=0.001667, momentum=0.9)` (`report/evidence/P2_training_optimizer_direct_log_20261104.json`); their link to the W&B runs is still inferred (no run id printed).
- Report: the code row no longer says that later commits changed only the report (they added output validation); service command corrected (run from `phase2` with `PYTHONPATH=src`); model-card link points to the copy in the source archive (the `main` copy is older).
- Packaging: `submission/build.sh` now writes `reviewed_source.zip` (committed files, PDFs excluded, refuses if a private ID is found) and puts the same members under `code/` in the report source ZIP; `RELEASE_MANIFEST.json` records `code_archive.identical_to_source_zip_code_folder`. Tests: `tests/test_release_packaging.py`.
- `tools/smoke_test_http.py` added (the real-weight smoke test used on 4 Nov, plus a synthetic two-animal request and decoded-image checks).
- Rubric comparison of the two Phase 2 reports, the inventory and the verification logs of the final candidate are in the Drive candidate folder (`00_RELEASE/AUDIT_AND_ACTIONS.md`, `03_Verification/`).

## 7. Owner decisions of 4 Nov (23:25) and how they were applied
- **Submission location**: submit from the Drive candidate folder. The team's files in `01_Report` and `04_Presentation` belong to another account and are read-only for the owner account, so they are left as the team's record; a pointer file at the top of `FINAL PHASE_2` names the candidate folder as the submission package.
- **Duplicate upload**: the byte-identical duplicate PDF in the candidate folder was moved to the Drive trash (recoverable for 30 days).
- **Student IDs (public repository)**: the tracked `SmartPetVision_Phase1.pdf` is replaced by a redacted copy (the four IDs on page 1 replaced by "(ID)"; text of pages 2–16 identical; no ID in extracted text, raw bytes or decompressed streams). The submitted original stays in the team Drive. Git history is **not** rewritten: a rewrite would change every commit hash cited by the report, the evidence and the manifests (for example `879b0e8`), and GitHub keeps pull-request refs and cached views, so it would not remove the old objects reliably. Release assets of `phase1-v1.0` (checkpoint and result ZIPs) could not be inspected from this environment.
- **Checks scope**: the checker's summary line now states the split (superseded free-form report vs submission report).
- **Manifest**: open items are split into `submission_gates`, `presentation_gates` and `optional_improvements`; GPU revalidation and retraining are optional.
- **Reproducible PDF**: `build.sh` dates the PDF by the commit time, so rebuilding a commit gives the same PDF bytes.
- **GitHub**: the candidate is pushed to `phase2-gap-closure` (PR #5) and merged into `main` with a merge commit (no squash or rebase), so every cited commit stays in `main`'s history.

## 8. Polish candidate (branch `phase2-polish`, local, 5 Nov; not pushed)
Bounded improvement pass on top of `17ed5bf` (stable package unchanged in Drive `REVIEW_CANDIDATE_20261104T1951Z`). No training, no new model-quality evaluation, no threshold or acceptance change.
- **Demo page** `GET /demo` (`src/spv/demo_page.py`, route in `serve.py`): upload-and-analyse view over `POST /analyze` (original, detections with mask, mask, cutout; verdict with reason messages; both policy flags; no species/confidence for a `no_animal` referral; empty, >15 MB, unreadable and service-down states). No external resources; API contract unchanged (4 tests). Chosen over the teammate's Gradio draft in Drive `05_Demo/gradio_app_CANDIDATE_20261104.py` (left unchanged) because it needs no new dependency (Gradio is not in the pinned requirements), runs in the same process and shows the original photo and reason messages; the draft also shows species and confidence for `no_animal`.
- **Demo check with real weights** (CPU cloud container, Chromium; not the presentation host): 7 labelled inputs, results in `report/evidence/P2_demo_page_check_20261104.json`; 71 s backup screen recording with a caption on every frame. Observation: a CC0 coffee-cup photo gave one false detection (cat 0.495) and was referred.
- **Smoke test tool** also checks `GET /demo`.
- **Report** (`submission/main.tex`): summary box, a results-at-a-glance table (value, population, setting, source for each headline result), the demo page in the service rows, a demo-check paragraph and a short conclusion (demonstrated / not demonstrated / next steps).
- **Presentation**: technical Q&A sheet `docs/PRESENTATION_QA.md`; rehearsal plan with a 2:30 demo slot and hand-off sentences, presentation-host checklist and fallback procedure in `docs/DEMO_AND_REHEARSAL_CHECKLIST.md`; model card notes on the demo display rule and the out-of-scope detection.

## 9. Final polish pass (branch `phase2-polish`, local, 5 Nov; not pushed)
On top of `34d13c9`; both earlier Drive packages stay unchanged. No training, no new model-quality evaluation, no threshold, weight or acceptance change.
- **Demo page**: decision wording "Ready for staff review — not approved for publication" / "Staff review required — see reasons"; species of a referral marked provisional; a score (even 1.000) described as a model score; short out-of-distribution note; Clear button; one request at a time; a new file, an error or Clear removes the previous result and releases the object URL; messages for 413 (size or megapixels), 415, 503 and an unexpected response; keyboard-reachable file chooser, focus moved to the verdict, narrow-screen layout; raw JSON keeps the API fields with a note for `no_animal`. No approval button, no browser storage, no external request. `/analyze` unchanged.
- **Tests**: 15 behavioural browser tests (`tests/test_demo_ui_browser.py`, Playwright, stand-in model, synthetic images; skipped where Playwright is absent, as in CI) and 3 launcher tests; mutation checks (scratch) confirmed that removing the no_animal rule, the wording, the duplicate-submit guard, the stale-result clearing, the URL release or the keyboard-reachable input makes at least one test fail.
- **Run instructions**: `docs/QUICKSTART.md`, `docs/TROUBLESHOOTING.md`, launcher `tools/run_service.py` (prerequisites, 127.0.0.1 only, no installs or downloads, `YOLO_OFFLINE=1`). The smoke test now uses `httpx` (declared) instead of `requests` (transitive); `spv.deploy` and `tools/validate_release.py`, which produced recorded evidence, still use `requests` and were not changed.
- **Evidence**: slide-11 referral reasons recounted from the f8manl0i per-photo file in Drive; identical to the team's reconciliation file (`report/evidence/P2_f8manl0i_referral_reasons_recount_20261105.json`, aggregates only). ONNX evidence index `docs/ONNX_EVIDENCE_INDEX.md` (no change to A9b).
- **Report**: "Data and code availability" row; launcher and QUICKSTART in the service row; the f8manl0i per-photo file location stated; demo-page update and repeated real-weight page check noted.
- **Team documents**: one-page rehearsal checklist (planned, not held), blank read-through sheet, two Q&A entries (availability; score 1.000).


## 10. Owner decisions of 5 Nov (10:21) and how they were applied
- **Package and report**: the owner chose this candidate (branch `phase2-polish`) and the template report as the one Phase 2 report; the Phase 1-layout report in `FINAL PHASE_2/01_Report` stays as the team's record and is not submitted.
- **Team members table**: rows now follow the Phase 1 report order (Abdulrahman, Yasser, Mohammed, Musaad) with the Phase 1 workstreams. Each member still reads the report and confirms their line; the read-through sheet stays blank until they fill it in.
- **Removed from the tracked tree** at the owner's decision: an unapproved draft document and a local working-notes file (both kept privately by the owner, not deleted). History is not rewritten.
- **A9b re-run** (new export, unchanged settings, CPU container): 0.0018 with one PyTorch thread, 0.0023 with two; not met. Stage-by-stage diagnosis in `report/evidence/M8_onnx_parity_recheck_A9b_20261105.json` (tool `tools/a9b_root_cause.py`); summary in `docs/ONNX_EVIDENCE_INDEX.md` ("Why A9b fails"). Bound unchanged; PyTorch serves.
- **Student IDs in Git history**: kept. Rewriting history would change the hash of every commit the report and evidence cite (for example `879b0e8`, `17ed5bf`, `cdcce9e`) and would not remove copies GitHub keeps for pull-request refs; current tracked files contain no IDs (the Phase 1 PDF in Git is the redacted copy).
- **W&B key**: revocation is done by the key owner in W&B account settings; a scan cannot show it (`P2_security_scan_20261104.json`: no credential pattern in any of the 31 commits on GitHub).
- **GitHub `main` merged in** (PR #6 report wording fixes, PR #7 repository README); in the one paragraph both branches changed, the correction about the per-photo file location and the reworded sentence are both kept. The project guide file and a vendored third-party design guide (`tools/skills/unslop`, not used by the project code) were removed from the tracked tree at the owner's decision and kept privately.

## 11. 6 Nov: merge of the segmentation workstream's 5 Nov additions into the template report (owner decision, 5 Nov 23:57)
- Source: files Mohammed uploaded to `FINAL PHASE_2` on 5 Nov (his `Final Phase 2.pdf`, 35 pages, is a fork of this report plus Sections 4.8/4.9; its `00_PHASE2_DOCTOR_REQUIREMENTS_AUDIT_20261105_AR.txt` quotes a PDF hash `65accd43…` while the uploaded PDF is `5565a682…`, and `REPORT_QA_FINAL.json` says 34 pages; his files are owned by him and were left unchanged).
- Carried over (verified): M5 Kaggle regression (means recomputed from the per-image CSVs of archive `2812a348…`; identical to the historical values; new with/without-border comparison on the same 300 IDs; SAM checkpoint identity), the corrected-label detector candidate (identity and operating-point audit files; test mAP as reported by the run), and the recall-by-size erratum (Oxford column had used COCO pixel bins; relative-bin recount 4/1,102, sum of found = 1,061 TP). New Section 4.8 of the report; three new evidence files; three checker rules (41/41).
- Not carried over: his Colab "derivative" T4 run (Section 4.8 in his report) because its archive was never delivered (his own appendix says so) and the Kaggle run supersedes it; a chronology sentence asserting a conflict between topic approval (19 Oct) and the first run (17 Oct) — both dates were already stated in this report and the owner decides how to present them; a citation and wording that attribute the work to automated agents (owner decision: no such traces); criterion labels A1/A2 used for A3/A4.
- Code unchanged: `spv.detect.recall_by_size` still computes COCO pixel bins (documented); the report names the definition it uses.

## 12. 6 Nov: full review against the guidelines and the example report; delivered commit pinned
- Review inputs: `Capstone_Project_Guidelines_UPDATED.pdf` (Tables 2, 5, 6, 7; R1–R10; §2.4) and the instructor's Phase 2 example report (18 pages). The submission already follows the example section by section; no rubric item is missing. Result: `docs/REQUIREMENTS_CHECKLIST_2026-11-06.md` (every requirement with status, evidence and owner; open items listed in its section 8).
- Report changes (`submission/main.tex`): (1) the exact delivered commit is stamped into the PDF by `build.sh` (`delivery_commit.tex`, generated and ignored; the build refuses a dirty `phase2/` tree), and Section 1 names it next to the results commit `879b0e8`; the two links that pointed to branch `main` (manifest, model card) now point to the results commit and to the delivered commit, so the PDF does not imply that `main` matches the package before PR #8 is merged (owner's point, 6 Nov 00:44). (2) A "Bonus evidence" row names the approval date and the team-collected set with its dataset-card section (rubric Table 6, bonus rows). (3) Three checklist items added (delivered commit; model card and go/no-go; notebooks checked). (4) The Section 1 field table is now an `xltabular` so it can break across pages (as a `tabularx` it had grown taller than a page and left pages 1–2 blank in a trial build). PDF 26 pages.
- Notebooks: none of the 15 can execute in this container (Drive, Oxford images, team photos, GPU or the W&B key are needed). New `tools/check_notebooks.py` (static: nbformat validity, syntax of every code cell, resource needs) with 4 unit tests; result 15/15 parse, `report/evidence/P2_notebook_static_check_20261106.json`. Inventory with executed-run evidence: `docs/NOTEBOOK_EXECUTION_PLAN.md`. Notebook 07 has no execution record in the repository and is listed as not run; notebook 11 Parts F and G remain optional and not run.
- Checks on this tree (6 Nov, before commit): 147 unit tests OK (Playwright installed), checker 41/41, ruff CI selection clean, preflight and launcher check pass, real-weight CPU HTTP smoke test (ready, ready, multiple_animals, no_animal; HTTP 200; policy flags correct).
- Not changed: weights, thresholds, acceptance criteria, A9b status, any result. Drive package `FINAL_6bd2420_20261105T2119Z` stays as it is until the successor package is uploaded; GitHub `main` unchanged (PR #8 awaits the owner's merge).

## 13. 6 Nov (evening): Phase 1 feedback (38/40) worked into the submission; honest statuses; in-place Drive update
- Inputs: the instructor's Phase 1 feedback sheet (Part 2 leakage −1; Part 13 independent reproduction −1; dated criteria; commit hash per run; harder test set; imbalance and per-class recall) and the owner's brief of 6 Nov 04:44 (fix in place, one requirement at a time, never mark unmet items complete).
- Leakage (Part 2): `tools/same_animal_groups.py` + 6 tests + frozen `docs/LEAKAGE_SAME_ANIMAL_PROTOCOL.md`; not executed on the images (dataset and ImageNet weights refused by the network policy, HTTP 403); all results remain split v1; retraining on v2 awaits approval. PARTIAL.
- Reproduction (Part 13): `tools/reproduce_phase2_eval.py` + 3 tests, `docs/INDEPENDENT_REPRODUCTION_PROCEDURE.md`, `REPRODUCTION_LOG.txt` header stating that no independent reproduction exists. BLOCKED on a teammate.
- Dated criteria: Drive metadata located an unexecuted integration-notebook copy with all ten criteria, Drive timestamp 18 Oct 11:19 UTC (`P2_prespecification_evidence_20261106.json`); the report's "written 15 Oct" claim replaced by this evidence and its limits. Provenance helper `src/spv/provenance.py` + 4 tests.
- Harder test set: `docs/INDEPENDENT_TEST_SET_PROTOCOL.md` (not started). Imbalance: stated as not done; controlled comparison needs approval.
- Report: new §1 subsection "Response to the Phase 1 feedback (38/40)"; criteria paragraph rewritten; bonus row reworded as team-stated with the 17/19 Oct conflict named; 28 pages. Model card and dataset card carry the same statements. Master checklist rewritten in the mandated format (`docs/REQUIREMENTS_CHECKLIST_2026-11-06.md`). The earlier wording "no rubric item was missing" is withdrawn: coverage was reviewed and the open items are listed.
- Not done, deliberately: no notebook edited; no training; no threshold or weight change; A9b unchanged; no Drive file deleted; Gmail not available in this session for the approval message.


## 14. 6 Nov (05:30–): external review of the f5bad5b package worked into the same files
- Inputs: the reviewer's notes on the f5bad5b summary (priorities 1–7 and four audit points: test counts per environment, the two archives, the old issues, the grouping tool), and the owner's instruction to close every point in its existing file (Drive, GitHub, W&B) without new folders.
- GitHub reference (priority 4): found that the owner merged PR #8 (head `81cd4fd`) into `main` at 01:14 UTC on 6 Nov (merge commit `bcc8d13`, CI green), after which `f5bad5b` had been pushed to `phase2-polish` with no open PR, so no CI ran on it. Opened PR #9 (`phase2-polish` → `main`) so that CI runs on the delivered commit; merging stays the owner's decision. Checklist S3 rewritten: the delivery is the commit printed in the PDF, not `main`.
- Package integrity (priority 3): the 9.25 MB `SmartPetVision_Phase2_source.zip` was downloaded from Drive on the owner's computer (file id unchanged), copied into the session and hashed: SHA-256 `afe9ffe279dec141a2046337adfbc04a4d83100d797aecaaf2a403b6549b5277`, equal to `SHA256SUMS`; `delivery_commit.tex` inside it names `f5bad5b`. The size-only entry of the previous readback is superseded. The same download-and-hash step is repeated for every later upload.
- Audit points: test counts now reported per environment (160 run / 0 skipped with Playwright and Chromium; 145 run + 15 skipped without Playwright), never merged; a table ties each archive name to its own size and hash source; new checklist section J keeps the four loose boxes, the 17/52 multi-animal photos and A9b visible with their report locations; the grouping tool's tests are stated to prove the algorithm on a toy set only.
- Team table: the owner stated on 6 Nov that the four members approved the "Team members" table as printed; recorded as the owner's statement (no signature file). Priorities 5–7 (deck, demo, recording, rehearsal, approval message, W&B key) are unchanged human actions; the owner ranked them below the instructor-facing files.
- Not possible here: the same-animal grouping on the images (dataset and ImageNet weights still refused by the network policy, HTTP 403, re-tested 6 Nov) and the independent reproduction (a teammate's action). All results remain split v1.

## 15. 6 Nov (12:14–17:30): same-animal grouping executed on the images; review aids; deck v9 candidate; no training

- Owner instruction (12:14 "نفذ") executed: `tools/same_animal_groups.py` run once on the 7,335 images in Colab (T4, code and requirements taken only from the delivered package `02_Code/reviewed_source.zip`, hash verified; split v1 rebuilt from `runs/data/metadata.csv` and checked 5,123/1,106/1,106 before the run). Result under the frozen rule: 6,541 similarity groups; 86 groups / 771 images cross the v1 partitions (test 135, val 127, train 509). Record: `report/evidence/P2_same_animal_grouping_run_20261106.json`; per-group table `report/evidence/identity_v2_20261106/crossing_groups_summary.csv`; full outputs in Drive `runs/data/identity_v2_20261106T0931Z/`.
- Owner instruction (12:46): no training; split v2 kept as an unapproved proposal. Review aids produced in the same Colab session (`review_crossing_groups.py`, Drive `06_Team_Actions`): 86 contact sheets in priority order (groups touching test first, then size, weak bridges, links near the threshold), chain analysis (13 groups of 10–114 images hold 574 of the 771 images; 12 groups have weak bridges below 0.92), an unfilled review log (no verdict recorded on anyone's behalf), a diagnostic sensitivity at 0.95 (18 groups / 115 images; the frozen threshold is unchanged) and a split-v2 integrity check (7,335 unique rows, 0 crossing groups, 4,903/1,382/1,050; val inflated by the chain groups).
- Wording fixed everywhere: "similarity groups crossing partitions under the protocol; human identity review not complete". The 771 images are not described as confirmed identity leakage.
- Documents updated in this revision: report §1 (Part 2 row), `docs/dataset_card.md`, `docs/model_card.md`, checklist A1, this section. Not changed: any weight, threshold, split used by the delivered checkpoints, A9b.
- Presentation (candidate, outside the frozen package manifest): deck v9 in Drive `04_Presentation` (new slide 3 with the feedback response in three columns fixed / disclosed only / not done; delivered commit and CI separated from the results commit on slides 1, 16, 19 and 21; the unsupported "previously exposed W&B key" sentence of v8 replaced by the neutral scan result; timings +1:00 after slide 3; notes and footers updated) with `SPEAKER_NOTES_v9.md` and `REHEARSAL_CHECKLIST_ONE_PAGE_v9.md`; v8 files untouched. Team review and a timed rehearsal are still pending.
- Still not done, and not claimable: human identity review, approved split v2, retraining, independent reproduction by a teammate, timed rehearsal, demo recording.
