# Requirements checklist — Phase 2, final presentation and bonus (status 6 November 2026)

Sources: *Capstone Project: Guidelines and Example Topics* (October 2026; page numbers below),
the instructor's Phase 2 example report (BoardSight, 18 pages) whose section structure the
submission follows, and the instructor's messages recorded in the audit tracker. This list
supersedes `REQUIREMENTS_CHECKLIST_AR.md` (31 October, superseded) and extends the package
`REQUIREMENTS_CROSSWALK.md` (5 November).

Status values: **Met** (verified in the repository or the evidence files today), **Met, human
step open** (the artefact exists; a team member still has to confirm or act), **Open** (not done),
**Optional**. "Where" names the report section (`submission/main.tex`, PDF
`SmartPetVision_Phase2.pdf`) or the file. Verification names the command or file read.

## 1. Phase 2 deliverables and metrics (guidelines Table 5 p.6, rubric Table 6 p.7; 45 points)

### M4 Object detection (15 points)

| # | Requirement | Where | Verification | Status |
|---|---|---|---|---|
| M4-1 | Annotations converted and verified visually | §2.1: conversion table, split table, data checks, label-check figure | `tools/verify_submission_evidence.py` (41/41); `M4_label_roundtrip.json` | Met |
| M4-2 | Detector fine-tuned (runs with W&B links; selected model and why) | §2.2: baseline + two runs, training curves, optimiser record | W&B `48vh56vh`, `2seovilv`; `P2_training_optimizer_*` | Met |
| M4-3 | Sweep of confidence and NMS thresholds with a justified operating point, chosen on validation | §2.3: 24-setting sweep table and figure, rule stated | `M4_threshold_sweep.csv` | Met |
| M4-4 | mAP50, mAP50–95, per-class AP on the test split (not at the deployment threshold) | §2.4 table; mAP at conf 0.001, P/R/F1 at the operating point | `M4_detection_test_metrics.json` | Met |
| M4-5 | Precision, recall, F1 at the operating point, with intervals | §2.4 (Wilson 95 % intervals) | checker | Met |
| M4-6 | Breakdown of false positives; recall by object size | §2.4–2.5: FP taxonomy table; size table (relative bins, erratum stated) | `M4_fp_breakdown.csv`; `P2_size_table_erratum_20261105.json` | Met |
| M4-7 | Galleries of false positives and false negatives | §2.5 figure (two galleries) | figures present in `submission/figures/` | Met |
| M4-8 | At least one failure case analysed; main causes; next fix | §2.5 (`chihuahua_156`; localisation as next fix) | report text | Met |
| M4-9 | Inference on images and video at the operating point | §2.6: 401 team photos; two CC-licensed clips with FPS and frame counts | `M4_team_detection_metrics.json`, `M4_video_inference.json` | Met |

### M5 Segmentation (15 points)

| # | Requirement | Where | Verification | Status |
|---|---|---|---|---|
| M5-1 | Masks generated or refined with SAM and verified visually; ground truth described | §3.1: trimap-derived body masks (ground truth); SAM 2.1-B oracle-box masks as baseline; visual gallery | `M5_gt_mask_check.jpg`; W&B `1jb5y453` | Met (adapted: trimaps are the ground truth, SAM is the baseline; stated in §3.1) |
| M5-2 | Fine-tuned segmentation model, runs table with two baselines, selected run and why | §3.2 | W&B `kd5ksgt3`, `3ztpz2tc` | Met |
| M5-3 | Thresholds and post-processing chosen on validation | §3.3 (mask confidence, match IoU, minimum share; served vs validated values stated) | `P2_gap_experiments_20261102.json` | Met |
| M5-4 | Mask mAP50–95, mean IoU, Dice on the test split, per split and per class | §3.4 tables | `M5_yolo_seg_test_metrics.json`, `M5_yolo_seg_coco.json`; Kaggle recomputation `P2_m5_kaggle_regression_20261105.json` | Met (per-class mask AP not recorded; stated) |
| M5-5 | Application measure | §3.4 "Application measure" | `P2_cpu_release_validation_f8manl0i.json` | Met (counts of mask-based referrals; agreement counts not computed, stated) |
| M5-6 | Error analysis with pattern table and at least one failure case | §3.5 (18 lowest-IoU gallery, pattern table, pug case) | figures present | Met |

### M8 Integration and deployment (15 points)

| # | Requirement | Where | Verification | Status |
|---|---|---|---|---|
| M8-1 | End-to-end pipeline with per-stage timing | §4.1 (p50/p95 per stage, two measurements kept apart) | `P2_gap_experiments_20261102.json`; `M8_benchmark_pytorch.json` | Met |
| M8-2 | Packaged pipeline with a manifest (model files + checksums, class order, input size, thresholds, versions) | §4.2; `runs/manifest.json`, `evidence/M8_manifest.json`; link pinned to commit `879b0e8` | `tools/preflight_release.py --strict-versions` (pass, 6 Nov) | Met |
| M8-3 | Input and output contracts | §4.3 | unit tests (`test_release_inputs.py`, `test_output_safety.py`) | Met |
| M8-4 | ONNX export of the principal model with a numerical-parity check | §4.4: classifier passes (A9a); **detector fails A9b** (0.0018–0.0023 vs 0.001), bound unchanged, cause diagnosed, PyTorch served | `M8_onnx_parity*.json`; `docs/ONNX_EVIDENCE_INDEX.md` | Met as a check; A9b recorded as FAIL |
| M8-5 | Working HTTP service (framework, endpoints, start-up checks, example request, figure) | §4.5; `GET /demo`, `tools/run_service.py`, `docs/QUICKSTART.md` | real-weight CPU smoke test 6 Nov (ready / ready / multiple_animals / no_animal, HTTP 200); 147 unit tests incl. 15 browser tests | Met |
| M8-6 | Benchmark: p50/p95/p99 latency, throughput, memory per backend, with warm-up and stated conditions | §4.6 table and figure (T4 and CPU rows never mixed) | `M8_benchmark_*.json`, `P2_cpu_http_validation_mu2tolkg.json` | Met |
| M8-7 | Verdict against acceptance criteria fixed in advance; for each FAIL: why, fix, cost; go/no-go | §1 criteria table (A1–A9b, fixed in Phase 1 / before the first run); §4.7 verdict table and decision | `M8_acceptance.csv`; checker | Met |
| M8-8 | Failure gallery and model card (failure modes, out-of-scope uses) | §2.5, §3.5, §4.7 last paragraph; `docs/model_card.md` ("Out of scope" section) | file read 6 Nov | Met |
| M8-9 | Live demonstration | `GET /demo` page; checklist `docs/DEMO_AND_REHEARSAL_CHECKLIST.md` | run in a CPU container, not yet on the presentation laptop | Met in software; **laptop run and recorded fallback open** (see P5, P6) |

## 2. Report form (example report structure; guidelines §4 p.6)

| # | Requirement | Where | Status |
|---|---|---|---|
| F-1 | Section 1: field table, team members (name, ID, workstream, contribution), points by team size, system overview figure + component table, acceptance criteria, follow-up on the Phase 1 data plan | §1 (student IDs from the untracked `private_ids.tex`) | Met, human step open (each member confirms their line) |
| F-2 | Sections 2–4 follow the example's subsections; Section 5 omitted (team of 4) | §2.1–2.6, §3.1–3.5, §4.1–4.7 (+ §4.8 regression runs, Conclusion) | Met |
| F-3 | Obstacles table (obstacle, module, impact, action, status); support needed | §5 | Met |
| F-4 | Submission checklist | §6 (17 items, all ticked) | Met |
| F-5 | Self-assessment with strongest evidence and known gap | §7 (41/45) | Met |
| F-6 | Every reported figure states split, input resolution and thresholds | every table caption/heading in §1–4 | Met for the tables and figures read on 5–6 Nov; each member re-checks their section during the read-through |
| F-7 | Code repository and commit hash: the delivered commit named explicitly; links pinned to commits, not to `main` | §1 (delivered commit stamped by `build.sh`; results commit `879b0e8`) | Met (6 Nov) |
| F-8 | Neutral academic voice; no invented results, names, roles or reviews | whole report; checker + read-through | Met for the automated checks; human read-through open |

## 3. Data policy and proposal requirements (guidelines §2.4 p.3, §3.1 p.4)

| # | Requirement | Evidence | Status |
|---|---|---|---|
| R1 | Computer vision as the core | three vision models; text only as output | Met |
| R2 | Publicly available annotated data (classification + boxes) | Oxford-IIIT Pet (trimaps → body boxes, documented in §2.1) | Met |
| R3 | Licence, privacy, ethics: no data in the repository, no redistribution, consent for team photos, no surveillance or medical claims | §1 "Data and code availability"; `docs/dataset_card.md`; `.gitignore`; archive scan (no photos, no credentials) | Met |
| R4 | Dataset not used in the course labs | Oxford-IIIT Pet | Met |
| R5 | Complete Phase 1 (split, dataset card, scratch vs fine-tuned, W&B) | Phase 1 report (submitted 28 Oct) | Met (Phase 1) |
| R6 | Detector with threshold analysis, mAP, FP/FN; segmentation against ground-truth masks; each with a function in the system | §2, §3; component table in §1 | Met |
| R7 | End-to-end pipeline, ONNX export with parity, HTTP service, benchmark, failure analysis, model card | §4 | Met (A9b fails and is reported as such) |
| R8 | Intended users, decision supported, quantitative acceptance criteria fixed before results | §1 summary and criteria table (A1–A9b) | Met |
| R9 | Feasibility | completed | Met |
| R10 | Four workstreams, one per member, each presents their own work | §1 team table; deck presenter plan | Met, human step open (presentation) |
| D-1 | Data must not be committed to code repositories or redistributed | no images in Git; weights in Drive | Met |
| D-2 | Personal data avoided; faces/homes: consent obtained | team photos private; consent stated in the dataset card | Met |

## 4. Final presentation (rubric Table 6 p.7, §6 Table 7 p.8; 15 points)

| # | Requirement | Where | Status | Owner |
|---|---|---|---|---|
| P1 | Failure analysis and decision (8 pts): failure gallery with explicit statements; go/no-go against criteria defined in advance; model card with out-of-scope uses | report §2.5, §3.5, §4.7; deck v8 | Met in documents | presenters |
| P2 | 15–20 minutes; clear line from problem to decision; structure of Table 7 (problem/users 2 min, data 2–3, Phase 1 2–3, Phase 2 5–6, demo 2–3, decision 2–3) | deck v8 (plan 18:30); `docs/REHEARSAL_CHECKLIST_ONE_PAGE.md` | **Open**: no timed rehearsal yet (owner decision: after the review) | all four |
| P3 | Every member presents their own contribution | presenter plan in the deck notes | Met, human step open | all four |
| P4 | Live demonstration on a clean runtime | `tools/run_service.py` + `/demo`; `docs/QUICKSTART.md` | **Open** on the presentation laptop (passed in a cloud CPU container 4–6 Nov) | demo presenter |
| P5 | Recorded fallback | script in Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md` | **Open** (to be recorded on the presentation laptop) | demo presenter |
| P6 | Figures reported with context (split, resolution, thresholds) on the slides | deck v8 | Met (slide-11 recount evidence 5 Nov) | — |

## 5. Bonus (rubric Table 6 p.8; up to +10)

| # | Requirement | Evidence | Status |
|---|---|---|---|
| B1 | Original project approved by the instructor before work began; R1–R10 met; all deliverables complete (+3) | approval 19 Oct 2026 stated in §1 "Topic" and "Bonus evidence"; the approval record itself is with the instructor | Met as stated by the team; chronology note: §1 also dates the first Phase 2 run 17 Oct (criteria A3–A9a written 15 Oct) — the owner should confirm the dates before submission |
| B2 | Team-collected images annotated by the team, processed with the full M1 protocol, documented in the dataset card, used in the evaluation (+7) | 401 photos / 503 boxes / masks; `docs/dataset_card.md` "Team-collected set" (collection, M1 processing, CVAT annotation, audit); used in §2.6, §3.4, §4.7; named in §1 "Bonus evidence" | Met |

## 6. Submission mechanics and instructor messages

| # | Requirement | Status | Owner |
|---|---|---|---|
| S1 | Upload `SmartPetVision_Phase2.pdf` and `SmartPetVision_Phase2_source.zip` by 15 Nov 2026 (one member) | **Open** | one member |
| S2 | Repository public (private repositories receive 0) | Met for `Yasser-20-alhajlah/smart-pet-vision` (GitHub API, 5 Nov); members' own repositories not checked | each member |
| S3 | GitHub reference version: the delivered commit is named in the PDF and in `CODE_ARCHIVE_INDEX.json`; `main` matches it only after PR #8 is merged | commit named (6 Nov); **merge of PR #8 open** (owner's click after review) | owner |
| S4 | No contradictions or unfinished notes; every member reads the submission | automated checks pass (checker 41/41, placeholder scan); **human read-through open** (`docs/TEAM_READTHROUGH_SIGNOFF.md`) | all four |
| S5 | Each member confirms their line in the team table | **Open** | each member |
| S6 | Student IDs: not in tracked files (`private_ids.tex` untracked; Phase 1 PDF in Git redacted); kept in Git history by decision (cited hashes) | Met (decided 5 Nov) | — |
| S7 | Previously exposed W&B API key revoked | **Open** (owner's action in W&B settings; cannot be verified from the repository) | owner |
| S8 | Notebooks: all 15 parse; executed runs cited by W&B run and evidence file; Part F / Part G of notebook 11 optional and not run | Met (`docs/NOTEBOOK_EXECUTION_PLAN.md`) | — |
| S9 | Checks after any change: 147 unit tests, checker 41/41, ruff CI selection, preflight, real-weight smoke test | Met on the delivered commit (6 Nov) | — |

## 7. Phase 1 (already graded; 40 points) — unchanged

Phase 1 was submitted on 28 October. Its deliverables (M1 dataset card and leakage-safe split, M2
scratch model, M3 fine-tuned model with W&B, reproducibility) are referenced, not re-submitted. An
optional resubmission with a change list is drafted in the Drive package
(`06_Team_Actions/PHASE1_RESUBMISSION_CHANGES_DRAFT.md`); no decision has been taken.

## 8. Open items in one list (what remains before 15 November)

1. Owner: merge PR #8 after review, so that `main` equals the delivered commit (S3).
2. Owner: revoke the old W&B key (S7); confirm the 19 Oct approval date against the 17 Oct first run (B1).
3. Each member: confirm their team-table line and sign the read-through sheet (S4, S5).
4. Demo presenter: run `tools/run_service.py` on the presentation laptop, record the fallback (P4, P5).
5. All four: one timed rehearsal, 15–20 minutes (P2).
6. One member: upload the PDF and the source ZIP from the Drive package (S1).
7. Optional: notebook 11 Part F (second detector recipe) and Part G (W&B logging) with the owner's approval.
