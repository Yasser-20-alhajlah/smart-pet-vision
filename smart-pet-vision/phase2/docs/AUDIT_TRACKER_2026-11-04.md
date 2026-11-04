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
| claude.ai Slides artifact deck | `LEUxXcJyLZzXV83uY2tiqt` | Not the presentation of record (S) |

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
