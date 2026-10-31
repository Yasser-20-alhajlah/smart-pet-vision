# Reproducing Smart Pet Vision

Goal: a team member reruns the whole project on a clean Colab runtime and gets the numbers below.

## Steps
1. Open Colab, Runtime > Change runtime type > T4 GPU.
2. Make W&B available: run `wandb login` once in the session, or store the key as a Colab secret named `WANDB_API_KEY` and load it with `from google.colab import userdata; os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")` (never paste the key into a cell or a file).
3. Add the shared folder `capstone ` to My Drive (a shortcut is enough), so that `smart-pet-vision` and the team-data folder `SmartPetVision_TeamData` sit side by side, e.g. `MyDrive/capstone /smart-pet-vision` and `MyDrive/capstone /SmartPetVision_TeamData`. The team data are kept outside the repository and found through `paths.team: ../SmartPetVision_TeamData` in `configs/config.yaml`. The setup cell of every notebook mounts Google Drive first, then finds the folder and checks that `configs/config.yaml` and `src/spv` exist; if it finds no copy or more than one, it stops and asks you to set `SPV_REPO` to the exact path.
4. Running a notebook end to end retrains and overwrites `runs/`. To reproduce without touching the submitted results, copy the folder without `runs/` (for example to `/content/repro/smart-pet-vision`), copy or link `SmartPetVision_TeamData` next to it (`/content/repro/SmartPetVision_TeamData`), and set `os.environ["SPV_REPO"]` to the copy before running. The 23 Oct 2026 reproduction was run this way, when the team data were still inside the repository.
5. Run notebooks 01, 02, 03, 04 and 05 in order. Measured on a T4 on 23 Oct 2026: 01 about 4 min, 02 about 41 min, 03 about 60 min, 04 about 83 min, 05 about 15 min.
6. Compare `runs/acceptance.csv` and the files below with the expected values.

## Expected values (run of 18-19 October 2026, seed 42)
| File | Value | Tolerance |
|---|---|---|
| runs/data/dataset_card_values.json | 7,335 images; split 5,123 / 1,106 / 1,106; leakage 0 | exact |
| runs/classification/comparison.csv | fine-tuned test accuracy 0.999; scratch 0.906 | +/- 0.01 (scratch +/- 0.03) |
| runs/detection/detection_test_metrics.json | mAP50 0.984; mAP50-95 0.850 | +/- 0.01 |
| runs/segmentation/yolo_seg_test_metrics.json | mIoU 0.923; Dice 0.959 | +/- 0.01 |
| runs/deploy/benchmark_pytorch.json (24 Oct section of notebook 05) | PyTorch p95 about 60 ms on T4 (58 ms in the 18-19 Oct run, 61 ms on 24 Oct) | depends on GPU load |
| runs/deploy/team_metrics.json | accuracy 0.970; referral rate 0.152 | +/- 0.01 |

GPU training is not guaranteed to be bit-exact across environments, so small differences within the tolerance are acceptable.

## W&B runs
- Submitted classification runs (group `full-rerun-2026-10-18`): scratch https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/gh7ayx6h, frozen https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/xsz5afr9, fine-tuned https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/8nvxbsn2
- Submitted detection, segmentation and deployment runs (same group): detector https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/2seovilv, YOLO11s-seg https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/3ztpz2tc, SAM baseline https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/1jb5y453, deployment and team set https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/fqow5ly7. Earlier runs of 17 Oct are tagged `superseded-by-full-rerun-2026-10-18`.
- Reproduction (classification runs): https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/groups/reproduction-2026-10-23
- Detection, segmentation and integration results of the reproduction are saved as files in `reproduction/2026-10-23/` (executed notebooks and the `runs/` JSON and CSV files).

## Reproduction record
| Reproduced by | Date | Environment | Result |
|---|---|---|---|
| Yasser Alhajlah | 2026-10-23 | Colab, Tesla T4, clean runtime; copy of the repository without `runs/` (split rebuilt from scratch); notebooks 01-05 | All notebooks finished (rc = 0) in 3 h 23 min. Dataset card values, split, comparison.csv, detection (mAP50 0.9836; TP/FP/FN 1061/46/45), segmentation (mIoU 0.9230; Dice 0.9586) and team metrics (accuracy 0.970; referral 0.152) identical to the submitted run; PyTorch p95 58.6 ms (submitted 58.3 ms); acceptance 9/10 as submitted (A9b detector ONNX parity 0.0018 > 0.001). |

## Changes after the reproduction (24 Oct 2026)
These changes follow the internal technical review. None of them changes a model, a split or a metric.
- Setup cell of notebooks 01, 02, 03 and 05: mounts Drive before finding the project (the same cell notebook 04 already used).
- Notebook 01: the review summary reads `ok` as 1/0 (y/n still accepted) and prints the completed audits; the review instructions were replaced by a description of what was done.
- Notebook 02: the draft instruction in the error-analysis cell was replaced by the final analysis.
- `src/spv/data.py`: an unreadable trimap is now dropped with the reason `unreadable_trimap` instead of stopping the run. All 7,349 trimaps were readable in both runs, so the cleaned set is unchanged.

## Phase 2 additions (24 Oct 2026)
These follow the internal review of Phase 2. No model was retrained and no test metric changed; the new cells read the saved weights and splits.
- Notebooks 03, 04 and 05 end with a section added on 24 Oct 2026, executed on a T4 the same day: per-class AP50-95 and P/R/F1, the classified error breakdown, a round-trip check of all exported YOLO labels, the detector on the team photos and on two Wikimedia Commons clips (03); a ground-truth mask check and SAM-trimap agreement (04); the service under uvicorn over HTTP with the PyTorch manifest, a benchmark of each backend in its own process, ONNX parity on real images, the acceptance table with A7 and A8 on the deployed PyTorch backend, the team evaluation rerun with the corrected labels (identical results), the referral analysis and a system failure gallery (05).
- The draft "For the report" cells of notebooks 03 and 04 were replaced by final text.
- `src/spv/detect.py`, `segment.py` and `deploy.py`: functions added for the analyses above; existing functions unchanged.
- `runs/acceptance.csv` was rewritten by the new section of notebook 05 (A7 from the uvicorn test, A8 from the PyTorch benchmark); the earlier version is kept in `reproduction/backup_before_phase2_fixes_2026-10-24/`.

## Team data moved out of the repository (26 Oct 2026)
The team-collected photos, masks, annotations and label files moved from `data/team/` inside the repository to the folder `SmartPetVision_TeamData` next to it, so that no data is committed to the repository. `configs/config.yaml` now sets `paths.team: ../SmartPetVision_TeamData`; the files inside the team folder are unchanged (a `README.md` was added), and no model, split or metric changed. Unprocessed originals moved to the private folder `Team_Collected_Dataset`. The previous configuration is kept in `reproduction/backup_before_teamdata_move_2026-10-26/`. The 23 Oct reproduction above used the old location `data/team/`.


## Execution review and regression commands

Use a copy of the project and preserve historical runs. The current source layout is src/spv, with tools and tests at the root. Run `PYTHONPATH=src python -m unittest discover -s tests -v` in the documented validation environment. Model-independent imports are lightweight; inference and training still require the full model dependencies. A skipped PyTorch construction test is reported honestly when PyTorch is absent.

For trained-weight CPU regression, set SPV_REPO in notebook 06 or run `python tools/validate_release.py --manifest runs/manifest.json --images ../SmartPetVision_TeamData/images --labels ../SmartPetVision_TeamData/labels.csv --output review_runs/NEW_RUN`. The command forces CPU visibility and refuses an existing output directory. Re-run the actual HTTP server, ONNX parity and isolated T4 benchmarks separately after the code changes. Passing ASGI tests with mocked models does not replace those checks.

`tools/audit_saved_evidence.py` recomputes saved-record arithmetic with explicit reference/archive inputs. `tools/redact_evidence_gallery.py` requires the exact original gallery SHA256 and creates a new lossless PNG, preserving all pixels outside explicit rectangles. Review new gallery coordinates manually if an image hash changes.


## Latest trained-weight repair and verification on 30 October 2026

The current pipeline retains independent instance evidence from the segmentation model. More than one animal in either model triggers referral; a selected-mask species conflict also triggers referral. Selecting one cut-out no longer discards evidence of another animal. Both model calls explicitly use square letterboxing (rect=False). The real-image ONNX comparison now applies the same square input policy to both backends, removing the prior rectangular-versus-square comparison defect. No confidence, NMS, acceptance threshold or trained weight was changed.

Mask generation rejects native-resolution and batch-coverage mismatches rather than silently cropping extra pixels. ONNX export helpers preserve existing exports and write a new detector candidate instead of replacing the original binary. Validation checks the entire image/label cohort before inference, records the independent segmentation evidence, and preserves dependency search paths when starting child HTTP or benchmark processes. The active Drive manifest now states the segmentation confidence, NMS and matching settings explicitly.

All 49 software tests passed with PyTorch available. A new trained-weight CPU run covered all 401 previously inspected team photos: 386/399 single-species predictions were correct, 81 photos were referred, 320 were provisionally ready, and 319/320 ready single-species predictions were correct. Seventeen of 52 multi-animal photos remain provisionally ready. Both mixed-species photos are now referred (0/2 ready). The earlier CPU record contained 387/399 correct predictions, 62 referrals, 339 ready photos, 27/52 multi-animal ready photos and 2/2 mixed-species ready photos. This comparison combines source, letterboxing and runtime differences; it is regression evidence and does not isolate a generalization gain or use an untouched holdout.

The portable validation CLI independently completed the same 401-photo cohort in a fresh Python process and reproduced 81 referrals, accuracy 0.9674185464 and 17 multi-animal ready cases. A separate real uvicorn process served 300/300 measured requests after five warm-ups; non-image input returned 415 and empty input returned 400. HTTP p50/p95/p99 were 276.56/375.64/586.21 ms on this CPU. These are actual model-service checks, rather than mocked inference. A Jupyter kernel could not start because this runtime rejects ZeroMQ TCP and IPC sockets; no notebook execution counts or outputs were fabricated.

Fresh-process CPU benchmarks used batch 1, 10 warm-ups and 200 measured runs cycling through the first 50 sorted regression images. PyTorch p50/p95/p99 were 258.10/319.45/351.27 ms, throughput 3.75695 images/s and peak RSS 1061.85 MiB. The diagnostic ONNX-detector pipeline measured 362.87/426.86/498.37 ms, 2.70372 images/s and 1231.45 MiB. The host was an AMD EPYC 9V74, Python 3.12.14; exact CPU runtime packages and thread counts accompany the results. GPU memory is not applicable and no new T4/A8 claim is made.

Fresh classifier ONNX parity passed: maximum absolute difference 2.980232238769531e-7 and full argmax agreement. After identical letterboxing, the final detector boxes on three documented real regression images differed by at most 0.0001220703125 pixels with identical counts and classes. The original raw-head A9b check still fails: 0.00201416015625 exceeds the unchanged 0.001 limit on four seeded inputs. Twenty-eight alternative fused/unfused, graph-optimization and decoding configurations were investigated without establishing a pass. The candidate binaries were not deployed. Final-box agreement is not relabelled as a raw-head pass.

The five original trained/export binaries remain SHA256-identical. Remaining release issues include the 17 missed multi-animal referrals, raw detector A9b, the current-build T4 criterion, complete original evaluator/provenance records, GitHub write access (pending Google device verification) and W&B cloud access and the team's final presentation. Human approval remains mandatory for every image. The original topic approval remains user-confirmed.

Current verification records: https://drive.google.com/drive/folders/1KE1ZgeYBzT4Rv05FJk4ufrkNNx6aeWfk

## Follow-up model-output validation (31 Oct 2026)

The source was hardened against malformed detector/classifier outputs and invalid masks, including the prior NaN-confidence path that could incorrectly return `ready`. The service now maps this failure to HTTP 503 without exposing internal details, and subpixel boxes produce a non-empty crop while requiring staff review. The saved run log records 62/62 regression tests passed with no skips and a successful `compileall`.

A fresh CPU replay of all 401 previously inspected team photos kept weights and thresholds unchanged and matched the preceding run across species, verdict, detector/segmenter counts, referral reasons, and segmentation species. Results: 386/399 single-species predictions correct; 320 provisional-ready; 81 referrals; 17/52 multiple-animal photos still provisional-ready; 0/2 mixed-species photos provisional-ready. This is compatibility evidence on a known cohort, not an independent holdout or a generalization claim. The current raw detector ONNX check remains 0.00201416015625 > 0.001. No new T4 benchmark or W&B cloud update is claimed.
