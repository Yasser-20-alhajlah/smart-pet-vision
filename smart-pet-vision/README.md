# Smart Pet Vision

AI251P Applied Computer Vision capstone. One photograph of a cat or dog goes in; the system returns
the species with a calibrated confidence, a full-body box, a body mask and cut-out, and a verdict:
**provisional ready, subject to staff approval for every image** or **refer to staff**.

Data: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., CVPR 2012),
CC BY-SA 4.0. Team-captured photos are used for external testing only.

## How to run (Google Colab, T4 GPU)

1. Add this folder to My Drive (a shortcut is enough), at most two folders deep, e.g. `MyDrive/capstone /smart-pet-vision`.
2. Open the notebooks in order. Each one starts with the same setup cell: it mounts Google Drive first, then finds the folder and checks that `configs/config.yaml` and `src/spv` exist; set `SPV_REPO` to the exact path if it finds none or more than one.
3. Running a notebook end to end retrains and overwrites `runs/`. To reproduce without touching the submitted results, copy the folder without `runs/` and set `os.environ["SPV_REPO"]` to the copy (see `docs/REPRODUCE.md`).

| Notebook | Milestone | Phase | Measured T4 time (23 Oct 2026) |
|---|---|---|---|
| `01_data_preparation` | M1 cleaning, de-duplication, split, review sheets | 1 | about 4 min |
| `02_classification` | M2-M3 scratch CNN vs. ResNet-50 frozen vs. fine-tuned | 1 | about 41 min |
| `03_detection` | M4 YOLO11s, threshold sweep, galleries | 2 | about 60 min |
| `04_segmentation` | M5 SAM baseline vs. YOLO11s-seg | 2 | about 83 min |
| `05_integration` | M8 pipeline, ONNX, HTTP service, benchmark, team set, A1-A9 | 2 | about 15 min |

Times were measured during the reproduction of 23 Oct 2026. Every setting lives in `configs/config.yaml`. Results are
written to `runs/` on Drive; images are re-downloaded to local disk each session, and the split is
fixed by `runs/data/metadata.csv` after the first run of notebook 01.

To log runs to W&B, set `wandb.enabled: true` and run `wandb login` once, or store the key as a Colab secret `WANDB_API_KEY` and load it into `os.environ` (never paste the key into a cell or a file).

## Repository

```
configs/config.yaml     all hyperparameters, thresholds and paths
src/spv/data.py         M1: cleaning, MD5 + pHash duplicate groups, group-aware split, masks, YOLO export
src/spv/classify.py     M2-M3: models, training, temperature scaling, ECE, per-breed errors
src/spv/detect.py       M4: YOLO training, threshold sweep on val, test metrics, FP/FN galleries
src/spv/segment.py      M5: SAM baseline, YOLO-seg, IoU/Dice with and without the trimap border
src/spv/pipeline.py     M8: detect -> crop -> classify -> segment -> verdict
src/spv/serve.py        FastAPI service (POST /analyze, GET /health)
src/spv/deploy.py       ONNX export + parity, latency benchmark, team-set preparation and evaluation
notebooks/              one notebook per stage
docs/                   dataset card, model card, REPRODUCE.md
```

## Design decisions worth reviewing

- **Leakage.** Exact duplicates are removed, near-duplicates (pHash Hamming <= 4) are grouped, and a
  group never crosses splits. Notebook 01 stops if the audit fails.
- **Body boxes.** The dataset has head boxes only. Body boxes are the tight box around trimap
  foreground + border, checked by eye on 300 images.
- **Thresholds.** Detector confidence/NMS and the classifier temperature are chosen on validation;
  test results are reported separately. The already inspected 401-photo set is retained for regression and must not be reused to tune thresholds.
- **Refer to staff** when: no animal, more than one animal, calibrated confidence < 0.90, detector and
  classifier disagree, or the mask covers < 5% of the frame.
- **SAM** is prompted with the ground-truth box, so it is a strong reference rather than a
  deployable competitor.

## Review guide for the team

| Reviewer | Check first | Run |
|---|---|---|
| Abdullrahman | `data.py`, notebook 01 review sheets, `classify.py`, notebook 02 | 01, 02 |
| Yasser | `detect.py`, notebook 03, operating-point choice | 03 |
| Mohammed | `segment.py`, notebook 04, border handling | 04 |
| Mosaed | `pipeline.py`, `serve.py`, `deploy.py`, notebook 05 | 05 |

Status (24 Oct 2026): both reviews are complete (`runs/data/review/labels/review.csv`, 200 images; `runs/data/review/boxes/review.csv`, 300 boxes; both confirmed by the team), and the 401 team photos with labels and masks were later moved outside the code repository to `../SmartPetVision_TeamData` on 26 October.


## Earlier software review on 30 October 2026

The source now uses one-to-one final-box association in the detector parity diagnostic, rejects misaligned segmentation arrays, records explicit segmentation confidence/NMS/matching settings, and decodes and runs inference off the HTTP event loop while serializing shared-model access. Body-mask pixel share and bounding-box share are separate metadata fields. Historical metadata is retained and must not be relabelled as newly computed mask area. The notebook serves the PyTorch manifest and treats ONNX as diagnostic.

Earlier software verification: 41 unit/ASGI/privacy tests discovered, 40 passed and one classifier-construction test skipped because PyTorch was unavailable. ASGI inference stages were mocked. Saved-record arithmetic and IDs were independently checked; all five downloaded weight/export SHA256 hashes match the prior CPU record. None of these checks is a new model-accuracy, trained-weight HTTP, ONNX parity or T4 benchmark. The original pinned FastAPI version and CPU PyTorch wheel could not be obtained in this environment; exact tested package versions are recorded in verification/environment.json.

Detector A9b still fails the recorded 0.001 criterion. The prior 401-image CPU regression still has 27 of 52 multi-animal images technically ready, and both mixed-species images remain ready. Every output requires staff approval; automatic publication is prohibited. Source changes must be rerun with the trained weights before claiming current-build performance. A visible-person region missed by the prior gallery redaction is covered in the lossless PNG M5_yolo_seg_failures.png; pixels outside that rectangle are identical to the decoded prior gallery. Metric files and original dataset images are unchanged.

[Current execution review documents and package](https://drive.google.com/drive/folders/1eiq8c7EKQaUCrnxk4Y1E0-SacJ67cSF3)

The release validator also rejects missing, extra, duplicate or invalid label IDs before inference, preventing an inner merge from silently shrinking the evaluation cohort. Two new regression tests cover exact image/label coverage and valid truth schema.


## Latest trained-weight repair and verification on 30 October 2026

The current pipeline retains independent instance evidence from the segmentation model. More than one animal in either model triggers referral; a selected-mask species conflict also triggers referral. Selecting one cut-out no longer discards evidence of another animal. Both model calls explicitly use square letterboxing (rect=False). The real-image ONNX comparison now applies the same square input policy to both backends, removing the prior rectangular-versus-square comparison defect. No confidence, NMS, acceptance threshold or trained weight was changed.

Mask generation rejects native-resolution and batch-coverage mismatches rather than silently cropping extra pixels. ONNX export helpers preserve existing exports and write a new detector candidate instead of replacing the original binary. Validation checks the entire image/label cohort before inference, records the independent segmentation evidence, and preserves dependency search paths when starting child HTTP or benchmark processes. The active Drive manifest now states the segmentation confidence, NMS and matching settings explicitly.

All 49 software tests passed with PyTorch available. A new trained-weight CPU run covered all 401 previously inspected team photos: 386/399 single-species predictions were correct, 81 photos were referred, 320 were provisionally ready, and 319/320 ready single-species predictions were correct. Seventeen of 52 multi-animal photos remain provisionally ready. Both mixed-species photos are now referred (0/2 ready). The earlier CPU record contained 387/399 correct predictions, 62 referrals, 339 ready photos, 27/52 multi-animal ready photos and 2/2 mixed-species ready photos. This comparison combines source, letterboxing and runtime differences; it is regression evidence and does not isolate a generalization gain or use an untouched holdout.

The portable validation CLI independently completed the same 401-photo cohort in a fresh Python process and reproduced 81 referrals, accuracy 0.9674185464 and 17 multi-animal ready cases. A separate real uvicorn process served 300/300 measured requests after five warm-ups; non-image input returned 415 and empty input returned 400. HTTP p50/p95/p99 were 276.56/375.64/586.21 ms on this CPU. These are actual model-service checks, rather than mocked inference. A Jupyter kernel could not start because this runtime rejects ZeroMQ TCP and IPC sockets; no notebook execution counts or outputs were fabricated.

Fresh-process CPU benchmarks used batch 1, 10 warm-ups and 200 measured runs cycling through the first 50 sorted regression images. PyTorch p50/p95/p99 were 258.10/319.45/351.27 ms, throughput 3.75695 images/s and peak RSS 1061.85 MiB. The diagnostic ONNX-detector pipeline measured 362.87/426.86/498.37 ms, 2.70372 images/s and 1231.45 MiB. The host was an AMD EPYC 9V74, Python 3.12.14; exact CPU runtime packages and thread counts accompany the results. GPU memory is not applicable and no new T4/A8 claim is made.

Fresh classifier ONNX parity passed: maximum absolute difference 2.980232238769531e-7 and full argmax agreement. After identical letterboxing, the final detector boxes on three documented real regression images differed by at most 0.0001220703125 pixels with identical counts and classes. The original raw-head A9b check still fails: 0.00201416015625 exceeds the unchanged 0.001 limit on four seeded inputs. Twenty-eight alternative fused/unfused, graph-optimization and decoding configurations were investigated without establishing a pass. The candidate binaries were not deployed. Final-box agreement is not relabelled as a raw-head pass.

The five original trained/export binaries remain SHA256-identical. Remaining release issues include the 17 missed multi-animal referrals, raw detector A9b, the current-build T4 criterion, complete original evaluator/provenance records, GitHub write access (pending Google device verification) and W&B cloud access and the team's final presentation. Human approval remains mandatory for every image. The original topic approval remains user-confirmed.

Current verification records: https://drive.google.com/drive/folders/1KE1ZgeYBzT4Rv05FJk4ufrkNNx6aeWfk

## Source-validation update (31 Oct 2026)

A follow-up validation added strict checks for detector and classifier outputs, safe handling for malformed segmentation masks, HTTP 503 on invalid model output, and non-empty crops for subpixel boxes. The source snapshot passed 62/62 regression tests with no skips in its recorded pinned environment and passed `compileall`. A fresh CPU replay of the already-inspected 401-photo set used unchanged weights and thresholds and matched the preceding record in species, verdict, detection and segmentation counts, reasons, and selected segmentation species: 386/399 single-species predictions correct, 320 provisional-ready, 81 referrals, 17/52 multi-animal photos still provisional-ready, and 0/2 mixed-species photos provisional-ready. This replay is not an independent test set.

The detector raw-head ONNX difference remains 0.00201416015625 against the unchanged 0.001 criterion. The current source has no new T4 benchmark or final team presentation; W&B cloud records also remain unverified. Every output still requires staff approval.
