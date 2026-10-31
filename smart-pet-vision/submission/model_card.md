# Model card - Smart Pet Vision system

Figures come from the JSON files under `runs/` named next to each item (models and test metrics from the full rerun of 18-19 Oct 2026 on a Tesla T4; analysis and deployment measurements of 24 Oct 2026, no model retrained). In the Phase 2 submission folder the same files carry an `M4_`, `M5_` or `M8_` prefix.

## Intended use
Shelter and adoption-platform staff upload one photo of a cat or dog. The system returns species,
confidence, body box, mask, cut-out and a verdict: ready for the listing, or refer to staff.

## Out of scope
Veterinary or health assessment; identifying or tracking individual animals or people; breeds or
species outside cats and dogs; video in the service (the detector was only demonstrated on two clips); surveillance; publishing a photo without the staff verdict.

## Components
| Stage | Model | Selected by | File |
|---|---|---|---|
| Detection | YOLO11s, conf 0.4, NMS IoU 0.5, input 640 | checkpoint: highest val box mAP50-95 (Ultralytics 8.4.182 fitness); thresholds: best val F1 with recall >= 0.80 | `runs/detection/operating_point.json` |
| Classification | ResNet-50 fine-tuned, temperature 0.05 | val macro-F1 (1.000); T fitted on val | `runs/classification/comparison.csv` |
| Segmentation | YOLO11s-seg, input 640, conf 0.25 | checkpoint: highest val mask mAP50-95 + box mAP50-95 (Ultralytics 8.4.182 fitness) | `runs/segmentation/weights.json` |

Classifier comparison on the test split (macro-F1): scratch CNN 0.892, frozen ResNet-50 0.999, fine-tuned 0.999.

## Results on the held-out test split (Oxford-IIIT Pet, n = 1,106, input 224 / 640)
| | Value | File |
|---|---|---|
| Accuracy / macro-F1 | 0.999 (95% CI 0.995-1.000) / 0.999 | `classification/finetune_test_metrics.json` |
| ECE raw -> calibrated | 0.0015 -> 0.0009 | same |
| Detection mAP50 / mAP50-95 | 0.984 / 0.850 (cat 0.994 / 0.888, dog 0.973 / 0.812) | `detection/detection_test_metrics.json`, `detection/detection_extra_metrics.json` |
| Detection P / R at operating point | 0.958 / 0.959 (46 FP: 34 localisation, 11 class confusion, 1 background; 45 FN: 34 localisation, 7 missed, 4 class confusion) | same |
| Mask mIoU / Dice (border ignored) | 0.923 / 0.959 (0.930 / 0.962); mask mAP50-95 0.875 | `segmentation/yolo_seg_test_metrics.json` |
| SAM 2.1-B baseline mIoU / Dice (300 images, GT box prompt) | 0.812 / 0.883 | `segmentation/sam_baseline_test_metrics.json` |
| p50 / p95 / p99 latency, batch 1, 10 warm-up, 200 runs, first 50 test images | PyTorch (deployed, detector on the T4) 38 / 61 / 78 ms, 24 img/s, 255 MB GPU, 1,733 MB RSS; ONNX detector on the ONNX Runtime CPU provider 281 / 364 / 702 ms, 3.3 img/s, 1,914 MB RSS | `deploy/benchmark_pytorch.json`, `deploy/benchmark_onnx_detector.json` |
| ONNX parity, max abs diff | classifier 3.0e-7 (passed); detector 1.8e-3 px, rel 2.8e-6 (**failed A9b, limit 1e-3**); on 50 real images raw output 0.0027 px, same detections on 49 of 50 images, matched boxes differ by up to 28.7 px | `deploy/onnx_parity.json`, `deploy/onnx_parity_real_images.json` |
| HTTP service (uvicorn, separate process, PyTorch manifest) | 300 / 300 requests succeeded; HTTP p95 135 ms; non-image upload 415, empty upload 400 | `deploy/service_http.json` |

## External test (team photos, n = 401; 399 single-species)
Classification: accuracy 0.970 (95% CI 0.948-0.983), macro-F1 0.970, referral rate 15.2%, errors that escaped review 1
(accuracy on accepted photos 0.997). Mask mIoU 0.910, Dice 0.936. File: `deploy/team_metrics.json`.
Detection (503 boxes): mAP50 0.771, mAP50-95 0.653; recall 0.732 at the operating point (cat 0.854, dog 0.651). File: `detection/team_detection_metrics.json`.
Before 4 label corrections (`data/team/label_corrections.csv`) the classification figures were accuracy 0.960 and 3 escaped errors
(`deploy/team_metrics_before_label_corrections.json`). The corrections came from reviewing only the
classifier-label disagreements, which can bias accuracy upward; both versions are kept.
The differences from the public split reflect domain, crop-selection and pipeline changes together; they do not isolate a pure classifier-transfer effect.

## Go / no-go
9 of 10 recorded project acceptance criteria are met; independent evidence of pre-results fixation remains unavailable (`runs/acceptance.csv`; A7 and A8 measured on the deployed PyTorch backend). A9b (detector ONNX parity
<= 0.001) is not met: the largest difference is 0.0018 px on box coordinates at 640 px scale. Decision: **no-go for automatic publishing and for ONNX detector deployment**. The PyTorch pipeline is suitable only for a staff-supervised demonstration or internal pilot in which every photo is reviewed before publication. The recorded classifier and detector test metrics do not demonstrate reliable rejection of multi-animal photos: 28 of 52 were accepted as ready. Both mixed-species photos were also accepted. This decision preserves the original failed A9b criterion.

## Known failure modes
- Small animals: on the team photos detection recall is 0.874 for animals filling at least 10% of the frame, 0.169 for 1-10% and 0.0 for under 1% (`detection/team_detection_metrics.json`).
- Multiple animals: the detector finds every animal in only 11 of 52 multi-animal team photos, so 28 of them were accepted as ready although the policy is to refer them; the rule also fired on 5 single-animal photos. In 12 team photos with visible animals no animal was detected (`deploy/team_referral_analysis.json`, `deploy/figures/system_failures.jpg`).
- Over-confident classifier: the fitted temperature hit the lower bound (0.05) because validation accuracy is ~1.0; reported confidence is temperature-scaled but not validated as calibrated on deployment photos,
  so most predictions carry confidence 1.00 (11 of the 12 team-set errors) and the low-confidence referral rule fired once on the team set.
  Referrals came from disagreement (22), multiple animals (22), no animal found (12), animal too small (10) and low confidence (1); 5 photos had more than one reason.
- Team-set misclassifications are dominated by dogs predicted as cats (9 of 12), several of them puppies; the one error accepted without review is team_0392 (adult husky predicted cat).
- Detection on the public split: errors are mostly localisation disagreements with loose trimap-derived boxes; class confusion hits small, curled-up or dressed dogs and long-haired or hairless cats.
- Segmentation worst breeds: beagle 0.889 IoU (fine-tuned); SAM baseline fails on close-ups where the box covers the whole frame (IoU about 0).

## Ethical considerations
Some errors trigger staff referral; recorded false-ready cases demonstrate that referral rules do not catch every error. Every photo requires human review. The verification threshold trades
staff workload (referral rate) against errors that escape review; both are reported. Owner consent is reported by the team for the collected photos; the private records were not independently inspected. Photos were stripped of metadata and annotated in CVAT by the four team members. The two demonstration videos are openly licensed Wikimedia Commons clips (CC0; CC BY 2.0, Roger Braunstein).

## Approval and verification boundary (25 Oct 2026 QA)
This model card describes a research prototype, not approval for unsupervised adoption-listing publication. An employee must review every photo. The model outputs and policy must be re-evaluated on new, independently labelled multi-animal and occlusion examples after any changes; the 401 external photos remain a previously inspected regression benchmark and cannot become a new untouched holdout on replay. Numeric A9b parity must be passed under its original unit and threshold before the ONNX detector is deployed.


## Prior CPU validation on 30 October 2026
The earlier CPU record, retained under new_evidence/, reports trained-weight inference on all 401 previously inspected team images: species accuracy 387/399, 62 referrals, 339 technically ready images, and 336 correct predictions among 337 ready single-species images. Both mixed-species images and 27 of 52 multi-animal images remain technically ready. The corresponding historical pre-CPU values are 61 referrals, 340 ready images and 337/338 correct ready single-species predictions. All 401 prior CPU rows require staff approval and prohibit automatic publication.

That earlier source snapshot also records 25 passing unit tests, 300/300 real HTTP requests, a fresh local-kernel notebook, CPU benchmarks and a classifier ONNX export with maximum difference 3.5763e-7. Those measurements predate the source changes below. Detector ONNX remained failed. The 18-second fallback replays earlier actual HTTP responses. The project topic approval is confirmed by the user; the dated pre-results target record has not been inspected.

## Earlier software review on 30 October 2026

The source now uses one-to-one final-box association in the detector parity diagnostic, rejects misaligned segmentation arrays, records explicit segmentation confidence/NMS/matching settings, and decodes and runs inference off the HTTP event loop while serializing shared-model access. Body-mask pixel share and bounding-box share are separate metadata fields. Historical metadata is retained and must not be relabelled as newly computed mask area. The notebook serves the PyTorch manifest and treats ONNX as diagnostic.

Earlier software verification: 41 unit/ASGI/privacy tests discovered, 40 passed and one classifier-construction test skipped because PyTorch was unavailable. ASGI inference stages were mocked. Saved-record arithmetic and IDs were independently checked; all five downloaded weight/export SHA256 hashes match the prior CPU record. None of these checks is a new model-accuracy, trained-weight HTTP, ONNX parity or T4 benchmark. The original pinned FastAPI version and CPU PyTorch wheel could not be obtained in this environment; exact tested package versions are recorded in verification/environment.json.

Detector A9b still fails the recorded 0.001 criterion. The prior 401-image CPU regression still has 27 of 52 multi-animal images technically ready, and both mixed-species images remain ready. Every output requires staff approval; automatic publication is prohibited. Source changes must be rerun with the trained weights before claiming current-build performance. A visible-person region missed by the prior gallery redaction is covered in the lossless PNG M5_yolo_seg_failures.png; pixels outside that rectangle are identical to the decoded prior gallery. Metric files and original dataset images are unchanged.

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

## Current source validation update (31 Oct 2026)

The 31 October source rejects invalid model outputs, maps the failure to HTTP 503, and requires staff review. The recorded environment passed 62/62 regression tests without skips and `compileall`. A replay on the previously inspected 401-photo set, with unchanged weights and thresholds, reproduced the previous outputs: 386/399 single-species predictions correct, 320 provisional-ready, 81 referred, 17/52 multi-animal photos still provisional-ready, and 0/2 mixed-species photos provisional-ready. This is not independent evaluation.

The detector raw-head ONNX difference remains 0.00201416015625 against the 0.001 target. Missed additional animals remain an unresolved safety limitation. No current-build T4 result is claimed. Every output requires human approval; automatic publication is prohibited.
