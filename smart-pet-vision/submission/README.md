# Smart Pet Vision Phase 2 execution review

Phase 2 comprises M4, M5 and M8, 15 points each, total 45, due 15 November 2026. The presentation is assessed separately for 15 points. The topic approval is confirmed by the user.

## Current source and verification
The canonical layout is src/spv, configs, notebooks, tools and tests. This execution review fixes final-box one-to-one matching, strict mask alignment, body-mask versus box area, explicit segmentation parameters, shared-model HTTP scheduling and portable validation paths. No model was trained or original data/weight replaced.

41 tests were discovered: 40 passed and one PyTorch construction test was skipped. The actual ASGI application was tested with mocked inference stages. Saved CSV arithmetic and IDs were recomputed and five downloaded model/export hashes matched the prior CPU record. These checks do not establish current trained-weight accuracy, numeric parity, real model HTTP behavior or T4 performance. See verification/ and requirements-validation.txt.

## Evidence provenance
Historical model results are in evidence/. The earlier 30 October trained-weight CPU measurements remain in new_evidence/; they predate the source changes in this review. They contain 401 regression cases, 387/399 correct species predictions, 62 referrals, 339 ready images, 336/337 correct ready single-species predictions and 27/52 multi-animal images still ready. All 401 saved outputs require staff approval and prohibit automatic publication. The 18-second fallback is a labelled replay of earlier HTTP responses, not a fresh live recording.

Detector ONNX remains failed at the unchanged project tolerance 0.001. The instructor requires export and a numerical parity check but does not prescribe that project-specific tolerance or a fixed mark deduction. The failed result is preserved.

## Documents and active locations
- Phase2_Report.docx and Phase2_Report.md contain the scientific report and a separate current verification section.
- model_card.md, DEMO.md and ../docs/ describe scope, data, reproduction and the presentation.
- The consolidated Arabic audit, complete requirement matrix, Modified_Files.csv and downloadable source/evidence package are in the [execution review folder](https://drive.google.com/drive/folders/1eiq8c7EKQaUCrnxk4Y1E0-SacJ67cSF3).
- Use the [additional privacy-reviewed segmentation gallery](https://drive.google.com/file/d/1vtKvgj_n9LpKferIZtZNODkWuDeWR6YA/view). The previous JPEG is preserved privately as a historical original; source data and metric files are unchanged.
- [Earlier verified classifier ONNX](https://drive.google.com/file/d/12yfNvAi27ZhK1fORoNghlmrxEgHosJvl/view) is supplied separately, without duplicating large binaries in the source archive.

## Reproduction
Use a copy of the source and preserve historical runs. In the documented validation environment run `PYTHONPATH=src python -m unittest discover -s tests -v`. For model validation install the original model dependencies separately, make the authorized weights/data available, then use notebook 06 or `python tools/validate_release.py --manifest runs/manifest.json --images ../SmartPetVision_TeamData/images --labels ../SmartPetVision_TeamData/labels.csv --output review_runs/NEW_RUN`. Existing output directories are rejected. Start the actual service with `PYTHONPATH=src SPV_MANIFEST=/absolute/path/to/manifest.json python -m uvicorn spv.serve:app --host 127.0.0.1 --port 8877` and rerun real HTTP, parity and hardware benchmarks separately. Restart an already-open notebook kernel before importing updated modules.

## Remaining work
Current-source trained-weight regression, actual HTTP testing, detector parity, independent small/multi-animal development and untouched holdout evaluation, T4 timing, two missing historical notebook04 execution counts, complete evaluator metadata, a dated original target record, private consent/provenance evidence and the final team presentation remain open. The public repository Yasser-20-alhajlah/smart-pet-vision is identified and readable; GitHub write access is currently blocked by a 403 integration response and a pending Google device-verification step, so no PR or merge exists yet. W&B cloud access is unavailable and its documentation is not updated. No grade is guaranteed.

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

## Follow-up source validation (31 Oct 2026)

The current source rejects malformed detector/classifier outputs and invalid segmentation masks, returns HTTP 503 on invalid model output, and safely handles subpixel boxes. The recorded pinned environment passed 62/62 regression tests without skips and passed `compileall`. A fresh CPU replay using unchanged weights and thresholds on the already-inspected 401-photo cohort reproduced all compared output fields: 386/399 single-species predictions correct, 320 provisional-ready, 81 referrals, 17/52 multiple-animal photos still provisional-ready, and 0/2 mixed-species photos provisional-ready. This is not an independent holdout.

Still open: detector raw-head ONNX parity (0.00201416015625 > 0.001), a safe response to missed additional animals, T4 validation of the current build, W&B cloud verification, complete provenance/evaluator records, and the final team presentation. Human review remains required for every image.
