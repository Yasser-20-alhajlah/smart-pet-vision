# Phase 2 demonstration protocol

Use an 18-minute rehearsal: problem and users 2 min; data 2 min; Phase 1 2 min; detection/segmentation with failures 5 min; live service 3 min; benchmarks and decision 3 min; transitions 1 min. Each member presents their contribution. A real team rehearsal has not been performed by this validation.

## Live service
Start a fresh runtime with the reviewed code and manifest; load the trained weights. Run the included 06_release_validation notebook with SPV_MANIFEST and SPV_TEAM_IMAGES set. Show an actual POST to /analyze, then the returned verdict, reasons, confidence, selected mask and timings. Review team_0005 (ready cat), team_0012 (disagreement referral), and team_0040 (known failure: mixed species and three animals, technically ready). Do not hide the failed counting policy. Every result requires staff approval and automatic publication is false.

## Checked fallback
The preserved validation log records new_evidence/M8_service_fallback.mp4 as 18 seconds at 24 fps with all 432 frames decoded; this review inspected its supplied frame sheet without decoding the full MP4 again. The earlier validation record reports that it replays three actual HTTP responses from its earlier source snapshot. It is labelled as a replay, not a screen recording, fresh live run, training result or T4 benchmark. Copy it to the presenting laptop and verify local playback before presentation. Use original response records in demo_responses.json to explain any case.

## Limits to state
The earlier CPU source record reports 25 unit tests, a fresh local notebook, 401 regression cases and 300 successful real HTTP requests. Current source verification is 40 passed tests and one skipped test, with mocked ASGI inference; trained-weight checks must be rerun. This does not demonstrate reliability on new animals or a new T4 result. Detector ONNX parity fails, 27/52 historical multi-animal images remain technically ready and one single-species classification error escapes. Topic approval is user-confirmed. The dated pre-results target record, private consent evidence, W&B access and team rehearsal still need their corresponding records.


## Current execution review

Use the PyTorch manifest for the supervised demonstration. A9b remains failed and ONNX is diagnostic only. The recorded fallback is replay of prior CPU HTTP responses, not a new live recording or T4 measurement. Rehearse 15-20 minutes with every member and test the current weights/service on a clean runtime; current ASGI regression tests use mocked inference stages.

## Source validation update (31 Oct 2026)

A later source snapshot passed 62/62 regression tests without skips and `compileall`; tests use controlled model outputs for the new output-validation paths. The 401-photo CPU replay reproduced the prior output records with unchanged weights and thresholds, so it is a regression check, not a new holdout. The current detector raw-head ONNX criterion remains failed, the current-build T4 measurement is outstanding, and every result still requires staff review. The team's final live demonstration has not yet been evidenced.
