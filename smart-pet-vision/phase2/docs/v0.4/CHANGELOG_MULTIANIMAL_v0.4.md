# Phase 2 v0.4: controlled multi-animal review candidate

- Based on Drive source snapshot dated 2026-10-30 (sha256-validated baseline).
- Changed `src/spv/pipeline.py` with default-OFF review feature and prospective audit/enforce modes.
- Added high-resolution detector + independent segmenter proposals; optional 2x2 tiled detector review for small animals.
- Added strict original-coordinate validation, deduplication, hard proposal cap, and fail-closed human review on runtime errors in enforce mode.
- Added per-image `secondary_review` telemetry and `secondary_review_ms`, without changing existing classifier, segmentation mask, or automatic-publication contract.
- Added candidate manifests; left `runs/manifest.json` intact.
- Added source/weight identity and paired baseline-vs-enforce evaluation harness for exactly matching image IDs.
- Not a training result. No independent holdout evaluation or trained-weight regression claimed.
- ONNX/export/Phase1 sources, trained weights, and historical evidence are unchanged.
