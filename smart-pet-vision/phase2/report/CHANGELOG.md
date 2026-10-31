# Phase 2 report source -- editorial revision record

## 31 October 2026 -- submission-focused edition
- Retained the submitted Phase 1 LaTeX style, page geometry, report sections and figure treatment.
- Presented Phase 2 modules in the assessed order: M4 Detection, M5 Segmentation and M8 Integration.
- Consolidated the report around recorded performance, reproducibility evidence and staff-review integration.
- Kept reported data populations, validation thresholds, numeric scores and ONNX comparisons unchanged.
- Refined the submission checklist, self-assessment and conclusion to avoid conflating development work with the evaluated pipeline.
- Rebuilt the PDF independently from the Overleaf ZIP and checked its layout and searchable text.

This is a report-source revision. It does not retrain models, alter model weights, modify the Phase 1 submission or change recorded evaluation data.

## 2026-10-31 — Review corrections (team)
- ONNX section: detector parity reported against pre-registered criterion A9 (0.0018 > 0.001, not met); added 50-image real-input result (98% same detections/classes, max box shift 28.7 px); 0.0020 (4 random inputs) and 3-image 0.0001 px figures kept with their source files now added to evidence/; removed claim that 0.001 was an "additional diagnostic threshold"; fixed "team's" rendering.
- New section "Acceptance criteria and go/no-go decision" from evidence/M8_acceptance.csv (A1–A8, A9a met; A9b not met). Decision: go for staff-supervised PyTorch demo; no-go for ONNX detector.
- Added false-positive gallery figure (rubric asks for FP and FN galleries).
- Added Oxford recall by size (1,105 large, recall 0.959; 1 medium) from M4_detection_extra_metrics.json.
- Video table: added species labels per clip; 4 dog labels in the cat-only clip.
- Benchmark provenance: table values come from evidence/M8_benchmark_*.json; W&B fqow5ly7 logged a separate repetition (values stated).
- Repository: Phase 2 code now in smart-pet-vision/phase2/ on branch phase2-m4-m5-m8 (30/30 snapshot hashes; tests pass).
- Multi-animal 17/52 "ready" explained (secondary animals missed by detector); 386/399 vs 387/399 explained.
- Submission checklist, self-assessment and obstacles table updated to state A9b not met.
- Added REQUIREMENTS_CHECKLIST_AR.md (33 items).
