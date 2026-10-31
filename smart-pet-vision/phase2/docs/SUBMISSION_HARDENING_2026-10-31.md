# Phase 2 submission hardening (31 October 2026)

## Status of each claim

| Claim | Status | Evidence |
|---|---|---|
| CPU regression on the 401-photo team set (source `051bd2c`, weight hashes checked) | Executed and verified; every report row recomputed from the run's `predictions.csv` | W&B `f8manl0i`; `report/evidence/P2_cpu_release_validation_f8manl0i.json` |
| Local CPU HTTP check, 300 requests, one worker | Executed and verified: 300/300, guards 300/300, p50/p95/p99 383.8/430.8/690.1 ms | W&B `mu2tolkg`; `report/evidence/P2_cpu_http_validation_mu2tolkg.json` |
| Detector ONNX parity A9b (<= 0.001) | **Not met** (0.0018; 0.0020; 0.00198-0.00201 in six settings). Threshold unchanged. Service uses PyTorch | `M8_onnx_parity*.json`, `M8_acceptance.csv` |
| Report numbers match evidence files | Checked automatically | `tools/verify_submission_evidence.py` (16 checks), CI |
| Notebooks 06/09/10 v2 | Written and syntax-checked; **not executed** (they need private Drive weights and photos) | `notebooks/*_v2.ipynb` |

## Integrity note on the executed notebook 06 used for `mu2tolkg`

The executed copy of `06_release_validation_READY.ipynb` (private Drive) contains a cell that, when the expected files were missing, wrote a `summary.json` from memory and a 10-row `predictions.csv` with placeholder labels into `03_Evidence/release_validation_20261031T150329Z/`. Effects, checked against the files:
- The HTTP timing and guard counts in `mu2tolkg` come from 300 real requests; they are not affected.
- The three demonstration images of `mu2tolkg` were chosen using the placeholder table; they are illustrative only and excluded from timing.
- The files `summary.json` (790 B) and `predictions.csv` (221 B) in that folder are **not evidence** and must not be cited. The real results are in `release_validation_20261031T144932Z/` (`predictions.csv`, 33,766 B, 401 rows), matching W&B `f8manl0i`.
Notebook 06 v2 removes this path: demonstration images come only from the same run's real predictions, and the cell stops otherwise.

## W&B hygiene
- `f166xpj4` and `i8giv0x1` both log the same saved M8 files (documentation, not inference). Neither should be cited as a new measurement. Notebook 10 v2 refuses to upload an identical file set again and uses `job_type="documentation"`.
- New inference runs from notebook 06 v2 use `job_type` `inference-validation` / `inference-http` and record commit, environment, weight hashes and data protocol; T4 and CPU values are never mixed under one key.

## ONNX: limited diagnosis (proposed, not yet run on the team weights)
Checked already: same weights (SHA-256), same random inputs (seed 0), eval mode, float32 input, raw head compared before post-processing; six export/runtime settings did not change the result.
New hypothesis: the gap sits in the four box-coordinate channels (pixel units), not in the class-score channels. `notebooks/09_onnx_channel_diagnosis_v2.ipynb` (CPU, about 5 minutes, free) tests it. A code check on the public `yolo11n.pt` gave box 0.0025 and score 1.8e-7, which only shows the tool works. Whatever the outcome, A9b stays not met; any per-channel figure is reported separately.

## Multi-animal review: protocol, blocked on data
The 17 multi-animal photos marked technically ready are cases where the detector found one animal; staff approval catches them. Tuning the opt-in review (`review_mode`) on these 52 photos and reporting on the same photos would be tuning on test. A valid experiment needs a separate set of multi-animal shelter photos split into validation (tune `review_conf`, `review_imgsz`) and holdout (report once, before/after). Until such photos exist, the baseline (`review_mode = off`) stays and staff approval remains mandatory.

## Rerun
1. `pip install -r smart-pet-vision/requirements.txt`
2. `cd smart-pet-vision/phase2 && PYTHONPATH=src python -m unittest discover -s tests && python tools/verify_submission_evidence.py`
3. Real-weight checks: run `notebooks/06_release_validation_v2.ipynb` in Colab (CPU), after setting the paths in its first cell.
