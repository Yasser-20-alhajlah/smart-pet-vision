# Notebook inventory and execution status (6 November 2026)

Every notebook under `phase2/notebooks/` is listed with what it needs to run, when it was executed
and which evidence file that execution left. The static check (`tools/check_notebooks.py`, result
`report/evidence/P2_notebook_static_check_20261106.json`) confirms that all 15 notebooks are valid
nbformat files and that every code cell parses once magics are removed; it does not execute them.

## What can and cannot run outside Colab/Kaggle

Thirteen of the fifteen notebooks mount Google Drive or read private assets (Oxford-IIIT Pet
images, the 401 team photos, the trained weights). Five need CUDA (`01`, `05` for inference; `02`,
`03`, `04`, `11` train). Four log to W&B and need the project key (`06_release_validation_v2`,
`10_*`, `11` Part G). None of these resources exists in a plain CPU container, so a notebook
re-run means a Colab or Kaggle session with the team Drive folder mounted and the key in Secrets.

What was run in the CPU container on 6 November instead (working tree at the delivered commit;
the three real weight files copied to the paths in `runs/manifest.json`):

| Check | Result |
|---|---|
| `python tools/check_notebooks.py --output report/evidence/P2_notebook_static_check_20261106.json` | 15/15 notebooks parse; needs recorded per notebook |
| `PYTHONPATH=src python -m unittest discover -s tests` (Playwright installed) | 147 tests, OK (143 before + 4 for the notebook checker) |
| `python tools/verify_submission_evidence.py` | 43/43 |
| `PYTHONPATH=src python tools/preflight_release.py --manifest runs/manifest.json --requirements ../requirements.txt --strict-versions`; `python tools/run_service.py --check-only` | pass |
| `python tools/smoke_test_http.py --image chelsea.png` (real weights, CPU, pinned venv) | ready, ready, `multiple_animals`, `no_animal`; HTTP 200; policy flags correct on every response |

These checks test behaviour of the delivered code with the delivered weights. They do not
re-measure accuracy or latency; those numbers come from the executed runs below.

## Inventory

| Notebook | Purpose | Needs | Executed (evidence) | Status |
|---|---|---|---|---|
| `00_ONNX_Repair_Verification_COLAB.ipynb` | A9b parity matrix: 4 random inputs + 50 team photos, several export/runtime settings | Drive, weights, team photos | 31 Oct–3 Nov, Colab CPU (`M8_onnx_final_agreement_team50_20261103.json` for the 50-photo cohort; `docs/ONNX_EVIDENCE_INDEX.md` rows 1 and 2b) | Executed; A9b not met, threshold unchanged; final boxes agree on the 50-photo cohort |
| `01_data_preparation.ipynb` | Body boxes from trimaps, YOLO labels, round-trip and split checks | Drive, Oxford images | 17–18 Oct (`M4_label_roundtrip.json`, figure `M4_yolo_label_check.jpg`) | Executed |
| `02_classification.ipynb` | Phase 1 classifier (reused by the pipeline) | Drive, Oxford, GPU, training | Phase 1 (Phase 1 report) | Executed in Phase 1; not re-run in Phase 2 |
| `03_detection.ipynb` | YOLO11s fine-tuning, threshold sweep, test, FP/FN galleries, video | Drive, Oxford, GPU, training | 18–19 Oct (W&B `48vh56vh`, `2seovilv`; `M4_*` evidence; executed copy `03_detection_executed.ipynb` in Drive) | Executed; rerun reproduced the first run |
| `04_segmentation.ipynb` | YOLO11s-seg fine-tuning and evaluation, SAM baseline | Drive, Oxford, GPU, training | 18–19 Oct (W&B `kd5ksgt3`, `3ztpz2tc`, `1jb5y453`; `M5_*`; executed copy in Drive) | Executed; rerun reproduced the first run |
| `05_integration.ipynb` | Pipeline, acceptance table, ONNX export and parity, benchmark, HTTP check | Drive, weights, team photos, T4 | 18–19 Oct (`M8_acceptance.csv`, `M8_benchmark_*.json`, `M8_onnx_parity.json`, `M8_service_http.json`; W&B `fqow5ly7`) | Executed |
| `06_release_validation.ipynb` | CPU regression on the team photos through `tools/validate_release.py` | weights, team photos | 31 Oct, Colab CPU (`P2_cpu_release_validation_f8manl0i.json`) | Executed; `06_release_validation_executed.ipynb` is the executed copy |
| `06_release_validation_executed.ipynb` | Executed copy of the above (outputs kept) | — | 31 Oct | Record only |
| `06_release_validation_v2.ipynb` | Same regression and HTTP check with W&B logging | Drive, weights, team photos, W&B key | 31 Oct (W&B `f8manl0i`, `mu2tolkg`; `P2_cpu_http_validation_mu2tolkg.json`) | Executed |
| `07_Multi_Animal_Validation_COLAB.ipynb` | Paired 401-photo regression of the multi-animal safety decision (v0.4) | Drive, weights, team photos | No executed copy or evidence file of this notebook is in the repository (`docs/v0.4/CHANGELOG_MULTIANIMAL_v0.4.md` states it claims no result) | **Not run as a notebook**; the multi-animal figure in the report (17 of 52 ready) comes from the 31 Oct CPU regression `P2_cpu_release_validation_f8manl0i.json`, and the T4 referral counts from `M8_team_referral_analysis.json` (notebook 05) |
| `09_onnx_channel_diagnosis_v2.ipynb` | Box channels vs score channels, float64 reference (diagnosis only) | Drive, weights | The same diagnosis was executed on 2 Nov inside notebook 11 (`P2_gap_experiments_20261102.json`, channel section); no separate output of this notebook is kept | Diagnosis executed via notebook 11 |
| `09_onnx_detector_parity_recheck.ipynb` | A9b re-check on the original protocol and on Oxford images | Drive, weights, Oxford download | 31 Oct–3 Nov (`M8_onnx_parity_recheck_4inputs.json`, `M8_onnx_parity_recheck_variants.json`, `M8_onnx_parity_recheck_A9b_20261103.json`) | Executed; 5 Nov re-run done by `tools/a9b_root_cause.py` in the CPU container (`M8_onnx_parity_recheck_A9b_20261105.json`) |
| `10_log_benchmark_evidence_wandb.ipynb` | Log the saved M8 evidence files to W&B | W&B key | 31 Oct (W&B `f166xpj4`, `i8giv0x1`) | Executed; superseded by the v2 logger |
| `10_log_evidence_documentation_v2.ipynb` | Documentation run with a digest guard against double logging | W&B key | Not run | **Optional, not run** (needs the key) |
| `11_phase2_gap_experiments.ipynb` | Baselines (COCO models as is), validation sweeps, per-stage p50/p95, ONNX test mAP, channel diagnosis; Part F optional YOLO11n recipe; Part G W&B logging | Drive, Oxford, T4, W&B key | 2 Nov, T4, code `879b0e8` (`P2_gap_experiments_20261102.json`) | Parts A–E executed; **Part F and Part G not run** |

## If a re-run is wanted before 15 November

Two items are open; neither is required for full credit under the rubric:

1. Notebook 11 Part F (a second training recipe, YOLO11n) needs about one T4 hour and the owner's
   compute approval. It would address the "one recipe per model" gap named in the self-assessment.
2. Notebook 11 Part G and `10_log_evidence_documentation_v2.ipynb` log already-saved evidence to
   W&B; they change no result and need the project key in Colab Secrets.

Re-running `03` or `04` is not planned: the reruns of 18–19 October already reproduced the first
runs, and a new run would create a checkpoint that the frozen report, manifest and weight hashes do
not describe.
