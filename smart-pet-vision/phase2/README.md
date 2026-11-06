# Smart Pet Vision — Phase 2 (M4 detection, M5 segmentation, M8 integration)

This folder holds the Phase 2 code. It is kept separate from the Phase 1 code in the parent folder. Some modules (`classify.py`, `common.py`, `data.py`, `config.yaml`) exist in both places and differ; the Phase 2 copies here are the ones the Phase 2 results use.

## Version

The files listed in the Phase 2 report's `evidence/verified_source_snapshot.json` (30 files, 30 October 2026) were committed unchanged in the first Phase 2 commit (SHA-256 match 30/30). The current version differs in two of them: `src/spv/pipeline.py` (opt-in v0.4 multi-animal review, `review_mode` off by default) and `notebooks/01_data_preparation.ipynb` (one wording change in a markdown cell). The later multi-animal development work (v0.4) is added in a separate commit so the two versions can be told apart.

## Layout

| Path | Content |
|---|---|
| `src/spv/detect.py` | YOLO11s body detector: training, threshold sweep, evaluation |
| `src/spv/segment.py`, `instance_alignment.py` | YOLO11s-seg masks, SAM 2.1 reference, mask/instance matching |
| `src/spv/pipeline.py` | Detect → classify → segment → ready/refer verdict |
| `src/spv/serve.py`, `demo_page.py` | FastAPI service: `POST /analyze`, `GET /health`; `GET /demo` browser page for the live demonstration (calls `/analyze` only, no external resources) |
| `src/spv/deploy.py` | ONNX export, parity check, latency/throughput benchmark |
| `notebooks/03_detection.ipynb`, `04_segmentation.ipynb`, `05_integration.ipynb` | Reproduction notebooks (Colab, T4) |
| `notebooks/06_release_validation_v2.ipynb` | CPU regression and local HTTP check with preflight (Colab, CPU) |
| `notebooks/09_onnx_channel_diagnosis_v2.ipynb`, `10_log_evidence_documentation_v2.ipynb` | ONNX box/score diagnosis; W&B documentation run that refuses duplicate uploads |
| `tools/preflight_release.py`, `tools/verify_submission_evidence.py` | Fail-fast checks before real-weight runs; report numbers vs saved evidence |
| `notebooks/11_phase2_gap_experiments.ipynb` | COCO-as-is baselines, validation sweeps, per-stage p50/p95, ONNX test mAP (Colab, T4) |
| `submission/` | **Final Phase 2 report on the instructor's template**; `./build.sh` builds the PDF, source ZIP and `RELEASE_MANIFEST.json` (SHA-256 of all submission files) |
| `report/` | Earlier free-form report (superseded for submission by `submission/`); `evidence/` is the source of every reported number |
| `docs/model_card.md`, `docs/DEMO_AND_REHEARSAL_CHECKLIST.md` | Model card; clean-install service run, demo requests, backup recording, four-person rehearsal |
| `docs/QUICKSTART.md`, `docs/TROUBLESHOOTING.md`, `tools/run_service.py` | Run instructions for the checked environments; launcher that checks prerequisites and serves on 127.0.0.1 only |
| `docs/ONNX_EVIDENCE_INDEX.md`, `tools/a9b_root_cause.py` | The four ONNX measurements kept apart (raw head A9b, final detections, task mAP, speed), their evidence files and reproduction; re-run of A9b with a stage-by-stage diagnosis of the gap |
| `docs/PRESENTATION_QA.md`, `docs/REHEARSAL_CHECKLIST_ONE_PAGE.md`, `docs/TEAM_READTHROUGH_SIGNOFF.md` | Technical Q&A; one-page rehearsal checklist; blank read-through sheet for the four members |
| `docs/AUDIT_TRACKER_2026-11-04.md` | Release-candidate audit (issues, evidence, state), change log and verification log |
| `docs/LEAKAGE_SAME_ANIMAL_PROTOCOL.md`, `tools/same_animal_groups.py`; `docs/INDEPENDENT_REPRODUCTION_PROCEDURE.md`, `tools/reproduce_phase2_eval.py`; `src/spv/provenance.py`; `docs/INDEPENDENT_TEST_SET_PROTOCOL.md` | Response to the Phase 1 feedback: same-animal grouping before splitting (frozen rules, not yet run on the images), independent reproduction procedure (not yet executed by a teammate), run provenance (commit, dirty state), shelter-style test-set protocol (not started) |
| `docs/REQUIREMENTS_CHECKLIST_2026-11-06.md`, `docs/NOTEBOOK_EXECUTION_PLAN.md`, `tools/check_notebooks.py` | Every requirement of the guidelines, the rubric and the example report with its status and owner; notebook inventory (needs, executed runs, evidence) and the static check of all 15 notebooks |
| `runs/manifest.json` | Model paths, thresholds, classifier temperature |
| `tests/` | Unit tests |

Weights (`*.pt`, `*.onnx`) and images are not committed (see `.gitignore`); they are kept in the team's Drive.

## Running the tests

```bash
cd smart-pet-vision/phase2
pip install -r ../requirements.txt pytest
PYTHONPATH=src:. python -m pytest -q tests
```

Result on 31 Oct 2026 (CPU, Python 3.13): snapshot version 49 passed; with the v0.4 multi-animal commit, 83 passed; with the submission checks, 102 passed. `python tools/verify_submission_evidence.py` checks that the report's numbers match `report/evidence/`. The report builds with `cd report && pdflatex main.tex` (twice).

## Running the service and the demo page

Step by step, with the weight paths and full SHA-256 values: `docs/QUICKSTART.md`; problems: `docs/TROUBLESHOOTING.md`.

```bash
cd smart-pet-vision/phase2
pip install -r ../requirements.txt
# copy the three private weight files from Drive to the paths in runs/manifest.json, then:
python tools/preflight_release.py --manifest runs/manifest.json --requirements ../requirements.txt --strict-versions
python tools/run_service.py      # checks again, then serves on 127.0.0.1:8000 only; never installs or downloads
# browser: http://127.0.0.1:8000/demo   (API: POST /analyze, GET /health, GET /docs)
```

Send one photo before the audience arrives: the first request loads the three models. Demo steps, expected results and recovery: `docs/DEMO_AND_REHEARSAL_CHECKLIST.md`.

## Acceptance status (criteria set before the Phase 2 runs)

Thresholds A1–A8 and A9a are in the team's code snapshot of 15 Oct 2026, before the first Phase 2 run (17 Oct); A9b (the same bound for the detector) was added no later than the first integration run. A1–A8 and A9a are met. **A9b, detector ONNX parity, is not met** (0.0018–0.0020 > 0.001). Notebook 09 (31 Oct) re-checked six export/runtime settings: 0.00198–0.00201, none within the limit (`report/evidence/M8_onnx_parity_recheck_variants.json`). The bound is not changed. The service therefore uses the PyTorch detector; the ONNX detector is not used. See the Phase 2 report, section "Acceptance criteria and go/no-go decision".

## CPU regression, 31 Oct 2026 (source `051bd2c`, same model code)

- W&B [f8manl0i](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/f8manl0i): 401 team photos, 386/399 single-species correct, 81 referrals, 17/52 multi-animal photos technically ready; publication guards on every photo. A regression on previously inspected photos, not a new test.
- W&B [mu2tolkg](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/mu2tolkg): local HTTP on CPU, one worker, 300/300 HTTP 200, guards 300/300, p50/p95/p99 383.8/430.8/690.1 ms. CPU figures; not comparable with the T4 figures used for A8.
