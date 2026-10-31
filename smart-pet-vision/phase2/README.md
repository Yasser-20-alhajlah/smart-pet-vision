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
| `src/spv/serve.py` | FastAPI service: `POST /analyze`, `GET /health` |
| `src/spv/deploy.py` | ONNX export, parity check, latency/throughput benchmark |
| `notebooks/03_detection.ipynb`, `04_segmentation.ipynb`, `05_integration.ipynb` | Reproduction notebooks (Colab, T4) |
| `runs/manifest.json` | Model paths, thresholds, classifier temperature |
| `tests/` | Unit tests |

Weights (`*.pt`, `*.onnx`) and images are not committed (see `.gitignore`); they are kept in the team's Drive.

## Running the tests

```bash
cd smart-pet-vision/phase2
pip install -r ../requirements.txt pytest
PYTHONPATH=src:. python -m pytest -q tests
```

Result on 31 Oct 2026 (CPU, Python 3.13): snapshot version 49 passed; with the v0.4 multi-animal commit, 83 passed.

## Acceptance status (criteria set before the Phase 2 runs)

Thresholds A1–A8 and A9a are in the team's code snapshot of 15 Oct 2026, before the first Phase 2 run (17 Oct); A9b (the same bound for the detector) was added no later than the first integration run. A1–A8 and A9a are met. **A9b, detector ONNX parity, is not met** (0.0018–0.0020 > 0.001). The service therefore uses the PyTorch detector; the ONNX detector is not used. See the Phase 2 report, section "Acceptance criteria and go/no-go decision".
