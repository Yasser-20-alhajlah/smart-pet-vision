# Demo, service instructions and rehearsal checklist (Phase 2)

Status 4 Nov 2026. Nothing in this file is marked done unless it was actually run; the boxes are for the team to tick.

## 1. Clean-install service run (Colab, CPU or T4)
Weights are private (Drive `MyDrive/capstone /smart-pet-vision/runs/`) and are never committed or published.

```bash
git clone -b phase2-gap-closure https://github.com/Yasser-20-alhajlah/smart-pet-vision.git   # use main after PR #5 is merged
cd smart-pet-vision/smart-pet-vision/phase2
pip install -r ../requirements.txt
# copy the private runs/ folder from Drive so that the paths in runs/manifest.json resolve
python tools/preflight_release.py --manifest runs/manifest.json   # weight files, hashes, versions; no model loading
PYTHONPATH=src python -m unittest discover -s tests               # 111 tests
SPV_MANIFEST=runs/manifest.json PYTHONPATH=src uvicorn spv.serve:app --host 127.0.0.1 --port 8000 --workers 1
```

Weight SHA-256 (must match `submission/RELEASE_MANIFEST.json` and `docs/model_card.md`):
detector `f48e4d8e…1da1cd4`, segmenter `868f3842…a5d8ee`, classifier `22dd58d4…2081ba`.

## 2. The three live requests
| # | Request | Expected | What it shows |
|---|---|---|---|
| 1 | `curl -s 127.0.0.1:8000/health` | `{"status":"ok", ...}` with the three model names | service loaded the manifest |
| 2 | `curl -s -F file=@<oxford_test_image>.jpg "127.0.0.1:8000/analyze?images=false"` | HTTP 200; detections, species, verdict (`ready` or `refer_to_staff` with reasons), `staff_approval_required: true`, `automatic_publication_allowed: false` | full pipeline and the publication guard |
| 3 | `curl -s -o /dev/null -w "%{http_code}" -F file=@notes.txt "127.0.0.1:8000/analyze"` | `415` (an empty file gives `400`) | input validation |

Choose the image for request 2 in advance from the Oxford test split and record its actual output during rehearsal; do not claim an output that was not observed. Do not show team photos on screen unless their owners agreed.

Timings to quote if asked (never combine them): T4 in-process p95 61.4 ms (A8); CPU HTTP p50/p95/p99 383.8/430.8/690.1 ms (W&B mu2tolkg).

## 3. Backup recording
- [ ] Record requests 1–3 from a clean Colab runtime, following Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md`.
- [ ] Check that no API key, token or personal path is visible on screen.
- [ ] Save to Drive `05_Demo` and play it once end to end.

## 4. Four-person rehearsal (presenters as in the deck notes, about 19 min)
| Order | Presenter | Slides | Points to state correctly |
|---|---|---|---|
| 1 | Abdulrahman Mastour Alharthi | title, criteria, data | split 5,123/1,106/1,106; test used once |
| 2 | Yasser Mansour Alhajlah | detection | AP (eval conf 0.001) kept apart from P/R at conf 0.40, NMS 0.50; COCO baseline mAP50 0.939 → 0.984 |
| 3 | Mohammed Adeeb Alrashdi | segmentation | foreground mIoU vs instance mask AP; SAM with a ground-truth box is an oracle reference |
| 4 | Musaad Obaied Saad Alqahtani | system, live demo, benchmark/ONNX, decision, failures, model card, close | A9b not met; PyTorch serves; 17/52 multi-animal limitation; staff approval mandatory |

- [ ] Full run-through with a timer; each presenter within their slot.
- [ ] Practise one switch to the backup recording.
- [ ] Every member can answer: why A9b fails and why the 0.001 bound was kept; why the 401 team photos are not a fresh holdout; why the 28.7 px figure is not used.
- [ ] Export the deck (Slides artifact → PPTX and PDF) to Drive `04_Presentation`.
