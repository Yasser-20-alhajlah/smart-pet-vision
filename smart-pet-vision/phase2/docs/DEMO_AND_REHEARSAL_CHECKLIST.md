# Demo, service instructions and rehearsal checklist (Phase 2)

Status 4 Nov 2026. Nothing in this file is marked done unless it was actually run; the boxes are for the team to tick.

## 1. Clean-install service run (Colab, CPU or T4)
Weights are private (Drive `MyDrive/capstone /smart-pet-vision/runs/`) and are never committed or published.

The release candidate's code is the Drive candidate folder's `02_Code/reviewed_source.zip` (the same files as `code/` in the report source ZIP). Use it, or the Git branch only if its head is the commit named in that ZIP's `CODE_ARCHIVE_INDEX.json`.

```bash
unzip reviewed_source.zip -d spv && cd spv/smart-pet-vision/phase2   # or: git clone -b phase2-gap-closure https://github.com/Yasser-20-alhajlah/smart-pet-vision.git
pip install -r ../requirements.txt
# runs/manifest.json is included; copy only the three private weight files from Drive to the paths it lists
python tools/preflight_release.py --manifest runs/manifest.json --requirements ../requirements.txt --strict-versions   # files, hashes, versions; no model loading
PYTHONPATH=src python -m unittest discover -s tests               # count for this candidate: see RELEASE_MANIFEST.json checks_at_build
python tools/smoke_test_http.py --image <authorised_pet_image>.png --out smoke_result.json   # optional rehearsal check, CPU, ~2 min
SPV_MANIFEST=runs/manifest.json PYTHONPATH=src uvicorn spv.serve:app --host 127.0.0.1 --port 8000 --workers 1
```

`tools/smoke_test_http.py` starts the service, recomputes the three weight hashes, runs preflight and sends: /health; the image with images=false and images=true (decodes the returned PNGs); a synthetic two-animal composite (the image twice; the verdict actually returned is recorded, a wrong `ready` included); a blank image; text, empty and >15 MB uploads (415/400/413). It checks both policy flags on every response. It is a functional check, not an accuracy or speed measurement.

Weight SHA-256 (must match `submission/RELEASE_MANIFEST.json` and `docs/model_card.md`):
detector `f48e4d8e…1da1cd4`, segmenter `868f3842…a5d8ee`, classifier `22dd58d4…2081ba`.

## 2. The live requests
| # | Request | Expected | What it shows |
|---|---|---|---|
| 1 | `curl -s 127.0.0.1:8000/health` | `{"status":"ok", ...}` with the three model names | service loaded the manifest |
| 2 | `curl -s -F file=@<oxford_test_image>.jpg "127.0.0.1:8000/analyze?images=false"` | HTTP 200; detections, species, verdict (`ready` or `refer_to_staff` with reasons), `staff_approval_required: true`, `automatic_publication_allowed: false` | full pipeline and the publication guard |
| 3 | Same as 2 with one authorised multi-animal image (a licensed public photo, or a team photo whose owner agreed) | Show the verdict actually returned, even if it is a wrong `ready`; this is the known 17/52 limitation | honest failure case |
| 4 | `curl -s -o /dev/null -w "%{http_code}" -F file=@notes.txt "127.0.0.1:8000/analyze"` | `415` (an empty file gives `400`) | input validation |

Executed once on 4 Nov in a CPU cloud container from a clean install of `1cab83f` (not the presentation host): /health 200; the CC0 scikit-image cat `chelsea.png` → ready, cat, both policy flags; blank image → no_animal; 415/400/413 (record: `report/evidence/P2_cpu_smoke_test_20261104.json`). `chelsea.png` can be used for request 2 if no Oxford image is at hand. Choose the image for request 2 in advance from the Oxford test split and record its actual output during rehearsal; do not claim an output that was not observed. Do not show team photos on screen unless their owners agreed.

Timings to quote if asked (never combine them): T4 in-process p95 61.4 ms (A8) and 70.9 ms (3 Nov repeat); CPU HTTP p50/p95/p99 383.8/430.8/690.1 ms (W&B mu2tolkg).

**Recovery during the session:** preflight fails with "weights not found" or a hash mismatch → restore the three files from Drive and rerun preflight (never substitute downloaded models); the service returns 503 or crashes → restart uvicorn once, then switch to the backup recording; no network or Colab runtime → play the backup recording.

## 3. Backup recording
- [x] (4 Nov, container) MP4 of the smoke-test run made from its transcript: `SmartPetVision_backup_recording_CPU_smoke_20261104.mp4` — useful as a last resort, but it is not the presentation-host demo.
- [ ] Record requests 1–4 on the presentation host (or a clean Colab runtime), following Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md`.
- [ ] Check that no API key, token or personal path is visible on screen.
- [ ] Save to Drive `05_Demo` and play it once end to end.

## 4. Four-person rehearsal (candidate deck `04_Presentation/SmartPetVision_Phase2_Presentation.pptx` in the Drive candidate folder, 17 slides + 3 appendix, about 19 min; the guideline range is 15–20 min)
| Slides | Presenter | Time | Points to state correctly |
|---|---|---|---|
| 1–4 | Abdulrahman Mastour Alharthi | 0:00–4:30 | split 5,123/1,106/1,106, test used once; Phase 1 0.906 vs 0.999; AdamW via `optimizer=auto`; operating point chosen on validation |
| 5–6 | Yasser Mansour Alhajlah | 4:30–6:50 | AP (all thresholds) kept apart from P/R/F1 at conf 0.40, NMS 0.50; COCO baseline mAP50 0.939 → 0.984 |
| 7–9 | Mohammed Adeeb Alrashdi | 6:50–10:25 | foreground mIoU vs instance mask AP; SAM with a ground-truth box is an oracle reference |
| 10–16 | Musaad Obaied Saad Alqahtani | 10:25–18:15 | 17/52 limitation; A9b not met, PyTorch serves; CPU vs T4 kept apart; live demo |
| 17 | All | 18:15–19:00 | GO staff-supervised PyTorch service; NO-GO ONNX detector |

The guidelines ask each member to present their own contribution. Slides 3–4 (box derivation, operating point) belong to the detection workstream; the team may move them to Yasser and give Abdulrahman more time on data and Phase 1.

- [ ] Full run-through with a timer; each presenter within their slot.
- [ ] Practise one switch to the backup recording.
- [ ] Every member can answer: why A9b fails and why the 0.001 bound was kept; why the 401 team photos are not a fresh holdout; why the 28.7 px figure is not used.
- [ ] Team approves the candidate deck (PPTX and PDF exported in the Drive candidate folder) before it replaces the deck in `04_Presentation`.
