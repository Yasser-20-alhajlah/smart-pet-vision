# Demo, service instructions and rehearsal checklist (Phase 2)

Status 5 Nov 2026. Nothing in this file is marked done unless it was actually run; the boxes are for the team to tick. Short versions: `QUICKSTART.md` (setup), `TROUBLESHOOTING.md`, `REHEARSAL_CHECKLIST_ONE_PAGE.md`.

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
# open http://127.0.0.1:8000/demo in the browser, then analyse one photo once (warm-up: the first request loads the models)
```

`tools/smoke_test_http.py` starts the service, recomputes the three weight hashes, runs preflight and sends: /health; the image with images=false and images=true (decodes the returned PNGs); a synthetic two-animal composite (the image twice; the verdict actually returned is recorded, a wrong `ready` included); a blank image; text, empty and >15 MB uploads (415/400/413). It checks both policy flags on every response. It is a functional check, not an accuracy or speed measurement.

Weight SHA-256 (must match `submission/RELEASE_MANIFEST.json` and `docs/model_card.md`):
detector `f48e4d8e…1da1cd4`, segmenter `868f3842…a5d8ee`, classifier `22dd58d4…2081ba`.

## 2. The live demonstration (browser page `GET /demo`, about 2 min 30 s)
The page is served by the same FastAPI process and only calls `POST /analyze`; it needs no internet access. It shows the original photo, the detections with the mask, the mask, the cutout, a large verdict (since 5 Nov: "Ready for staff review — not approved for publication" / "Staff review required — see reasons") with the reason messages, and both policy flags of the response. For a `no_animal` referral it shows no species or confidence; for other referrals the species is marked provisional. A Clear button resets the page; a new file or an error removes the previous result. Empty files and files over 15 MB are stopped in the browser; unreadable files show the service's 415 message, over-dimensioned images the 413 message, invalid model output the 503 message; if the service is down, the page says so. Start it with `python tools/run_service.py` (checks first, 127.0.0.1 only); see `QUICKSTART.md` and `TROUBLESHOOTING.md`.

| # | Action on the page | What to say |
|---|---|---|
| 1 | Show the header strip: staff approval required, automatic publication disabled | the system proposes, staff decide |
| 2 | Upload the Oxford test image chosen during rehearsal (or the CC0 photo `chelsea.png`) | verdict, four panels, flags; the result is a proposal for staff |
| 3 | Upload a photo without a pet, or the synthetic blank image (say "synthetic") | `no_animal` referral; the species field is deliberately not shown |
| 4 | Upload a text file renamed `.png` | the 415 message; the service rejects non-images |
| 5 | Optional, only if time remains: the synthetic two-cat composite (say "synthetic; the 17/52 real multi-animal limitation is unchanged") | `multiple_animals` referral |

Observed on 4 Nov with the earlier page wording (READY CANDIDATE / REFER TO STAFF) in a CPU cloud container (real weights, code commit `1721627`, Chromium; **not the presentation host**; evidence `report/evidence/P2_demo_page_check_20261104.json`): CC0 cat photo → READY CANDIDATE, cat, 1 detection; CC0 coffee-cup photo (no pet) → REFER TO STAFF (one false box, cat 0.495; classifier "dog" 1.000; reasons disagreement, segmentation_unmatched, animal_too_small); synthetic two-cat composite → REFER TO STAFF (multiple_animals); synthetic blank image → REFER TO STAFF (no_animal), species hidden; text file → HTTP 415 message; empty and 16 MB files → stopped in the browser. Both policy flags were correct on every response; the browser requested no host other than the service; no file was written in the service working tree. The coffee-cup result is a useful honest failure example (out-of-scope object referred), not a false-positive rate.

Observed on 5 Nov with the revised page (real weights, code `cdcce9e` started with `tools/run_service.py`, CPU cloud container, Chromium; **not the presentation host**; evidence `report/evidence/P2_demo_page_check_20261105.json`): CC0 cat → "Ready for staff review — not approved for publication"; a text file after that result → HTTP 415 message and the earlier result removed; synthetic two-cat composite → "Staff review required — see reasons" (`multiple_animals`, species marked provisional); synthetic blank image → `no_animal`, identification not shown; CC0 coffee cup → three reasons, species "dog" score 1.000 marked provisional; empty and 16 MB files stopped in the browser; Clear reset the page; at 390 px width no horizontal scrolling. Only the service host was requested; nothing was stored in the browser; no file other than Python caches was written in the service folder.

### Command-line fallback (if the browser fails)
| # | Request | Expected | What it shows |
|---|---|---|---|
| 1 | `curl -s 127.0.0.1:8000/health` | `{"status":"ok", ...}` with the three model names | service loaded the manifest |
| 2 | `curl -s -F file=@<oxford_test_image>.jpg "127.0.0.1:8000/analyze?images=false"` | HTTP 200; detections, species, verdict (`ready` or `refer_to_staff` with reasons), `staff_approval_required: true`, `automatic_publication_allowed: false` | full pipeline and the publication guard |
| 3 | Same as 2 with one authorised multi-animal image (a licensed public photo, or a team photo whose owner agreed) | Show the verdict actually returned, even if it is a wrong `ready`; this is the known 17/52 limitation | honest failure case |
| 4 | `curl -s -o /dev/null -w "%{http_code}" -F file=@notes.txt "127.0.0.1:8000/analyze"` | `415` (an empty file gives `400`) | input validation |

If a no-animal image is shown, say that the `species` field is still filled from the whole frame and is not meaningful for a `no_animal` referral (model card, known failure modes).

Executed once on 4 Nov in a CPU cloud container from a clean install of `1cab83f` (not the presentation host): /health 200; the CC0 scikit-image cat `chelsea.png` → ready, cat, both policy flags; blank image → no_animal; 415/400/413 (record: `report/evidence/P2_cpu_smoke_test_20261104.json`). `chelsea.png` can be used for request 2 if no Oxford image is at hand. Choose the image for request 2 in advance from the Oxford test split and record its actual output during rehearsal; do not claim an output that was not observed. Do not show team photos on screen unless their owners agreed.

Timings to quote if asked (never combine them): T4 in-process p95 61.4 ms (A8) and 70.9 ms (3 Nov repeat); CPU HTTP p50/p95/p99 383.8/430.8/690.1 ms (W&B mu2tolkg).

**Recovery during the session:** preflight fails with "weights not found" or a hash mismatch → restore the three files from Drive and rerun preflight (never substitute downloaded models); the service returns 503 or crashes → restart uvicorn once, then switch to the backup recording; no network or Colab runtime → play the backup recording.

## 3. Backup recording
- [x] (4 Nov, container) MP4 of the smoke-test run made from its transcript: `SmartPetVision_backup_recording_CPU_smoke_20261104.mp4` — useful as a last resort, but it is not the presentation-host demo.
- [x] (4 Nov, container) 71 s screen recording of the demo page with real weights on CPU, code commit `1721627`: `SmartPetVision_demo_page_backup_recording_CPU_container_1721627.mp4`. A caption on every frame states the date, commit, CPU container and that it is not the presentation host; inputs are labelled CC0 or SYNTHETIC. It is the closest available fallback until a presentation-host recording exists.
- [x] (5 Nov, container) 56 s screen recording of the revised page with real weights on CPU, code `cdcce9e`: `SmartPetVision_demo_page_backup_recording_CPU_container_cdcce9e.mp4` (CC0 cat, SYNTHETIC blank image, SYNTHETIC two-cat composite, text file, Clear). Caption on every frame: date, container, not the presentation host, commit. Replaces the 4 Nov recording as the first fallback because it shows the current wording; both are cloud-container recordings.
- [ ] Record requests 1–4 on the presentation host (the laptop used on the day), following Drive `05_Demo/DEMO_PLAN_AND_BACKUP_RECORDING_SCRIPT.md`.
- [ ] Check that no API key, token or personal path is visible on screen.
- [ ] Save to Drive `05_Demo` and play it once end to end.

## 4. Four-person rehearsal plan (polish-candidate deck, 17 slides + 3 appendix; target 18:30, guideline 15–20 min)
The guidelines ask each member to present their own contribution and recommend 2–3 min of live demonstration. This plan therefore gives the detection slides 3–4 to Yasser (detection workstream) and gives the demo 2:30. It is a proposal: the team confirms or changes it at the first timed run. The time boxes printed on the slides follow this plan.

| Slides | Presenter | Time | Points to state correctly | Hand-off sentence |
|---|---|---|---|---|
| 1–2 | Abdulrahman Mastour Alharthi | 0:00–2:40 | users and decision; criteria fixed before results (shown on slide 17); split 5,123/1,106/1,106, test used once; Phase 1 0.906 vs 0.999 | "Yasser shows how we find each animal." |
| 3–6 | Yasser Mansour Alhajlah | 2:40–6:45 | body boxes from trimaps; operating point on validation (conf 0.40, NMS 0.50); AP over all thresholds kept apart from P/R/F1 at the operating point; COCO-as-is 0.939 → 0.984 mAP50; team-photo recall 0.732 | "Mohammed explains how we outline the animal." |
| 7–9 | Mohammed Adeeb Alrashdi | 6:45–9:55 | union-mask mIoU vs instance mask AP; trimap border; SAM with a ground-truth box is an oracle reference, not a controlled comparison | "Musaad puts the three models together." |
| 10–15 | Musaad Obaied Saad Alqahtani | 9:55–14:45 | 17/52 limitation; A9b not met, PyTorch serves; CPU vs T4 and HTTP vs in-process never combined | "Now the system running." |
| 16 | Musaad (demo), Yasser at the keyboard if preferred | 14:45–17:15 | demo steps 1–4 of section 2; switch to the recording after one failed retry | "We close with the decision." |
| 17 | All (one sentence each) | 17:15–18:30 | GO staff-supervised PyTorch service; NO-GO ONNX detector; limitations | — |

Timing rule: if any block runs over by more than 20 s, the next presenter shortens their spoken text, not the demo; total must stay under 20:00.

- [ ] Full run-through with a timer; each presenter within their slot.
- [ ] Practise one switch to the backup recording.
- [ ] Every member can answer: why A9b fails and why the 0.001 bound was kept; why the 401 team photos are not a fresh holdout; why the 28.7 px figure is not used. Questions and short answers: `docs/PRESENTATION_QA.md`.
- [ ] Team approves the candidate deck (PPTX and PDF exported in the Drive candidate folder) before it replaces the deck in `04_Presentation`.

## 5. Presentation-host checklist (the day before and 30 min before)
- [ ] Host chosen (a laptop; a Colab runtime cannot show the demo page without a public tunnel, which we do not use) and the same host used for the rehearsal; Python and `../requirements.txt` installed from the release archive.
- [ ] Three weight files copied from Drive; `preflight_release.py --strict-versions` passes (no other weights, ever).
- [ ] Service started from `smart-pet-vision/phase2` with `python tools/run_service.py` (all checks `ok`); `http://127.0.0.1:8000/health` returns the three model names.
- [ ] `http://127.0.0.1:8000/demo` open in a browser tab, zoom about 90 % so the verdict and the four panels fit on the projector; one warm-up photo analysed.
- [ ] Demo images in one local folder, named to show their status (Oxford test, CC0, SYNTHETIC); no team photos unless the owner agreed; no personal paths, keys or notebook tokens on screen.
- [ ] Backup recording opened in a second window and played once; slides in presenter view; timer visible to the presenters.
- [ ] No internet needed during the demo (the page loads nothing external; the launcher sets `YOLO_OFFLINE=1`).

**Fallback procedure (decide in under 30 s):** page shows "could not be reached" or a 503 → restart the service once (Ctrl+C, `python tools/run_service.py`, one warm-up photo); still failing, or any freeze over 30 s → say "we switch to the recording made from the same code" and play the recording; never improvise a result that was not shown. The recording is from a cloud container, not this host: say so.
