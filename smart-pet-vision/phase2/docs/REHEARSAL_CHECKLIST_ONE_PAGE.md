# Rehearsal checklist — one page

**Status: planned, not held.** Nothing below has been done until the team ticks it. Full details:
`DEMO_AND_REHEARSAL_CHECKLIST.md` (sections 4–5); questions: `PRESENTATION_QA.md`.

Rehearsal date: ________  Host computer (the one used on the day): ________  Timer: ________

| Slides | Presenter (own workstream) | Planned | Actual | Must be said correctly |
|---|---|---|---|---|
| 1–2 | Abdulrahman — data and classification | 0:00–2:40 | ____ | criteria fixed before results; split 5,123 / 1,106 / 1,106; test used once |
| 3–6 | Yasser — detection | 2:40–6:45 | ____ | operating point chosen on validation (conf 0.40, NMS 0.50); AP kept apart from P/R/F1; team-photo recall 0.732 |
| 7–9 | Mohammed — segmentation | 6:45–9:55 | ____ | mIoU 0.923 vs mask AP; SAM with a ground-truth box is an oracle reference |
| 10–15 | Musaad — integration | 9:55–14:45 | ____ | 17 of 52 multi-animal photos wrongly ready; A9b failed, PyTorch serves; CPU and T4 numbers never mixed |
| 16 | Musaad — live demo (2:30) | 14:45–17:15 | ____ | ready ≠ approved; no_animal hides the species; say "synthetic" for synthetic images |
| 17 | All, one sentence each | 17:15–18:30 | ____ | GO staff-supervised PyTorch service; NO-GO ONNX detector |

Total must stay under 20:00. If a block overruns by more than 20 s, the next presenter shortens their talk, not the demo.

**Before the run**
- [ ] `python tools/run_service.py` on the host: all checks `ok`; `/demo` open; one warm-up photo analysed.
- [ ] Demo images in one folder, names show their status (Oxford test / CC0 / SYNTHETIC); no team photos unless the owner agreed.
- [ ] Backup recording opened in a second window. The existing recordings were made in a **cloud container**, not on this host; say so if played.

**During the run**
- [ ] Each presenter within their slot (write the actual time above).
- [ ] Demo: one ready photo, one no-pet or blank image, one invalid file; read the verdict wording aloud.
- [ ] Practise the fallback once: restart once, then play the recording.

**After the run**
- [ ] Every member answers Q8 (A9b), Q2 (team photos are not a holdout) and Q6 (operating point) from `PRESENTATION_QA.md` without notes.
- [ ] Record on the presentation host a new backup recording and note: date, host, commit, "presentation host".
- [ ] Tick only what was done; record the actual total time: ________

Completed by (names, date): ______________________________
