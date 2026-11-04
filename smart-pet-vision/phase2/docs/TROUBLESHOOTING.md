# Troubleshooting

Start with `python tools/run_service.py --check-only`: it names the first failing prerequisite.

| Symptom | Likely cause | What to do |
|---|---|---|
| `FAIL  packages importable` | Environment not created from `../requirements.txt` | Activate the virtual environment, then `pip install -r ../requirements.txt` yourself (the launcher never installs) |
| `PREFLIGHT FAILED` … missing weight | A weight file is not at the path in `runs/manifest.json` | Place the three files as in `QUICKSTART.md` step 3; the two `best.pt` files go to different folders |
| `PREFLIGHT FAILED` … SHA-256 mismatch | Wrong or partly downloaded file, or detector and segmenter swapped | Download again from the team Drive; compare with the table in `QUICKSTART.md`. Do not edit `EXPECTED_SHA256` |
| `PREFLIGHT FAILED` … version | Installed package differs from the pin | Use a new virtual environment with the pinned file. `--allow-version-drift` starts anyway but the run is then not the recorded environment |
| `FAIL  port 8000 free` | Another service is running | `--port 8010`, or find and stop it: `ss -ltnp \| grep 8000`, then `kill <PID>` |
| `FAIL  loopback address only` | A non-loopback `--host` was given | Intended: the service is not exposed beyond this computer |
| First analysis takes much longer | Models load on the first request | Analyse one photo before presenting |
| Page: "could not be reached" | Service stopped or wrong address | Check the terminal; restart with `python tools/run_service.py`; reload the page |
| Page: HTTP 413 | File over 15 MB, or more than 25 megapixels when decoded | Use a smaller or downscaled photo |
| Page: HTTP 415 | Not a readable image (for example a renamed text file) | Choose a JPEG or PNG |
| Page: HTTP 503 | The model output failed validation; no verdict is given | Expected safety behaviour; the photo needs staff review. If every photo gives 503, restart the service and rerun preflight |
| Cutout or mask empty, `segmentation_unmatched` | The segmenter found no mask matching the selected animal | Expected referral; staff review |
| Smoke test exits early | Port 8765 in use, or weights missing | Stop other services or pass `--port`; run the preflight first |
| Browser tests skipped | Playwright or Chromium not installed | Optional for developers; the other tests still run |
| Slow start without internet | Ultralytics online checks | The launcher sets `YOLO_OFFLINE=1`; set it yourself for the manual command |

**During the presentation:** restart the service once (Ctrl+C, then `python tools/run_service.py`,
then one warm-up photo). If it still fails, play the backup recording and say that it is a recording,
where it was made and from which code (`DEMO_AND_REHEARSAL_CHECKLIST.md`).

**Timings:** times shown by the page or the smoke test describe that machine at that moment. Do not
compare them with the T4 benchmark in the report, and do not mix CPU and GPU numbers.
