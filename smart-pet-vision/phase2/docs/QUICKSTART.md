# Quickstart — run the Smart Pet Vision service and demo page

Scope: start the PyTorch service on one computer, open the demo page, run one functional smoke test.
This is not an accuracy or speed evaluation. If a step fails, see `TROUBLESHOOTING.md`.

## Environments actually checked

| Environment | What was run | When / source |
|---|---|---|
| Linux x86_64 cloud container, Python 3.13.16, CPU only; new virtual environment with `../requirements.txt` from PyPI (default Linux wheels) | unit tests, preflight, real-weight HTTP smoke test, demo page with real weights | 4–5 Nov 2026; logs in the release package `03_Verification` |
| Google Colab, Python 3.13, Tesla T4 (and Colab CPU runtime) | training, evaluation, benchmarks, `notebooks/06_release_validation_v2.ipynb` | Oct–2 Nov 2026, code `879b0e8`; see the report |

| macOS arm64, Python 3.12.14, CPU; PyTorch 2.15.0, Ultralytics 8.4.210 (versions differ from the pinned environment) | real-weight loopback HTTP functional check, three weight hashes, demo HTML, four image-analysis requests, three invalid uploads and three decoded PNGs; explicit `--allow-version-drift` | 12 Nov 2026; `report/evidence/P2_local_cpu_smoke_20261112/` |

Not checked: Windows, a pinned macOS installation, and the computer used for the final presentation. The 12 November check demonstrates the recorded local configuration only, not full platform support or T4 performance.
Commands below are for a Linux shell.

## A. Local shell (Linux)

**1. Source.** Either clone the repository, or extract `reviewed_source.zip` (release package `02_Code`).
All later commands run from `smart-pet-vision/phase2`.

```bash
python3 -m zipfile -t reviewed_source.zip && unzip -q reviewed_source.zip -d spv && cd spv/smart-pet-vision/phase2
```

**2. Environment** (new virtual environment; several GB, the torch wheels are large):

```bash
python3.13 -m venv ../../.venv && . ../../.venv/bin/activate
pip install -r ../requirements.txt
```

**3. Private weights.** The three weight files are not in Git or in the source archives. Ask the
owner (Yasser) for access to the team Drive folder `MyDrive/capstone /smart-pet-vision/runs/`, download
the three files and place them at the paths in `runs/manifest.json`:

| Model | Path (relative to `phase2/`) | Bytes | SHA-256 |
|---|---|---|---|
| Detector (YOLO11s) | `runs/detect/yolo11s_det/weights/best.pt` | 19,159,066 | `f48e4d8ea5c925c18a4b43744c2f697bfc01a0a473492275f000848125da1cd4` |
| Segmenter (YOLO11s-seg) | `runs/segment/yolo11s_seg/weights/best.pt` | 20,495,332 | `868f3842bf0e05c8b936d3e51e6e81b7f44fdcf12f7b666e5041bf1d47a5d8ee` |
| Classifier (ResNet-50, fine-tuned) | `runs/classification/finetune.pt` | 94,366,917 | `22dd58d471f2dbb39eef10331f208d668a569b1feae378171df8074fbe2081ba` |

Both `best.pt` files have the same name: keep each in its own folder. Never commit them.

**4. Strict preflight** (reads files only; checks paths, SHA-256, operating point and pinned versions):

```bash
PYTHONPATH=src python tools/preflight_release.py --manifest runs/manifest.json --requirements ../requirements.txt --strict-versions
```

**5. Start the service** (checks the prerequisites again, binds to 127.0.0.1 only, never installs or downloads anything):

```bash
python tools/run_service.py            # or: --port 8010 ; --check-only to check without starting
```

Equivalent manual command: `SPV_MANIFEST=runs/manifest.json PYTHONPATH=src YOLO_OFFLINE=1 uvicorn spv.serve:app --host 127.0.0.1 --port 8000 --workers 1`

**6. Open the demo page:** `http://127.0.0.1:8000/demo`. Analyse one photo before an audience
arrives: the first request loads the three models. API: `POST /analyze`, `GET /health`, `GET /docs`.

**7. One functional smoke test** (starts its own service on port 8765, so keep the ports different
from step 5; use an authorised single-animal test photo):

```bash
python tools/smoke_test_http.py --image path/to/authorised_photo.png --out smoke/smoke_result.json
```

Expected last line: every value `true` and `"decoded_png_count": 3`. It checks behaviour and the
two policy flags only, not accuracy or speed. The command stops before starting HTTP if preflight fails and exits nonzero if its response checks fail. It uses CPU. A different installed-library environment requires the explicit option `--allow-version-drift`; this records the differences while retaining file, weight-hash and manifest checks. Such a check does not validate the pinned environment.

**8. Stop.** Press Ctrl+C in the service terminal. If a service is left running, find it with
`ss -ltnp | grep 8000` and stop that process ID with `kill <PID>`.

**Tests (optional).** `PYTHONPATH=src python -m unittest discover -s tests`. The 15 browser tests in
`tests/test_demo_ui_browser.py` need Playwright and Chromium, which are not in `../requirements.txt`;
without them they are reported as skipped.

## B. Google Colab

`notebooks/06_release_validation_v2.ipynb` (CPU runtime): it mounts Drive, clones the repository at
the fixed revision in its first cell (`REV`, set to `051bd2c`; change it to test other code), reads
the manifest and weights directly from `MyDrive/capstone /smart-pet-vision/runs/`, runs the preflight
(version differences are recorded, not fatal), a CPU regression on the private team photos (access to
`SmartPetVision_TeamData` needed) and a local HTTP check inside the Colab machine. The demo page is
not reachable from your own browser in Colab without a tunnel; do not open a public tunnel. For the
live demonstration use a local computer (section A).

On 12 November, the local Git checkout was also checked with Playwright 1.65.1 and Chromium Headless Shell: all 164 unittest cases passed, including the 15 browser cases, with zero skips. These UI tests use a stand-in pipeline and synthetic uploads; they do not re-evaluate model accuracy or replace the presentation-host rehearsal. See `report/evidence/P2_local_cpu_smoke_20261112/git_browser_verification_20261112.json`.
