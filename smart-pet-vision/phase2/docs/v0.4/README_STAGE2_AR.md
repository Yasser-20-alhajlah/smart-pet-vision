# Smart Pet Vision Phase 2 — Stage 2 ONNX verification (31 October 2026)

## Status: IN PROGRESS / NOT A RELEASE

- Phase 1 is **closed and unchanged**. GitHub `main` and original Drive files were not modified.
- The project source is a local frozen copy of the latest Drive snapshot. All 30 original SHA256 hashes match the approved snapshot of 30 October 2026.
- Detector `best.pt` and `best.onnx` source hashes are stored in `verification/stage2/APPROVED_BASELINE_MANIFEST_2026-10-30.json`. They are **not included** in this archive; use the original Drive files, do not rename/retrain/replace them.
- The original raw-head ONNX parity **failed**: max abs difference 0.00201416015625 > 0.001 on four seeded inputs. The three-image final-box check (max 0.0001220703125 pixels) does **not** establish a raw-head pass.
- New Stage 2 diagnostics were added under `tools/` and tested using synthetic tensors and source-hash checks. No trained-weight inference is claimed here.
- The previously inspected 401-photo cohort remains a regression set, **not an independent new test**. Known false provisional-ready multi-animal cases: 17/52; manual approval still required for every image.

## Which files changed?

All 30 files covered by the original source snapshot were **left byte-identical**. Only new files were added:

- `tools/onnx_parity_diagnostics.py`: per-channel absolute-error analysis, source SHA and image manifest verification.
- `tools/run_onnx_stage2.py`: read-only, fail-closed trained-weight parity runner; writes JSON even when dependencies or inputs are missing.
- `tools/freeze_onnx_cohort.py`: one-time manifest creation for at least 50 photos. Review the selected images and cohort makeup before freezing/using for the official gate.
- `tests/test_onnx_parity_diagnostics.py`: 8 synthetic/unit tests.
- `verification/stage2/*`: reference manifest, local test log and run-status evidence.

## Actual Stage 2 command (CPU parity, in a model-capable environment)

Install the original model dependencies on a dedicated environment (see `verification/stage2/model_environment_reference.json`). Do not reinstall from the executed training notebook on top of the submitted runs.

1. Select at least 50 representative images from the allowed test/regression pool, retaining their original image IDs and source; verify the selection. Example cohort freeze (creates a new file, refuses overwriting an existing file):

```bash
python tools/freeze_onnx_cohort.py \
  --image-root /path/to/team_images \
  --output /path/to/new_cohort_50.json --count 50
```

The helper chooses alphabetically from the supplied folder; to avoid bias, supply a **pre-curated folder** with a variety of dogs, cats, sizes and poses. Verify the list manually. It is a technical parity cohort, not an independent model-quality benchmark.

2. Run the real parity gate; copy the original Drive assets without modifications:

```bash
python tools/run_onnx_stage2.py \
  --repo /path/to/SmartPetVision_Phase2_Stage2_ONNX_2026-10-31 \
  --weights /path/to/best.pt \
  --onnx /path/to/best.onnx \
  --image-root /path/to/team_images \
  --cohort /path/to/new_cohort_50.json \
  --baseline verification/stage2/APPROVED_BASELINE_MANIFEST_2026-10-30.json \
  --output /path/to/new_unique_parity_run
```

The gate explicitly matches the approved source SHA256 and both actual model SHA256, uses four seeded raw inputs plus 50+ same-input real images, then compares final NMS outputs separately. It returns `PASS`, `FAIL` or `BLOCKED`, creates a **new** output directory and will not overwrite evidence. It does not authorize automatic publication even if ONNX candidate parity passes. This is not a GPU/T4 benchmark.

3. For a numeric failure, inspect `random.per_run[].channels` and `largest_errors`, as well as `real.per_image`; isolate whether the maximum error is among box coordinates (channels 0–3), classification logits/scores, or ONNX export numerical precision. Keep the original `0.001` unchanged.

## Conditions still blocking final approval

- Real-weight ONNX raw-head parity must pass the unmodified 0.001 project criterion, or retain the recorded failure and keep PyTorch as the deployed backend.
- The final working source must be rerun because its `segment.py` and `deploy.py` hashes differ from the saved 401-image run; no previous result is silently relabeled a new measurement.
- New independent validation for multiple animals; policy presently misses some multiple-animal images.
- T4 benchmark and full M4/M5 evaluation metadata; complete consent/provenance and W&B run/config/artifact trace.
- Source is a read-only mirror. Do not merge to GitHub main or replace Phase 1 until the team approves a scoped Phase 2 branch/PR.

## Sources

- Drive [source folder](https://drive.google.com/drive/folders/1DxON6E4Agi0kfsXJvtt3poCDgRciekuS)
- Drive [Model Repair Verification](https://drive.google.com/drive/folders/1KE1ZgeYBzT4Rv05FJk4ufrkNNx6aeWfk)
- [Approved source audit](../SPV_Phase2_Source_Audit_2026-10-30/AUDIT_DECISION_AR.md) (relative reference in the working environment)
