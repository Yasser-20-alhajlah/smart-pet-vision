# Independent reproduction procedure (Phase 1 feedback, Part 13) — status BLOCKED

**What the instructor said:** "Yasser's reproduction is said to match, but no reproduced numbers or
run IDs are shown next to the originals, and `REPRODUCTION_LOG.txt` is empty." Required: show the
reproduced numbers and run IDs next to the originals, and commit the log.

**Current state (6 Nov 2026):** `REPRODUCTION_LOG.txt` at the repository root holds no executed
reproduction. The only recorded reproduction is Yasser's own Phase 1 rerun of 23 Oct
(`docs/REPRODUCE.md`); it is not independent because he authored the original runs, and no
numbers were placed beside the originals. **No team member other than the author has executed
a reproduction of the Phase 1 or Phase 2 results.** The procedure below is ready; executing it
is a human action for Abdulrahman, Mohammed or Musaad. Nothing below may be filled in by anyone
else, and an automated rerun is not an independent reproduction.

## Level 1 — evaluation reproduction (about 15 minutes on a T4; required)
Re-evaluates the delivered checkpoints on the Oxford test split exactly as the reported numbers
were produced, and writes original vs reproduced side by side with predefined tolerances
(`tools/reproduce_phase2_eval.py`; tolerances 0.005 absolute, fixed before any run).

```bash
# 1. clean clone at the delivered commit (printed on page 1 of the report; also CODE_ARCHIVE_INDEX.json)
git clone https://github.com/Yasser-20-alhajlah/smart-pet-vision.git && cd smart-pet-vision
git checkout <delivered commit>
# 2. pinned environment
python -m venv .venv && . .venv/bin/activate
pip install -r smart-pet-vision/requirements.txt
# 3. assets: Oxford-IIIT Pet (torchvision download, notebook 01 cell 2) and the three weight files
#    from Drive MyDrive/capstone /smart-pet-vision/runs/ at the paths in phase2/runs/manifest.json
sha256sum runs/detect/yolo11s_det/weights/best.pt      # f48e4d8ea5c925c18a4b43744c2f697bfc01a0a473492275f000848125da1cd4
sha256sum runs/segment/yolo11s_seg/weights/best.pt     # 868f3842bf0e05c8b936d3e51e6e81b7f44fdcf12f7b666e5041bf1d47a5d8ee
sha256sum runs/classification/finetune.pt              # 22dd58d471f2dbb39eef10331f208d668a569b1feae378171df8074fbe2081ba
# 4. the split v1 artefacts: runs/data/metadata.csv (Drive runs/) and data/processed/ (rebuilt by notebook 01 from the same metadata, or copied from Drive)
# 5. run (your own name; nobody else's)
cd smart-pet-vision/phase2
PYTHONPATH=src python tools/reproduce_phase2_eval.py --operator "<Full Name>" \
  --det-weights runs/detect/yolo11s_det/weights/best.pt --seg-weights runs/segment/yolo11s_seg/weights/best.pt \
  --det-yaml ../data/processed/yolo_detect/data.yaml --seg-yaml ../data/processed/yolo_segment/data.yaml \
  --processed ../data/processed --metadata ../runs/data/metadata.csv --image-dir ../data/raw/oxford-iiit-pet/images \
  --out report/evidence/reproduction/reproduction_eval_<name>_<YYYYMMDD>.json
```
Exit status 0 means every metric is within tolerance; the JSON holds operator, date, command,
commit, dirty state, weight hashes, environment, originals, reproduced values and differences.

## Level 2 — training reproduction (about 1 T4 hour per model; recommended)
Run notebooks `03_detection.ipynb` and `04_segmentation.ipynb` on a copy of the repository
without `runs/` (`SPV_REPO` set to the copy, as in `docs/REPRODUCE.md`), with W&B logging on,
then run Level 1 on the new checkpoints. Record the new W&B run ids. Training is not bit-exact
on GPU; the comparison uses the same tolerances and reports the differences as they are.

## What to commit afterwards
1. The JSON from Level 1 (and Level 2) under `report/evidence/reproduction/`.
2. `REPRODUCTION_LOG.txt`: one block per execution — operator, date, commit, environment
   (`pip freeze` path), seed, weight hashes, W&B run ids, the side-by-side table (original /
   reproduced / difference / tolerance / verdict), and the operator's statement that they ran it.
3. The table also goes into Section 1 of the Phase 2 report ("Response to the Phase 1
   feedback") and into the model card.

Until these exist, every document that mentions reproduction must say that the independent
reproduction has **not** been executed.

## Original numbers and run ids to compare against
| Metric | Original | Run / evidence |
|---|---|---|
| Detector test mAP50 / mAP50–95 | 0.9836 / 0.8500 | W&B `2seovilv`; `M4_detection_test_metrics.json` |
| Segmenter mask mAP50 / mAP50–95 | 0.9931 / 0.8745 | W&B `3ztpz2tc`; `M5_yolo_seg_coco.json` |
| Segmenter mIoU / Dice (test, conf 0.25) | 0.9230 / 0.9586 | `M5_yolo_seg_test_metrics.json` |
| Classifier test accuracy (Phase 1) | see Phase 1 report | W&B `8nvxbsn2` |
