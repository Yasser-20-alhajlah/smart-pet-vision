# Technical Q&A sheet for the final presentation

Short answers for the four presenters. Every number keeps its split, hardware and setting; the source is in brackets (sections refer to the Phase 2 report `submission/main.tex`). If a question goes beyond this sheet, say what was measured and what was not.

## Data and evaluation

**Q1. How do you know there is no leakage between train and test?**
Exact and near-duplicates (MD5 and pHash groups) were assigned whole to one split before splitting: 0 groups cross splits; split 5,123 / 1,106 / 1,106, seed 42 (Phase 1; report §2.1). All models, thresholds and settings were chosen on validation; the test split was evaluated once. What we cannot exclude: the same individual animal in different photos (identity) and overlap between ImageNet/COCO pretraining images and Oxford test images. A Phase 1 similarity analysis (removing test images with a close training match) left fine-tuned accuracy at 0.997–0.999, but it is a diagnostic, not a proof (Phase 1 report §2.4).

**Q2. Are the 401 team photos a held-out test set?**
No. They are evaluation-only (never used for training or for choosing thresholds), but the team had already inspected them, so they are not a fresh holdout. We use them to show where the system fails (small and secondary animals), not to claim accuracy (report §2.6).

**Q3. Why is mAP measured at confidence 0.001 and not at your operating point?**
mAP integrates over all thresholds, so it is computed at 0.001 as Ultralytics does. Precision, recall and F1 are reported separately at the operating point (conf 0.40, NMS 0.50). Mixing the two would be the "mAP at the deployment threshold" mistake (report §2.4).

## Models

**Q4. What did fine-tuning add over pretrained models?**
Detector, Oxford test, 640 px: COCO YOLO11s used as is mAP50 0.939 / mAP50–95 0.728 → fine-tuned 0.984 / 0.850. Segmenter, test: COCO YOLO11s-seg as is mask mAP50–95 0.686, mIoU 0.830 → fine-tuned 0.875, 0.923. Classifier (Phase 1, test, 224 px): scratch CNN 0.906 → fine-tuned ResNet-50 0.999. Only one fine-tuning recipe per model was trained; a rerun reproduced the metrics (W&B 2seovilv, 3ztpz2tc) (report §2.2, §3.2).

**Q5. Which optimiser did you use? Your notebook says lr0 = 0.01.**
The training call sets no optimiser, so Ultralytics 8.4.182 applied `optimizer=auto`, which for 3,240 iterations selects AdamW with lr0 0.001667 and ignores lr0 = 0.01. Evidence: the executed training notebooks print "optimizer=auto found, ignoring lr0=0.01" and "AdamW(lr=0.001667, momentum=0.9)"; the saved training arguments and the W&B learning-rate curves match. The link between those notebook logs and the two W&B run ids is inferred from identical settings, not printed (report §2.2; `report/evidence/P2_training_optimizer_direct_log_20261104.json`).

**Q6. How did you choose the operating point?**
On the validation split only: sweep of conf 0.05–0.70 × NMS 0.50/0.60/0.70 (24 settings), rule fixed in advance: highest F1 with recall ≥ 0.80. F1 is flat at 0.958 between conf 0.30 and 0.50; chosen conf 0.40, NMS 0.50 (report §2.3).

**Q7. Is the SAM comparison fair?**
It is not a controlled comparison, and we do not present it as one. SAM 2.1-B was prompted with the ground-truth box (an oracle the system never has), not fine-tuned, run at native resolution on 300 sampled test images. On those 300 images YOLO11s-seg has mIoU 0.925 vs SAM 0.812; with trimap border pixels ignored SAM is higher (0.948 vs 0.934). So most of the gap is the border convention of the Oxford masks, not better segmentation (report §3.4).

## Integration and deployment

**Q8. Why did ONNX fail and why didn't you relax the 0.001 bound?**
A9b (raw detector head, max abs difference ≤ 0.001, CPU, 4 random inputs, seed 0) gave 0.0018–0.0020; six export/runtime settings gave 0.00198–0.00201. The difference sits in the pixel-unit box channels (0.0018) while class scores differ by about 1.5e-7; against a float64 PyTorch run, float32 PyTorch itself differs by 0.0018 and ONNX by 0.00096, which points to float32 rounding rather than a wrong export, but that diagnosis uses four random inputs. Task-level values are close, not identical (Oxford test mAP50 0.98252 ONNX vs 0.98357 PyTorch). The criterion was fixed before the results, so changing it now would be moving the goalposts: A9b stays failed, the service runs PyTorch, and the ONNX detector is NO-GO (report §4.4, §4.7).

**Q9. What happens with photos of several animals?**
It is the main known failure. On the 52 multi-animal team photos, all animals were found in only 11, and 17 photos were marked technically ready because a second animal was missed (small-animal recall 0/10, medium 0.169). A synthetic two-cat image is referred, but that does not fix real photos. The mitigation is procedural: every photo, ready or referred, needs staff approval (report §2.6, §4.7; model card).

**Q10. How were the latencies measured? Why do the numbers differ?**
They are different measurements and are never combined. A8: Tesla T4, PyTorch, in-process, batch 1, first 50 Oxford test images cycled, 10 warm-up and 200 timed calls: p50/p95/p99 38.5/61.4/78.4 ms, 24.1 img/s, 255 MB GPU, 1,733 MB RSS. HTTP on T4: 46.3/135.3/178.3 ms. HTTP on CPU only (W&B mu2tolkg, 300 requests, 5 warm-up): 383.8/430.8/690.1 ms. Per-stage p50 on 1,106 validation images (separate run): detect 16.2, classify 10.1, segment 22.1 ms. The times shown on the demo page come from a functional check and are not benchmarks (report §4.1, §4.6).

**Q11. What stops the system from publishing a wrong photo?**
Every response carries `staff_approval_required: true` and `automatic_publication_allowed: false`; they are set in the pipeline, checked by unit tests, by the 300-request HTTP checks and by the smoke test, and displayed on the demo page. "Ready" means "no referral reason was raised", not "publish". An invalid model output returns HTTP 503 and never a ready verdict (report §4.2, §4.3).

**Q12. Why does the API return a species for a photo with no animal?**
The classifier always runs (on the whole frame when no box is found), so the field is filled; it is not an identification. The model card says so, and the demo page hides species and confidence for a `no_animal` referral. The API contract was left unchanged.

**Q13. Can another person reproduce your results?**
Requirements are pinned; `runs/manifest.json` holds every path and threshold; `tools/preflight_release.py` checks the three weight SHA-256 values and library versions before a run; the code and source ZIPs are byte-reproducible and the PDF build is dated by the commit. Unit tests and the evidence checker (report numbers vs saved evidence files) run in CI. Real-weight smoke tests passed from clean installs on CPU. Not re-done: training and the T4 benchmarks on the final candidate code; weights are private (Drive), so reproduction needs team access (report §4.2; `submission/RELEASE_MANIFEST.json`).

**Q14. What would you do with one more week?**
Collect new, independent multi-animal photos as a fresh test set; retrain the detector with the four corrected training boxes and more small/secondary animals, selecting on validation and testing once; define an ONNX parity test on decisions, fixed before the run; repeat the T4 benchmark on the final code.

## Demo questions

**Q15. Is the live demo the same code as the report?**
Same models, thresholds and referral rule. All reported metrics come from code `879b0e8`. Later, `src/spv` gained fail-closed handling of invalid model outputs (HTTP 503 or a referral, never ready), a crop-rounding change of at most one pixel, a wall-time timing field and the demo page `GET /demo`, which only calls `POST /analyze`. None of these changes a threshold; the timings and team-photo results were not re-measured on the newer code (report §4.2).

**Q16. The coffee cup was detected as a cat. Is that a bug?**
It is a real false detection (score 0.495, above the 0.40 operating point) on an object the system is not designed for. The classifier said "dog" with confidence 1.000, which shows that its confidence means little outside cat and dog crops. The referral rule caught it (disagreement, no matching mask, too small), so it went to staff. One example is not a rate (`report/evidence/P2_demo_page_check_20261104.json`).

**Q17. Are your code and data public?**
The repository is public. Oxford-IIIT Pet is public (CC BY-SA 4.0) and is downloaded from the official page, not redistributed. The 401 team photos stay private to the team: they were given with their owners' consent, are used only for evaluation and can show people and homes; only aggregate results are published. The trained weights are in the team Drive with their SHA-256 values in `docs/QUICKSTART.md` (report, "Data and code availability").

**Q18. The page says "score 1.000". Is the species then certain?**
No. It is the classifier's temperature-scaled score, calibrated on Oxford validation photos. On other photo types it can be 1.000 and wrong (the coffee cup gave "dog" at 1.000). That is why a referred photo's species is shown as provisional and every photo still needs staff approval.
