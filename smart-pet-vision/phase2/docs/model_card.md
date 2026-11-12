# Model card: Smart Pet Vision (Phase 2 release)

## Intended use
Assist shelter and adoption-platform staff who review one candidate listing photo of a cat or a dog. The service returns the detected animal(s), the species with a temperature-scaled confidence, a body mask, and a technical verdict: `ready` or `refer_to_staff` with reasons. **Every photo, ready or referred, requires staff approval; automatic publication is disabled** (`staff_approval_required: true`, `automatic_publication_allowed: false` in every response).

## Out of scope
Automatic publication; identifying individual animals or people; tracking; veterinary or medical use; species other than cats and dogs; unsupervised public-facing decisions.

## Models (release manifest `runs/manifest.json`)
| Component | Model | Input | Weights SHA-256 |
|---|---|---|---|
| Detector | YOLO11s, fine-tuned 40 epochs (W&B 2seovilv) | 640 px letterbox | f48e4d8ea5c925c18a4b43744c2f697bfc01a0a473492275f000848125da1cd4 |
| Segmenter | YOLO11s-seg, fine-tuned 40 epochs (W&B 3ztpz2tc) | 640 px letterbox | 868f3842bf0e05c8b936d3e51e6e81b7f44fdcf12f7b666e5041bf1d47a5d8ee |
| Classifier | ResNet-50, ImageNet weights, fine-tuned (Phase 1, W&B 8nvxbsn2) | 224 px crop | 22dd58d471f2dbb39eef10331f208d668a569b1feae378171df8074fbe2081ba |

Operating point (chosen on the Oxford validation split): detector confidence 0.40, NMS IoU 0.50; classifier temperature 0.05, verification threshold 0.90; minimum matched-mask share 0.05.

## Data
Oxford-IIIT Pet (CC BY-SA 4.0), 7,335 images after cleaning, group-aware split 5,123 / 1,106 / 1,106 (seed 42; split v1: MD5 and pHash groups only, so different photos of the same animal can still cross train and test — the Phase 1 assessment deducted a point for this; the same-animal grouping protocol in `docs/LEAKAGE_SAME_ANIMAL_PROTOCOL.md` was run once on 6 Nov 2026 and found 86 similarity groups (771 images) crossing the v1 partitions under the frozen rule — the owner reviewed them on 7 Nov 2026 and recorded all 86 as clear duplicates, so 135 of the 1,106 test images (12.2%) have a duplicate in training or validation and the split-v1 test numbers may be optimistic (`report/evidence/P2_same_animal_review_owner_20261107.json`); split v2 (4,903 train / 1,382 validation / 1,050 test) was used for one supplementary retraining of the three models on 7 November (report Section 4.10); these weights were not promoted, and the performance table below remains split v1; record `report/evidence/P2_same_animal_grouping_run_20261106.json`). A re-evaluation of the delivered checkpoints from the package used on 6 Nov 2026 (commit 609c0c5) (Colab T4) reproduced all six test metrics exactly (box mAP50 and mAP50–95, mask mAP50 and mAP50–95, mean IoU, Dice; difference 0.0, tolerance 0.005); it was run by a second team member, Mohammed Adeeb Alrashdi, who followed it from start to end and agrees with the result (statement of the project owner, 11 November 2026); the Colab record shows the owner's computer and account (`report/evidence/P2_reproduction_run_20261106.json`, `REPRODUCTION_LOG.txt`). The execution record and owner attribution are documented in REPRODUCTION_LOG.txt; independent operator identity is not established by the account metadata alone. No training run has been independently reproduced (`docs/INDEPENDENT_REPRODUCTION_PROCEDURE.md`, Level 2). Body boxes and masks derived from the trimaps. A private set of 401 team photos (owner consent; metadata removed) is used only for supplementary evaluation and is not committed to the repository.

## Performance (Oxford test split unless stated)
| Measure | Value | Conditions |
|---|---|---|
| Detection mAP50 / mAP50-95 | 0.984 / 0.850 | 640 px, conf 0.001 evaluation |
| Detection P / R / F1 | 0.958 / 0.959 / 0.959 | conf 0.40, NMS 0.50 |
| Mask mAP50-95; mean IoU; mean Dice | 0.875; 0.923; 0.959 | 640 px, conf 0.25 |
| Classification accuracy / macro-F1 | 0.999 / 0.999 | 224 px (Phase 1) |
| Team photos: detection recall | 0.732 (dog 0.651) | 401 photos, 503 animals, conf 0.40 |
| Team photos: single-species species accuracy | 386/399 | CPU run W&B f8manl0i |
| End-to-end p95 latency | 61 ms (T4, in-process); 431 ms (CPU HTTP, W&B mu2tolkg) | batch 1 |
| Per-stage p50 / p95 (validation, notebook 11) | detect 16.2/17.2, classify 10.1/11.7, segment 22.1/23.4, total 48.6/51.3 ms | T4, in-process, 1,106 val images, code 879b0e8; separate run from the A8 benchmark |

### Baselines (not fine-tuned, COCO-pretrained weights used as is; notebook 11)
| Measure | Baseline | Fine-tuned (above) |
|---|---|---|
| Detection mAP50 / mAP50-95, test | 0.939 / 0.728 | 0.984 / 0.850 |
| Detection P / R / F1 at conf 0.40, NMS 0.50, test | 0.937 / 0.919 / 0.928 | 0.958 / 0.959 / 0.959 |
| Mask mAP50-95; mean IoU; mean Dice, test | 0.686; 0.830; 0.893 | 0.875; 0.923; 0.959 |

SAM 2.1-B prompted with the ground-truth box (mIoU 0.812 on 300 test images) is an oracle reference, not an autonomous baseline.

Validation sweep for mask confidence (0.10-0.55, 1,106 validation images) gave mask mIoU 0.92001-0.92240; the highest was at 0.10, the service keeps 0.25 (difference 0.00024 mIoU; referrals at 0.10 were not evaluated). The minimum mask share 0.05 was confirmed on validation (2 of 1,106 images flagged).

## Known failure modes
- Missed secondary or small animals: on team photos 17 of 52 multi-animal photos were marked technically ready because only one animal was detected. The opt-in second detection pass (`review_mode: enforce`) lowers this to 10 of 52 on the same, previously inspected photos but refers 127 instead of 45 of 349 single-animal photos (6 Nov supplementary run; not an independent holdout); it stays off in the served manifest.
- Class balance (supplementary, 6 Nov): a detector trained with cat images oversampled ×2 (same recipe, split v1, seed 42) scored higher on the validation split (mAP50–95 0.854 vs 0.844; dog recall 0.955 vs 0.940 at its own operating point). It was not evaluated on the test split and is not served.
- Lower recall on non-Oxford photos (team set recall 0.732 vs 0.959 on Oxford).
- Dark dogs in clothing detected as cats (detector/classifier disagreement then triggers referral).
- Masks bleed into similar-coloured blankets or clothing; occlusion truncates masks.
- When no animal is detected (`no_animal`), the classifier still runs on the whole frame and the response still carries a `species` and `confidence` (a grey test image returned "cat" with 0.98 in the 4 Nov smoke test). These fields describe the animal only when the verdict is `ready`; for a referral, staff should read the reasons, not the species. The demo page (`GET /demo`) therefore does not display species or confidence for a `no_animal` referral, marks the species of any other referral as provisional, and describes a score (even 1.000) as a model score, not proof; the API response itself is unchanged.
- Objects that are not pets can produce a detection: in the 4 Nov demo-page check a CC0 photo of a coffee cup gave one box (cat, score 0.495, above the 0.40 operating point) and the classifier returned "dog" with confidence 1.000 on that crop; the photo was referred (disagreement, no matching mask, too small). One example, not a false-positive rate (`report/evidence/P2_demo_page_check_20261104.json`); classifier confidence is not meaningful on non-pet crops.
- A two-animal photo is referred only when the detector finds both animals (`multiple_animals`); a synthetic composite of two clearly separated cats was referred, while 17 of 52 real multi-animal team photos were not.
- The ONNX detector export does not meet the raw numerical parity criterion A9b (max abs diff 0.0018-0.0023 vs 0.001, including a 5 Nov re-run). A detector candidate retrained on four corrected training boxes (Kaggle, 5 Nov) is not served: lower validation mAP50-95 (0.839 vs 0.844); see report Section 4.8. A per-channel diagnosis on four random inputs places the difference in the pixel-unit box channels (0.00183; class scores about 1.5e-7). Task-level values on the Oxford test split are close, not identical (mAP50 0.98252 vs 0.98357, mAP50-95 0.85176 vs 0.85000, ONNX vs PyTorch). A9b stays failed and the service uses PyTorch.

## Ethical considerations
Photos may show people or homes; the service returns no identity information and keeps no uploaded image. Team photos are kept in private storage only.


### Supplementary local HTTP verification, 12 November 2026
The delivered PyTorch weights were checked against all three recorded SHA-256 values and used in a CPU-only macOS loopback smoke test. Health and demo HTML returned 200; all four image-analysis requests succeeded with both publication-policy flags intact; three PNG outputs decoded; invalid text/empty/oversized uploads returned 415/400/413. The public CC0 Chelsea cat, its mirrored two-copy composite and a gray blank are behaviour probes, not an accuracy set. The composite and blank were referred.

This used Python 3.12.14 and explicitly recorded `--allow-version-drift` (PyTorch 2.15.0, Ultralytics 8.4.210). It does not establish pinned-version compatibility, T4 performance, independent Multi-Pet accuracy or presentation-host readiness. Default strict preflight correctly refused the differing environment before starting HTTP. The tool now stops on preflight failure and returns nonzero if its response checks fail. Evidence: `report/evidence/P2_local_cpu_smoke_20261112/`.

### Saved split-v2 hybrid HTTP measurement reviewed on 12 November
A retained 8 November receipt matches notebook cell 52: 10 warm-up and 200 measured sequential requests; 200/200 HTTP 200; p50/p95/p99 310.3/434.9/466.3 ms and 2.96 requests/s. The ONNX detector used CPUExecutionProvider inside a T4 session; classifier/segmenter used the available T4. These are torchvision Oxford official-test images, not established as the project's split-v2 test cohort. All responses were held for review. No accuracy, Multi-Pet improvement, ONNX-parity pass or promotion is established. Exact HTTP code revision and memory were not recorded. The resumed notebook retains two errors and later outputs; it is not a clean top-to-bottom execution.

The three v2 checkpoints and three historical Phase 1 classifiers were freshly downloaded and matched to recorded SHA-256 values, with ZIP CRC checks. This does not recover the separate 5 November research checkpoints or establish independent reproduction of the original classification training. See `report/evidence/P2_supplementary_F_hybrid_http_20261108_reviewed_20261112.json` and `P2_checkpoint_availability_recheck_20261112.json`. Served weights and thresholds remain unchanged.
