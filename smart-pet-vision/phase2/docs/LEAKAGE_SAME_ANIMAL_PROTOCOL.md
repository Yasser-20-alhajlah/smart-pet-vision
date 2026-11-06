# Same-animal grouping protocol (Phase 1 feedback, Part 2) — frozen 6 November 2026

**Status: PARTIAL.** The tool and the rules below are implemented and tested
(`tools/same_animal_groups.py`, `tests/test_same_animal_groups.py`); the grouping has **not been
executed** on the 7,335 Oxford-IIIT Pet images, because this environment has neither the images
nor the ImageNet weights (both downloads are refused by the network policy). Every reported Phase
1 and Phase 2 number therefore still rests on the split of 17 October (`split v1`: MD5 + pHash
groups only), and all delivered checkpoints were trained on `split v1`.

## What the instructor asked
"Only near-identical images are grouped (pHash linked 84 images into 41 groups), so different
photos of the same pet can fall in train and test. Your own check finds 22 test images with a
same-breed training image of similarity ≥ 0.95 and 133 ≥ 0.90; the leak is measured, not removed.
Group likely same-animal photos (e.g. embedding similarity within each breed) before splitting."

## Rules fixed before any run (not to be tuned after seeing results)
| Rule | Value | Reason |
|---|---|---|
| Feature model | torchvision ResNet-50, ImageNet weights `DEFAULT` (IMAGENET1K_V2), `fc` removed, 224 px centre crop, L2-normalised | the features the Phase 1 sensitivity check used (`src/spv/supplement.py`); not trained on this data |
| Candidate scope | pairs **within the same breed** only | different breeds cannot be the same animal; breed labels are given |
| Link threshold | cosine similarity **≥ 0.90** | the lower of the two cutoffs the instructor quoted; links are transitive (connected components) |
| Review band | 0.85 ≤ sim < 0.90 | listed in `pairs_for_review.csv` for visual review; not linked unless a reviewer says so |
| Merged with | MD5 and pHash (Hamming ≤ 4) groups of split v1 | derived and near-identical images stay together |
| Review decisions | `overrides.csv` (image_a, image_b, link/unlink), reviewer named, dated | similarity is evidence, not identity; decisions are recorded, never silently applied |
| Split v2 | Phase 1 `group_stratified_split` on the identity groups, fractions 0.70/0.15/0.15, seed 42 | same splitter and seed as v1; only the groups change |
| Acceptance of v2 | `cross_partition_check.json`: 0 groups crossing partitions | the test the instructor's comment implies |

## Procedure (Colab or Kaggle; GPU optional, about 10 minutes on a T4)
```bash
cd smart-pet-vision/phase2
python tools/same_animal_groups.py \
  --images-csv  ../runs/data/metadata.csv \          # image, breed, species, md5, phash, group, split (split v1 artefact)
  --image-dir   ../data/raw/oxford-iiit-pet/images \
  --weights DEFAULT --threshold 0.90 \
  --split ../runs/data/metadata.csv \                # split v1 column -> cross_partition_check.json
  --new-split --seed 42 \
  --out ../runs/data/identity_v2
```
Then: (1) a team member reviews `pairs_for_review.csv` (all linked pairs and the review band) on a
contact sheet, records decisions in `overrides.csv` with name and date, and reruns with
`--overrides`; (2) commit `identity_groups.csv`, `pairs_for_review.csv`, `overrides.csv`,
`grouping_params.json`, `cross_partition_check.json` and `split_v2.csv` under
`report/evidence/identity_v2/`, naming the commit of the code that produced them; (3) copy the
summary counts into the dataset card and Section 1 of the Phase 2 report.

## What follows from the result
- `cross_partition_check.json` on **split v1** quantifies the leak in the delivered results (number
  of identity groups and images that cross train/val/test). It replaces "the leak is measured,
  not removed" with "N groups / M images cross partitions under the frozen rule".
- Every delivered checkpoint (classifier `finetune.pt`, detector `2seovilv`, segmenter `3ztpz2tc`)
  was trained on split v1. Evaluating them on the v2 test split is **not** leakage-safe, because
  v2 test images may have been in v1 train. A leakage-safe result needs retraining on v2 train
  (about one T4 hour per model with the delivered recipes), evaluated once on v2 test and reported
  separately from the historical v1 numbers. **Retraining needs the owner's approval for compute
  and time and is not started.**
- Residual uncertainty remains: similarity ≥ 0.90 can link different animals of the same breed and
  can miss the same animal in very different poses; the review decisions bound this only for the
  pairs reviewed.

## Evidence of freezing
The rules are in this file and in `FROZEN_THRESHOLD = 0.90` / `REVIEW_BAND = 0.05` of
`tools/same_animal_groups.py`; the Git commit that adds them precedes any run of the tool on the
real images.
