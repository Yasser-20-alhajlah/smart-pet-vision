# Dataset card - Smart Pet Vision

Figures come from `runs/data/dataset_card_values.json` (notebook 01), `runs/data/review/` and the team-data files in the `SmartPetVision_TeamData` folder, which sits next to the repository folder (not inside it).
Last updated from the full rerun of 18-19 Oct 2026.

## Sources
| Set | Content | Use | Licence |
|---|---|---|---|
| Oxford-IIIT Pet (Parkhi et al., 2012) | 37 breeds (12 cat, 25 dog), species and breed labels, head box, pixel trimap | train / val / test | CC BY-SA 4.0 |
| Team-collected | 401 photos of cats and dogs (178 cat, 221 dog, 2 mixed), 503 body boxes, polygon masks | external test only | supplied by team members (104 taken by pet owners), annotated by the team, owner consent; not redistributed |

## Cleaning
- Unreadable files, images under 32 px, and images without a usable trimap are dropped: 14 dropped (empty trimap).
- Byte-identical files (MD5) keep one copy: 0 exact duplicates found.
- Near-duplicates (pHash Hamming <= 4): 45 pairs, merged transitively (union-find) into 41 groups covering 84 images (largest group 3).
- After cleaning: 7,335 images, 2,366 cats, 4,969 dogs, 37 breeds. Median image size 500 x 375.

## Split
70/15/15 by image count, stratified by breed, whole duplicate groups assigned to one split.
Counts: train 5,123 / val 1,106 / test 1,106. Distinct split groups (including singletons), recomputed from metadata.csv: train 5,091 / validation 1,103 / test 1,098. Leakage audit: groups crossing splits = 0, MD5 crossing splits = 0 (passed).

| Split | Images | Cats | Dogs | Cat share | Smallest breed |
|---|---|---|---|---|---|
| Train | 5,123 | 1,652 | 3,471 | 32.2% | 128 |
| Val | 1,106 | 357 | 749 | 32.3% | 28 |
| Test | 1,106 | 357 | 749 | 32.3% | 28 |

Residual risk: the dataset has no per-animal identifier, so different photos of the same pet can fall in different splits. Notebook 02b performs a **train/test image-similarity sensitivity analysis** using the nearest same-breed training neighbour (ImageNet ResNet-50 features). Excluding 22-790 images above cutoffs 0.95-0.75 leaves the selected model at 0.997-0.999 accuracy on the retained subsets (`runs/supplement/identity_effect.csv`). Filtering also changes the test population, and image similarity is not an animal identity label. **The causal effect of identity leakage is unmeasured; it cannot be excluded or bounded from this analysis.** The team-collected set (photos supplied by team members or pet owners and reported to have no MD5/pHash match with Oxford-IIIT Pet) is an external test of generalisation to different photo sources. It cannot establish identity-disjointness within Oxford-IIIT Pet or prove zero possible pretraining overlap.
Test accuracy is reported with a 95% Wilson interval (fine-tuned ResNet-50: 0.999, CI 0.995-1.000).

## Derived annotations and audits
- **Body mask** = trimap foreground + border. The border band is ambiguous; segmentation is also scored with it ignored.
- **Body box** = tight box around the body mask. Review of 300 random images: 4 flagged (1.3%, 95% CI 0.5-3.4%), all loose boxes around a small animal; none misses the animal (`runs/data/review/boxes/review.csv`).
- **Label audit** of 200 random images: 0 species errors (0%, 95% upper bound 1.5%); 4 breed labels doubtful (2.0%, 95% CI 0.8-5.0%) (`runs/data/review/labels/review.csv`).
- Both audits were done visually on the contact sheets on 19 Oct 2026 (`reviewer` column) and confirmed by the team (`owner_confirmed` column). Notebook 01 prints the summary from these files (`ok`: 1 correct, 0 flagged). Upper bound: rule of three; intervals: Wilson.

## Known limitations
- Mostly one large, centred animal per image (1,105 of 1,106 test boxes are "large"); web and pet-site photos are cleaner than shelter photos, so small or partly visible animals are under-represented.
- Class imbalance: about two dogs per cat; handled by breed stratification and macro-F1, no class weights.
- The same animal may appear in more than one split (no animal identity labels). Similarity-filtered subset scores do not quantify or bound an identity-leakage effect.
- Possible overlap between ImageNet pretraining images and Oxford-IIIT Pet (both web-collected) cannot be excluded.
- Breed is not a protected attribute, but per-breed error rates are reported because some breeds look alike across species.

## Team-collected set
- **Collection.** 414 photos received from team members; 13 excluded as suspected stock or generated images; 401 kept. Who took the kept photos: 104 were taken by the pets' owners and sent as screenshots (unedited, per team statement); the other 297 were supplied by team members (170 with phone-camera EXIF, 127 without camera metadata). `source_map.csv` does not record which of these 297 a team member photographed personally and which a team member received from an owner, so the set is described as team-collected and team-annotated, not as photographed entirely by the team. Owner consent reported by the team for all 401; signed or recorded permission/provenance evidence is retained privately and not added to the code repository.
- **Processing (same M1 protocol).** EXIF orientation applied, then all metadata including GPS removed; resized to a maximum side of 2048 px; MD5 and pHash checked within the set and against Oxford-IIIT Pet (no exact or near duplicates found); quality audit for blur, size and black bars (`SmartPetVision_TeamData/quality_audit.csv`). No identifiable faces; animals not disturbed.
- **Annotation.** CVAT (CVAT for images 1.1). An automatic draft (detector boxes and segmentation polygons) was reviewed and corrected in CVAT by the four team members, one part each (part 1: 101 images; parts 2-4: 100 images each). Images changed from the draft: part 1 4, part 2 9, part 3 100 (46 with redrawn polygons), part 4 5. Four species labels were corrected by the annotators (team_0144, team_0354, team_0391, team_0392) and four more after a visual review of the classifier-label disagreements (team_0065, team_0076, team_0097, team_0209; `SmartPetVision_TeamData/label_corrections.csv`). Per-part notes are in `SmartPetVision_TeamData/annotation/done/`.
- **Full audit (notebook 02b, `runs/supplement/team_audit.json`).** All 401 photos readable, each with a mask of the same size and a label (401/401/401); 503 boxes, matching `n_animals` for every photo; no EXIF metadata left; no photo above 2048 px; no exact duplicates; median mask share 0.28; species 178 cat, 221 dog, 2 mixed. Every photo is rendered with its boxes in `SmartPetVision_TeamData/audit_02b/team_sheet_00-08.jpg` (kept with the team data, outside the repository).
- **Labels used in evaluation.** `SmartPetVision_TeamData/labels.csv` (image, species, n_animals), `SmartPetVision_TeamData/team_boxes.csv` and `SmartPetVision_TeamData/masks/` (union of the polygons per image). The 2 mixed cat-and-dog images are excluded from classification metrics.
- **Use.** External test only: never used for training, threshold selection or model selection.
- **External test result (classifier).** Fine-tuned ResNet-50 applied to the detector crop in the integrated pipeline (notebook 05, Phase 2 results; `runs/deploy/team_metrics.json` in the Phase 2 submission): accuracy 0.970 (387 of 399 single-species photos; Wilson 95% CI 0.948-0.983), macro-F1 0.970; 0.960 before the four post-review label corrections, which came from reviewing only classifier-label disagreements and can bias accuracy upward.
- **Storage.** The processed set (images, masks, annotations, label and audit files) is kept outside the code repository in `SmartPetVision_TeamData`, shared only within the team and, for grading, with the instructor; the code reads it through `paths.team: ../SmartPetVision_TeamData` in `configs/config.yaml`. Unprocessed originals, which may carry camera metadata, are kept privately and are never shared. No team photo file is stored in the repository or redistributed.


## Final evidence review - 28 October 2026

Checkpoint reload checks establish agreement of aggregate test accuracy, not identical predictions on every image. `correct_count_difference` replaces the misleading historical field `images_different`; underlying values are unchanged. Shared training-loss and validation-accuracy/F1 histories agree within 1e-12; epoch times differ. Training accuracy and validation loss are from the logging rerun. See `docs/REVIEW_CHANGES_28OCT.md`.

The 401 team-collected and team-annotated photos remain one dataset, as confirmed by Mohammed on 28 October. Existing source records are retained; this is not a claim that the team personally captured every photo. The set has already been evaluated. Future model improvements require a new independent final test set. Saved pipeline predictions and a metric verification script are included in `runs/team_evaluation/`.
