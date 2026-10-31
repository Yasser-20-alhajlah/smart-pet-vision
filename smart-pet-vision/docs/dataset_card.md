# Dataset card - Smart Pet Vision

Figures come from `runs/data/dataset_card_values.json` (notebook 01), `runs/data/review/` and the team-data files in the `SmartPetVision_TeamData` folder, which sits next to the repository folder (not inside it).
Last updated from the full rerun of 18-19 Oct 2026.

## Sources
| Set | Content | Use | Licence |
|---|---|---|---|
| Oxford-IIIT Pet (Parkhi et al., 2012) | 37 breeds (12 cat, 25 dog), species and breed labels, head box, pixel trimap | train / val / test | CC BY-SA 4.0 |
| Team-captured | 401 phone photos of cats and dogs (178 cat, 221 dog, 2 mixed), 503 body boxes, polygon masks | external test only | collected by the team with owner consent; not redistributed |

## Cleaning
- Unreadable files, images under 32 px, and images without a usable trimap are dropped: 14 dropped (empty trimap).
- Byte-identical files (MD5) keep one copy: 0 exact duplicates found.
- Near-duplicates (pHash Hamming <= 4): 45 pairs, merged transitively (union-find) into 41 groups covering 84 images (largest group 3).
- After cleaning: 7,335 images, 2,366 cats, 4,969 dogs, 37 breeds. Median image size 500 x 375.

## Split
70/15/15 by image count, stratified by breed, whole duplicate groups assigned to one split.
Counts: train 5,123 / val 1,106 / test 1,106. Leakage audit: groups crossing splits = 0, MD5 crossing splits = 0 (passed).

| Split | Images | Cats | Dogs | Cat share | Smallest breed |
|---|---|---|---|---|---|
| Train | 5,123 | 1,652 | 3,471 | 32.2% | 128 |
| Val | 1,106 | 357 | 749 | 32.3% | 28 |
| Test | 1,106 | 357 | 749 | 32.3% | 28 |

Residual risk: the dataset has no per-animal identifier, so different photos of the same pet can fall in different splits. The size of this effect is not measured, and the MD5/pHash checks cannot establish identity independence. The team-captured set (photos supplied by team members or pet owners and reported to have no MD5/pHash match with Oxford-IIIT Pet) is an external test of generalisation to different photo sources. It cannot establish identity-disjointness within Oxford-IIIT Pet or prove zero possible pretraining overlap.
Test accuracy is reported with a 95% Wilson interval (fine-tuned ResNet-50: 0.999, CI 0.995-1.000).

## Derived annotations and audits
- **Body mask** = trimap foreground + border. The border band is ambiguous; segmentation is also scored with it ignored.
- **Body box** = tight box around the body mask. Review of 300 random images: 4 flagged (1.3%, 95% CI 0.5-3.4%), all loose boxes around a small animal; none misses the animal (`runs/data/review/boxes/review.csv`; copy in the submission: `boxes_review_filled.csv`).
- **Label audit** of 200 random images: 0 species errors (0%, 95% upper bound 1.5%); 4 breed labels doubtful (2.0%, 95% CI 0.8-5.0%) (`runs/data/review/labels/review.csv`; copy in the submission: `labels_review_filled.csv`).
- Both audits were done visually on the contact sheets with AI assistance on 19 Oct 2026 (`reviewer` column) and confirmed by the team (`owner_confirmed` column). Notebook 01 prints the summary from these files (`ok`: 1 correct, 0 flagged). The earlier upper bound used the rule of three. The execution review independently computed the exact one-sided 95% binomial upper bound as 1.4867%; other reported intervals are Wilson intervals.

## Known limitations
- Mostly one large, centred animal per image (1,105 of 1,106 test boxes are "large"); web and pet-site photos are cleaner than shelter photos, so small or partly visible animals are under-represented.
- Class imbalance: about two dogs per cat; handled by breed stratification and macro-F1, no class weights.
- The same individual animal may appear in more than one split (no identity labels); the effect size is not measured.
- Possible overlap between ImageNet pretraining images and Oxford-IIIT Pet (both web-collected) cannot be excluded.
- Breed is not a protected attribute, but per-breed error rates are reported because some breeds look alike across species.

## Team-captured set
- **Collection.** 414 photos received from team members; 13 excluded as suspected stock or generated images; 401 kept. Provenance of the kept set: camera EXIF present for 170, no camera metadata for 127, owner-contributed screenshots for 104 (taken by the pet owner, unedited, per team statement). Owner consent reported by the team for all 401; signed or recorded permission/provenance evidence should be retained privately and not added to a public code repository.
- **Processing (same M1 protocol).** EXIF orientation applied, then all metadata including GPS removed; resized to a maximum side of 2048 px; MD5 and pHash checked within the set and against Oxford-IIIT Pet (no exact or near duplicates found); quality audit for blur, size and black bars (`SmartPetVision_TeamData/quality_audit.csv`). The team reports that animals were not disturbed. The gallery review found a person region requiring an additional display redaction; a full raw-image privacy audit is not certified here.
- **Annotation.** CVAT (CVAT for images 1.1). An automatic draft (detector boxes and segmentation polygons) was reviewed and corrected in CVAT by the four team members, one part each (part 1: 101 images; parts 2-4: 100 images each). Images changed from the draft: part 1 4, part 2 9, part 3 100 (46 with redrawn polygons), part 4 5. Four species labels were corrected by the annotators (team_0144, team_0354, team_0391, team_0392) and four more after a visual review of the classifier-label disagreements (team_0065, team_0076, team_0097, team_0209; `SmartPetVision_TeamData/label_corrections.csv`). Per-part notes are in `SmartPetVision_TeamData/annotation/done/`.
- **Labels used in evaluation.** `SmartPetVision_TeamData/labels.csv` (image, species, n_animals), `SmartPetVision_TeamData/team_boxes.csv` and `SmartPetVision_TeamData/masks/` (union of the polygons per image). The 2 mixed cat-and-dog images are excluded from classification metrics.
- **Use.** External test only: never used for training, threshold selection or model selection.
- **Storage.** The processed set (images, masks, annotations, label and audit files) is kept outside the code repository in `capstone/SmartPetVision_TeamData`, shared only within the team; the code reads it through `paths.team: ../SmartPetVision_TeamData` in `configs/config.yaml`. Unprocessed originals, which may carry camera metadata, are kept privately in `capstone/Team_Collected_Dataset` and are never shared. No team photo file is stored in the repository or redistributed; some result figures in `runs/` show team photos as evaluation evidence.


## Area field clarification from execution review

Historical metadata.csv uses the field mask_fraction for bounding-box area divided by image area. It remains unchanged as historical evidence. Newly generated annotations now provide box_fraction for that quantity and mask_fraction for the actual binary body-mask pixel share. A synthetic 3-of-9-pixel mask verified the difference from its 4-of-9-pixel bounding box. No source dataset image, split or label was replaced.

The nine supplied evidence galleries were visually reviewed; a visible-person region in the YOLO segmentation failure sheet required one additional display-only redaction. The PNG output preserves all decoded pixels outside the recorded rectangle. This gallery review does not certify consent or anonymization of every raw image.
