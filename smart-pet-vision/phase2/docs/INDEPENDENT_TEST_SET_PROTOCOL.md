# Independent shelter-style test and calibration set — collection protocol (status OPEN)

The Phase 1 feedback recommends a harder, independent shelter-style test and calibration set
(cluttered backgrounds, poor light, partial views) because the Oxford test accuracy is saturated.
No such set exists: the 401 team photos were collected for Phase 1, have been evaluated by every
delivered model (Sections 2.6, 3.4 and 4.7 of the Phase 2 report) and cannot be called untouched.
This protocol is fixed before any collection so that the new set is independent by construction.

## Scope
- Target: 300–500 photos of cats and dogs in shelter-like conditions; at least 40 % with clutter,
  25 % in poor light, 25 % with a partly visible animal, 15 % with two or more animals; roughly
  1 cat : 1 dog (the training data is 1 : 2).
- Sources: photos taken by team members or supplied by owners/shelters with written consent;
  no web images, no stock or generated images, no photo already in the team set.
- Two disjoint parts, assigned at collection time by a fixed rule (photo index mod 2):
  **calibration** (temperature, thresholds) and **final test** (used once, after all choices).

## Screening (before any model sees the photos)
1. MD5 and pHash (Hamming ≤ 4) against Oxford-IIIT Pet and the 401 team photos: matches excluded.
2. ResNet-50 feature similarity ≥ 0.90 (the frozen rule of `LEAKAGE_SAME_ANIMAL_PROTOCOL.md`)
   against Oxford train/val and the team set: matches listed for review and excluded unless a
   reviewer records that the animal is different.
3. Metadata stripped; a `source_map.csv` records who took each photo and the consent record id.

## Annotation
Body boxes and polygon masks in CVAT by one annotator, 20 % re-annotated by a second annotator
(box IoU and mask IoU reported); species label from the owner or shelter, checked by the
annotator; a `README` in the set folder states the counts and the audit result.

## Use
- Models and thresholds are never tuned on the final test part.
- Report accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, detector recall by
  size and by visibility, segmentation IoU/Dice, with sample counts and Wilson intervals.
- The set is private (consent); only aggregate results are published.

## Status
Not started. Owner: the team (collection and consent); estimated effort two to three weeks.
This item does not block the Phase 2 submission; it blocks any claim of generalisation beyond the
already-evaluated team photos.
