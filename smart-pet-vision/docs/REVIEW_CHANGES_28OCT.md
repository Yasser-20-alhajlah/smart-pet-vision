# Review changes 28 October 2026

Scope: Phase 1, M1-M3, out of 40. This review used the current Drive snapshot and recovered previous review requests and notes. It does not certify the complete Phase 2 system.

## Changes

- Corrected the PDF's checkpoint claim: equal aggregate accuracy does not establish per-image prediction agreement.
- Permanently redacted the visible face in PDF Figure 5 for display only; model input data and numerical results were not altered.
- Rewrote similarity interpretation with exact counts (1105/1106 and 315/316); no causal leakage bound is claimed.
- Clarified that shared logging-rerun histories agree within 1e-12, while runtime differs. Added history_comparison_verified.csv.
- Clarified that future improvements require a new independent final test. The 401-photo set is already evaluated.
- Added saved team pipeline predictions/metrics before and after label corrections and a standalone verification script. This is supplemental Phase 2 pipeline evidence, not a new Phase 1 whole-image experiment.
- Corrected notebook 02b's field images_different to correct_count_difference, and set per_image_agreement_verified=false. Saved numbers are unchanged; corrected JSON interpretation is not a new model run.
- Removed notebook 02b's user-specific default Drive path. SPV_REPO or unambiguous repository discovery is used.
- Changed notebook 02b's team contact-sheet output to the private team dataset's audit_02b directory, outside the code repository.
- Cleared outputs in corrected 02b source. Historical executed 02b is preserved under notebooks/executed with a prominent explanatory note; the original bytes are also retained in the local review backup.
- Recomputed repository and local package SHA-256 manifests. The original manifest had one mismatch (02b).
- Kept the 401 photos as one team-collected and team-annotated set. Existing provenance was not rewritten as personal capture.

## Verification

52 saved-evidence checks pass: source syntax, counts, duplicate-group/MD5 split crossings, audit samples/completion, confusion-matrix-derived scores, model weight hashes, shared histories, best validation epochs, notebook errors and source/executed code agreement for 01 and 02, pinned requirements and retained team metadata coverage. The 414-row source map includes 13 excluded photos; retained rows are matched by filename to the 401 labels.

Saved team classification metrics were independently recomputed: 387/399 correct, accuracy 0.9699248120, macro-F1 0.9696639465 after four post-review label corrections; before, 383/399, accuracy 0.9598997494. Two mixed images are excluded. Disagreement-only review can bias corrected accuracy upward.

No GPU training, fresh whole-dataset model inference or fresh independent label review was performed in this turn. The full 401 private photos were not re-inspected pixel by pixel; saved audit results were checked against file inventories and tabular evidence.

## Remaining external conditions

- Delivery access policy confirmed by Mohammed: Drive remains private to four team accounts; the team handles public W&B access. Instructor Drive access is not required by the user's chosen delivery workflow. No sharing was expanded.
- W&B access block resolved after explicit user authorization and sign-in. Existing scientific report updated/published; frozen final-test notes updated; historical sources retained.
- Updated user clarification: instructor approval has not yet been granted. Any earlier approval-by-19-October statement is superseded and withdrawn.
- The recorded guideline review identifies Phase 1 weights as 12/20/8; data bonus up to +7 and topic +3 are assessed on the full project and are not guaranteed by this review.
- No Blackboard submission or receipt was produced.
- The current editable Overleaf source was not available in the reviewed snapshot. PDF edits must be synchronized before any future re-export.

## Independent-audit reconciliation follow-up
Compared October 26 review package against current final folder; 12 additional function/curve checks passed. That earlier pass qualified a team-reported approval claim; the later user clarification below supersedes it and states approval pending. Completed Drive evidence synchronization. The older source ZIP remains historical. The draft described at that stage has since been published as documented below. See the final reconciliation PDF.

## Final strengthening pass: actual new evidence
Added fresh checkpoint inference with per-image logits/probabilities, independent metric/ECE recomputation, image-MD5 verification and environment/source provenance. Restored a current Git bundle and rebuilt the report with exact page text/pixel comparison. Inspected the original downloaded instructor template. No new training, W&B publication, approval evidence or teammate claim was invented. See RELEASE_VERSION.json at root and docs/RELEASE_IMPROVEMENTS.md.

## User correction: topic approval
Mohammed explicitly clarified that approval will be requested after sending the project. The report and current source now state approval pending, not approved on or before 19 October. No original-topic bonus is certified.

## Live W&B publication
Published current Phase 1 tables, split/method details, complete-rerun curve provenance, independent inference evidence and version links in the existing scientific report on 28 October. Preserved historical narrative and all five original run sets. Updated frozen notes with best checkpoint, final scores, calibration and fingerprints. No original run summary metrics were overwritten.
