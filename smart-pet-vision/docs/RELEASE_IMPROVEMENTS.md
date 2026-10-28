# Final improvement pass - 28 October 2026

## Report rebuild
The corrected PDF is not currently reproducible from a clean clone: the documented `report_source/build_report.py` and its required base PDF are absent. A local `tools/build_report.py` is untracked, and its required `tools/assets/base_report.pdf` is also absent. The report should be treated as a separately supplied deliverable until the authorized inputs are included in the repository or a versioned release. The instructor template was directly inspected from the user's downloaded Capstone_Phase1_Template_Overleaf.zip and confirms M1 12, classification 20 and reproducibility 8, including repository/commit and teammate reproduction fields.

## Version provenance
A new review-time Git commit/bundle identifies current tracked source. It is not a historical training commit. Generated run outputs, weights and personal team photographs remain excluded from Git; release manifests link the delivered evidence. Teacher acceptance of a downloadable Git bundle versus a hosted repository remains outside this review.

## Fresh inference
A separate verifier `tools/evaluate_checkpoints.py` restores all three existing checkpoints, checks each validation/test image MD5, exports per-image logits/probabilities and recomputes scores. It uses saved temperatures for comparison, refits temperatures only on validation as a diagnostic, and never trains or selects a model on test performance. Results belong in a new verification directory; the original results are preserved.

For this pass the timm Hugging Face mirror was used because the Oxford HTTP archive was slow. Before inference all 2,212 retained validation/test images matched the original metadata MD5 exactly. Sources: https://huggingface.co/datasets/timm/oxford-iiit-pet and https://www.robots.ox.ac.uk/~vgg/data/pets/ . Images are not included in the release.

## External evidence
Mohammed clarified on 28 October that instructor approval has not yet been granted and will be sought after submission. The earlier statement that approval was confirmed by 19 October is withdrawn. Do not claim prior approval or a confirmed original-topic bonus. After Mohammed explicitly authorized forge.coreweave.com and signed in, the existing W&B scientific report was updated and published on 28 October. Current Phase 1 tables and provenance were added; historical narrative/panels retained. Frozen final-test notes were updated. The published section is preserved in docs/WANDB_CURRENT_RESULTS_UPDATE.md. No original run metric was overwritten and no new training run was invented.

## Reproduction boundary
Fresh checkpoint inference is different from a new clean training reproduction by another teammate. The original teammate reproduction record remains team-reported unless actual original run outputs are supplied. No Blackboard upload is performed in this pass.

## Actual fresh-inference result
Completed on local macOS Apple MPS, Python 3.12.14, torch 2.11.0, float32; 28 October 2026 18:27 UTC (see environment.json for exact times). All 2,212 validation/test image bytes matched submitted MD5. All three test accuracy values and confusion matrices match saved results exactly. Saved-temperature ECE differences: scratch -6.8982e-9, frozen +1.2395e-9, fine-tuned 0.0. Validation refits vary slightly by environment (scratch 0.822748 versus saved 0.823074); original temperatures/results were preserved. Fine-tuned remains clipped at 0.05. Independent standard-library recomputation of accuracy, macro-F1, confusion matrices and ECE passed from the newly exported CSVs for all three models.

Fresh frozen and fine-tuned predictions differ on two images despite equal accuracy; this confirms why aggregate accuracy must not be used as evidence of identical predictions. No original per-image Oxford predictions existed, so per-image agreement with the historical GPU run is still unverifiable. Exports are new inference evidence, not recovered original logits. See runs/verification_2026-10-28/.
