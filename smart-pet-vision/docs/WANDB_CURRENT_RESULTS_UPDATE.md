# Published Phase 1 update — 28 October 2026

Published in the existing scientific report; historical narrative/panels retained.

**Use this section for the current M1–M3 submission.** The sections below dated 26 October are retained as historical analysis, including superseded figures. In particular, the earlier frozen test value of 99.37% is superseded for the submitted checkpoints; the current frozen result is **99.9096% (1105/1106)**. Detection, segmentation and deployment material below belongs to Phase 2 and was not re-audited in this update.



### Data and selection protocol

Oxford-IIIT Pet (CC BY-SA 4.0): 7349 original images, 14 empty trimaps excluded, 7335 retained (2366 cats, 4969 dogs; 37 breeds). Train/validation/test: 5123/1106/1106. Seed 42. Exact MD5 and pHash duplicate groups do not cross splits. There are no authoritative individual-animal IDs, so identity independence is not established. Similarity filtering is sensitivity analysis, not a causal leakage bound. Label audit: 0 species errors in 200, rule-of-three 95% upper bound 1.5%; four doubtful breed labels in 200 and four loose derived boxes in 300.

Whole-image binary cat/dog classification, 224×224 input, ImageNet normalization. Scratch: custom CNN; frozen and fully fine-tuned: ResNet-50. Training uses random resized crops (scale 0.5–1.0), horizontal flip and color jitter; validation/test use resize and center crop. Select the first checkpoint attaining maximum validation macro-F1, not the final-epoch Summary and not the best test score. Fine-tuned is selected by validation macro-F1. These comparisons do not isolate training strategy from model size and learning-rate differences.



| Submitted training run | Best epoch | Best validation accuracy | Best validation macro-F1 |
| --- | --- | --- | --- |
| scratch — gh7ayx6h | 25 | 0.927667270 | 0.917515932 |
| frozen — xsz5afr9 | 6 | 0.998191682 | 0.997928839 |
| fine-tuned — 8nvxbsn2 | 6 | 1.000000000 | 1.000000000 |



### Current final test results — 1106 images



| Model | Correct/n | Accuracy | Macro-F1 | ECE raw | ECE calibrated | Temperature |
| --- | --- | --- | --- | --- | --- | --- |
| Scratch | 1002/1106 | 0.9059674503 | 0.8922996255 | 0.0135853811 | 0.0247622819 | 0.8230744600 |
| Frozen | 1105/1106 | 0.9990958409 | 0.9989651820 | 0.0109669426 | 0.0012126884 | 0.4895465076 |
| Fine-tuned | 1105/1106 | 0.9990958409 | 0.9989651820 | 0.0014543669 | 0.0009041591 | 0.05 (lower bound) |

Confusion matrices, rows true and columns predicted, class order cat/dog: scratch [[304,53],[51,698]]; frozen [[356,1],[0,749]]; fine-tuned [[356,1],[0,749]]. Decisions use argmax, not a confidence threshold. Wilson 95% accuracy intervals: scratch [0.887,0.922], transfer models [0.995,1.000]. Selected fine-tuned per-class precision/recall/F1: cat 1.000000/0.997199/0.998597 (n=357), dog 0.998667/1.000000/0.999333 (n=749). Binary top-2 is trivially 1.0 and provides no additional discrimination.

Temperature is fitted on validation only; ECE uses 15 equal-width confidence bins. Calibration worsens scratch ECE. The selected temperature reaching the 0.05 lower bound and near-separable validation data limit calibration conclusions. Small ECE differences do not establish statistical significance or shelter-domain calibration. A 0.90 referral threshold is provisional; staff approval remains required before publishing.



### Learning curves and independent checkpoint verification

The original runs did not log train accuracy or validation loss. Full-logging reruns bvns5vir (scratch), 4ixtdkiq (frozen) and b3t0emrs (fine-tuned) supply train loss, train accuracy, validation loss, validation accuracy and validation macro-F1. Shared original/rerun training loss and validation accuracy/F1 histories agree within 1e-12; runtime differs. Added curves are rerun measurements, not recovered original logs. Submitted checkpoints were retained.

Fresh float32 inference on Apple MPS on 28 October checked all 2212 validation/test JPEGs against original MD5. All three test accuracies and confusion matrices matched the saved results exactly; calibrated ECE differences were below 7e-9. Test/validation logits, per-image probabilities and environment records are saved in runs/verification_2026-10-28/. Independent standard-library recomputation of accuracy, macro-F1, matrices and ECE passed. This was fresh checkpoint inference, not retraining, an original historical-logit recovery, or teammate clean-training reproduction. Frozen and fine-tuned predictions differ on two images despite equal aggregate accuracy.



### Team data, provenance and version

The 401 team-collected and annotated photos remain one dataset (401 masks; 503 boxes). Two mixed-species images are excluded from binary image-level scoring. Saved integrated-pipeline evidence on detector crops: before label corrections 383/399 (0.9598997494); after four corrections 387/399 (0.9699248120), macro-F1 0.9696639465. This is supplemental Phase 2 pipeline evidence, not a new Phase 1 whole-image experiment. Disagreement-focused label correction can bias the score upward. The set has already been evaluated; future improvements require a new independent final test.

Verification source Git commit: c43589ec6fa1042d9ce680326d42a018ebc2031c. This is a new review-time source snapshot, not a historical training commit. The final source revision and bundle SHA-256 are recorded in RELEASE_VERSION.json; restore Phase1_Source.bundle. The corrected PDF is not currently reproducible from a clean clone because the documented build script and its required base PDF are not included. Full semantic original LaTeX was not recovered. The team's 23 October reproduction record remains distinct from the new automated checkpoint verification.

**Topic approval is pending.** Mohammed clarified on 28 October that instructor approval will be requested after sending the project. No approval by 19 October is claimed. The original-topic bonus is not established. Phase 1 is assessed out of 40 (12/20/8); any self-score is not an awarded grade.

[Current final submission folder](https://drive.google.com/drive/folders/1M7frAxmWt-eJO2p2FxHP5Yc3Ejn8J4FO) · [Current Phase 1 PDF (includes dataset-card summary)](https://drive.google.com/file/d/1gQPsRRZtiwHGhcvW-bX3oRULDl20U--9/view) · [Source and evidence package](https://drive.google.com/file/d/1uGf_mViUfX7DeIKZWt6o10-wCMr1qskR/view) · [Scratch training](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/gh7ayx6h) · [Frozen training](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/xsz5afr9) · [Fine-tuned training](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/8nvxbsn2)

The Drive files remain private to the team under the user's access policy; these links do not promise anonymous access. The current numerical evidence is stated here publicly. Oxford reference: Parkhi et al. (2012), Cats and Dogs; calibration reference: Guo et al. (2017), On Calibration of Modern Neural Networks.



## Historical report — 26 October 2026

The following original narrative and figures are preserved for traceability. For current Phase 1 submission values, use the verified section above.
