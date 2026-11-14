# Smart Pet Vision — Phase 1 report (M1–M3), standalone version

This is the current Phase 1 report text for Weights & Biases. It stands on its own: it covers data
foundations (M1) and cat/dog classification (M2–M3) only. Detection, segmentation and deployment are
reported in the Phase 2 report and are not repeated here. Values that were replaced are listed once, in
Appendix H, and are not used anywhere else in this document.

Sources of every number: `SmartPetVision_Phase1.pdf` (16 pages), `docs/WANDB_CURRENT_RESULTS_UPDATE.md`
(section of 28 October 2026) and `docs/WANDB_FROZEN_FINAL_NOTES.txt`. No model was trained for this
version; it only unifies the text and the sources of the figures.

## 1. Data and selection protocol

Oxford-IIIT Pet (CC BY-SA 4.0): 7349 original images, 14 empty trimaps excluded, 7335 retained (2366 cats,
4969 dogs; 37 breeds). Train/validation/test: 5123/1106/1106. Seed 42. Exact MD5 and pHash duplicate groups
do not cross splits. There are no authoritative individual-animal IDs, so identity independence is not
established. Similarity filtering is sensitivity analysis, not a causal leakage bound. Label audit: 0 species
errors in 200, rule-of-three 95% upper bound 1.5%; four doubtful breed labels in 200 and four loose derived
boxes in 300.

Whole-image binary cat/dog classification, 224×224 input, ImageNet normalization. Scratch: custom CNN;
frozen and fully fine-tuned: ResNet-50. Training uses random resized crops (scale 0.5–1.0), horizontal flip
and color jitter; validation/test use resize and center crop. The retained checkpoint is the first one
attaining maximum validation macro-F1, not the final-epoch summary and not the best test score. The
fine-tuned model is selected by validation macro-F1. These comparisons do not isolate training strategy
from model size and learning-rate differences.

| Submitted training run | Best epoch | Best validation accuracy | Best validation macro-F1 |
| --- | --- | --- | --- |
| scratch — gh7ayx6h | 25 | 0.927667270 | 0.917515932 |
| frozen — xsz5afr9 | 6 | 0.998191682 | 0.997928839 |
| fine-tuned — 8nvxbsn2 | 6 | 1.000000000 | 1.000000000 |

## 2. Final test results — 1106 images

| Model | Correct/n | Accuracy | Macro-F1 | ECE raw | ECE calibrated | Temperature |
| --- | --- | --- | --- | --- | --- | --- |
| Scratch | 1002/1106 | 0.9059674503 | 0.8922996255 | 0.0135853811 | 0.0247622819 | 0.8230744600 |
| Frozen | 1105/1106 | 0.9990958409 | 0.9989651820 | 0.0109669426 | 0.0012126884 | 0.4895465076 |
| Fine-tuned | 1105/1106 | 0.9990958409 | 0.9989651820 | 0.0014543669 | 0.0009041591 | 0.05 (lower bound) |

Confusion matrices, rows true and columns predicted, class order cat/dog: scratch [[304,53],[51,698]];
frozen [[356,1],[0,749]]; fine-tuned [[356,1],[0,749]]. Decisions use argmax, not a confidence threshold.
Wilson 95% accuracy intervals: scratch [0.887,0.922], transfer models [0.995,1.000]. Selected fine-tuned
per-class precision/recall/F1: cat 1.000000/0.997199/0.998597 (n=357), dog 0.998667/1.000000/0.999333
(n=749). Binary top-2 is trivially 1.0 and provides no additional discrimination.

Temperature is fitted on validation only; ECE uses 15 equal-width confidence bins. Calibration worsens
scratch ECE. The selected temperature reaching the 0.05 lower bound and near-separable validation data limit
calibration conclusions. Small ECE differences do not establish statistical significance or shelter-domain
calibration. A 0.90 referral threshold is provisional; staff approval remains required before publishing.

## 3. Learning curves

The curves in this report are drawn from the full-logging reruns, one run per model. The submitted runs
(Section 1) did not log training accuracy or validation loss, so they cannot supply these panels; they are
shown only in Appendix H.

| Model | Run that feeds the curves | Logged keys used (x = `epoch`) |
| --- | --- | --- |
| Scratch CNN | bvns5vir | `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1` |
| Frozen ResNet-50 | 4ixtdkiq | `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1` |
| Fine-tuned ResNet-50 | b3t0emrs | `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1` |

<!-- PANELS: curves-scratch -->

<!-- PANELS: curves-frozen -->

<!-- PANELS: curves-finetuned -->

The reruns repeat the three training configurations with the same configuration, seed and data order.
Shared training-loss, validation-accuracy and macro-F1 histories of the submitted runs and the reruns agree
within 1e-12; runtimes differ. Training accuracy and validation loss are rerun measurements, not recovered
original logs. The submitted checkpoints were retained.

Scratch appears under-trained: at epoch 25 training accuracy is approximately 0.914 and validation accuracy
0.928; training loss is approximately 0.204 and validation loss 0.193. Fine-tuned validation loss reaches
approximately 0.0018 at epoch 6 and rises to 0.0035 by epoch 10, consistent with mild overfitting after the
selected checkpoint. Frozen validation loss decreases from approximately 0.063 to 0.013.

## 4. Verification

Checkpoint inference, 28 October 2026: fresh float32 inference on Apple MPS checked all 2212 validation/test
JPEGs against original MD5. All three test accuracies and confusion matrices matched the saved results
exactly; calibrated ECE differences were below 7e-9. Independent standard-library recomputation of accuracy,
macro-F1, matrices and ECE passed. This was fresh checkpoint inference, not retraining. Frozen and fine-tuned
predictions differ on two images despite equal aggregate accuracy.

Clean retraining, 29 October 2026: all three classifiers were retrained and evaluated in a separate Google
Colab T4 working directory using seed 42 (25, 8 and 10 epochs). Across all recorded epochs the five logged
learning-curve metrics matched the full-logging reruns of Section 3, and the test results matched. This
verifies same-seed reproducibility; it does not estimate variability across independent seeds.

| Model | Clean-retraining run | Test accuracy | Test macro-F1 |
| --- | --- | --- | --- |
| Scratch | 9fay48ne | 0.905967 | 0.892300 |
| Frozen | qlzja26v | 0.999096 | 0.998965 |
| Fine-tuned | 2ptjc52f | 0.999096 | 0.998965 |

## 5. Team data and version

The 401 team-collected and annotated photos remain one dataset (401 masks; 503 boxes). Phase 1 training does
not use them. Two mixed-species images are excluded from binary image-level scoring. Saved integrated-pipeline
evidence on detector crops: before label corrections 383/399 (0.9598997494); after four corrections 387/399
(0.9699248120), macro-F1 0.9696639465. This is supplemental pipeline evidence, not a Phase 1 whole-image
experiment. Disagreement-focused label correction can bias the score upward. The set has already been
evaluated; future improvements require a new independent final test.

Published source snapshot: edcc29f2e035b329e577cd04bf6092dc1f4ae04c (29 October 2026); it does not identify
the historical training commit. Checkpoints and saved results are in release
[phase1-v1.0](https://github.com/Yasser-20-alhajlah/smart-pet-vision/releases/tag/phase1-v1.0).

Runs: [scratch gh7ayx6h](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/gh7ayx6h) ·
[frozen xsz5afr9](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/xsz5afr9) ·
[fine-tuned 8nvxbsn2](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/8nvxbsn2) ·
[rerun bvns5vir](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/bvns5vir) ·
[rerun 4ixtdkiq](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/4ixtdkiq) ·
[rerun b3t0emrs](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/b3t0emrs)

References: Parkhi et al. (2012), Cats and Dogs; Guo et al. (2017), On Calibration of Modern Neural Networks.

## Appendix H. Historical record (superseded; not used for the current results)

Nothing in this appendix is a current result. It is kept so that earlier statements can be traced.

| Item | Earlier value or source | Replaced by |
| --- | --- | --- |
| Frozen model, test accuracy | 99.37%, from the superseded evaluation run allez66m | 99.9096% (1105/1106), retained checkpoint of run xsz5afr9 (Section 2) |
| Learning-curve panels | Submitted runs gh7ayx6h, xsz5afr9, 8nvxbsn2, which logged training loss, validation accuracy and validation macro-F1 only | Full-logging reruns bvns5vir, 4ixtdkiq, b3t0emrs (Section 3) |
| Narrative and figures of 26 October 2026 | Earlier W&B report "Smart Pet Vision: A Scientific Report on Classification, Detection and Segmentation" | Sections 1–5 of this document |
| Detection, segmentation and deployment text | Same earlier report | The Phase 2 report; outside Phase 1 |

<!-- PANELS: historical-original-runs -->

Earlier report, kept unchanged for traceability:
https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/reports/Smart-Pet-Vision-A-Scientific-Report-on-Classification-Detection-and-Segmentation--VmlldzoxODE4NTQ1Mw

The team's reproduction record of 23 October 2026 (clean Colab T4, by the author of the original runs) is a
historical record and is distinct from the verification of 28–29 October in Section 4.
