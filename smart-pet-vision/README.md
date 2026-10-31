# Smart Pet Vision

AI251P Applied Computer Vision capstone. The Phase 1 deliverable covers whole-image cat/dog
classification: data foundations (M1), CNN training and evaluation (M2), and transfer learning
with W&B experiment tracking (M3). Detection, segmentation, integrated deployment and staff-referral
behavior are Phase 2 scope and are not evaluated as Phase 1 results.

**Phase 2 (M4 detection, M5 segmentation, M8 integration)** is in [`phase2/`](phase2/README.md): code, tests, the Phase 2 report (`phase2/report/SmartPetVision_Phase2_Report.pdf`) and the requirements checklist.

Data: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., CVPR 2012),
CC BY-SA 4.0. Team-collected photos are used for external testing only.

## How to run (Google Colab, T4 GPU)

1. Add this folder to My Drive (a shortcut is enough), at most two folders deep, e.g. `MyDrive/capstone /smart-pet-vision`.
2. For Phase 1 training, run notebooks 01 and 02 in order. Their setup cells mount Google Drive, find the project and check that `configs/config.yaml` and `src/spv` exist; set `SPV_REPO` to the exact path if no unique copy is found.
3. Running a notebook end to end retrains and overwrites `runs/`. To reproduce without touching the submitted results, copy the folder without `runs/` and set `os.environ["SPV_REPO"]` to the copy (see `docs/REPRODUCE.md`).

| Notebook | Milestone | Phase | Approximate T4 runtime (run-specific) |
|---|---|---|---|
| `01_data_preparation` | M1 cleaning, de-duplication, split, review sheets | 1 | about 4 min |
| `02_classification` | M2-M3 scratch CNN vs. ResNet-50 frozen vs. fine-tuned | 1 | about 41 min |
| `02b_phase1_supplement` | Optional Phase 1 supplemental checks and reruns: checkpoint check, learning curves, image-similarity sensitivity analysis and team-set audit; outputs go to `runs/supplement/` | 1 | about 45 min |
| `03_independent_checkpoint_verification` | Optional fresh checkpoint inference; not training reproduction | 1 | Run-specific |

Notebooks 01 and 02 timings were recorded on 23 Oct 2026; notebook 02b was run on 28 Oct 2026. Notebook 03 verifies restored checkpoints using fresh inference; it does not retrain the models or establish a teammate training reproduction. Every setting lives in `configs/config.yaml`. Results are
written to `runs/` on Drive; images are re-downloaded to local disk each session, and the split is
fixed by `runs/data/metadata.csv` after the first run of notebook 01.

To log runs to W&B, set `wandb.enabled: true` and run `wandb login` once, or store the key as a Colab secret `WANDB_API_KEY` and load it into `os.environ` (never paste the key into a cell or a file).

## Repository

```
configs/config.yaml     all hyperparameters, thresholds and paths
src/spv/data.py         M1: cleaning, MD5 + pHash duplicate groups, group-aware split, masks, YOLO export
src/spv/classify.py     M2-M3: models, training, temperature scaling, ECE, per-breed errors
src/spv/common.py       configuration, seeding, W&B tracking, JSON helpers
src/spv/supplement.py   checkpoint hashes, learning-curve plots, image-similarity sensitivity analysis, team-set audit
notebooks/              Source notebooks; executed notebooks with stored outputs
                        are not included in the current public package.
docs/                   Dataset card and reproduction documentation.
runs/                   Restore data/, classification/ and verification_2026-10-28/
                        from Phase1_Results_v2.zip. The public archive does not
                        include runs/supplement/ or runs/team_evaluation/.
```

## Design decisions worth reviewing

- **Leakage.** Exact duplicates are removed, near-duplicates (pHash Hamming <= 4) are grouped, and a
  group never crosses splits. Notebook 01 stops if the audit fails. This does not separate different
  photos of the same animal (the dataset has no identity labels). Notebook 02b performs a similarity-based
  sensitivity analysis using nearest same-breed training neighbours: the selected model's accuracy remains
  0.997-0.999 across the retained subsets. This cannot measure or bound the causal effect of identity leakage;
  filtering changes the test-set composition and verified pet identities are unavailable.
- **Body boxes.** The dataset has head boxes only. Body boxes are the tight box around trimap
  foreground + border, checked by eye on 300 images.
- **Thresholds.** The classifier temperature is chosen on validation; the test split is read once per model.
- **Refer to staff** (Phase 2 system behavior, outside the Phase 1 evaluation) when: no animal, more than one animal, calibrated
  confidence < 0.90, detector and classifier disagree, or the mask covers < 5% of the frame. The 0.90 threshold is
  provisional: the temperature fitted on the near-separable validation split is not a reliable probability scale,
  so the threshold will be fixed on a harder, independent calibration set.

Status (24 Oct 2026): both reviews are complete (`runs/data/review/labels/review.csv`, 200 images; `runs/data/review/boxes/review.csv`, 300 boxes; both confirmed by the team), and the 401 team photos with labels and masks are in the folder `SmartPetVision_TeamData` next to this repository (moved out of `data/team/` on 26 Oct 2026; see `docs/REPRODUCE.md`).


## Final evidence review - 28 October 2026

Checkpoint reload checks establish agreement of aggregate test accuracy, not identical predictions on every image. `correct_count_difference` replaces the misleading historical field `images_different`; underlying values are unchanged. Shared training-loss and validation-accuracy/F1 histories agree within 1e-12; epoch times differ. Training accuracy and validation loss are from the logging rerun. See `docs/REVIEW_CHANGES_28OCT.md`.

The 401 team-collected and team-annotated photos remain one dataset, as confirmed by Mohammed on 28 October. Existing source records are retained; this is not a claim that the team personally captured every photo. The set has already been evaluated. Future model improvements require a new independent final test set. Pipeline evaluation evidence under `runs/team_evaluation/` is supplemental Phase 2 evidence; it is not included in the public Results v2 archive and is outside the Phase 1 classification scope.

## Repository and Version Control

- **GitHub Repository:** https://github.com/Yasser-20-alhajlah/smart-pet-vision
- **Version identification:** use the Git commit of the downloaded
  checkout. Published source revisions are not necessarily the revision
  used during the original training.
- **Current submission instructions:** [README_SUBMISSION.md](../README_SUBMISSION.md)

## Model Weights and Checkpoints

Due to GitHub file size limits, the model weights (`.pt` files) and raw result files are not hosted in the main repository.
They have been uploaded to **GitHub Releases**.

Download them from the [release page](https://github.com/Yasser-20-alhajlah/smart-pet-vision/releases).