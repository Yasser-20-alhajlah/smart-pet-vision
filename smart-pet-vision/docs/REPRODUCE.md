# Reproducing Smart Pet Vision

Scope: reproduce Phase 1 (M1-M3) only on a clean Colab T4 runtime.

## Steps
1. Open Colab, Runtime > Change runtime type > T4 GPU.
2. Make W&B available: run `wandb login` once in the session, or store the key as a Colab secret named `WANDB_API_KEY` and load it with `import os; from google.colab import userdata; os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")` (never paste the key into a cell or a file).
3. Add the shared folder `smart-pet-vision` to My Drive (a shortcut is enough), at most two folders deep, e.g. `MyDrive/capstone /smart-pet-vision`. The setup cell of every notebook mounts Google Drive first, then finds the folder and checks that `configs/config.yaml` and `src/spv` exist; if it finds no copy or more than one, it stops and asks you to set `SPV_REPO` to the exact path.
4. Running a notebook end to end retrains and overwrites `runs/`. To reproduce without touching the submitted results, copy the folder without `runs/` (for example to `/content/repro/smart-pet-vision`) and set `os.environ["SPV_REPO"]` to the copy before running. The 23 Oct 2026 reproduction was run this way.
5. Run notebooks 01 and 02 in order. Measured on a T4 on 23 Oct 2026: 01 about 4 min; 02 about 41 min. Notebooks 03-05 belong to Phase 2 and are not required here.
6. Compare the dataset metadata and classification files below. Use the selected checkpoint, not the final-epoch W&B Summary, for validation model selection.

## Expected values (run of 18-19 October 2026, seed 42)
| File | Value | Tolerance |
|---|---|---|
| runs/data/dataset_card_values.json | 7,335 images; split 5,123 / 1,106 / 1,106; leakage 0 | exact |
| runs/classification/comparison.csv | fine-tuned test accuracy 0.999; scratch 0.906 | +/- 0.01 (scratch +/- 0.03) |

GPU training is not guaranteed to be bit-exact across environments, so small differences within the tolerance are acceptable.

## W&B runs
- Submitted classification runs (group `full-rerun-2026-10-18`): scratch https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/gh7ayx6h, frozen https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/xsz5afr9, fine-tuned https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/8nvxbsn2
- Reproduction (classification runs): https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/groups/reproduction-2026-10-23

## Reproduction record
| Reproduced by | Date | Environment | Result |
|---|---|---|---|
| Yasser Alhajlah | 2026-10-23 | Clean Colab Tesla T4, repository copy without runs/; split rebuilt | Phase 1 notebooks 01 and 02 completed; saved data/split and classification comparison match the submitted results. The original full-project record also includes Phase 2; it is not required to reproduce Phase 1. |

## Changes after the reproduction (24 Oct 2026)
These changes follow the internal technical review. None of them changes a model, a split or a metric.
- Setup cell of notebooks 01, 02, 03 and 05: mounts Drive before finding the project (the same cell notebook 04 already used).
- Notebook 01: the review summary reads `ok` as 1/0 (y/n still accepted) and prints the completed audits; the review instructions were replaced by a description of what was done.
- Notebook 02: the draft instruction in the error-analysis cell was replaced by the final analysis.
- `src/spv/data.py`: an unreadable trimap is now dropped with the reason `unreadable_trimap` instead of stopping the run. All 7,349 trimaps were readable in both runs, so the cleaned set is unchanged.


## Report revision (26 October 2026)
The report uses the instructor template. Notebook 02 explanatory markdown was corrected; code cells and stored outputs were preserved. No model was retrained for this revision. Original audit assistance and reviewer provenance were retained. Historical runs do not contain training accuracy or validation loss; those curves must not be invented.

## Best checkpoint versus final epoch
Scratch: epoch 25; frozen: first maximum at epoch 6; fine-tuned: epoch 6 (validation accuracy and macro-F1 both 1.0). Fine-tuned final epoch 10 has macro-F1 0.998965182024. Saved test metrics use the retained checkpoint.


## Source notebook consistency check (26 October 2026)
The source notebook `notebooks/02_classification.ipynb` now uses the same error-analysis text as the executed submission notebook. Only markdown changed; source code cells, execution counts and stored outputs were preserved. No training run was performed for this synchronization.


## Final evidence review - 28 October 2026

Checkpoint reload checks establish agreement of aggregate test accuracy, not identical predictions on every image. `correct_count_difference` replaces the misleading historical field `images_different`; underlying values are unchanged. Shared training-loss and validation-accuracy/F1 histories agree within 1e-12; epoch times differ. Training accuracy and validation loss are from the logging rerun. See `docs/REVIEW_CHANGES_28OCT.md`.

The 401 team-collected and team-annotated photos remain one dataset, as confirmed by Mohammed on 28 October. Existing source records are retained; this is not a claim that the team personally captured every photo. The set has already been evaluated. Future model improvements require a new independent final test set. Saved pipeline predictions and a metric verification script are included in `runs/team_evaluation/`.

## New independent checkpoint verification (28 October 2026)
Use the fixed release source and submitted checkpoint bytes. With original Oxford JPEGs available, run:

    python tools/evaluate_checkpoints.py --images /absolute/path/to/images --output /absolute/path/to/new_verification --device cpu
    python tools/verify_prediction_exports.py /absolute/path/to/new_verification

Use --device cuda on Colab T4 or --device mps on Apple Silicon. The output must be a new directory. Every validation/test JPEG is checked against metadata MD5 first. This exports fresh test/validation logits and test probabilities, preserves saved temperatures and results, and refits temperature on validation only as a diagnostic. This does not retrain, select a new model or establish teammate clean training reproduction. See notebooks/03_independent_checkpoint_verification.ipynb and docs/RELEASE_IMPROVEMENTS.md for actual execution and environment.

`report_source/build_report.py` rebuilds the corrected PDF from a privacy-redacted base plus editable corrections. 16/16 pages match the current report's text and rendered pixels. This is current revision source, not recovered original LaTeX.
