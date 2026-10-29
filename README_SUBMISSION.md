# Smart Pet Vision - Phase 1 submission (AI251P Applied Computer Vision)

Team: Smart Pet Vision. Phase 1 covers M1 (data foundations) and M2-M3 (classification).

| Name | Student ID |
|---|---|
| Abdulrahman Mastour Alharthi | 225478 |
| Yasser Mansour Alhajlah | 225481 |
| Mohammed Adeeb Alrashdi | 225488 |
| Musaad Obaied Saad Alqahtani | 225471 |

## Phase 1 scope

Phase 1 covers:
- M1: data inspection, cleaning, duplicate grouping, splitting, audits and dataset documentation.
- M2: CNN training and evaluation.
- M3: transfer learning and W&B experiment tracking.

The task is whole-image binary classification: cat or dog.
Detection, segmentation and integrated deployment are outside the Phase 1 evaluation.

## Public submission files

- [Phase 1 PDF](./smart-pet-vision/SmartPetVision_Phase1.pdf)
- [Source code and notebooks](./smart-pet-vision/)
- [Dataset card](./smart-pet-vision/docs/dataset_card.md)
- [Configuration](./smart-pet-vision/configs/config.yaml)
- [Pinned requirements](./smart-pet-vision/requirements.txt)
- [GitHub Release: phase1-v1.0](https://github.com/Yasser-20-alhajlah/smart-pet-vision/releases/tag/phase1-v1.0)
- [Checkpoint archive](https://github.com/Yasser-20-alhajlah/smart-pet-vision/releases/download/phase1-v1.0/Phase1_Checkpoints.zip)
- [Results and verification archive](https://github.com/Yasser-20-alhajlah/smart-pet-vision/releases/download/phase1-v1.0/Phase1_Results_v2.zip)
- [W&B project](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision)
- [Phase 1 W&B report](https://forge.coreweave.com/wandb/yasser-alhajlah-kaust/smart-pet-vision/reports/Smart-Pet-Vision-Phase-1-Data-Foundations-CNN-Classification-and-Transfer-Learning--VmlldzoxODE4NTQ1Mw)

The linked PDF is the final Phase 1 submission report.

## Restoring the saved evidence

Use a separate working copy. Do not overwrite the original submitted results.

1. Download the repository.
2. Download and extract Phase1_Results_v2.zip.
3. Inside the extracted folder named "مجلد جديد", locate:
   - data/
   - classification/
   - verification_2026-10-28/
4. Copy these three folders into smart-pet-vision/runs/.
5. The results archive already includes the three checkpoints.
   Phase1_Checkpoints.zip provides the same checkpoint files separately.

The resulting project layout should be:

    smart-pet-vision/
      configs/config.yaml
      requirements.txt
      src/spv/
      notebooks/
      runs/
        data/
          metadata.csv
          dataset_card_values.json
          review/
        classification/
          scratch.pt
          frozen.pt
          finetune.pt
          comparison.csv
          train_info.json
          scratch_history.csv
          frozen_history.csv
          finetune_history.csv
        verification_2026-10-28/
          environment.json
          scratch_predictions.csv
          frozen_predictions.csv
          finetune_predictions.csv

Do not place the enclosing "مجلد جديد" folder inside runs/.

## Checking saved predictions without retraining

From the inner smart-pet-vision project directory, run:

    python tools/verify_prediction_exports.py runs/verification_2026-10-28

This recomputes accuracy, macro-F1, confusion matrices and 15-bin ECE
from the exported prediction CSV files. It writes a recomputation report
in the verification directory.

This checks saved prediction evidence. It does not load the models,
perform fresh inference or reproduce training.

## Reproducing training

Use a separate project copy and a clean Colab T4 runtime.

1. Open notebooks/01_data_preparation.ipynb.
2. Set SPV_REPO to the inner project directory if automatic discovery fails.
3. Follow the setup cell, install the pinned requirements and authenticate
   W&B if logging is enabled.
4. Run notebook 01, then notebooks/02_classification.ipynb.
5. Compare the generated data counts, histories and classification results
   with the reference values below.

Oxford-IIIT Pet is downloaded by the code at runtime.
Private team photographs are not required for notebooks 01 and 02.

Running these notebooks retrains the models and writes new results.
Do not run them over the only copy of the submitted evidence.

The documented training environment is Colab T4. Identical results
across different software and hardware environments are not guaranteed.

## Retraining verification — 29 October 2026

| Model | Run ID | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| [Scratch](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/9fay48ne) | `9fay48ne` | 0.905967 | 0.892300 |
| [Frozen](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/qlzja26v) | `qlzja26v` | 0.999096 | 0.998965 |
| [Fine-tuned](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/runs/2ptjc52f) | `2ptjc52f` | 0.999096 | 0.998965 |

## Reference results

Dataset after cleaning: 7,335 images.
Train / validation / test: 5,123 / 1,106 / 1,106.

| Model | Best validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Scratch CNN | 0.917516 | 0.905967 | 0.892300 |
| Frozen ResNet-50 | 0.997929 | 0.999096 | 0.998965 |
| Fine-tuned ResNet-50 | 1.000000 | 0.999096 | 0.998965 |

Checkpoint selection uses validation macro-F1, not test performance.
The first best epochs are 25, 6 and 6 respectively.
Final-epoch W&B summaries must not be confused with best-checkpoint results.

## Experiment provenance

Original submitted training runs:
- Scratch: gh7ayx6h
- Frozen: xsz5afr9
- Fine-tuned: 8nvxbsn2

Latest full-logging reruns:
- Scratch: bvns5vir
- Frozen: 4ixtdkiq
- Fine-tuned: b3t0emrs

The additional training-accuracy and validation-loss curves come from
the full-logging reruns. They are not recovered original training logs.

Current Git commits identify published source revisions, not necessarily
the source revision used during the original training.

## Current availability limits

The inspected public repository and release do not currently provide:
- MANIFEST_SHA256.txt
- RELEASE_VERSION.json
- Phase1_Source.bundle
- Executed notebooks with stored outputs
- The runs/supplement/ evidence directory

These files must not be described as publicly included until uploaded.

REPRODUCTION_LOG.txt is currently empty.

A PDF-editing script exists at tools/build_report.py, but its required
tools/assets/base_report.pdf is absent. Rebuilding the PDF from a clean
checkout is therefore not currently supported.

Historical references to a local Drive submission package do not establish
that all of that package's files are available in this public repository.

## Scientific and administrative limitations

- Duplicate-group separation does not establish animal-identity independence.
- Image-similarity filtering is a sensitivity analysis, not a causal bound
  on leakage.
- Calibration improvements do not establish reliable confidence on
  shelter-domain photographs.
- Instructor topic approval remains pending.
- Any 40/40 in the PDF is the team's self-assessment, not an awarded grade.

## Privacy and submission

Do not upload API keys, private team photographs, owner-consent records
or unprocessed photographs containing personal metadata.

Follow the live Blackboard assignment instructions for required files.
Additional archives should be submitted only if allowed or requested.
Retain the submission receipt.
