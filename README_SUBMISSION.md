# Smart Pet Vision - Phase 1 submission (AI251P Applied Computer Vision)

Team: Smart Pet Vision. Phase 1 covers M1 (data foundations) and M2-M3 (classification).

| Name | Student ID |
|---|---|
| Abdulrahman Mastour Alharthi | 225478 |
| Yasser Mansour Alhajlah | 225481 |
| Mohammed Adeeb Alrashdi | 225488 |
| Musaad Obaied Saad Alqahtani | 225471 |

## What is submitted on Blackboard

| File | Content |
|---|---|
| `SmartPetVision_Phase1.pdf` | Phase 1 report (instructor template), with an Appendix A dataset-card summary; the full card is in `smart-pet-vision/docs/dataset_card.md` |
| Supporting code/results ZIP (only if the live Blackboard assignment accepts it) | Phase 1 code, notebooks and permitted results only; do **not** include the private team photographs by default |

**Submission rule:** The Blackboard instructions reviewed by the team explicitly require `SmartPetVision_Phase1.pdf`. A ZIP is not an independently confirmed mandatory attachment. Upload additional files only when the live assignment allows or requests them. One team member submits and retains the receipt.

**Current report and source:** [Phase 1 PDF](https://drive.google.com/file/d/1gQPsRRZtiwHGhcvW-bX3oRULDl20U--9/view) · [Phase 1 source snapshot](https://drive.google.com/drive/folders/1_8Plxl5jx9DJT5nlbv1E17TfPGpx_PaU) · [W&B](https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision). Drive remains private to the four team members. The team manages public W&B access for the instructor.

## Contents of this folder

```
final phase1/
  README_SUBMISSION.md        this file
  MANIFEST_SHA256.txt         SHA-256 of all repository files including supplement and team-evaluation evidence (fixed identifier)
  SmartPetVision_Phase1.pdf   the report (same file as the one uploaded separately)
  smart-pet-vision/           Phase 1 code repository
    README.md                 project overview and how to run
    requirements.txt          all 19 dependencies pinned to the exact versions of the 18-19 Oct 2026 run
    configs/config.yaml       the single configuration file (seed 42, all paths and hyperparameters)
    src/spv/                  data.py (M1), classify.py (M2-M3), common.py (configuration, seeding, W&B),
                              supplement.py (checkpoint check, learning curves, image-similarity sensitivity, team-set audit)
    notebooks/                01_data_preparation, 02_classification (source) and corrected 02b_phase1_supplement source
    notebooks/executed/       01, 02 and historical 02b with stored outputs
    docs/                     dataset_card.md, REPRODUCE.md
    runs/data/                split (metadata.csv), near-duplicate groups, dropped files, dataset statistics,
                              dataset_card_values.json, label audit (review/labels), box review (review/boxes)
    runs/classification/      best checkpoints (scratch.pt, frozen.pt, finetune.pt), comparison.csv, train_info.json,
                              training histories, test metrics (JSON), error lists and figures (learning curves,
                              confusion matrices, reliability diagrams, error galleries)
    runs/supplement/          outputs of notebook 02b: checkpoint hashes and test-accuracy check, full learning curves
                              (curves/), image-similarity sensitivity analysis (identity_*), team-set audit summary (team_audit.json)
  SmartPetVision_TeamData/    team-collected external test set: 401 photos (images/), 401 masks (masks/),
                              CVAT annotations (annotation/done/), labels, boxes, label corrections,
                              source map, quality audit, README, and audit_02b/ (notebook 02b: all 401 photos with their boxes,
                              per-image audit). Private to the team; excluded from the supporting ZIP
```

## How to reproduce

1. Keep `smart-pet-vision/` and `SmartPetVision_TeamData/` side by side, as in this folder. The configuration finds the team data through `paths.team: ../SmartPetVision_TeamData`. Phase 1 (notebooks 01 and 02) does not read the team data; it supports the dataset card and the external test.
2. Put the folder in Google Drive and open the notebooks in Colab with a T4 GPU. If the setup cell does not find the project by itself, set `os.environ["SPV_REPO"]` to the path of `smart-pet-vision`.
3. Run `notebooks/01_data_preparation.ipynb`, then `notebooks/02_classification.ipynb` (about 4 and 41 minutes on a T4). Oxford-IIIT Pet is downloaded at run time from the official VGG page to the runtime disk; it is not in this folder.
4. Optional: run `notebooks/02b_phase1_supplement.ipynb` (about 45 minutes); it retrains for additional logging; numeric outputs go to `runs/supplement/` and private review images to `SmartPetVision_TeamData/audit_02b/`.
5. Compare `runs/classification/comparison.csv` and `runs/data/dataset_card_values.json` with the expected values in `docs/REPRODUCE.md`. Running the notebooks overwrites `runs/`; work on a copy to keep the submitted results.
6. Check the files against `MANIFEST_SHA256.txt` (from the folder that contains it): `sha256sum -c MANIFEST_SHA256.txt`.
7. W&B logging: run `wandb login`, or store the key as the Colab secret `WANDB_API_KEY`. The key is never written in a cell or a file.

## Weights & Biases

- Project: https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision
- Phase 1 runs (group `full-rerun-2026-10-18`): scratch `gh7ayx6h`, frozen `xsz5afr9`, fine-tuned `8nvxbsn2`; reproduction group `reproduction-2026-10-23`
- Full-logging rerun (notebook 02b, 28 Oct 2026; same values as the submitted runs plus training accuracy and validation loss): scratch `bvns5vir`, frozen `4ixtdkiq`, fine-tuned `b3t0emrs`
- W&B access has been reported as public by the team; independently confirm the links open for the intended grader
- Summary report: https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision/reports/Smart-Pet-Vision-A-Scientific-Report-on-Classification-Detection-and-Segmentation--VmlldzoxODE4NTQ1Mw

## Not included, and why

- **Oxford-IIIT Pet images**: downloaded by the code at run time from the official source (CC BY-SA 4.0); not redistributed.
- **Phase 2 material** (notebooks 03-05, the detection, segmentation, pipeline, service and deployment modules, the model card and their results): part of the Phase 2 submission. `configs/config.yaml` still holds the Phase 2 settings because it is the single file used for every run.
- **Unprocessed team photos** and the 13 excluded photos: kept privately by the team (they may carry camera metadata).
- **Local W&B logs** (`wandb/`) written by the notebook 02b run: the runs are on the W&B project instead.

## Version note

The project was developed in a shared Google Drive folder, not in git, so there is no commit hash. This folder is a snapshot of 28 October 2026, identified by `MANIFEST_SHA256.txt`. The reported results are traced through the W&B runs above and the executed notebooks in `smart-pet-vision/notebooks/executed/`. The submitted models were not replaced: notebook 02b repeated the three trainings only to log training accuracy and validation loss, and its checkpoints were not kept. Changes after the submitted run: the explanatory markdown of notebook 02 (code cells and outputs unchanged, 26 October), the paths to the team data in `README.md` and `docs/` (team data moved out of the repository on 26 October), on 27 October the Phase 1 scope of `README.md` and the provenance and external-test lines of `docs/dataset_card.md`, and on 28 October the name of the team set ("team-collected" instead of "team-captured") in `README.md`, `docs/dataset_card.md` and a comment of `configs/config.yaml` (no setting changed), and the storage line of `docs/dataset_card.md`. Also on 28 October: the best checkpoints of the submitted run were added to `runs/classification/` (copied from the run folder of 18 October; reloading them reproduces the submitted test accuracy exactly); `classify.train` now also logs training accuracy and validation loss (no change to training: shared logged loss/accuracy/F1 histories agree within 1e-12; runtimes differ); `src/spv/supplement.py` and notebook 02b were added with their outputs in `runs/supplement/`; `README.md` and `docs/dataset_card.md` were updated with the image-similarity sensitivity analysis and the team-set audit. The audit's contact sheets of the team photos were moved from `runs/supplement/team_review/` to `SmartPetVision_TeamData/audit_02b/`, so that no team photo sits in the repository.

## Scientific interpretation and final QA (28 October 2026)

- **Similarity is not identity:** nearest same-breed train/test similarity and filtered-subset accuracy are a *sensitivity analysis*. Animal IDs are unavailable; the study cannot establish identity-disjointness, a causal leakage effect or a numeric upper bound on leakage. The selected classifier achieved 1,105/1,106 accuracy (0.999096) on the full held-out test split; after excluding the 790 cases above the 0.75 similarity cutoff, accuracy was 315/316 (0.996835). The exclusion changes the evaluated population.
- **Calibration:** the small ECE difference was observed, but statistical significance was not assessed; confidence is not guaranteed calibrated for shelter photos.
- **Review policy:** `ready` is a recommendation for staff review, **not** authorization to publish automatically. Phase 1 delivers the binary Cat/Dog classifier; detection, segmentation and publishing controls belong to Phase 2.
- **Self-assessment:** any 40/40 reported in the PDF is the **team's self-score**, not an awarded grade.
- **Integrity:** `MANIFEST_SHA256.txt` covers all delivered repository files, including generated supplement evidence. PACKAGE_SHA256.txt additionally covers the local report and supporting documents. The private team dataset is maintained separately.
- **Privacy:** avoid uploading `SmartPetVision_TeamData/` into a public ZIP or repository. If an instructor needs to examine it, provide access under the project's consent and institution rules.

Final QA involved scientific wording, source-link and documentation edits; it did not alter model weights, experiment histories, training datasets or test metrics.


## Review closure
The corrected 02b source has outputs cleared; the prior executed record is retained separately. Saved team evaluation evidence and a metric-check script are included. No GPU training or fresh model inference was performed during this review. See REVIEW_FINAL_AR.docx and smart-pet-vision/docs/REVIEW_CHANGES_28OCT.md for remaining access, approval and bonus conditions. The supporting ZIP contains no private team photographs.

## Independent audit reconciliation (28 October, final pass)
See SmartPetVision_Phase1_Audit_Reconciled.pdf for the October 26 audit versus current final comparison. Use this final folder, not the historical 8-page REVIEW package. 52 saved-evidence checks and 12 focused function/curve checks passed. W&B live matching, original approval evidence and clean GPU reproduction remain outside completed verification.

SOURCE_AND_EVIDENCE.zip excludes .pt checkpoint bytes to stay below the connector upload limit. Existing checkpoints remain under smart-pet-vision/runs/classification/ in Drive; restore them to that path before checking the full repository manifest. Historical runs/supplement/repo_sha256.txt is a run-time snapshot, not the final release manifest. MANIFEST_SHA256.txt is authoritative for final repository bytes.

## Final evidence improvement pass
A complete current Git source snapshot is provided as Phase1_Source.bundle with full commit and bundle hash in RELEASE_VERSION.json. Restore with git clone Phase1_Source.bundle smart-pet-vision. It is newly created at review time, not original training history. Generated run evidence and weights are separate from Git and identified by the release manifests.

The corrected 16-page PDF can be rebuilt from report_source/build_report.py and its preserved redacted base; a restored Git checkout produced identical text and rendered pixels on all 16 pages. This is current PDF revision source, not recovered original LaTeX.

Fresh inference of all three checkpoints was completed on the local Mac using exactly fingerprint-matching validation/test JPEGs. Accuracy and confusion matrices match saved results; per-image logits/probabilities and independent ECE recalculation are now in runs/verification_2026-10-28/. No original result, checkpoint or training split was replaced. See docs/RELEASE_IMPROVEMENTS.md. Live W&B edits and original approval/teammate training-reproduction proof remain external conditions.

## Latest user clarification - topic approval pending
Mohammed confirmed that instructor approval has not yet been granted and will be sought after sending the project. Earlier approval-by-19-October statements are withdrawn. Technical readiness is separate from instructor topic acceptance. No original-topic bonus is certified.

## Live W&B publication complete
The existing scientific report now contains the current Phase 1 tables and verification provenance, with historical content explicitly retained. Frozen final-test notes were completed. Access was obtained after Mohammed authorized the Forge domain and signed in; the earlier access block is resolved. Topic approval remains pending, per Mohammed.
