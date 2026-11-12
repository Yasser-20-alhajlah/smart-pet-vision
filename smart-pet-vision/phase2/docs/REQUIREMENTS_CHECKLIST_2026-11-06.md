# Master requirements checklist — Smart Pet Vision Phase 2

Updated 12 November 2026. Earlier dated execution details are retained with their original scope. Current delivery is `FINAL PHASE_2 (PDF and source ZIP only)`; the PDF and archive index give its exact commit. The official Phase 2 assignment text supplied by the team is verified; the original Phase 1 template, original Phase 2 template and actual one-page 38/40 feedback have now been opened. The older general guidelines conflict with the later phase-specific templates and do not override them.

Legend: `[ ]` not reviewed · `[~]` in progress / partial · `[!]` blocked or failed · `[x]` verified
within the stated scope. `[x]` is used only after an evidence-based check in this session; "the
check was executed" is kept distinct from "the acceptance criterion passed". Sources: *Capstone
Project Guidelines* (October 2026; Tables 2, 5, 6, 7; §2.4; R1–R10), the instructor's Phase 2
example report, the **Phase 1 feedback sheet (38/40)**, and the instructor's messages recorded in
the audit tracker. Every item names its evidence; every human step names its owner.

> Updated on 6 Nov 2026 after the merge of PR #9 into `main` (item S3 and open item 3 only). The copy inside the
> delivered commit `f44fd4c` (and therefore inside the report's source ZIP and `02_Code/reviewed_source.zip`) predates
> the merge and still lists it as open; nothing else differs.

Delivered revision: the commit printed on page 1 of the report and in `CODE_ARCHIVE_INDEX.json`;
historical results: commit `879b0e8` (2 Nov) and the runs of 17–19 Oct (W&B `2seovilv`,
`3ztpz2tc`, `1jb5y453`, `fqow5ly7`), all on **split v1**.

## A. Explicit instructor-requested corrections (Phase 1 feedback)

### A1 `[x]` Leakage-safe split: group likely same-animal photos before splitting (Part 2, −1)
- **7 Nov 2026 update (owner review):** the owner reviewed the 86 groups himself and recorded `clear duplicate` for all 86 (reviewer YASSER; date and note fields empty); record `report/evidence/P2_same_animal_review_owner_20261107.json`, log in Drive `review/review_log_Yasser_20261107_1.xlsx`. Consequence: 135 of 1,106 test images (12.2%) have a duplicate across partitions; split v1 results may be optimistic. Split v2 was not promoted. On 7 November all three recipes were retrained once on v2 as supplementary evidence, recorded in `P2_supplementary_F_split_v2_retrain_20261107.json`.
- Original issue: only MD5/pHash groups (84 images in 41 groups); 22 test images with a same-breed training image at similarity ≥ 0.95 and 133 at ≥ 0.90; leak measured, not removed.
- Files changed: `tools/same_animal_groups.py` (new), `tests/test_same_animal_groups.py` (6 tests), `docs/LEAKAGE_SAME_ANIMAL_PROTOCOL.md` (frozen rules), `docs/dataset_card.md` (Phase 1), `docs/model_card.md`, report §1 "Response to the Phase 1 feedback".
- Repair: within-breed ResNet-50 feature links at cosine ≥ 0.90, transitive, merged with MD5/pHash groups; review band 0.85–0.90 and recorded review decisions; cross-partition test; split v2 with the unchanged Phase 1 splitter (seed 42). Rules frozen before any run.
- Verification: `PYTHONPATH=src python -m unittest tests.test_same_animal_groups` → 6/6 OK (synthetic features). **Run on the real images on 6 Nov 2026** (Colab T4, code from the delivered package, split v1 rebuilt and verified 5,123/1,106/1,106 first): 6,541 similarity groups, 4,361 automatic links, 12,647 review-band pairs; **86 groups / 771 images cross the v1 partitions** (test 135, val 127, train 509); 13 chain groups of 10–114 images (links at the 0.90 floor, visibly many different animals) hold 574 of them; 54 groups touch test; diagnostic 0.95 threshold → 18 groups / 115 images. Split v2 proposal 4,903/1,382/1,050, 0 crossing, integrity checked (val inflated to 18.8 % by the chain groups).
- Evidence: `report/evidence/P2_same_animal_grouping_run_20261106.json`, `report/evidence/identity_v2_20261106/crossing_groups_summary.csv`; full outputs, 86 contact sheets and the unfilled review log in Drive `runs/data/identity_v2_20261106T0931Z/`.
- Status wording: *86 groups crossing partitions; owner review (7 Nov) recorded all 86 as clear duplicates; 135 of 1,106 test images affected*. Not the served split; supplementary v2 retraining is recorded separately.
- Current scope: v2 is 4,903/1,382/1,050 with no crossing similarity group under the recorded procedure. The supplementary detector/segmenter/classifier runs exist; headline results and served weights remain v1. The requirement is likely same-animal grouping, not a certified identity registry; v2 implements it before splitting. Individual identity and the historical v1 scores remain limited; no retroactive grade change is claimed.

### A2 `[x]` Independent reproduction with numbers and run IDs beside the originals (Part 13, −1)
- **6 Nov 2026 update:** Level 1 executed from the delivered package (Colab T4, 15:24 UTC): all six metrics identical to the originals (difference 0.0, tolerance 0.005); record `report/evidence/P2_reproduction_run_20261106.json`, output `report/evidence/reproduction/`, block 1 of `REPRODUCTION_LOG.txt`. Operator field "mohammed"; the owner states that Mohammed Adeeb Alrashdi ran it on the owner's computer; the Colab record shows the owner's account. **11 Nov 2026:** Mohammed's written agreement received and quoted in block 1 of the log. No personal signature form is an established instructor submission requirement. The saved Colab account alone does not establish the physical operator; operator attribution remains an attribution limitation; the official feedback does not require another classifier-training run. Level 2 is optional and was not run.
- Original issue: `REPRODUCTION_LOG.txt` empty; the only rerun (23 Oct) was by the author.
- Files changed: `tools/reproduce_phase2_eval.py` (new), `tests/test_reproduce_phase2_eval.py` (3 tests), `docs/INDEPENDENT_REPRODUCTION_PROCEDURE.md`, `REPRODUCTION_LOG.txt` (distinguishes the saved evaluation run from independently operated classifier training), report §1.
- Repair: evaluation reproduction script writing original vs reproduced side by side with tolerances fixed in advance (0.005 absolute), operator, commit, dirty state, weight hashes, environment; Level 2 (training rerun) described.
- Verification: 3/3 tests (originals loaded from evidence equal the reported numbers; tolerance verdicts; no invented metrics). The saved Level 1 execution of 6 November is recorded above; no fresh GPU execution occurred in this review.
- The explicit documentation repair is complete: original W&B source run IDs, reproduced execution ID, original/reproduced numbers and the nonempty committed log are linked. This does not assert a new training run or change the awarded grade.

### A3 `[x]` Dated record that the acceptance criteria were fixed before testing (within the stated scope)
- Original issue: the report asserted "written 15 October" without a dated record.
- Files changed: `report/evidence/P2_prespecification_evidence_20261106.json`, report §1 criteria paragraph.
- Repair/verification: Drive metadata read through the API: unexecuted notebook copy with all ten criteria and the final targets, Drive timestamp 18 Oct 2026 11:19 UTC (10,484 B, SHA-256 `aca90fa4…`), before the executed integration runs of 19 Oct 00:47 and 09:18 UTC; original notebook file created 15 Oct 16:55 UTC (content changed later). The report now states exactly this, including that the first detector training run (17 Oct) precedes the earliest dated copy of the criteria text.
- Remaining limitation: no record earlier than 18 Oct carries the criteria text; W&B config and summaries were read publicly on 12 November without a key; creation times were not part of that query. Status `[x]` for "dated record located and reported honestly", not for "prespecification before every run".

### A4 `[~]` Commit hash in every run
- Repair: `src/spv/provenance.py` (`record()`: commit, branch, dirty files or archive index or source fingerprint, environment; `file_sha256`) with 4 tests; `tools/reproduce_phase2_eval.py` uses it. Evidence files of 2 and 5 Nov already carry `source_commit`; the runs of 17–19 Oct do not (historical, cannot be changed).
- Verification: `tests.test_provenance` 4/4; `python src/spv/provenance.py` prints the current commit. Future detector and segmenter training now records source identity through `training_provenance.py` at the actual training entry points. Historical runs are unchanged. New code is tested without performing training.

### A5 `[x]` Per-class recall; class imbalance (improvement recommendations)
- Per-class P/R/F1 at the operating point are in report §2.4 (cat 0.983 / dog 0.948 recall) from `M4_detection_extra_metrics.json`; per-species IoU/Dice in §3.4. `[x]` for reporting.
- Imbalance: no class weights or balanced sampler in the delivered detector (stated in §1 and the dataset card). Controlled comparison run on 6 Nov with the owner's approval (report §4.9, experiment E; `P2_supplementary_E_class_balance_val_20261106.json`): cats oversampled ×2, same recipe, split v1 and seed; validation only — mAP50–95 0.844 → 0.854, F1 0.958 → 0.964, dog recall 0.940 → 0.955. Test split not used; not promoted. `[x]` for the comparison.

### A6 `[!]` Harder independent shelter-style test and calibration set
- Not collected; `docs/INDEPENDENT_TEST_SET_PROTOCOL.md` fixes scope, screening against Oxford and the team set (MD5/pHash/feature similarity), annotation and use before collection. Owner: team; 2–3 weeks. Blocks generalisation claims only.

## B. Explicit graded Phase 2 requirements (Table 5 / Table 6; 45 points)

| ID | Requirement (source) | Evidence (verified this session) | Status |
|---|---|---|---|
| M4-1 | Annotations converted and verified visually (T5) | §2.1 tables/figure; `M4_label_roundtrip.json`; checker 41/41 | `[x]` split v1 |
| M4-2 | Fine-tuned detector, runs with W&B links, selection reason (example §2.2) | §2.2; `2seovilv`; optimiser evidence | `[x]` |
| M4-3 | Threshold sweep on validation, justified operating point (T5) | §2.3; `M4_threshold_sweep.csv` | `[x]` |
| M4-4 | mAP50, mAP50–95, per-class AP at conf 0.001; P/R/F1 at the op (T5) | §2.4; `M4_detection_test_metrics.json`, `_extra_metrics.json` | `[x]` |
| M4-5 | FP breakdown, recall by size, FP/FN galleries, failure case (T5, example §2.5) | §2.4–2.5; size-table erratum evidence | `[x]` |
| M4-6 | Inference on images and video (T5) | §2.6; `M4_video_inference.json` | `[x]` |
| M5-1 | Ground-truth masks and SAM comparison with conditions stated (T5) | §3.1 (trimaps = GT; SAM 2.1-B oracle box = baseline; border effect quantified in §3.4) | `[x]` |
| M5-2 | Fine-tuned segmenter, two baselines, selected run (example §3.2) | §3.2; `3ztpz2tc` | `[x]` |
| M5-3 | Thresholds on validation (T5) | §3.3; `P2_gap_experiments_20261102.json` | `[x]` |
| M5-4 | Mask mAP50–95, mIoU, Dice on test (T5) | §3.4; `M5_*`; Kaggle recomputation | `[x]`; per-class mask AP not recorded (stated) |
| M5-5 | Error analysis, failure case (T5) | §3.5 | `[x]` |
| M8-1 | End-to-end pipeline, per-stage timing (T5) | §4.1 | `[x]` |
| M8-2 | Manifest with hashes, class order, input size, thresholds, versions (T5) | §4.2; `runs/manifest.json`; strict preflight pass | `[x]` |
| M8-3 | Input/output contracts (example §4.3) | §4.3; unit tests | `[x]` |
| M8-4 | ONNX export with parity check (T5) | §4.4: A9a pass; **A9b FAIL 0.0018–0.0023 vs 0.001, bound unchanged, PyTorch served** | `[x]` check done; criterion failed and reported |
| M8-5 | HTTP service (T5) | §4.5; real-weight CPU smoke test; 15 browser tests | `[x]` |
| M8-6 | p50/p95/p99, throughput, memory, stated conditions, warm-up (T5) | §4.6 (T4 and CPU rows never mixed) | `[x]` |
| M8-7 | Verdict vs criteria fixed in advance; for each FAIL why/fix/cost; go/no-go (T5) | §4.7; criteria dating per A3 | `[x]` |
| M8-8 | Failure gallery, model card with out-of-scope uses (T5) | §2.5/§3.5/§4.7; `docs/model_card.md` | `[x]` |
| M8-9 | Live demonstration on a clean runtime (T5) | `/demo` + launcher; run only in a cloud CPU container | `[!]` laptop run and recording open (owner/presenter) |

## C. Report form and consistency (example report; §4 of the guidelines)
| ID | Requirement | Status / evidence |
|---|---|---|
| F-1 | Section 1 field table, team table, points, overview, criteria, Phase 1 data-plan follow-up, **Phase 1 feedback response** | `[x]` present; contributions reported by the team; no additional signature condition |
| F-2 | Sections 2–4 as in the example; §4.8 regression runs; Conclusion | `[x]` |
| F-3 | Obstacles table; submission checklist; self-assessment | `[x]` |
| F-4 | Every figure states split, resolution, thresholds | `[x]` for tables read in this pass; members re-check their sections `[!]` |
| F-5 | Delivered commit stamped in the PDF; links pinned to commits; source ZIP rebuilds | `[x]` (build.sh; recompile log in the package) |
| F-6 | Report, slides, model card, dataset card, checklist and manifests agree | `[~]` report, cards, checklist updated in this pass; the current presentation and its embedded/external notes are synchronized; the final release manifest identifies the version |

## D. Data policy, proposal requirements (§2.4, §3.1)
R1–R10 and D-1/D-2: `[x]` as in the previous checklist (vision core; public annotated data; no data in Git; consent reported for the team photos — "owner consent reported by the team; records kept privately"; dataset not from the labs; Phase 1 complete; detector/segmenter with functions; pipeline/ONNX/HTTP/benchmark/model card; users and criteria; feasibility; four workstreams). Privacy: current code archives exclude `private_ids.tex` and submitted PDFs. The instructor ZIP/PDF retain the required team identifiers privately. Historical Git objects were not rewritten; no blanket full-history privacy certification is claimed.

## E. Bonus conditions (Table 6; +10 max) — `[~]` team-stated, not verified
- B1 original topic approved before work began: approval date 19 Oct stated by the team; the approval message is not in the delivered material and was not located in this session; the first training run (17 Oct) precedes 19 Oct. Owner: Yasser (locate the message; explain or correct the dates).
- B2 team-collected set processed with the M1 protocol, documented in the dataset card, used in evaluation: documented and used (`[x]` for documentation); the +7 is the instructor's decision.

## F. Presentation (Table 6, 15 points; Table 7)
| ID | Requirement | Status / owner |
|---|---|---|
| P1 | Failure gallery, go/no-go, model card out-of-scope | `[x]` in documents; feedback response present in the current presentation |
| P2 | 15–20 min, Table 7 structure, timed rehearsal | `[!]` no rehearsal yet (owner decision: after review) |
| P3 | Every member presents their own work | `[!]` human |
| P4/P5 | Live demo on the presentation laptop; recorded fallback | `[!]` not done |

## G. Software verification (this pass)
| Check | Result |
|---|---|
| Unit tests, 160 collected (`PYTHONPATH=src python -m unittest discover -s tests`) | Measured 6 Nov on the delivered revision; each environment is reported on its own and the counts are never merged: (a) Git checkout, Python 3.13, PyTorch, Playwright and Chromium: **160 run, 0 skipped, OK**; (b) clean extraction of `reviewed_source.zip` (not a Git checkout), same interpreter: **159 run + 1 skipped** (`test_provenance` needs a Git checkout), OK; (c) clean extraction in the pinned virtual environment without Playwright: **144 run + 16 skipped** (15 browser tests + the Git test), OK. CI (`phase2-tests.yml`) runs in a Git checkout with `requirements.txt` and no Chromium, so its count is 145 run + 15 skipped. Logs: `03_Verification/logs/05_*` (c) and `10_*` (b). |
| Evidence checker | 49/49 |
| ruff (CI selection) | clean |
| Notebook static check | 15/15 parse; no notebook executed in this environment (needs Drive/Oxford/team photos/GPU/W&B) — `docs/NOTEBOOK_EXECUTION_PLAN.md` |
| Real-weight HTTP smoke test | run on the delivered revision (log in the package) |

Delivered archives (two different files; each name is tied to its own size and SHA-256 in the package `SHA256SUMS` and `RELEASE_MANIFEST.json`):

| File | What it is | Size and hash |
|---|---|---|
| `01_Report/SmartPetVision_Phase2_source.zip` | report source for the instructor: `main.tex`, figures, `delivery_commit.tex`, plus the code under `code/` (PDFs excluded) | size and SHA-256 in `SHA256SUMS`; the hash is re-verified on a copy **downloaded from Drive** after every upload (`00_RELEASE/DRIVE_READBACK.json`, field `sha256_verified`) — a size match alone is not accepted |
| `02_Code/reviewed_source.zip` | code archive only, with `CODE_ARCHIVE_INDEX.json` (tree hash and delivered commit) | size and SHA-256 in `SHA256SUMS`; verified after download in the same readback |
| `01_Report/SmartPetVision_Phase2.pdf` | the report; page 1 prints the delivered commit | size and SHA-256 in `SHA256SUMS`; verified after download |

## H. Submission mechanics
| ID | Item | Status / owner |
|---|---|---|
| S1 | Upload PDF + source ZIP by 15 Nov | `[!]` one member |
| S2 | Public repository | `[x]` (`Yasser-20-alhajlah/smart-pet-vision`) |
| S3 | GitHub reference version | `[x]` the delivery is **the commit printed on page 1 of the PDF** (also in `CODE_ARCHIVE_INDEX.json`), not `main`. State on 6 Nov: PR #8 (head `81cd4fd`) was merged into `main` by the owner at 01:14 UTC (merge commit `bcc8d13`, CI green); the later delivered commits (`f5bad5b` to `f44fd4c`) went through PR #9 (CI green on `f44fd4c`, run 42247384173), which the owner merged into `main` on 6 Nov at 18:04 UTC (merge commit `a35be4c`; CI on `main` green, run 42267368025). PR #9 is closed. The delivery is still the commit `f44fd4c`, not `main`. |
| S4 | Human read-through, no contradictions | automated checks `[x]`; team-members table: the owner stated on 6 Nov that the four members approved the table as printed (no individual signature file exists in the package) `[~]`; full read-through of the delivered report `[!]` all four |
| S5 | W&B key revoked | `[!]` owner; not verifiable from the repository |
| S6 | Drive package files replaced in place with readback | `[x]` `00_RELEASE/DRIVE_READBACK.json`: every file's size, and the SHA-256 of the PDF, the source ZIP and the code archive computed on copies downloaded from Drive (the 9.25 MB source ZIP via the owner's computer, 6 Nov) |

## J. Known issues that stay open and visible (not closed by any later report)
| Issue | Where it is reported | Status |
|---|---|---|
| Four loose training boxes (M3 audit) | report Section 1, "Follow-up on the Phase 1 data-improvement plan", row "Re-derive the four loose body boxes": 4 train labels corrected on 3 Nov in a separate label set; the served detector was **not** retrained on them; the candidate trained on the corrected labels gave no validation gain and was not promoted (Section 4.8, "Regression runs of 5 November") | `[~]` corrected labels exist; served metrics do not reflect them |
| 17 of 52 multi-animal team photos marked `ready` | report summary (page 1), Section 1 "Results at a glance", Section 2.6 (team photos) and Section 4.7 "Verdict against the acceptance criteria"; evidence `P2_cpu_release_validation_f8manl0i` | `[!]` open limitation; staff approval remains mandatory Supplementary run of 6 Nov (report §4.9, experiment D): the opt-in second detection pass gives 10 of 52 at the cost of 82 more single-animal referrals (45 → 127 of 349); post-hoc on the same photos, review stays off. |
| A9b detector ONNX parity 0.0018–0.0023 vs bound 0.001 | report Section 1 (criteria table), Section 4.4 "ONNX export and parity", Section 4.7 verdict table; `M8_onnx_parity_recheck_A9b_20261105.json` | `[!]` **not met**; bound unchanged; PyTorch served |
| Same-animal grouping tool: tests prove the algorithm on a toy set; on the images it ran twice with identical results (6 Nov, `bcc5337` and `609c0c5`) | A1 above; `docs/LEAKAGE_SAME_ANIMAL_PROTOCOL.md` (review band, transitive links to be audited) | `[x]` grouping implemented for v2; certified identities are not an added requirement |

## I. Current next actions and scope
1. Presenters: test the real service on the presentation computer, record fallback video and hold a timed rehearsal; every member explains their own work. These are future team actions, not executions claimed by this review.
2. One member: submit the two report files through Blackboard within the official deadline. No Blackboard upload was performed here.
3. Account owner: identify and revoke the exposed W&B key. This security action is unverified and separate from the official report requirements.
4. Retain the disclosed ONNX A9b failure, multi-animal limitations, split-v1 leakage and missing historical SAM settings. Further research is distinct from correcting documentation.
5. The original Phase 1 and Phase 2 templates and original 38/40 assessment were opened; their source hashes and literal requirement mapping are in OFFICIAL_REQUIREMENTS_RECONCILIATION_20261112.md. PCB examples remain illustrative.

## K. Verification scope
- Prior published f30264d baseline: local Phase 2 165 passed; CI 150 passed and 15 browser skips. These are separate executions.
- This official-source revision adds five focused bootstrap/provenance regressions, all passed locally without training. The current full CI result is recorded in the release metadata; do not infer a GPU accuracy run from unit tests.
- Saved evidence checks pass 49/49; all 15 selected notebooks parse. Historical saved outputs are preserved.
- Public W&B read returned 44 runs; 84 shared numeric summary fields of the three selected runs matched the saved snapshot. No API key, new run or historical edit was used.
- The real-weight CPU functional receipt and all T4 measurements retain their original environments and source identities; actual presentation-host testing remains the presenters' task.
