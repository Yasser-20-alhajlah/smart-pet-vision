# Phase 1 report source

The public `SmartPetVision_Phase1.pdf` is a historical copy with student IDs redacted. The current instructor PDF and matching editable LaTeX source are retained privately in the [Final Phase 1 folder](https://drive.google.com/drive/folders/1M7frAxmWt-eJO2p2FxHP5Yc3Ejn8J4FO). Use the canonical `SmartPetVision_Phase1.pdf` and `SmartPetVision_Phase1_source.zip` identified by that folder's current README.

Extract the source ZIP into a new directory and compile root `main.tex` with pdfLaTeX or Tectonic. The consolidated source retains historical benchmark results and separately labels the 5 November experiments and their checkpoint-availability limit. The current software revision is recorded in the report and its code archive; this does not recover the unknown historical training commit.

`tools/build_report.py` is a tracked legacy PDF-overlay script. Its historical `tools/assets/base_report.pdf` input is absent from this checkout. It is not the LaTeX builder for the current report and is not needed when building the supplied source ZIP.

Phase 2 has a separate LaTeX project under `phase2/submission/`. Private student identifiers belong only in instructor-facing materials, never in public source commits.
