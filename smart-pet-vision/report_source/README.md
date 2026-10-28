# Rebuild the current corrected report

This is a reproducible PDF revision source, not a recovered original 16-page LaTeX project. The preserved base PDF plus editable replacement paragraphs/rectangles in build_report.py regenerate the current corrected report, including permanent display-only face redaction and links. The older 8-page Overleaf ZIP is historical.

Install requirements.txt in a separate environment, then run python report_source/build_report.py. Output: report_source/SmartPetVision_Phase1.pdf; edit record: pdf_changes.json. Compare page text and rendered pixels, since PDF document IDs can differ bytewise.

The output is not automatically promoted to submission: visually inspect before replacing the root final PDF. No metric, model weight or image used for evaluation is changed. Full semantic editing of the original layout still needs its original authoring source.
