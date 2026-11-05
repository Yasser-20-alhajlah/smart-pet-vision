#!/usr/bin/env bash
# Build SmartPetVision_Phase2.pdf, the source ZIP, reviewed_source.zip and RELEASE_MANIFEST.json. Usage: ./build.sh [extra files to fingerprint]
set -euo pipefail
cd "$(dirname "$0")"
# Date the PDF by the commit, not the clock, so rebuilding the same commit gives the same PDF bytes.
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" FORCE_SOURCE_DATE=1
[ -f private_ids.tex ] || echo "note: private_ids.tex not found; student IDs will print as (ID)"
# Stamp the exact delivery commit into the PDF (generated file, ignored by Git; refuses a dirty phase2 tree).
if [ -n "$(git status --porcelain -- ..)" ]; then echo "error: uncommitted changes under phase2/; commit first so the PDF names the delivered commit" >&2; exit 1; fi
printf '\\newcommand{\\deliverycommit}{%s}\n\\newcommand{\\deliverybranch}{%s}\n' "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" > delivery_commit.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
cp main.pdf SmartPetVision_Phase2.pdf
rm -f SmartPetVision_Phase2_source.zip reviewed_source.zip
# Code archive of the committed revision (no PDFs; refuses if a private ID is found), then the
# report source ZIP = main.tex + figures + private_ids.tex + the same code members under code/.
python3 tools/make_code_archive.py reviewed_source.zip
python3 tools/make_code_archive.py --source-zip SmartPetVision_Phase2_source.zip reviewed_source.zip \
  main.tex figures/* $( [ -f private_ids.tex ] && echo private_ids.tex )
python3 tools/make_release_manifest.py "$@"   # optional: extra delivered files to fingerprint (e.g. the deck)
echo "built SmartPetVision_Phase2.pdf, SmartPetVision_Phase2_source.zip, reviewed_source.zip and RELEASE_MANIFEST.json"
