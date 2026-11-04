#!/usr/bin/env bash
# Build SmartPetVision_Phase2.pdf, the source ZIP and RELEASE_MANIFEST.json. Usage: ./build.sh [extra files to fingerprint]
set -euo pipefail
cd "$(dirname "$0")"
[ -f private_ids.tex ] || echo "note: private_ids.tex not found; student IDs will print as (ID)"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
cp main.pdf SmartPetVision_Phase2.pdf
rm -f SmartPetVision_Phase2_source.zip
zip -q SmartPetVision_Phase2_source.zip main.tex figures/* $( [ -f private_ids.tex ] && echo private_ids.tex )
python3 tools/make_release_manifest.py "$@"   # optional: extra delivered files to fingerprint (e.g. the deck)
echo "built SmartPetVision_Phase2.pdf, SmartPetVision_Phase2_source.zip and RELEASE_MANIFEST.json"
