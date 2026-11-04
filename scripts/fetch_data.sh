#!/usr/bin/env bash
# Downloads the raw training sources into $NW_DATA/raw (about 6 GB of downloads).
#
#   PDMX       public-domain MusicXML from MuseScore.com (Zenodo record 15571083)
#   ASAP       classical piano scores (MusicXML), pinned commit
#   MUSCIMA++  handwritten pages with symbol masks (CC BY-NC-SA 4.0), plus the
#              CVC-MUSCIMA images they annotate
set -euo pipefail
source "$(dirname "$0")/common.sh"

PDMX_RECORD=https://zenodo.org/api/records/15571083/files
ASAP_REPO=https://github.com/fosfrancesco/asap-dataset.git
ASAP_COMMIT=afc815c75c42e83a79c03feb6da8a35e77d4c6b8
MUSCIMA_PP=https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip
CVC_MUSCIMA=https://github.com/apacha/OMR-Datasets/releases/download/datasets/CVCMUSCIMA_SR.zip

fetch_pdmx() {
  download "$PDMX_RECORD/PDMX.csv/content" "$RAW/pdmx/PDMX.csv" 30392ccf38bb63ce70e7afae70f9c88c
  download "$PDMX_RECORD/mxl.tar.gz/content" "$RAW/pdmx/mxl.tar.gz" 49ffd75ecf5489c0be6d41182eb11ff7
  tar xzf "$RAW/pdmx/mxl.tar.gz" -C "$RAW/pdmx"
  rm "$RAW/pdmx/mxl.tar.gz"
}

fetch_asap() {
  rm -rf "$RAW/asap"
  git init -q "$RAW/asap"
  git -C "$RAW/asap" fetch -q --depth 1 "$ASAP_REPO" "$ASAP_COMMIT"
  git -C "$RAW/asap" checkout -q FETCH_HEAD
}

fetch_muscima() {
  download "$MUSCIMA_PP" "$RAW/muscima/MUSCIMA-pp_v2.0.zip" e13c13f0719feb1ed8f3419c1960bd9e
  download "$CVC_MUSCIMA" "$RAW/muscima/CVCMUSCIMA_SR.zip" 9e3c9b9d899a68be00a1de3e344ebb64
  unzip -q -o "$RAW/muscima/MUSCIMA-pp_v2.0.zip" -d "$RAW/muscima"
  unzip -q -o "$RAW/muscima/CVCMUSCIMA_SR.zip" -d "$RAW/muscima"
}

step fetch-pdmx fetch_pdmx
step fetch-asap fetch_asap
step fetch-muscima fetch_muscima
