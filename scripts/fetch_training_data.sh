#!/usr/bin/env bash
# Downloads what training needs into $NW_DATA/raw (about 30 GB of downloads), each checked
# against an md5 or a pinned commit:
#
#   PDMX       public-domain MusicXML from MuseScore.com (Zenodo record 15571083)
#   ASAP       classical piano scores (MusicXML), pinned commit
#   MUSCIMA++  handwritten pages with symbol masks (CC BY-NC-SA 4.0), plus the
#              CVC-MUSCIMA images they annotate
#   OLiMPiC    synthetic systems (CC BY-SA 4.0) for training the sequence model; scanned
#              systems (CC BY-SA 4.0), whose dev split monitors training, and the Lieder
#              corpus metadata (CC0), to leave out dev songs our sources also hold; the
#              authors' code (MIT), whose linearization turns MusicXML into our labels and back
#   GrandStaff piano excerpts (images) with GrandStaff-LMX transcriptions (CC BY-SA 4.0),
#              for training the sequence model
#   MusiCorpus historical manuscripts, mostly handwritten, with MusicXML transcriptions (CC BY-NC-SA
#              4.0): its piano grand staves train the sequence model; its test split is a sealed
#              benchmark (scores only)
#   Smashcima  assets for our fork's handwritten crops: MUSCIMA++ 1.0 glyphs (CC BY-NC-SA
#              4.0) and paper from the Moravian Library's scans, checked against checksums
#   MuseScore  3.6.2 (GPL), the engraver of the MuseScore training crops, into tools/musescore
#
# A source already in place is not fetched again. Archives are deleted once unpacked.
set -euo pipefail
source "$(dirname "$0")/common.sh"

PDMX_RECORD=https://zenodo.org/api/records/15571083/files
ASAP_REPO=https://github.com/fosfrancesco/asap-dataset.git
ASAP_COMMIT=afc815c75c42e83a79c03feb6da8a35e77d4c6b8
MUSCIMA_PP=https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip
CVC_MUSCIMA=https://github.com/apacha/OMR-Datasets/releases/download/datasets/CVCMUSCIMA_SR.zip
OLIMPIC_RELEASE=https://github.com/ufal/olimpic-icdar24/releases/download/datasets
LIEDER_SCORES=https://raw.githubusercontent.com/OpenScore/Lieder/38c5db510224d9facdc4b08d741fc788cfb58ea8/data/scores.yaml
OLIMPIC_REPO=https://github.com/ufal/olimpic-icdar24.git
OLIMPIC_COMMIT=f4f45ac523ebe09832342270a68b1b4368f63da2
GRANDSTAFF_IMAGES=https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz
MUSICORPUS=https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/fdf9ba56-1854-45fc-a8d2-311dfb3ad8c3/content
MUSESCORE=https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021370-x86_64.AppImage

download "$PDMX_RECORD/PDMX.csv/content" "$RAW/pdmx/PDMX.csv" 30392ccf38bb63ce70e7afae70f9c88c
if [ ! -d "$RAW/pdmx/mxl" ]; then
  download "$PDMX_RECORD/mxl.tar.gz/content" "$RAW/pdmx/mxl.tar.gz" 49ffd75ecf5489c0be6d41182eb11ff7
  unpack "$RAW/pdmx/mxl.tar.gz" "$RAW/pdmx" mxl
  rm "$RAW/pdmx/mxl.tar.gz"
fi

checkout "$ASAP_REPO" "$ASAP_COMMIT" "$RAW/asap"

if [ ! -d "$RAW/muscima/v2.0" ]; then
  download "$MUSCIMA_PP" "$RAW/muscima/MUSCIMA-pp_v2.0.zip" e13c13f0719feb1ed8f3419c1960bd9e
  unpack "$RAW/muscima/MUSCIMA-pp_v2.0.zip" "$RAW/muscima" v2.0
  rm "$RAW/muscima/MUSCIMA-pp_v2.0.zip"
fi
if [ ! -d "$RAW/muscima/CvcMuscima-Distortions" ]; then
  download "$CVC_MUSCIMA" "$RAW/muscima/CVCMUSCIMA_SR.zip" 9e3c9b9d899a68be00a1de3e344ebb64
  unpack "$RAW/muscima/CVCMUSCIMA_SR.zip" "$RAW/muscima" CvcMuscima-Distortions
  rm "$RAW/muscima/CVCMUSCIMA_SR.zip"
fi
if [ ! -d "$RAW/olimpic/olimpic-1.0-scanned" ]; then
  download "$OLIMPIC_RELEASE/olimpic-1.0-scanned.2024-02-12.tar.gz" "$RAW/olimpic/scanned.tar.gz" cca3555c8a469fc7d9d349fb66040a40
  unpack "$RAW/olimpic/scanned.tar.gz" "$RAW/olimpic" olimpic-1.0-scanned
  rm "$RAW/olimpic/scanned.tar.gz"
fi
download "$LIEDER_SCORES" "$RAW/olimpic/lieder_scores.yaml" f9b27916cb7d3ebd5c8d5af50a6ad7d6
checkout "$OLIMPIC_REPO" "$OLIMPIC_COMMIT" "$RAW/olimpic/code/repo"
if [ ! -d "$RAW/olimpic/olimpic-1.0-synthetic" ]; then
  download "$OLIMPIC_RELEASE/olimpic-1.0-synthetic.2024-02-12.tar.gz" "$RAW/olimpic/synthetic.tar.gz" efaeb053fc85a10a74316d32aab13e22
  unpack "$RAW/olimpic/synthetic.tar.gz" "$RAW/olimpic" olimpic-1.0-synthetic
  rm "$RAW/olimpic/synthetic.tar.gz"
fi
# GrandStaff-LMX holds the transcriptions; its images come from the original GrandStaff
# release and are extracted into the same directories.
if [ ! -d "$RAW/olimpic/grandstaff-lmx" ]; then
  download "$OLIMPIC_RELEASE/grandstaff-lmx.2024-02-12.tar.gz" "$RAW/olimpic/grandstaff-lmx.tar.gz" 9aca0658471b4f9a328de5766c980744
  download "$GRANDSTAFF_IMAGES" "$RAW/olimpic/grandstaff.tgz" f8245becbf534edc7afe241ac244602b
  rm -rf "$RAW/olimpic/.grandstaff" && mkdir -p "$RAW/olimpic/.grandstaff"
  tar xzf "$RAW/olimpic/grandstaff-lmx.tar.gz" -C "$RAW/olimpic/.grandstaff"
  tar xzf "$RAW/olimpic/grandstaff.tgz" -C "$RAW/olimpic/.grandstaff/grandstaff-lmx" 2>/dev/null
  mv "$RAW/olimpic/.grandstaff/grandstaff-lmx" "$RAW/olimpic/grandstaff-lmx"
  rm -rf "$RAW/olimpic/.grandstaff" "$RAW/olimpic/grandstaff-lmx.tar.gz" "$RAW/olimpic/grandstaff.tgz"
fi

if [ ! -d "$RAW/musicorpus/musicorpus_v1.0" ]; then
  download "$MUSICORPUS" "$RAW/musicorpus/musicorpus_v1.0.zip" 413a500ec1b61184fbb1938aa1244117
  unpack "$RAW/musicorpus/musicorpus_v1.0.zip" "$RAW/musicorpus" musicorpus_v1.0
  rm "$RAW/musicorpus/musicorpus_v1.0.zip"
fi

(cd "$ROOT/train" && "$SMASHCIMA_PY" install_smashcima_assets.py)

musescore="$ROOT/tools/musescore"
if [ ! -d "$musescore/squashfs-root" ]; then
  download "$MUSESCORE" "$musescore/musescore.AppImage" 8a7c100a19fd78e1c43fccd5c1db27a6
  chmod +x "$musescore/musescore.AppImage"
  rm -rf "$musescore/.extract" && mkdir "$musescore/.extract"
  (cd "$musescore/.extract" && ../musescore.AppImage --appimage-extract >/dev/null)
  mv "$musescore/.extract/squashfs-root" "$musescore/squashfs-root" && rmdir "$musescore/.extract"
fi

log "training data ready"
