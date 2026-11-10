#!/usr/bin/env bash
# Downloads what the benchmarks and comparisons need, beyond the training data
# (fetch_training_data.sh), into $NW_DATA/raw (about 22 GB), each checked against an md5 or a
# pinned commit. None of it is trained on.
#
#   OLiMPiC    IMSLP scans of OpenScore Lieder songs, as whole pages, with their system
#              locations (CC BY-SA 4.0); the authors' code (MIT: TEDn metric, linearization)
#              and their trained Zeus model (CC BY-SA 4.0), to compare against
#   Mutopia    piano pieces engraved by LilyPond (CC or public domain), with sources, PDFs and
#              MIDI (eval/mutopia_pieces.txt)
#   Transcoda  its authors' trained model (CC BY 4.0), to compare against, their metric's
#              code, and their Polish benchmark (evaluation only)
#   Copista    copisteria's trained weights (Apache-2.0), to compare against (the code is
#              checked out by setup_compare_env.sh)
#   Audiveris  5.11.0 (AGPL-3.0), to compare against
#   MusiCorpus historical manuscripts with MusicXML transcriptions (CC BY-NC-SA 4.0); its test
#              split is a sealed benchmark (scores only)
#
# A source already in place is not fetched again. Archives are deleted once unpacked.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OLIMPIC_RELEASE=https://github.com/ufal/olimpic-icdar24/releases/download/datasets
OLIMPIC_REPO=https://github.com/ufal/olimpic-icdar24.git
OLIMPIC_COMMIT=f4f45ac523ebe09832342270a68b1b4368f63da2
ZEUS_MODEL=https://github.com/ufal/olimpic-icdar24/releases/download/zeus-release/zeus-olimpic-1.0-2024-02-12.model.tar.gz
MUTOPIA_REPO=https://github.com/MutopiaProject/MutopiaProject.git
MUTOPIA_COMMIT=2144afd6f52d56c5b6995b8b589ef1268b3139f0
MUTOPIA_FILES=https://www.mutopiaproject.org/ftp
TRANSCODA_REVISION=b529f8aa5d996d9224df3395b5b92d0867343c91
TRANSCODA_MODEL=https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1/resolve/$TRANSCODA_REVISION
TRANSCODA_REPO=https://github.com/btrkeks/Transcoda.git
TRANSCODA_COMMIT=d4e2e687d5679ae96ca4aa6f01e06a5b338cd488
POLISH_SCORES=https://huggingface.co/datasets/btrkeks/polish-scores/resolve/55428225335e21c0e4d77a58b32dbb2b58e25954
COPISTA_MODEL=https://huggingface.co/kobimusic/copista-28m/resolve/44464d11234a22ba23634ec8eb1e31e3b012021a
AUDIVERIS_DEB=https://github.com/Audiveris/audiveris/releases/download/5.11.0/Audiveris-5.11.0-ubuntu24.04-x86_64.deb
MUSICORPUS=https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/fdf9ba56-1854-45fc-a8d2-311dfb3ad8c3/content

if [ ! -d "$RAW/olimpic/sources/imslp_systems" ]; then
  download "$OLIMPIC_RELEASE/olimpic-1.0-sources-for-scanned.2024-02-12.tar.gz.tgz" "$RAW/olimpic/sources.tgz" 2d2e7d3a4e05714b6e36a5b25a3dc801
  unpack "$RAW/olimpic/sources.tgz" "$RAW/olimpic/sources" corpus_to_imslp imslp_pdfs imslp_pngs imslp_systems
  rm "$RAW/olimpic/sources.tgz"
fi
checkout "$OLIMPIC_REPO" "$OLIMPIC_COMMIT" "$RAW/olimpic/code/repo"
if [ ! -d "$RAW/olimpic/code/zeus-olimpic-1.0-2024-02-12.model" ]; then
  download "$ZEUS_MODEL" "$RAW/olimpic/code/zeus.tgz" c6dfa1175a740c145015073b481bdf51
  unpack "$RAW/olimpic/code/zeus.tgz" "$RAW/olimpic/code" zeus-olimpic-1.0-2024-02-12.model
  rm "$RAW/olimpic/code/zeus.tgz"
fi

while read -r file md5; do
  download "$TRANSCODA_MODEL/$file" "$RAW/transcoda/b529f8a/$file" "$md5"
done <<'FILES'
  config.json 0618595302f62804e5a91d990f1641d3
  configuration_transcoda.py c7b02494bb76a12a5e1ac1b52502448a
  encoder.py eaac77672a86c5884aa23f3fa7a3793b
  frontends.py 3036d17150816df7fa9da84a456f59da
  generation_config.json 19e59b52c8844dcee524c3eb4ba7a7b2
  model.safetensors c501a672d8a0afa1f4420a1152aea835
  modeling_transcoda.py 9f5666b88d6c614c3a6328ab6d47907c
  special_tokens_map.json 10c2fbdc1ca5ac8f0312bebcfb4a3e74
  tokenizer.json 3de8e02236440f73774af2466e892cda
  tokenizer_config.json 138bf24fefffb08643d3bb5d52c72f79
  vision_frontend.py 58d977b416979c208f209f15eec4e067
FILES

checkout "$TRANSCODA_REPO" "$TRANSCODA_COMMIT" "$RAW/transcoda_repo/src"
download "$POLISH_SCORES/data/train-00000-of-00001.parquet" "$RAW/polish_scores/polish.parquet" 8d6f6a9c3ab07e16018d4b7477143a20
download "$COPISTA_MODEL/evidence-2m.pt" "$RAW/copisteria/src/models/evidence-2m.pt" 34e71c035eaf7f4f423806046da38009
download "$COPISTA_MODEL/small/v7_obj-recall-30m_best.pt" "$RAW/copisteria/src/models/small/v7_obj-recall-30m_best.pt" \
  ab5d420ac94c5cddd7b392a4dd93281f
download "$AUDIVERIS_DEB" "$RAW/audiveris/audiveris.deb" 392b146938867806fcc4bdfe8af13ddf
[ -d "$RAW/audiveris/unpacked" ] || dpkg -x "$RAW/audiveris/audiveris.deb" "$RAW/audiveris/unpacked"

checkout "$MUTOPIA_REPO" "$MUTOPIA_COMMIT" "$RAW/mutopia/repo"
while read -r _ piece pdf_md5 midi_md5; do
  name=$(basename "$piece")
  download "$MUTOPIA_FILES/$piece/$name-a4.pdf" "$RAW/mutopia/pieces/$name-a4.pdf" "$pdf_md5"
  download "$MUTOPIA_FILES/$piece/$name.mid" "$RAW/mutopia/pieces/$name.mid" "$midi_md5"
done < "$ROOT/eval/mutopia_pieces.txt"

if [ ! -d "$RAW/musicorpus/musicorpus_v1.0" ]; then
  download "$MUSICORPUS" "$RAW/musicorpus/musicorpus_v1.0.zip" 413a500ec1b61184fbb1938aa1244117
  unpack "$RAW/musicorpus/musicorpus_v1.0.zip" "$RAW/musicorpus" musicorpus_v1.0
  rm "$RAW/musicorpus/musicorpus_v1.0.zip"
fi

log "benchmark data ready"
