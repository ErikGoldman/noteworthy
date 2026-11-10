# Shared settings and helpers for the pipeline stages; sourced, not run.
#
# Every setting can be overridden from the environment, e.g. NW_PAGES=200 for a quick try.
# NW_DATA and NW_RUNS move the (large) data and model directories elsewhere.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
ZEUS_PY="$ROOT/.venv-zeus/bin/python"
TRANSCODA_PY="$ROOT/.venv-transcoda/bin/python"
MUSICDIFF_PY="$ROOT/.venv-musicdiff/bin/python"
HOMR_PY="$ROOT/.venv-homr/bin/python"
COPISTERIA_PY="$ROOT/.venv-copisteria/bin/python"
SMASHCIMA_PY="$ROOT/.venv-smashcima/bin/python"
NW_DATA=${NW_DATA:-$ROOT/data}
NW_RUNS=${NW_RUNS:-$ROOT/runs}
NW_PAGES=${NW_PAGES:-12000}                 # synthetic training pages
NW_BASE_STEPS=${NW_BASE_STEPS:-60000}       # training from scratch
NW_FINETUNE_STEPS=${NW_FINETUNE_STEPS:-20000}
NW_EVAL_EVERY=${NW_EVAL_EVERY:-5000}
NW_SEQ_STEPS=${NW_SEQ_STEPS:-192000}       # sequence model (train_seq.sh)
NW_RENDER_WORKERS=${NW_RENDER_WORKERS:-$(nproc)}
NW_TRAIN_WORKERS=${NW_TRAIN_WORKERS:-$(( $(nproc) < 10 ? $(nproc) : 10 ))}
NW_SMASHCIMA_WORKERS=${NW_SMASHCIMA_WORKERS:-6}  # each holds MUSCIMA++'s glyphs in memory (about 2 GB)
NW_BENCH_COUNT=${NW_BENCH_COUNT:-100}

RAW="$NW_DATA/raw"
NW_SEQ_RUN=${NW_SEQ_RUN:-sequence}         # the sequence model's run directory under NW_RUNS
# What recognize.sh uses and the benchmarks score: the segmentation network, and the sequence
# model's averaged weights at the end of its fixed schedule (best.pt is chosen on real scans).
SEGMENTER="$NW_RUNS/finetune/model.onnx"
SEQ_CHECKPOINT="$NW_RUNS/$NW_SEQ_RUN/average_$(printf '%06d' "$NW_SEQ_STEPS").pt"
export MC_ASSETS_CACHE="$RAW/smashcima"   # where Smashcima keeps its glyphs and paper

log() { echo "[$(date +%H:%M:%S)] $*"; }

# The scores every crop builder draws from; a sample's index is its line in this list.
make_score_list() {
  [ -f "$NW_DATA/render_scores.txt" ] && return
  (cd "$ROOT/train" && "$PY" list_scores.py "$RAW/pdmx" "$RAW/asap" "$NW_DATA")
}

# Runs a command that opens windows (MuseScore) on the display, or on a virtual one without.
with_display() {
  if [ -n "${DISPLAY:-}" ]; then "$@"; else xvfb-run -a "$@"; fi
}

# Each stage checks for its own outputs and skips work already done, so the pipeline can
# be rerun at any point without downloading or recomputing finished parts. Outputs are
# written under a temporary name and moved into place, so an interrupted step never
# looks finished.

# Downloads URL to DEST unless DEST already has the right md5.
download() {
  local url=$1 dest=$2 md5=$3
  if [ -e "$dest" ] && echo "$md5  $dest" | md5sum -c --quiet - 2>/dev/null; then
    return
  fi
  mkdir -p "$(dirname "$dest")"
  log "downloading $url"
  # A dropped connection resumes where it stopped (-C -); plain --retry gives up on those.
  curl -fL --retry 5 --retry-all-errors -C - -o "$dest.partial" "$url"
  echo "$md5  $dest.partial" | md5sum -c --quiet - || { echo "checksum mismatch for $url" >&2; exit 1; }
  mv "$dest.partial" "$dest"
}

# Unpacks ARCHIVE (zip or tar.gz) and moves the listed top-level entries into DEST_DIR,
# in order, so the last entry's presence means all are in place.
unpack() {
  local archive=$1 dest=$2
  shift 2
  local tmp="$dest/.unpack"
  rm -rf "$tmp" && mkdir -p "$tmp"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$tmp" ;;
    *) tar xzf "$archive" -C "$tmp" ;;
  esac
  for entry in "$@"; do rm -rf "${dest:?}/$entry" && mv "$tmp/$entry" "$dest/$entry"; done
  rm -rf "$tmp"
}

# Clones REPO at COMMIT into DEST, unless DEST is already at that commit.
checkout() {
  local repo=$1 commit=$2 dest=$3
  [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$commit" ] && return
  log "fetching $repo"
  rm -rf "$dest.partial"
  git init -q "$dest.partial"
  git -C "$dest.partial" fetch -q --depth 1 "$repo" "$commit"
  git -C "$dest.partial" checkout -q FETCH_HEAD
  rm -rf "$dest" && mv "$dest.partial" "$dest"
}
