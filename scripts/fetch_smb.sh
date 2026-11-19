#!/usr/bin/env bash
# Downloads SMB, PRAIG's Sheet Music Benchmark (CC BY-NC 4.0), into $NW_DATA/raw/smb, pinned and
# checked. It is gated on Hugging Face: access is granted to a person's own account, so this
# cannot be done for them. Without it, the script says how to get it and goes on only with the
# user's permission: a yes at the prompt, or NW_WITHOUT_SMB=1 where no one is there to answer
# (reproduce.sh runs this first, before a day of training, for that reason). Only the pages in
# eval/smb_pages.txt are scored (docs/EVALUATION_PROTOCOL.md).
set -euo pipefail
source "$(dirname "$0")/common.sh"

SMB_REVISION=1b75cc87bb47791c647f4bd1e7a27016246bc1dd
SMB_MD5=efe7b4d292c4e1a6756646cf1cbb8aa0
smb="$RAW/smb/data/test-00000-of-00001.parquet"

directions() {
  cat >&2 <<EOF

SMB could not be fetched: $1
It is gated on Hugging Face, so it needs your own account. To include it:
  1. Request access at https://huggingface.co/datasets/PRAIG/SMB (approved by hand; its card
     says to email the author if no answer comes)
  2. Install the Hugging Face CLI:  curl -LsSf https://hf.co/cli/install.sh | bash
  3. Log in with a read token:      hf auth login
  4. Run this again.
Everything else works without it; only the SMB benchmark row is left out.
EOF
}

# Permission to go on without SMB: NW_WITHOUT_SMB=1, or a yes from someone at the terminal.
without_smb_permitted() {
  [ "${NW_WITHOUT_SMB:-0}" = 1 ] && return 0
  if [ -t 0 ]; then
    local answer
    read -r -p "Go on without SMB? [y/N] " answer
    [[ $answer == [yY]* ]]
  else
    echo "No terminal to ask: set NW_WITHOUT_SMB=1 to go on without SMB." >&2
    return 1
  fi
}

fetch() {
  command -v hf > /dev/null || { directions "the Hugging Face CLI (hf) is not installed"; return 1; }
  hf auth whoami > /dev/null 2>&1 || { directions "not logged in to Hugging Face (hf auth login)"; return 1; }
  log "downloading SMB"
  hf download PRAIG/SMB data/test-00000-of-00001.parquet --repo-type dataset --revision "$SMB_REVISION" \
    --local-dir "$RAW/smb" > /dev/null || { directions "the download was refused; has access been granted to this account?"; return 1; }
}

if [ ! -f "$smb" ] && ! fetch; then
  if without_smb_permitted; then
    log "going on without SMB"
    exit 0
  fi
  exit 1
fi
echo "$SMB_MD5  $smb" | md5sum -c --quiet -
