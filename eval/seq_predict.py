"""The system-crop sequence model (train/seq) on every OLiMPiC scanned crop of a split.

usage: seq_predict.py CHECKPOINT --split dev|test --out DIR [--batch N]

Writes DIR/samples/<song>/<system>.lmx and its delinearized .musicxml, the layout
zeus_comparison.py uses, so both are scored the same way:
  olimpic_benchmark.py systems --predictions DIR --out DIR     (note F1)
  zeus_comparison.py tedn DIR --samples LIST                   (TEDn, in .venv-zeus)
and prints the symbol error rate (SER) over the whole split.
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import Levenshtein
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
OLIMPIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "olimpic"
SCANNED = OLIMPIC / "olimpic-1.0-scanned"
sys.path.insert(0, str(OLIMPIC / "code" / "repo"))

from app.linearization.Delinearizer import Delinearizer  # noqa: E402
from app.symbolic.part_to_score import part_to_score  # noqa: E402
from seq.data import SystemDataset, collate, samples, split_names  # noqa: E402
from seq.model import Seq2Seq  # noqa: E402
from seq.vocab import Vocab  # noqa: E402
from train_seq import VOCABULARY  # noqa: E402


def delinearize(lmx: str) -> str:
    """MusicXML of an LMX string. A sequence the delinearizer cannot parse becomes an empty
    part, as OLiMPiC's TEDn evaluation treats it (its error reporting itself can crash)."""
    delinearizer = Delinearizer(errout=None)
    try:
        delinearizer.process_text(lmx)
        part = delinearizer.part_element
    except Exception:
        part = ET.Element("part")
    return ET.tostring(part_to_score(part).getroot(), encoding="unicode", xml_declaration=True)


def load_model(checkpoint: Path, vocab: Vocab, device) -> Seq2Seq:
    state = torch.load(checkpoint, map_location=device)
    model = Seq2Seq(len(vocab), max_tokens=state["args"]["max_tokens"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    device = torch.device("cuda")
    vocab = Vocab.load(VOCABULARY)
    model = load_model(args.checkpoint, vocab, device)
    names = split_names(SCANNED, args.split)
    loader = torch.utils.data.DataLoader(SystemDataset(samples(SCANNED, names), vocab, False, model.max_tokens),
                                         batch_size=args.batch, num_workers=6, collate_fn=collate)
    errors = total = 0
    done = 0
    for images, widths, tokens in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths)
        for name, p, g in zip(names[done:], predicted, tokens.tolist()):
            lmx, gold = vocab.decode(p), vocab.decode(g[1:])
            errors += Levenshtein.distance(lmx.split(), gold.split())
            total += len(gold.split())
            path = args.out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".lmx").write_text(lmx + "\n")
            path.with_suffix(".musicxml").write_text(delinearize(lmx))
        done += len(predicted)
    print(f"SER = {100 * errors / total:.2f}% over {done} systems ({args.split})")


if __name__ == "__main__":
    main()
