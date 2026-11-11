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
from fractions import Fraction
from pathlib import Path

import Levenshtein
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from seq.lmx_tuplets import dots_mended, extra_beams_dropped, imply_tuplets  # noqa: E402
OLIMPIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "olimpic"
SCANNED = OLIMPIC / "olimpic-1.0-scanned"
sys.path.insert(0, str(OLIMPIC / "code" / "repo"))

from app.linearization.Delinearizer import Delinearizer  # noqa: E402
from app.symbolic.part_to_score import part_to_score  # noqa: E402
from seq.data import SystemDataset, collate, samples, split_names  # noqa: E402
from seq.model import Seq2Seq  # noqa: E402
from seq.ottava import add_octave_shifts, shifts_of, sounding  # noqa: E402
from seq.barlines import add_barlines, without_barlines  # noqa: E402
from seq.time_symbols import add_time_symbols, without_time_symbols  # noqa: E402
from seq.tuplet_numbers import hide_numbers, without_hidden_numbers  # noqa: E402
from seq.vocab import Vocab  # noqa: E402
from train_seq import VOCABULARY, comparable  # noqa: E402


def rewind_to_measure_start(part: ET.Element) -> None:
    """Sets every backup to the measure's start. Over 99% of them go there in OLiMPiC's gold,
    and the model errs on long measures (5/4, 12/4), putting a whole staff after the other."""
    for measure in part.findall("measure"):
        position = 0
        for element in measure:
            if element.tag == "note" and element.find("chord") is None and element.find("grace") is None:
                position += int(element.findtext("duration", 0))
            elif element.tag == "forward":
                position += int(element.findtext("duration", 0))
            elif element.tag == "backup":
                element.find("duration").text = str(position)
                position = 0


def delinearize(lmx: str, bar: Fraction | None = None) -> str:
    """MusicXML of an LMX string, unmarked tuplets and overbeamed groups restored (bar: see
    imply_tuplets) and ottava tokens drawn as signs over sounding pitches. Tokens OLiMPiC's
    delinearizer does not know are taken out and drawn afterwards: barlines and time symbols
    first, hidden tuplet numbers after the tuplet repairs, which drop one with a group they
    clear. A sequence the
    delinearizer cannot parse becomes an empty part, as OLiMPiC's TEDn evaluation treats it (its
    error reporting itself can crash)."""
    delinearizer = Delinearizer(errout=None)
    numbered, barlines = without_barlines(sounding(lmx))
    plain, time_symbols = without_time_symbols(numbered)
    restored, hidden_numbers = without_hidden_numbers(dots_mended(extra_beams_dropped(imply_tuplets(plain, bar), bar), bar))
    try:
        delinearizer.process_text(restored)
        part = delinearizer.part_element
        rewind_to_measure_start(part)
        add_octave_shifts(part, shifts_of(lmx))
        add_barlines(part, barlines)
        add_time_symbols(part, time_symbols)
        hide_numbers(part, hidden_numbers)
    except Exception:
        part = ET.Element("part")
    return ET.tostring(part_to_score(part).getroot(), encoding="unicode", xml_declaration=True)


def load_model(checkpoint: Path, device) -> Seq2Seq:
    state = torch.load(checkpoint, map_location=device)
    args = state["args"]
    # Tokens are only ever added at the end, so an older model's ids still hold.
    model = Seq2Seq(len(state["model"]["out.weight"]), max_tokens=args["max_tokens"], height=args.get("height", 256),
                    row_stride=args.get("row_stride", 32)).to(device)
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
    model = load_model(args.checkpoint, device)
    names = split_names(SCANNED, args.split)
    loader = torch.utils.data.DataLoader(SystemDataset(samples(SCANNED, names), vocab, False, model.max_tokens, model.height),
                                         batch_size=args.batch, num_workers=6, collate_fn=collate)
    errors = total = 0
    done = 0
    for images, widths, tokens in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths)
        for name, p, g in zip(names[done:], predicted, tokens.tolist()):
            lmx, gold = comparable(vocab.decode(p)), comparable(vocab.decode(g[1:]))
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
