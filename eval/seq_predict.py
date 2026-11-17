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
from seq.lmx_tuplets import ALL_RULES, PageRule, with_page_rules  # noqa: E402
OLIMPIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "olimpic"
SCANNED = OLIMPIC / "olimpic-1.0-scanned"
sys.path.insert(0, str(OLIMPIC / "code" / "repo"))

from app.linearization.Delinearizer import Delinearizer  # noqa: E402
from app.symbolic.part_to_score import part_to_score  # noqa: E402
from seq.data import SystemDataset, collate, samples, split_names  # noqa: E402
from seq.model import Seq2Seq, StaffSeq2Seq  # noqa: E402
from seq.ottava import add_octave_shifts, shifts_of, sounding  # noqa: E402
from seq.barlines import add_barlines, without_barlines  # noqa: E402
from seq.time_symbols import add_time_symbols, without_time_symbols  # noqa: E402
from seq.tuplet_numbers import hide_numbers, without_hidden_numbers  # noqa: E402
from seq.label_form import LabelForm  # noqa: E402
from seq.vocab import Vocab  # noqa: E402
from train_seq import VOCABULARY, as_lmx, comparable, meter_starts  # noqa: E402


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


def delinearize(lmx: str, bar: Fraction | None = None, rules: frozenset[PageRule] = ALL_RULES) -> str:
    """MusicXML of an LMX string, the page rules in rules applied (seq.lmx_tuplets.with_page_rules:
    unmarked tuplets and overbeamed groups restored; bar: see imply_tuplets) and ottava tokens
    drawn as signs over sounding pitches. Tokens OLiMPiC's delinearizer does not know are taken
    out and drawn afterwards: barlines and time symbols first, hidden tuplet numbers after the
    tuplet repairs, which drop one with a group they clear. A sequence the delinearizer cannot
    parse becomes an empty part, as OLiMPiC's TEDn evaluation treats it (its error reporting
    itself can crash)."""
    delinearizer = Delinearizer(errout=None)
    numbered, barlines = without_barlines(sounding(lmx))
    plain, time_symbols = without_time_symbols(numbered)
    restored, hidden_numbers = without_hidden_numbers(with_page_rules(plain, bar, rules))
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


def load_model(checkpoint: Path, device) -> Seq2Seq | StaffSeq2Seq:
    """A run's model: StaffSeq2Seq if it trained on staff crops (train_seq.py --staves)."""
    state = torch.load(checkpoint, map_location=device)
    args = state["args"]
    # Tokens are only ever added at the end, so an older model's ids still hold.
    vocab_size = len(state["model"]["out.weight"])
    if args.get("staves"):
        model = StaffSeq2Seq(vocab_size, max_tokens=args["max_tokens"], row_stride=args.get("row_stride", 32),
                             blocks=args.get("blocks", 1), across_columns=args.get("across_columns", 1),
                             staff_outward=args.get("staff_outward", 0.0),
                             meter_context=args.get("meter_context", False),
                             staff_cut_between=args.get("staff_cut_between", False),
                             label_form=LabelForm.from_options(args)).to(device)
    else:
        model = Seq2Seq(vocab_size, max_tokens=args["max_tokens"], height=args.get("height", 256),
                        row_stride=args.get("row_stride", 32), meter_context=args.get("meter_context", False),
                        relative_rows=args.get("relative_rows", False),
                        row_windows=args.get("row_windows", 0), interline_scaled=args.get("interline_scale", False),
                        staff_anchored=args.get("staff_anchored", False),
                        staff_bands=args.get("staff_bands", False),
                        staff_windows=args.get("staff_windows", False),
                        label_form=LabelForm.from_options(args)).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def reading_dataset(model, pairs: list, vocab: Vocab):
    """(dataset, collate) that reads (sample, staves) pairs as `model` was trained to: staff crops,
    systems at staff scale, or systems at the model's fixed height (staves unused)."""
    from seq.staff_data import InterlineSystemDataset, StaffDataset, collate_staves, collate_systems
    if isinstance(model, StaffSeq2Seq):
        return StaffDataset(pairs, vocab, False, model.max_tokens, outward=model.staff_outward,
                            meter_context=model.meter_context, cut_between=model.staff_cut_between,
                            form=model.label_form), collate_staves
    if model.interline_scaled:
        return InterlineSystemDataset(pairs, vocab, False, model.max_tokens, meter_context=model.meter_context,
                                      form=model.label_form), collate_systems
    return SystemDataset([s for s, _ in pairs], vocab, False, model.max_tokens, model.height, model.meter_context,
                         model.label_form), collate


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
    if isinstance(model, StaffSeq2Seq):
        sys.exit("seq_predict.py reads whole crops; a StaffSeq2Seq run is scored by train_seq.py's dev check and seq_pages.py")
    names = split_names(SCANNED, args.split)
    loader = torch.utils.data.DataLoader(SystemDataset(samples(SCANNED, names), vocab, False, model.max_tokens, model.height,
                                                       model.meter_context, model.label_form),
                                         batch_size=args.batch, num_workers=6, collate_fn=collate)
    errors = total = 0
    done = 0
    for images, widths, tokens in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths, starts=meter_starts(tokens, vocab))
        for name, p, g in zip(names[done:], predicted, tokens.tolist()):
            lmx, gold = comparable(as_lmx(model, vocab, p)), comparable(as_lmx(model, vocab, g[1:]))
            errors += Levenshtein.distance(lmx.split(), gold.split())
            total += len(gold.split())
            path = args.out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".lmx").write_text(lmx + "\n")
            path.with_suffix(".musicxml").write_text(delinearize(lmx))
        done += len(predicted)
    told = ", each given its label's meter" if model.meter_context else ""
    print(f"SER = {100 * errors / total:.2f}% over {done} systems ({args.split}{told})")


if __name__ == "__main__":
    main()
