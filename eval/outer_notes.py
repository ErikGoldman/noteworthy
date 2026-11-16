"""How well a sequence model reads the music beyond a system's outer staff lines: notes under
ottava signs and notes far above the top staff or below the bottom one, where a StaffSeq2Seq
staff crop runs out of room (EXPERIMENTS_STAFF_MODEL.md). For comparing short training runs
that change the crops: overall SER is too blunt, since such notes are under 1% of its edits.

usage: outer_notes.py CHECKPOINT --out PREDICTIONS.jsonl [--limit N] [--device cuda|cpu]
       outer_notes.py --report PREDICTIONS.jsonl

The crops are OLiMPiC's dev systems, synthetic and scanned (the same 1,422 systems, printed and
scanned), labelled with pitches as printed and ottava tokens (build_olimpic_written.py, SPLIT
dev, in data/olimpic_written_dev with their staff positions). Every crop is read; reported:
  - notes under a sign: right only if their pitch and their ottava token both are;
  - notes FAR_STEPS or more beyond the system's outer lines and under no sign;
  - ottava tokens written in crops with no sign (false signs);
  - SER over all crops, and all other notes, as a check that nothing else got worse.
"Wrong" is a label token substituted or missed in the Levenshtein alignment of the reading.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import Levenshtein
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from seq.data import samples, split_names  # noqa: E402
from seq.ottava import OCTAVES_OF_TOKEN  # noqa: E402
from seq.vocab import Vocab  # noqa: E402
from train_seq import VOCABULARY, as_lmx, meter_starts, with_staves  # noqa: E402
from seq_predict import load_model, reading_dataset  # noqa: E402

DATA = Path(os.environ.get("NW_DATA", ROOT / "data")) / "olimpic_written_dev"
SOURCES = ("synthetic", "scanned")
FAR_STEPS = 6  # three ledger lines beyond the outer line
PITCH = re.compile(r"^([A-G])(\d)$")
STEPS = "CDEFGAB"
# The steps of each clef's outer lines, bottom and top: notes beyond them are outside the staff.
STAFF_LINES = {"G2": ("E4", "F5"), "F4": ("G2", "A3")}


def step(pitch: str) -> int:
    letter, octave = PITCH.match(pitch).groups()
    return int(octave) * 7 + STEPS.index(letter)


def note_staves(tokens: list[str]) -> list[int | None]:
    """Each note's tokens' staff. LMX names a note's staff among the attributes after its pitch
    and duration (staff:N) only when it changes; a note naming none, as most chord notes and
    runs do, is on the staff of the note before it."""
    starts = [i for i, t in enumerate(tokens)
              if (PITCH.match(t) and (i == 0 or tokens[i - 1] != "chord")) or t in ("rest", "chord")]
    out, current = [None] * len(tokens), 1
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(tokens)
        end = next((i for i in range(start + 1, end) if tokens[i] in ("measure", "backup", "forward")), end)
        named = [int(t[6:]) for t in tokens[start:end] if t.startswith("staff:")]
        current = named[0] if named else current
        out[start:end] = [current] * (end - start)
    return out


def note_kinds(tokens: list[str]) -> dict[int, str]:
    """The kind of each pitch token of a system's label: "under a sign", "far outside" (FAR_STEPS
    beyond the top staff's top line or the bottom staff's bottom line), or "elsewhere"
    (any other note on a treble or bass staff). Pitches on other clefs are left out."""
    staves, clefs, kinds = note_staves(tokens), {}, {}
    # The header gives every staff a clef, notes or none: the bottom staff is the last one named.
    count = max((clef_staff(tokens, i) for i, t in enumerate(tokens) if t.startswith("clef:")), default=1)
    for i, token in enumerate(tokens):
        if token.startswith("clef:"):
            clefs[clef_staff(tokens, i)] = token[5:]
        if not PITCH.match(token) or staves[i] not in clefs or clefs[staves[i]] not in STAFF_LINES:
            continue
        if i + 1 < len(tokens) and tokens[i + 1] in OCTAVES_OF_TOKEN:
            kinds[i] = "under a sign"
            continue
        bottom, top = (step(p) for p in STAFF_LINES[clefs[staves[i]]])
        above, below = step(token) - top, bottom - step(token)
        outer = (staves[i] == 1 and above >= FAR_STEPS) or (staves[i] == count and below >= FAR_STEPS)
        kinds[i] = "far outside" if outer else "elsewhere"
    return kinds


def clef_staff(tokens: list[str], at: int) -> int:
    """The staff of the clef token at `at`: the staff:N after it; a one-staff label names none."""
    named = at + 1 < len(tokens) and tokens[at + 1].startswith("staff:")
    return int(tokens[at + 1][6:]) if named else 1


def wrong_label_tokens(reading: list[str], label: list[str]) -> set[int]:
    return {g for op, _, g in Levenshtein.editops(reading, label) if op in ("replace", "insert")}


def report(rows: list[dict]) -> None:
    print(f"{len(rows)} crops")
    for source in SOURCES + ("all",):
        chosen = [r for r in rows if source == "all" or r["source"] == source]
        notes, edits, total, false_signs = Counter(), 0, 0, 0
        for r in chosen:
            reading, label = r["reading"].split(), r["label"].split()
            wrong = wrong_label_tokens(reading, label)
            for i, kind in note_kinds(label).items():
                missed = i in wrong or (kind == "under a sign" and i + 1 in wrong)
                notes[kind, missed] += 1
            edits += Levenshtein.distance(reading, label)
            total += len(label)
            if not any(t in OCTAVES_OF_TOKEN for t in label):
                false_signs += sum(t in OCTAVES_OF_TOKEN for t in reading)
        print(f"== {source}: SER {edits / total:.4f}, false ottava tokens {false_signs}")
        for kind in ("under a sign", "far outside", "elsewhere"):
            n = notes[kind, True] + notes[kind, False]
            print(f"   {kind:<14} {notes[kind, True] / max(1, n):>6.1%} wrong of {n}")


def read(args) -> list[dict]:
    """args.data: the set's directory, a subdirectory per source and staff_positions beside them."""
    device = torch.device(args.device)
    vocab = Vocab.load(VOCABULARY)
    model = load_model(args.checkpoint, device)
    rows = []
    for source in SOURCES:
        root = args.data / source
        items = samples(root, split_names(root, "dev"))[:args.limit]
        paired = with_staves(items, root, args.data / "staff_positions")
        items = [s for s, _ in paired]
        dataset, batched = reading_dataset(model, paired, vocab)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=batched)
        done = 0
        with torch.no_grad():
            for images, widths, tokens in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    predicted = model.generate(images.to(device), widths, starts=meter_starts(tokens, vocab))
                for sample, p, g in zip(items[done:], predicted, tokens.tolist()):
                    rows.append({"source": source, "name": str(sample.image.relative_to(root).with_suffix("")),
                                 "reading": as_lmx(model, vocab, p), "label": as_lmx(model, vocab, g[1:])})
                done += len(predicted)
                print(f"{source}: {done}/{len(items)}", file=sys.stderr, flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, help="where the readings are saved, one JSON line a crop")
    ap.add_argument("--report", type=Path, help="report on readings saved earlier, reading nothing")
    ap.add_argument("--data", type=Path, default=DATA, help="the crops to read (default: OLiMPiC dev's, two staves)")
    ap.add_argument("--limit", type=int, help="the first N crops of each source only (a quick check)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    if args.report:
        rows = [json.loads(line) for line in args.report.open()]
    elif args.checkpoint and args.out:
        rows = read(args)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        ap.error("give CHECKPOINT and --out, or --report")
    report(rows)


if __name__ == "__main__":
    main()
