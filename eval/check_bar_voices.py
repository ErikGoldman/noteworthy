"""The voices LMX bars hold, as seq.bar_voices counts them from the tokens, against the MusicXML they
were linearized from, where each note carries its own <staff>, <voice> and print-object: the check
that every caller of bar_voices (the voice-count label form, training's two-voice repeat, spec 4's
two-voice class) reads voices as the score has them. Run it before changing the parser
(seq.lmx_tuplets.parse_measure) or bar_voices; it should find no disagreement, or list each.

usage: check_bar_voices.py [--every K] [--synthetic N] [--examples E] [--fixture OUT.jsonl]

Scores: every Kth of the render list's training scores (default 10), cut into systems as the crop
builders cut them (build_seq_dataset.break_into_systems, seeded by the score's index), and N
synthetic pieces (train/synthetic_bars.py: two voices on a staff by design). Per bar and staff:
  notes    voices with notes (grace notes and chord members too, not rests) that are not
           print-object="no", on the staff each names: seq.bar_voices.voices_with_notes
  written  stretches between backups holding a note or rest, on the staff of their first:
           seq.bar_voices.voices_written
--fixture writes up to E bars of each kind of bar that once broke a rule (a voice opening with a
grace note, a clef change opening the bar or inside it, a whole-bar rest, a hidden rest, a
cross-staff note, a stretch opening with a forward, tuplets in two voices) with their counts from
the MusicXML: train/test_bar_voices.py's cases.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_seq_dataset import (DATA, break_into_systems, is_training_score, linearize, load_musicxml,  # noqa: E402
                               piano_part, split_part_to_systems, without_octave_clefs)
from omr.pieces import pdmx_license_conflicts  # noqa: E402
from seq.bar_voices import voices_with_notes, voices_written  # noqa: E402
from seq.label_form import bars_of  # noqa: E402
from seq.lmx_tuplets import PITCH  # noqa: E402
from synthetic_bars import synthetic_score  # noqa: E402

Counts = dict[int, int]


def staff_of(element: ET.Element) -> int:
    return int(element.findtext("staff", "1"))


def notes_in_musicxml(measure: ET.Element) -> Counts:
    voices: dict[int, set] = {}
    for note in measure.findall("note"):
        if note.get("print-object") != "no" and note.find("rest") is None:
            voices.setdefault(staff_of(note), set()).add(note.findtext("voice", "1"))
    return {staff: len(v) for staff, v in voices.items()}


def written_in_musicxml(measure: ET.Element) -> Counts:
    counts: Counter = Counter()
    stretch: list[ET.Element] = []
    for element in list(measure) + [ET.Element("backup")]:
        if element.tag == "backup":
            heads = [e for e in stretch if e.tag == "note"]
            if heads:
                counts[staff_of(heads[0])] += 1
            stretch = []
        elif element.tag in ("note", "forward"):
            stretch.append(element)
    return dict(counts)


def kinds_of(measure: ET.Element, tokens: list[str]) -> set[str]:
    """Which of the bars that once broke a rule this one is."""
    kinds = set()
    stretches = [[]]
    for element in measure:
        if element.tag == "backup":
            stretches.append([])
        elif element.tag in ("note", "forward"):
            stretches[-1].append(element)
    for stretch in filter(None, stretches):
        first = stretch[0]
        if first.tag == "forward":
            kinds.add("stretch opening with a forward")
        elif first.find("grace") is not None:
            kinds.add("voice opening with a grace note")
        if len({staff_of(n) for n in stretch if n.tag == "note"}) > 1:
            kinds.add("cross-staff note")
    first_head = next((i for i, t in enumerate(tokens) if PITCH.match(t) or t in ("rest", "forward")), len(tokens))
    if any(t.startswith("clef:") for t in tokens[:first_head]):
        kinds.add("clef opening the bar")
    if any(t.startswith("clef:") for t in tokens[first_head:]):
        kinds.add("clef inside the bar")
    if "rest:measure" in tokens:
        kinds.add("whole-bar rest")
    if "print-object:no" in tokens:
        kinds.add("hidden note or rest")
    if any(n > 1 for n in notes_in_musicxml(measure).values()) and measure.find("note/time-modification") is not None:
        kinds.add("tuplets in two voices")
    return kinds


def system_bars(part: ET.Element) -> list[tuple[ET.Element, list[str]]]:
    """The part's systems, each bar's MusicXML with its LMX tokens."""
    out = []
    for page in split_part_to_systems(part):
        for system in page.systems:
            tokens = linearize(system.part).split()
            measures = system.part.findall("measure")
            bars = bars_of(tokens)
            if len(bars) == len(measures):
                out += list(zip(measures, bars))
    return out


def training_parts(every: int):
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    listed = (DATA / "render_scores.txt").read_text().split()
    for index in range(0, len(listed), every):
        if not is_training_score(listed[index], conflicts):
            continue
        try:
            part = piano_part(load_musicxml(Path(listed[index])))
        except Exception:  # noqa: BLE001 - a score the builders cannot read is not theirs either
            continue
        if part is not None:
            without_octave_clefs(part)
            break_into_systems(part, random.Random(index))
            yield f"render list {index}", part


def synthetic_parts(n: int):
    for index in range(n):
        part = piano_part(synthetic_score(index, 8)[0])
        break_into_systems(part, random.Random(index))
        yield f"synthetic {index}", part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--synthetic", type=int, default=500)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--fixture", type=Path)
    args = ap.parse_args()
    tally, disagreements, fixture, kinds_seen = Counter(), [], [], Counter()
    sources = list(training_parts(args.every)) + list(synthetic_parts(args.synthetic))
    for name, part in sources:
        try:
            bars = system_bars(part)
        except Exception:  # noqa: BLE001 - as the builders: a system that will not linearize is left out
            tally["systems left out"] += 1
            continue
        for measure, tokens in bars:
            truth = {"notes": notes_in_musicxml(measure), "written": written_in_musicxml(measure)}
            read = {"notes": voices_with_notes(tokens), "written": voices_written(tokens)}
            tally["bars"] += 1
            for question in truth:
                if truth[question] != read[question]:
                    tally[f"{question} disagrees"] += 1
                    disagreements.append((name, question, truth[question], read[question], " ".join(tokens)))
            for kind in kinds_of(measure, tokens):
                kinds_seen[kind] += 1
                if kinds_seen[kind] <= args.examples:
                    fixture.append({"kind": kind, "source": name, "bar": " ".join(tokens),
                                    "notes": truth["notes"], "written": truth["written"]})
    print(f"{len(sources)} scores; {dict(tally)}")
    print("bars of each kind:", dict(kinds_seen))
    for name, question, truth, read, bar in disagreements[:20]:
        print(f"  {name}: {question} MusicXML {truth} tokens {read}\n    {bar[:300]}")
    if args.fixture:
        args.fixture.write_text("".join(json.dumps(case) + "\n" for case in fixture))
    sys.exit(1 if disagreements else 0)


if __name__ == "__main__":
    main()
