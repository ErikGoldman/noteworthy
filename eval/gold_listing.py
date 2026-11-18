"""A Polish page's gold bar by bar, for review against the print: each bar's kern lines, numbered
as polish_overrides.jsonl addresses them (the unchanged page), and below them what the scorer
reads from the gold as corrected, staff by staff: pitches with each accidental drawn (!) or not
(hidden), length, onset in the bar (@, in quarters), beams per level, rests.

usage: gold_listing.py UNCHANGED.krn CORRECTED.krn [PRINT_FIXES.jsonl]
With PRINT_FIXES.jsonl (omr_ned.py --gold-print-fixes), the page's fixes are set as the scorer sets them.
Bars are counted from 0, as the scorer counts them (a pickup or a bar carried over from the
previous page is bar 0). Run with the musicdiff venv."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from omr_ned import authors_metric, parsed_as_authors_do, print_fixes, with_print_fixes


def kern_bars(text: str) -> list[list[tuple[int, str]]]:
    """Each bar's lines with their numbers from 1; a barline ends its bar."""
    bars: list[list[tuple[int, str]]] = [[]]
    for number, line in enumerate(text.split("\n"), 1):
        bars[-1].append((number, line))
        if line.startswith("="):
            bars.append([])
    return [bar for bar in bars if any(not line.startswith(("*", "!", "=")) and line for _, line in bar)]


def accidental(pitch) -> str:
    shown = pitch.accidental
    if shown is None:
        return ""
    return f"[{shown.name}{'!' if shown.displayStatus else ' hidden' if shown.displayStatus is False else '?'}]"


def described(note, measure) -> str:
    onset = Fraction(note.getOffsetInHierarchy(measure))
    if note.isRest:
        return f"rest {note.quarterLength:g} @{onset}" + (" (hidden)" if note.style.hideObjectOnPrint else "")
    beams = "/".join(beam.type + (f"-{beam.direction}" if beam.direction else "") for beam in note.beams.beamsList)
    pitches = "+".join(pitch.nameWithOctave + accidental(pitch) for pitch in note.pitches)
    return f"{pitches} {note.quarterLength:g} @{onset}" + (f" <{beams}>" if beams else "")


def main() -> None:
    unchanged, corrected = Path(sys.argv[1]), Path(sys.argv[2])
    metric = authors_metric()
    score = parsed_as_authors_do(metric, corrected, prediction=False)
    if len(sys.argv) > 3:
        with_print_fixes(score, print_fixes(Path(sys.argv[3]), unchanged.stem), metric.m21)
    measures = [list(part.getElementsByClass("Measure")) for part in score.parts]
    lines = kern_bars(unchanged.read_text())
    staff_names = ["upper", "lower"] if len(measures) == 2 else [f"staff {i + 1}" for i in range(len(measures))]
    print(f"{unchanged.stem}: {len(lines)} bars in the kern, {[len(m) for m in measures]} as the scorer reads the corrected gold")
    for index in range(max(len(lines), *map(len, measures))):
        print(f"\n=== bar {index}")
        for number, line in lines[index] if index < len(lines) else []:
            print(f"  {number:4d}  {line}")
        for name, staff in zip(staff_names, measures):
            if index < len(staff):
                for number, voice in enumerate(staff[index].voices or [staff[index]], 1):
                    print(f"  {name} voice {number}: " + "  ".join(described(n, staff[index]) for n in voice.notesAndRests))


if __name__ == "__main__":
    main()
