"""How well a sequence model reads the rhythm of bars with tuplets, numbered and unnumbered: the
staff model's largest single weakness on Polish dev (EXPERIMENTS_STAFF_MODEL.md), where a run of
triplets numbered only at first is read as plain notes after that.

usage: tuplet_bars.py READINGS.jsonl     (written by outer_notes.py: OLiMPiC dev, synthetic and
                                          scanned, labels with pitches as printed)

Each crop's label and reading are cut into bars, and each bar reduced to its rhythm: durations,
dots, tuplet ratios, rests, chords, backups and forwards, in order (pitches, beams, stems and
the rest left out). The bars are aligned by their rhythms (Levenshtein over the bar sequence),
and a label's bar is read right if it is matched to an equal bar. Reported for bars with tuplets
whose numbers are all shown, bars with a hidden number (`tuplet:hidden`), and bars without
tuplets, by source; and the tuplet-ratio tokens read against the label's, per crop.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import Levenshtein

DURATIONS = {"breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th"}
RATIO = re.compile(r"^\d+in\d+$")
STRUCTURE = {"dot", "rest", "chord", "backup", "forward"}


def bars(lmx: str) -> list[list[str]]:
    out: list[list[str]] = []
    for token in lmx.split():
        if token == "measure" or not out:
            out.append([])
        if token != "measure":
            out[-1].append(token)
    return out


def rhythm(bar: list[str]) -> str:
    return " ".join(t for t in bar if t in DURATIONS or t in STRUCTURE or RATIO.match(t))


def kind(bar: list[str]) -> str:
    if "tuplet:hidden" in bar:
        return "tuplets, a number hidden"
    if any(RATIO.match(t) for t in bar):
        return "tuplets, numbers shown"
    return "no tuplets"


def right_bars(reading: str, label: str) -> list[tuple[str, bool]]:
    """(kind, read right) for each of the label's bars."""
    gold = bars(label)
    gold_rhythms, read_rhythms = [rhythm(b) for b in gold], [rhythm(b) for b in bars(reading)]
    right = [False] * len(gold)
    for op, g0, g1, _, _ in Levenshtein.opcodes(gold_rhythms, read_rhythms):
        if op == "equal":
            right[g0:g1] = [True] * (g1 - g0)
    return [(kind(b), ok) for b, ok in zip(gold, right)]


def ratio_recall(reading: str, label: str) -> tuple[int, int]:
    """(tuplet-ratio tokens of the label also in the reading, counted as a multiset; label's)."""
    gold = Counter(t for t in label.split() if RATIO.match(t))
    read = Counter(t for t in reading.split() if RATIO.match(t))
    return sum((gold & read).values()), sum(gold.values())


def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).open()]
    for source in sorted({r["source"] for r in rows}) + ["all"]:
        chosen = [r for r in rows if source == "all" or r["source"] == source]
        counts, found, total = Counter(), 0, 0
        for r in chosen:
            for k, ok in right_bars(r["reading"], r["label"]):
                counts[k, ok] += 1
            f, t = ratio_recall(r["reading"], r["label"])
            found, total = found + f, total + t
        print(f"== {source}: tuplet-ratio tokens read {found / max(1, total):.1%} of {total}")
        for k in ("tuplets, numbers shown", "tuplets, a number hidden", "no tuplets"):
            n = counts[k, True] + counts[k, False]
            print(f"   {k:<26} {counts[k, True] / max(1, n):>6.1%} of {n} bars read with the right rhythm")


if __name__ == "__main__":
    main()
