"""What a label's bar holds, and whether a reading's bar matches it: spec 4's classes and test
(docs/EVAL_SETS.md), from runs/overnight_2026-11-14's compare_bars.py, voice_events.py and
cause_bars.py, which scored them on OLiMPiC first (EXPERIMENTS.md, "Spec 4 on OLiMPiC", 2026-11-18).

Bars are compared as drawn: implied tuplets marked (seq.lmx_tuplets), gaps spelled one way
(seq.label_form: a hidden rest and a forward of its length draw the same page) and a gap that ends
a voice left out, as it draws nothing. A reading's bars are aligned to the label's by rhythm
(tuplet_bars.rhythm: durations, dots, ratios, rests, chords, backups and forwards), and a label bar
is right when the alignment matches it to a bar of the same rhythm. Pitches are not compared: a
chord tone missing counts against a bar, a wrong pitch does not. (The first scorer's doc called
this equality; its numbers were rhythm's, found 2026-11-19.)

Classes, from the label alone: two voices drawn on a staff (seq.bar_voices, the parser training
uses too), a chord of three or more notes, a tuplet
whose number is shown, one whose number is hidden; a bar holding two counts under both, one
holding none is "none of these".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from seq.label_form import BACKUP_TO_BAR, with_backups_spelled, with_canonical_gaps  # noqa: E402
from seq.bar_voices import has_two_voices_on_a_staff  # noqa: E402
from seq.lmx_tuplets import imply_tuplets  # noqa: E402
from tuplet_bars import DURATIONS, RATIO, rhythm  # noqa: E402

TWO_VOICES, CHORDS, TUPLET_SHOWN, TUPLET_HIDDEN, NONE = (
    "two voices on a staff", "chord of 3+", "tuplet, number shown", "tuplet, number hidden", "none of these")
CLASSES = [TWO_VOICES, CHORDS, TUPLET_SHOWN, TUPLET_HIDDEN, NONE]
PITCH = re.compile(r"^[A-G][#b]*-?\d$")


# ---- bars as drawn ---------------------------------------------------------------------------

def bars_of(tokens: list[str]) -> list[list[str]]:
    """Each bar's tokens from its `measure` on; anything before the first is left out."""
    starts = [i for i, t in enumerate(tokens) if t == "measure"]
    return [tokens[s:e] for s, e in zip(starts, starts[1:] + [len(tokens)])]


def is_length(token: str) -> bool:
    return token in DURATIONS or token == "dot" or bool(RATIO.match(token))


def without_closing_gaps(tokens: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        end = i + 1
        while tokens[i] == "forward" and end < len(tokens) and is_length(tokens[end]):
            end += 1
        closes_a_voice = tokens[i] == "forward" and (end == len(tokens) or tokens[end] in (BACKUP_TO_BAR, "measure"))
        out += [] if closes_a_voice else tokens[i:end]
        i = end
    return out


def bars_drawn(lmx: str) -> list[list[str]]:
    canonical = with_canonical_gaps(imply_tuplets(lmx)).split()
    return bars_of(with_backups_spelled(" ".join(without_closing_gaps(canonical))).split())


def pairs(gold: list[list[str]], read: list[list[str]]) -> list[tuple[bool, list[str] | None]]:
    """(read right, the one reading bar it is paired with or None) for each label bar."""
    out: list[tuple[bool, list[str] | None]] = [(False, None)] * len(gold)
    for op, g0, g1, r0, r1 in Levenshtein.opcodes([rhythm(b) for b in gold], [rhythm(b) for b in read]):
        if op == "equal":
            out[g0:g1] = [(True, b) for b in read[r0:r1]]
        elif op == "replace" and g1 - g0 == r1 - r0:
            out[g0:g1] = [(False, b) for b in read[r0:r1]]
    return out


def right_bars(label: str, reading: str) -> list[tuple[list[str], bool]]:
    """Each label bar as drawn, and whether a reading's bar of the same rhythm is aligned to it."""
    gold = bars_drawn(label)
    return [(bar, right) for bar, (right, _) in zip(gold, pairs(gold, bars_drawn(reading)))]


# ---- classes ---------------------------------------------------------------------------------

def largest_chord(bar: list[str]) -> int:
    """Notes in the bar's largest chord: each further note of a chord follows a `chord` token."""
    largest, size, previous = 1, 1, None
    for token in bar:
        if PITCH.match(token):
            size = size + 1 if previous == "chord" else 1
            largest = max(largest, size)
        previous = token
    return largest


def classes_of(bar: list[str]) -> set[str]:
    out = set()
    if has_two_voices_on_a_staff(bar):
        out.add(TWO_VOICES)
    if largest_chord(bar) >= 3:
        out.add(CHORDS)
    if any(RATIO.match(t) for t in bar):
        out.add(TUPLET_HIDDEN if "tuplet:hidden" in bar else TUPLET_SHOWN)
    return out or {NONE}
