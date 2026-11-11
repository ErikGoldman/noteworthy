"""What a two-spine (piano) **kern file draws where: its measures, the clef, key and meter in
force on each staff and where they change, and its notes' pitches, measure by measure and staff
by staff.

Times are in quarter notes from the start of the file. A data line starts at the time the
shortest note still sounding on the line before it ends; grace notes take no time. Spines split
(`*^`) and merge (`*v`) into voices of the same staff: the left spine is staff 2 (the lower),
the right staff 1, as MusicXML numbers a piano's staves. A file whose two spines merge into one
is read up to the merger: OLiMPiC's GrandStaff-LMX cut those short the same way, as GrandStaff's
images draw no notes after it.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

STAFF_OF_SPINE = (2, 1)
CLEF, KEY, METER = "clef", "key", "meter"
SIGNATURE = re.compile(r"\*(?:clef(?P<clef>[A-G]v*\^*\d)|k\[(?P<key>[^]]*)\]|M(?P<meter>\d+/\d+))$")
NOTE = re.compile(r"([a-gA-G])\1*(#+|-+|n)?")
DURATION = re.compile(r"(\d+)(?:%(\d+))?(\.*)")
ALTER = {"#": 1, "##": 2, "-": -1, "--": -2, "n": 0, None: 0}


class Unreadable(ValueError):
    pass


@dataclass(frozen=True)
class Change:
    time: Fraction
    staff: int
    kind: str  # CLEF, KEY or METER
    value: str | int  # a clef's sign and line ("G2", "F4"), a key's fifths, a meter ("3/4")


@dataclass
class Kern:
    opening: dict[tuple[str, int], str | int] = field(default_factory=dict)  # (kind, staff): value
    changes: list[Change] = field(default_factory=list)  # with notes after them
    bar_times: list[Fraction] = field(default_factory=list)  # each measure's start but the first
    end: Fraction = Fraction(0)
    pitches: Counter = field(default_factory=Counter)  # (measure, staff, step, alter, octave), grace notes left out
    grace_notes: int = 0
    cut_at_merger: bool = False
    after_notes: set[tuple[str, int]] = field(default_factory=set)  # (kind, staff) first given after notes

    def measure_at(self, time: Fraction) -> int:
        """0-based; a time on a barline is the start of the measure after it."""
        return bisect.bisect_right(self.bar_times, time)

    def measure_start(self, measure: int) -> Fraction:
        return self.bar_times[measure - 1] if measure else Fraction(0)

    @property
    def measure_count(self) -> int:
        return 1 + sum(t < self.end for t in self.bar_times)


@dataclass
class _Spine:
    staff: int
    ends: Fraction = Fraction(0)  # when its note sounding now ends


def duration(token: str) -> Fraction:
    """A note or rest's length in quarters; grace notes (q, Q) have none."""
    match = DURATION.search(token)
    if "q" in token.lower() or match is None:
        return Fraction(0)
    reciprocal, scale, dots = match.groups()
    whole = {"0": Fraction(2), "00": Fraction(4)}.get(reciprocal) or Fraction(1, int(reciprocal))
    length = 4 * whole * int(scale or 1)
    return length * (2 - Fraction(1, 2 ** len(dots)))


def pitch(note: str) -> tuple[str, int, int] | None:
    """(step, alter, octave) of one note of a token; None for a rest."""
    if "r" in note:
        return None
    match = NOTE.search(note)
    if match is None:
        return None
    letters, accidental = match.group(0).rstrip("#-n"), match.group(2)
    step = letters[0].upper()
    octave = 3 + len(letters) if letters[0].islower() else 4 - len(letters)
    return step, ALTER[accidental], octave


def read(text: str) -> Kern:
    lines = [line for line in text.splitlines() if line and not line.startswith("!")]
    reader = _Reader(lines)
    for i, line in enumerate(lines):
        tokens = line.split("\t")
        if line.startswith("**"):
            reader.spines = [_Spine(staff) for staff in STAFF_OF_SPINE[:len(tokens)]]
        elif len(tokens) != len(reader.spines):
            raise Unreadable(f"line {i + 1} has {len(tokens)} tokens for {len(reader.spines)} spines")
        elif line.startswith("*"):
            if _merges_the_staves(reader.spines, tokens):
                reader.kern.cut_at_merger = reader.notes_after[i + 1]
                break
            reader.interpretation(tokens, i)
        elif line.startswith("="):
            reader.barline()
        else:
            reader.data(tokens)
    reader.kern.end = max([reader.time] + [spine.ends for spine in reader.spines])
    return reader.kern


class _Reader:
    def __init__(self, lines: list[str]):
        self.kern = Kern()
        self.spines: list[_Spine] = []
        self.in_force: dict[tuple[str, int], str | int] = {}
        self.time = Fraction(0)
        self.seen_notes = False
        self.notes_after = [False] * (len(lines) + 1)
        for i in range(len(lines) - 1, -1, -1):
            self.notes_after[i] = self.notes_after[i + 1] or _is_data(lines[i])

    def interpretation(self, tokens: list[str], line: int) -> None:
        for spine, token in zip(self.spines, tokens):
            if signature := SIGNATURE.match(token):
                kind = signature.lastgroup
                self.signature(kind, spine.staff, _value(kind, signature.group(kind)), line)
        self.spines = _split_and_merge(self.spines, tokens)

    def signature(self, kind: str, staff: int, value: str | int, line: int) -> None:
        if (kind, staff) not in self.in_force:
            if self.seen_notes:
                self.kern.after_notes.add((kind, staff))
            self.kern.opening[kind, staff] = value
        elif self.in_force[kind, staff] != value and self.notes_after[line + 1]:
            self.kern.changes.append(Change(self.time, staff, kind, value))
        self.in_force[kind, staff] = value

    def barline(self) -> None:
        bars = self.kern.bar_times
        if self.time > 0 and (not bars or bars[-1] < self.time):
            bars.append(self.time)

    def data(self, tokens: list[str]) -> None:
        self.seen_notes = True
        measure = self.kern.measure_at(self.time)
        started = []
        for spine, token in zip(self.spines, tokens):
            if token == ".":
                continue
            started.append(duration(token.split(" ")[0]))
            spine.ends = self.time + started[-1]
            notes = [p for note in token.split(" ") if (p := pitch(note)) is not None]
            if started[-1]:
                for p in notes:
                    self.kern.pitches[(measure, spine.staff, *p)] += 1
            else:  # GrandStaff's images draw them all, its labels often none
                self.kern.grace_notes += len(notes)
        if any(started):  # a line of grace notes only takes no time
            self.time = min(spine.ends for spine in self.spines if spine.ends > self.time)


def _value(kind: str, text: str) -> str | int:
    if kind == KEY:  # *k[f#c#] is two sharps, *k[b-e-a-] three flats
        return text.count("#") - text.count("-")
    return text


def _is_data(line: str) -> bool:
    return not line.startswith(("*", "=", "!"))


def _merges_the_staves(spines: list[_Spine], tokens: list[str]) -> bool:
    """All spines merge into one: what is left is no longer two staves."""
    return all(token == "*v" for token in tokens) and len({spine.staff for spine in spines}) > 1


def _split_and_merge(spines: list[_Spine], tokens: list[str]) -> list[_Spine]:
    after: list[_Spine] = []
    i = 0
    while i < len(tokens):
        token, spine = tokens[i], spines[i]
        if token == "*^":
            after += [spine, _Spine(spine.staff, spine.ends)]
        elif token == "*v":
            merged = i
            while merged < len(tokens) and tokens[merged] == "*v" and spines[merged].staff == spine.staff:
                merged += 1
            after.append(_Spine(spine.staff, max(s.ends for s in spines[i:merged])))
            i = merged
            continue
        elif token != "*-":
            after.append(spine)
        i += 1
    return after
