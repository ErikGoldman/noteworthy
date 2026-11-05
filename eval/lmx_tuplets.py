"""Restores tuplets a transcription left out, from the arithmetic of the bar.

Engravers stop printing the number once a passage has settled into triplets or sextuplets,
and then nothing but the bar's length tells them from plain notes. The sequence model reads
such notes perfectly and writes them plain, so their voice overfills its measure. Where
reading a voice's beam groups of 3 and 6 as tuplets fills the measure exactly, this does so.

A crop rarely shows a time signature, so the bar's length defaults to what most of its
measures say (measure_evidence): an unmarked group only ever lengthens a voice.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

LENGTHS = {"maxima": 32, "long": 16, "breve": 8, "whole": 4, "half": 2, "quarter": 1, "eighth": Fraction(1, 2),
           "16th": Fraction(1, 4), "32nd": Fraction(1, 8), "64th": Fraction(1, 16), "128th": Fraction(1, 32),
           "256th": Fraction(1, 64), "512th": Fraction(1, 128), "1024th": Fraction(1, 256)}
TUPLET = re.compile(r"(\d+)in(\d+)$")
PITCH = re.compile(r"[A-G]\d$")
TUPLET_OF_GROUP = {3: ("3in2", Fraction(2, 3)), 6: ("6in4", Fraction(2, 3))}


@dataclass
class Note:
    duration_at: int  # index of its duration token
    length: Fraction
    advances: bool  # False for chord members and grace notes
    marked: bool  # already part of a tuplet
    beam_group: int | None = None


@dataclass
class Voice:
    notes: list[Note] = field(default_factory=list)
    skipped: Fraction = Fraction(0)  # forwards
    measure_rest: bool = False

    def total(self) -> Fraction:
        return self.skipped + sum((n.length for n in self.notes if n.advances), Fraction(0))

    def beam_groups(self) -> dict[int, list[Note]]:
        groups: dict[int, list[Note]] = {}
        for note in self.notes:
            if note.beam_group is not None:
                groups.setdefault(note.beam_group, []).append(note)
        return groups


def length_at(tokens: list[str], at: int) -> tuple[Fraction, bool]:
    """A duration token's length with its dots and tuplet ratio, and whether it has a ratio."""
    base = Fraction(LENGTHS[tokens[at]])
    length, dot = base, Fraction(1, 2)
    for token in tokens[at + 1:]:
        ratio = TUPLET.match(token)
        if ratio:
            return length * int(ratio[2]) / int(ratio[1]), True
        if token != "dot":
            break
        length, dot = length + base * dot, dot / 2
    return length, False


def parse_measure(tokens: list[str], start: int, end: int) -> list[Voice]:
    voices, voice = [], Voice()
    depth, group, next_group = 0, None, 0
    chord = grace = False
    head = None
    note = None
    for i in range(start, end):
        token = tokens[i]
        if token in ("grace", "grace:slash"):
            grace = True
        elif token == "chord":
            chord = True
        elif token == "rest:measure":
            voice.measure_rest = True
        elif token in ("forward", "backup", "rest") or PITCH.match(token):
            head = token
        elif token in LENGTHS and head:
            length, marked = length_at(tokens, i)
            if head == "backup":
                voices.append(voice)
                voice = Voice()
            elif head == "forward":
                voice.skipped += length
            else:
                note = Note(i, length, not (chord or grace), marked, group if depth else None)
                voice.notes.append(note)
            head, chord, grace = None, False, False
        elif token == "beam:begin":
            if depth == 0:
                group, next_group = next_group, next_group + 1
                if note:
                    note.beam_group = group
            depth += 1
        elif token == "beam:end":
            depth = max(0, depth - 1)
    voices.append(voice)
    return voices


def measures_of(tokens: list[str]) -> list[list[Voice]]:
    starts = [i for i, t in enumerate(tokens) if t == "measure"]
    return [parse_measure(tokens, s, e) for s, e in zip(starts, starts[1:] + [len(tokens)])]


def written_bar(tokens: list[str]) -> Fraction | None:
    """The first time signature's bar length, in quarters."""
    beats = next((t for t in tokens if t.startswith("beats:")), None)
    kind = next((t for t in tokens if t.startswith("beat-type:")), None)
    if beats is None or kind is None:
        return None
    return Fraction(4 * int(beats.split(":")[1]), int(kind.split(":")[1]))


def measure_evidence(measure: list[Voice]) -> Fraction | None:
    """A measure's own idea of the bar. A voice with no candidates cannot shrink, so the bar is
    at least the longest of those; if all have candidates, the shortest is the best guess."""
    voices = [v for v in measure if v.notes and not v.measure_rest]
    fixed = [v.total() for v in voices if not tuplet_candidates(v)]
    return max(fixed) if fixed else min((v.total() for v in voices), default=None)


def likely_bar(measures: list[list[Voice]]) -> Fraction | None:
    votes = Counter(e for e in map(measure_evidence, measures) if e)
    return min(votes, key=lambda bar: (-votes[bar], bar)) if votes else None


def tuplet_candidates(voice: Voice) -> list[tuple[list[Note], str, Fraction]]:
    """Unmarked beam groups of 3 or 6 equal notes, with their tuplet token and ratio."""
    found = []
    for notes in voice.beam_groups().values():
        timed = [n for n in notes if n.advances]
        if len(timed) in TUPLET_OF_GROUP and len({n.length for n in timed}) == 1 and not any(n.marked for n in notes):
            found.append((notes, *TUPLET_OF_GROUP[len(timed)]))
    return found


def starts_on_their_division(voice: Voice, candidates: list[tuple[list[Note], str, Fraction]]) -> bool:
    """Whether every candidate, read as a tuplet, starts on a multiple of its own span. A plain
    group fitted to the beat by a rest (16th rest, three 16ths) does not, as a tuplet."""
    ratio_of = {id(n): ratio for notes, _, ratio in candidates for n in notes}
    position = voice.skipped
    span_left: dict[int, Fraction] = {}
    for notes, _, ratio in candidates:
        span_left[id(notes[0])] = sum((n.length * ratio for n in notes if n.advances), Fraction(0))
    for note in voice.notes:
        span = span_left.get(id(note))
        if span is not None and position % span:
            return False
        if note.advances:
            position += note.length * ratio_of.get(id(note), 1)
    return True


def insertions_for(voice: Voice, bar: Fraction) -> list[tuple[int, str]]:
    """Where tuplet tokens go if they make the voice fill the bar exactly."""
    if voice.measure_rest or voice.total() <= bar:
        return []
    candidates = tuplet_candidates(voice)
    saved = sum((n.length * (1 - ratio) for notes, _, ratio in candidates for n in notes if n.advances), Fraction(0))
    if voice.total() - saved != bar or not starts_on_their_division(voice, candidates):
        return []
    return [(n.duration_at + 1, token) for notes, token, _ in candidates for n in notes]


def imply_tuplets(lmx: str, bar: Fraction | None = None) -> str:
    """The LMX with tuplets the bar's arithmetic implies. bar is the length in force from earlier
    systems, for one that does not print its own time signature."""
    tokens = lmx.split()
    measures = measures_of(tokens)
    bar = written_bar(tokens) or bar or likely_bar(measures)
    if bar is None:
        return lmx
    insertions = [i for measure in measures for voice in measure for i in insertions_for(voice, bar)]
    for at, token in sorted(insertions, reverse=True):
        tokens.insert(at, token)
    return " ".join(tokens)
