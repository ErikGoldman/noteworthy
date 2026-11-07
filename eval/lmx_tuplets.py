"""Restores tuplets a transcription left out, from the arithmetic of the bar.

Engravers stop printing the number once a passage has settled into triplets or sextuplets,
and then nothing but the bar's length tells them from plain notes. The sequence model reads
such notes perfectly and writes them plain, so their voice overfills its measure. Where
reading a voice's beam groups of 3 and 6 as tuplets fills the measure exactly, this does so.

A crop rarely shows a time signature, so the bar's length defaults to what most of its
measures say (measure_evidence): an unmarked group only ever lengthens a voice.

The model errs the other way too: reading into a triplet passage, it marks the plain notes
before it as triplets, and their voice falls short of the bar. Where clearing the marks from
a voice's first or last notes fills the measure exactly and leaves whole triplets, this does so.
It also reads a beam too many on a group, and that voice falls short as well: where giving one
group's notes one beam fewer fills the measure exactly, extra_beams_dropped does so.
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
TWO_THIRDS_TUPLETS = {token for token, _ in TUPLET_OF_GROUP.values()}


@dataclass
class Note:
    duration_at: int  # index of its duration token
    length: Fraction
    advances: bool  # False for chord members and grace notes
    tuplet_at: int | None  # index of its tuplet token
    beam_group: int | None = None

    @property
    def marked(self) -> bool:
        return self.tuplet_at is not None


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


def length_at(tokens: list[str], at: int) -> tuple[Fraction, int | None]:
    """A duration token's length with its dots and tuplet ratio, and where the ratio is."""
    base = Fraction(LENGTHS[tokens[at]])
    length, dot = base, Fraction(1, 2)
    for i, token in enumerate(tokens[at + 1:], at + 1):
        ratio = TUPLET.match(token)
        if ratio:
            return length * int(ratio[2]) / int(ratio[1]), i
        if token != "dot":
            break
        length, dot = length + base * dot, dot / 2
    return length, None


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
            length, tuplet_at = length_at(tokens, i)
            if head == "backup":
                voices.append(voice)
                voice = Voice()
            elif head == "forward":
                voice.skipped += length
            else:
                note = Note(i, length, not (chord or grace), tuplet_at, group if depth else None)
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


def two_thirds_events(voice: Voice, tokens: list[str]) -> list[list[Note]]:
    """The voice's notes marked 3in2 or 6in4, each with the rest of its chord."""
    events: list[list[Note]] = []
    for note in voice.notes:
        if note.advances:
            events.append([])
        if events and note.marked and tokens[note.tuplet_at] in TWO_THIRDS_TUPLETS:
            events[-1].append(note)
    return [event for event in events if event]


def runaway_tuplets(voice: Voice, tokens: list[str], bar: Fraction) -> list[int]:
    """Where tuplet tokens go away if that makes the voice fill the bar exactly. A plain note
    is half as long again as its tuplet form."""
    if voice.measure_rest or voice.total() >= bar:
        return []
    events = two_thirds_events(voice, tokens)
    for count in range(1, len(events) + 1):
        for cleared, kept in ((events[:count], events[count:]), (events[-count:], events[:-count])):
            gained = sum((event[0].length / 2 for event in cleared), Fraction(0))
            if voice.total() + gained == bar and len(kept) % 3 == 0:
                return [note.tuplet_at for event in cleared for note in event]
    return []


def imply_tuplets(lmx: str, bar: Fraction | None = None) -> str:
    """The LMX with tuplets the bar's arithmetic implies. bar is the length in force from earlier
    systems, for one that does not print its own time signature."""
    tokens = lmx.split()
    measures = measures_of(tokens)
    bar = written_bar(tokens) or bar or likely_bar(measures)
    if bar is None:
        return lmx
    edits = [(at, token) for measure in measures for voice in measure for at, token in insertions_for(voice, bar)]
    edits += [(at, None) for measure in measures for voice in measure for at in runaway_tuplets(voice, tokens, bar)]
    for at, token in sorted(edits, key=lambda edit: edit[0], reverse=True):
        if token is None:
            del tokens[at]
        else:
            tokens.insert(at, token)
    return " ".join(tokens)


ONE_BEAM_FEWER = {"16th": "eighth", "32nd": "16th", "64th": "32nd"}
NOTE_START = {"rest", "backup", "forward", "measure", "chord", "grace", "grace:slash"}


def overbeamed_group(voice: Voice, tokens: list[str], bar: Fraction) -> list[Note] | None:
    """The voice's one beam group of plain equal notes (no dots, no tuplets) whose doubling
    fills the bar exactly; None if there is no such group, or more than one."""
    if voice.measure_rest or not voice.notes or voice.total() >= bar:
        return None
    fitting = []
    for notes in voice.beam_groups().values():
        timed = [n for n in notes if n.advances]
        names = {tokens[n.duration_at] for n in timed}
        plain = all(not n.marked and n.length == LENGTHS[tokens[n.duration_at]] for n in notes)
        if len(names) == 1 and names <= ONE_BEAM_FEWER.keys() and plain \
                and sum((n.length for n in timed), Fraction(0)) == bar - voice.total():
            fitting.append(notes)
    return fitting[0] if len(fitting) == 1 else None


def last_beam_token(tokens: list[str], duration_at: int) -> int | None:
    """Where the last beam token of a note is: they follow its duration, up to the next note."""
    last = None
    for i in range(duration_at + 1, len(tokens)):
        if tokens[i] in NOTE_START or PITCH.match(tokens[i]):
            break
        if tokens[i].startswith("beam:"):
            last = i
    return last


def extra_beams_dropped(lmx: str, bar: Fraction | None = None) -> str:
    """The LMX with one beam fewer on groups the bar's arithmetic says were read with one too
    many (16ths written as 32nds: Polish 057). bar as for imply_tuplets."""
    tokens = lmx.split()
    measures = measures_of(tokens)
    bar = written_bar(tokens) or bar or likely_bar(measures)
    if bar is None:
        return lmx
    notes = [n for measure in measures for voice in measure for n in (overbeamed_group(voice, tokens, bar) or [])]
    for note in notes:
        tokens[note.duration_at] = ONE_BEAM_FEWER[tokens[note.duration_at]]
    beams = [last_beam_token(tokens, n.duration_at) for n in notes]
    for at in sorted((b for b in beams if b is not None), reverse=True):
        del tokens[at]
    return " ".join(tokens)


def bar_fill(lmx: str) -> float:
    """The share of voices that fill the bar once tuplets are restored: a check of a reading
    that needs no ground truth, since a misread rhythm rarely still adds up."""
    tokens = imply_tuplets(lmx).split()
    measures = measures_of(tokens)
    bar = written_bar(tokens) or likely_bar(measures)
    fills = [v.total() == bar for measure in measures for v in measure if v.notes and not v.measure_rest]
    return sum(fills) / len(fills) if fills else 0.0
