"""Restores tuplets a transcription left out, from the arithmetic of the bar.

Engravers stop printing the number once a passage has settled into triplets or sextuplets,
and then nothing but the bar's length tells them from plain notes. The sequence model reads
such notes perfectly and writes them plain, so their voice overfills its measure. Where
reading a voice's beam groups of 3, 5 and 6 as tuplets fills the measure exactly, this does so.

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
from math import gcd

LENGTHS = {"maxima": 32, "long": 16, "breve": 8, "whole": 4, "half": 2, "quarter": 1, "eighth": Fraction(1, 2),
           "16th": Fraction(1, 4), "32nd": Fraction(1, 8), "64th": Fraction(1, 16), "128th": Fraction(1, 32),
           "256th": Fraction(1, 64), "512th": Fraction(1, 128), "1024th": Fraction(1, 256)}
TUPLET = re.compile(r"(\d+)in(\d+)$")
PITCH = re.compile(r"[A-G]\d$")
TUPLET_OF_GROUP = {3: ("3in2", Fraction(2, 3)), 5: ("5in4", Fraction(4, 5)), 6: ("6in4", Fraction(2, 3))}


@dataclass
class Note:
    duration_at: int  # index of its duration token
    length: Fraction
    advances: bool  # False for chord members and grace notes
    tuplet_at: int | None  # index of its tuplet token
    beam_group: int | None = None
    inner_group: int | None = None  # under the group's second beam
    grace: bool = False

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
        return self.grouped_by(lambda note: note.beam_group)

    def inner_groups(self) -> dict[int, list[Note]]:
        return self.grouped_by(lambda note: note.inner_group)

    def grouped_by(self, group_of) -> dict[int, list[Note]]:
        groups: dict[int, list[Note]] = {}
        for note in self.notes:
            if group_of(note) is not None:
                groups.setdefault(group_of(note), []).append(note)
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
    inner, next_inner = None, 0
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
                note = Note(i, length, not (chord or grace), tuplet_at, group if depth else None,
                            inner if depth > 1 else None, grace)
                voice.notes.append(note)
            head, chord, grace = None, False, False
        elif token == "beam:begin":
            if depth == 0:
                group, next_group = next_group, next_group + 1
                if note:
                    note.beam_group = group
            elif depth == 1:
                inner, next_inner = next_inner, next_inner + 1
                if note:
                    note.inner_group = inner
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


def as_tuplet(notes: list[Note]) -> tuple[list[Note], str, Fraction] | None:
    """An unmarked group of 3, 5 or 6 equal notes, with its tuplet token and ratio."""
    timed = [n for n in notes if n.advances]
    if len(timed) in TUPLET_OF_GROUP and len({n.length for n in timed}) == 1 and not any(n.marked for n in notes):
        return notes, *TUPLET_OF_GROUP[len(timed)]
    return None


def tuplet_candidates(voice: Voice) -> list[tuple[list[Note], str, Fraction]]:
    """Beam groups that could be tuplets, or failing that, groups under their second beam: an
    eighth then three 16ths a triplet (Polish 055)."""
    inner_groups = voice.inner_groups()
    found = []
    for notes in voice.beam_groups().values():
        whole = as_tuplet(notes)
        inner = [as_tuplet(group) for group in inner_groups.values() if group[0] in notes]
        found += [whole] if whole else [c for c in inner if c]
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


def marked_events(voice: Voice) -> list[list[Note]]:
    """The voice's notes marked as tuplets, each with the rest of its chord."""
    events: list[list[Note]] = []
    for note in voice.notes:
        if note.advances:
            events.append([])
        if events and note.marked:
            events[-1].append(note)
    return [event for event in events if event]


def ratio_of(event: list[Note], tokens: list[str]) -> tuple[int, int]:
    """The event's tuplet as (actual, normal): 3in2 is (3, 2)."""
    actual, normal = TUPLET.match(tokens[event[0].tuplet_at]).groups()
    return int(actual), int(normal)


def gain_if_plain(event: list[Note], tokens: list[str]) -> Fraction:
    actual, normal = ratio_of(event, tokens)
    return event[0].length * (actual - normal) / normal


def whole_tuplets(events: list[list[Note]], tokens: list[str]) -> bool:
    """Each kind of tuplet comes in whole groups: threes for 3in2 and 6in4 (a sextuplet is two
    triplets), fives for 5in4."""
    counts = Counter(ratio_of(event, tokens) for event in events)
    return all(count % (actual // gcd(actual, normal)) == 0 for (actual, normal), count in counts.items())


def runaway_tuplets(voice: Voice, tokens: list[str], bar: Fraction) -> list[int]:
    """Where tuplet tokens go away if that makes the voice fill the bar exactly and leaves whole
    tuplets. The model marks plain notes beside triplets as triplets, and four plain 16ths as a
    quintuplet (Polish 018, 026); the short voice then shifts every note of the other staff."""
    if voice.measure_rest or voice.total() >= bar:
        return []
    events = marked_events(voice)
    for count in range(1, len(events) + 1):
        for cleared, kept in ((events[:count], events[count:]), (events[-count:], events[:-count])):
            gained = sum((gain_if_plain(event, tokens) for event in cleared), Fraction(0))
            if voice.total() + gained == bar and whole_tuplets(kept, tokens):
                return [at for event in cleared for note in event for at in tuplet_tokens(tokens, note)]
    return []


def tuplet_tokens(tokens: list[str], note: Note) -> list[int]:
    """The note's ratio token and its bracket's start or stop, which go with it."""
    brackets = [i for i in following_tokens(tokens, note.duration_at) if tokens[i].startswith("tuplet:")]
    return [note.tuplet_at, *brackets]


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
    """The voice's one group of equal notes, a beam group or the notes under one's second beam,
    whose doubling fills the bar exactly; None if there is no such group, or more than one. No
    dots; a tuplet only whole (042: triplet 32nds for 16ths, beamed with an eighth)."""
    if voice.measure_rest or not voice.notes or voice.total() >= bar:
        return None
    groups = {tuple(n.duration_at for n in notes): notes
              for notes in [*voice.beam_groups().values(), *voice.inner_groups().values()]}
    fitting = [notes for notes in groups.values() if doubling_fills(notes, tokens, bar - voice.total())]
    return fitting[0] if len(fitting) == 1 else None


def doubling_fills(notes: list[Note], tokens: list[str], missing: Fraction) -> bool:
    timed = [n for n in notes if n.advances]
    # Every note, graces and chord members too, as every one is redrawn: a grace eighth read
    # into a 16th's beam (Polish 065, seq514) has no beam to lose.
    names = {tokens[n.duration_at] for n in notes}
    undotted = all(tokens[n.duration_at + 1] != "dot" for n in notes if n.duration_at + 1 < len(tokens))
    alike = len({tokens[n.tuplet_at] if n.marked else None for n in notes}) == 1
    return len(names) == 1 and names <= ONE_BEAM_FEWER.keys() and undotted and alike \
        and sum((n.length for n in timed), Fraction(0)) == missing


def following_tokens(tokens: list[str], duration_at: int) -> range:
    """Where a note's marks are: they follow its duration, up to the next note."""
    end = next((i for i in range(duration_at + 1, len(tokens)) if tokens[i] in NOTE_START or PITCH.match(tokens[i])),
               len(tokens))
    return range(duration_at + 1, end)


def last_beam_token(tokens: list[str], duration_at: int) -> int | None:
    return max((i for i in following_tokens(tokens, duration_at) if tokens[i].startswith("beam:")), default=None)


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


def events_of(voice: Voice) -> list[list[Note]]:
    """Each advancing note with the chord notes after it; grace notes belong to none."""
    events: list[list[Note]] = []
    for note in voice.notes:
        if note.advances:
            events.append([note])
        elif events and not note.grace:
            events[-1].append(note)
    return events


def dots(tokens: list[str], note: Note) -> int:
    count = 0
    while note.duration_at + 1 + count < len(tokens) and tokens[note.duration_at + 1 + count] == "dot":
        count += 1
    return count


def dot_read_in(voice: Voice, tokens: list[str], bar: Fraction) -> list[Note] | None:
    """The voice's one dotted event whose dot, taken away, makes it fill the bar exactly; None
    if no event or several would. Only an overfull voice: a short one may be a second voice
    that stops before the barline, and dotting it to fill made OLiMPiC crops worse."""
    if voice.measure_rest or not voice.notes or voice.total() <= bar:
        return None
    fitting = [event for event in events_of(voice) if not event[0].marked and dots(tokens, event[0]) == 1
               and voice.total() - LENGTHS[tokens[event[0].duration_at]] / 2 == bar]
    return fitting[0] if len(fitting) == 1 else None


def filled(measure: list[Voice], bar: Fraction) -> bool:
    """Whether some voice fills the bar: a short bar before a repeat or ending is short in every
    voice (069 bar 12), and a page with no time signature (096) has only a guessed bar."""
    return any(v.total() == bar for v in measure if v.notes and not v.measure_rest)


def dots_mended(lmx: str, bar: Fraction | None = None) -> str:
    """The LMX with one dot taken away in voices it makes fill the bar exactly, in measures where
    another voice already does: a dot read where there is none (Polish 030 bar 21). bar as for
    imply_tuplets."""
    tokens = lmx.split()
    measures = measures_of(tokens)
    bar = written_bar(tokens) or bar or likely_bar(measures)
    if bar is None:
        return lmx
    events = (dot_read_in(voice, tokens, bar) for measure in measures if filled(measure, bar) for voice in measure)
    for at in sorted((n.duration_at + 1 for event in events if event for n in event), reverse=True):
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
