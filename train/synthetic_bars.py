"""Synthetic piano pieces, each holding one thing the model reads worst, set in otherwise plain
music: a passage of tuplets, two voices on a staff, or dense chords (TODO, "Where the cost is":
tuplets, notes at another onset, later voices, chord tones). Some hold none, so that plain bars
stay plain. Features never overlap for now: a piece with tuplets has one voice a staff and light
chords.

usage: synthetic_bars.py --out DIR --scores N [--measures M]

The music: a melody in the right hand, mostly stepwise and on chord tones at the beat, over a
diatonic progression, plain rhythms and a longer value closing each phrase; the left hand in one
accompaniment pattern a piece (block chords, broken chords, bass and chord, held basses). The
feature is then drawn into it:
  - tuplets: one kind a piece (3:2 eighths most, quarter-eighth, 16th triplets, 6:4, 5:4, quarter
    triplets over a half; duplets and quadruplets in compound time), as passages of a few beats in
    the melody, or the accompaniment's broken chords in triplets;
  - two voices: an inner line of longer values with rests under the melody, or a held bass under
    the accompaniment, stems apart;
  - dense chords: three- and four-note chords in both hands.
A piece is M bars (default 8); the crop builders cut it into systems of 2 to 6, all in its feature.

Writes DIR/scores/<index>.musicxml, DIR/scores.txt and DIR/features.tsv (each score's feature), for
the crop builders like any score list: build_seq_dataset.py --scores DIR/scores.txt,
build_musescore_crops.py likewise. Score k is drawn from random.Random(f"synthetic-bars:{k}")
alone, so any subset or order of scores reproduces the same files. A twentieth are held out by
omr.held_out, as for any score; the builders skip them.
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

DIVISIONS = 840  # per quarter: 16ths in triplets and quintuplets come out whole
QUARTERS = {"whole": Fraction(4), "half": Fraction(2), "quarter": Fraction(1), "eighth": Fraction(1, 2),
            "16th": Fraction(1, 4)}
SHORT = {"q": "quarter", "h": "half", "w": "whole", "8": "eighth", "16": "16th"}
BEAM_LEVELS = {"eighth": 1, "16th": 2}
STEPS = "CDEFGAB"
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
ACCIDENTAL_OF_ALTER = {1: "sharp", -1: "flat", 0: "natural"}

PLAIN, TUPLETS, TWO_VOICES, DENSE_CHORDS = "plain", "tuplets", "two voices", "dense chords"
FEATURES = {PLAIN: 0.2, TUPLETS: 0.3, TWO_VOICES: 0.25, DENSE_CHORDS: 0.25}
METERS = {(4, 4): 35, (3, 4): 25, (2, 4): 12, (6, 8): 15, (2, 2): 5, (9, 8): 4, (12, 8): 4}
STAFF_CLEFS = {1: (("G", 2), ("F", 4), 0.05), 2: (("F", 4), ("G", 2), 0.1)}  # usual, other, other's share

# A beat's rhythm, as "RATIO? values" (a value is SHORT[..], "." dots it); plain melodies.
MELODY_BEAT = {"q": 6, "8 8": 6, "8. 16": 2, "16 16 8": 1, "8 16 16": 1, "16 16 16 16": 1}
MELODY_COMPOUND_BEAT = {"q.": 4, "q 8": 4, "8 8 8": 4, "8. 16 8": 1}
LONG_NOTE = 0.2  # of a melody's beats that hold over the next one
CADENCE_EVERY = 4  # bars a phrase; its last bar closes on one long value
TUPLETS_SIMPLE = {"3:2 8 8 8": 6, "3:2 q 8": 2, "3:2 16 16 16 | 8": 1, "6:4 16 16 16 16 16 16": 2,
                  "5:4 16 16 16 16 16": 1}
TUPLETS_OVER_HALF = "3:2 q q q"
TUPLETS_COMPOUND = {"2:3 8 8": 3, "4:3 8 8 8 8": 1}
TUPLET_BAR = 0.7          # bars of a tuplet piece's melody holding a passage
TUPLET_ACCOMPANIMENT = 0.3  # tuplet pieces whose accompaniment's broken chords are triplets instead
ACCOMPANIMENTS = {"block": 3, "broken": 4, "bass and chord": 3, "held bass": 1}
PROGRESSION = {0: (3, 4, 5), 1: (4, 6), 2: (5, 3), 3: (4, 0, 1), 4: (0, 5), 5: (1, 3, 4), 6: (0,)}  # degree: next
LIGHT_CHORDS = 0.1        # melody notes made chords outside the dense-chord pieces
DENSE_CHORDS_SHARE = 0.7
INNER_REST = 0.25
CHROMATIC = 0.03


@dataclass
class Event:
    kind: str                       # quarter, eighth, ...
    dots: int = 0
    ratio: tuple[int, int] | None = None
    tuplet: str | None = None       # "start" or "stop" on a group's first or last event
    rest: bool = False
    pitches: list[int] = field(default_factory=list)   # diatonic steps, C0 = 0, lowest first
    alters: list[int] = field(default_factory=list)
    shown: list[str | None] = field(default_factory=list)   # the accidental drawn on each
    stem: str | None = None
    beams: list[tuple[int, str]] = field(default_factory=list)

    @property
    def quarters(self) -> Fraction:
        value = QUARTERS[self.kind] * (Fraction(3, 2) if self.dots else 1)
        return value * Fraction(self.ratio[1], self.ratio[0]) if self.ratio else value


Beat = list[list[Event]]  # a beat's beam groups


def rhythm(text: str) -> Beat:
    """A rhythm's groups of events: a tuplet group, or plain values."""
    groups = []
    for part in text.split("|"):
        words = part.split()
        ratio = tuple(map(int, words.pop(0).split(":"))) if ":" in words[0] else None
        events = [Event(SHORT[w.rstrip(".")], dots=int(w.endswith(".")), ratio=ratio) for w in words]
        if ratio:
            events[0].tuplet, events[-1].tuplet = "start", "stop"
        groups.append(events)
    return groups


def pick(rng: random.Random, weighted: dict):
    return rng.choices(list(weighted), weights=list(weighted.values()))[0]


@dataclass(frozen=True)
class Meter:
    beats: int
    beat_type: int

    @property
    def compound(self) -> bool:
        return self.beat_type == 8 and self.beats % 3 == 0

    @property
    def counted_beats(self) -> int:
        """Dotted quarters in compound meters, else quarters."""
        return self.beats // 3 if self.compound else self.beats * 4 // self.beat_type

    @property
    def quarters(self) -> Fraction:
        return Fraction(4 * self.beats, self.beat_type)

    @property
    def beat(self) -> str:
        return "q." if self.compound else "q"


@dataclass(frozen=True)
class Piece:
    """One score's choices."""
    meter: Meter
    fifths: int
    clefs: dict[int, tuple[str, int]]
    feature: str
    accompaniment: str
    tuplet: str | None          # the piece's tuplet rhythm, for its melody's passages
    triplet_accompaniment: bool
    two_voice_staff: int | None


def drawn_piece(rng: random.Random) -> Piece:
    meter = Meter(*pick(rng, METERS))
    feature = pick(rng, FEATURES)
    tuplet = triplet_accompaniment = None
    if feature == TUPLETS:
        triplet_accompaniment = not meter.compound and rng.random() < TUPLET_ACCOMPANIMENT
        if not triplet_accompaniment:
            kinds = TUPLETS_COMPOUND if meter.compound else TUPLETS_SIMPLE | ({TUPLETS_OVER_HALF: 1} if meter.counted_beats % 2 == 0 else {})
            tuplet = pick(rng, kinds)
    two_voice_staff = rng.choice([1, 1, 2]) if feature == TWO_VOICES else None
    if triplet_accompaniment:
        accompaniment = "broken"
    elif feature == DENSE_CHORDS or two_voice_staff == 2:  # chords to thicken; a held bass of its own
        accompaniment = rng.choice(["block", "bass and chord"])
    else:
        accompaniment = pick(rng, ACCOMPANIMENTS)
    return Piece(meter, rng.randint(-6, 6),
                 {staff: (other if rng.random() < share else usual) for staff, (usual, other, share) in STAFF_CLEFS.items()},
                 feature, accompaniment, tuplet, bool(triplet_accompaniment), two_voice_staff)


# ---- harmony and pitch ------------------------------------------------------------------------

def tonic(fifths: int) -> int:
    """The major key's tonic, as a step letter's index: each fifth up is four steps."""
    return (4 * fifths) % 7


def progression(bars: int, rng: random.Random) -> list[int]:
    """Scale degrees (0 = I), one a bar, opening on I and closing each phrase on V or I."""
    degrees = [0]
    while len(degrees) < bars:
        closing = (len(degrees) + 1) % CADENCE_EVERY == 0
        degrees.append(rng.choice((4, 0)) if closing else rng.choice(PROGRESSION[degrees[-1]]))
    return degrees


def chord_tones(degree: int, fifths: int) -> set[int]:
    root = tonic(fifths) + degree
    return {(root + k) % 7 for k in (0, 2, 4)}


def nearest(step: int, letters: set[int], low: int, high: int) -> int:
    candidates = [s for s in range(low, high + 1) if s % 7 in letters]
    return min(candidates, key=lambda s: (abs(s - step), s))


def chord_below(top: int, letters: set[int], size: int) -> list[int]:
    """size chord tones from top downwards, lowest first."""
    tones, step = [top], top - 1
    while len(tones) < size:
        if step % 7 in letters:
            tones.append(step)
        step -= 1
    return sorted(tones)


def chord_above(bass: int, letters: set[int], size: int) -> list[int]:
    tones, step = [bass], bass + 1
    while len(tones) < size:
        if step % 7 in letters:
            tones.append(step)
        step += 1
    return tones


@dataclass(frozen=True)
class Range:
    low: int
    high: int


def middle_line(clef: tuple[str, int]) -> int:
    """The step of the staff's middle line: B4 for a treble clef, D3 for a bass clef."""
    return 4 * 7 + 6 if clef == ("G", 2) else 3 * 7 + 1


# ---- the hands --------------------------------------------------------------------------------

def notes(beats: list[Beat]) -> list[Event]:
    return [e for beat in beats for group in beat for e in group if not e.rest]


def melody_rhythm(piece: Piece, bar: int, rng: random.Random) -> list[Beat]:
    """A bar of the melody, beat by beat; a phrase's last bar one long value; a tuplet piece's
    bars a passage of its tuplet over some beats in a row."""
    meter, n = piece.meter, piece.meter.counted_beats
    if (bar + 1) % CADENCE_EVERY == 0:
        return [long_value(meter, n)]
    passage = range(0)
    if piece.tuplet and rng.random() < TUPLET_BAR:
        span = 2 if piece.tuplet == TUPLETS_OVER_HALF else 1
        length = rng.randint(1, n // span) * span
        first = rng.randrange(0, n - length + 1, span)
        passage = range(first, first + length)
    beats, beat = [], 0
    while beat < n:
        if beat in passage:
            beats.append(rhythm(piece.tuplet))
            beat += 2 if piece.tuplet == TUPLETS_OVER_HALF else 1
        elif beat + 1 < n and beat + 1 not in passage and rng.random() < LONG_NOTE:
            beats.append(long_value(meter, 2))
            beat += 2
        else:
            beats.append(rhythm(pick(rng, MELODY_COMPOUND_BEAT if meter.compound else MELODY_BEAT)))
            beat += 1
    return beats


def long_value(meter: Meter, beats: int) -> Beat:
    """A value lasting `beats` beats, or two where no one value does (3 or 4 dotted quarters)."""
    if meter.compound:
        return rhythm({1: "q.", 2: "h.", 3: "h. | q.", 4: "h. | h."}[beats])
    return rhythm({1: "q", 2: "h", 3: "h.", 4: "w"}[beats])


def melody_pitches(beats: list[Beat], degree: int, piece: Piece, start: int, voice: Range,
                   rng: random.Random) -> int:
    """On each beat a chord tone near the last note, between beats steps; returns the last step."""
    letters, step = chord_tones(degree, piece.fifths), start
    for beat in beats:
        for k, event in enumerate(e for group in beat for e in group):
            if k == 0:
                step = nearest(step + rng.choice([-2, -1, 0, 1, 2]), letters, voice.low, voice.high)
            else:
                step = min(voice.high, max(voice.low, step + rng.choice([-2, -1, -1, 1, 1, 2])))
            event.pitches = [step]
    return step


def accompaniment(piece: Piece, degree: int, hand: Range, rng: random.Random) -> list[Beat]:
    """A bar of the piece's accompaniment over the chord (of four notes in a dense-chord piece)."""
    letters, meter = chord_tones(degree, piece.fifths), piece.meter
    bass = nearest(hand.low + 3, {(tonic(piece.fifths) + degree) % 7}, hand.low, hand.high)
    size = 4 if piece.feature == DENSE_CHORDS else 3
    chord = chord_above(nearest(bass + 2, letters, bass + 1, hand.high), letters, size)
    beats: list[Beat] = []
    for beat in range(meter.counted_beats):
        if piece.accompaniment == "block":
            beats.append(rhythm(meter.beat))
            beats[-1][0][0].pitches = [bass] + chord if beat == 0 else chord
        elif piece.accompaniment == "broken":
            figure = "3:2 8 8 8" if piece.triplet_accompaniment else ("8 8 8" if meter.compound else "8 8")
            beats.append(rhythm(figure))
            line = [bass, chord[1], chord[2], chord[1]]
            for k, event in enumerate(beats[-1][0]):
                event.pitches = [line[k % len(line)] if beat == 0 or k else chord[0]]
        elif piece.accompaniment == "bass and chord":
            beats.append(rhythm(meter.beat))
            beats[-1][0][0].pitches = [bass] if beat % 2 == 0 else chord
        else:
            beats.append(long_value(meter, meter.counted_beats))
            for group in beats[-1]:
                group[0].pitches = [bass, bass + 7]
            break
    return beats


def inner_line(piece: Piece, degree: int, voice: Range, rng: random.Random) -> list[Beat]:
    """A second voice of longer values under the melody, chord tones, with rests."""
    beats, letters = [], chord_tones(degree, piece.fifths)
    step = (voice.low + voice.high) // 2
    n, beat = piece.meter.counted_beats, 0
    while beat < n:
        length = 2 if beat + 1 < n and rng.random() < 0.5 else 1
        beats.append(long_value(piece.meter, length))
        for group in beats[-1]:
            for event in group:
                event.rest = rng.random() < INNER_REST
                step = nearest(step + rng.choice([-1, 0, 1]), letters, voice.low, voice.high)
                event.pitches = [] if event.rest else [step]
        beat += length
    return beats


def thickened(beats: list[Beat], degree: int, piece: Piece, share: float, sizes: tuple[int, ...],
              rng: random.Random) -> None:
    letters = chord_tones(degree, piece.fifths)
    for event in notes(beats):
        if rng.random() < share:
            event.pitches = chord_below(event.pitches[-1], letters, rng.choice(sizes))


# ---- notation ---------------------------------------------------------------------------------

def key_alters(fifths: int) -> dict[int, int]:
    order = SHARP_ORDER if fifths > 0 else FLAT_ORDER
    return {STEPS.index(s): (1 if fifths > 0 else -1) for s in order[:abs(fifths)]}


def with_accidentals(voices: list[list[Beat]], fifths: int, chromatic: set[int], rng: random.Random) -> None:
    """Each note's alter (the key's, or off it for the chromatic ones) and the accidental an engraver
    draws: where it differs from what is in force on that line in the bar, the staff's voices in time
    order."""
    key = key_alters(fifths)
    timed = []
    for beats in voices:
        at = Fraction(0)
        for event in (e for beat in beats for group in beat for e in group):
            timed.append((at, event))
            at += event.quarters
    in_force: dict[int, int] = {}
    for _, event in sorted(timed, key=lambda pair: pair[0]):
        event.alters, event.shown = [], []
        for k, step in enumerate(event.pitches):
            alter = key.get(step % 7, 0)
            if id(event) in chromatic and k == len(event.pitches) - 1:
                alter = rng.choice([a for a in (-1, 1) if a != alter] or [0])
            shown = ACCIDENTAL_OF_ALTER[alter] if in_force.get(step, key.get(step % 7, 0)) != alter else None
            in_force[step] = alter
            event.alters.append(alter)
            event.shown.append(shown)


def with_stems_and_beams(beats: list[Beat], stem: str | None, middle: int) -> None:
    """One stem direction a beam group (the voice's own if the staff has two, else away from the
    group's middle), and beams over its runs of eighths and shorter, broken by rests."""
    for group in (g for beat in beats for g in beat):
        sounding = [e for e in group if not e.rest]
        mean = sum(sum(e.pitches) / len(e.pitches) for e in sounding) / max(1, len(sounding))
        direction = stem or ("down" if mean >= middle else "up")
        for event in sounding:
            event.stem = direction if event.kind != "whole" else None
        for run in beamable_runs(group):
            set_beams(run)


def beamable_runs(group: list[Event]) -> list[list[Event]]:
    runs, current = [], []
    for event in group:
        if not event.rest and event.kind in BEAM_LEVELS:
            current.append(event)
        else:
            runs, current = runs + [current], []
    return [r for r in runs + [current] if len(r) > 1]


def set_beams(run: list[Event]) -> None:
    for level in (1, 2):
        has = [BEAM_LEVELS[e.kind] >= level for e in run]
        for k, event in enumerate(run):
            if not has[k]:
                continue
            before, after = k > 0 and has[k - 1], k + 1 < len(run) and has[k + 1]
            if level == 1 or before or after:
                kind = "continue" if before and after else "begin" if after else "end"
            else:
                kind = "forward hook" if k == 0 else "backward hook"
            event.beams.append((level, kind))


def sub(parent: ET.Element, tag: str, text: str | None = None, **attributes) -> ET.Element:
    element = ET.SubElement(parent, tag, attributes)
    if text is not None:
        element.text = text
    return element


def write_event(measure: ET.Element, event: Event, staff: int, voice: str) -> None:
    duration = str(int(event.quarters * DIVISIONS))
    for k, step in enumerate(event.pitches or [None]):
        note = sub(measure, "note")
        if k:
            sub(note, "chord")
        if step is None:
            sub(note, "rest")
        else:
            pitch = sub(note, "pitch")
            sub(pitch, "step", STEPS[step % 7])
            if event.alters[k]:
                sub(pitch, "alter", str(event.alters[k]))
            sub(pitch, "octave", str(step // 7))
        sub(note, "duration", duration)
        sub(note, "voice", voice)
        sub(note, "type", event.kind)
        for _ in range(event.dots):
            sub(note, "dot")
        if step is not None and event.shown[k]:
            sub(note, "accidental", event.shown[k])
        if event.ratio:
            modification = sub(note, "time-modification")
            sub(modification, "actual-notes", str(event.ratio[0]))
            sub(modification, "normal-notes", str(event.ratio[1]))
        if k == 0 and event.stem:
            sub(note, "stem", event.stem)
        sub(note, "staff", str(staff))
        for level, kind in event.beams if k == 0 else []:
            sub(note, "beam", kind, number=str(level))
        if k == 0 and event.tuplet:
            sub(sub(note, "notations"), "tuplet", type=event.tuplet, number="1")


def attributes(measure: ET.Element, piece: Piece) -> None:
    element = sub(measure, "attributes")
    sub(element, "divisions", str(DIVISIONS))
    sub(sub(element, "key"), "fifths", str(piece.fifths))
    time = sub(element, "time")
    sub(time, "beats", str(piece.meter.beats))
    sub(time, "beat-type", str(piece.meter.beat_type))
    sub(element, "staves", "2")
    for staff, (sign, line) in piece.clefs.items():
        clef = sub(element, "clef", number=str(staff))
        sub(clef, "sign", sign)
        sub(clef, "line", str(line))


# ---- a piece ----------------------------------------------------------------------------------

def staff_voices(piece: Piece, bar: int, degree: int, melody_step: int, rng: random.Random) -> tuple[dict[int, list[list[Beat]]], int]:
    """Each staff's voices for one bar (upper voice first), and where the melody ends."""
    upper, lower = middle_line(piece.clefs[1]), middle_line(piece.clefs[2])
    # A staff of two voices keeps them apart: the melody above the inner line, the chords above the bass.
    melody_range = Range(upper, upper + 9) if piece.two_voice_staff == 1 else Range(upper - 4, upper + 9)
    hand = Range(lower - 3, lower + 5) if piece.two_voice_staff == 2 else Range(lower - 9, lower + 4)
    melody = melody_rhythm(piece, bar, rng)
    melody_step = melody_pitches(melody, degree, piece, melody_step, melody_range, rng)
    left = accompaniment(piece, degree, hand, rng)
    if piece.feature == DENSE_CHORDS:
        thickened(melody, degree, piece, DENSE_CHORDS_SHARE, (3, 4), rng)
    elif piece.feature != TWO_VOICES:
        thickened(melody, degree, piece, LIGHT_CHORDS, (2,), rng)
    staves = {1: [melody], 2: [left]}
    if piece.two_voice_staff == 1:
        staves[1].append(inner_line(piece, degree, Range(upper - 7, upper - 1), rng))
    elif piece.two_voice_staff == 2:
        staves[2] = [left, inner_line(piece, degree, Range(lower - 10, lower - 4), rng)]
    return staves, melody_step


def synthetic_score(index: int, measures: int) -> tuple[ET.Element, str]:
    """The score and its feature."""
    rng = random.Random(f"synthetic-bars:{index}")
    piece = drawn_piece(rng)
    score = ET.Element("score-partwise", version="3.1")
    sub(sub(sub(score, "part-list"), "score-part", id="P1"), "part-name", "Piano")
    part = sub(score, "part", id="P1")
    melody_step = middle_line(piece.clefs[1])
    for bar, degree in enumerate(progression(measures, rng)):
        measure = sub(part, "measure", number=str(bar + 1))
        if bar == 0:
            attributes(measure, piece)
        staves, melody_step = staff_voices(piece, bar, degree, melody_step, rng)
        write_bar(measure, piece, staves, rng)
    return score, piece.feature


def write_bar(measure: ET.Element, piece: Piece, staves: dict[int, list[list[Beat]]], rng: random.Random) -> None:
    """Each staff's voices, a backup before every voice but the bar's first."""
    first = True
    for staff, voices in staves.items():
        middle = middle_line(piece.clefs[staff])
        # Passing chromatic notes in the melody only: in an accompaniment's repeated tones they
        # alternate with the key's (E flat, E, E flat) as no one writes.
        chromatic = {id(e) for e in notes(voices[0]) if rng.random() < CHROMATIC} if staff == 1 else set()
        with_accidentals(voices, piece.fifths, chromatic, rng)
        for k, beats in enumerate(voices):
            with_stems_and_beams(beats, (("up", "down")[k] if len(voices) > 1 else None), middle)
            if not first:
                sub(sub(measure, "backup"), "duration", str(int(piece.meter.quarters * DIVISIONS)))
            for event in (e for beat in beats for group in beat for e in group):
                write_event(measure, event, staff, str((1 if staff == 1 else 5) + k))
            first = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores", type=int, required=True)
    ap.add_argument("--measures", type=int, default=8)
    args = ap.parse_args()
    scores = args.out / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    paths, features = [], []
    for index in range(args.scores):
        path = scores / f"{index:06d}.musicxml"
        score, feature = synthetic_score(index, args.measures)
        ET.ElementTree(score).write(path, encoding="UTF-8", xml_declaration=True)
        paths.append(str(path.resolve()))
        features.append(f"{path.name}\t{feature}")
    (args.out / "scores.txt").write_text("\n".join(paths) + "\n")
    (args.out / "features.tsv").write_text("score\tfeature\n" + "\n".join(features) + "\n")
    print(f"{len(paths)} scores in {scores}")


if __name__ == "__main__":
    main()
