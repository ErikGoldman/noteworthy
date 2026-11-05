"""Ground truth of a Mutopia piece from its MIDI file and LilyPond source.

LilyPond's MIDI holds the written notes at exact rational times, one track per staff, so
the notes and their measures can be read from it directly. It has no bar lines: measures
are laid out from its time signatures, after the pickup that only the source states
(\\partial). That fails for some sources; see midi_matches_print.
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import mido

from score_events import NoteEvent

ONSET_GRID = 96       # denominators of written onsets, in quarters, divide this
MIN_ON_GRID = 0.98


def midi_matches_print(lilypond: str) -> bool:
    """False when the MIDI would not give the printed measures and notes: repeats played
    out, free time, no bar lines, a pickup in mid-piece (which restarts the bars), or
    grace notes, which sound as notes and shift the notes around them."""
    return (not re.search(r"unfoldRepeats|cadenzaOn|BarLine|\\grace|\\acciaccatura|\\appoggiatura|\\slashedGrace", lilypond)
            and len(re.findall(r"\\partial", lilypond)) <= 1)


def onsets_on_grid(events: list[NoteEvent]) -> bool:
    """Whether the onsets are written rhythm. MIDI quirks no source check foresees show up
    as onsets like 25/192 of a quarter."""
    on_grid = sum(ONSET_GRID % e.onset.denominator == 0 for e in events)
    return on_grid >= MIN_ON_GRID * len(events)


def pickup_quarters(lilypond: str) -> Fraction:
    """Length of the piece's pickup, 0 if it has none: \\partial 4, 8., 4*3."""
    partial = re.search(r"\\partial\s*(\d+)(\.?)(?:\*(\d+))?", lilypond)
    if not partial:
        return Fraction(0)
    return Fraction(4, int(partial[1])) * (Fraction(3, 2) if partial[2] else 1) * int(partial[3] or 1)


def _sounding_notes(midi: mido.MidiFile) -> tuple[list[tuple[Fraction, int, Fraction, int]], dict[Fraction, Fraction]]:
    """(start, pitch, duration, track) of each note, and measure length by the time it
    takes effect; all in quarters."""
    notes, measure_length = [], {Fraction(0): Fraction(4)}
    for track_index, track in enumerate(midi.tracks):
        ticks, started = 0, {}
        for message in track:
            ticks += message.time
            now = Fraction(ticks, midi.ticks_per_beat)
            if message.type == "time_signature":
                measure_length[now] = Fraction(4 * message.numerator, message.denominator)
            elif message.type == "note_on" and message.velocity:
                started.setdefault(message.note, []).append(now)
            elif message.type in ("note_on", "note_off") and started.get(message.note):
                start = started[message.note].pop(0)
                notes.append((start, message.note, now - start, track_index))
    return notes, measure_length


def _measure_starts(measure_length: dict[Fraction, Fraction], pickup: Fraction, end: Fraction) -> list[Fraction]:
    changes = sorted(measure_length.items())
    length_at = lambda time: [length for start, length in changes if start <= time][-1]
    starts, position = [Fraction(0)], pickup or length_at(Fraction(0))
    while position < end:
        starts.append(position)
        position += length_at(position)
    return starts


def midi_events(path: Path, pickup: Fraction) -> list[NoteEvent]:
    notes, measure_length = _sounding_notes(mido.MidiFile(path))
    starts = _measure_starts(measure_length, pickup, max((s + d for s, _, d, _ in notes), default=Fraction(0)))
    events = []
    for start, pitch, duration, track in notes:
        measure = max(i for i, s in enumerate(starts) if s <= start)
        events.append(NoteEvent(measure, start - starts[measure], pitch, duration, track))
    return sorted(events)
