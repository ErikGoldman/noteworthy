"""Reduce scores to comparable note events.

Both the hand-written ground-truth format (*.gt, see sleepless.gt) and MusicXML
are reduced to NoteEvents: (measure index, staff, onset, MIDI pitch, duration).
Durations and onsets are in quarter notes, as exact Fractions.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
GT_PITCH = re.compile(r"^([A-G])(#|##|b|bb)?(-?\d)$")
GT_EVENT = re.compile(r"^(16|32|1|2|4|8)(\.*)(.+?)(~?)$")
ALTER = {None: 0, "#": 1, "##": 2, "b": -1, "bb": -2}


@dataclass(frozen=True, order=True)
class NoteEvent:
    measure: int
    onset: Fraction
    midi: int
    duration: Fraction
    staff: int = field(compare=False)


@dataclass
class GtNote:
    """A ground-truth event as written, kept for MusicXML export."""
    duration: Fraction
    dots: int
    base: int                   # 1, 2, 4, ... (whole, half, quarter, ...)
    pitches: list[tuple[str, int, int]]  # (step, alter, octave); empty = rest/space
    is_space: bool
    tie: bool


@dataclass
class GtScore:
    header: dict[str, list[str]]
    # measures[m][staff] -> list of voices -> list of GtNote
    measures: dict[int, dict[int, list[list[GtNote]]]]

    def events(self) -> list[NoteEvent]:
        out = []
        for m, staves in self.measures.items():
            for staff, voices in staves.items():
                for voice in voices:
                    t = Fraction(0)
                    for note in voice:
                        for step, alter, octave in note.pitches:
                            out.append(NoteEvent(m, t, midi_of(step, alter, octave), note.duration, staff))
                        t += note.duration
        return sorted(out)


def midi_of(step: str, alter: int, octave: int) -> int:
    return 12 * (octave + 1) + STEP_SEMITONES[step] + alter


def _parse_gt_event(token: str) -> GtNote:
    m = GT_EVENT.match(token)
    if not m:
        raise ValueError(f"bad event {token!r}")
    base, dots, what, tie = int(m.group(1)), len(m.group(2)), m.group(3), bool(m.group(4))
    duration = Fraction(4, base) * (2 - Fraction(1, 2 ** dots))
    if what in ("r", "s"):
        return GtNote(duration, dots, base, [], what == "s", tie)
    pitches = []
    for p in what.split("+"):
        pm = GT_PITCH.match(p)
        if not pm:
            raise ValueError(f"bad pitch {p!r} in {token!r}")
        pitches.append((pm.group(1), ALTER[pm.group(2)], int(pm.group(3))))
    return GtNote(duration, dots, base, pitches, False, tie)


def parse_gt(path: str | Path) -> GtScore:
    header: dict[str, list[str]] = {}
    measures: dict[int, dict[int, list[list[GtNote]]]] = {}
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("header "):
            key, *vals = line.split()[1:]
            header.setdefault(key, []).append(" ".join(vals))
            continue
        try:
            loc, body = line.split(":", 1)
            measure, staff = (int(v) for v in loc.split())
            voices = [[_parse_gt_event(tok) for tok in v.split()] for v in body.split("|")]
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
        measures.setdefault(measure, {})[staff] = voices
    return GtScore(header, dict(sorted(measures.items())))


def _read_musicxml_root(path: Path) -> ET.Element:
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF"))
            return ET.fromstring(z.read(name))
    return ET.parse(path).getroot()


def parse_musicxml(path: str | Path) -> list[NoteEvent]:
    """Events from a partwise MusicXML file. Parts are numbered as consecutive staves."""
    root = _read_musicxml_root(Path(path))
    events = []
    staff_offset = 0
    for part in root.iter("part"):
        divisions = 1
        max_staff = 1
        for m_index, measure in enumerate(part.iter("measure")):
            t = Fraction(0)
            last_onset = Fraction(0)
            for el in measure:
                if el.tag == "attributes":
                    divisions = int(el.findtext("divisions", divisions))
                    max_staff = max(max_staff, int(el.findtext("staves", 1)))
                elif el.tag == "backup":
                    t -= Fraction(int(el.findtext("duration")), divisions)
                elif el.tag == "forward":
                    t += Fraction(int(el.findtext("duration")), divisions)
                elif el.tag == "note":
                    if el.find("grace") is not None:
                        continue
                    dur = Fraction(int(el.findtext("duration", 0)), divisions)
                    onset = last_onset if el.find("chord") is not None else t
                    pitch = el.find("pitch")
                    if pitch is not None:
                        staff = staff_offset + int(el.findtext("staff", 1))
                        midi = midi_of(pitch.findtext("step"), int(float(pitch.findtext("alter", 0))),
                                       int(pitch.findtext("octave")))
                        events.append(NoteEvent(m_index, onset, midi, dur, staff))
                    if el.find("chord") is None:
                        last_onset = t
                        t += dur
        staff_offset += max_staff
    return sorted(events)
