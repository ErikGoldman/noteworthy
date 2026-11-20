"""A system's tuplet drawing carried into MuseScore's own file, which MuseScore 3.6.2 honours where
it ignores the same in MusicXML (show-number, bracket, placement: build_musescore_crops.py).

The system's MusicXML says what each group shows (seq.tuplet_display); MuseScore converts it to
.mscx, and each of its <Tuplet>s is given the matching group's numberType (0 a number, 1 a ratio,
2 none), bracketType (1 a bracket, 2 none) and direction, and with the score's `far` share a
number offset away from its notes. Groups pair by staff, bar and order (document order in both
files, a staff's voices one after another); a pair must agree in ratio and in every note's value
and pitches, since two 3:2 groups of two voices would pass a ratio check alone. A system whose
groups do not all pair is not drawn: most often a group MuseScore does not keep as a tuplet (two
eighths marked 3:2). Pitches are compared as pitch classes: under an ottava the system's MusicXML
gives them as printed (with_label_ottavas), MuseScore's file as they sound.

The look's small numbers and arcs are drawn on Verovio's SVG only (seq.tuplet_svg).
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from seq.tuplet_display import TupletGroup, TupletLook, tuplet_groups

RATIO, NO_NUMBER = "1", "2"  # numberType; 0, a number, is the default
BRACKET, NO_BRACKET = "1", "2"
DIRECTION = {"above": "up", "below": "down"}
FAR_SPATIUMS = (1.5, 4.0)
# A number set far from its notes must stay on the page: the builder's page leaves half an inch above
# and below (build_musescore_crops.PAGE, about 7 spatiums), which a number high over its notes and moved
# 4 more can leave, and MuseScore draws nothing off the page. A page with room for any of them, its
# height grown by the same; the crop is cut to the ink, so where the system sits does not show.
ROOMY_PAGE = {"pageHeight": "9", "pageEvenTopMargin": "2", "pageOddTopMargin": "2",
              "pageEvenBottomMargin": "2", "pageOddBottomMargin": "2"}
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
GRACE_TAGS = ("acciaccatura", "appoggiatura", "grace4", "grace16", "grace32", "grace8after", "grace16after", "grace32after")

Event = tuple[str, tuple[int, ...]]  # a note's value and its pitch classes (none for a rest)


def midi(pitch: ET.Element) -> int:
    alter = round(float(pitch.findtext("alter", "0")))
    return 12 * (int(pitch.findtext("octave")) + 1) + STEPS[pitch.findtext("step")] + alter


def musicxml_events(part: ET.Element, group: TupletGroup) -> list[Event]:
    """The group's notes, each with the chord notes that follow it."""
    notes = part.findall("measure")[group.measure].findall("note")
    events, members = [], {id(n) for n in group.notes}
    for k, note in enumerate(notes):
        if id(note) not in members:
            continue
        chord = [note] + list(next_chord_notes(notes, k))
        pitches = tuple(sorted(midi(n.find("pitch")) % 12 for n in chord if n.find("pitch") is not None))
        events.append((note.findtext("type", ""), pitches))
    return events


def next_chord_notes(notes: list[ET.Element], k: int):
    for n in notes[k + 1:]:
        if n.find("chord") is None:
            return
        yield n


def mscx_groups(score: ET.Element) -> dict[tuple[int, int], list[tuple[ET.Element, list[Event]]]]:
    """Each staff's and bar's <Tuplet>s in order, with their events (grace chords left out)."""
    found: dict[tuple[int, int], list[tuple[ET.Element, list[Event]]]] = {}
    for staff_index, staff in enumerate(score.findall("Staff"), 1):
        for measure_index, measure in enumerate(staff.findall("Measure")):
            groups = found.setdefault((staff_index, measure_index), [])
            for voice in measure.findall("voice"):
                open_group = None
                for element in voice:
                    if element.tag == "Tuplet" and open_group is None:
                        open_group = (element, [])
                        groups.append(open_group)
                    elif element.tag in ("Chord", "Rest") and open_group is not None and not is_grace(element):
                        pitches = tuple(sorted(int(n.findtext("pitch")) % 12 for n in element.findall("Note")))
                        open_group[1].append((element.findtext("durationType", ""), pitches))
                    elif element.tag == "endTuplet":
                        open_group = None
    return found


def is_grace(chord: ET.Element) -> bool:
    return any(chord.find(tag) is not None for tag in GRACE_TAGS)


def ratio_of(tuplet: ET.Element) -> tuple[int, int]:
    return int(tuplet.findtext("actualNotes", "0")), int(tuplet.findtext("normalNotes", "0"))


def paired(part: ET.Element, score: ET.Element) -> list[tuple[TupletGroup, ET.Element]] | None:
    """Each MusicXML group with its <Tuplet>, or None if they do not all pair."""
    in_mscx = mscx_groups(score)
    by_place: dict[tuple[int, int], list[TupletGroup]] = {}
    for group in tuplet_groups(part):
        by_place.setdefault((int(group.staff), group.measure), []).append(group)
    if sum(map(len, by_place.values())) != sum(map(len, in_mscx.values())):
        return None
    pairs = []
    for place, groups in by_place.items():
        tuplets = in_mscx.get(place, [])
        if len(tuplets) != len(groups):
            return None
        for group, (tuplet, events) in zip(groups, tuplets):
            if ratio_of(tuplet) != (group.actual, group.normal) or events != musicxml_events(part, group):
                return None
            pairs.append((group, tuplet))
    return pairs


def set_property(tuplet: ET.Element, tag: str, value: str) -> None:
    """MuseScore reads a tuplet's properties in any order."""
    element = tuplet.find(tag)
    if element is None:
        element = ET.SubElement(tuplet, tag)
    element.text = value


def draw_as_musicxml_says(group: TupletGroup, tuplet: ET.Element, look: TupletLook, rng: random.Random) -> None:
    start = group.start
    if start.get("show-number") == "none":
        set_property(tuplet, "numberType", NO_NUMBER)
        set_property(tuplet, "bracketType", NO_BRACKET)
        return
    if start.get("show-number") == "both":
        set_property(tuplet, "numberType", RATIO)
    if start.get("bracket") in ("yes", "no"):
        set_property(tuplet, "bracketType", BRACKET if start.get("bracket") == "yes" else NO_BRACKET)
    side = start.get("placement")
    if side in DIRECTION:
        set_property(tuplet, "direction", DIRECTION[side])
        if rng.random() < look.far:
            set_number_offset(tuplet, (-1 if side == "above" else 1) * rng.uniform(*FAR_SPATIUMS))


def set_number_offset(tuplet: ET.Element, dy: float) -> None:
    number = tuplet.find("Number")
    if number is not None:
        number.insert(1, ET.Element("offset", x="0", y=f"{dy:.2f}"))


def with_roomy_page(score: ET.Element) -> None:
    style = score.find("Style")
    if style is None:
        style = ET.SubElement(score, "Style")
    for tag, value in ROOMY_PAGE.items():
        set_property(style, tag, value)


def drawn_in_mscx(musicxml: Path, mscx: Path, look: TupletLook, rng: random.Random) -> bool:
    """Rewrites mscx with the MusicXML's tuplet drawing; False (file untouched) if they do not pair."""
    part = ET.parse(musicxml).getroot().find("part")
    tree = ET.parse(mscx)
    score = tree.getroot().find("Score")
    pairs = paired(part, score)
    if pairs is None:
        return False
    for group, tuplet in pairs:
        draw_as_musicxml_says(group, tuplet, look, rng)
    if look.far:
        with_roomy_page(score)
    tree.write(mscx, encoding="UTF-8", xml_declaration=True)
    return True
