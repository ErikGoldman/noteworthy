"""Ottavas in LMX.

OLiMPiC's LMX has no ottava: a note under an 8va sign is labelled with the pitch it sounds, so
a model has to add the octave silently and hold it from the sign onward. That hidden state
goes wrong in ways nothing downstream can see: a segno taken for an `8` lifts whole bars, and
a short `8...` line is held past its end.

Our labels give the pitch as printed and mark each note under a sign with a token straight
after its pitch. Each note says for itself whether it is shifted, so there is no state to
hold, and `sounding` turns such LMX back into OLiMPiC's form.

A system is taken alone, as it is drawn: a sign started in an earlier system has no effect.
"""
from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction

OCTAVES_OF_TOKEN = {"ottava:8va": 1, "ottava:15ma": 2, "ottava:8vb": -1, "ottava:15mb": -2}
TOKEN_OF_OCTAVES = {octaves: token for token, octaves in OCTAVES_OF_TOKEN.items()}
OCTAVES_OF_SIZE = {8: 1, 15: 2}
TOKEN_ATTRIBUTE = "ottava-token"  # on a <note> of a written_part, for the linearizer
PITCH = re.compile(r"([A-G])(\d)")


def octaves_of(sign: ET.Element) -> int:
    """How far above the printed notes they sound. MusicXML's type says which way the notes
    were moved for printing: `down` is the 8va sign."""
    kind = sign.get("type")
    if kind not in ("up", "down"):
        return 0
    octaves = OCTAVES_OF_SIZE[int(sign.get("size", 8))]
    return octaves if kind == "down" else -octaves


def placed_elements(part: ET.Element) -> Iterator[tuple[ET.Element, tuple[int, Fraction], ET.Element]]:
    """Each note and direction, with its measure and its place: measure index, then position."""
    for index, measure in enumerate(part.findall("measure")):
        position = onset = Fraction(0)
        for element in measure:
            duration = Fraction(element.findtext("duration", "0"))
            if element.tag == "direction":
                yield measure, (index, position), element
            elif element.tag == "backup":
                position -= duration
            elif element.tag == "forward":
                position += duration
            elif element.tag == "note":
                if element.find("chord") is None:
                    onset = position
                    position += Fraction(0) if element.find("grace") is not None else duration
                yield measure, (index, onset), element


def notes_under_ottavas(part: ET.Element) -> list[tuple[ET.Element, int]]:
    """Each pitched note under a sign, with its octaves. A sign holds from its place in the
    measure until its stop, on its own staff, whichever voice a note is in."""
    signs, notes = defaultdict(list), []
    for _, place, element in placed_elements(part):
        staff = int(element.findtext("staff", "1"))
        if element.tag == "direction":
            # A stop sorts before a start at the same place.
            signs[staff] += [(place, abs(octaves_of(s)), octaves_of(s)) for s in element.iter("octave-shift")]
        elif element.find("pitch") is not None:
            notes.append((place, staff, element))
    for staff in signs:
        signs[staff].sort(key=lambda sign: sign[:2])
    under = []
    for place, staff, note in notes:
        held = bisect_right(signs[staff], place, key=lambda sign: sign[0])
        if held and signs[staff][held - 1][2]:
            under.append((note, signs[staff][held - 1][2]))
    return under


def move_octaves(note: ET.Element, octaves: int) -> None:
    pitch = note.find("pitch/octave")
    pitch.text = str(int(pitch.text) + octaves)


def written_part(part: ET.Element) -> ET.Element:
    """A copy of the part with the pitches as printed, its shifted notes tagged with their token."""
    written = copy.deepcopy(part)
    for note, octaves in notes_under_ottavas(written):
        move_octaves(note, -octaves)
        note.set(TOKEN_ATTRIBUTE, TOKEN_OF_OCTAVES[octaves])
    return written


def shifts_of(lmx: str) -> list[int]:
    """Octaves above the printed pitch of each note, in token order."""
    shifts: list[int] = []
    for token in lmx.split():
        if PITCH.fullmatch(token):
            shifts.append(0)
        elif token in OCTAVES_OF_TOKEN and shifts:
            shifts[-1] = OCTAVES_OF_TOKEN[token]
    return shifts


def shift_direction(kind: str, octaves: int, staff: str, number: int) -> ET.Element:
    """A <direction> holding an octave-shift; `kind` is start or stop."""
    direction = ET.Element("direction", placement="above" if octaves > 0 else "below")
    size = str(next(s for s, o in OCTAVES_OF_SIZE.items() if o == abs(octaves)))
    moved = "down" if octaves > 0 else "up"
    ET.SubElement(ET.SubElement(direction, "direction-type"), "octave-shift",
                  type=moved if kind == "start" else "stop", size=size, number=str(number))
    ET.SubElement(direction, "staff").text = staff
    return direction


def add_octave_shifts(part: ET.Element, shifts: list[int]) -> None:
    """Draws the signs on a part whose pitches sound: MusicXML keeps the sounding pitch under
    an octave-shift. `shifts` gives each pitched note's octaves, in document order. A sign
    belongs to a staff, not a voice, so it spans everything at the places it covers. A part
    that lost notes in delinearizing keeps no signs: on the wrong notes they would be worse
    than none."""
    pitched = [(measure, place, note) for measure, place, note in placed_elements(part)
               if note.tag == "note" and note.find("pitch") is not None]
    if len(pitched) != len(shifts) or not any(shifts):
        return
    staves = defaultdict(dict)
    for (measure, place, note), octaves in zip(pitched, shifts):
        _, notes, shift = staves[note.findtext("staff", "1")].get(place, (measure, [], 0))
        staves[note.findtext("staff", "1")][place] = measure, [*notes, note], shift or octaves
    for staff, places in staves.items():
        onsets = [places[place] for place in sorted(places)]
        held = 0
        for (measure, notes, octaves), following in zip(onsets, [*onsets[1:], None]):
            if octaves and octaves != held:
                measure.insert(list(measure).index(notes[0]), shift_direction("start", octaves, staff, int(staff)))
            held = octaves
            if octaves and (following is None or following[2] != octaves):
                measure.insert(list(measure).index(notes[-1]) + 1, shift_direction("stop", octaves, staff, int(staff)))


def sounding(lmx: str) -> str:
    """LMX in OLiMPiC's form: ottava tokens applied to the pitch before them and removed. A
    token with no pitch before it means nothing, so it goes too."""
    out: list[str] = []
    for token in lmx.split():
        if token not in OCTAVES_OF_TOKEN:
            out.append(token)
        elif out and (pitch := PITCH.fullmatch(out[-1])):
            out[-1] = f"{pitch[1]}{int(pitch[2]) + OCTAVES_OF_TOKEN[token]}"
    return " ".join(out)
