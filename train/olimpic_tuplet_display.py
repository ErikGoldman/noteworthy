"""Whether each OLiMPiC tuplet group draws its number, from the song's own MuseScore file.

OLiMPiC's labels come from OpenScore Lieder's MusicXML exports, which record a hidden number only
when MuseScore's numberType says so (2, on the tuplet or as the score's style), not when the tuplet
is made invisible (<visible>0</visible>, which hides number and bracket). Most songs hide numbers
the second way, so their labels call 46% of all groups printed where the page draws none
(EXPERIMENTS.md, "OLiMPiC's tuplet labels against the scores' own MuseScore files").

Pairing: OLiMPiC's systems keep the score's measure numbers as MuseScore 3 exports them
(exported_measure_numbers), and within a measure both list the groups in document order, the
upper staff's voices before the lower's. A song's systems are paired in page order, so a number
repeated after a section break is taken at its next occurrence past the last measure paired.

A song whose groups do not all pair (no .mscx, a measure or group missing, a ratio that differs)
is left out by the caller, never relabelled in part (decided 2026-11-17: about 8 songs, 380 of
23,308 groups; MuseScore 4 files, kept by the repo for 4 songs, are not read)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HIDDEN_NUMBER_TYPE = "2"


@dataclass(frozen=True)
class ScoreTuplet:
    ratio: str      # "3in2": actual notes in normal notes, as the labels' time-modification
    hidden: bool    # number not drawn


class Unpaired(Exception):
    """A song's groups cannot all be matched to its score's."""


def is_irregular(measure: ET.Element) -> bool:
    """MuseScore 3 writes <irregular>1</irregular>, 2.x an empty <irregular/>."""
    flag = measure.find("irregular")
    return flag is not None and (flag.text or "1").strip() != "0"


def restarts_numbering(base: ET.Element) -> bool:
    for layout_break in base.iter("LayoutBreak"):
        if layout_break.findtext("subtype") == "section":
            return layout_break.findtext("startWithMeasureOne", "1") != "0"
    return False


def exported_measure_numbers(first_staff: ET.Element) -> list[str]:
    """Each measure's number in MuseScore 3's MusicXML: a first irregular measure (a pickup) is
    "0", later irregular ones "X1", "X2", ..., the rest count from 1, plus their noOffset; a
    section break that starts again at measure one restarts the count. Frames carry no number,
    only, at times, the break."""
    numbers, regular, irregular, restart = [], 1, 1, False
    for base in first_staff:
        if base.tag not in ("Measure", "HBox", "VBox", "TBox", "FBox"):
            continue
        if restart:
            regular, irregular = 1, 1
        if base.tag == "Measure":
            regular += int(base.findtext("noOffset", "0"))
            if is_irregular(base) and regular + irregular == 2:
                numbers.append("0")
            elif is_irregular(base):
                numbers.append(f"X{irregular}")
                irregular += 1
            else:
                numbers.append(str(regular))
                regular += 1
        restart = restarts_numbering(base)
    return numbers


def piano_staves(score: ET.Element) -> list[ET.Element]:
    """The piano part's staves: the part named for a piano, else the last (two songs set a
    harpsichord), at their score-level ids."""
    parts = score.findall("Part")
    named = [p for p in parts if "piano" in (p.findtext("Instrument/trackName") or p.findtext("trackName") or "").lower()]
    part = (named or parts)[-1]
    first = 1 + sum(len(p.findall("Staff")) for p in parts[:parts.index(part)])
    staves = {staff.get("id"): staff for staff in score.findall("Staff")}
    return [staves[str(first + k)] for k in range(len(part.findall("Staff")))]


def tuplets_in(measure: ET.Element, style_number_type: str) -> list[ScoreTuplet]:
    """The measure's tuplet groups in document order (voice by voice); a group is defined once,
    by the <Tuplet> holding its ratio."""
    return [ScoreTuplet(f"{t.findtext('actualNotes')}in{t.findtext('normalNotes')}",
                        t.findtext("visible") == "0" or t.findtext("numberType", style_number_type) == HIDDEN_NUMBER_TYPE
                        or t.findtext("Number/visible") == "0")
            for t in measure.iter("Tuplet") if t.find("actualNotes") is not None]


ScoreMeasures = dict[str, list[tuple[int, list[ScoreTuplet]]]]


def score_tuplets(mscx: Path) -> ScoreMeasures:
    """For each exported measure number, each measure bearing it (in score order): its index in
    the score and its piano groups, upper staff first."""
    score = ET.parse(mscx).getroot().find("Score")
    style_number_type = score.findtext("Style/tupletNumberType", "0")
    numbers = exported_measure_numbers(score.find("Staff"))
    measures = [[m for m in staff if m.tag == "Measure"] for staff in piano_staves(score)]
    if any(len(staff) != len(numbers) for staff in measures):
        raise Unpaired(f"{mscx.name}: piano staves hold {[len(s) for s in measures]} measures, the score {len(numbers)}")
    by_number = defaultdict(list)
    for index, number in enumerate(numbers):
        by_number[number].append((index, [t for staff in measures for t in tuplets_in(staff[index], style_number_type)]))
    return dict(by_number)


def song_score(lieder: Path, song: str, path: str) -> Path:
    """The song's .mscx: at its scores.yaml path, or (the repo has renamed some composers'
    directories since OLiMPiC was made) anywhere by its file name, lc<song>.mscx."""
    found = sorted((lieder / "scores" / path).glob("*.mscx")) or sorted((lieder / "scores").rglob(f"lc{song}.mscx"))
    if len(found) != 1:
        raise Unpaired(f"{len(found)} .mscx files found ({path})")
    return found[0]


def tuplet_starts(measure: ET.Element) -> list[tuple[ET.Element, str]]:
    """The measure's <tuplet type="start"> elements in document order, with their note's ratio."""
    return [(tuplet, f"{note.findtext('time-modification/actual-notes')}in{note.findtext('time-modification/normal-notes')}")
            for note in measure.iter("note") for tuplet in note.findall("notations/tuplet") if tuplet.get("type") == "start"]


def has_tuplets(part: ET.Element) -> bool:
    return any(t.get("type") == "start" for t in part.iter("tuplet"))


def set_display(tuplet: ET.Element, hidden: bool) -> None:
    """As the score draws it; a hidden tuplet draws neither number nor bracket."""
    if hidden:
        tuplet.set("show-number", "none")
        tuplet.set("bracket", "no")
    else:
        tuplet.attrib.pop("show-number", None)


def with_score_display(parts: list[ET.Element], score: ScoreMeasures) -> int:
    """Sets every tuplet start of a song's systems (their parts, in page order) as the score draws
    it; returns how many it changed. A measure is paired with the first measure of its number past
    the last one paired (OLiMPiC leaves some systems out). Raises Unpaired, having changed nothing,
    if any group has no match of the same ratio."""
    last, pairs = -1, []
    for part in parts:
        for measure in part.findall("measure"):
            number, starts = measure.get("number"), tuplet_starts(measure)
            later = [(index, groups) for index, groups in score.get(number, []) if index > last]
            if not later:
                if starts:
                    raise Unpaired(f"measure {number}: not in the score past measure index {last}")
                continue
            last, groups = later[0]
            if [ratio for _, ratio in starts] != [group.ratio for group in groups]:
                raise Unpaired(f"measure {number}: the systems' groups {[r for _, r in starts]}, the score's {[g.ratio for g in groups]}")
            pairs += [(tuplet, group.hidden) for (tuplet, _), group in zip(starts, groups)]
    changed = sum((tuplet.get("show-number") == "none") != hidden for tuplet, hidden in pairs)
    for tuplet, hidden in pairs:
        set_display(tuplet, hidden)
    return changed
