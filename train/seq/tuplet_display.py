"""How a score's tuplets are drawn: which numbers engravers leave out, and how the rest are set.

Hiding follows what real scores do (EXPERIMENTS.md, "When real scores hide a tuplet's number",
2026-11-15, from PDMX, ASAP and OLiMPiC, all MuseScore engravings): it is first the score's
choice, most showing every number and a few hiding all; the rest hide by place in a run of like
groups, about half at the first and nearly nine in ten from the fifth, print the number again at
a voice's first tuplet in a system and for a tuplet alone in its bar, and mostly keep 5s and 7s.
The labels follow: a hidden group's start carries show-number="none", which the linearizer writes
as `tuplet:hidden` (seq.tuplet_numbers).

The look is how a printed number is set, which the labels do not record: above or below its
notes, with a bracket or without, as 3 or 3:2, and, drawn by the engraver's own means
(build_seq_dataset.py, build_musescore_crops.py), far from its notes, small, or in an arc over
an unbeamed group. Real prints the model misreads set their numbers so (EXPERIMENTS.md, "Printed
tuplets whose ratio is left out, by eye", 2026-11-16). An engraver keeps one look through a
score, so it is drawn once per score.
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum

# Of scores with tuplets: PDMX hides in 2% of them, ASAP in 68%, OLiMPiC in 21%. Weighted toward
# hiding, since the model meets the passages that hide far more often than PDMX's share says.
SHOW_ALL, BY_RUN, HIDE_ALL = "show all", "by run", "hide all"
SCORE_HIDING = {SHOW_ALL: 0.65, BY_RUN: 0.28, HIDE_ALL: 0.07}
# A mix for teaching that a number may or may not be drawn: half the scores hide by run, so most
# of their systems show some groups' numbers and hide others' (TODO, hidden-tuplet fine-tunes).
OFTEN_HIDING = {SHOW_ALL: 0.25, BY_RUN: 0.5, HIDE_ALL: 0.25}
HIDING_MIXES = {"engravers": SCORE_HIDING, "often": OFTEN_HIDING}
# Within the scores that hide some (PDMX / ASAP / OLiMPiC agree to a few points):
HIDDEN_BY_RUN_PLACE = (0.50, 0.65, 0.72, 0.78, 0.87)  # the 1st, 2nd, 3rd, 4th, 5th and later group
HIDDEN_AT_VOICES_FIRST_IN_SYSTEM = 0.58  # against 0.77 later
HIDDEN_ALONE_IN_BAR = 0.47  # against 0.77 with other tuplets
HIDDEN_ODD_GROUPING = 0.10  # 5:4 5% in PDMX, 30% in ASAP; 7s never in PDMX
COMMON_GROUPINGS = {3, 6}


@dataclass
class TupletGroup:
    """A tuplet group as drawn: its <tuplet type="start">, and where it stands."""
    start: ET.Element
    measure: int
    staff: str
    actual: int
    normal: int
    value: str
    run_place: int = 1
    first_in_voice: bool = False
    alone_in_bar: bool = False
    notes: list[ET.Element] = field(default_factory=list)

    def follows(self, other: TupletGroup | None) -> bool:
        """Of a run with other: the same ratio and value, nothing between them in the voice."""
        return other is not None and (self.actual, self.normal, self.value) == (other.actual, other.normal, other.value)


def tuplet_groups(part: ET.Element) -> list[TupletGroup]:
    """The part's groups in document order. A group nested in another counts as part of it; a
    voice's grace notes and a chord's later notes neither join nor break a run."""
    groups: list[TupletGroup] = []
    open_group: dict[tuple[str, str], TupletGroup] = {}
    just_closed: dict[tuple[str, str], TupletGroup | None] = {}
    voices_with_groups: set[tuple[str, str]] = set()
    for index, measure in enumerate(part.findall("measure")):
        for note in measure.findall("note"):
            if note.find("grace") is not None or note.find("chord") is not None:
                continue
            voice = (note.findtext("staff", "1"), note.findtext("voice", "1"))
            start = note.find("notations/tuplet[@type='start']")
            group = open_group.get(voice)
            if group is None and start is not None:
                group = new_group(start, note, index, voice[0])
                previous = just_closed.get(voice)
                group.run_place = previous.run_place + 1 if group.follows(previous) else 1
                group.first_in_voice = voice not in voices_with_groups
                voices_with_groups.add(voice)
                groups.append(group)
                open_group[voice] = group
            if group is None:
                just_closed[voice] = None
                continue
            group.notes.append(note)
            if note.find("notations/tuplet[@type='stop']") is not None:
                just_closed[voice] = open_group.pop(voice)
    for group in groups:
        group.alone_in_bar = sum(g.measure == group.measure for g in groups) == 1
    return groups


def new_group(start: ET.Element, note: ET.Element, measure: int, staff: str) -> TupletGroup:
    modification = note.find("time-modification")
    actual = int(modification.findtext("actual-notes", "3")) if modification is not None else 3
    normal = int(modification.findtext("normal-notes", "2")) if modification is not None else 2
    return TupletGroup(start, measure, staff, actual, normal, note.findtext("type", ""))


# ---- hiding ----------------------------------------------------------------------------------

def score_hiding(rng: random.Random, mix: dict[str, float] = SCORE_HIDING) -> str:
    return rng.choices(list(mix), weights=list(mix.values()))[0]


def hidden_share(group: TupletGroup) -> float:
    """How often an engraver hiding by run leaves this group's number out."""
    share = HIDDEN_BY_RUN_PLACE[min(group.run_place, len(HIDDEN_BY_RUN_PLACE)) - 1]
    if group.first_in_voice:
        share = min(share, HIDDEN_AT_VOICES_FIRST_IN_SYSTEM)
    if group.alone_in_bar:
        share = min(share, HIDDEN_ALONE_IN_BAR)
    if group.actual not in COMMON_GROUPINGS:
        share = min(share, HIDDEN_ODD_GROUPING)
    return share


def hide_as_engravers_do(part: ET.Element, hiding: str, rng: random.Random) -> None:
    """Hides numbers (and their brackets) in one system as a score of this hiding would."""
    for group in tuplet_groups(part):
        if hiding == HIDE_ALL or (hiding == BY_RUN and rng.random() < hidden_share(group)):
            group.start.set("show-number", "none")
            group.start.set("bracket", "no")


# ---- the look --------------------------------------------------------------------------------

@dataclass(frozen=True)
class TupletLook:
    """How one score sets its printed numbers. None leaves the choice to the engraver."""
    side: str | None = None       # "above" or "below" the notes
    bracket: str | None = None    # "yes" or "no"
    ratio_shown: bool = False     # 3:2 rather than 3
    far: float = 0.0              # share of numbers set well away from their notes
    small: float = 0.0            # share of numbers drawn small
    arc: float = 0.0              # share of bracketed groups drawn with an arc instead


PLAIN_LOOK = TupletLook()


def tuplet_look(rng: random.Random) -> TupletLook:
    def some(chance: float, low: float, high: float) -> float:
        return rng.uniform(low, high) if rng.random() < chance else 0.0
    return TupletLook(side=rng.choices([None, "above", "below"], [0.5, 0.3, 0.2])[0],
                      bracket=rng.choices([None, "yes", "no"], [0.6, 0.25, 0.15])[0],
                      ratio_shown=rng.random() < 0.05,
                      far=some(0.4, 0.2, 0.8), small=some(0.3, 0.3, 1.0), arc=some(0.4, 0.5, 1.0))


def set_look(part: ET.Element, look: TupletLook) -> None:
    """The look's MusicXML part (side, bracket, ratio) on each printed group's start."""
    for group in tuplet_groups(part):
        if group.start.get("show-number") == "none":
            continue
        if look.side:
            group.start.set("placement", look.side)
        if look.bracket:
            group.start.set("bracket", look.bracket)
        if look.ratio_shown:
            group.start.set("show-number", "both")


@dataclass(frozen=True)
class TupletDrawing:
    """A builder's tuplet options: a hiding mix of HIDING_MIXES (None: the builder's own rule),
    and varied looks."""
    hiding: str | None = None
    varied_looks: bool = False

    @property
    def changed(self) -> bool:
        return self.hiding is not None or self.varied_looks


@dataclass(frozen=True)
class ScoreTuplets:
    """One score's choices, drawn once and applied to each of its systems. hiding None: the
    builder's own rule, so that a run varying the look alone changes nothing else."""
    hiding: str | None
    look: TupletLook

    @staticmethod
    def drawn(drawing: TupletDrawing, rng: random.Random) -> ScoreTuplets:
        """Draws from rng only what the drawing changes, so a builder without either option makes
        the crops it made before."""
        return ScoreTuplets(score_hiding(rng, HIDING_MIXES[drawing.hiding]) if drawing.hiding else None,
                            tuplet_look(rng) if drawing.varied_looks else PLAIN_LOOK)
