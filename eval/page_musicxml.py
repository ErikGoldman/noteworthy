"""Finishes a page's joined MusicXML, for readers that fill gaps with their own defaults.

The delinearizer works a system at a time and leaves optional attributes out. music21 reads a
<grace> with no slash as slashed and a <tuplet> with no bracket as bracketed, neither of which
the model said, and its one slur stack pairs a slur with the other staff's across a barline.
A tie joins two notes of one pitch, so one that does not is completed or removed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict

MAX_SLUR_NUMBER = 16  # MusicXML's number-level


def pair_slurs(part: ET.Element) -> None:
    """Numbers slurs afresh, each stop closing the latest slur open on its own staff. A stop with
    nothing open goes: its start is on an earlier page, and alone it would pair with a stranger."""
    open_slurs = defaultdict(list)
    free = list(range(1, MAX_SLUR_NUMBER + 1))
    for note in part.iter("note"):
        staff = note.findtext("staff", "1")
        for notations in note.findall("notations"):
            for slur in sorted(notations.findall("slur"), key=lambda s: s.get("type") != "stop"):
                if slur.get("type") == "start":
                    number = free.pop(0) if free else MAX_SLUR_NUMBER
                    slur.set("number", str(number))
                    open_slurs[staff].append(number)
                elif open_slurs[staff]:
                    number = open_slurs[staff].pop()
                    slur.set("number", str(number))
                    free = sorted([*free, number])
                else:
                    notations.remove(slur)


def pitch_of(note: ET.Element) -> tuple | None:
    pitch = note.find("pitch")
    return None if pitch is None else (pitch.findtext("step"), pitch.findtext("alter", "0"), pitch.findtext("octave"))


def remove_tie(note: ET.Element, kind: str) -> None:
    for tie in note.findall("tie"):
        if tie.get("type") == kind:
            note.remove(tie)
    for notations in note.findall("notations"):
        for tied in notations.findall("tied"):
            if tied.get("type") == kind:
                notations.remove(tied)


def has_tie(note: ET.Element, kind: str) -> bool:
    return any(t.get("type") == kind for t in note.findall("tie"))


def events_by_voice(part: ET.Element) -> dict[tuple[str, str], list[list[ET.Element]]]:
    """Each voice's notes grouped into events: a note and the chord notes after it."""
    events = defaultdict(list)
    for note in part.iter("note"):
        voice = (note.findtext("staff", "1"), note.findtext("voice", "1"))
        if note.find("chord") is not None and events[voice]:
            events[voice][-1].append(note)
        else:
            events[voice].append([note])
    return events


def add_tie(note: ET.Element, kind: str) -> None:
    # MusicXML puts <tie> after <duration> and before <voice>.
    index = list(note).index(note.find("voice")) if note.find("voice") is not None else len(note)
    note.insert(index, ET.Element("tie", type=kind))
    notations = note.find("notations")
    if notations is None:
        notations = ET.SubElement(note, "notations")
    notations.append(ET.Element("tied", type=kind))


def mend_ties(part: ET.Element) -> None:
    """Each tie joins the same pitch in the voice's next event: a half-written tie whose other
    note is there is completed, and one with no such note goes. The model sometimes ties a grace
    note to the different note its slur leads to (Polish 017), or leaves out a tie's stop (093)."""
    for events in events_by_voice(part).values():
        for before, after in zip([[]] + events, events + [[]]):
            in_before = {pitch_of(n): n for n in before}
            in_after = {pitch_of(n): n for n in after}
            for note in before:
                if has_tie(note, "start"):
                    mend(note, in_after.get(pitch_of(note)), "start", "stop")
            for note in after:
                if has_tie(note, "stop"):
                    mend(note, in_before.get(pitch_of(note)), "stop", "start")


def mend(note: ET.Element, partner: ET.Element | None, kind: str, partner_kind: str) -> None:
    """A tie end on `note`, whose other end belongs on `partner`, the same pitch next door."""
    if partner is None:
        remove_tie(note, kind)
    elif not has_tie(partner, partner_kind):
        add_tie(partner, partner_kind)


def state_grace_slashes(part: ET.Element) -> None:
    for grace in part.iter("grace"):
        grace.set("slash", grace.get("slash", "no"))


def state_tuplet_brackets(part: ET.Element) -> None:
    """A beam already groups a tuplet, so only unbeamed ones take a bracket."""
    for note in part.iter("note"):
        for tuplet in note.findall("notations/tuplet"):
            if tuplet.get("type") == "start":
                tuplet.set("bracket", "no" if note.find("beam") is not None else "yes")


def finish_score(score: ET.Element) -> None:
    """Also clears the part name: the delinearizer's is a constant, not something read."""
    for name in score.iter("part-name"):
        name.text = ""
    for part in score.findall("part"):
        pair_slurs(part)
        mend_ties(part)
        state_grace_slashes(part)
        state_tuplet_brackets(part)
