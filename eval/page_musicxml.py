"""Finishes a page's joined MusicXML, for readers that fill gaps with their own defaults.

The delinearizer works a system at a time and leaves optional attributes out. music21 reads a
<grace> with no slash as slashed and a <tuplet> with no bracket as bracketed, neither of which
the model said, and its one slur stack pairs a slur with the other staff's across a barline.
A tie joins two notes of one pitch, so one that does not is completed or removed, and a slur
that does is a tie. A clef change at a barline goes before it, as printed.
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


def slur_ends(event: list[ET.Element], kind: str) -> dict[str, tuple[ET.Element, ET.Element]]:
    """The event's slur starts or stops by number, each with the <notations> holding it."""
    return {slur.get("number"): (notations, slur) for note in event for notations in note.findall("notations")
            for slur in notations.findall("slur") if slur.get("type") == kind}


def plain(event: list[ET.Element]) -> bool:
    """No grace note and no articulation: a grace note slurred to its own pitch (Polish 017) and
    portato, a slur over dotted repeated notes, are slurs."""
    return all(note.find("grace") is None and note.find("notations/articulations") is None for note in event)


def ties_from_slurs(part: ET.Element) -> None:
    """A slur from one event to the voice's next, over exactly the same pitches, is a tie: the
    print draws both with one arc, and the model reads about two in five ties as slurs. Where
    the pitches differ even in part (049, a chord moving under a slur) it stays a slur."""
    for events in events_by_voice(part).values():
        for before, after in zip(events, events[1:]):
            pitches = {pitch_of(note) for note in before}
            if None in pitches or pitches != {pitch_of(note) for note in after} or not (plain(before) and plain(after)):
                continue
            stops = slur_ends(after, "stop")
            for number, (notations, slur) in slur_ends(before, "start").items():
                if number in stops:
                    notations.remove(slur)
                    stop_notations, stop = stops[number]
                    stop_notations.remove(stop)
                    tie_event(before, after)


def tie_event(before: list[ET.Element], after: list[ET.Element]) -> None:
    for note in before:
        if not has_tie(note, "start"):
            add_tie(note, "start")
    for note in after:
        if not has_tie(note, "stop"):
            add_tie(note, "stop")


def clefs_before_barlines(part: ET.Element) -> None:
    """A clef change that opens a bar goes to the end of the bar before. The print draws a change
    at a barline before the barline, and gold puts it there; the model writes it as the next bar's
    first sign (Polish 095, 108). The first bar's clefs are the system's own and stay. Appended
    last, the clef lands at the bar's end because the delinearizer ends each bar on a voice that
    fills it; a bar ending in <backup> would put it earlier."""
    measures = part.findall("measure")
    for before, measure in zip(measures, measures[1:]):
        opening = next((child for child in measure if child.tag in ("attributes", "note", "backup", "forward")), None)
        if opening is None or opening.tag != "attributes" or opening.find("clef") is None:
            continue
        clefs = opening.findall("clef")
        for clef in clefs:
            opening.remove(clef)
        ET.SubElement(before, "attributes").extend(clefs)
        if len(opening) == 0:
            measure.remove(opening)


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
        ties_from_slurs(part)
        mend_ties(part)
        clefs_before_barlines(part)
        state_grace_slashes(part)
        state_tuplet_brackets(part)
