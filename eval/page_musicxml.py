"""Finishes a page's joined MusicXML, for readers that fill gaps with their own defaults.

The delinearizer works a system at a time and leaves optional attributes out. music21 reads a
<grace> with no slash as slashed and a <tuplet> with no bracket as bracketed, neither of which
the model said, and its one slur stack pairs a slur with the other staff's across a barline.
A tie joins two notes of one pitch, so one that does not is completed or removed, and a slur (or
a half arc at a system break) that does is a tie. A clef change at a barline goes before it, as
printed, and a head two voices share shows its accidental in both. A tuplet the model left
unmarked is closed, with no number.
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


def measure_index(part: ET.Element) -> dict[ET.Element, int]:
    return {note: index for index, measure in enumerate(part.findall("measure")) for note in measure.iter("note")}


def events_by_voice(part: ET.Element, across_staves: bool = False) -> dict[tuple[str, ...], list[list[ET.Element]]]:
    """Each voice's notes grouped into events: a note and the chord notes after it. By default
    a voice's notes on each staff are taken apart; across_staves keeps a voice that crosses to
    the other staff in one line."""
    events = defaultdict(list)
    for note in part.iter("note"):
        voice = (note.findtext("voice", "1"),) if across_staves else (note.findtext("staff", "1"), note.findtext("voice", "1"))
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
    note to the different note its slur leads to (Polish 017), or leaves out a tie's stop (093).
    A tie in the page's last bar may run on to the next page, as gold has it (081), and stays."""
    measure_of = measure_index(part)
    last_bar = len(part.findall("measure")) - 1
    for events in events_by_voice(part).values():
        for before, after in zip([[]] + events, events + [[]]):
            in_before = {pitch_of(n): n for n in before}
            in_after = {pitch_of(n): n for n in after}
            runs_off_page = not after and bool(before) and measure_of[before[0]] == last_bar
            for note in before:
                if has_tie(note, "start") and not runs_off_page:
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


# Portato is a slur over repeated notes with these marks. An accent says nothing of the arc: a
# tie often starts on an accented chord (Polish 039, 079).
PORTATO_MARKS = {"staccato", "tenuto", "detached-legato", "staccatissimo", "spiccato"}


def plain(event: list[ET.Element]) -> bool:
    """No grace note and no portato mark: a grace note slurred to its own pitch (Polish 017) and
    portato are slurs."""
    return all(note.find("grace") is None and not any(mark.tag in PORTATO_MARKS for mark in note.iterfind("notations/articulations/*"))
               for note in event)


def mark_new_system(measure: ET.Element) -> None:
    measure.insert(0, ET.Element("print", {"new-system": "yes"}))


def starts_system(measure: ET.Element) -> bool:
    return any(p.get("new-system") == "yes" for p in measure.findall("print"))


def ties_across_system_breaks(part: ET.Element) -> None:
    """A tie over a system break is printed as two half arcs, and the model, seeing one system,
    reads each as a slur end with no partner (Polish 070, 108, 056), or as a tie end. Where the
    voice's last event before the break and first after it have the same notes, either half is a
    tie. Run before pair_slurs, which would pair such a stray stop with an unrelated slur. A
    voice's notes on the other staff are not next to each other (062: one cross-staff note a beat)."""
    measure_of = measure_index(part)
    measures = part.findall("measure")
    for events in events_by_voice(part, across_staves=True).values():
        for before, after in zip(events, events[1:]):
            if not (starts_system(measures[measure_of[after[0]]]) and measure_of[before[0]] < measure_of[after[0]]):
                continue
            partners = tie_partners(before, after, across_barline=True) if plain(before) and plain(after) else []
            if partners and half_arc_drawn(before, after):
                for notations, slur in [*slur_ends(before, "start").values(), *slur_ends(after, "stop").values()]:
                    notations.remove(slur)
                tie_pairs(partners)


def half_arc_drawn(before: list[ET.Element], after: list[ET.Element]) -> bool:
    return bool(slur_ends(before, "start") or slur_ends(after, "stop")
                or any(has_tie(n, "start") for n in before) or any(has_tie(n, "stop") for n in after))


def ties_from_slurs(part: ET.Element) -> None:
    """A slur from one event to the voice's next, whose notes all sound on into it, is a tie: the
    print draws both with one arc, and the model reads about two in five ties as slurs. The next
    event may add notes, as when a held note is joined by a chord (Polish 096, 055); where a note
    of the first does not go on (049, a chord moving under a slur) it stays a slur."""
    measure_of = measure_index(part)
    for events in events_by_voice(part).values():
        for before, after in zip(events, events[1:]):
            if not (plain(before) and plain(after)):
                continue
            partners = tie_partners(before, after, measure_of[before[0]] != measure_of[after[0]])
            stops = slur_ends(after, "stop")
            for number, (notations, slur) in slur_ends(before, "start").items():
                if partners and number in stops:
                    notations.remove(slur)
                    stop_notations, stop = stops[number]
                    stop_notations.remove(stop)
                    tie_pairs(partners)


def tie_partners(before: list[ET.Element], after: list[ET.Element], across_barline: bool) -> list[tuple]:
    """Each note of `before` with the note of `after` it would be tied to, or none if any note
    has no partner. A partner has the same pitch, or across a barline the same staff place where
    `after` writes no accidental of its own: a tie carries the bar's accidental over the barline,
    so a note read as slurred lost it to the key (Polish 039: C natural tied into a bar of C sharp)."""
    if any(pitch_of(note) is None for note in before + after):
        return []
    by_pitch = {pitch_of(note): note for note in after}
    if all(pitch_of(note) in by_pitch for note in before):
        return [(note, by_pitch[pitch_of(note)]) for note in before]
    by_place = {staff_place(note): note for note in after}
    if (across_barline and len(after) == len(before) and all(note.find("accidental") is None for note in after)
            and all(staff_place(note) in by_place for note in before)):
        return [(note, by_place[staff_place(note)]) for note in before]
    return []


def staff_place(note: ET.Element) -> tuple[str, str]:
    step, _, octave = pitch_of(note)
    return step, octave


def tie_pairs(pairs: list[tuple[ET.Element, ET.Element]]) -> None:
    """Ties each pair, the second note taking the first's alteration."""
    for first, second in pairs:
        if not has_tie(first, "start"):
            add_tie(first, "start")
        if not has_tie(second, "stop"):
            add_tie(second, "stop")
        set_alter(second.find("pitch"), pitch_of(first)[1])


def set_alter(pitch: ET.Element, alter: str) -> None:
    """MusicXML puts <alter> between <step> and <octave>, and leaves it out for none."""
    element = pitch.find("alter")
    if alter == "0":
        if element is not None:
            pitch.remove(element)
    elif element is not None:
        element.text = alter
    else:
        element = ET.Element("alter")
        element.text = alter
        pitch.insert(list(pitch).index(pitch.find("step")) + 1, element)


def clefs_before_barlines(part: ET.Element) -> None:
    """A clef change that opens a bar goes to the end of the bar before. The print draws a change
    at a barline before the barline, and gold puts it there; the model writes it as the next bar's
    first sign (Polish 095, 108). That includes a lower staff's clef after the upper staff's notes
    and a <backup>, as when a system opens in a new clef that the print shows as a courtesy clef
    at the end of the line before (081, 091). The first bar's clefs are the system's own and stay.
    Appended last, the clef lands at the bar's end because the delinearizer ends each bar on a
    voice that fills it; a bar ending in <backup> would put it earlier."""
    measures = part.findall("measure")
    for before, measure in zip(measures, measures[1:]):
        for opening in attributes_at_bar_start(measure):
            clefs = opening.findall("clef")
            for clef in clefs:
                opening.remove(clef)
            ET.SubElement(before, "attributes").extend(clefs)
            if len(opening) == 0:
                measure.remove(opening)


def attributes_at_bar_start(measure: ET.Element) -> list[ET.Element]:
    """The bar's <attributes> holding a clef that come before any of its time passes in their voice."""
    return [element for element, onset in timed(measure)
            if element.tag == "attributes" and onset == 0 and element.find("clef") is not None]


def state_grace_slashes(part: ET.Element) -> None:
    for grace in part.iter("grace"):
        grace.set("slash", grace.get("slash", "no"))


def tuplet_groups(part: ET.Element) -> list[tuple[ET.Element, list[ET.Element]]]:
    """Each tuplet's start mark with the notes of its voice from that start to its stop."""
    groups = []
    for events in events_by_voice(part).values():
        open_group = None
        for event in events:
            for note in event:
                start = next((t for t in note.findall("notations/tuplet") if t.get("type") == "start"), None)
                if start is not None:
                    open_group = (start, [])
                    groups.append(open_group)
                if open_group is not None:
                    open_group[1].append(note)
                if any(t.get("type") == "stop" for t in note.findall("notations/tuplet")):
                    open_group = None
    return groups


def state_tuplet_brackets(part: ET.Element) -> None:
    """A beam already groups a tuplet, so only unbeamed ones take a bracket. Any beam in the
    group counts: a triplet that opens with a rest (Polish 003) has none on its first note."""
    for start, notes in tuplet_groups(part):
        start.set("bracket", "no" if any(note.find("beam") is not None for note in notes) else "yes")


def tuplets_closed(part: ET.Element) -> None:
    """Groups the tuplet notes the model left unmarked, which imply_tuplets gave a ratio but no
    start or stop: music21 reads such notes as one group running on (Polish 077, 096). A group
    closes once it fills its ratio's count of its shortest note, within one bar, and shows no
    number, as the model saw none. Notes inside a group the model opened are left to it."""
    measure_of = measure_index(part)
    for events in events_by_voice(part).values():
        group, inside, bar = [], False, None
        for event in events:
            first = event[0]
            if measure_of[first] != bar:
                group, inside, bar = [], False, measure_of[first]
            marks = [tuplet.get("type") for note in event for tuplet in note.findall("notations/tuplet")]
            if inside or marks or first.find("time-modification") is None or first.find("grace") is not None:
                inside = still_open(inside, marks)
                group = []
                continue
            group.append(first)
            if fills_its_ratio(group):
                mark_hidden_tuplet(group[0], group[-1])
                group = []


def still_open(inside: bool, marks: list[str]) -> bool:
    if "stop" in marks:
        return False
    return inside or "start" in marks


def fills_its_ratio(group: list[ET.Element]) -> bool:
    actual = int(group[0].findtext("time-modification/actual-notes"))
    shortest = min(duration(note) for note in group)
    return len(group) > 1 and sum(duration(note) for note in group) == actual * shortest


def mark_hidden_tuplet(first: ET.Element, last: ET.Element) -> None:
    for note, kind in ((first, "start"), (last, "stop")):
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        attributes = {"type": kind, "bracket": "no", "show-number": "none"} if kind == "start" else {"type": kind}
        notations.append(ET.Element("tuplet", attributes))


def duration(element: ET.Element) -> int:
    return int(element.findtext("duration", "0"))


def timed(measure: ET.Element):
    """Each of the measure's elements with where it starts, in divisions. A chord note starts with
    the note before it, and a grace note takes no time."""
    position = start = 0
    for element in measure:
        if element.tag == "backup":
            position -= duration(element)
        elif element.tag == "forward":
            position += duration(element)
        elif element.tag == "note" and element.find("chord") is None:
            start = position
            if element.find("grace") is None:
                position += duration(element)
        yield element, (start if element.tag == "note" else position)


def notes_by_onset(measure: ET.Element) -> dict[int, list[ET.Element]]:
    """A measure's notes other than grace notes, by where they start in divisions."""
    onsets = defaultdict(list)
    for element, onset in timed(measure):
        if element.tag == "note" and element.find("grace") is None:
            onsets[onset].append(element)
    return onsets


def shared_heads_shown_alike(part: ET.Element) -> None:
    """Voices sounding one pitch at once on one staff share a printed head, and so its accidental.
    The model writes the accidental in one voice only (Polish 037, 088); gold in each."""
    for measure in part.findall("measure"):
        for notes in notes_by_onset(measure).values():
            heads = defaultdict(list)
            for note in notes:
                if note.find("pitch") is not None:
                    heads[(note.findtext("staff", "1"), pitch_of(note))].append(note)
            for unison in heads.values():
                shown = next((note.find("accidental") for note in unison if note.find("accidental") is not None), None)
                for note in unison:
                    if shown is not None and note.find("accidental") is None:
                        add_accidental(note, shown.text)


def add_accidental(note: ET.Element, kind: str) -> None:
    # MusicXML puts <accidental> after <type> and its <dot>s.
    before = [child for child in note if child.tag in ("type", "dot")]
    index = list(note).index(before[-1]) + 1 if before else len(note)
    accidental = ET.Element("accidental")
    accidental.text = kind
    note.insert(index, accidental)


TIME_SYMBOLS = {("4", "4"): "common", ("2", "2"): "cut"}


def time_symbols_drawn(part: ET.Element) -> None:
    """4/4 and 2/2 as C and cut C. The model's tokens give only the numbers, and the prints of
    this era draw these two meters as symbols (Polish 077, 079)."""
    for time in part.iter("time"):
        symbol = TIME_SYMBOLS.get((time.findtext("beats"), time.findtext("beat-type")))
        if symbol:
            time.set("symbol", symbol)


def finish_score(score: ET.Element) -> None:
    """Also clears the part name: the delinearizer's is a constant, not something read."""
    for name in score.iter("part-name"):
        name.text = ""
    for part in score.findall("part"):
        ties_across_system_breaks(part)
        pair_slurs(part)
        ties_from_slurs(part)
        mend_ties(part)
        clefs_before_barlines(part)
        shared_heads_shown_alike(part)
        state_grace_slashes(part)
        state_tuplet_brackets(part)
        time_symbols_drawn(part)
        tuplets_closed(part)  # after the brackets: a group with no number shown takes no bracket either
