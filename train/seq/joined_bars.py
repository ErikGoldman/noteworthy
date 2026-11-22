"""Bars drawn as one, so that a model can find where a bar ends only by its notes adding up to the
meter (TODO, priority 1, item 4). Fed every note before it right, (l) finds a bar's end 98% of the
time with the barline and 18% without (EXPERIMENTS.md, "Three diagnostics without training"): the
line is its only cue. Erasing barlines in the pixels leaves the engraver's extra space around them
(Verovio pads each bar: a boundary drawn with an invisible barline is still 26% wider than the space
between two notes), so the bars are joined in the MusicXML before it is drawn instead: Verovio does
not count beats and spaces a long bar's notes evenly. The label keeps every bar.

A boundary is joined only where nothing is written at it: no barline element (repeats, endings,
double bars), no change of time, key, clef or divisions opening the next bar (an engraver marks
those at a bar's start anyway), and no whole-bar rest on either side (it is drawn centred in its
bar, and often written with no note type, so as a plain rest Verovio draws nothing: four bars of
rest became one drawn rest, 2026-11-20). Nor is a bar with a clef or other attributes after its
first note, or an ottava sign: they act on one staff from where they stand, and written voice by
voice (below) they would act on other notes (octaves moved in a few systems, 2026-11-20).

The joined bar is written voice by voice: each voice's notes from all its bars in one run, a
`forward` in that voice wherever it is silent, and one `backup` to the bar's start between voices.
Appending one bar's content after another's (voice 1, backup, voice 2, then voice 1 again...) is
valid MusicXML, but Verovio's import then moved notes in 598 of 615 joined systems, a voice after a
mid-bar backup drawn a bar late (its timemap against the plain drawing's, 2026-11-20).
Each element that is not a note (directions, harmony, attributes) travels with the note after it."""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass

CHANGES = ("time", "key", "clef", "divisions")
JOINED_ODDS = 0.5


@dataclass
class Event:
    """A note (with the elements written before it) at its onset, in divisions from the joined bar's start."""
    voice: str
    staff: str | None
    onset: int
    advance: int
    elements: list[ET.Element]


def duration(child: ET.Element) -> int:
    return int(child.findtext("duration", "0"))


def events_of(measure: ET.Element, offset: int) -> tuple[list[Event], list[ET.Element], int]:
    """(the measure's notes as events, the elements after its last note, its length)."""
    cursor = furthest = last_onset = 0
    out, pending = [], []
    for child in measure:
        if child.tag in ("backup", "forward"):
            cursor += duration(child) if child.tag == "forward" else -duration(child)
            furthest = max(furthest, cursor)
            continue
        if child.tag != "note":
            pending.append(child)
            continue
        chord, grace = child.find("chord") is not None, child.find("grace") is not None
        onset = last_onset if chord else cursor
        advance = 0 if chord or grace else duration(child)
        out.append(Event(child.findtext("voice", "1"), child.findtext("staff"), offset + onset, advance, pending + [child]))
        pending = []
        if not chord:
            last_onset = cursor
        cursor += advance
        furthest = max(furthest, cursor)
    return out, pending, furthest


def element(tag: str, length: int, voice: str | None = None, staff: str | None = None) -> ET.Element:
    out = ET.Element(tag)
    ET.SubElement(out, "duration").text = str(length)
    if voice is not None:
        ET.SubElement(out, "voice").text = voice
    if staff is not None:
        ET.SubElement(out, "staff").text = staff
    return out


def joined(measures: list[ET.Element]) -> ET.Element:
    """One measure holding the measures' notes, written voice by voice (see the module's note).
    ValueError where a voice's notes overlap in time (not a voice Verovio can draw as one)."""
    events, trailing, total = [], [], 0
    for measure in measures:
        found, after, length = events_of(measure, total)
        events += found
        trailing += after
        total += length
    out = ET.Element("measure", measures[0].attrib)
    voices = list(dict.fromkeys(e.voice for e in events))
    for k, voice in enumerate(voices):
        if k:
            out.append(element("backup", total))
        cursor, staff = 0, None
        for event in sorted((e for e in events if e.voice == voice), key=lambda e: e.onset):  # stable: chords, graces keep their order
            staff = event.staff or staff
            if event.onset > cursor:
                out.append(element("forward", event.onset - cursor, voice, staff))
            elif event.onset < cursor and event.advance:
                raise ValueError(f"voice {voice}'s notes overlap at {event.onset}")
            out.extend(event.elements)
            cursor = max(cursor, event.onset + event.advance)
        if cursor < total:
            out.append(element("forward", total - cursor, voice, staff))
    out.extend(trailing)
    return out


def has_whole_bar_rest(measure: ET.Element) -> bool:
    return any(rest.get("measure") == "yes" for rest in measure.iter("rest"))


def changes_within(measure: ET.Element) -> bool:
    """Attributes after the measure's first note, or an ottava sign anywhere in it."""
    first_note = next((k for k, child in enumerate(measure) if child.tag == "note"), len(measure))
    late_attributes = any(child.tag == "attributes" for child in list(measure)[first_note:])
    return late_attributes or measure.find("direction/direction-type/octave-shift") is not None


def joinable(measure: ET.Element, following: ET.Element) -> bool:
    if measure.find("barline") is not None or following.find("barline") is not None:
        return False
    if any(has_whole_bar_rest(m) or changes_within(m) for m in (measure, following)):
        return False
    if any(p.get("new-system") == "yes" or p.get("new-page") == "yes" for p in following.findall("print")):
        return False
    return not any(attributes.find(change) is not None
                   for attributes in following.findall("attributes") for change in CHANGES)


def runs_of(chosen: list[int]) -> list[list[int]]:
    """Consecutive chosen boundaries as runs: boundaries 2 and 3 join measures 2, 3 and 4."""
    out: list[list[int]] = []
    for k in chosen:
        if out and out[-1][-1] == k - 1:
            out[-1].append(k)
        else:
            out.append([k])
    return out


def with_bars_joined(part: ET.Element, rng: random.Random, skip: tuple[int, int] = (0, 0)) -> int:
    """Joins each joinable boundary between the part's measures at JOINED_ODDS (at least one where
    any can be), leaving the first skip[0] and last skip[1] measures (a drawn system's neighbours)
    as they are. Returns how many boundaries were joined; ValueError as joined()."""
    measures = part.findall("measure")
    own = measures[skip[0]:len(measures) - skip[1]]
    boundaries = [k for k in range(len(own) - 1) if joinable(own[k], own[k + 1])]
    if not boundaries:
        return 0
    chosen = [k for k in boundaries if rng.random() < JOINED_ODDS] or [rng.choice(boundaries)]
    for run in runs_of(chosen):
        group = own[run[0]:run[-1] + 2]
        at = list(part).index(group[0])
        for measure in group:
            part.remove(measure)
        part.insert(at, joined(group))
    return len(chosen)
