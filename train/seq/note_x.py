"""Where each note and rest of a label is drawn across its crop (TODO, priority 1: a position output, so
that columns, which notes stand together, come from the page and not from a reading's own onsets).

For a system the dataset builder renders (build_seq_dataset.render_system), each label head (a note or
rest as the linearizer writes it, chord members and grace notes included, in the label's order) gets the
x of its notehead (or rest) in the crop as a share of the crop's width, or None where nothing of it is
drawn (a hidden rest) or it cannot be matched. The drawing and the label are matched through the MEI
Verovio made of the same MusicXML: each MusicXML note's (measure, staff, onset, pitch) against each MEI
note's, in order; the SVG element of an MEI note carries its id.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction

from lxml import etree

SVG = "{http://www.w3.org/2000/svg}"
MEI = "{http://www.music-encoding.org/ns/mei}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"
MEI_CONTAINERS = {"beam", "tuplet", "btrem", "ftrem", "graceGrp"}
MEI_TIMED = {"note", "chord", "rest", "space", "mRest", "mSpace"}
TRANSLATE = re.compile(r"translate\(\s*([-\d.eE]+)")
ONSET_DENOMINATOR = 48  # onsets matched as the nearest simple fraction: Verovio's ticks drift in tuplets


def snapped(onset: Fraction) -> Fraction:
    return onset.limit_denominator(ONSET_DENOMINATOR)


def page_x(svg: bytes) -> dict[str, float]:
    """Each drawn note's, rest's and chord's x on the page as layout_svg sized it, in pixels: its first
    glyph's position in Verovio's units, plus the page margin, scaled by the page's px per unit."""
    root = etree.fromstring(svg)
    inner = root.find(f"{SVG}svg")
    vb = [float(v) for v in inner.get("viewBox").split()]
    px_per_unit = float(root.get("width").removesuffix("px")) / vb[2]
    margin = next((g for g in inner.iter(f"{SVG}g") if "page-margin" in (g.get("class") or "")), None)
    offset = float(TRANSLATE.search(margin.get("transform")).group(1)) if margin is not None and margin.get("transform") else 0.0
    out = {}
    for g in inner.iter(f"{SVG}g"):
        if (g.get("class") or "").split()[:1] in (["note"], ["rest"], ["chord"]):
            use = next(g.iter(f"{SVG}use"), None)
            if use is None:
                continue
            found = TRANSLATE.search(use.get("transform") or "")
            x = float(found.group(1)) if found else use.get("x")
            if x is not None:
                out[g.get("id")] = (float(x) + offset - vb[0]) * px_per_unit
    return out


def pitch_of(step: str | None, octave: str | None) -> str:
    return f"{step.upper()}{octave}" if step and octave else "rest"


def musicxml_heads(part: ET.Element) -> list[tuple[int, str, Fraction, str]]:
    """Each note and rest of the part, in document order: (measure index, staff, onset in quarters, pitch
    or "rest"); a chord member at its chord's onset, a grace note at the note it precedes."""
    out = []
    divisions = 1
    for m, measure in enumerate(part.findall("measure")):
        time = last = Fraction(0)
        for child in measure:
            if child.tag == "attributes" and child.findtext("divisions"):
                divisions = int(child.findtext("divisions"))
            elif child.tag in ("backup", "forward"):
                step = Fraction(int(child.findtext("duration", "0")), divisions)
                time += step if child.tag == "forward" else -step
            elif child.tag == "note":
                chord, grace = child.find("chord") is not None, child.find("grace") is not None
                onset = last if chord else time
                pitch = child.find("pitch")
                out.append((m, child.findtext("staff", "1"), snapped(onset),
                            pitch_of(pitch.findtext("step"), pitch.findtext("octave")) if pitch is not None else "rest"))
                if not chord and not grace:
                    last = time
                    time += Fraction(int(child.findtext("duration", "0")), divisions)
    return out


def mei_heads(mei: str) -> list[tuple[str, tuple[int, str, Fraction, str]]]:
    """Each MEI note and rest Verovio drew: (its id, (measure index, staff, onset in quarters, pitch or
    "rest")); a chord's notes at the chord's onset."""
    root = ET.fromstring(mei)
    ppq = {sd.get("n"): int(sd.get("ppq")) for sd in root.iter(f"{MEI}staffDef") if sd.get("ppq")}
    out = []
    for m, measure in enumerate(root.iter(f"{MEI}measure")):
        for staff in measure.iter(f"{MEI}staff"):
            n = staff.get("n")
            for layer in staff.iter(f"{MEI}layer"):
                for element, ticks in timed(layer):
                    tag = element.tag.removeprefix(MEI)
                    onset = snapped(Fraction(ticks, ppq.get(n, 1)))
                    if tag == "rest":
                        out.append((element.get(XMLID), (m, n, onset, "rest")))
                    elif tag == "note":
                        out.append((element.get(XMLID), (m, n, onset, pitch_of(element.get("pname"), element.get("oct")))))
                    elif tag == "chord":
                        out += [(note.get(XMLID), (m, n, onset, pitch_of(note.get("pname"), note.get("oct"))))
                                for note in element.iter(f"{MEI}note")]
    return out


def timed(element: ET.Element, cursor: int = 0) -> list[tuple[ET.Element, int]]:
    """Each timed element under element in order, with its onset in ticks (dur.ppq; grace notes none)."""
    out = []
    for child in element:
        tag = child.tag.removeprefix(MEI)
        if tag in MEI_CONTAINERS:
            inner = timed(child, cursor)
            out += inner
            cursor += sum(int(e.get("dur.ppq", "0")) for e, _ in inner if e.get("grace") is None
                          and e.tag.removeprefix(MEI) in MEI_TIMED)
        elif tag in MEI_TIMED:
            out.append((child, cursor))
            if child.get("grace") is None:
                cursor += int(child.get("dur.ppq", "0"))
    return out


def head_x(part: ET.Element, mei: str, svg: bytes, crop_left: float, crop_width: float) -> list[float | None]:
    """Each MusicXML note's and rest's x in the crop as a share of its width (the label's heads, in order):
    crop_left is the page pixel the crop's left edge stands at. None where no drawn MEI note matches."""
    drawn = page_x(svg)
    pool: dict[tuple, list[str]] = defaultdict(list)
    for id_, key in mei_heads(mei):
        pool[key].append(id_)
    out = []
    for key in musicxml_heads(part):
        ids = pool.get(key)
        id_ = ids.pop(0) if ids else None
        x = drawn.get(id_) if id_ else None
        out.append(None if x is None else round((x - crop_left) / crop_width, 5))
    return out
