"""Renders system crops with LMX transcriptions from our training scores, for the
sequence model (train_seq.py).

usage: build_seq_dataset.py --scores LIST.txt --out DIR [--systems-per-score K] [--workers W]
                            [--unmarked-tuplets] [--thick-beams] [--round-heads] [--marks]
                            [--engraver-hiding] [--varied-tuplet-looks]

Held-out scores (omr.held_out) and PDMX scores with a license conflict are left out. Each two-staff (piano) part is cut into systems of a few measures, which OLiMPiC's code
turns into self-contained parts: clefs and key repeated at each system start, the time
signature only where the score prints one (split_part_to_systems), as an engraver lays
them out. K random systems per score are rendered alone with Verovio under random
engraving options and cropped to their ink with random margins. Their LMX comes from
OLiMPiC's linearizer, as for the OLiMPiC and GrandStaff-LMX datasets.

With --unmarked-tuplets only systems with tuplets are taken, drawn without their numbers
(the first may stay): engravers mark the first triplets of a passage and leave the rest,
and a model that only ever saw numbers reads the others as plain notes.

Otherwise one system in five with tuplets hides every number (the first may stay), unless
--engraver-hiding hides them as real scores do: by score, then by place in a run of like groups
(seq.tuplet_display). --varied-tuplet-looks sets each score's printed numbers its own way: above
or below, bracketed or not, 3 or 3:2, some far from their notes, small, or in an arc (TODO 3c).

With --thick-beams beams are drawn heavier, as on old plates and in copyists' hands, where
a single beam can be as thick as a printed double one. A model that only ever saw thin
beams counts them by thickness and reads such eighths as 16ths.

With --round-heads filled noteheads are round dots, as in copyists' hands.

With --marks random segnos and codas are added above bars, and an ottava passage over some
systems: 8va above either staff or 8vb below the lower, signed `8va` or `8` alone, its line dashed
or dotted, as plates vary (Polish 110 has a dotted `8` over each staff). A segno looks like an `8`,
and only 4% of real crops have any ottava. Bass-clef sharps follow old plates, with A sharp on the
top line: a model that knows only the modern layout reads five of them as three.

Labels give pitches as printed, ottavas as tokens (seq.ottava).

Writes DIR/samples/<score index>/s<k>.png + .lmx, and DIR/samples.train.txt listing
every sample. A score's directory is written whole or not at all, so an interrupted
run resumes where it stopped. Scores that hang inside Verovio are listed in
DIR/hung.txt and skipped by later runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import verovio
from lxml import etree

from omr.fonts import use_smufl_fonts
from omr.held_out import is_held_out
from omr.pieces import pdmx_license_conflicts
from omr.render import FONTS, SVG_NS, UNITS_PER_INTERLINE_PER_UNIT, layout_svg
from seq.staff_positions import Staves, staff_count
from seq.ottava import TOKEN_ATTRIBUTE, move_octaves, notes_under_ottavas, written_part
from seq.barlines import barline_tokens_of, with_barline_tokens
from seq.time_symbols import TOKEN_OF_SYMBOL, token_of, with_time_symbol
from seq.tuplet_display import PLAIN_LOOK, ScoreTuplets, TupletDrawing, TupletLook, hide_as_engravers_do, set_look
from seq.tuplet_numbers import HIDDEN, START, starts_hidden_tuplet
from seq.tuplet_svg import restyle_tuplets

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("NW_DATA", ROOT / "data"))
sys.path.insert(0, str(DATA / "raw" / "olimpic" / "code" / "repo"))
from app.linearization.Linearizer import Linearizer  # noqa: E402
from app.linearization.vocabulary import PITCH_TOKENS  # noqa: E402
from app.symbolic.part_to_score import part_to_score  # noqa: E402
from app.symbolic.split_part_to_systems import split_part_to_systems  # noqa: E402

MEASURES_PER_SYSTEM = (2, 6)
MAX_TOKENS = 1000          # the model's sequence limit, with room for <bos>/<eos>
INTERLINE_PX = (14.0, 26.0)
MARGIN_INTERLINES = (0.5, 3.0)
HUNG_SCORE_S = 300         # a score not done by then is hung inside Verovio
PART_NAME_HIDDEN = 0.85    # only a score's first system shows the instrument name
KEEP_FIRST_TUPLET_NUMBER = 0.5
TUPLET_NUMBERS_HIDDEN = 0.2  # of systems with tuplets, as engravers do once a passage settles
# Only a few percent of PDMX scores mark C or cut time, and only a score's first system shows it.
METER_SYMBOL_DRAWN = 0.3
METERS_WITH_SYMBOLS = {("4", "4"): "common", ("2", "2"): "cut"}
SINGLE_BEAM_THICKENING = (1.0, 2.6)
STACKED_BEAM_THICKENING = 1.35  # beyond this, stacked beams touch and their number is lost
ROUND_HEAD_RADIUS_INTERLINES = (0.42, 0.55)
# Some plates cut naturals small and thin, and at a page scan's resolution they look like sharps
# (Polish 055, 022, 087); Verovio's are large and clear.
NATURALS_SHRUNK = 0.3
# French and other 19th-century plates draw the quarter rest as a mirrored eighth rest (SMuFL
# restQuarterOld). Verovio never does; on OLiMPiC's scans seq512 reads it as an eighth rest and
# seq513 as a half rest (197 of 8,314 visible rests).
OLD_QUARTER_RESTS = 0.2
QUARTER_REST, OLD_QUARTER_REST = "E4E5", "E4F2"
NATURAL_SCALE = (0.6, 0.85)
FILLED_NOTEHEAD = "#E0A4"  # SMuFL noteheadBlack; Verovio appends a per-file suffix
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
SVG_UNITS_PER_INTERLINE = 180
NOTEHEAD_CENTRE_UNITS = 106  # a glyph is placed by its left edge
REPEAT_MARK_CHANCE = 0.4
REPEAT_MARKS = ("segno", "segno", "coda")
OTTAVA_PASSAGE_CHANCE = 0.5
PLAIN_LINE_CHANCE = 0.25
SYSTEM_END_PASSAGE_CHANCE = 0.4  # Polish 023: an 8va line across whole systems
HOOKLESS_OTTAVA_CHANCE = 0.3
OTTAVA_PASSAGE_NOTES = (3, 40)
OTTAVA_DASHES = ("36 72", "18 54", "18 36", "60 60", "90 45")  # Verovio's own is the first
DOTTED_OTTAVA_CHANCE = 0.5
OTTAVA_DOTS = ("1 40", "1 60", "1 90")  # round caps make each 1-unit dash a dot
OTTAVA_LINE_THICKNESS = (0.1, 0.35)  # Verovio's default is 0.2 interlines
ALTERNATIVE_OTTAVA_SYMBOL_CHANCE = 0.5  # `8` alone, as on older plates, for `8va`
# Polish 023: an upright bold `8` over a chain of short slanted dashes. The model reads the sign
# when both are made Verovio's (an italic `8`, dots) and misses it when either is left as printed,
# and every music font's glyph is italic: plates set the sign in their own type.
TEXT_OTTAVA_CHANCE = 0.4
TEXT_OTTAVA_INTERLINES = (1.3, 2.3)  # the font size
TEXT_OTTAVA_FONTS = ("DejaVu Serif", "DejaVu Sans", "Ubuntu")
# SMuFL 8, 8va, 8vb, 8ba, 8vb (not 15 or 22), and what plates print for each
TEXT_OF_OTTAVA_GLYPH = {"#E510": ("8", "8."), "#E511": ("8", "8va", "8va.", "8ve"), "#E512": ("8", "8vb"),
                        "#E513": ("8", "8ba"), "#E51C": ("8", "8vb")}
CHAINED_OTTAVA_CHANCE = 0.3
CHAIN_LINK_INTERLINES = (0.25, 0.45)  # one dash and the gap after it
HIGHEST_OCTAVE, LOWEST_OCTAVE = 7, 1
F_CLEF, SHARP, NATURAL = "#E062", "#E262", "#E261"  # SMuFL; Verovio appends a per-file suffix
OCTAVE_UNITS = 3.5 * SVG_UNITS_PER_INTERLINE


@dataclass(frozen=True)
class Style:
    """Departures from Verovio's engraving, for hands it does not draw."""
    thick_beams: bool = False
    round_heads: bool = False
    marks: bool = False


def is_training_score(source: str, conflicts: set[str]) -> bool:
    """Not held out, and not a PDMX score with a license conflict (pdmx_license_conflicts)."""
    return bool(source.strip()) and not is_held_out(Path(source)) and Path(source).name not in conflicts


def load_musicxml(path: Path) -> ET.Element:
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile").get("full-path")
            return ET.fromstring(z.read(rootfile))
    return ET.parse(path).getroot()


def piano_part(score: ET.Element) -> ET.Element | None:
    """The first part notated on two staves."""
    for part in score.findall("part"):
        staves = part.find("measure/attributes/staves")
        if staves is not None and staves.text == "2":
            return part
    return None


def without_octave_clefs(part: ET.Element) -> None:
    """Plain clefs for octave clefs (a treble clef with an 8 below, 2% of piano parts). Our clef
    tokens have no octave and labels give sounding pitch, so a crop drawn with the 8 would put
    every note an octave from its label's; drawn plain, the notes sit where they sound."""
    for clef in part.iter("clef"):
        for change in clef.findall("clef-octave-change"):
            clef.remove(change)


def break_into_systems(part: ET.Element, rng: random.Random) -> None:
    """Replaces the part's layout with system breaks every few measures."""
    measures = part.findall("measure")
    for m in measures:
        for p in m.findall("print"):
            m.remove(p)
    k = 0
    while k < len(measures):
        if k:
            measures[k].insert(0, ET.Element("print", {"new-system": "yes"}))
        k += rng.randint(*MEASURES_PER_SYSTEM)


def hide_part_name(score: ET.Element) -> None:
    for name in score.iter("part-name"):
        name.text = ""


def has_tuplets(part: ET.Element) -> bool:
    return part.find(".//time-modification") is not None


def hide_tuplet_numbers(part: ET.Element, rng: random.Random) -> None:
    starts = [t for t in part.iter("tuplet") if t.get("type") == "start"]
    keep_first = rng.random() < KEEP_FIRST_TUPLET_NUMBER
    for tuplet in starts[1:] if keep_first else starts:
        tuplet.set("show-number", "none")
        tuplet.set("bracket", "no")


def draw_tuplet_numbers(part: ET.Element, unmarked: bool, tuplets: ScoreTuplets, rng: random.Random) -> None:
    """Which numbers the system hides: all for the unmarked set (the first may stay), else as the
    score's engraver would (tuplets.hiding), else the blind rule; then the score's look."""
    if unmarked:
        hide_tuplet_numbers(part, rng)
    elif tuplets.hiding is not None:
        hide_as_engravers_do(part, tuplets.hiding, rng)
    elif has_tuplets(part) and rng.random() < TUPLET_NUMBERS_HIDDEN:
        hide_tuplet_numbers(part, rng)
    set_look(part, tuplets.look)


def draw_some_meters_as_symbols(part: ET.Element, rng: random.Random) -> None:
    """Some 4/4 and 2/2 signatures drawn as C and cut-C, for labels and image alike."""
    for time in part.iter("time"):
        symbol = METERS_WITH_SYMBOLS.get((time.findtext("beats"), time.findtext("beat-type")))
        if symbol and time.get("symbol") not in TOKEN_OF_SYMBOL and rng.random() < METER_SYMBOL_DRAWN:
            time.set("symbol", symbol)


class WrittenLinearizer(Linearizer):
    """OLiMPiC's linearizer, with each tagged note's ottava token straight after its pitch,
    each measure's barline signs as tokens (seq.barlines), time signatures' symbols
    (seq.time_symbols) and tuplets whose number is not drawn (seq.tuplet_numbers)."""
    def process_measure(self, measure: ET.Element):
        start = len(self.output_tokens)
        super().process_measure(measure)
        self.output_tokens[start:] = with_barline_tokens(self.output_tokens[start:], *barline_tokens_of(measure))

    def process_time_signature(self, time: ET.Element):
        super().process_time_signature(time)
        self.output_tokens[:] = with_time_symbol(self.output_tokens, token_of(time))

    def process_note(self, note: ET.Element, measure: ET.Element):
        self._ottava_token = note.get(TOKEN_ATTRIBUTE)
        self._hides_tuplet_number = starts_hidden_tuplet(note)
        super().process_note(note, measure)

    def _emit(self, token: str):
        super()._emit(token)
        if token in PITCH_TOKENS and self._ottava_token:
            self.output_tokens.append(self._ottava_token)
        if token == START and self._hides_tuplet_number:
            self.output_tokens.append(HIDDEN)


def linearize(part: ET.Element) -> str:
    linearizer = WrittenLinearizer(errout=None)
    linearizer.process_part(written_part(part))
    return " ".join(linearizer.output_tokens)


def ottavas_drawn_as_labelled(tk: verovio.toolkit, part: ET.Element, neighbours: tuple[int, int] = (0, 0)) -> bool:
    """False if Verovio shifted other notes than the labels say. A sign holds for its whole
    staff, as in MuseScore, but Verovio leaves out some voices of the same staff. neighbours:
    the measures drawn before and after the system (with_neighbours), which are not counted."""
    labelled = len(notes_under_ottavas(part))
    if not labelled:
        return True
    measures = re.split(r"<measure\b", tk.getMEI())[1:]
    own = measures[neighbours[0]:len(measures) - neighbours[1]]
    notes = [dict(re.findall(r'(oct(?:\.ges)?)="(\d+)"', note)) for m in own for note in re.findall(r"<note [^>]*>", m)]
    return labelled == sum(note.get("oct") != note.get("oct.ges", note.get("oct")) for note in notes)


def direction(child: ET.Element, staff: int, placement: str = "above") -> ET.Element:
    element = ET.Element("direction", placement=placement)
    ET.SubElement(element, "direction-type").append(child)
    ET.SubElement(element, "staff").text = str(staff)
    return element


def add_repeat_mark(part: ET.Element, rng: random.Random) -> None:
    """A segno or coda at the start of a random bar. Neither changes a single label."""
    measure = rng.choice(part.findall("measure"))
    first_note = next(i for i, e in enumerate(measure) if e.tag == "note")
    measure.insert(first_note, direction(ET.Element(rng.choice(REPEAT_MARKS)), 1))


def first_voice_notes(measure: ET.Element, staff: int) -> list[int]:
    """Where in the measure the staff's first voice has a note, chord members and graces left out."""
    on_staff = [(i, e) for i, e in enumerate(measure) if e.tag == "note" and e.findtext("staff", "1") == str(staff)]
    voice = on_staff[0][1].findtext("voice", "1") if on_staff else None
    return [i for i, e in on_staff if e.findtext("voice", "1") == voice and e.find("chord") is None and e.find("grace") is None]


def passage(part: ET.Element, staff: int, rng: random.Random) -> tuple[tuple[ET.Element, int], tuple[ET.Element, int]] | None:
    """A random run of the staff's first voice, within a bar or across several: where its first
    note is, and where the element after its last note (and that note's chord) goes. Some runs
    take the whole system from one end, as a sign carried over a system break does."""
    places = [(m, i) for m in part.findall("measure") for i in first_voice_notes(m, staff)]
    shortest, longest = OTTAVA_PASSAGE_NOTES
    if len(places) < shortest:
        return None
    start = rng.randrange(len(places) - shortest + 1)
    stop = min(len(places), start + rng.randint(shortest, longest)) - 1
    if rng.random() < SYSTEM_END_PASSAGE_CHANCE:
        start, stop = (0, stop) if rng.random() < 0.5 else (start, len(places) - 1)
    (first_measure, first), (last_measure, last) = places[start], places[stop]
    after_last = last + 1
    while after_last < len(last_measure) and last_measure[after_last].find("chord") is not None:
        after_last += 1
    return (first_measure, first), (last_measure, after_last)


def spanned(part: ET.Element, staff: int, placement: str, start: ET.Element, stop: ET.Element, rng: random.Random) -> bool:
    """Puts start and stop around a random passage of the staff; False if it has none."""
    found = passage(part, staff, rng)
    if found is None:
        return False
    (first_measure, first), (last_measure, after_last) = found
    # Stop first: in a one-bar passage the start's insertion would move it.
    last_measure.insert(after_last, direction(stop, staff, placement))
    first_measure.insert(first, direction(start, staff, placement))
    return True


def add_ottava_passage(part: ET.Element, rng: random.Random) -> None:
    """An 8va over a run of either staff or an 8vb under the lower one. The notes move the other
    way first, so they are drawn where the music has them and only the sign is new."""
    if next(part.iter("octave-shift"), None) is not None:
        return
    staff, kind, placement = rng.choice([(1, "down", "above"), (2, "up", "below"), (2, "down", "above")])
    if not spanned(part, staff, placement, ET.Element("octave-shift", type=kind, size="8"),
                   ET.Element("octave-shift", type="stop", size="8"), rng):
        return
    for note, octaves in notes_under_ottavas(part):
        move_octaves(note, octaves)
        if not LOWEST_OCTAVE <= int(note.findtext("pitch/octave")) <= HIGHEST_OCTAVE:
            raise ValueError("the passage leaves the piano's range")


def add_plain_line(part: ET.Element, rng: random.Random) -> None:
    """A solid line with no sign between the staves, over a run of notes, and no label: plates
    have such lines, and a model that knows only 8va lines reads one as a sign (Polish 049)."""
    staff, placement = rng.choice([(1, "below"), (2, "above")])
    spanned(part, staff, placement, ET.Element("bracket", type="start", number="2", **{"line-end": "none", "line-type": "solid"}),
            ET.Element("bracket", type="stop", number="2", **{"line-end": "none"}), rng)


def add_marks(part: ET.Element, rng: random.Random) -> None:
    if rng.random() < REPEAT_MARK_CHANCE:
        add_repeat_mark(part, rng)
    if rng.random() < OTTAVA_PASSAGE_CHANCE:
        add_ottava_passage(part, rng)
    if rng.random() < PLAIN_LINE_CHANCE:
        add_plain_line(part, rng)


EDGE_MARGINS = ("rightMarginClef", "rightMarginKeySig", "rightMarginMeterSig", "leftMarginRightBarLine")
EDGE_MARGIN_UNITS = (1.0, 2.0)   # Verovio's default and its maximum, in half staff spaces


def engraving_options(rng: random.Random) -> dict:
    return {
        "font": rng.choice(FONTS), "breaks": "none", "adjustPageWidth": True, "adjustPageHeight": True,
        "pageMarginLeft": 0, "pageMarginRight": 0, "pageMarginTop": 0, "pageMarginBottom": 0,
        "header": "none", "footer": "none", "unit": 9.0, "scale": 100,
        "spacingStaff": rng.randint(6, 16), "spacingLinear": rng.uniform(0.15, 0.4),
        "spacingNonLinear": rng.uniform(0.45, 0.75), "staffLineWidth": rng.uniform(0.1, 0.3),
        "stemWidth": rng.uniform(0.1, 0.35), "barLineWidth": rng.uniform(0.15, 0.5),
        "ledgerLineThickness": rng.uniform(0.15, 0.4), "beamMaxSlope": rng.randint(5, 20),
        "svgBoundingBoxes": False, "svgHtml5": False,
        # Room at a system's edges, from Verovio's tightest to as much as it allows (prints leave more):
        # where a tie or slur runs on from or to the next system (with_neighbours), its half arc
        # fills it. Every system gets it, so that the room itself tells nothing.
        **{margin: rng.uniform(*EDGE_MARGIN_UNITS) for margin in EDGE_MARGINS},
    }


def ottava_options(rng: random.Random) -> dict:
    return {"octaveAlternativeSymbols": rng.random() < ALTERNATIVE_OTTAVA_SYMBOL_CHANCE,
            "octaveLineThickness": rng.uniform(*OTTAVA_LINE_THICKNESS)}


def polygon_points(polygon: etree._Element) -> list[tuple[float, float]]:
    return [tuple(map(float, point.split(","))) for point in polygon.get("points").split()]


def thickened(corners: list[tuple[float, float]], factor: float) -> list[tuple[float, float]]:
    """A beam's four corners, spread about its centre line at each end."""
    by_x = sorted(range(4), key=lambda i: corners[i][0])
    out = list(corners)
    for top, bottom in (by_x[:2], by_x[2:]):
        middle = (corners[top][1] + corners[bottom][1]) / 2
        for i in (top, bottom):
            out[i] = (corners[i][0], middle + (corners[i][1] - middle) * factor)
    return out


def overlap_in_x(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    return min(x for x, _ in a) < max(x for x, _ in b) and min(x for x, _ in b) < max(x for x, _ in a)


def thicken_beams(svg: bytes, factor: float) -> bytes:
    """The SVG with single beams thickened by factor, stacked ones as far as they stay apart."""
    root = etree.fromstring(svg)
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if "beam" not in group.get("class", "").split():
            continue
        polygons = [p for p in group.findall(f"{{{SVG_NS}}}polygon") if len(p.get("points").split()) == 4]
        corners = [polygon_points(p) for p in polygons]
        for polygon, own in zip(polygons, corners):
            stacked = any(other is not own and overlap_in_x(own, other) for other in corners)
            drawn = thickened(own, min(factor, STACKED_BEAM_THICKENING) if stacked else factor)
            polygon.set("points", " ".join(f"{x:.1f},{y:.1f}" for x, y in drawn))
    return etree.tostring(root)


def round_noteheads(svg: bytes, radius_interlines: float) -> bytes:
    """The SVG with filled noteheads drawn as circles."""
    root = etree.fromstring(svg)
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if "notehead" not in group.get("class", "").split():
            continue
        for glyph in group.findall(f"{{{SVG_NS}}}use"):
            position = re.match(r"translate\(([-\d.]+),\s*([-\d.]+)\)", glyph.get("transform", ""))
            if position and glyph.get(XLINK_HREF, "").startswith(FILLED_NOTEHEAD):
                circle = etree.Element(f"{{{SVG_NS}}}circle", cx=str(float(position[1]) + NOTEHEAD_CENTRE_UNITS),
                                       cy=position[2], r=str(radius_interlines * SVG_UNITS_PER_INTERLINE))
                group.replace(glyph, circle)
    return etree.tostring(root)


def chained(line: etree._Element, link: float) -> None:
    """The straight line as a chain of short dashes slanted up to the right."""
    x0, y, x1, _ = map(float, re.findall(r"[-\d.]+", line.get("d")))
    rise = link * 0.3
    links = [f"M{x:.0f} {y + rise:.0f} L{x + link * 0.6:.0f} {y - rise:.0f}" for x in np.arange(x0, x1 - link * 0.6, link)]
    line.set("d", " ".join(links))
    line.attrib.pop("stroke-dasharray", None)
    line.set("stroke-linecap", "round")


@dataclass(frozen=True)
class OttavaType:
    font: str
    bold: bool
    italic: bool
    size_interlines: float
    choice: float  # picks the wording among a glyph's

    @staticmethod
    def drawn(rng: random.Random) -> "OttavaType":
        return OttavaType(rng.choice(TEXT_OTTAVA_FONTS), rng.random() < 0.6, rng.random() < 0.3,
                          rng.uniform(*TEXT_OTTAVA_INTERLINES), rng.random())


def with_text_ottava(group: etree._Element, glyph: etree._Element, wordings: tuple[str, ...], kind: OttavaType) -> None:
    """The sign set in a text font in place of Verovio's glyph, centred on its line, which then
    starts just after it."""
    x = float(re.match(r"translate\(([-\d.]+),", glyph.get("transform"))[1])
    size = kind.size_interlines * SVG_UNITS_PER_INTERLINE
    wording = wordings[int(kind.choice * len(wordings))]
    lines = group.findall(f"{{{SVG_NS}}}path")
    line_y = float(re.findall(r"[-\d.]+", lines[0].get("d"))[1]) if lines else float(re.search(r",\s*([-\d.]+)\)", glyph.get("transform"))[1])
    text = etree.Element(f"{{{SVG_NS}}}text", x=f"{x:.0f}", y=f"{line_y + 0.36 * size:.0f}",
                         attrib={"font-family": kind.font, "font-size": f"{size:.0f}",
                                 "font-weight": "bold" if kind.bold else "normal", "font-style": "italic" if kind.italic else "normal"})
    text.text = wording
    group.replace(glyph, text)
    for line in lines:
        _, y, x1, _ = re.findall(r"[-\d.]+", line.get("d"))
        line.set("d", f"M{x + (0.6 * len(wording) + 0.15) * size:.0f} {y} L{x1} {y}")


def with_headroom(root: etree._Element, units: float) -> None:
    """More room above the drawing, at the same scale: a sign set in type can stand taller than
    the glyph Verovio sized the page for."""
    inner = root.find(f"{{{SVG_NS}}}svg")
    x, y, width, height = map(float, inner.get("viewBox").split())
    px_per_unit = float(root.get("height").removesuffix("px")) / height
    inner.set("viewBox", f"{x} {y - units} {width} {height + units}")
    root.set("height", f"{int(round((height + units) * px_per_unit))}px")


def restyle_ottava_lines(svg: bytes, rng: random.Random) -> bytes:
    """Dashed, dotted or chained, as plates vary, and at times set in a text font; Verovio draws
    only its own dashes and its music font's italic glyph."""
    dotted = rng.random() < DOTTED_OTTAVA_CHANCE
    dashes = rng.choice(OTTAVA_DOTS if dotted else OTTAVA_DASHES)
    chain = rng.uniform(*CHAIN_LINK_INTERLINES) * SVG_UNITS_PER_INTERLINE if rng.random() < CHAINED_OTTAVA_CHANCE else None
    text_type = OttavaType.drawn(rng) if rng.random() < TEXT_OTTAVA_CHANCE else None
    set_in_type = False
    root = etree.fromstring(svg)
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if "octave" in group.get("class", "").split():
            if text_type:
                for glyph in group.findall(f"{{{SVG_NS}}}use"):
                    wordings = TEXT_OF_OTTAVA_GLYPH.get(glyph.get(XLINK_HREF, "")[:5])
                    if wordings:
                        with_text_ottava(group, glyph, wordings, text_type)
                        set_in_type = True
            for line in group.findall(f"{{{SVG_NS}}}path"):
                if chain:
                    chained(line, chain)
                else:
                    line.set("stroke-dasharray", dashes)
                    line.set("stroke-linecap", "round" if dotted else "butt")
            if rng.random() < HOOKLESS_OTTAVA_CHANCE:  # the sign goes on past the system's end
                for hook in group.findall(f"{{{SVG_NS}}}polyline"):
                    group.remove(hook)
    if set_in_type:
        with_headroom(root, text_type.size_interlines * SVG_UNITS_PER_INTERLINE)
    return etree.tostring(root)


def groups_of_class(root: etree._Element, name: str) -> list[etree._Element]:
    return [g for g in root.iter(f"{{{SVG_NS}}}g") if name in g.get("class", "").split()]


def raise_bass_a_sharp(svg: bytes) -> bytes:
    """The SVG with the fifth sharp of bass-clef signatures an octave up, on the top line."""
    root = etree.fromstring(svg)
    for staff in groups_of_class(root, "staff"):
        glyphs = lambda group: group.findall(f".//{{{SVG_NS}}}use")
        if not any(u.get(XLINK_HREF, "").startswith(F_CLEF) for clef in groups_of_class(staff, "clef") for u in glyphs(clef)):
            continue
        for signature in groups_of_class(staff, "keySig"):
            sharps = [u for u in glyphs(signature) if u.get(XLINK_HREF, "").startswith(SHARP)]
            if len(sharps) >= 5:
                x, y = re.match(r"translate\(([-\d.]+),\s*([-\d.]+)\)", sharps[4].get("transform")).groups()
                sharps[4].set("transform", sharps[4].get("transform").replace(f"translate({x}, {y})", f"translate({x}, {float(y) - OCTAVE_UNITS})"))
    return etree.tostring(root)


def shrink_naturals(svg: bytes, factor: float) -> bytes:
    """The SVG with the naturals before notes drawn smaller, from their left edge."""
    root = etree.fromstring(svg)
    for accidental in groups_of_class(root, "accid"):
        for glyph in accidental.findall(f"{{{SVG_NS}}}use"):
            if glyph.get(XLINK_HREF, "").startswith(NATURAL):
                glyph.set("transform", re.sub(r"scale\(([-\d.]+),\s*([-\d.]+)\)",
                                              lambda m: f"scale({float(m[1]) * factor}, {float(m[2]) * factor})",
                                              glyph.get("transform", "")))
    return etree.tostring(root)


def glyph_path(font: str, code: str) -> str:
    """The outline of a SMuFL glyph in one of Verovio's fonts (its files have no namespace)."""
    return etree.parse(str(Path(verovio.__file__).parent / "data" / font / f"{code}.xml")).find("path").get("d")


def old_quarter_rests(svg: bytes, font: str) -> bytes:
    """The SVG with its quarter rests drawn in the old style, by redefining the glyph."""
    root = etree.fromstring(svg)
    for definition in root.iter(f"{{{SVG_NS}}}g"):
        if definition.get("id", "").startswith(QUARTER_REST + "-"):
            definition.find(f"{{{SVG_NS}}}path").set("d", glyph_path(font, OLD_QUARTER_REST))
    return etree.tostring(root)


PITCH_TOKEN = re.compile(r"^[A-G]\d$")


def open_at_an_edge(lmx: str) -> bool:
    """A tie or slur that runs on from the system before or into the one after: a page draws it as
    a half arc from or to the system's edge"""
    open_ties, slurs, pitch = set(), 0, None
    for token in lmx.split():
        if PITCH_TOKEN.match(token):
            pitch = token
        elif token == "tied:start":
            open_ties.add(pitch)
        elif token == "tied:stop":
            if pitch not in open_ties:
                return True
            open_ties.discard(pitch)
        elif token == "slur:start":
            slurs += 1
        elif token == "slur:stop":
            if slurs == 0:
                return True
            slurs -= 1
    return bool(open_ties) or slurs > 0


def notes_only(measure: ET.Element) -> ET.Element:
    """A neighbouring measure with nothing but its notes: no clef, key or time of its own (Verovio
    would draw a courtesy signature at the line's end) and no directions (an ottava would reach
    into the system, against its label). Only ties and slurs then cross the break."""
    measure = copy.deepcopy(measure)
    for child in list(measure):
        if child.tag == "direction":
            measure.remove(child)
        elif child.tag == "attributes":
            for change in child.findall("clef") + child.findall("key") + child.findall("time"):
                child.remove(change)
    return measure


def notes_only_attributes(attributes: ET.Element) -> ET.Element:
    """The system's opening attributes for the measure set before it, without its time signature:
    a time change that opens the system must still be drawn there"""
    attributes = copy.deepcopy(attributes)
    for time in attributes.findall("time"):
        attributes.remove(time)
    return attributes


def with_neighbours(systems: list, i: int) -> tuple[ET.Element, tuple[int, int]]:
    """The system's part between the last measure of the system before and the first of the one
    after, each on a line of its own, so that Verovio splits the ties and slurs across the breaks
    as on a page; and how many measures stand before and after it (the first is also the system's
    line among those drawn)"""
    part = copy.deepcopy(systems[i].part)
    first = part.find("measure")
    before_count = after_count = 0
    if i > 0:
        before = notes_only(systems[i - 1].part.findall("measure")[-1])
        before.insert(0, notes_only_attributes(first.find("attributes")))
        part.insert(list(part).index(first), before)
        first.insert(0, ET.Element("print", {"new-system": "yes"}))
        before_count = 1
    if i + 1 < len(systems):
        after = notes_only(systems[i + 1].part.find("measure"))
        after.insert(0, ET.Element("print", {"new-system": "yes"}))
        part.append(after)
        after_count = 1
    return part, (before_count, after_count)


def only_line(svg: bytes, line: int) -> bytes:
    """The page with one system left on it"""
    root = etree.fromstring(svg)
    for k, system in enumerate(groups_of_class(root, "system")):
        if k != line:
            system.getparent().remove(system)
    return etree.tostring(root)


# Printed systems are stretched to the page's width; ours are drawn as tight as they go. A share is
# spread out instead, by laying the system out on a page this much wider than it needs.
JUSTIFIED = 0.6
JUSTIFIED_STRETCH = (1.05, 1.35)


def justify(tk: verovio.toolkit, musicxml: str, options: dict, stretch: float) -> None:
    """Lays the loaded score out again, its lines spread over a page `stretch` times their natural
    width (in Verovio's units, the width of the page it drew). Verovio justifies a lone line only
    with encoded breaks."""
    natural = float(re.search(r'width="([\d.]+)px"', tk.renderToSVG(1)).group(1))
    options |= {"adjustPageWidth": False, "pageWidth": round(natural * stretch), "breaks": "encoded"}
    tk.setOptions(options)
    tk.loadData(musicxml)


def staff_lines_only(svg: bytes) -> bytes:
    """The page with nothing drawn but its staves' lines (each staff group's own paths), and the
    styles and definitions they are drawn with."""
    root = etree.fromstring(svg)
    lines = [p for staff in groups_of_class(root, "staff") for p in staff if p.tag == f"{{{SVG_NS}}}path"]
    styling = [e for e in root.iter(f"{{{SVG_NS}}}style", f"{{{SVG_NS}}}defs")]
    keep = set(lines) | set(styling) | {a for e in lines + styling for a in e.iterancestors()}
    for element in [e for e in root.iter() if e not in keep and e.getparent() in keep]:
        element.getparent().remove(element)
    return etree.tostring(root)


def drawn_staves(lines: np.ndarray) -> Staves | None:
    """Staff centres and interline in an image of staff lines alone: its rows of ink, five to a
    staff. None unless the lines come in fives."""
    rows = np.nonzero((lines < 250).any(axis=1))[0]
    if len(rows) == 0:
        return None
    centres = [run.mean() for run in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)]
    if len(centres) % 5:
        return None
    staves = [centres[i:i + 5] for i in range(0, len(centres), 5)]
    return Staves(tuple(float(s[2]) for s in staves), float(np.mean([np.diff(s).mean() for s in staves])))


def rasterized(svg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(cairosvg.svg2png(bytestring=svg, background_color="white"), np.uint8),
                        cv2.IMREAD_GRAYSCALE)


def render_system(tk: verovio.toolkit, musicxml: str, rng: random.Random, style: Style = Style(),
                  line: int | None = None, look: TupletLook = PLAIN_LOOK) -> tuple[np.ndarray, Staves | None] | None:
    """The system as a grayscale image cropped to its ink plus a random margin, and its staves
    there, from the drawing (None if its lines do not come in fives). With line, the score holds
    neighbouring measures on lines of their own (with_neighbours), and only that line is drawn."""
    options = engraving_options(rng)
    if style.marks:
        options |= ottava_options(rng)
    if line is not None:
        options["breaks"] = "encoded"
    tk.setOptions(options)
    if not tk.loadData(musicxml) or tk.getPageCount() != 1:
        return None
    if rng.random() < JUSTIFIED:
        justify(tk, musicxml, options, rng.uniform(*JUSTIFIED_STRETCH))
        if tk.getPageCount() != 1:
            return None
    interline = rng.uniform(*INTERLINE_PX)
    drawn = tk.renderToSVG(1)
    if look != PLAIN_LOOK:
        drawn = restyle_tuplets(drawn.encode(), look, UNITS_PER_INTERLINE_PER_UNIT * options["unit"], rng).decode()
    svg = layout_svg(tk, drawn, interline / (UNITS_PER_INTERLINE_PER_UNIT * options["unit"]))
    if style.thick_beams:
        svg = thicken_beams(svg, rng.uniform(*SINGLE_BEAM_THICKENING))
    if style.round_heads:
        svg = round_noteheads(svg, rng.uniform(*ROUND_HEAD_RADIUS_INTERLINES))
    if style.marks:
        svg = raise_bass_a_sharp(restyle_ottava_lines(svg, rng))
    if rng.random() < NATURALS_SHRUNK:
        svg = shrink_naturals(svg, rng.uniform(*NATURAL_SCALE))
    if rng.random() < OLD_QUARTER_RESTS:
        svg = old_quarter_rests(svg, options["font"])
    if line is not None:
        svg = only_line(svg, line)
    use_smufl_fonts()  # music symbols in text (tempo marks) need the SMuFL fonts in cairo
    image = rasterized(svg)
    ys, xs = np.nonzero(image < 128)
    if len(ys) == 0:
        return None
    m = [int(rng.uniform(*MARGIN_INTERLINES) * interline) for _ in range(4)]
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    staves = drawn_staves(rasterized(staff_lines_only(svg)))
    in_crop = staves and Staves(tuple(c - ys.min() + m[0] for c in staves.centres), staves.interline)
    return cv2.copyMakeBorder(crop, m[0], m[1], m[2], m[3], cv2.BORDER_CONSTANT, value=255), in_crop


def write_sample(stem: Path, image: np.ndarray, staves: Staves, lmx: str) -> None:
    """A crop, its label, and its staves as drawn (find_staff_positions.py reads them)."""
    cv2.imwrite(str(stem.with_suffix(".png")), image)
    stem.with_suffix(".lmx").write_text(lmx + "\n")
    stem.with_suffix(".staves.json").write_text(json.dumps([*staves.centres, staves.interline]))


def row_resized(row: float, factor: float) -> float:
    """Where a row lands when its image is resized by factor: cv2.resize scales pixel centres,
    not edges, so row 0 of an image halved lies between rows 0 and 1 of the original."""
    return (row + 0.5) * factor - 0.5


def build_score(job: tuple[int, str, Path, int, bool, Style, TupletDrawing]) -> int:
    """Renders up to K systems of one score; returns how many."""
    index, source, out, k, unmarked_tuplets, style, drawing = job
    target = out / "samples" / f"{index:06d}"
    if target.exists():
        return len(list(target.glob("*.lmx")))
    rng = random.Random(index)
    try:
        part = piano_part(load_musicxml(Path(source)))
        if part is None:
            return 0
        without_octave_clefs(part)
        break_into_systems(part, rng)
        systems = [s for page in split_part_to_systems(part) for s in page.systems]
        in_order = list(systems)
        if unmarked_tuplets:
            systems = [s for s in systems if has_tuplets(s.part)]
    except Exception:
        return 0
    tuplets = ScoreTuplets.drawn(drawing, rng)
    rng.shuffle(systems)
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    tk = verovio.toolkit()
    tk.setOptions({"xmlIdSeed": index})
    written = 0
    for system in systems:
        if written == k:
            break
        try:
            if style.marks:
                add_marks(system.part, rng)
            draw_some_meters_as_symbols(system.part, rng)
            draw_tuplet_numbers(system.part, unmarked_tuplets, tuplets, rng)
            lmx = linearize(system.part)
            drawn, neighbours = (with_neighbours(in_order, in_order.index(system)) if open_at_an_edge(lmx)
                                 else (system.part, None))
            score = part_to_score(drawn).getroot()
            if rng.random() < PART_NAME_HIDDEN:
                hide_part_name(score)
            musicxml = ET.tostring(score, encoding="unicode", xml_declaration=True)
            line = neighbours[0] if neighbours else None
            rendered = render_system(tk, musicxml, rng, style, line, tuplets.look) if 10 <= len(lmx.split()) <= MAX_TOKENS else None
            # A system Verovio broke over several lines draws more staves than the label has.
            drawn_right = (rendered is not None and rendered[1] is not None
                           and len(rendered[1].centres) == staff_count(lmx)
                           and ottavas_drawn_as_labelled(tk, system.part, neighbours or (0, 0)))
        except Exception:
            # Nothing may escape to the pool: CairoError cannot be unpickled, which kills
            # the pool's result thread and silently stalls every later score.
            continue
        if not drawn_right:
            continue
        write_sample(partial / f"s{written}", *rendered, lmx)
        written += 1
    partial.rename(target)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems-per-score", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--unmarked-tuplets", action="store_true")
    ap.add_argument("--thick-beams", action="store_true")
    ap.add_argument("--round-heads", action="store_true")
    ap.add_argument("--marks", action="store_true")
    ap.add_argument("--engraver-hiding", action="store_true", help="hide tuplet numbers as engravers do")
    ap.add_argument("--varied-tuplet-looks", action="store_true", help="set printed tuplet numbers in varied ways")
    args = ap.parse_args()
    sources = args.scores.read_text().split("\n")
    hung_log = args.out / "hung.txt"
    hung = set(hung_log.read_text().split("\n")) if hung_log.exists() else set()
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    style = Style(args.thick_beams, args.round_heads, args.marks)
    drawing = TupletDrawing(args.engraver_hiding, args.varied_tuplet_looks)
    jobs = [(i, s, args.out, args.systems_per_score, args.unmarked_tuplets, style, drawing)
            for i, s in enumerate(sources)
            if s not in hung and is_training_score(s, conflicts)]
    total = 0
    # "spawn", not fork: Verovio keeps its font resources in C++ static state that forked
    # workers cannot use ("font resources are not available"); see build_dataset.py.
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pending = [(job, pool.apply_async(build_score, (job,))) for job in jobs]
        for done, (job, result) in enumerate(pending, 1):
            try:
                total += result.get(timeout=HUNG_SCORE_S)
            except mp.TimeoutError:
                # Verovio occasionally never returns on a score. Its worker stays stuck
                # until the pool is terminated; the score is skipped from now on.
                with hung_log.open("a") as f:
                    f.write(job[1] + "\n")
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} scores, {total} systems", flush=True)
        pool.terminate()
    for partial in (args.out / "samples").glob("*.partial"):
        shutil.rmtree(partial)
    kept = {f"{job[0]:06d}" for job in jobs}
    names = sorted(f"samples/{p.parent.name}/{p.stem}" for p in (args.out / "samples").glob("[0-9]*/*.lmx")
                   if p.parent.name in kept)
    (args.out / "samples.train.txt").write_text("\n".join(names) + "\n")
    print(f"{len(names)} systems from {len(jobs)} scores")


if __name__ == "__main__":
    main()
