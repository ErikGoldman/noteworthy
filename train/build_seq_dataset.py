"""Renders system crops with LMX transcriptions from our training scores, for the
sequence model (train_seq.py).

usage: build_seq_dataset.py --scores LIST.txt --out DIR [--systems-per-score K] [--workers W]
                            [--unmarked-tuplets] [--thick-beams] [--round-heads] [--marks]

Held-out scores (omr.held_out) and PDMX scores with a license conflict are left out. Each two-staff (piano) part is cut into systems of a few measures, which OLiMPiC's code
turns into self-contained parts: clefs and key repeated at each system start, the time
signature only where the score prints one (split_part_to_systems), as an engraver lays
them out. K random systems per score are rendered alone with Verovio under random
engraving options and cropped to their ink with random margins. Their LMX comes from
OLiMPiC's linearizer, as for the OLiMPiC and GrandStaff-LMX datasets.

With --unmarked-tuplets only systems with tuplets are taken, drawn without their numbers
(the first may stay): engravers mark the first triplets of a passage and leave the rest,
and a model that only ever saw numbers reads the others as plain notes.

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
from seq.ottava import TOKEN_ATTRIBUTE, move_octaves, notes_under_ottavas, written_part

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
SINGLE_BEAM_THICKENING = (1.0, 2.6)
STACKED_BEAM_THICKENING = 1.35  # beyond this, stacked beams touch and their number is lost
ROUND_HEAD_RADIUS_INTERLINES = (0.42, 0.55)
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
HIGHEST_OCTAVE, LOWEST_OCTAVE = 7, 1
F_CLEF, SHARP = "#E062", "#E262"  # SMuFL; Verovio appends a per-file suffix
OCTAVE_UNITS = 3.5 * SVG_UNITS_PER_INTERLINE


@dataclass(frozen=True)
class Style:
    """Departures from Verovio's engraving, for hands it does not draw."""
    thick_beams: bool = False
    round_heads: bool = False
    marks: bool = False


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


def hide_tuplet_numbers(score: ET.Element, rng: random.Random) -> None:
    starts = [t for t in score.iter("tuplet") if t.get("type") == "start"]
    keep_first = rng.random() < KEEP_FIRST_TUPLET_NUMBER
    for tuplet in starts[1:] if keep_first else starts:
        tuplet.set("show-number", "none")
        tuplet.set("bracket", "no")


class WrittenLinearizer(Linearizer):
    """OLiMPiC's linearizer, with each tagged note's ottava token straight after its pitch."""
    def process_note(self, note: ET.Element, measure: ET.Element):
        self._ottava_token = note.get(TOKEN_ATTRIBUTE)
        super().process_note(note, measure)

    def _emit(self, token: str):
        super()._emit(token)
        if token in PITCH_TOKENS and self._ottava_token:
            self.output_tokens.append(self._ottava_token)


def linearize(part: ET.Element) -> str:
    linearizer = WrittenLinearizer(errout=None)
    linearizer.process_part(written_part(part))
    return " ".join(linearizer.output_tokens)


def ottavas_drawn_as_labelled(tk: verovio.toolkit, part: ET.Element) -> bool:
    """False if Verovio shifted other notes than the labels say. A sign holds for its whole
    staff, as in MuseScore, but Verovio leaves out some voices of the same staff."""
    labelled = len(notes_under_ottavas(part))
    if not labelled:
        return True
    notes = [dict(re.findall(r'(oct(?:\.ges)?)="(\d+)"', note)) for note in re.findall(r"<note [^>]*>", tk.getMEI())]
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


def restyle_ottava_lines(svg: bytes, rng: random.Random) -> bytes:
    """Dashed or dotted, as plates vary; Verovio draws only its own dashes."""
    dotted = rng.random() < DOTTED_OTTAVA_CHANCE
    dashes = rng.choice(OTTAVA_DOTS if dotted else OTTAVA_DASHES)
    root = etree.fromstring(svg)
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if "octave" in group.get("class", "").split():
            for line in group.findall(f"{{{SVG_NS}}}path"):
                line.set("stroke-dasharray", dashes)
                line.set("stroke-linecap", "round" if dotted else "butt")
            if rng.random() < HOOKLESS_OTTAVA_CHANCE:  # the sign goes on past the system's end
                for hook in group.findall(f"{{{SVG_NS}}}polyline"):
                    group.remove(hook)
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


def render_system(tk: verovio.toolkit, musicxml: str, rng: random.Random, style: Style = Style()) -> np.ndarray | None:
    """The system as a grayscale image cropped to its ink plus a random margin."""
    options = engraving_options(rng)
    if style.marks:
        options |= ottava_options(rng)
    tk.setOptions(options)
    if not tk.loadData(musicxml) or tk.getPageCount() != 1:
        return None
    interline = rng.uniform(*INTERLINE_PX)
    svg = layout_svg(tk, tk.renderToSVG(1), interline / (UNITS_PER_INTERLINE_PER_UNIT * options["unit"]))
    if style.thick_beams:
        svg = thicken_beams(svg, rng.uniform(*SINGLE_BEAM_THICKENING))
    if style.round_heads:
        svg = round_noteheads(svg, rng.uniform(*ROUND_HEAD_RADIUS_INTERLINES))
    if style.marks:
        svg = raise_bass_a_sharp(restyle_ottava_lines(svg, rng))
    use_smufl_fonts()  # music symbols in text (tempo marks) need the SMuFL fonts in cairo
    png = cairosvg.svg2png(bytestring=svg, background_color="white")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(image < 128)
    if len(ys) == 0:
        return None
    m = [int(rng.uniform(*MARGIN_INTERLINES) * interline) for _ in range(4)]
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.copyMakeBorder(crop, m[0], m[1], m[2], m[3], cv2.BORDER_CONSTANT, value=255)


def build_score(job: tuple[int, str, Path, int, bool, Style]) -> int:
    """Renders up to K systems of one score; returns how many."""
    index, source, out, k, unmarked_tuplets, style = job
    target = out / "samples" / f"{index:06d}"
    if target.exists():
        return len(list(target.glob("*.lmx")))
    rng = random.Random(index)
    try:
        part = piano_part(load_musicxml(Path(source)))
        if part is None:
            return 0
        break_into_systems(part, rng)
        systems = [s for page in split_part_to_systems(part) for s in page.systems]
        if unmarked_tuplets:
            systems = [s for s in systems if has_tuplets(s.part)]
    except Exception:
        return 0
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
            lmx = linearize(system.part)
            score = part_to_score(system.part).getroot()
            if rng.random() < PART_NAME_HIDDEN:
                hide_part_name(score)
            if unmarked_tuplets:
                hide_tuplet_numbers(score, rng)
            musicxml = ET.tostring(score, encoding="unicode", xml_declaration=True)
            image = render_system(tk, musicxml, rng, style) if 10 <= len(lmx.split()) <= MAX_TOKENS else None
            drawn_right = image is not None and ottavas_drawn_as_labelled(tk, system.part)
        except Exception:
            # Nothing may escape to the pool: CairoError cannot be unpickled, which kills
            # the pool's result thread and silently stalls every later score.
            continue
        if not drawn_right:
            continue
        cv2.imwrite(str(partial / f"s{written}.png"), image)
        (partial / f"s{written}.lmx").write_text(lmx + "\n")
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
    args = ap.parse_args()
    sources = args.scores.read_text().split("\n")
    hung_log = args.out / "hung.txt"
    hung = set(hung_log.read_text().split("\n")) if hung_log.exists() else set()
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    style = Style(args.thick_beams, args.round_heads, args.marks)
    jobs = [(i, s, args.out, args.systems_per_score, args.unmarked_tuplets, style)
            for i, s in enumerate(sources)
            if s.strip() and s not in hung and not is_held_out(Path(s)) and Path(s).name not in conflicts]
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
