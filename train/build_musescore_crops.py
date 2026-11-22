"""System crops with LMX transcriptions engraved by MuseScore 3.6.2, a second engraver beside
Verovio (build_seq_dataset.py), for the sequence model (train_seq.py).

usage: build_musescore_crops.py --scores LIST.txt --out DIR [--systems-per-score K]
                                [--renderers R] [--scores-per-batch B] [--limit N]
                                [--musescore SQUASHFS_ROOT] [--tuplet-hiding {engravers,often}] [--varied-tuplet-looks]

Scores, systems and labels are chosen as for the Verovio crops: the same training scores, the
first two-staff part cut into systems of a few measures, each system made self-contained by
OLiMPiC's split_part_to_systems and labelled by build_seq_dataset.linearize, some meters drawn as
C or cut-C and some tuplet numbers hidden. Each system is engraved alone, as its own one-page
score on a wide page, so that a system MuseScore draws otherwise than its label says costs only
itself. Such a system is found in the page's SVG and left out: broken in two (four staves, or a
second page), with more or fewer barlines than its measures, or a barline that differs between
the staves. A system ending in a plain barline is engraved with an empty measure after it, which
the crop leaves out, because MuseScore closes every score with a final bar (with_spare_measure).
The crop is the page's ink (its alpha channel) up to the system's closing barline, at a random
interline, with random margins.

Each score is engraved in one of STYLE_VARIANTS MuseScore styles (music font, line widths,
spacing), drawn from the score's index, so a score's crops do not depend on which batch it was
rendered in, and a run stopped and restarted writes what an uninterrupted one would.

MuseScore draws every tuplet number of a MusicXML file (show_every_tuplet_number). With
--tuplet-hiding or --varied-tuplet-looks the numbers are hidden and set as build_seq_dataset.py's
options say (seq.tuplet_display), through MuseScore's own file, which it honours: each system is
converted to .mscx, its tuplets drawn there (mscx_tuplets), and the .mscx engraved. A system whose
tuplets do not pair between the two files is left out ("tuplets do not pair" in its outcome).

Output: DIR/samples/<index>/s<k>.png + .lmx + .staves.json (the staves as drawn), an outcome.json per score saying what became of
each candidate system (or why the score was left out whole), and DIR/samples.train.txt; train_seq.py reads it as --data DIR:train. A score's
directory appears only once complete, so the run may be killed at any time and restarted.
"""
from __future__ import annotations

import argparse
import ctypes
import functools
import json
import math
import multiprocessing as mp
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seq_dataset import (DATA, INTERLINE_PX, MARGIN_INTERLINES, MAX_TOKENS, PART_NAME_HIDDEN,  # noqa: E402
                               ROOT, break_into_systems, draw_some_meters_as_symbols,
                               hide_part_name, is_training_score, linearize, without_octave_clefs,
                               load_musicxml, open_at_an_edge, part_to_score, piano_part,
                               row_resized, split_part_to_systems, write_sample)
from seq.staff_positions import Staves  # noqa: E402
from omr.pieces import pdmx_license_conflicts  # noqa: E402
from seq.barlines import REPEAT_START, barline_tokens_of  # noqa: E402
from seq.ottava import add_octave_shifts, shifts_of  # noqa: E402
from seq.tuplet_display import HIDING_MIXES, ScoreTuplets, TupletDrawing, hide_as_engravers_do, set_look  # noqa: E402
from mscx_tuplets import drawn_in_mscx  # noqa: E402

MIN_TOKENS = 10
NICENESS = 10
STYLE_VARIANTS = 64
STARTUP_TIMEOUT_S = 30
TIMEOUT_PER_SYSTEM_S = 5
# (music font, its text font, weight). MuseJazz and Petaluma are hand-drawn styles, a few
# percent as handwriting is of our training mix.
MUSIC_FONTS = (("Emmentaler", "MScore Text", 4), ("Bravura", "Bravura Text", 3), ("Leland", "Leland Text", 3),
               ("Gonville", "Gootville Text", 2), ("MuseJazz", "MuseJazz Text", 0.5), ("Petaluma", "Petaluma Text", 0.5))
# MuseScore style values in staff spaces, MuseScore's defaults within each range. Systems are
# not justified, so measureSpacing alone sets how loosely they are spaced.
STYLE_RANGES = {"staffLineWidth": (0.06, 0.13), "stemWidth": (0.08, 0.16), "beamWidth": (0.4, 0.6),
                "barWidth": (0.12, 0.24), "ledgerLineWidth": (0.12, 0.22), "akkoladeDistance": (5.0, 9.0),
                "measureSpacing": (1.0, 1.5), "minNoteDistance": (0.2, 0.4)}
# A wide, short page in inches, so that MuseScore seldom breaks a system and the PNG stays small.
PAGE = {"pageWidth": 26, "pageHeight": 5, "pagePrintableWidth": 25, "pageEvenLeftMargin": 0.5,
        "pageOddLeftMargin": 0.5, "pageEvenTopMargin": 0.5, "pageEvenBottomMargin": 0.5,
        "pageOddTopMargin": 0.5, "pageOddBottomMargin": 0.5, "lastSystemFillLimit": 1}
LINES_PER_STAFF = 5
STAVES_PER_SYSTEM = 2
SVG_NS = "http://www.w3.org/2000/svg"
WRITTEN = "written"


@dataclass(frozen=True)
class ExpectedDrawing:
    """What the page must show for the label to hold."""
    barlines: int  # see barlines_drawn
    spare_measure: bool  # see with_spare_measure


@dataclass(frozen=True)
class LabelledSystem:
    name: str
    musicxml: str
    lmx: str
    expected: ExpectedDrawing
    tuplets: ScoreTuplets | None = None  # drawn through MuseScore's own file (mscx_tuplets); None: every number shown


@dataclass(frozen=True)
class Staff:
    top: float
    bottom: float
    interline: float

    @property
    def centre_row(self) -> float:
        """Its middle line's row in the page's PNG, whose pixels are the SVG's units: pixel k
        covers k to k + 1, so a line drawn at y is centred on row y - 0.5."""
        return (self.top + self.bottom) / 2 - 0.5


@dataclass(frozen=True)
class MuseScore:
    """The unpacked AppImage. Its AppRun forks rather than execs the binary, so a timeout would
    kill only the script and leave MuseScore running: the binary is started directly."""
    root: Path

    @property
    def binary(self) -> Path:
        return self.root / "bin" / "mscore-portable"

    def environment(self) -> dict[str, str]:
        # Qt's print support asks CUPS for printers at start-up and waits a minute for an
        # answer that never comes on WSL; a server path that cannot exist fails at once.
        return os.environ | {"APPDIR": str(self.root), "APPIMAGE": str(self.root / "AppRun"),
                             "DISPLAY": os.environ.get("DISPLAY", ":0"), "CUPS_SERVER": "/nonexistent/cups.sock"}


# ---- systems and labels ----------------------------------------------------------------------

def closes_with_own_barline(part: ET.Element) -> bool:
    """The last measure ends in a final or double bar or a repeat, which MuseScore draws at a
    score's end as written."""
    return bool(barline_tokens_of(part.findall("measure")[-1])[1])


def barlines_drawn(part: ET.Element) -> int:
    """The system's opening barline, each measure's closing one, and a repeat sign opening the
    first measure, which MuseScore draws after the clefs and key (further on, one joins the
    previous measure's barline)."""
    measures = part.findall("measure")
    return 1 + len(measures) + (REPEAT_START in barline_tokens_of(measures[0])[0])


def with_spare_measure(part: ET.Element) -> None:
    """An empty measure after the system's last, cut off the crop again (closing_barline_right),
    for a system ending in a plain barline. MuseScore closes a score with a final bar, and it
    ignores a MusicXML regular barline there. Not after a measure with its own closing barline:
    MuseScore then draws that on the upper staff only and a plain one below."""
    ET.SubElement(part, "measure", number="spare")


def show_every_tuplet_number(part: ET.Element) -> None:
    """MuseScore 3.6 draws every tuplet's number, ignoring MusicXML's show-number="none", so
    its crops label every number shown, the source's hidden ones too. Hidden numbers are
    taught by the Verovio crops, which draw them as labelled."""
    for tuplet in part.iter("tuplet"):
        tuplet.attrib.pop("show-number", None)


def without_octave_shifts(part: ET.Element) -> None:
    for measure in part.findall("measure"):
        for direction in measure.findall("direction"):
            for kind in direction.findall("direction-type"):
                if kind.find("octave-shift") is not None:
                    direction.remove(kind)
            if direction.find("direction-type") is None:
                measure.remove(direction)


def with_label_ottavas(part: ET.Element, lmx: str) -> bool:
    """Redraws the system's ottava signs as its label gives them, each closed within the system;
    False if they cannot be. A sign that runs on into the next system has no stop here, and
    MuseScore drops an unclosed sign, drawing its notes where they sound: the label would then
    give an ottava, at printed pitches, that the image does not show."""
    shifts = shifts_of(lmx)
    without_octave_shifts(part)
    add_octave_shifts(part, shifts)
    return not any(shifts) or next(part.iter("octave-shift"), None) is not None


def labelled_system(part: ET.Element, name: str, rng: random.Random,
                    tuplets: ScoreTuplets | None = None) -> LabelledSystem | None:
    """The system's label and its score for MuseScore, or None if its label is out of bounds, or
    if a tie or slur runs on across one of its edges: engraved alone, the system shows none of the
    half arc a page draws there. (The Verovio crops draw it from neighbouring measures; MuseScore
    would stretch a system that is not a page's last to the page's width.)"""
    draw_some_meters_as_symbols(part, rng)
    show_every_tuplet_number(part)
    if tuplets is not None:
        draw_tuplets(part, tuplets, rng)
    lmx = linearize(part)
    if open_at_an_edge(lmx):
        return None
    if not MIN_TOKENS <= len(lmx.split()) <= MAX_TOKENS or not with_label_ottavas(part, lmx):
        return None
    score = part_to_score(part).getroot()
    if rng.random() < PART_NAME_HIDDEN:
        hide_part_name(score)
    expected = ExpectedDrawing(barlines_drawn(part), spare_measure=not closes_with_own_barline(part))
    if expected.spare_measure:
        with_spare_measure(part)
    return LabelledSystem(name, ET.tostring(score, encoding="unicode", xml_declaration=True), lmx, expected, tuplets)


def draw_tuplets(part: ET.Element, tuplets: ScoreTuplets, rng: random.Random) -> None:
    """The score's hiding (else every number shown, as without a drawing) and look."""
    if tuplets.hiding is not None:
        hide_as_engravers_do(part, tuplets.hiding, rng)
    set_look(part, tuplets.look)


def labelled_systems(source: Path, rng: random.Random, k: int,
                     drawing: TupletDrawing = TupletDrawing()) -> list[LabelledSystem]:
    """Up to k random systems of the score's piano part. Raises ValueError without one."""
    part = piano_part(load_musicxml(source))
    if part is None:
        raise ValueError("no two-staff part")
    without_octave_clefs(part)
    break_into_systems(part, rng)
    tuplets = ScoreTuplets.drawn(drawing, rng) if drawing.changed else None
    parts = [s.part for page in split_part_to_systems(part) for s in page.systems]
    rng.shuffle(parts)
    chosen = []
    for part in parts:
        if len(chosen) == k:
            break
        system = labelled_system(part, f"candidate{len(chosen)}", rng, tuplets)
        if system is not None:
            chosen.append(system)
    return chosen


# ---- engraving style -------------------------------------------------------------------------

def style_variant(index: int) -> int:
    return random.Random(f"style:{index}").randrange(STYLE_VARIANTS)


def engraving_style(variant: int) -> str:
    """A MuseScore style file (.mss) for one of the variants."""
    rng = random.Random(f"variant:{variant}")
    music, text, _ = rng.choices(MUSIC_FONTS, weights=[w for *_, w in MUSIC_FONTS])[0]
    values = {"musicalSymbolFont": music, "musicalTextFont": text, **PAGE,
              **{name: f"{rng.uniform(*bounds):.3f}" for name, bounds in STYLE_RANGES.items()}}
    style = etree.Element("Style")
    for name, value in values.items():
        etree.SubElement(style, name).text = str(value)
    root = etree.Element("museScore", version="3.02")
    root.append(style)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True).decode()


def write_settings(config: Path, style_file: Path) -> None:
    """A settings folder of the renderer's own: MusicXML imports take the style from it (the
    command line's --style applies to MuseScore files only), and parallel renderers do not
    share settings."""
    ini = config / "MuseScore" / "MuseScore3.ini"
    ini.parent.mkdir(parents=True, exist_ok=True)
    ini.write_text(f"[application]\ntelemetry\\allowed=false\n\n[import]\nstyle\\styleFile={style_file}\n")


# ---- MuseScore -------------------------------------------------------------------------------

def die_with_parent() -> None:
    """The kernel kills this process when its parent dies. Run in each worker and in each
    MuseScore, so that a run killed even with SIGKILL leaves neither behind: orphaned workers
    would go on rendering the batches queued to them, beside the restarted run."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


def as_musescore_child() -> None:
    """Runs in MuseScore's process before exec. It segfaults on some files: no core dumps."""
    die_with_parent()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@dataclass(frozen=True)
class Outputs:
    """What a conversion writes beside its input, and whether it has."""
    suffixes: tuple[str, ...]
    written: Callable[[Path], bool]

    def conversion(self, source: Path) -> dict:
        stem = source.with_suffix("")
        return {"in": str(source), "out": [f"{stem}{suffix}" for suffix in self.suffixes]}


def run_musescore(musescore: MuseScore, musicxmls: list[Path], work: Path, outputs: Outputs) -> bool:
    """Converts in one MuseScore process, in order; False if it fails or hangs."""
    (work / "job.json").write_text(json.dumps([outputs.conversion(m) for m in musicxmls]))
    timeout = STARTUP_TIMEOUT_S + TIMEOUT_PER_SYSTEM_S * len(musicxmls)
    try:
        done = subprocess.run([str(musescore.binary), "-c", str(work / "config"), "-j", str(work / "job.json")],
                              env=musescore.environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout, preexec_fn=as_musescore_child)
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def engrave(musescore: MuseScore, musicxmls: list[Path], work: Path, style: str, outputs: Outputs) -> set[Path]:
    """Converts the files in as few MuseScore processes as it takes; returns those it crashed
    or hung on. MuseScore stops at the first such file, so the rest go to a new process."""
    (work / "style.mss").write_text(style)
    write_settings(work / "config", work / "style.mss")
    failed = set()
    remaining = musicxmls
    while remaining and not run_musescore(musescore, remaining, work, outputs):
        culprit = next((i for i, m in enumerate(remaining) if not outputs.written(m)), len(remaining))
        failed.update(remaining[culprit:culprit + 1])
        remaining = remaining[culprit + 1:]
    return failed


def page_files(musicxml: Path, page: int) -> tuple[Path, Path]:
    """MuseScore numbers each exported page: s0.png becomes s0-1.png, s0-2.png..."""
    stem = musicxml.with_suffix("")
    return stem.with_name(f"{stem.name}-{page}.png"), stem.with_name(f"{stem.name}-{page}.svg")


def converted(musicxml: Path) -> bool:
    """The SVG is written after the PNG."""
    return page_files(musicxml, 1)[1].exists()


PAGES = Outputs((".png", ".svg"), converted)
MSCX = Outputs((".mscx",), lambda source: source.with_suffix(".mscx").exists())
CRASHED = "musescore crashed or hung"


# ---- finding the system on the page ----------------------------------------------------------

def staff_lines(svg: bytes) -> list[tuple[float, float, float]]:
    """Each staff line as (y, left, right)."""
    lines = []
    for polyline in etree.fromstring(svg).iter(f"{{{SVG_NS}}}polyline"):
        if polyline.get("class") == "StaffLines":
            (x0, y0), (x1, _) = [tuple(map(float, p.split(","))) for p in polyline.get("points").split()[:2]]
            lines.append((y0, min(x0, x1), max(x0, x1)))
    return lines


def staves(lines: list[tuple[float, float, float]]) -> list[Staff]:
    """Staff lines grouped into staves, top to bottom: five lines that start and end together.
    Raises ValueError for lines that do not form whole staves."""
    by_extent: dict[tuple[int, int], list[float]] = {}
    for y, left, right in lines:
        by_extent.setdefault((round(left), round(right)), []).append(y)
    found = []
    for ys in by_extent.values():
        ys.sort()
        if len(ys) % LINES_PER_STAFF:
            raise ValueError(f"{len(ys)} staff lines of one extent")
        for k in range(0, len(ys), LINES_PER_STAFF):
            staff = ys[k:k + LINES_PER_STAFF]
            found.append(Staff(staff[0], staff[-1], (staff[-1] - staff[0]) / (LINES_PER_STAFF - 1)))
    return sorted(found, key=lambda s: s.top)


@dataclass(frozen=True, order=True)
class Stroke:
    """One staff's stroke of a barline: MuseScore draws each staff's from its top line, down
    to the next staff's when the barline spans both."""
    left: float
    right: float
    top: float


def barline_strokes(svg: bytes) -> list[Stroke]:
    strokes = []
    for polyline in etree.fromstring(svg).iter(f"{{{SVG_NS}}}polyline"):
        if polyline.get("class") == "BarLine":
            (x, top), (_, bottom) = [tuple(map(float, p.split(","))) for p in polyline.get("points").split()[:2]]
            half = float(polyline.get("stroke-width", 0)) / 2
            strokes.append(Stroke(x - half, x + half, min(top, bottom)))
    return strokes


def barlines(strokes: list[Stroke], interline: float) -> list[list[Stroke]]:
    """The strokes grouped into barlines, left to right: a double or final bar's strokes lie
    within an interline of each other."""
    groups: list[list[Stroke]] = []
    for stroke in sorted(strokes):
        if groups and stroke.left - max(s.right for s in groups[-1]) < interline:
            groups[-1].append(stroke)
        else:
            groups.append([stroke])
    return groups


def drawn_alike_on_both_staves(barline: list[Stroke], upper: Staff, lower: Staff) -> bool:
    """MuseScore draws some barlines it imports on the upper staff only, a final bar there and
    a plain one below (seen at a system's last measure), and then the image and label differ."""
    def pattern(staff: Staff) -> list[tuple[int, int]]:
        return sorted((round(s.left), round(s.right)) for s in barline if abs(s.top - staff.top) < upper.interline)
    return pattern(upper) == pattern(lower)


def closing_barline_right(strokes: list[Stroke], upper: Staff, lower: Staff, expected: ExpectedDrawing) -> float:
    """The right edge of the system's closing barline: the last, or the last before the spare
    measure's. Raises ValueError if the system has more or fewer barlines than its measures
    call for, or any differ between its staves."""
    found = barlines(strokes, upper.interline)
    system = found[:-1] if expected.spare_measure else found
    if len(system) != expected.barlines:
        raise ValueError("barlines do not match measures")
    if not all(drawn_alike_on_both_staves(b, upper, lower) for b in system):
        raise ValueError("barline differs between staves")
    return max(s.right for s in system[-1])


@dataclass(frozen=True)
class SystemOnPage:
    staves: Staves  # in the page's rows
    right: float  # where the crop ends: the system's closing barline


def system_on_page(svg: bytes, expected: ExpectedDrawing) -> SystemOnPage:
    """The page's one two-staff system. Raises ValueError for any other page."""
    try:
        lines, strokes = staff_lines(svg), barline_strokes(svg)
    except etree.XMLSyntaxError as e:
        raise ValueError("unreadable SVG") from e
    found = staves(lines)
    if len(found) != STAVES_PER_SYSTEM:
        raise ValueError(f"{len(found)} staves")
    upper, lower = found
    return SystemOnPage(Staves((upper.centre_row, lower.centre_row), upper.interline),
                        closing_barline_right(strokes, upper, lower, expected))


def ink_crop(rgba: np.ndarray, system: SystemOnPage, rng: random.Random) -> tuple[np.ndarray, Staves]:
    """The system's ink (the page's alpha up to its closing barline) black on white, at a
    random interline, with random margins; and its staves there."""
    ink = rgba[:, :math.ceil(system.right) + 1, 3]
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        raise ValueError("no ink")
    if ys.min() == 0 or ys.max() == ink.shape[0] - 1 or xs.min() == 0:
        raise ValueError("cut by the page edge")
    crop = 255 - ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    target = rng.uniform(*INTERLINE_PX)
    scale = target / system.staves.interline
    height = max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    m = [int(rng.uniform(*MARGIN_INTERLINES) * target) for _ in range(4)]
    factor = height / crop.shape[0]
    staves = Staves(tuple(row_resized(c - ys.min(), factor) + m[0] for c in system.staves.centres),
                    system.staves.interline * factor)
    return cv2.copyMakeBorder(resized, m[0], m[1], m[2], m[3], cv2.BORDER_CONSTANT, value=255), staves


def engraved_system(musicxml: Path, expected: ExpectedDrawing, rng: random.Random) -> tuple[np.ndarray, Staves]:
    """The crop MuseScore made of the system, and its staves there. Raises ValueError if it
    did not engrave it as one system on one page."""
    png, svg = page_files(musicxml, 1)
    if not png.exists() or not svg.exists() or png.stat().st_size == 0:
        raise ValueError("not engraved")
    if page_files(musicxml, 2)[0].exists():
        raise ValueError("two pages")
    system = system_on_page(svg.read_bytes(), expected)
    rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("unreadable page")
    return ink_crop(rgba, system, rng)


# ---- scores ----------------------------------------------------------------------------------

SystemChooser = Callable[[Path, random.Random], list[LabelledSystem]]  # a score's systems to engrave


@dataclass
class ScoreJob:
    index: int
    source: str
    systems: list[LabelledSystem]
    outcome: dict[str, str]

    @staticmethod
    def prepared(index: int, source: str, choose: SystemChooser) -> "ScoreJob":
        try:
            return ScoreJob(index, source, choose(Path(source), random.Random(index)), {})
        except Exception as e:  # a score we cannot read is left out, as by build_seq_dataset
            return ScoreJob(index, source, [], {"score": f"unreadable: {type(e).__name__}"})

    def crop_rng(self, system: LabelledSystem) -> random.Random:
        """A system's own, so that one MuseScore fails on does not change the others' crops."""
        return random.Random(f"crop:{self.index}:{system.name}")

    def tuplet_rng(self, system: LabelledSystem) -> random.Random:
        return random.Random(f"tuplets:{self.index}:{system.name}")


def score_dir(out: Path, index: int) -> Path:
    return out / "samples" / f"{index:06d}"


def system_file(work: Path, job: ScoreJob, system: LabelledSystem) -> Path:
    return work / f"{job.index:06d}" / f"{system.name}.musicxml"


def write_scores(jobs: list[ScoreJob], work: Path, musescore: MuseScore, style: str, out: Path) -> None:
    """Engraves the scores' systems in one style and writes each score's crops."""
    systems: dict[Path, tuple[ScoreJob, LabelledSystem]] = {}
    for job in jobs:
        for system in job.systems:
            path = system_file(work, job, system)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(system.musicxml)
            systems[path] = (job, system)
    plain = [m for m, (_, system) in systems.items() if system.tuplets is None]
    not_drawn = {m: CRASHED for m in engrave(musescore, plain, work, style, PAGES)}
    not_drawn |= engrave_with_tuplets(musescore, {m: js for m, js in systems.items() if js[1].tuplets is not None}, work, style)
    for job in jobs:
        write_score(job, work, not_drawn, out)


def engrave_with_tuplets(musescore: MuseScore, systems: dict[Path, tuple[ScoreJob, LabelledSystem]], work: Path,
                         style: str) -> dict[Path, str]:
    """MusicXML to MuseScore's file, the tuplets drawn there (mscx_tuplets), then to the page;
    returns the systems not drawn and why."""
    not_drawn = {m: CRASHED for m in engrave(musescore, list(systems), work, style, MSCX)}
    mscxs = []
    for musicxml, (job, system) in systems.items():
        if musicxml in not_drawn:
            continue
        if not drawn_in_mscx(musicxml, musicxml.with_suffix(".mscx"), system.tuplets.look, job.tuplet_rng(system)):
            not_drawn[musicxml] = "tuplets do not pair"
            continue
        mscxs.append(musicxml.with_suffix(".mscx"))
    return not_drawn | {m.with_suffix(".musicxml"): CRASHED for m in engrave(musescore, mscxs, work, style, PAGES)}


def write_score(job: ScoreJob, work: Path, not_drawn: dict[Path, str], out: Path) -> None:
    """The score's crops, labels and outcome, moved into place only once all are written."""
    partial = score_dir(out, job.index).with_suffix(".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    written = 0
    for system in job.systems:
        musicxml = system_file(work, job, system)
        if musicxml in not_drawn:
            job.outcome[system.name] = not_drawn[musicxml]
            continue
        try:
            image, staves = engraved_system(musicxml, system.expected, job.crop_rng(system))
        except ValueError as e:
            job.outcome[system.name] = str(e)
            continue
        write_sample(partial / f"s{written}", image, staves, system.lmx)
        job.outcome[system.name] = WRITTEN
        written += 1
    (partial / "outcome.json").write_text(json.dumps({"source": job.source, **job.outcome}) + "\n")
    partial.rename(score_dir(out, job.index))


@dataclass(frozen=True)
class Batch:
    """Scores (index, source) of one style variant, for one MuseScore process."""
    variant: int
    scores: list[tuple[int, str]]


@dataclass(frozen=True)
class RunSettings:
    out: Path
    musescore: MuseScore
    choose: SystemChooser  # module-level (or a partial of one): the spawned workers unpickle it


def render_batch(batch: Batch, run: RunSettings) -> int:
    """Renders the batch in a worker; returns how many scores it held."""
    work = run.out / "work" / f"{batch.scores[0][0]:06d}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        jobs = [ScoreJob.prepared(index, source, run.choose) for index, source in batch.scores]
        write_scores(jobs, work, run.musescore, engraving_style(batch.variant), run.out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return len(batch.scores)


def batches(scores: list[tuple[int, str]], size: int) -> list[Batch]:
    """The scores grouped by style variant, `size` at a time."""
    by_variant: dict[int, list[tuple[int, str]]] = {}
    for score in scores:
        by_variant.setdefault(style_variant(score[0]), []).append(score)
    return [Batch(variant, group[i:i + size]) for variant, group in sorted(by_variant.items())
            for i in range(0, len(group), size)]


# ---- the run ---------------------------------------------------------------------------------

def discard_unfinished(out: Path) -> None:
    """What a killed run leaves: half-written scores and MuseScore's job folders."""
    for partial in (out / "samples").glob("*.partial"):
        shutil.rmtree(partial)
    shutil.rmtree(out / "work", ignore_errors=True)


def write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_sample_list(out: Path, indices: set[int]) -> int:
    names = sorted(f"samples/{p.parent.name}/{p.stem}" for p in (out / "samples").glob("[0-9]*/*.lmx")
                   if p.parent.name.isdigit() and int(p.parent.name) in indices)
    write_atomically(out / "samples.train.txt", "".join(f"{n}\n" for n in names))
    return len(names)


def outcomes(out: Path, indices: set[int]) -> Counter:
    """What became of the finished scores' systems, and why any score was left out whole."""
    tally: Counter = Counter()
    for index in indices:
        path = score_dir(out, index) / "outcome.json"
        if path.exists():
            outcome = json.loads(path.read_text())
            outcome.pop("source")
            tally.update(f"score: {v}" if k == "score" else v for k, v in outcome.items())
    return tally


def raise_on_sigterm() -> None:
    """`kill` stops the run as Ctrl-C does, through the clean-up below."""
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems-per-score", type=int, default=4)
    ap.add_argument("--renderers", type=int, default=6, help="MuseScore processes at once")
    ap.add_argument("--scores-per-batch", type=int, default=4, help="scores per MuseScore process")
    ap.add_argument("--limit", type=int, help="only the first N scores of the list")
    ap.add_argument("--musescore", type=Path, default=ROOT / "tools" / "musescore" / "squashfs-root")
    ap.add_argument("--tuplet-hiding", choices=sorted(HIDING_MIXES),
                    help="hide tuplet numbers by a mix of seq.tuplet_display: as engravers do, or often")
    ap.add_argument("--varied-tuplet-looks", action="store_true", help="set printed tuplet numbers in varied ways")
    args = ap.parse_args()
    raise_on_sigterm()
    os.nice(NICENESS)
    sources = args.scores.read_text().split("\n")[:args.limit]
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    scores = [(i, s) for i, s in enumerate(sources) if is_training_score(s, conflicts)]
    indices = {i for i, _ in scores}
    discard_unfinished(args.out)
    (args.out / "samples").mkdir(parents=True, exist_ok=True)
    pending = [(i, s) for i, s in scores if not score_dir(args.out, i).exists()]
    work = batches(pending, args.scores_per_batch)
    drawing = TupletDrawing(args.tuplet_hiding, args.varied_tuplet_looks)
    run = RunSettings(args.out, MuseScore(args.musescore),
                      functools.partial(labelled_systems, k=args.systems_per_score, drawing=drawing))
    print(f"{len(scores)} scores, {len(pending)} to render in {len(work)} batches", flush=True)
    start = time.time()
    done = 0
    # "spawn": workers must not inherit the parent's signal handler or its pool.
    pool = mp.get_context("spawn").Pool(args.renderers, initializer=die_with_parent)
    try:
        for n in pool.imap_unordered(functools.partial(render_batch, run=run), work):
            done += n
            if done % 200 < n:
                print(f"{done}/{len(pending)} scores, {(time.time() - start) / done:.2f} s per score", flush=True)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
        discard_unfinished(args.out)
        systems = write_sample_list(args.out, indices)
        print(f"{systems} systems from {len(scores)} scores; outcomes: {dict(outcomes(args.out, indices))}")


if __name__ == "__main__":
    main()
