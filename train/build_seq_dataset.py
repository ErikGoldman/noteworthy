"""Renders system crops with LMX transcriptions from our training scores, for the
sequence model (train_seq.py).

usage: build_seq_dataset.py --scores LIST.txt --out DIR [--systems-per-score K] [--workers W]
                            [--unmarked-tuplets] [--thick-beams]

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
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
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

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("NW_DATA", ROOT / "data"))
sys.path.insert(0, str(DATA / "raw" / "olimpic" / "code" / "repo"))
from app.linearization.Linearizer import Linearizer  # noqa: E402
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


def linearize(part: ET.Element) -> str:
    linearizer = Linearizer(errout=None)
    linearizer.process_part(part)
    return " ".join(linearizer.output_tokens)


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


def render_system(tk: verovio.toolkit, musicxml: str, rng: random.Random, thick_beams: bool = False) -> np.ndarray | None:
    """The system as a grayscale image cropped to its ink plus a random margin."""
    options = engraving_options(rng)
    tk.setOptions(options)
    if not tk.loadData(musicxml) or tk.getPageCount() != 1:
        return None
    interline = rng.uniform(*INTERLINE_PX)
    svg = layout_svg(tk, tk.renderToSVG(1), interline / (UNITS_PER_INTERLINE_PER_UNIT * options["unit"]))
    if thick_beams:
        svg = thicken_beams(svg, rng.uniform(*SINGLE_BEAM_THICKENING))
    use_smufl_fonts()  # music symbols in text (tempo marks) need the SMuFL fonts in cairo
    png = cairosvg.svg2png(bytestring=svg, background_color="white")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(image < 128)
    if len(ys) == 0:
        return None
    m = [int(rng.uniform(*MARGIN_INTERLINES) * interline) for _ in range(4)]
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.copyMakeBorder(crop, m[0], m[1], m[2], m[3], cv2.BORDER_CONSTANT, value=255)


def build_score(job: tuple[int, str, Path, int, bool, bool]) -> int:
    """Renders up to K systems of one score; returns how many."""
    index, source, out, k, unmarked_tuplets, thick_beams = job
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
            lmx = linearize(system.part)
            score = part_to_score(system.part).getroot()
            if rng.random() < PART_NAME_HIDDEN:
                hide_part_name(score)
            if unmarked_tuplets:
                hide_tuplet_numbers(score, rng)
            musicxml = ET.tostring(score, encoding="unicode", xml_declaration=True)
            image = render_system(tk, musicxml, rng, thick_beams) if 10 <= len(lmx.split()) <= MAX_TOKENS else None
        except Exception:
            # Nothing may escape to the pool: CairoError cannot be unpickled, which kills
            # the pool's result thread and silently stalls every later score.
            continue
        if image is None:
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
    args = ap.parse_args()
    sources = args.scores.read_text().split("\n")
    hung_log = args.out / "hung.txt"
    hung = set(hung_log.read_text().split("\n")) if hung_log.exists() else set()
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    jobs = [(i, s, args.out, args.systems_per_score, args.unmarked_tuplets, args.thick_beams)
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
