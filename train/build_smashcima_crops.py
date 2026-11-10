"""Handwritten system crops with LMX transcriptions, drawn by Smashcima (our fork, ../Smashcima)
from MUSCIMA++ writers' symbols, for the sequence model (train_seq.py).

usage (in .venv-smashcima, from the repo root):
  MC_ASSETS_CACHE=data/raw/smashcima .venv-smashcima/bin/python train/build_smashcima_crops.py \
      --scores LIST.txt --out DIR [--systems-per-score K] [--workers W] [--limit N]

Scores, systems and labels are chosen as for the Verovio crops (build_seq_dataset.py): the same
training scores, the first two-staff part cut into systems of a few measures, each made
self-contained by OLiMPiC's split_part_to_systems and labelled by linearize. Before labelling,
each system loses what Smashcima does not draw and can go without (drawable_system): grace notes,
ornaments, fermatas, arpeggios, strong accents, ottava signs (its notes then sit where they
sound, as the label gives them) and the numbers of tuplets other than triplets. A system whose
label still holds a token Smashcima cannot draw (UNDRAWN_TOKENS: 32nd notes and shorter, breves,
hidden rests, double flats, stemless notes) is left out.

Each system is drawn on its own page, sized to it, in one MUSCIMA++ writer's hand on a scanned
paper texture, and resized to a random interline. Glyphs come only from MUSCIMA++ pages outside
its two test sets, which are our handwritten dev and test pages (the fork leaves them out).

Output: DIR/samples/<index>/s<k>.png + .lmx and DIR/samples.train.txt, read by train_seq.py as
--data DIR:train. A score's directory appears only once complete, so the run may be killed at
any time and restarted.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seq_dataset import (DATA, INTERLINE_PX, MARGIN_INTERLINES, MAX_TOKENS,  # noqa: E402
                               PART_NAME_HIDDEN,
                               break_into_systems, draw_some_meters_as_symbols, hide_part_name,
                               is_training_score, linearize, load_musicxml, part_to_score,
                               piano_part, split_part_to_systems, without_octave_clefs)
from build_musescore_crops import without_octave_shifts, write_sample_list  # noqa: E402
from omr.pieces import pdmx_license_conflicts  # noqa: E402

MIN_TOKENS = 10
NICENESS = 10

# Tokens of a label that Smashcima cannot draw: a system holding one is left out.
UNDRAWN_TOKENS = {"32nd", "64th", "128th", "256th", "512th", "1024th", "breve", "long", "maxima",
                  "print-object:no", "flat-flat", "stem:none"}
STRIPPED_NOTATIONS = ("fermata", "arpeggiate", "non-arpeggiate", "ornaments", "technical",
                      "glissando", "slide", "dynamics", "accidental-mark", "other-notation")
DRAWN_ARTICULATIONS = {"staccato", "accent", "tenuto"}

# Page around one system, in millimetres (MUSCIMA++'s staff space is about 2.4 mm).
STAFF_HEIGHT_MM = 4 * 2.435
STAFF_GAP_MM = (9.0, 16.0)
"Between the two staves of the system"
PAGE_PADDING_MM = 30.0
"Above and below the system on its page: room for any ledger lines and marks, cut off again"
SIDE_PADDING_MM = 3.0
MM_PER_COLUMN = (5.5, 8.0)
"Width per note column (a distinct onset in a measure); Smashcima packs columns tightly"
MM_PER_MEASURE = 5.0
HEADER_MM = 32.0


def without_graces(part: ET.Element) -> None:
    for measure in part.findall("measure"):
        for note in measure.findall("note"):
            if note.find("grace") is not None:
                measure.remove(note)


def without_undrawn_notations(part: ET.Element) -> None:
    """Strips marks Smashcima does not draw and whose loss leaves the music intact, so that
    neither the label nor the image has them"""
    for notations in part.iter("notations"):
        for child in list(notations):
            if child.tag in STRIPPED_NOTATIONS:
                notations.remove(child)
            elif child.tag == "articulations":
                for articulation in list(child):
                    if articulation.tag not in DRAWN_ARTICULATIONS:
                        child.remove(articulation)


def with_only_triplet_numbers(part: ET.Element) -> None:
    """Hides the number of every tuplet but a triplet's: Smashcima draws only 3s (MUSCIMA++ has
    no other tuplet numbers worth sampling). The label then says the number is hidden."""
    for note in part.iter("note"):
        actual = note.findtext("time-modification/actual-notes")
        for tuplet in note.findall("notations/tuplet"):
            number = tuplet.findtext("tuplet-actual/tuplet-number") or actual
            if tuplet.get("type") == "start" and number != "3":
                tuplet.set("show-number", "none")


def drawable_system(part: ET.Element) -> None:
    without_graces(part)
    without_undrawn_notations(part)
    without_octave_shifts(part)
    with_only_triplet_numbers(part)


def label_drawn(lmx: str) -> bool:
    tokens = lmx.split()
    return MIN_TOKENS <= len(tokens) <= MAX_TOKENS and not UNDRAWN_TOKENS.intersection(tokens)


@dataclass
class LabelledSystem:
    musicxml: str
    lmx: str
    measures: int
    columns: int
    "Distinct onsets summed over the measures: how much room the system needs"


def note_columns(part: ET.Element) -> int:
    """Distinct onsets of notes and rests, measure by measure"""
    total = 0
    for measure in part.findall("measure"):
        onset, onsets = 0, set()
        for child in measure:
            duration = int(child.findtext("duration") or 0)
            if child.tag == "backup":
                onset -= duration
            elif child.tag == "forward":
                onset += duration
            elif child.tag == "note" and child.find("chord") is None:
                onsets.add(onset)
                onset += duration
        total += len(onsets)
    return total


def labelled_systems(source: Path, k: int, rng: random.Random) -> list[LabelledSystem]:
    """Up to k random drawable systems of the score's piano part"""
    part = piano_part(load_musicxml(source))
    if part is None:
        return []
    without_octave_clefs(part)
    break_into_systems(part, rng)
    parts = [s.part for page in split_part_to_systems(part) for s in page.systems]
    rng.shuffle(parts)
    chosen = []
    for system in parts:
        if len(chosen) == k:
            break
        drawable_system(system)
        draw_some_meters_as_symbols(system, rng)
        lmx = linearize(system)
        if not label_drawn(lmx):
            continue
        score = part_to_score(system).getroot()
        if rng.random() < PART_NAME_HIDDEN:
            hide_part_name(score)
        chosen.append(LabelledSystem(ET.tostring(score, encoding="unicode", xml_declaration=True),
                                     lmx, len(system.findall("measure")), note_columns(system)))
    return chosen


class Drawer:
    """One Smashcima model per worker process, reseeded per system"""

    def __init__(self):
        import smashcima as sc
        from smashcima.geometry import Vector2
        from smashcima.synthesis.page.SimplePageSynthesizer import PageSetup
        self.sc, self.Vector2, self.PageSetup = sc, Vector2, PageSetup
        self.model = sc.orchestration.BaseHandwrittenModel()

    def page_for(self, system: LabelledSystem, rng: random.Random):
        width = HEADER_MM + system.measures * MM_PER_MEASURE + system.columns * rng.uniform(*MM_PER_COLUMN)
        height = 2 * PAGE_PADDING_MM + 2 * STAFF_HEIGHT_MM + rng.uniform(*STAFF_GAP_MM)
        return self.PageSetup(size=self.Vector2(width + 2 * SIDE_PADDING_MM, height),
                              padding_top=PAGE_PADDING_MM, padding_bottom=PAGE_PADDING_MM,
                              padding_left=SIDE_PADDING_MM, padding_right=SIDE_PADDING_MM, staff_count=2)

    def ink_box_px(self, page, dpi: float) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) of everything drawn on the page but the paper, in pixels"""
        points = []
        for sprite, transform in self.sc.scene.Sprite.traverse_sprites(page.space):
            if sprite.space is page.space:  # the paper texture
                continue
            for corner in ((0, 0), (sprite.pixel_width, 0), (0, sprite.pixel_height),
                           (sprite.pixel_width, sprite.pixel_height)):
                points.append(transform.apply_to(self.Vector2(*corner)))
        origin = page.view_box.rectangle
        to_px = dpi / 25.4
        xs = [(p.x - origin.x) * to_px for p in points]
        ys = [(p.y - origin.y) * to_px for p in points]
        return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))

    def draw(self, system: LabelledSystem, rng: random.Random) -> np.ndarray:
        """The system as a grayscale crop at a random interline, with random margins of its
        paper"""
        self.model.rng.seed(rng.random())
        self.model.page_synthesizer.page_setup = self.page_for(system, rng)
        scene = self.model(data=system.musicxml, format=".musicxml")
        if len(scene.pages) != 1:
            raise ValueError("system broke onto a second page")
        page = scene.pages[0]
        gray = cv2.cvtColor(scene.render(page)[:, :, :3], cv2.COLOR_BGR2GRAY)
        interline_px = 28.75 * scene.dpi / 300  # MUSCIMA++'s staff space at the render DPI
        left, top, right, bottom = self.ink_box_px(page, scene.dpi)
        margins = [int(rng.uniform(*MARGIN_INTERLINES) * interline_px) for _ in range(4)]
        top, bottom = max(0, top - margins[0]), min(gray.shape[0], bottom + margins[1])
        left, right = max(0, left - margins[2]), min(gray.shape[1], right + margins[3])
        if top == 0 or bottom == gray.shape[0] or right == gray.shape[1]:
            raise ValueError("system reaches the page edge")
        crop = gray[top:bottom, left:right]
        scale = rng.uniform(*INTERLINE_PX) / interline_px
        return cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)


DRAWER: Drawer | None = None


def start_worker() -> None:
    global DRAWER
    os.nice(NICENESS)
    DRAWER = Drawer()


def score_dir(out: Path, index: int) -> Path:
    return out / "samples" / f"{index:06d}"


def build_score(job: tuple[int, str, Path, int]) -> int:
    """Draws up to k systems of one score; returns how many"""
    index, source, out, k = job
    target = score_dir(out, index)
    rng = random.Random(index)
    try:
        systems = labelled_systems(Path(source), k, rng)
    except Exception:
        systems = []
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    written = 0
    for system in systems:
        try:
            image = DRAWER.draw(system, rng)
        except Exception:
            # Nothing may escape to the pool (see build_seq_dataset.build_score).
            continue
        cv2.imwrite(str(partial / f"s{written}.png"), image)
        (partial / f"s{written}.lmx").write_text(system.lmx + "\n")
        written += 1
    partial.rename(target)
    return written


def discard_unfinished(out: Path) -> None:
    for partial in (out / "samples").glob("*.partial"):
        shutil.rmtree(partial)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems-per-score", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N scores of the list")
    args = ap.parse_args()
    os.nice(NICENESS)
    sources = args.scores.read_text().split("\n")[:args.limit]
    conflicts = pdmx_license_conflicts(DATA / "raw" / "pdmx")
    scores = [(i, s) for i, s in enumerate(sources) if is_training_score(s, conflicts)]
    indices = {i for i, _ in scores}
    discard_unfinished(args.out)
    (args.out / "samples").mkdir(parents=True, exist_ok=True)
    pending = [(i, s, args.out, args.systems_per_score) for i, s in scores
               if not score_dir(args.out, i).exists()]
    print(f"{len(scores)} scores, {len(pending)} to draw", flush=True)
    start, done, systems = time.time(), 0, 0
    # "spawn": forked workers would share the parent's state and Smashcima's caches.
    with mp.get_context("spawn").Pool(args.workers, initializer=start_worker) as pool:
        try:
            for n in pool.imap_unordered(build_score, pending, chunksize=4):
                done += 1
                systems += n
                if done % 500 == 0:
                    print(f"{done}/{len(pending)} scores, {systems} systems, "
                          f"{(time.time() - start) / done:.2f} s per score", flush=True)
        finally:
            discard_unfinished(args.out)
            print(f"{write_sample_list(args.out, indices)} systems from {len(scores)} scores")


if __name__ == "__main__":
    main()
