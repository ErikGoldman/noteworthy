"""Spec 4's sets by cause from rendered scores (docs/EVAL_SETS.md): system crops of unseen held-out
PDMX scores, engraved by MuseScore, one set a class (bar_classes): systems holding a printed tuplet
number (printed_tuplets), two voices on a staff (two_voices), a chord of three or more notes
(chords), and printed_tuplets' systems again with every tuplet number hidden (hidden_tuplets). Read by `outer_notes.py CHECKPOINT --data DIR/<set> --sources clean degraded`, scored by
`cause_bars.py`.

usage: build_cause_sets.py --out DIR [--sets NAME ...] [--renderers R] [--musescore SQUASHFS_ROOT]

Scores: the render list's unseen ones (held_out_copies.is_unseen: never trained on, no copy in a
training set), in list order, indexed by their place in the list so that a score's systems and
style are those build_musescore_crops.py would draw for it. Systems: the score cut into systems
as for the training crops, and, for each set, the first SYSTEMS_PER_SCORE whose label holds a bar
of its class, in score order, chosen by the label alone; a system may be in more than one set.
MuseScore 3.6.2 draws every tuplet's number, and the labels say so
(build_musescore_crops.show_every_tuplet_number): the few numbers authors hide are drawn and
labelled shown here. Authors' hidden numbers cannot make a set: unseen PDMX hides numbers in 6
scores (EXPERIMENTS.md, 2026-11-19). So hidden_tuplets hides every number of printed_tuplets'
systems (drawn through MuseScore's own file, as build_musescore_crops.py --tuplet-hiding does): the
same music, a test of reading a tuplet whose number is not drawn, not of when engravers hide one.
Hiding every number draws nothing from the rng, so the same systems are chosen.

Output, for each set: DIR/<set>/clean and DIR/<set>/degraded (the same crops through
omr.augment.capture_degradation, as synthetic_benchmark.py --degrade), each samples/<index>/s<k>.png
+ .lmx with samples.dev.txt, and DIR/<set>/staff_positions/{clean,degraded}.json; and
eval/cause_sets.txt, every set's systems and their scores, committed as the record of the sets.
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from held_out_copies import DATA, is_unseen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_musescore_crops import (LabelledSystem, MuseScore, RunSettings, batches, die_with_parent,  # noqa: E402
                                   discard_unfinished, labelled_system, render_batch)
from build_seq_dataset import break_into_systems, load_musicxml, piano_part, split_part_to_systems, without_octave_clefs  # noqa: E402
from find_staff_positions import positions_of  # noqa: E402
from omr.augment import capture_degradation  # noqa: E402
from seq.data import Sample  # noqa: E402
from bar_classes import CHORDS, TUPLET_HIDDEN, TUPLET_SHOWN, TWO_VOICES, bars_drawn, classes_of  # noqa: E402
from seq.tuplet_display import HIDE_ALL, PLAIN_LOOK, ScoreTuplets  # noqa: E402

@dataclass(frozen=True)
class CauseSet:
    kind: str                    # bar_classes' class a chosen system must hold
    numbers_hidden: bool = False  # every tuplet number hidden, before the system is labelled


SETS = {"printed_tuplets": CauseSet(TUPLET_SHOWN), "two_voices": CauseSet(TWO_VOICES), "chords": CauseSet(CHORDS),
        "hidden_tuplets": CauseSet(TUPLET_HIDDEN, numbers_hidden=True)}
EVERY_NUMBER_HIDDEN = ScoreTuplets(HIDE_ALL, PLAIN_LOOK)
SYSTEMS_PER_SCORE = 5
SPLIT = "dev"
RECORD = Path(__file__).resolve().parent / "cause_sets.txt"


def holds(system: LabelledSystem, kind: str) -> bool:
    return any(kind in classes_of(bar) for bar in bars_drawn(system.lmx))


def first_systems_holding(source: Path, rng: random.Random, chosen: CauseSet) -> list[LabelledSystem]:
    """The first SYSTEMS_PER_SCORE systems with a bar of the class. Raises ValueError without a
    two-staff part. Every system is labelled, kept or not, so the rng's draws (meters drawn as
    symbols, part names) do not depend on which are kept, nor on the set."""
    part = piano_part(load_musicxml(source))
    if part is None:
        raise ValueError("no two-staff part")
    without_octave_clefs(part)
    break_into_systems(part, rng)
    parts = [s.part for page in split_part_to_systems(part) for s in page.systems]
    tuplets = EVERY_NUMBER_HIDDEN if chosen.numbers_hidden else None
    labelled = [labelled_system(p, f"candidate{n}", rng, tuplets) for n, p in enumerate(parts)]
    return [s for s in labelled if s is not None and holds(s, chosen.kind)][:SYSTEMS_PER_SCORE]


def unseen_scores() -> list[tuple[int, str]]:
    listed = (DATA / "render_scores.txt").read_text().split()
    return [(i, s) for i, s in enumerate(listed) if is_unseen(Path(s))]


def engrave_clean(scores: list[tuple[int, str]], chosen: CauseSet, out: Path, musescore: MuseScore, renderers: int) -> None:
    (out / "samples").mkdir(parents=True)
    run = RunSettings(out, musescore, functools.partial(first_systems_holding, chosen=chosen))
    with mp.get_context("spawn").Pool(renderers, initializer=die_with_parent) as pool:
        for done, _ in enumerate(pool.imap_unordered(functools.partial(render_batch, run=run), batches(scores, 4)), 1):
            print(f"batch {done}", flush=True)
    discard_unfinished(out)


def sample_names(root: Path) -> list[str]:
    return sorted(f"samples/{p.parent.name}/{p.stem}" for p in (root / "samples").glob("[0-9]*/*.lmx"))


def crop_rng(name: str) -> np.random.Generator:
    """samples/<index>/s<k>'s own generator."""
    _, index, system = name.split("/")
    return np.random.default_rng([int(index), int(system.removeprefix("s"))])


def write_degraded(clean: Path, degraded: Path, names: list[str]) -> None:
    """Each crop degraded with its own generator, seeded by its name, so the set is the same
    whatever order or subset it is built in."""
    for name in names:
        stem, target = clean / name, degraded / name
        target.parent.mkdir(parents=True, exist_ok=True)
        staves = json.loads(stem.with_suffix(".staves.json").read_text())
        image = cv2.imread(str(stem.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(target.with_suffix(".png")), capture_degradation(crop_rng(name), image, staves[-1]))
        for suffix in (".lmx", ".staves.json"):
            shutil.copyfile(stem.with_suffix(suffix), target.with_suffix(suffix))


def write_split(root: Path, names: list[str], positions: Path) -> None:
    (root / f"samples.{SPLIT}.txt").write_text("".join(f"{n}\n" for n in names))
    staves = {n: positions_of(Sample(root / f"{n}.png", root / f"{n}.lmx")) for n in names}
    (positions / f"{root.name}.json").write_text(json.dumps({"split": SPLIT, "staves": staves}))


def write_record(names_by_set: dict[str, list[str]], sources: dict[int, str]) -> None:
    """The record's lines for these sets, replacing theirs; other sets' lines are kept."""
    lines = [f"{name}\t{n}\t{sources[int(n.split('/')[1])]}\n" for name, names in names_by_set.items() for n in names]
    kept = [line for line in RECORD.read_text().splitlines(keepends=True)[1:] if line.split("\t")[0] not in names_by_set] \
        if RECORD.exists() else []
    RECORD.write_text("# spec 4's rendered sets (build_cause_sets.py): set, sample, its score\n" + "".join(kept + lines))


def build_set(scores: list[tuple[int, str]], chosen: CauseSet, out: Path, musescore: MuseScore, renderers: int) -> list[str]:
    clean, degraded, positions = out / "clean", out / "degraded", out / "staff_positions"
    engrave_clean(scores, chosen, clean, musescore, renderers)
    names = sample_names(clean)
    write_degraded(clean, degraded, names)
    positions.mkdir()
    for root in (clean, degraded):
        write_split(root, names, positions)
    print(f"{out.name}: {len(names)} systems from {len({n.split('/')[1] for n in names})} scores", flush=True)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--renderers", type=int, default=6)
    ap.add_argument("--musescore", type=Path, default=ROOT / "tools" / "musescore" / "squashfs-root")
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"{args.out} exists: build into a fresh directory")
    scores = unseen_scores()
    print(f"{len(scores)} unseen held-out scores", flush=True)
    musescore = MuseScore(args.musescore)
    names = {name: build_set(scores, SETS[name], args.out / name, musescore, args.renderers) for name in args.sets}
    write_record(names, dict(scores))


if __name__ == "__main__":
    main()
