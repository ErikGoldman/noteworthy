"""GrandStaff-LMX's training set with C and cut time (seq.time_symbols) and barline signs
(seq.barlines) as tokens.

usage: build_grandstaff_written.py GRANDSTAFF_ROOT OUT_DIR (--rebuilt REBUILT_DIR | --release-labels)

REBUILT_DIR is rebuild_grandstaff_lmx.py's output: a system takes its mended LMX from there (the
clefs, keys and meters its source has), keeps the release's where the rebuild gave it back
unchanged, and is left out where it disagrees with its source. That cleaning (2026-11-11) also
leaves out the blank images, the systems cut at a merger and those with grace notes, below.
--release-labels leaves the cleaning out whole, for TODO 3d's run: the release's LMX everywhere,
only the filters that came before and after it (tuplets, signs not matched up, ties at an edge).

GrandStaff's images draw a `*met(c)` or `*met(c|)` of the **kern source as a symbol, whatever
the meter (a Scarlatti 3/8 shows C), and its repeats, double and final bars, but its LMX spells
only the numbers and has no barlines. The labels here add their tokens; images are linked, not
copied. A system whose signs cannot be matched up with its LMX (time signatures or measures not
the same in number) is left out, and so is every system with a tuplet: its images often draw no
number where the label says one is shown (the later groups of a run, or none at all), and draw a
sextuplet as one 6 where the label has two triplets (checked by eye, 2026-11-10: about a third of
16 sampled). Labels cannot say which; the images are given, so they cannot be redrawn. Half the
systems
link GrandStaff's distorted image (a camera's warp and light) in place of the clean render. A
system whose image is blank is left out: 29 of GrandStaff's images, most of them distorted ones,
are one flat grey or nearly black. And a system whose source merges the two staves into one spine
with notes after it: Verovio drew GrandStaff's image with no notes after the merger, only empty
measures, and OLiMPiC's build cut the label at it, so the label lacks measures the image draws. And every system with a grace note in its source: GrandStaff's
images draw each one as a stemless notehead, often under a stray dash, like no printed grace note,
and two thirds of their labels have none (checked by eye, 2026-11-11: drawn in all 8 sampled).
And every system with a tie from the system before or into the one after: its image draws the
half arc, but its label has no tie (OLiMPiC's build lost them: MuseScore drops a tie with no note
to end on), and the other sets label such arcs (EXPERIMENTS.md, 2026-11-13).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2

from rebuild_grandstaff_lmx import CORRECT, MENDED
from seq import kern as kern_reader
from seq.barlines import with_kern_barline_tokens
from seq.time_symbols import with_kern_time_symbols

DISTORTED_SHARE = 0.5
# GrandStaff's blank images, a quarter the size: no contrast (one grey), or nearly all black.
BLANK_CONTRAST, BLANK_BRIGHTNESS = 8, 60


def is_blank(image: Path) -> bool:
    pixels = cv2.imread(str(image), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return pixels.std() < BLANK_CONTRAST or pixels.mean() < BLANK_BRIGHTNESS


def cleaning_leaves_out(name: str, status: dict[str, str], source, image: Path) -> str | None:
    """Why the 2026-11-11 cleaning leaves this system out, or None."""
    if status[name] not in (CORRECT, MENDED):
        return "unlike their source"
    if source.cut_at_merger:
        return "with notes only before a merger of the staves"
    if source.grace_notes:
        return "with grace notes"
    if is_blank(image):
        return "with a blank image"
    return None


def always_left_out(source, lmx: str | None) -> str | None:
    """Why this system is left out whichever labels it takes, or None."""
    if source.ties_at_edges:
        return "with a tie across an edge"
    if lmx is None:
        return "with signs that cannot be matched up"
    if "tuplet:start" in lmx.split():
        return "with tuplets"
    return None


def readable_source(kern: str):
    try:
        return kern_reader.read(kern)
    except kern_reader.Unreadable:
        return None


def written_label(lmx: str, kern: str) -> str | None:
    lmx = with_kern_time_symbols(lmx.strip(), kern)
    return lmx and with_kern_barline_tokens(lmx, kern)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="GrandStaff-LMX")
    ap.add_argument("out", type=Path)
    labels = ap.add_mutually_exclusive_group(required=True)
    labels.add_argument("--rebuilt", type=Path, help="rebuild_grandstaff_lmx.py's output: the cleaned set")
    labels.add_argument("--release-labels", action="store_true", help="the release's labels, the cleaning left out (TODO 3d)")
    args = ap.parse_args()
    root, out = args.root, args.out
    status = (dict(line.split("\t")[::2] for line in (args.rebuilt / "systems.tsv").read_text().splitlines())
              if args.rebuilt else None)
    kept, left_out = [], Counter()
    for name in (root / "samples.train.txt").read_text().split():
        kern = (root / f"{name}.krn").read_text()
        source = readable_source(kern)
        image = root / (f"{name}_distorted.jpg" if random.Random(name).random() < DISTORTED_SHARE else f"{name}.jpg")
        labelled = args.rebuilt if status is not None and status[name] == MENDED else root
        lmx = written_label((labelled / f"{name}.lmx").read_text(), kern)
        if source is None:
            reason = "whose source's spines cannot be followed"
        else:
            reason = ((cleaning_leaves_out(name, status, source, image) if status is not None else None)
                      or always_left_out(source, lmx))
        if reason:
            left_out[reason] += 1
            continue
        target = out / f"{name}.lmx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(lmx + "\n")
        target.with_suffix(".jpg").unlink(missing_ok=True)
        target.with_suffix(".jpg").symlink_to(image.resolve())
        kept.append(name)
    (out / "samples.train.txt").write_text("\n".join(kept) + "\n")
    print(f"{len(kept)} systems kept; left out: " + ", ".join(f"{n} {why}" for why, n in left_out.most_common()))


if __name__ == "__main__":
    main()
