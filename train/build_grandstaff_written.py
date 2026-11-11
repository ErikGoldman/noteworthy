"""GrandStaff-LMX's training set with C and cut time (seq.time_symbols) and barline signs
(seq.barlines) as tokens.

usage: build_grandstaff_written.py GRANDSTAFF_ROOT REBUILT_DIR OUT_DIR

REBUILT_DIR is rebuild_grandstaff_lmx.py's output: a system takes its mended LMX from there (the
clefs, keys and meters its source has), keeps the release's where the rebuild gave it back
unchanged, and is left out where it disagrees with its source.

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
"""
from __future__ import annotations

import random
import sys
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


def main() -> None:
    root, rebuilt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    status = dict(line.split("\t")[::2] for line in (rebuilt / "systems.tsv").read_text().splitlines())
    kept, unmatched, with_tuplets, unlike_source, cut, graced, blank = [], 0, 0, 0, 0, 0, 0
    for name in (root / "samples.train.txt").read_text().split():
        if status[name] not in (CORRECT, MENDED):
            unlike_source += 1
            continue
        labelled = rebuilt if status[name] == MENDED else root
        kern = (root / f"{name}.krn").read_text()
        source = kern_reader.read(kern)
        if source.cut_at_merger:
            cut += 1
            continue
        if source.grace_notes:
            graced += 1
            continue
        lmx = with_kern_time_symbols((labelled / f"{name}.lmx").read_text().strip(), kern)
        lmx = lmx and with_kern_barline_tokens(lmx, kern)
        if lmx is None:
            unmatched += 1
            continue
        if "tuplet:start" in lmx.split():
            with_tuplets += 1
            continue
        image = root / (f"{name}_distorted.jpg" if random.Random(name).random() < DISTORTED_SHARE else f"{name}.jpg")
        if is_blank(image):
            blank += 1
            continue
        target = out / f"{name}.lmx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(lmx + "\n")
        target.with_suffix(".jpg").unlink(missing_ok=True)
        target.with_suffix(".jpg").symlink_to(image.resolve())
        kept.append(name)
    (out / "samples.train.txt").write_text("\n".join(kept) + "\n")
    print(f"{len(kept)} systems kept; left out: {unlike_source} unlike their source, {unmatched} with signs "
          f"that cannot be matched up, {with_tuplets} with tuplets, {cut} with notes only before a merger of "
          f"the staves, {graced} with grace notes, {blank} with a blank image")


if __name__ == "__main__":
    main()
