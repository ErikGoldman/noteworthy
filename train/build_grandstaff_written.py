"""GrandStaff-LMX's training set with C and cut time (seq.time_symbols) and barline signs
(seq.barlines) as tokens.

usage: build_grandstaff_written.py GRANDSTAFF_ROOT OUT_DIR

GrandStaff's images draw a `*met(c)` or `*met(c|)` of the **kern source as a symbol, whatever
the meter (a Scarlatti 3/8 shows C), and its repeats, double and final bars, but its LMX spells
only the numbers and has no barlines. The labels here add their tokens; images are linked, not
copied. A system whose signs cannot be matched up with its LMX (time signatures or measures not
the same in number) is left out. Half the systems
link GrandStaff's distorted image (a camera's warp and light) in place of the clean render.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from seq.barlines import with_kern_barline_tokens
from seq.time_symbols import with_kern_time_symbols

DISTORTED_SHARE = 0.5


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    kept, unmatched = [], 0
    for name in (root / "samples.train.txt").read_text().split():
        kern = (root / f"{name}.krn").read_text()
        lmx = with_kern_time_symbols((root / f"{name}.lmx").read_text().strip(), kern)
        lmx = lmx and with_kern_barline_tokens(lmx, kern)
        if lmx is None:
            unmatched += 1
            continue
        target = out / f"{name}.lmx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(lmx + "\n")
        target.with_suffix(".jpg").unlink(missing_ok=True)
        image = f"{name}_distorted.jpg" if random.Random(name).random() < DISTORTED_SHARE else f"{name}.jpg"
        target.with_suffix(".jpg").symlink_to((root / image).resolve())
        kept.append(name)
    (out / "samples.train.txt").write_text("\n".join(kept) + "\n")
    print(f"{len(kept)} systems kept, {unmatched} left out with signs that cannot be matched up")


if __name__ == "__main__":
    main()
