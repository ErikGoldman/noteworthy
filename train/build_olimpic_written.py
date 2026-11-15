"""An OLiMPiC split (by default the synthetic training set) with pitches as printed and ottava
tokens (seq.ottava).

usage: build_olimpic_written.py OLIMPIC_ROOT OUT_DIR [SPLIT]

OLIMPIC_ROOT is olimpic-1.0-synthetic or olimpic-1.0-scanned (whose systems are the synthetic
dev and test systems, printed and scanned); SPLIT train, dev or test, written to
OUT_DIR/samples.SPLIT.txt.

Labels come from each system's MusicXML; images are linked, not copied. A system that opens
with the stop of a sign from an earlier system is left out: its file does not say which way
the sign went, and the image shows the notes shifted. A system wholly inside a long sign
has no trace of it, so it stays, with its sounding pitches.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from build_seq_dataset import linearize


def opens_inside_a_sign(part: ET.Element) -> bool:
    first_on_staff = {}
    for direction in part.iter("direction"):
        for sign in direction.iter("octave-shift"):
            first_on_staff.setdefault(direction.findtext("staff", "1"), sign.get("type"))
    return "stop" in first_on_staff.values()


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    kept = []
    for name in (root / f"samples.{split}.txt").read_text().split():
        part = ET.parse(root / f"{name}.musicxml").find("part")
        if opens_inside_a_sign(part):
            continue
        target = out / f"{name}.lmx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(linearize(part) + "\n")
        target.with_suffix(".png").unlink(missing_ok=True)
        target.with_suffix(".png").symlink_to((root / f"{name}.png").resolve())
        kept.append(name)
    (out / f"samples.{split}.txt").write_text("\n".join(kept) + "\n")
    print(f"{len(kept)} systems kept")


if __name__ == "__main__":
    main()
