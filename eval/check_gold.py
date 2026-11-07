"""Checks gold as the scorer reads it, after gold_overrides.py.

usage: check_gold.py PUBLISHED_DIR CHANGED_DIR OVERRIDES.jsonl

Runs in .venv-musicdiff. Loads the authors' metric first: that registers converter21, whose
kern reader is the one OMR-NED sees. music21's own reader gives different notes, beams and
ottavas for the same file, so a check made with it says nothing about scores. A changed page
must keep every note the published page parses to, unless an override cut bars that are not on
it, and have equally long parts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omr_ned import authors_metric


def main() -> None:
    published, changed, overrides = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    shortened = {o["page"] for o in map(json.loads, overrides.read_text().splitlines()) if "end_after" in o}
    parse = authors_metric().m21.converter.parse
    sys.setrecursionlimit(100000)
    failed = 0
    for page in sorted(changed.glob("*.krn")):
        before, after = (parse(str(directory / page.name), forceSource=True) for directory in (published, changed))
        notes_before, notes_after = (sum(len(n.pitches) for n in score.recurse().notes) for score in (before, after))
        bars = [len(part.getElementsByClass("Measure")) for part in after.parts]
        if (notes_after < notes_before and page.stem not in shortened) or len(set(bars)) > 1:
            print(f"{page.stem}: notes {notes_before} to {notes_after}, bars {bars}")
            failed += 1
    print(f"{failed} pages failed")
    sys.exit(bool(failed))


if __name__ == "__main__":
    main()
