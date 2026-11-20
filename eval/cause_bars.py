"""Spec 4's measure (docs/EVAL_SETS.md): label bars read with the right rhythm, by what the bar holds (bar_classes),
a share per piece, then the mean over pieces. A piece is a crop's directory: an OLiMPiC song, or a
held-out PDMX score of build_cause_sets.py's sets. A piece with fewer than MIN_BARS bars of a class
is left out of that class; a class with fewer than MIN_PIECES pieces is reported as too few to
decide. Reported per readings file and source (OLiMPiC's synthetic and scanned, or a set's clean
and degraded crops).

usage: cause_bars.py READINGS.jsonl [...] [--olimpic-clean]
  READINGS: outer_notes.py's or read_samples.py's readings (a row's "reading", or the first of its
  "readings"). --olimpic-clean keeps OLiMPiC dev's songs with no copy in a training source
  (data/olimpic_dev_clean.txt), as spec 4 on OLiMPiC is scored (EXPERIMENTS.md, 2026-11-18).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bar_classes import CLASSES, classes_of, right_bars

MIN_BARS, MIN_PIECES = 5, 20
OLIMPIC_CLEAN = Path(__file__).resolve().parent.parent / "data" / "olimpic_dev_clean.txt"

Counts = dict[str, dict[str, list[int]]]  # piece -> class -> [right, bars]


def reading_of(row: dict) -> str:
    return row["readings"][0] if "readings" in row else row["reading"]


def by_piece(rows: list[dict]) -> Counts:
    out: Counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        piece = str(Path(row["name"]).parent)
        for bar, right in right_bars(row["label"], reading_of(row)):
            for name in classes_of(bar):
                out[piece][name][0] += right
                out[piece][name][1] += 1
    return out


def summary(pieces: Counts) -> dict[str, tuple[float | None, int]]:
    """class -> (mean over pieces of the share right, or None if too few; pieces counted)."""
    out = {}
    for name in CLASSES:
        shares = [right / bars for counts in pieces.values() for right, bars in [counts.get(name, [0, 0])] if bars >= MIN_BARS]
        out[name] = (sum(shares) / len(shares) if len(shares) >= MIN_PIECES else None, len(shares))
    return out


def cell(share: float | None, pieces: int) -> str:
    return f"too few ({pieces})" if share is None else f"{100 * share:.1f}% ({pieces})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("readings", type=Path, nargs="+")
    ap.add_argument("--olimpic-clean", action="store_true")
    args = ap.parse_args()
    clean = set(OLIMPIC_CLEAN.read_text().split()) if args.olimpic_clean else None
    print(f"{'readings':48s}{'source':>10s}" + "".join(f"{name:>24s}" for name in CLASSES))
    for path in args.readings:
        rows = [r for r in map(json.loads, path.open()) if clean is None or r["name"] in clean]
        for source in sorted({r.get("source", "") for r in rows}):
            cells = summary(by_piece([r for r in rows if r.get("source", "") == source]))
            print(f"{str(path)[-48:]:48s}{source:>10s}" + "".join(f"{cell(*c):>24s}" for c in cells.values()))


if __name__ == "__main__":
    main()
