"""One row per kind of input, so no single piece stands for the recognizer.

usage: scorecard.py RUN_DIR [BASELINE_DIR] [--split dev|test]

Reads the results.json the benchmarks write under RUN_DIR (see ROWS); rows not run are
marked so. Each F1 comes with a 95% range from resampling whole pieces, as an item's
neighbours in a piece share its engraving and notation. With BASELINE_DIR each row also
shows the change from it, resampling the same pieces for both: a range that excludes 0
is a difference these pieces can support. A row of one piece has no range.

Decide on dev; test is held out (look at its numbers, never at its errors), and Night
Album counts as test. Rows measure different things and are not averaged into one score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from note_scoring import gt_notes

ROWS = {"rendered print, Verovio": "verovio", "rendered print, MuseScore": "musescore",
        "real scans, system crops": "olimpic-systems", "real scans, whole pages": "olimpic-pages",
        "print, varied repertoire": "mutopia", "handwriting": "handwritten"}
TEST_ONLY_ROWS = {"Night Album": "night-album"}
RESAMPLES = 2000


def piece_counts(path: Path) -> dict[str, np.ndarray]:
    """[correct, predicted, ground truth] notes of each piece."""
    pieces: dict[str, np.ndarray] = {}
    for item in json.loads(path.read_text()).values():
        t = item["tally"]
        counts = np.array([t.get("correct", 0), t.get("pred", 0), gt_notes(t)], dtype=np.int64)
        pieces[item["piece"]] = pieces.get(item["piece"], 0) + counts
    return pieces


def f1(counts: np.ndarray) -> np.ndarray:
    """F1 of [..., (correct, predicted, ground truth)] counts."""
    return 2 * counts[..., 0] / np.maximum(1, counts[..., 1] + counts[..., 2])


def resampled_weights(pieces: int) -> np.ndarray:
    """How often each piece is drawn in each resample, the same for every row of a size."""
    return np.random.default_rng(0).multinomial(pieces, np.full(pieces, 1 / pieces), size=RESAMPLES)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def row_text(name: str, run: Path, baseline: Path | None) -> str:
    if not run.exists():
        return f"{name:26s} not run"
    pieces = piece_counts(run)
    table = np.stack(list(pieces.values()))
    total = table.sum(0)
    text = f"{name:26s} {len(pieces):6d} {total[2]:6d}  {f1(total):.3f}"
    if len(pieces) > 1:
        low, high = interval(f1(resampled_weights(len(pieces)) @ table))
        text += f" [{low:.3f}, {high:.3f}]"
    else:
        text += " " * 15
    if baseline is None or not baseline.exists():
        return text
    theirs = piece_counts(baseline)
    shared = sorted(pieces.keys() & theirs.keys())
    ours, other = (np.stack([p[k] for k in shared]) for p in (pieces, theirs))
    change = f1(ours.sum(0)) - f1(other.sum(0))
    text += f"  {change:+.3f}"
    if len(shared) > 1:
        weights = resampled_weights(len(shared))
        low, high = interval(f1(weights @ ours) - f1(weights @ other))
        text += f" [{low:+.3f}, {high:+.3f}]"
    if len(shared) < len(pieces):
        text += f" (on {len(shared)} shared pieces)"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("baseline", type=Path, nargs="?")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    suffix = "-test" if args.split == "test" else ""
    rows = {name: directory + suffix for name, directory in ROWS.items()}
    if args.split == "test":
        rows |= TEST_ONLY_ROWS
    print(f"{'':26s} {'pieces':>6s} {'notes':>6s}  F1    95% range" + ("        change, 95% range" if args.baseline else ""))
    for name, directory in rows.items():
        print(row_text(name, args.run / directory / "results.json",
                       args.baseline / directory / "results.json" if args.baseline else None))


if __name__ == "__main__":
    main()
