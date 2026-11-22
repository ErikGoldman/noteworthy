"""The scores of a render list whose piano part holds a tuplet, in list order: the hidden-tuplet sets
are drawn from these alone, since a score without one adds no crop that teaches hiding.

usage: scores_with_tuplets.py LIST OUT [--workers N]
A score that cannot be read is left out and counted on stderr."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

from build_seq_dataset import load_musicxml, piano_part


def holds_a_tuplet(source: str) -> bool | None:
    """Whether the score's piano part holds a tuplet; None if it cannot be read."""
    try:
        part = piano_part(load_musicxml(Path(source)))
    except Exception:  # noqa: BLE001 - any unreadable file is counted, not fatal
        return None
    return part is not None and part.find(".//notations/tuplet") is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("list", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    scores = args.list.read_text().split()
    with mp.Pool(args.workers) as pool:
        found = pool.map(holds_a_tuplet, scores, chunksize=64)
    kept = [s for s, holds in zip(scores, found) if holds]
    args.out.write_text("".join(f"{s}\n" for s in kept))
    print(f"{len(kept)} of {len(scores)} scores hold a tuplet; {found.count(None)} unreadable, left out", file=sys.stderr)


if __name__ == "__main__":
    main()
