"""Benchmark pages that share a piece with training data: for each page (or song, a directory of
system files), the largest share of its melody fragments (select_mutopia.melody_grams) found in
any one training piece. At COPY_SHARE or more it counts as a copy, and is reported apart
(docs/EVALUATION_PROTOCOL.md, Overlap). Counts only: nothing of the benchmark is printed but
names and shares.

usage: benchmark_overlap.py --training LIST --benchmark NAME=GLOB [--benchmark NAME=GLOB ...]
                            [--by-directory] [--out RESULTS.json]

LIST names one training MusicXML per line; a piece is a file. GLOB finds the benchmark's MusicXML
(relative to the working directory); with --by-directory a benchmark item is each file's
directory (an OLiMPiC song).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

from score_events import parse_musicxml
from select_mutopia import COPY_SHARE, melody_grams


def grams_of(path: str) -> set:
    try:
        return melody_grams(parse_musicxml(Path(path)))
    except Exception:  # noqa: BLE001 - an unreadable file shares nothing
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", type=Path, required=True)
    ap.add_argument("--benchmark", action="append", required=True)
    ap.add_argument("--by-directory", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    training = [line for line in args.training.read_text().split("\n") if line.strip()]
    training_grams = {piece: grams_of(piece) for piece in training}
    results = {}
    for spec in args.benchmark:
        name, pattern = spec.split("=", 1)
        items: dict[str, set] = defaultdict(set)
        for path in sorted(glob(pattern, recursive=True)):
            items[str(Path(path).parent) if args.by_directory else path] |= grams_of(path)
        shares = {item: max((len(grams & t) / len(grams) for t in training_grams.values()), default=0.0) if grams else 0.0
                  for item, grams in items.items()}
        copies = sorted((share, item) for item, share in shares.items() if share >= COPY_SHARE)
        print(f"{name}: {len(shares)} items, {len(copies)} at or over {COPY_SHARE} "
              f"(largest share {max(shares.values(), default=0):.3f})")
        for share, item in copies:
            print(f"  {share:.3f}  {Path(item).name}")
        results[name] = shares
    print(f"training pieces {len(training)}, unreadable {sum(not g for g in training_grams.values())}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
