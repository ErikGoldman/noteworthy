"""Benchmark items that share a piece with training data: for each page (or song, a directory of
system files), the largest share of its melody fragments (select_mutopia.melody_grams) found in
any one training unit. At COPY_SHARE or more it counts as a copy (docs/EVALUATION_PROTOCOL.md,
Overlap). Counts only: nothing of the benchmark is written but names and shares.

usage: benchmark_overlap.py [--pages NAME=GLOB ...] [--songs NAME=GLOB ...] [--training LIST]
                            [--workers W] [--out SHARES.tsv]

GLOB finds the benchmark's MusicXML (relative to the working directory), or @FILE lists it. With --pages an item is a
file; with --songs it is each file's directory (an OLiMPiC song, a directory of systems). Training is every unit as trained on
(held_out_copies: the render list less its held-out scores, GrandStaff's movements, OLiMPiC's
songs, MusiCorpus's manuscripts); --training LIST instead takes one MusicXML a unit, for a source
not yet trained on.

Fragments are taken per file and pooled per unit: a GrandStaff crop or an OLiMPiC system numbers
its bars from 0, so pooling their events first stacks them and finds nothing (EXPERIMENTS.md,
"SMB", 2026-11-19). A fragment across two files is left out.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

from held_out_copies import DATA, Unit, grandstaff_units, musicorpus_units, olimpic_units, render_list_units
from score_events import parse_musicxml
from select_mutopia import COPY_SHARE, melody_grams

Grams = set[tuple[int, ...]]

_index: dict[tuple[int, ...], list[str]] = {}  # fragment: benchmark items holding it


def file_grams(path: Path) -> Grams:
    return melody_grams(parse_musicxml(path))


def unit_grams(files: list[Path]) -> Grams:
    return set().union(*(file_grams(f) for f in files))


def one_crop_per_bar_range(unit: Unit) -> Unit:
    """GrandStaff renders each bar range in several keys; the fragments are intervals, so one crop of
    each range (in whichever key comes first: some ranges have no untransposed crop) holds them all."""
    source, name, files = unit
    by_range: dict[str, Path] = {}
    for f in sorted(files):
        by_range.setdefault(f.stem[f.stem.index("m-"):], f)
    return source, name, sorted(by_range.values())


def trained_units() -> list[Unit]:
    scores = [Path(line) for line in (DATA / "render_scores.txt").read_text().split()]
    return (render_list_units(scores) + [one_crop_per_bar_range(u) for u in grandstaff_units()]
            + olimpic_units() + musicorpus_units())


def listed_units(training_list: Path) -> list[Unit]:
    return [("listed", line, [Path(line)]) for line in training_list.read_text().split()]


def set_index(index: dict[tuple[int, ...], list[str]]) -> None:
    global _index
    _index = index


def shared_counts(unit: Unit) -> tuple[Unit, Counter | None]:
    """How many of each item's fragments this unit holds; None if the unit will not parse."""
    try:
        grams = unit_grams(unit[2])
    except Exception:  # noqa: BLE001 - counted and listed by the caller
        return unit, None
    return unit, Counter(item for fragment in grams & _index.keys() for item in _index[fragment])


def largest_shares(items: dict[str, Grams], units: list[Unit], workers: int) -> tuple[dict[str, tuple[float, str]], list[Unit]]:
    """For each item, its largest share in one unit and that unit ("source:name"); the units that would not parse."""
    index = defaultdict(list)
    for item, fragments in items.items():
        for fragment in fragments:
            index[fragment].append(item)
    best = {item: (0.0, "") for item in items}
    unparsed = []
    with mp.Pool(workers, set_index, (dict(index),)) as pool:
        for done, (unit, counts) in enumerate(pool.imap_unordered(shared_counts, units, chunksize=16), 1):
            if counts is None:
                unparsed.append(unit)
                continue
            for item, n in counts.items():
                best[item] = max(best[item], (n / len(items[item]), f"{unit[0]}:{unit[1]}"))
            if done % 5000 == 0:
                print(f"{done}/{len(units)} training units", flush=True)
    print(f"training units {dict(Counter(u[0] for u in units))}; unparsed {dict(Counter(u[0] for u in unparsed))}", flush=True)
    return best, unparsed


def benchmark_items(pattern: str, by_directory: bool) -> tuple[dict[str, Grams], list[str]]:
    """(item: fragments, files that would not parse); an item with an unparsed file is listed, not hidden."""
    items: dict[str, Grams] = defaultdict(set)
    unparsed = []
    paths = Path(pattern[1:]).read_text().split() if pattern.startswith("@") else glob(pattern, recursive=True)
    for path in sorted(paths):
        item = str(Path(path).parent) if by_directory else path
        try:
            items[item] |= file_grams(Path(path))
        except Exception:  # noqa: BLE001 - listed by the caller
            unparsed.append(path)
    return dict(items), unparsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", action="append", default=[])
    ap.add_argument("--songs", action="append", default=[])
    ap.add_argument("--training", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    benchmarks = {}
    for spec, by_directory in [(s, False) for s in args.pages] + [(s, True) for s in args.songs]:
        name, pattern = spec.split("=", 1)
        items, unparsed = benchmark_items(pattern, by_directory)
        print(f"{name}: {len(items)} items, {len(unparsed)} files unparsed: {' '.join(unparsed) or '-'}", flush=True)
        benchmarks[name] = items
    every_item = {(name, item): grams for name, items in benchmarks.items() for item, grams in items.items() if grams}
    keyed = {f"{name}\t{item}": grams for (name, item), grams in every_item.items()}
    units = listed_units(args.training) if args.training else trained_units()
    best, _ = largest_shares(keyed, units, args.workers)
    lines = ["benchmark\titem\tfragments\tlargest_share\tnearest"]
    for name, items in benchmarks.items():
        shares = {item: best.get(f"{name}\t{item}", (0.0, "")) for item in items}
        copies = sorted(((s, n, item) for item, (s, n) in shares.items() if s >= COPY_SHARE), reverse=True)
        print(f"{name}: {len(items)} items, {len(copies)} at or over {COPY_SHARE} "
              f"(largest share {max((s for s, _ in shares.values()), default=0):.3f}); no fragments: "
              f"{sum(not g for g in items.values())}")
        for share, nearest, item in copies:
            print(f"  {share:.3f}  {Path(item).name}  {nearest}")
        lines += [f"{name}\t{item}\t{len(items[item])}\t{s:.3f}\t{n}" for item, (s, n) in sorted(shares.items())]
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
