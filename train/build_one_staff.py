"""Takes a set of two-staff system crops apart into one crop per staff (seq.one_staff), for
training and scoring the models on one staff at a time (EXPERIMENTS.md, (k)).

usage: build_one_staff.py ROOT SPLIT POSITIONS_DIR OUT_ROOT OUT_POSITIONS_DIR [--names LIST] [--workers N]

ROOT is a directory of <sample>.png|.jpg + <sample>.lmx with samples.<SPLIT>.txt, its staff
positions in POSITIONS_DIR/<ROOT's name>.json (find_staff_positions.py). Writes, for each system
that splits, OUT_ROOT/<sample>-staff1.png (the upper staff) and -staff2.png with their .lmx, and
OUT_ROOT/samples.<SPLIT>.txt; their positions in OUT_POSITIONS_DIR/<OUT_ROOT's name>.json, in the
same form; and OUT_ROOT/outcomes.json, how many systems split and why the others did not.
--names takes these systems of the split only (OLiMPiC dev's clean list). Images are written
grayscale, as the models read them.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import cv2

from seq.data import Sample, samples, split_names
from seq.one_staff import Unsplittable, staff_images, staff_labels
from seq.staff_positions import Staves
from seq.staff_data import load_positions

STAFF_NAMES = ("staff1", "staff2")


def split_system(job: tuple[Sample, Staves | None, Path, str]) -> tuple[list[tuple[str, list[float]]], str | None]:
    """The system's staves written under OUT_ROOT as (name, position), or none and why."""
    sample, staves, out_root, name = job
    if staves is None:
        return [], "no staff positions"
    try:
        labels = staff_labels(sample.lmx.read_text())
        crops = staff_images(cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE), staves)
    except Unsplittable as reason:
        return [], str(reason)
    written = []
    for staff_name, label, (image, position) in zip(STAFF_NAMES, labels, crops):
        staff = f"{name}-{staff_name}"
        (out_root / staff).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / f"{staff}.png"), image)
        (out_root / f"{staff}.lmx").write_text(label)
        written.append((staff, [round(position.centres[0], 2), round(position.interline, 3)]))
    return written, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("split")
    ap.add_argument("positions", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("out_positions", type=Path)
    ap.add_argument("--names", type=Path, help="split only these systems of SPLIT")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()
    names = args.names.read_text().split() if args.names else split_names(args.root, args.split)
    found = load_positions(args.positions, args.root)
    jobs = [(sample, found.get(name), args.out_root, name) for sample, name in zip(samples(args.root, names), names)]
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_positions.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(split_system, jobs, chunksize=64)
    staves = {staff: position for written, _ in results for staff, position in written}
    left_out = Counter(reason for _, reason in results if reason)
    (args.out_root / f"samples.{args.split}.txt").write_text("".join(f"{s}\n" for s in staves))
    (args.out_positions / f"{args.out_root.name}.json").write_text(json.dumps({"split": args.split, "staves": staves}))
    outcomes = {"systems": len(jobs), "split": len(jobs) - sum(left_out.values()), "left out": dict(left_out.most_common())}
    (args.out_root / "outcomes.json").write_text(json.dumps(outcomes, indent=1) + "\n")
    print(f"{args.root.name}: {outcomes['split']} of {len(jobs)} systems split; left out {dict(left_out.most_common())}",
          flush=True)


if __name__ == "__main__":
    main()
