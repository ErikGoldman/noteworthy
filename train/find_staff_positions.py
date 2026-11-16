"""Each system crop's staff positions, for StaffSeq2Seq (seq.staff_data): as drawn, where the
crop's builder recorded them (<sample>.staves.json: build_seq_dataset.py, build_musescore_crops.py,
build_smashcima_crops.py), else found once by seq.staves.

usage: find_staff_positions.py OUT_DIR ROOT:SPLIT [ROOT:SPLIT ...] [--workers N]

For each ROOT (a directory of <sample>.png|.jpg + <sample>.lmx with samples.<SPLIT>.txt), writes
OUT_DIR/<ROOT's name>.json: {"split": SPLIT, "staves": {sample: [centre, ..., interline] or null}},
centres in the image's own rows, top to bottom, as many as the sample's LMX has staves; null
where they are not found. Found ones: on 300 crops, by the segmenter's staves (the recognizer's
layout, which inference uses) and by eye where the two disagree, about 9 in 10 of the crops the
finder once missed are right, and 97 in 100 of the rest (EXPERIMENTS.md).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2

from seq.data import samples, split_names
from seq.staff_positions import staff_count
from seq.staves import system_staves


def positions_of(sample) -> list[float] | None:
    drawn = sample.image.with_suffix(".staves.json")
    if drawn.exists():
        return json.loads(drawn.read_text())
    image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
    staves = system_staves(image, staff_count(sample.lmx.read_text()))
    return None if staves is None else [round(c, 2) for c in staves.centres] + [round(staves.interline, 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool:
        for spec in args.sets:
            root, split = spec.rsplit(":", 1)
            root = Path(root)
            names = split_names(root, split)
            found = pool.map(positions_of, samples(root, names), chunksize=64)
            staves = dict(zip(names, found))
            (args.out / f"{root.name}.json").write_text(json.dumps({"split": split, "staves": staves}))
            print(f"{root.name}: {sum(f is not None for f in found)} of {len(names)} split", flush=True)


if __name__ == "__main__":
    main()
