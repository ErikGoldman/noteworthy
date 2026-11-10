"""Contact sheets of training crops whose label holds a token, to check by eye that the image
draws what the label says. See docs/TRAINING_DATA_CHECKS.md for what to look at and why.

usage (from the repo root):
  python train/crop_sheets.py CROP_DIR TOKEN [TOKEN...] [--count N] [--tail] [--out DIR]

CROP_DIR is a crop set (e.g. data/seq514_data/seq2/musescore). A crop is chosen when its label
holds any of the tokens; a token ending in `*` matches by prefix (`tremolo*`). --tail shows only
the right third of each crop, for marks at a system's end. One PNG per call, in --out.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROW_HEIGHT, SHEET_WIDTH = 300, 1800


def matches(tokens: list[str], wanted: list[str]) -> bool:
    return any(t.startswith(w[:-1]) if w.endswith("*") else t == w for t in tokens for w in wanted)


def chosen_crops(crops: Path, wanted: list[str], count: int) -> list[str]:
    names = (crops / "samples.train.txt").read_text().split()
    random.Random(" ".join(wanted)).shuffle(names)
    chosen = []
    for name in names:
        label = crops / f"{name}.lmx"
        if label.exists() and matches(label.read_text().split(), wanted):
            chosen.append(name)
            if len(chosen) == count:
                break
    return chosen


def sheet_row(image: np.ndarray, name: str, tail: bool) -> np.ndarray:
    if tail:
        image = image[:, int(image.shape[1] * 0.65):]
    width = min(SHEET_WIDTH, int(image.shape[1] * ROW_HEIGHT / image.shape[0]))
    row = cv2.resize(image, (width, ROW_HEIGHT), interpolation=cv2.INTER_AREA)
    row = np.pad(row, ((0, 0), (0, SHEET_WIDTH - width)), constant_values=255)
    cv2.putText(row, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    return np.pad(row, ((0, 8), (0, 0)), constant_values=128)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crops", type=Path)
    ap.add_argument("tokens", nargs="+")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("crop_sheets"))
    args = ap.parse_args()
    names = chosen_crops(args.crops, args.tokens, args.count)
    if not names:
        raise SystemExit("no crop holds those tokens")
    rows = [sheet_row(cv2.imread(str(args.crops / f"{n}.png"), cv2.IMREAD_GRAYSCALE), n, args.tail) for n in names]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.crops.name}_{'_'.join(t.strip('*').replace(':', '-') for t in args.tokens)}.png"
    cv2.imwrite(str(path), np.vstack(rows))
    print(path, *names)


if __name__ == "__main__":
    main()
