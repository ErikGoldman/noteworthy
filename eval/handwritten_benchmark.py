"""Handwritten benchmark: MUSCIMA++ pages recognized whole and scored by note F1.

usage: handwritten_benchmark.py --binary NOTEWORTHY --model MODEL.onnx --out DIR
                                [--split dev|test] [--jobs N] [--threads T]
       handwritten_benchmark.py --predictions DIR [--split dev|test]
       handwritten_benchmark.py scans --out DIR [--split dev|test]

With --predictions, another recognizer's DIR/<page>.musicxml is scored instead (pages
from the scans mode: DIR/<page>.png, black ink on white).

Ground truth comes from the pages' MuNG annotations (muscima_truth.py); measures whose
rhythm cannot be read from the graph take part in measure alignment but are not scored.
dev is MUSCIMA++'s writer-dependent test set, test its writer-independent one, held out
like Night Album (look at its number, never at its errors); omr.muscima keeps both out of
training. Every writer copied the same 20 pieces, so the pieces themselves are always in
training in other hands.

Data (scripts/fetch_training_data.sh): $NW_DATA/raw/muscima.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from muscima_truth import page_truth
from note_scoring import report, score
from results_file import save_results
from score_events import parse_musicxml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from omr.muscima import Page, dev_pages, test_pages  # noqa: E402

MUSCIMA = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "muscima"


def write_scan(page: Page, path: Path) -> None:
    """The page as black ink on white, as the recognizer expects (CVC-MUSCIMA is inverted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), 255 - cv2.imread(str(page.image), cv2.IMREAD_GRAYSCALE))


def piece_of(page_name: str) -> str:
    """Page names are w<writer>_p<piece>, and every writer copied the same pieces."""
    return page_name.split("_")[1]


def recognize(page: Page, args: argparse.Namespace) -> Path | str:
    """The recognizer's MusicXML for the page, or its error."""
    scan = args.out / f"{page.name}.png"
    prediction = scan.with_suffix(".musicxml")
    write_scan(page, scan)
    run = subprocess.run([args.binary, str(scan), "--model", args.model, "-o", str(prediction), "--threads", str(args.threads)],
                         capture_output=True, text=True)
    return prediction if run.returncode == 0 else run.stderr.strip()[-200:]


def bench(page: Page, args: argparse.Namespace) -> Counter | str:
    prediction = args.predictions / f"{page.name}.musicxml" if args.predictions else recognize(page, args)
    if isinstance(prediction, str):
        return prediction
    if not prediction.exists():
        return f"no prediction {prediction}"
    truth = page_truth(page.annotation)
    unscored = frozenset(k for k, m in enumerate(truth) if not m.scored)
    return score([e for m in truth for e in m.events], parse_musicxml(prediction), unscored)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["recognize", "scans"], default="recognize")
    ap.add_argument("--binary")
    ap.add_argument("--model")
    ap.add_argument("--predictions", type=Path, help="score these instead of recognizing")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if (args.mode == "scans" or not args.predictions) and not args.out:
        ap.error("--out is needed to write scans")
    if args.mode == "recognize" and not args.predictions and not (args.binary and args.model):
        ap.error("--binary and --model are needed unless --predictions is given")
    pages = (dev_pages if args.split == "dev" else test_pages)(MUSCIMA)
    if args.mode == "scans":
        for page in pages:
            write_scan(page, args.out / f"{page.name}.png")
        return
    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for page, result in zip(pages, pool.map(lambda p: bench(p, args), pages)):
            if isinstance(result, str):
                print(f"{page.name}: failed: {result}")
            else:
                results[page.name] = result
    save_results(results, (args.predictions or args.out) / "results.json", piece_of)
    report(results, "pages", show_worst=args.split != "test")


if __name__ == "__main__":
    main()
