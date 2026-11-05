"""Benchmark on piano music outside OLiMPiC's one genre: Mutopia pieces, engraved by
LilyPond, whole PDFs recognized as one score and scored by note F1.

usage: mutopia_benchmark.py --binary NOTEWORTHY --model MODEL.onnx --out DIR
                            [--split dev|test] [--jobs N] [--threads T]
       mutopia_benchmark.py --predictions DIR [--split dev|test]

With --predictions, another recognizer's DIR/<name>.musicxml is scored instead.

The pieces are listed in mutopia_pieces.txt (select_mutopia.py), with ground truth from
their MIDI (mutopia_truth.py). They are clean digital engravings: this measures variety
of repertoire and engraving, not scan quality. Pieces with grace notes, free time or
played-out repeats have no reliable MIDI truth and are left out, so those are not tested
here. test is held out like Night Album (look at its number, never at its errors).

Data (scripts/fetch_data.sh): $NW_DATA/raw/mutopia.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mutopia_truth import midi_events, pickup_quarters
from note_scoring import report, score
from results_file import save_results
from score_events import parse_musicxml

ROOT = Path(__file__).resolve().parent.parent
MUTOPIA = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "mutopia"
PIECES = Path(__file__).with_name("mutopia_pieces.txt")


def pieces(split: str) -> list[str]:
    """Paths under Mutopia's ftp/ of the split's pieces."""
    rows = [line.split() for line in PIECES.read_text().splitlines()]
    return [path for row_split, path, *_ in rows if row_split == split]


def recognize(name: str, args: argparse.Namespace) -> Path | str:
    """The recognizer's MusicXML for the piece, or its error."""
    prediction = args.out / f"{name}.musicxml"
    run = subprocess.run([args.binary, str(MUTOPIA / "pieces" / f"{name}-a4.pdf"), "--model", args.model, "-o", str(prediction),
                          "--threads", str(args.threads)], capture_output=True, text=True)
    return prediction if run.returncode == 0 else run.stderr.strip()[-200:]


def bench(piece: str, args: argparse.Namespace) -> Counter | str:
    name = Path(piece).name
    prediction = args.predictions / f"{name}.musicxml" if args.predictions else recognize(name, args)
    if isinstance(prediction, str):
        return prediction
    if not prediction.exists():
        return f"no prediction {prediction}"
    source = (MUTOPIA / "repo" / "ftp" / piece / f"{name}.ly").read_text(errors="replace")
    return score(midi_events(MUTOPIA / "pieces" / f"{name}.mid", pickup_quarters(source)), parse_musicxml(prediction))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary")
    ap.add_argument("--model")
    ap.add_argument("--predictions", type=Path, help="score these instead of recognizing")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if not args.predictions and not (args.binary and args.model and args.out):
        ap.error("--binary, --model and --out are needed unless --predictions is given")
    results_dir = args.predictions or args.out
    results_dir.mkdir(parents=True, exist_ok=True)
    chosen = pieces(args.split)
    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for piece, result in zip(chosen, pool.map(lambda p: bench(p, args), chosen)):
            if isinstance(result, str):
                print(f"{piece}: failed: {result}")
            else:
                results[piece] = result
    save_results(results, results_dir / "results.json", lambda piece: piece)
    report(results, "pieces", show_worst=args.split != "test")


if __name__ == "__main__":
    main()
