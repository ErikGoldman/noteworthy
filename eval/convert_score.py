"""Converts a directory of scores between **kern and MusicXML with converter21.

usage: convert_score.py SOURCE OUT --to kern|musicxml

Runs in .venv-musicdiff. OMR-NED is only comparable within one format: the same music as
MusicXML and as **kern differs by several percent, so two systems are scored after
conversion to the same form. A file that cannot be converted is left out, which scores 100.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import converter21
import music21

SUFFIX = {"kern": ".krn", "musicxml": ".musicxml"}
WRITER = {"kern": "humdrum", "musicxml": "musicxml"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--to", choices=SUFFIX, required=True)
    args = ap.parse_args()
    converter21.register()
    args.out.mkdir(parents=True, exist_ok=True)
    source_suffix = SUFFIX["musicxml" if args.to == "kern" else "kern"]
    failed = 0
    paths = sorted(args.source.glob("*" + source_suffix))
    for path in paths:
        try:
            score = music21.converter.parse(str(path), forceSource=True, acceptSyntaxErrors=True) if args.to == "musicxml" \
                else music21.converter.parse(str(path), forceSource=True)
            score.write(WRITER[args.to], fp=args.out / (path.stem + SUFFIX[args.to]))
        except Exception as e:  # converters raise many kinds on damaged scores
            failed += 1
            print(f"{path.name}: {type(e).__name__}: {str(e)[:100]}")
    print(f"{len(paths) - failed} of {len(paths)} converted")


if __name__ == "__main__":
    main()
