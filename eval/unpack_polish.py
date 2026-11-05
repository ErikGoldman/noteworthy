"""Unpacks Transcoda's Polish benchmark for omr_ned.py.

usage: unpack_polish.py PARQUET OUT

Writes OUT/pages/<n>.png, OUT/gold/<n>.krn and .musicxml as published, and
OUT/gold_terminated/<n>.krn, the same with a **kern header and one terminator per open spine.
The published text has neither, and the authors' metric adds them from the first line; a page
that ends with another spine count then fails the strict parse of the gold and comes out
empty, so it scores 100 against anything, itself included (34 of 112 pages).

Runs in .venv-musicdiff. Evaluation only: the dataset's card forbids training on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq


def spines_after(fields: list[str]) -> int:
    count, i = 0, 0
    while i < len(fields):
        if fields[i] == "*v":
            while i < len(fields) and fields[i] == "*v":
                i += 1
            count += 1
        else:
            count += 2 if fields[i] == "*^" else 1
            i += 1
    return count


def terminated(kern: str) -> str:
    lines = [line for line in kern.strip().split("\n") if line]
    first = len(lines[0].split("\t"))
    spines = first
    for line in lines:
        if line.startswith("*"):
            spines = spines_after(line.split("\t"))
    return "\n".join(["\t".join(["**kern"] * first), *lines, "\t".join(["*-"] * spines)]) + "\n"


def main() -> None:
    parquet, out = Path(sys.argv[1]), Path(sys.argv[2])
    for name in ("pages", "gold", "gold_terminated"):
        (out / name).mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(pq.read_table(parquet).to_pylist()):
        name = f"{i:03d}"
        (out / "pages" / f"{name}.png").write_bytes(row["image"]["bytes"])
        (out / "gold" / f"{name}.krn").write_text(row["transcription_kern"])
        (out / "gold" / f"{name}.musicxml").write_text(row["transcription_xml"])
        (out / "gold_terminated" / f"{name}.krn").write_text(terminated(row["transcription_kern"]))


if __name__ == "__main__":
    main()
