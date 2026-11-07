"""Applies reviewed corrections to Polish gold, leaving the published files untouched.

usage: gold_overrides.py GOLD_DIR OVERRIDES.jsonl OUT_DIR [--standard corrected|print]

GOLD_DIR is unpack_polish.py's output; OUT_DIR receives every page, changed or not, so a scorer
can take it in GOLD_DIR's place. Each override says what a reviewer saw on the printed page,
whichever system that helps. See docs/EVALUATION_PROTOCOL.md for the two standards:
  corrected  errors under any convention; gold's own conventions stay
  print      corrected, and notes on the staff they are drawn on, not the hand that plays them

One override per line of OVERRIDES.jsonl:
  page, line   the gold line it applies to, numbered from 1 in the unchanged page
  source       sha256 of the unchanged page; a mismatch stops the run, since the line numbers
               would then point at something nobody reviewed
  standard     corrected or print
  reason       what is wrong, as seen on the print
and one of:
  insert_after lines to add after `line` (tabs between columns)
  move_up      {"last_line": n, "lowest": midi}: from `line` to `last_line`, notes of the left
               spine from that pitch up go to a new voice on the upper staff
Gold's own text is not repeated here: the dataset's licence is a placeholder, not a grant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SEMITONE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
NOTE = re.compile(r"([a-g]+|[A-G]+)(#+|-+|n)?")
RHYTHM = re.compile(r"\d+(?:%\d+)?\.*")
BEAM_MARKS = re.compile(r"[LJKk]")


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def midi(note: str) -> int | None:
    """Pitch of one kern note, None for a rest."""
    found = NOTE.search(note)
    if "r" in note or not found:
        return None
    letters, accidental = found[1], found[2] or ""
    octave = 3 + len(letters) if letters[0].islower() else 4 - len(letters)
    alteration = accidental.count("#") - accidental.count("-")
    return 12 * (octave + 1) + SEMITONE[letters[0].lower()] + alteration


def rest_like(field: str) -> str:
    return RHYTHM.search(field)[0] + "r"


def divided(field: str, lowest: int) -> tuple[str, str]:
    """A left-spine field as what stays and what moves up, each a rest if it would be empty.
    Beams go from both: the print's beam joins the staves, and no kern spine can hold that."""
    if field == ".":
        return ".", "."
    notes = BEAM_MARKS.sub("", field).split()
    moving = [note for note in notes if (midi(note) or 0) >= lowest]
    staying = [note for note in notes if note not in moving]
    return " ".join(staying) or rest_like(field), " ".join(moving) or rest_like(field)


def moved_up(lines: list[str], lowest: int) -> list[str]:
    """The lines with their high left-spine notes in a new last column, opened and closed around them."""
    width = len(lines[0].split("\t"))
    keep = ["*"] * (width - 1)
    out = ["\t".join([*keep, "*^"])]
    for line in lines:
        fields = line.split("\t")
        if line.startswith("="):
            out.append("\t".join([*fields, fields[-1]]))
        elif line.startswith("*"):
            out.append("\t".join([*fields, "*"]))
        else:
            left, upper = divided(fields[0], lowest)
            out.append("\t".join([left, *fields[1:], upper]))
    out.append("\t".join([*keep, "*v", "*v"]))
    return out


def after_spine_operations(fields: list[str], columns: list[tuple]) -> list[tuple]:
    """Column states once *^ and *v have applied: a split's new column starts empty, and a merge
    keeps the state with more beams open."""
    out, i = [], 0
    while i < len(fields):
        if fields[i] == "*^":
            out += [columns[i], (0, None)]
            i += 1
        elif fields[i] == "*v":
            merged = next((j for j in range(i, len(fields)) if fields[j] != "*v"), len(fields))
            out.append(max(columns[i:merged], key=lambda state: state[0]))
            i = merged
        else:
            out.append(columns[i])
            i += 1
    return out


def beamable(field: str) -> bool:
    """Shorter than a quarter note, tuplets included."""
    return int(re.match(r"\D*(\d+)", field)[1]) > 4


def beams_closed(text: str) -> str:
    """Closes each beam left open on its column's last beamable note. Gold has groups whose L
    has no J, and the scorer's parser then drops the group's beams, leaving flags. A beam
    ends at a barline or at a note too long for one, so this needs no review of the print."""
    lines = text.split("\n")
    columns: list[tuple[int, tuple[int, int] | None]] = []  # beams open, (line, column) of the last note in them

    def close(state: tuple[int, tuple[int, int] | None]) -> None:
        depth, last = state
        if depth and last:
            row = lines[last[0]].split("\t")
            row[last[1]] += "J" * depth
            lines[last[0]] = "\t".join(row)

    for at, line in enumerate(lines):
        fields = line.split("\t")
        if line.startswith("**"):
            columns = [(0, None)] * len(fields)
        elif line.startswith("="):
            for state in columns:
                close(state)
            columns = [(0, None)] * len(fields)
        elif line.startswith("*"):
            if "*^" in fields or "*v" in fields:
                columns = after_spine_operations(fields, columns)
        elif line and not line.startswith("!"):
            for i, field in enumerate(fields):
                if field == "." or midi(field.split()[0]) is None:
                    continue
                if not beamable(field):
                    close(columns[i])
                    columns[i] = 0, None
                    continue
                opened = max(note.count("L") for note in field.split())
                ended = max(note.count("J") for note in field.split())
                columns[i] = max(0, columns[i][0] + opened - ended), (at, i)
    return "\n".join(lines)


def corrected(text: str, overrides: list[dict]) -> str:
    lines = text.split("\n")
    # From the bottom up, so that earlier line numbers still mean the unchanged page.
    for override in sorted(overrides, key=lambda o: o["line"], reverse=True):
        if "insert_after" in override:
            lines[override["line"]:override["line"]] = override["insert_after"]
        else:
            first, last = override["line"] - 1, override["move_up"]["last_line"]
            lines[first:last] = moved_up(lines[first:last], override["move_up"]["lowest"])
    return "\n".join(lines)


def load_overrides(path: Path, standards: set[str]) -> dict[str, list[dict]]:
    by_page = defaultdict(list)
    for line in path.read_text().splitlines():
        override = json.loads(line)
        if override["standard"] in standards:
            by_page[override["page"]].append(override)
    return by_page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path)
    ap.add_argument("overrides", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--standard", choices=["corrected", "print"], default="corrected")
    args = ap.parse_args()
    overrides = load_overrides(args.overrides, {"corrected", args.standard})
    args.out.mkdir(parents=True, exist_ok=True)
    for page in sorted(args.gold.glob("*.krn")):
        text = page.read_text()
        for override in overrides.get(page.stem, []):
            if override["source"] != source_hash(text):
                sys.exit(f"{page.stem}: gold changed since line {override['line']} was reviewed")
        (args.out / page.name).write_text(beams_closed(corrected(text, overrides.get(page.stem, []))))
    print(f"{sum(map(len, overrides.values()))} overrides on {len(overrides)} pages, standard {args.standard}")


if __name__ == "__main__":
    main()
