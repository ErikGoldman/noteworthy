"""Applies reviewed corrections to Polish gold, leaving the published files untouched.

usage: gold_overrides.py GOLD_DIR OVERRIDES.jsonl OUT_DIR [--standard corrected|print]

GOLD_DIR is unpack_polish.py's output; OUT_DIR receives every page, changed or not, so a scorer
can take it in GOLD_DIR's place. Each override says what a reviewer saw on the printed page,
whichever system that helps. See docs/EVALUATION_PROTOCOL.md for the two standards:
  corrected  errors under any convention; gold's own conventions stay
  print      corrected, and notes on the staff they are drawn on, not the hand that plays them;
             one slur per chord, as drawn

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
  move_down    {"last_line": n}: from `line` to `last_line`, notes of the right spine go to a new
               voice on the lower staff
  end_after    true: gold stops at `line`; what follows is not on this page
  staff_down   {"last_line": n, "lower_columns": k}: from `line` to `last_line`, with no spine
               changes between, every column of the upper staff (those after the first k) goes to a
               new voice on the lower staff, the upper staff left blank
  show_accidental {"column": c, "note": i}: the i-th note (from 0) of the field at `line`, column c,
               shows an accidental the key or bar already gives, as the print does (kern's `X`)
  hide_rests   {"last_line": n}: from `line` to `last_line`, rests become rests not drawn: gold
               pads its voices with rests the print does not have
  articulate   {"column": n, "mark": "'" or "^" or "`"}: adds a staccato, accent or wedge to the
               note at `line` in that column, counted from 0; with "ambiguous": true it is a mark
               the print leaves between dot and wedge, stored as a staccato and flagged for
               omr_ned.py --ambiguous-marks
  slur         {"column": n, "marks": "(" or ")" or ")("}: a slur the print draws starts or ends
               (or both) at the note at `line` in that column
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
# Humdrum's way to flag single notes; the scorer's parser turns it into the note's colour.
AMBIGUOUS_SIGNIFIER = "i"
AMBIGUOUS_COLOR = "#abcdef"
AMBIGUOUS_DECLARATION = f'!!!RDF**kern: {AMBIGUOUS_SIGNIFIER} = marked note, color="{AMBIGUOUS_COLOR}"'


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


def hidden_rest_like(field: str) -> str:
    """A rest of the field's length that is not drawn (kern's `yy`), which the scorer ignores. A
    move leaves gaps on both sides that the print fills with nothing: the vacated staff stays
    blank, and the moved notes are one voice beamed across (038, 062, 047, 067, 012, 058)."""
    return RHYTHM.search(field)[0] + "ryy"


def with_rests_hidden(line: str) -> str:
    if line.startswith(("=", "*", "!")):
        return line
    return "\t".join(re.sub(r"^(\d+(?:%\d+)?\.*r)(?!yy)", r"\1yy", field) for field in line.split("\t"))


def divided(field: str, lowest: int, keep_beams: bool) -> tuple[str, str]:
    """A spine field as what stays and what moves, each a hidden rest if it would be empty. A beam the
    move splits between the staves goes from both sides: no kern spine can hold it. A field
    that goes whole keeps it, when its group goes the same way (`keep_beams`)."""
    if field == ".":
        return ".", "."
    notes = BEAM_MARKS.sub("", field).split()
    moving = [note for note in notes if (midi(note) or 0) >= lowest]
    staying = [note for note in notes if note not in moving]
    if keep_beams and not staying:
        return hidden_rest_like(field), field
    if keep_beams and not moving:
        return field, hidden_rest_like(field)
    return " ".join(staying) or hidden_rest_like(field), " ".join(moving) or hidden_rest_like(field)


def goes_whole(field: str, lowest: int) -> str | None:
    """Which way all of a field's notes go, None if the move splits them."""
    notes = BEAM_MARKS.sub("", field).split()
    moving = sum((midi(note) or 0) >= lowest for note in notes)
    return "move" if moving == len(notes) else "stay" if moving == 0 else None


def beams_kept(fields: list[str], lowest: int) -> list[bool]:
    """For each field of one column (barlines and interpretations included, and ignored),
    whether its beam marks survive the move: those of a beam group whose fields all go the
    same way, whole."""
    kept = [True] * len(fields)
    group: list[int] = []
    depth = 0
    for i, field in enumerate(fields):
        if field == "." or field.startswith(("=", "*", "!")):
            continue
        opened, closed = field.count("L"), field.count("J")
        if depth or opened:
            group.append(i)
        depth = max(0, depth + opened - closed)
        if group and not depth:
            ways = {goes_whole(fields[j], lowest) for j in group}
            for j in group:
                kept[j] = len(ways) == 1 and None not in ways
            group = []
    for j in group:  # a group the range cuts off: its other end is not moved with it
        kept[j] = False
    return kept


def moved_up(lines: list[str], lowest: int) -> list[str]:
    """The lines with their high left-spine notes in a new last column, opened and closed around them."""
    width = len(lines[0].split("\t"))
    keep = ["*"] * (width - 1)
    out = ["\t".join([*keep, "*^"])]
    kept = beams_kept([line.split("\t")[0] for line in lines], lowest)
    for line, keep_beams in zip(lines, kept):
        fields = line.split("\t")
        if line.startswith("="):
            out.append("\t".join([*fields, fields[-1]]))
        elif line.startswith("*"):
            out.append("\t".join([*fields, "*"]))
        else:
            fields[0], upper = divided(fields[0], lowest, keep_beams)
            out.append("\t".join([*fields, upper]))
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


def beam_count(field: str) -> int:
    """How many beams a note of this length takes: 1 for an eighth, 2 for a 16th, tuplets by
    their written value (a triplet eighth, 12, has one)."""
    return max(0, int(re.match(r"\D*(\d+)", field)[1]).bit_length() - 3)


def beams_closed(text: str) -> str:
    """Closes each beam left open on its column's last beamable note. Gold has groups whose L
    has no J, and the scorer's parser then drops the group's beams, leaving flags. A beam
    ends at a barline, at a note too long for one, or where a note would carry more beams than
    its length allows, so this needs no review of the print."""
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
                if opened and columns[i][0] + opened > beam_count(field):
                    # More beams than the note has: gold ended the last group with too few J
                    # (`16J` for `16JJ`), and the next group's L nests in it (023's Trio).
                    close(columns[i])
                    columns[i] = 0, None
                columns[i] = max(0, columns[i][0] + opened - ended), (at, i)
    return "\n".join(lines)


def moved_down(lines: list[str]) -> list[str]:
    """The lines with the last column's notes in a new column before it, on the lower staff."""
    width = len(lines[0].split("\t"))
    keep = ["*"] * (width - 2)
    out = ["\t".join([*keep, "*^", "*"])]
    everything = 0  # every note moves
    kept = beams_kept([line.split("\t")[-1] for line in lines], everything)
    for line, keep_beams in zip(lines, kept):
        fields = line.split("\t")
        if line.startswith("="):
            out.append("\t".join([*fields, fields[-1]]))
        elif line.startswith("*"):
            out.append("\t".join([*fields, "*"]))
        else:
            staying, moving = divided(fields[-1], everything, keep_beams)
            out.append("\t".join([*fields[:-1], moving, staying]))
    out.append("\t".join([*keep, "*v", "*v", "*"]))
    return out


def staff_moved_down(lines: list[str], lower: int) -> list[str]:
    """The lines with all of the upper staff's columns as new voices of the lower staff, and the
    upper staff hidden rests. The lower staff's last column is split once per moved column, and
    they are merged back at the end."""
    width = len(lines[0].split("\t"))
    upper = width - lower
    out = []
    for added in range(upper):
        columns = lower + added
        out.append("\t".join(["*"] * (columns - 1) + ["*^"] + ["*"] * upper))
    for line in lines:
        fields = line.split("\t")
        if line.startswith("="):
            out.append("\t".join(fields[:lower] + fields[lower:] + fields[lower:]))
        elif line.startswith(("*", "!")):  # the upper staff's own: clefs must not reach the lower
            out.append("\t".join(fields[:lower] + [fields[0][0]] * upper + fields[lower:]))
        else:
            upper_fields = fields[lower:]
            out.append("\t".join(fields[:lower] + upper_fields + [hidden_rest_like(f) if f != "." else "." for f in upper_fields]))
    out.append("\t".join(["*"] * (lower - 1) + ["*v"] * (upper + 1) + ["*"] * upper))
    return out


def with_accidental_shown(note: str) -> str:
    """The note with `X` after its pitch and accidental, which makes the accidental drawn."""
    found = NOTE.search(note)
    return note[:found.end()] + "X" + note[found.end():]


def with_marks(line: str, column: int, marks: str) -> str:
    """The line with signifiers added to the field in that column: kern reads them anywhere in a
    token, and on a chord they land on its last note."""
    fields = line.split("\t")
    fields[column] += marks
    return "\t".join(fields)


def one_slur_per_chord(text: str) -> str:
    """Each chord's slur marks on its first note only. Gold often marks every note of a chord
    (`4E( 4G( 4e(`), which the scorer reads as one slur per note where the print draws one arc:
    465 slurs too many on the 49 two-staff dev pages."""
    def field_with_one_slur(field: str) -> str:
        notes = field.split(" ")
        for mark in "()":
            bearing = [i for i, note in enumerate(notes) if mark in note]
            for i in bearing[1:]:
                notes[i] = notes[i].replace(mark, "")
        return " ".join(notes)
    return "\n".join(line if line.startswith(("*", "!", "=")) else "\t".join(map(field_with_one_slur, line.split("\t")))
                     for line in text.split("\n"))


def corrected(text: str, overrides: list[dict]) -> str:
    lines = text.split("\n")
    # From the bottom up, so that earlier line numbers still mean the unchanged page. On one
    # line, articulations come first: a move opens its range by inserting a line there.
    for override in sorted(overrides, key=lambda o: (o["line"], "articulate" in o), reverse=True):
        first = override["line"] - 1
        if "articulate" in override:
            mark = override["articulate"]
            lines[first] = with_marks(lines[first], mark["column"], mark["mark"] + (AMBIGUOUS_SIGNIFIER if mark.get("ambiguous") else ""))
        elif "slur" in override:
            lines[first] = with_marks(lines[first], override["slur"]["column"], override["slur"]["marks"])
        elif "show_accidental" in override:
            shown = override["show_accidental"]
            fields = lines[first].split("\t")
            notes = fields[shown["column"]].split(" ")
            notes[shown["note"]] = with_accidental_shown(notes[shown["note"]])
            fields[shown["column"]] = " ".join(notes)
            lines[first] = "\t".join(fields)
        elif "insert_after" in override:
            lines[first + 1:first + 1] = override["insert_after"]
        elif "end_after" in override:
            lines[first + 1:] = ["\t".join(["*-"] * len(lines[first].split("\t"))), ""]
        elif "staff_down" in override:
            last = override["staff_down"]["last_line"]
            lines[first:last] = staff_moved_down(lines[first:last], override["staff_down"]["lower_columns"])
        elif "hide_rests" in override:
            last = override["hide_rests"]["last_line"]
            lines[first:last] = [with_rests_hidden(line) for line in lines[first:last]]
        elif "move_down" in override:
            last = override["move_down"]["last_line"]
            lines[first:last] = moved_down(lines[first:last])
        else:
            last = override["move_up"]["last_line"]
            lines[first:last] = moved_up(lines[first:last], override["move_up"]["lowest"])
    if any(o.get("articulate", {}).get("ambiguous") for o in overrides):
        lines = [*[line for line in lines if line], AMBIGUOUS_DECLARATION, ""]
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
        text = beams_closed(corrected(text, overrides.get(page.stem, [])))
        (args.out / page.name).write_text(one_slur_per_chord(text) if args.standard == "print" else text)
    print(f"{sum(map(len, overrides.values()))} overrides on {len(overrides)} pages, standard {args.standard}")


if __name__ == "__main__":
    main()
