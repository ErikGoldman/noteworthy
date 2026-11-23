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
  move_up      {"last_line": n, "notes": {"<gold line>": [i, ...]}, "column": c}: from `line` to
               `last_line`, the listed notes (by place in column c's field, from 0; c is the left
               spine if left out) are drawn on the upper staff and go to a new voice there, after
               the last column; a lower staff split into several spines names the one that holds them
  move_down    {"last_line": n, "notes": {"<gold line>": [i, ...]}, "column": c}: from `line` to
               `last_line`, the listed notes (every note if "notes" is left out) of column c (the
               last if left out; columns as on `line`) are drawn on the staff below its own and go
               to a new last voice of that staff; an upper staff already in two voices is fine
  end_after    true: gold stops at `line`; what follows is not on this page
  staff_down   {"last_line": n, "lower_columns": k}: from `line` to `last_line`, with no spine
               changes between, every column of the upper staff (those after the first k) goes to a
               new voice on the lower staff, the upper staff left blank
  slash_grace  {"column": c}: the grace notes of the field at `line`, column c, are slashed
  beams        {"column": c, "marks": m}: the field's beam marks become m, as the print groups the
               notes: one L (opens) or J (closes) per beam, K or k for a partial beam right or left,
               "" for none; kern reads a chord's on any note, so they go on its last
  show_accidental {"column": c, "note": i}: the i-th note (from 0) of the field at `line`, column c,
               shows an accidental the key or bar already gives, as the print does (kern's `X`;
               a natural on a note gold writes with none, kern's `n`). No op hides one: the
               scorer's parser ignores kern's `y`, and draws what the key and bar call for
  hide_rests   {"last_line": n}: from `line` to `last_line`, rests become rests not drawn: gold
               pads its voices with rests the print does not have
  hide_rest    {"column": c}: the rest of the field at `line`, column c, is not drawn: one voice's
               padding where the staff's other voice, or this one elsewhere, has its rests printed
  articulate   {"column": n, "mark": "'" or "^" or "`"}: adds a staccato, accent or wedge to the
               note at `line` in that column, counted from 0; with "ambiguous": true it is a mark
               the print leaves between dot and wedge, stored as a staccato and flagged for
               omr_ned.py --ambiguous-marks
  slur         {"column": n, "marks": "(" or ")" or ")("}: a slur the print draws starts or ends
               (or both) at the note at `line` in that column
  tuplet_numbers "hidden" or "shown": from `line` on, the print draws tuplet numbers or does not
               (kern's *Xtuplet and *tuplet before that line); gold says nothing of it, so the
               scorer otherwise takes every number as shown
  replace      {"last_line": n, "lines": [...]}: lines `line` to `last_line` become these, a
               passage transcribed afresh from the print where gold is wrong beyond a local fix
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
SLUR_MARKS = re.compile(r"[()]")
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


def rest_hidden(field: str) -> str:
    return re.sub(r"^(\d+(?:%\d+)?\.*r)(?!yy)", r"\1yy", field)


def with_rests_hidden(line: str) -> str:
    if line.startswith(("=", "*", "!")):
        return line
    return "\t".join(map(rest_hidden, line.split("\t")))


def divided(field: str, moves: set[int], keep_beams: bool) -> tuple[str, str]:
    """A spine field as what stays and what moves, each a hidden rest if it would be empty. A beam the
    move splits between the staves goes from both sides: no kern spine can hold it. A field
    that goes whole keeps it, when its group goes the same way (`keep_beams`). A chord's slur is
    one arc: when the chord splits, it stays with the notes that stay."""
    if field == ".":
        return ".", "."
    notes = BEAM_MARKS.sub("", field).split()
    moving = [note for i, note in enumerate(notes) if i in moves]
    staying = [note for i, note in enumerate(notes) if i not in moves]
    moving = without_slurs_staying_has(moving, staying)
    if keep_beams and not staying:
        return hidden_rest_like(field), field
    if keep_beams and not moving:
        return field, hidden_rest_like(field)
    return " ".join(staying) or hidden_rest_like(field), " ".join(moving) or hidden_rest_like(field)


def without_slurs_staying_has(moving: list[str], staying: list[str]) -> list[str]:
    """The moving notes without the slur marks a staying note of the same chord already carries."""
    marks = {mark for note in staying for mark in SLUR_MARKS.findall(note)}
    return [SLUR_MARKS.sub(lambda m: "" if m[0] in marks else m[0], note) for note in moving]


def every_note(field: str) -> set[int]:
    return set(range(len(BEAM_MARKS.sub("", field).split())))


def goes_whole(field: str, moves: set[int]) -> str | None:
    """Which way all of a field's notes go, None if the move splits them."""
    count = len(BEAM_MARKS.sub("", field).split())
    moving = len(moves & set(range(count)))
    return "move" if moving == count else "stay" if moving == 0 else None


def beams_kept(fields: list[str], moves: list[set[int]]) -> list[bool]:
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
            ways = {goes_whole(fields[j], moves[j]) for j in group}
            for j in group:
                kept[j] = len(ways) == 1 and None not in ways
            group = []
    for j in group:  # a group the range cuts off: its other end is not moved with it
        kept[j] = False
    return kept


def moved_up(lines: list[str], moves: list[set[int]], column: int = 0) -> list[str]:
    """The lines with the chosen notes of `column` (moves, per line) in a new last column, opened and
    closed around them."""
    width = len(lines[0].split("\t"))
    keep = ["*"] * (width - 1)
    out = ["\t".join([*keep, "*^"])]
    kept = beams_kept([line.split("\t")[column] for line in lines], moves)
    for line, line_moves, keep_beams in zip(lines, moves, kept):
        fields = line.split("\t")
        if line.startswith("="):
            out.append("\t".join([*fields, fields[-1]]))
        elif line.startswith("*"):
            out.append("\t".join([*fields, "*"]))
        else:
            fields[column], upper = divided(fields[column], line_moves, keep_beams)
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
                depth = columns[i][0] + opened - ended
                if depth < 0:
                    stray_beam_ends_mended(lines, columns[i][1], at, i, -depth)
                columns[i] = max(0, depth), (at, i)
    return "\n".join(lines)


def stray_beam_ends_mended(lines: list[str], before: tuple[int, int] | None, at: int, column: int, stray: int) -> None:
    """A J with no beam open, which makes the scorer's parser drop every beam in the bar. Where the
    column's note just before ended a group, that group ended a note early (`16cccJ 8aaJ` for
    two 16ths and an eighth under one beam: Polish 026, 042) and loses its J instead. Otherwise
    the stray J goes."""
    row, field_at = (before[0], before[1]) if before and "J" in field_of(lines, before) else (at, column)
    fields = lines[row].split("\t")
    fields[field_at] = without_last(fields[field_at], "J", stray)
    lines[row] = "\t".join(fields)


def field_of(lines: list[str], where: tuple[int, int]) -> str:
    return lines[where[0]].split("\t")[where[1]]


def without_last(text: str, mark: str, count: int) -> str:
    for _ in range(count):
        cut = text.rfind(mark)
        if cut < 0:
            break
        text = text[:cut] + text[cut + 1:]
    return text


def staves_of_columns(lines: list[str]) -> list[int]:
    """Each column's staff after these lines, 0 the lowest (kern's spines run from the bottom staff
    up): one column a staff from the header, then through every split and merge of voices."""
    staves: list[int] = []
    for line in lines:
        fields = line.split("\t")
        if line.startswith("**"):
            staves = list(range(len(fields)))
        elif line.startswith("*") and ("*^" in fields or "*v" in fields):
            staves = staves_after_spine_operations(fields, staves)
    return staves


def staves_after_spine_operations(fields: list[str], staves: list[int]) -> list[int]:
    out, i = [], 0
    while i < len(fields):
        if fields[i] == "*^":
            out += [staves[i], staves[i]]
            i += 1
        elif fields[i] == "*v":
            out.append(staves[i])
            i = next((j for j in range(i, len(fields)) if fields[j] != "*v"), len(fields))
        else:
            out.append(staves[i])
            i += 1
    return out


def is_data(line: str) -> bool:
    return bool(line) and not line.startswith(("*", "!", "="))


def held_over(lines: list[str], start: int, column: int) -> bool:
    """Whether the first note line from `start` continues a note in that column (`.`): a voice
    split there would cut a sounding note, and the scorer's parser then reads no parts at all
    (030, 056). A merge across a held note it reads (051)."""
    following = next((line for line in lines[start:] if is_data(line)), None)
    return following is not None and following.split("\t")[column] == "."


def moved_down(lines: list[str], first: int, last: int, moves: list[set[int]] | None, source: int | None) -> list[str]:
    """Lines first to last with column `source`'s chosen notes (moves, per line; None: all of
    them) in a new column after the last of the staff below, opened and closed around them."""
    staves = staves_of_columns(lines[:first])
    source = len(staves) - 1 if source is None else source
    split = max(i for i, staff in enumerate(staves) if staff == staves[source] - 1)
    if held_over(lines, first, split):
        raise ValueError(f"move_down from line {first + 1}: column {split} holds a note across the split")
    column_fields = [line.split("\t")[source] for line in lines[first:last]]
    if moves is None:
        moves = [every_note(field) for field in column_fields]
    kept = beams_kept(column_fields, moves)
    opened = ["*^" if i == split else "*" for i in range(len(staves))]
    out = ["\t".join(opened)]
    for line, line_moves, keep_beams in zip(lines[first:last], moves, kept):
        fields = line.split("\t")
        if not is_data(line):
            if "*^" in fields or "*v" in fields:
                raise ValueError(f"move_down from line {first + 1}: voices split or merged inside the range")
            added = fields[split] if line.startswith("=") else line[0]
        else:
            fields[source], added = divided(fields[source], line_moves, keep_beams)
        out.append("\t".join([*fields[:split + 1], added, *fields[split + 1:]]))
    out.append("\t".join("*v" if i in (split, split + 1) else "*" for i in range(len(staves) + 1)))
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
    """The note with `X` after its accidental, which makes it drawn. A note with none takes an
    explicit natural, `n`, which the scorer's parser always draws; `X` alone it ignores."""
    found = NOTE.search(note)
    return note[:found.end()] + ("X" if found[2] else "n") + note[found.end():]


def with_accidental_display(line: str, column: int, note: int, display) -> str:
    fields = line.split("\t")
    notes = fields[column].split(" ")
    notes[note] = display(notes[note])
    fields[column] = " ".join(notes)
    return "\t".join(fields)


TUPLET_NUMBERS = {"hidden": "*Xtuplet", "shown": "*tuplet"}


def tuplet_numbers_line(before: str, display: str) -> str:
    """An interpretation for every spine of the line it goes before."""
    return "\t".join([TUPLET_NUMBERS[display]] * len(before.split("\t")))


def with_beams(line: str, column: int, marks: str) -> str:
    fields = line.split("\t")
    notes = [BEAM_MARKS.sub("", note) for note in fields[column].split(" ")]
    notes[-1] += marks
    fields[column] = " ".join(notes)
    return "\t".join(fields)


def with_graces_slashed(line: str, column: int) -> str:
    """The line with that field's grace notes slashed: kern's `qq` is an unslashed grace, `q` a slashed one."""
    fields = line.split("\t")
    fields[column] = fields[column].replace("qq", "q")
    return "\t".join(fields)


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


def listed_moves(notes: dict[str, list[int]], first: int, last: int) -> list[set[int]]:
    """For each line from first to last (0-based, last excluded), the notes listed by its 1-based number."""
    return [set(notes.get(str(n + 1), [])) for n in range(first, last)]


def order_on_page(override: dict) -> tuple[int, int]:
    """Applied highest first: on one line, a replace (the line transcribed afresh, so the marks of
    other overrides go onto it, not under it), then articulations, then the rest, then a
    tuplet-number line inserted before the line."""
    kind = 3 if "replace" in override else 2 if "articulate" in override else 0 if "tuplet_numbers" in override else 1
    return override["line"], kind


def corrected(text: str, overrides: list[dict]) -> str:
    lines = text.split("\n")
    # From the bottom up, so that earlier line numbers still mean the unchanged page. On one
    # line (order_on_page), line edits come before a move, which opens its range by inserting a
    # line there, and a tuplet-number line, inserted before the line, comes last.
    for override in sorted(overrides, key=order_on_page, reverse=True):
        first = override["line"] - 1
        if "tuplet_numbers" in override:
            lines.insert(first, tuplet_numbers_line(lines[first], override["tuplet_numbers"]))
        elif "articulate" in override:
            mark = override["articulate"]
            lines[first] = with_marks(lines[first], mark["column"], mark["mark"] + (AMBIGUOUS_SIGNIFIER if mark.get("ambiguous") else ""))
        elif "slur" in override:
            lines[first] = with_marks(lines[first], override["slur"]["column"], override["slur"]["marks"])
        elif "beams" in override:
            lines[first] = with_beams(lines[first], override["beams"]["column"], override["beams"]["marks"])
        elif "slash_grace" in override:
            lines[first] = with_graces_slashed(lines[first], override["slash_grace"]["column"])
        elif "hide_rest" in override:
            fields = lines[first].split("\t")
            fields[override["hide_rest"]["column"]] = rest_hidden(fields[override["hide_rest"]["column"]])
            lines[first] = "\t".join(fields)
        elif "show_accidental" in override:
            at = override["show_accidental"]
            lines[first] = with_accidental_display(lines[first], at["column"], at["note"], with_accidental_shown)
        elif "insert_after" in override:
            lines[first + 1:first + 1] = override["insert_after"]
        elif "replace" in override:
            lines[first:override["replace"]["last_line"]] = override["replace"]["lines"]
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
            notes = override["move_down"].get("notes")
            moves = None if notes is None else listed_moves(notes, first, last)
            lines[first:last] = moved_down(lines, first, last, moves, override["move_down"].get("column"))
        else:
            last = override["move_up"]["last_line"]
            lines[first:last] = moved_up(lines[first:last], listed_moves(override["move_up"]["notes"], first, last),
                                         override["move_up"].get("column", 0))
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
