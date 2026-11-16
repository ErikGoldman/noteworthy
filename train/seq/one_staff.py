"""A two-staff system taken apart into two one-staff systems, label and image, to train and score
the models on one staff at a time (EXPERIMENTS.md, (k)). Free of torch, as
staff_positions.py, for the builders.

Labels. A bar of LMX is a header (key, meter, each staff's clef), then runs of notes separated
by backups: each run one voice as MusicXML writes it. A note names its staff (staff:N) only when
it changes, and the linearizer forgets the staff, voice and stem at each bar and backup, so every
run names its own. A staff's label keeps the header's key and meter and its own clef, and its own
runs, the backups between them recounted over the runs left out; staff tokens go, as OLiMPiC's
linearizer writes a one-staff part. What cannot be put on one staff without guessing leaves the
system out (Unsplittable): a run with notes on both staves (a voice crossing between the hands,
a beam or chord across them), a clef change among the other staff's notes, a key or meter change
within a bar, notes and backups that do not add up, a whole-bar rest of unknown length, an
ottava sign between the staves.

Images. Between the staves (half an interline clear of their outer lines) each piece of ink is
given to a staff (gap_owners): to the one it reaches, both if it runs right through (barlines,
the brace), and a piece reaching neither (a slur, a dynamic, a dot) to the staff on whose side of
the middle it lies. Each staff's crop ends at its own last ink there, the other's painted out.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from .barlines import BARLINES
from .lmx_tuplets import LENGTHS, PITCH, length_at
from .staff_positions import Staves

STAVES = (1, 2)
NOTE_PREFIXES = {"print-object:no", "grace", "grace:slash", "chord"}
MOVES = ("backup", "forward")
HEADER_TOKEN = re.compile(r"(key:fifths:-?\d+|time|time:common|time:cut|beats:\d+|beat-type:\d+|clef:[CFG]\d|staff:\d)$")
METER_OR_KEY = re.compile(r"(key:fifths:-?\d+|time|beats:\d+|beat-type:\d+)$")
# Signs drawn between the staves: below the upper, above the lower. Alone in the gap, nothing
# says whose they are (gap_owners would guess by their place, and PDMX's sit as near the other staff).
OTTAVAS_BETWEEN = ({"ottava:8vb", "ottava:15mb"}, {"ottava:8va", "ottava:15ma"})
PAPER, UPPER, LOWER, BOTH = 0, 1, 2, 3  # what a piece of the gap's ink is, by staff
# Ink, as seq.data.blank_margins has it: darker than mid-grey.
INK_BELOW = 128
# The gap's rows start this many interlines clear of the staves' outer lines, so that staff lines
# (tilted ones too) are not taken for ink in the gap.
LINE_CLEARANCE = 0.5
HALO_INTERLINES = 0.1  # painted out around the other staff's ink, each side (painted_out)


class Unsplittable(ValueError):
    pass


@dataclass(frozen=True)
class Run:
    tokens: list[str]
    staff: int | None  # None for a run of moves alone
    length: Fraction | None  # how far it moves the cursor; None with a whole-bar rest


@dataclass(frozen=True)
class Bar:
    opening: list[str]  # barline tokens straight after `measure`
    header: list[str]
    runs: list[Run]
    backups: list[Fraction]  # backups[k] comes before runs[k + 1]
    closing: list[str]  # barline tokens at the bar's end


def staff_labels(lmx: str) -> tuple[str, str]:
    """The upper and lower staff's labels of a two-staff system's LMX; Unsplittable if they cannot
    be had without guessing."""
    bars = bars_of(lmx.split())
    labels = tuple(" ".join(token for bar in bars for token in bar_on_staff(bar, staff)) for staff in STAVES)
    if any(sign in label.split() for label, signs in zip(labels, OTTAVAS_BETWEEN) for sign in signs):
        raise Unsplittable("an ottava sign between the staves")
    return labels


def bars_of(tokens: list[str]) -> list[Bar]:
    starts = [i for i, t in enumerate(tokens) if t == "measure"]
    if not starts or starts[0] != 0:
        raise Unsplittable("tokens before the first bar")
    return [bar_of(tokens[s + 1:e]) for s, e in zip(starts, starts[1:] + [len(tokens)])]


def bar_of(tokens: list[str]) -> Bar:
    """A bar's tokens after its `measure`."""
    begin = next((i for i, t in enumerate(tokens) if t not in BARLINES), len(tokens))
    end = len(tokens)
    while end > begin and tokens[end - 1] in BARLINES:
        end -= 1
    body = next((i for i in range(begin, end) if starts_music(tokens[i])), end)
    header = tokens[begin:body]
    if not all(HEADER_TOKEN.match(t) for t in header):
        raise Unsplittable("an unexpected token before the bar's notes")
    runs, backups, start, i = [], [], body, body
    while i < end:
        if tokens[i] != "backup":
            i += 1
            continue
        runs.append(run_of(tokens[start:i]))
        back = Fraction(0)
        while i < end and tokens[i] == "backup":
            back += LENGTHS[tokens[i + 1]]
            i += 2
        backups.append(back)
        start = i
    runs.append(run_of(tokens[start:end]))
    return Bar(tokens[:begin], header, runs, backups, tokens[end:])


def starts_music(token: str) -> bool:
    return token in NOTE_PREFIXES or token in MOVES or token == "rest" or PITCH.match(token) is not None


def run_of(tokens: list[str]) -> Run:
    if any(METER_OR_KEY.match(t) for t in tokens):
        raise Unsplittable("a key or meter change within a bar")
    clef_staves = {int(tokens[i + 1][6:]) for i, t in enumerate(tokens) if t.startswith("clef:")}
    note_staves = {int(t[6:]) for i, t in enumerate(tokens)
                   if t.startswith("staff:") and not (i and tokens[i - 1].startswith("clef:"))}
    has_notes = any(t == "rest" or PITCH.match(t) for t in tokens)
    if not note_staves | clef_staves <= set(STAVES):
        raise Unsplittable("a third staff")
    if len(note_staves) > 1:
        raise Unsplittable("a voice on both staves")
    if has_notes and not note_staves:
        raise Unsplittable("notes with no staff")
    if len(note_staves | clef_staves) > 1:
        raise Unsplittable("a clef change among the other staff's notes")
    # A run of a clef change alone (moved to its place by forwards) is that staff's.
    staff = next(iter(note_staves | clef_staves), None)
    return Run(tokens, staff, run_length(tokens))


def run_length(tokens: list[str]) -> Fraction | None:
    """How far a run moves the cursor: its notes but chord notes and grace notes, and its forwards.
    None for a whole-bar rest, whose length LMX does not write."""
    length, head, timeless = Fraction(0), None, False
    for i, token in enumerate(tokens):
        if token in ("chord", "grace"):
            timeless = True
        elif token == "rest:measure":
            return None
        elif token in ("rest", "forward") or PITCH.match(token):
            head = token
        elif token in LENGTHS and head:
            if head == "forward" or not timeless:
                length += length_at(tokens, i)[0]
            head, timeless = None, False
    return length


def bar_on_staff(bar: Bar, staff: int) -> list[str]:
    out = ["measure", *bar.opening, *header_on_staff(bar.header, staff)]
    kept = [k for k, run in enumerate(bar.runs) if run.staff == staff]
    for previous, k in zip([None, *kept], kept):
        out += moves(cursor_move(bar, previous, k))
        out += [t for t in bar.runs[k].tokens if not t.startswith("staff:")]
    return out + bar.closing


def header_on_staff(header: list[str], staff: int) -> list[str]:
    """The key and meter, and the staff's own clef, with no staff token."""
    out, i = [], 0
    while i < len(header):
        if header[i].startswith("clef:"):
            if header[i + 1] == f"staff:{staff}":
                out.append(header[i])
            i += 2
        else:
            out.append(header[i])
            i += 1
    return out


def cursor_move(bar: Bar, previous: int | None, k: int) -> Fraction:
    """From the end of run `previous` (the bar's start if None) to the start of run k: the runs
    between, less the backups between. A move between two runs must go back; a forward, or none,
    would leave the voice, stem and staff the linearizer keeps unreset, and its tokens would not
    be the linearizer's."""
    if previous is None:
        skipped, backups = range(0, k), bar.backups[:k]
    else:
        skipped, backups = range(previous + 1, k), bar.backups[previous:k]
    lengths = [whole_bar_rest_length(bar, j) if bar.runs[j].length is None else bar.runs[j].length for j in skipped]
    move = sum(lengths, Fraction(0)) - sum(backups, Fraction(0))
    if previous is None and move < 0:
        raise Unsplittable("notes and backups that do not add up")
    if previous is not None and move >= 0:
        raise Unsplittable("a voice that does not back up to its start")
    return move


def whole_bar_rest_length(bar: Bar, j: int) -> Fraction:
    """The length of run j, a whole-bar rest, which LMX does not write: the backup after it, if
    that goes back to the bar's start. It does where the bar's longest voice of written length is
    as long (1,922 of 1,961 whole-bar rests sampled), and where every voice is a whole-bar rest,
    as each starts the bar."""
    written = [run.length for run in bar.runs if run.length is not None and run.staff is not None]
    if written and bar.backups[j] != max(written):
        raise Unsplittable("a whole-bar rest whose backup is not the bar's length")
    return bar.backups[j]


def moves(distance: Fraction) -> list[str]:
    """A backup (distance below 0) or forward over |distance|, in the types OLiMPiC's linearizer
    spells it with: each at most once, longest first."""
    kind, remaining, out = ("backup" if distance < 0 else "forward"), abs(distance), []
    for name, length in sorted(LENGTHS.items(), key=lambda kv: -kv[1]):
        if length <= remaining:
            out += [kind, name]
            remaining -= length
    if remaining:
        raise Unsplittable("a move no note types add up to")
    return out


def staff_images(image: np.ndarray, staves: Staves) -> tuple[tuple[np.ndarray, Staves], tuple[np.ndarray, Staves]]:
    """The upper and lower staff's crops of a two-staff system (dark ink on light), each with
    its staff's position in its own rows: each ends at its own last ink in the gap between the
    staves (gap_owners), the other staff's ink there painted over in the paper's tone."""
    upper, lower = staves.centres
    top, bottom = gap_rows(staves)
    owner = gap_owners(image[top:bottom], (upper + lower) / 2 - top)
    halo = 2 * max(1, round(HALO_INTERLINES * staves.interline)) + 1
    paper = np.median(image)
    upper_rows, lower_rows = (np.flatnonzero((owner == staff).any(axis=1)) for staff in (UPPER, LOWER))
    upper_end = top + (upper_rows[-1] + 1 if len(upper_rows) else 0)
    lower_start = top + (lower_rows[0] if len(lower_rows) else bottom - top)
    upper_crop, lower_crop = image[:upper_end].copy(), image[lower_start:].copy()
    upper_crop[top:][painted_out(owner, LOWER, halo)[:upper_end - top]] = paper
    lower_crop[:bottom - lower_start][painted_out(owner, UPPER, halo)[lower_start - top:]] = paper
    return (upper_crop, Staves((upper,), staves.interline)), (lower_crop, Staves((lower - lower_start,), staves.interline))


def painted_out(owner: np.ndarray, other: int, halo: int) -> np.ndarray:
    """The other staff's ink and the paper around it, `halo` pixels across: the grey edge that
    anti-aliasing or blur gives a symbol lies outside its ink, and left, it outlines the symbol."""
    around = cv2.dilate((owner == other).astype(np.uint8), np.ones((halo, halo), np.uint8)).astype(bool)
    return around & ((owner == other) | (owner == PAPER))


def gap_rows(staves: Staves) -> tuple[int, int]:
    """The rows between the staves, LINE_CLEARANCE clear of their outer lines."""
    upper, lower = staves.centres
    clearance = (2 + LINE_CLEARANCE) * staves.interline
    top, bottom = math.ceil(upper + clearance), math.floor(lower - clearance) + 1
    if bottom - top < 2:
        raise Unsplittable("staves too close to cut between")
    return top, bottom


def gap_owners(gap: np.ndarray, middle: float) -> np.ndarray:
    """Whose each pixel of the gap's ink is (PAPER where there is none), by the piece of ink it is
    in: the upper staff's if the piece reaches the gap's top (a stem, a note's ledger lines), the
    lower's if it reaches its bottom, BOTH if it runs right through (a barline, the brace), and a
    piece touching neither (a slur, a dynamic, a dot) the staff on whose side of `middle` its
    centre lies."""
    count, parts, _, centres = cv2.connectedComponentsWithStats((gap < INK_BELOW).astype(np.uint8), connectivity=8)
    reaches_top, reaches_bottom = np.zeros(count, bool), np.zeros(count, bool)
    reaches_top[parts[0]], reaches_bottom[parts[-1]] = True, True
    owners = np.where(centres[:, 1] < middle, UPPER, LOWER)
    owners[reaches_top & ~reaches_bottom] = UPPER
    owners[reaches_bottom & ~reaches_top] = LOWER
    owners[reaches_top & reaches_bottom] = BOTH
    owners[0] = PAPER
    return owners[parts]
