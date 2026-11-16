"""How a model's labels are written beyond LMX itself, and back: options a model is trained with,
recorded in its checkpoint, applied to each label as it is read for training and taken out of each
reading before anything else sees it.

voice_counts: after each `measure`, one `voices:N` per staff the bar uses, in staff order: how many
voices that staff holds in the bar (VOICE_COUNT_CAP or more as the cap). The decoder commits to a
bar's voices before writing them; without it, it notices a staff's second voice (fed the right
history it writes the backup) yet drops or merges it reading on its own (EXPERIMENTS.md,
"Polyphony: three experiments toward a root cause").

canonical_gaps: one way to write each gap and each return to the bar's start. A run of gaps in a
voice (forwards, rests not drawn) becomes forwards spelled as the greedy sum of note values
(forward takes no dot, and the delinearizer adds up a run); a run of backups that goes back to the
bar's start becomes `backup:bar`. LMX writes a note's voice and staff only where they change, so a
rest not drawn that carried them hands them on to the next note or rest. Fed the right history, the model gets a second voice's entry gap
wrong ten times as often as a note's length, and labels spell one gap several ways (a hidden rest,
a forward, two forwards); 98% of backups go to the bar's start, so their length is arithmetic the
model need not do. Readings need only `backup:bar` spelled out again, from the time its voice has
reached; a backup after a whole-bar rest (whose length the tokens do not give) is left as written.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from fractions import Fraction

from .lmx_tuplets import LENGTHS, PITCH, length_at

VOICE_COUNT_CAP = 4
VOICE_COUNT_TOKENS = [f"voices:{n}" for n in range(1, VOICE_COUNT_CAP + 1)]
STAFF = re.compile(r"^staff:(\d+)$")
BACKUP_TO_BAR = "backup:bar"
PREFIXES = {"print-object:no", "grace", "grace:slash", "chord"}
SPELLED = sorted(((name, Fraction(value)) for name, value in LENGTHS.items() if value <= 4), key=lambda nv: -nv[1])
SPELLED_TRIPLETS = [(name, value * Fraction(2, 3)) for name, value in SPELLED]
HEADS = re.compile(r"^([A-G]-?\d|rest|rest:measure|forward)$")


@dataclass(frozen=True)
class LabelForm:
    voice_counts: bool = False
    canonical_gaps: bool = False

    @classmethod
    def from_options(cls, options: dict) -> LabelForm:
        """From a checkpoint's args (older ones lack the options: plain LMX)."""
        return cls(voice_counts=options.get("voice_counts", False), canonical_gaps=options.get("canonical_gaps", False))

    def options(self) -> dict:
        return asdict(self)

    def label(self, lmx: str) -> str:
        if self.canonical_gaps:
            lmx = with_canonical_gaps(lmx)
        return with_voice_counts(lmx) if self.voice_counts else lmx

    def reading(self, lmx: str) -> str:
        if self.voice_counts:
            lmx = without_voice_counts(lmx)
        return with_backups_spelled(lmx) if self.canonical_gaps else lmx


def voices_per_staff(bar: list[str]) -> dict[int, int]:
    """Each staff's voices in a bar: its stretches between backups that write something there,
    a stretch belonging to the staff its first staff token names."""
    counts: dict[int, int] = {}
    for tokens in stretches(bar):
        staff = next((int(m[1]) for t in tokens if (m := STAFF.match(t))), None)
        if staff is not None and any(HEADS.match(t) for t in tokens):
            counts[staff] = counts.get(staff, 0) + 1
    return counts


def stretches(bar: list[str]) -> list[list[str]]:
    """A bar cut at its backups (each followed by its length, if any)."""
    out: list[list[str]] = [[]]
    for token in bar:
        if token in ("backup", BACKUP_TO_BAR):
            out.append([])
        else:
            out[-1].append(token)
    return out


def with_voice_counts(lmx: str) -> str:
    tokens = lmx.split()
    starts = [i for i, t in enumerate(tokens) if t == "measure"]
    out = tokens[:starts[0]] if starts else tokens
    for s, e in zip(starts, starts[1:] + [len(tokens)]):
        counts = voices_per_staff(tokens[s + 1:e])
        out += ["measure", *(VOICE_COUNT_TOKENS[min(counts[staff], VOICE_COUNT_CAP) - 1] for staff in sorted(counts)),
                *tokens[s + 1:e]]
    return " ".join(out)


def without_voice_counts(lmx: str) -> str:
    return " ".join(t for t in lmx.split() if t not in VOICE_COUNT_TOKENS)


def elements_of(bar: list[str]) -> list[list[str]]:
    """A bar's tokens in runs: each note, rest, forward or backup with what is written with it (its
    prefixes before, its length, marks and staff after), and the tokens before the first."""
    out: list[list[str]] = [[]]
    for k, token in enumerate(bar):
        heads_here = token in PREFIXES or token in ("rest", "rest:measure", "forward", "backup", BACKUP_TO_BAR) \
            or PITCH.match(token)
        if heads_here and not (k and bar[k - 1] in PREFIXES):
            out.append([])
        out[-1].append(token)
    return out


def length_of(element: list[str]) -> Fraction | None:
    """The time a note, rest, forward or backup takes (None for a whole-bar rest)."""
    if "rest:measure" in element:
        return None
    at = next((k for k, t in enumerate(element) if t in LENGTHS), None)
    return length_at(element, at)[0] if at is not None else Fraction(0)


def advances(element: list[str]) -> bool:
    return not ({"chord", "grace", "grace:slash"} & set(element)) and element[0] not in ("backup", BACKUP_TO_BAR)


def is_gap(element: list[str]) -> bool:
    """A forward, or a rest not drawn, carrying nothing but its voice, length and staff."""
    plain = [t for t in element if not (t.startswith(("voice:", "staff:")) or t in LENGTHS or re.match(r"^\d+in\d+$", t))]
    return plain in (["forward"], ["print-object:no", "rest"])


def spelled(length: Fraction, head: str) -> list[str] | None:
    """`head VALUE ...` adding up to length: plain note values largest first where they suffice,
    else triplet values (`VALUE 3in2`) for the part plain ones cannot reach; None if neither can."""
    values = plain_values(length)
    if values is None:
        values = next((v for name, t in SPELLED_TRIPLETS if t <= length
                       for v in [with_triplet(name, plain_values(length - t))] if v is not None), None)
    return None if values is None else [token for value in values for token in (head, *value)]


def plain_values(length: Fraction) -> list[tuple[str, ...]] | None:
    values = []
    for name, value in SPELLED:
        while length >= value:
            values.append((name,))
            length -= value
    return values if length == 0 else None


def with_triplet(name: str, rest: list[tuple[str, ...]] | None) -> list[tuple[str, ...]] | None:
    return None if rest is None else [(name, "3in2"), *rest]


def with_canonical_gaps(lmx: str) -> str:
    return " ".join(token for bar in bars_of(lmx.split()) for token in canonical_bar(bar))


def bars_of(tokens: list[str]) -> list[list[str]]:
    """Each bar from its `measure` on, after whatever comes before the first."""
    starts = [i for i, t in enumerate(tokens) if t == "measure"]
    if not starts:
        return [tokens]
    before = [tokens[:starts[0]]] if starts[0] else []
    return before + [tokens[s:e] for s, e in zip(starts, starts[1:] + [len(tokens)])]


def with_voice_and_staff(element: list[str], run: list[list[str]]) -> list[str]:
    """element given the last voice and staff a run of gaps named, where it names none itself (the
    voice after its head, the staff at its end, as the linearizer writes them)."""
    voice = next((t for e in reversed(run) for t in reversed(e) if t.startswith("voice:")), None)
    staff = next((t for e in reversed(run) for t in reversed(e) if t.startswith("staff:")), None)
    out = list(element)
    if voice and not any(t.startswith("voice:") for t in out):
        head = next(k for k, t in enumerate(out) if t not in PREFIXES)
        out.insert(head + 1, voice)
    if staff and not any(t.startswith("staff:") for t in out):
        out.append(staff)
    return out


class BarClock:
    """Where the voice being written stands in its bar, in quarters, and whether every length passed
    so far is known (a whole-bar rest's is not)."""

    def __init__(self) -> None:
        self.time, self.known = Fraction(0), True

    def passes(self, element: list[str]) -> None:
        if element[0] == BACKUP_TO_BAR:
            self.time = Fraction(0)
            return
        length = length_of(element)
        if length is None:
            self.known = False
        elif element[0] == "backup":
            self.time -= length
        elif advances(element):
            self.time += length

    def backup_to_bar(self) -> list[str] | None:
        """The backups back to the bar's start, if the way back is known and can be spelled."""
        return spelled(self.time, "backup") if self.known else None


def canonical_bar(bar: list[str]) -> list[str]:
    elements = elements_of(bar)
    out, clock = [], BarClock()
    k = 0
    while k < len(elements):
        element = elements[k]
        if element and element[0] == "backup":
            run_end = k
            while run_end < len(elements) and elements[run_end][0] == "backup":
                run_end += 1
            run = elements[k:run_end]
            only_lengths = all(len(e) <= 3 for e in run)   # `backup VALUE [RATIO]`, nothing else written with it
            back = sum((length_of(e) for e in run), Fraction(0))
            if only_lengths and back == clock.time and clock.backup_to_bar() is not None:
                out.append(BACKUP_TO_BAR)
            else:
                out += [t for e in run for t in e]
        elif element and is_gap(element):
            run_end = k
            while run_end < len(elements) and is_gap(elements[run_end]):
                run_end += 1
            run = elements[k:run_end]
            forwards = spelled(sum((length_of(e) for e in run), Fraction(0)), "forward")
            if forwards is None:
                out += [t for e in run for t in e]
            else:
                out += forwards
                if run_end < len(elements) and elements[run_end][0] not in ("backup", "forward"):
                    elements[run_end] = with_voice_and_staff(elements[run_end], run)
        else:
            run_end, run = k + 1, [element]
            out += element
        for e in run:
            if e:
                clock.passes(e)
        k = run_end
    return out


def with_backups_spelled(lmx: str) -> str:
    """Each `backup:bar` written out as backups as long as its voice has run; one whose length the
    tokens do not give (after a whole-bar rest) is dropped."""
    out = []
    for bar in bars_of(lmx.split()):
        clock = BarClock()
        for element in elements_of(bar):
            if not element:
                continue
            out += (clock.backup_to_bar() or []) + element[1:] if element[0] == BACKUP_TO_BAR else element
            clock.passes(element)
    return " ".join(out)
