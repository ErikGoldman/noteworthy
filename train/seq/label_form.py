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

clock: after each note, rest or gap that carries a voice past a whole quarter of its bar, `clock:N`,
the quarter it has reached (CLOCK_CAP the most), so a bar written right ends `clock:L` before the
next `measure`, L its length. The model closes a bar where it sees the barline and does not add up
its notes (EXPERIMENTS.md, "Probes: the model does not count beats"): three triplet eighths read as
plain reach the beat one note early, and written out, that is an error the loss sees. Readings have
the tokens taken out first. A voice's clock stops at a whole-bar rest, whose length is not written.

onsets: before each note, rest or gap that moves its voice on, where in the bar it starts, factorized
as `beat:N` (whole quarters, ONSET_BEAT_CAP the most) and `sub:P/Q` (the rest, `sub:0` on the beat; Q
one of ONSET_DENOMINATORS): `beat:0 sub:0 C4 eighth 3in2 beat:0 sub:1/3 D4 eighth 3in2 ...`. A length
is then the gap between two positions, local arithmetic, where the clock left a running sum to do;
and the model reads positions off the page well (it writes the image's clock where its own notes
say otherwise, EXPERIMENTS.md, "Is the clock counted or read off the page?"). Precedent: REMI's Bar
and Position tokens (Huang & Yang 2020), on a grid that here must hold thirds. Chord members and
graces take none (they share their note's onset or take no time); a voice's onsets stop after a
whole-bar rest, whose length is not written, and where a position is past the cap or off the grid.
Readings have the tokens taken out first.

positions: every run of backups and of gaps (forwards, rests not drawn) becomes one `at:P`, the
position in the bar, in quarters, where the next event starts (TODO step 3a): an onset the decoder
already knows from the other voices, where a forward's length is a sum it must work out and gets
wrong ten times as often as a note's (EXPERIMENTS.md, "Is the entry gap itself hard?"). Reading,
`at:P` is a backup when P is behind where the voice stands and the next event names another voice,
a forward when ahead. A position behind its voice with no change of voice cannot be placed, nor one
in a bar whose time is not known: it is read as no gap and reported (UnplacedPosition), never
dropped silently. A run left as written: where
the bar's time is not known (after a whole-bar rest) or falls before its start (a label backing up
past it), where a backup carries more than its length,
or where the way to P cannot be spelled again.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction

from .bar_voices import voices_written
from .lmx_tuplets import LENGTHS, PITCH, length_at

VOICE_COUNT_CAP = 4
VOICE_COUNT_TOKENS = [f"voices:{n}" for n in range(1, VOICE_COUNT_CAP + 1)]
STAFF_OR_VOICE = re.compile(r"^(staff|voice):\d+$")
BACKUP_TO_BAR = "backup:bar"
PREFIXES = {"print-object:no", "grace", "grace:slash", "chord"}
SPELLED = sorted(((name, Fraction(value)) for name, value in LENGTHS.items() if value <= 4), key=lambda nv: -nv[1])
SPELLED_TRIPLETS = [(name, value * Fraction(2, 3)) for name, value in SPELLED]
POSITION = re.compile(r"^at:(\d+(?:/\d+)?)$")
CLOCK_CAP = 16
CLOCK_TOKENS = [f"clock:{n}" for n in range(1, CLOCK_CAP + 1)]
CLOCK = re.compile(r"^clock:\d+$")
ONSET_BEAT_CAP = 15
ONSET_DENOMINATORS = (2, 3, 4, 6, 8, 12, 16)
ONSET_SUBS = ["sub:0"] + sorted({f"sub:{Fraction(p, q)}" for q in ONSET_DENOMINATORS for p in range(1, q)},
                                key=lambda t: Fraction(t[4:]))
ONSET_TOKENS = [f"beat:{n}" for n in range(ONSET_BEAT_CAP + 1)] + ONSET_SUBS
ONSET = re.compile(r"^(beat:\d+|sub:\d+(/\d+)?)$")


@dataclass(frozen=True)
class LabelForm:
    voice_counts: bool = False
    canonical_gaps: bool = False
    clock: bool = False
    onsets: bool = False
    interleaved: bool = False  # each bar's voices in time order (with_interleaving); implies canonical_gaps

    def __post_init__(self):
        if self.interleaved and not self.canonical_gaps:
            raise ValueError("interleaved needs canonical_gaps: its lanes start at backup:bar")

    @classmethod
    def from_options(cls, options: dict) -> LabelForm:
        """From a checkpoint's args (older ones lack the options: plain LMX)."""
        return cls(voice_counts=options.get("voice_counts", False), canonical_gaps=options.get("canonical_gaps", False),
                   clock=options.get("clock", False), onsets=options.get("onsets", False),
                   interleaved=options.get("interleaved", False))

    def options(self) -> dict:
        return asdict(self)

    def label(self, lmx: str) -> str:
        if self.canonical_gaps:
            lmx = with_canonical_gaps(lmx)
        if self.interleaved:
            lmx = with_interleaving(lmx)
        if self.voice_counts:
            lmx = with_voice_counts(lmx)
        if self.clock:
            lmx = with_clock(lmx)
        return with_onsets(lmx) if self.onsets else lmx

    def reading(self, lmx: str) -> str:
        if self.onsets:
            lmx = without_onsets(lmx)
        if self.clock:
            lmx = without_clock(lmx)
        if self.voice_counts:
            lmx = without_voice_counts(lmx)
        if self.interleaved:
            lmx = without_interleaving(lmx)
        return with_backups_spelled(with_hidden_bar_rests_restored(lmx)) if self.canonical_gaps else lmx


def voices_per_staff(bar: list[str]) -> dict[int, int]:
    """Each staff's voices in a bar as the label writes them (seq.bar_voices.voices_written), the
    bar cut at its `backup:bar` tokens first, which have no length for the parser to go back by."""
    counts: Counter = Counter()
    for tokens in stretches(bar):
        counts.update(voices_written(tokens))
    return dict(counts)


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
            or PITCH.match(token) or POSITION.match(token)
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


def draws_nothing(element: list[str]) -> bool:
    """A forward, or a rest not drawn, carrying nothing but its voice, length (dotted or not) and staff."""
    plain = [t for t in element if not (t.startswith(("voice:", "staff:")) or t in LENGTHS or t == "dot"
                                        or re.match(r"^\d+in\d+$", t))]
    return plain in (["forward"], ["print-object:no", "rest"])


def is_gap(element: list[str]) -> bool:
    """A gap canonical_gaps respells: forwards take no dot, so a dotted one is left as written."""
    return draws_nothing(element) and "dot" not in element


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


def clocked_bar(bar: list[str]) -> list[str]:
    out, clock, reached = [], BarClock(), 0
    for element in elements_of(bar):
        out += element
        if not element:
            continue
        clock.passes(element)
        if element[0] in ("backup", BACKUP_TO_BAR):
            reached = int(clock.time) if clock.known else reached
        elif clock.known and advances(element) and int(clock.time) > reached:
            reached = int(clock.time)
            if reached <= CLOCK_CAP:
                out.append(f"clock:{reached}")
    return out


def with_clock(lmx: str) -> str:
    tokens = lmx.split()
    bars = bars_of(tokens)
    head = tokens[:len(tokens) - sum(map(len, bars))]
    return " ".join(head + [t for bar in bars for t in clocked_bar(bar)])


def without_clock(lmx: str) -> str:
    return " ".join(t for t in lmx.split() if not CLOCK.match(t))


def onset_tokens(time: Fraction) -> list[str] | None:
    """`beat:N sub:P/Q` for a position in the bar, or None where it is past the cap or off the grid."""
    beat, sub = int(time), time - int(time)
    token = "sub:0" if sub == 0 else f"sub:{sub}"
    return [f"beat:{beat}", token] if 0 <= beat <= ONSET_BEAT_CAP and token in ONSET_SUBS else None


def onset_bar(bar: list[str]) -> list[str]:
    out, clock = [], BarClock()
    for k, element in enumerate(elements_of(bar)):
        is_event = k > 0  # elements_of puts the tokens before the bar's first event (measure, clefs) first
        if is_event and advances(element) and clock.known and (position := onset_tokens(clock.time)):
            out += position
        out += element
        if element:
            clock.passes(element)
    return out


def with_onsets(lmx: str) -> str:
    tokens = lmx.split()
    bars = bars_of(tokens)
    head = tokens[:len(tokens) - sum(map(len, bars))]
    return " ".join(head + [t for bar in bars for t in onset_bar(bar)])


def without_onsets(lmx: str) -> str:
    return " ".join(t for t in lmx.split() if not ONSET.match(t))


MAX_LANES = 8
LANE = re.compile(r"^lane:(\d+)$")
LANE_TOKENS = [f"lane:{k}" for k in range(1, MAX_LANES + 1)]


def lanes_of(elements: list[list[str]]) -> list[list[list[str]]] | None:
    """A canonical bar's elements after its head cut into lanes at its backups: at a `backup:bar` (dropped:
    the next lane starts at the bar's start, implied) and at a partial `backup` (kept as its lane's first
    element: a voice coming in partway); None with more than MAX_LANES."""
    lanes: list[list[list[str]]] = [[]]
    for element in elements:
        if element == [BACKUP_TO_BAR]:
            lanes.append([])
        elif element[0] == "backup":
            lanes.append([element])
        else:
            lanes[-1].append(element)
    return lanes if len(lanes) <= MAX_LANES else None


def onsets_of_lanes(lanes: list[list[list[str]]]) -> list[list[Fraction]]:
    """Each element's onset from the bar's start, lane by lane: a chord member at its head's (a grace
    note's chord at the grace note's), a grace note at the note after it, a lane's opening backup where it
    takes the time back to, a whole-bar rest (no length) at its lane's start. A backup going back past the
    bar's start (after a whole-bar rest, whose length is not written) starts its lane at the bar's start:
    onsets only order the elements, and deinterleaved_bar gives each lane back as it was whatever they are."""
    out, end = [], Fraction(0)
    for lane in lanes:
        time = Fraction(0)
        if lane and lane[0][0] == "backup":
            time = max(end - (length_of(lane[0]) or Fraction(0)), Fraction(0))
        onsets, head = [], time
        for element in lane:
            if element[0] == "backup":
                onsets.append(time)
            elif "chord" in element:
                onsets.append(head)  # its head's, a grace note's too
            else:
                onsets.append(time)
                head = time
                if advances(element):
                    time += length_of(element) or Fraction(0)
        out.append(onsets)
        end = time
    return out


def interleaved_bar(bar: list[str]) -> list[str] | None:
    """A canonical bar with its lanes' elements in time order: by onset, then lane, then as written;
    `lane:K` before an element whose lane is not the one before it (the first lane needs none at the
    bar's start). None where the bar has no lanes to interleave (lanes_of)."""
    elements = elements_of(bar)
    lanes = lanes_of(elements[1:])
    if lanes is None:
        return None
    onsets = onsets_of_lanes(lanes)
    timed = sorted(((onset, k, n, element) for k, lane in enumerate(lanes)
                    for n, (onset, element) in enumerate(zip(onsets[k], lane))),
                   key=lambda item: item[:3])
    out, current = list(elements[0]), 0
    for _, k, _, element in timed:
        if k != current:
            out.append(LANE_TOKENS[k])
            current = k
        out += element
    return out


def deinterleaved_bar(bar: list[str]) -> list[str]:
    """interleaved_bar undone: each lane's tokens in order, a lane joined to the one before by `backup:bar`
    unless it opens with its own backup (lanes with no element are kept empty, so the next keeps its number)."""
    lanes: list[list[str]] = [[]]
    head, current, started = [], 0, False
    for token in bar:
        lane = LANE.match(token)
        if lane:
            current, started = int(lane[1]) - 1, True
            while len(lanes) <= current:
                lanes.append([])
            continue
        if not started and not lanes[0] and not is_head(token) and token not in PREFIXES:
            head.append(token)
            continue
        started = True
        lanes[current].append(token)
    out = head + lanes[0]
    for lane in lanes[1:]:
        out += (lane if lane[:1] == ["backup"] else [BACKUP_TO_BAR] + lane)
    return out


def is_head(token: str) -> bool:
    return token in ("rest", "rest:measure", "forward", "backup", BACKUP_TO_BAR) or bool(PITCH.match(token))


def with_interleaving(lmx: str) -> str:
    """Canonical LMX (with_canonical_gaps) with each bar's voices interleaved in time (interleaved_bar); a
    bar that cannot be is left as written (deinterleaved_bar reads it back unchanged; interleaving_failures
    counts them)."""
    tokens = lmx.split()
    bars = bars_of(tokens)
    out = []
    for bar in bars:
        done = interleaved_bar(bar) if bar[:1] == ["measure"] else None
        out += done if done is not None else bar
    return " ".join(out)


def interleaving_failures(lmx: str) -> int:
    """How many of the canonical LMX's bars with_interleaving leaves as written."""
    return sum(interleaved_bar(bar) is None for bar in bars_of(lmx.split()) if bar[:1] == ["measure"])


def without_interleaving(lmx: str) -> str:
    return " ".join(t for bar in bars_of(lmx.split()) for t in (deinterleaved_bar(bar) if bar[:1] == ["measure"] else bar))


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


def has_rest_head(written: list[str]) -> bool:
    """Whether the tokens written so far end in a rest's head, with its voice or staff after it."""
    return written[-1:] == ["rest"] or (written[-2:-1] == ["rest"] and bool(STAFF_OR_VOICE.match(written[-1])))


def with_hidden_bar_rests_restored(lmx: str) -> str:
    """A hidden whole-bar rest (`print-object:no rest voice:N rest:measure`) is a gap, so
    with_canonical_gaps takes its hidden head away and leaves `rest:measure voice:N`; a drawn one
    keeps its `rest`. So a `rest:measure` with no `rest` before it is a hidden one, and is written
    out whole again (without its head it converts to a note with neither pitch nor length)."""
    tokens, out, k = lmx.split(), [], 0
    while k < len(tokens):
        if tokens[k] == "rest:measure" and not has_rest_head(out):
            voice = tokens[k + 1:k + 2] if tokens[k + 1:k + 2] and tokens[k + 1].startswith("voice:") else []
            out += ["print-object:no", "rest", *voice, "rest:measure"]
            k += 1 + len(voice)
        else:
            out.append(tokens[k])
            k += 1
    return " ".join(out)


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


@dataclass(frozen=True)
class UnplacedPosition:
    """An `at:` read as no gap: behind where its voice stands with the voice going on, or in a bar
    whose time is not known (reached None, after a whole-bar rest)."""
    bar: int                    # the bar's `measure`, counted from 0
    voice: str                  # the voice token in force ("" before any)
    position: Fraction
    reached: Fraction | None


def moves_in_time(element: list[str]) -> bool:
    """A backup, or a gap with a length: a hidden whole-bar rest's head (`print-object:no rest
    voice:N`, its `rest:measure` an element of its own) has none, and stays with its rest."""
    return bool(element) and (element[0] == "backup" or (draws_nothing(element) and any(t in LENGTHS for t in element)))


def names_another_voice(element: list[str], voice: str) -> bool:
    return any(t.startswith("voice:") and t != voice for t in element)


def placeable(position: Fraction, reached: Fraction, following: list[str], voice: str) -> bool:
    """Whether a reading can place `at:position`: a way there is spelled, and going back starts
    another voice (a voice that backs up into itself, as some GrandStaff labels do, cannot be told
    from a misread position)."""
    return way_to(position, reached) is not None and (position >= reached or names_another_voice(following, voice))


def way_to(position: Fraction, reached: Fraction) -> list[str] | None:
    """The backups or forwards that take a voice standing at reached to position."""
    if position < reached:
        return spelled(reached - position, "backup")
    return spelled(position - reached, "forward") if position > reached else []


def with_positions(lmx: str) -> str:
    return " ".join(token for bar in bars_of(lmx.split()) for token in positioned_bar(bar))


def positioned_bar(bar: list[str]) -> list[str]:
    elements = elements_of(bar)
    out, clock, k, voice = [], BarClock(), 0, ""
    while k < len(elements):
        if not moves_in_time(elements[k]):
            out += elements[k]
            clock.passes(elements[k])
            voice = next((t for t in elements[k] if t.startswith("voice:")), voice)
            k += 1
            continue
        run_end = k
        while run_end < len(elements) and moves_in_time(elements[run_end]):
            run_end += 1
        run, before = elements[k:run_end], clock.time
        for element in run:
            clock.passes(element)
        plain_backups = all(len(e) <= 3 for e in run if e[0] == "backup")   # `backup VALUE [RATIO]`
        following = with_voice_and_staff(elements[run_end], run) if run_end < len(elements) else []
        if clock.known and clock.time >= 0 and plain_backups and placeable(clock.time, before, following, voice):
            out.append(f"at:{clock.time}")
            if following:
                elements[run_end] = following
        else:
            out += [t for e in run for t in e]
            voice = next((t for e in reversed(run) for t in reversed(e) if t.startswith("voice:")), voice)
        k = run_end
    return out


def with_gaps_from_positions(lmx: str) -> tuple[str, list[UnplacedPosition]]:
    """Each `at:P` written out as the backups or forwards that reach P, and the positions that
    could not be placed."""
    out, unplaced, number = [], [], -1
    for bar in bars_of(lmx.split()):
        number += bar[:1] == ["measure"]
        elements = [e for e in elements_of(bar) if e]
        clock, voice = BarClock(), ""
        for k, element in enumerate(elements):
            position = POSITION.match(element[0])
            if not position:
                out += element
                clock.passes(element)
                voice = next((t for t in element if t.startswith("voice:")), voice)
                continue
            target = Fraction(position[1])
            following = elements[k + 1] if k + 1 < len(elements) else []
            if not (clock.known and placeable(target, clock.time, following, voice)):
                unplaced.append(UnplacedPosition(number, voice, target, clock.time if clock.known else None))
                continue
            way = way_to(target, clock.time)
            out += way + element[1:]
            for move in elements_of(way)[1:]:
                clock.passes(move)
    return " ".join(out), unplaced
