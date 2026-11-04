"""Note-level ground truth for MUSCIMA++ pages, derived from their MuNG symbol graphs.

MUSCIMA++ annotates symbols and how they relate, not notes in time. From the graph:
  pitch     mung2midi's pitch inference (clefs, key signatures, accidentals, ledger lines)
  duration  each notehead's own links: head kind, beams, flags, augmentation dots, tuplet
            numeral (mung2midi's duration inference fails on notes shared by two stems)
  measure   systems are the staves joined by measure separators, which also split them
            into measures; measures are numbered on through the page's systems
  onset     not annotated: each staff's chords and rests are read left to right through
            the measure as one chain, or as one chain per stem direction (rests in both)
            when the staff holds two voices
A measure is scored only when every staff's chains end at one common length. Where they
do not (writers' shorthand, notes shared by two stems, unmarked tuplets, multi-measure
rests), the measure keeps its notes for measure alignment but is not scored.
"""
from __future__ import annotations

import contextlib
import io
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from mung.io import read_nodes_from_file
from mung.node import Node
from mung2midi.inference import PitchInferenceEngine

from score_events import NoteEvent

HEAD_VALUE = {"noteheadWhole": Fraction(4), "noteheadHalf": Fraction(2), "noteheadFull": Fraction(1)}
REST_VALUE = {"restWhole": Fraction(4), "restHalf": Fraction(2), "restQuarter": Fraction(1), "rest8th": Fraction(1, 2),
              "rest16th": Fraction(1, 4), "rest32nd": Fraction(1, 8), "rest64th": Fraction(1, 16)}
FLAG_STROKES = {"flag8thUp": 1, "flag8thDown": 1, "flag16thUp": 2, "flag16thDown": 2,
                "flag32ndUp": 3, "flag32ndDown": 3, "flag64thUp": 4, "flag64thDown": 4}
TUPLET_RATIO = {"numeral3": Fraction(2, 3), "numeral5": Fraction(4, 5), "numeral6": Fraction(4, 6), "numeral7": Fraction(4, 7)}
NUMERAL_VALUE = {f"numeral{d}": d for d in range(10)}
# Symbols whose duration this reading cannot place: the whole measure goes unscored.
UNREADABLE = {"multiMeasureRest"}


@dataclass
class Item:
    """A chord (heads sharing a stem, or stacked stemless heads) or a rest."""
    x: int
    duration: Fraction
    stem_dir: int = 0          # +1 stem up, -1 down, 0 none (rests, whole notes)
    midis: list[int] = field(default_factory=list)
    readable: bool = True


@dataclass
class TruthMeasure:
    events: list[NoteEvent]
    scored: bool


class _Graph:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.by_id = {n.id: n for n in nodes}

    def linked(self, node: Node, classes) -> list[Node]:
        return [self.by_id[o] for o in node.outlinks if self.by_id[o].class_name in classes]

    def duration(self, node: Node) -> Fraction:
        if node.class_name in REST_VALUE:
            d = REST_VALUE[node.class_name]
        else:
            d = HEAD_VALUE[node.class_name]
            if node.class_name == "noteheadFull":
                strokes = len(self.linked(node, {"beam"})) + sum(FLAG_STROKES[f.class_name] for f in self.linked(node, FLAG_STROKES))
                d /= 2 ** strokes
        d *= 2 - Fraction(1, 2 ** len(self.linked(node, {"augmentationDot"})))
        for tuplet in self.linked(node, {"tuple"}):
            for numeral in self.linked(tuplet, TUPLET_RATIO):
                d *= TUPLET_RATIO[numeral.class_name]
        return d


def _load_without_ties(annotation: Path) -> list[Node]:
    # mung2midi's pitch inference rejects ties over three notes; pitch does not need ties.
    nodes = [n for n in read_nodes_from_file(str(annotation)) if n.class_name not in ("tie", "slur")]
    kept = {n.id for n in nodes}
    for n in nodes:
        n.outlinks = [o for o in n.outlinks if o in kept]
        n.inlinks = [i for i in n.inlinks if i in kept]
    return nodes


def _pitches(nodes: list[Node]) -> dict[int, int]:
    """MIDI pitch by notehead id. The engine logs and prints freely; silenced."""
    logging.disable(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return PitchInferenceEngine().infer_pitches(nodes)
    finally:
        logging.disable(logging.NOTSET)


def _systems(graph: _Graph) -> list[list[Node]]:
    """Staves grouped into systems (joined by a measure separator), top to bottom."""
    staves = {n.id: n for n in graph.nodes if n.class_name == "staff"}
    parent = {s: s for s in staves}

    def root(s: int) -> int:
        while parent[s] != s:
            s = parent[s]
        return s

    for sep in (n for n in graph.nodes if n.class_name == "measureSeparator"):
        joined = [o for o in sep.outlinks if o in staves]
        for s in joined[1:]:
            parent[root(s)] = root(joined[0])
    groups = defaultdict(list)
    for s in staves.values():
        groups[root(s.id)].append(s)
    systems = [sorted(g, key=lambda s: s.top) for g in groups.values()]
    return sorted(systems, key=lambda g: g[0].top)


def _staff_of(graph: _Graph, node: Node) -> int | None:
    staff = graph.linked(node, {"staff"})
    return staff[0].id if staff else None


def _stemless_chords(heads: list[Node]) -> list[list[Node]]:
    """Stemless heads of one staff grouped into chords by horizontal overlap."""
    chords: list[list[Node]] = []
    for h in sorted(heads, key=lambda h: h.left):
        if chords and h.left < max(c.right for c in chords[-1]):
            chords[-1].append(h)
        else:
            chords.append([h])
    return chords


def _items(graph: _Graph, staff_ids: set[int], pitches: dict[int, int]) -> dict[int, list[Item]]:
    """Items by staff id, for the staves of one system."""
    by_stem: dict[tuple[int, int], list[Node]] = defaultdict(list)
    stemless: dict[int, list[Node]] = defaultdict(list)
    items: dict[int, list[Item]] = defaultdict(list)
    for n in graph.nodes:
        staff = _staff_of(graph, n)
        if staff not in staff_ids:
            continue
        if n.class_name in HEAD_VALUE:
            stems = graph.linked(n, {"stem"})
            if stems:
                by_stem[(stems[0].id, staff)].append(n)
            else:
                stemless[staff].append(n)
            if len(stems) > 1:
                items[staff].append(Item(n.left, Fraction(0), readable=False))
        elif n.class_name in REST_VALUE:
            items[staff].append(Item(n.left, graph.duration(n)))
        elif n.class_name in UNREADABLE:
            items[staff].append(Item(n.left, Fraction(0), readable=False))

    def chord_item(heads: list[Node], stem_dir: int) -> Item:
        durations = {graph.duration(h) for h in heads}
        return Item(min(h.left for h in heads), min(durations), stem_dir, [pitches.get(h.id) for h in heads],
                    readable=len(durations) == 1 and all(h.id in pitches for h in heads))

    for (stem_id, staff), heads in by_stem.items():
        stem = graph.by_id[stem_id]
        items[staff].append(chord_item(heads, 1 if stem.top < min(h.top for h in heads) else -1))
    for staff, heads in stemless.items():
        items[staff].extend(chord_item(chord, 0) for chord in _stemless_chords(heads))
    return items


def _chains(items: list[Item], voices: bool) -> list[list[Item]]:
    ordered = sorted(items, key=lambda i: i.x)
    if not voices:
        return [ordered]
    return [[i for i in ordered if i.stem_dir >= 0], [i for i in ordered if i.stem_dir <= 0]]


def _length(chain: list[Item]) -> Fraction:
    return sum((i.duration for i in chain), Fraction(0))


def _fills(items: list[Item], length: Fraction) -> list[list[Item]] | None:
    """The staff's chains if they all end at `length`: one chain, else one per voice."""
    if len(items) == 1 and items[0].duration == REST_VALUE["restWhole"]:
        return [items]  # a whole rest fills any measure
    for voices in (False, True):
        chains = _chains(items, voices)
        if all(_length(c) == length for c in chains if c):
            return chains
    return None


def _measure(staff_items: dict[int, list[Item]], staff_number: dict[int, int], measure: int,
             lengths: list[Fraction]) -> TruthMeasure:
    """The measure's notes, scored if every staff fills the first of `lengths` it can."""
    readable = all(i.readable for items in staff_items.values() for i in items)
    for length in lengths if readable else []:
        chains = {s: _fills(items, length) for s, items in staff_items.items()}
        if all(c is not None for c in chains.values()):
            events = []
            for s, staff_chains in chains.items():
                onset_of: dict[int, Fraction] = {}
                for chain in staff_chains:
                    onset = Fraction(0)
                    for item in chain:
                        onset_of.setdefault(id(item), onset)
                        onset += item.duration
                for item in staff_items[s]:
                    events += [NoteEvent(measure, onset_of[id(item)], m, item.duration, staff_number[s]) for m in item.midis]
            return TruthMeasure(events, scored=True)
    events = [NoteEvent(measure, Fraction(0), m, item.duration, staff_number[s])
              for s, items in staff_items.items() for item in items for m in item.midis if m is not None]
    return TruthMeasure(events, scored=False)


def _time_signature_length(graph: _Graph, signature: Node) -> Fraction | None:
    """Measure length in quarters of a timeSignature node: two numerals, top over bottom,
    or common/cut time."""
    symbols = graph.linked(signature, {"timeSigCommon", "timeSigCutCommon", *NUMERAL_VALUE})
    if any(n.class_name == "timeSigCommon" for n in symbols):
        return Fraction(4)
    if any(n.class_name == "timeSigCutCommon" for n in symbols):
        return Fraction(4)
    numerals = sorted(symbols, key=lambda n: n.top)
    if len(numerals) != 2:
        return None
    beats, beat_type = (NUMERAL_VALUE[n.class_name] for n in numerals)
    return Fraction(4 * beats, beat_type)


@dataclass
class _RawMeasure:
    staff_items: dict[int, list[Item]]
    staff_number: dict[int, int]
    time_signature: Fraction | None  # measure length set by a time signature drawn in it


def _raw_measures(graph: _Graph, pitches: dict[int, int]) -> list[_RawMeasure]:
    separators = [n for n in graph.nodes if n.class_name == "measureSeparator"]
    signatures = [n for n in graph.nodes if n.class_name == "timeSignature"]
    out = []
    for staves in _systems(graph):
        staff_ids = {s.id for s in staves}
        bounds = sorted({sep.left for sep in separators if staff_ids & set(sep.outlinks)})
        index = lambda x: sum(b <= x for b in bounds)
        by_measure: dict[int, dict[int, list[Item]]] = defaultdict(lambda: defaultdict(list))
        for staff, items in _items(graph, staff_ids, pitches).items():
            for item in items:
                by_measure[index(item.x)][staff].append(item)
        signature_at = {index(n.left): _time_signature_length(graph, n) for n in signatures
                        if _staff_of(graph, n) in staff_ids}
        # A separator at the system's start leaves an empty measure before it; skipped.
        for k in sorted(by_measure):
            out.append(_RawMeasure(by_measure[k], {s.id: n for n, s in enumerate(staves, 1)}, signature_at.get(k)))
    return out


def _possible_lengths(items: list[Item]) -> set[Fraction]:
    """Lengths the staff's items can fill: as one chain, or as two voices that agree."""
    out = set()
    for voices in (False, True):
        totals = {_length(c) for c in _chains(items, voices) if c}
        if len(totals) == 1:
            out |= totals
    return out


def _most_common_length(measures: list[_RawMeasure]) -> Fraction | None:
    """The length most staves can fill: the meter of a page that starts without a time
    signature."""
    lengths = Counter(n for m in measures for items in m.staff_items.values() for n in _possible_lengths(items))
    return lengths.most_common(1)[0][0] if lengths else None


def page_truth(annotation: Path) -> list[TruthMeasure]:
    graph = _Graph(_load_without_ties(annotation))
    raw = _raw_measures(graph, _pitches(graph.nodes))
    meter = _most_common_length(raw)
    measures = []
    for k, m in enumerate(raw):
        meter = m.time_signature or meter
        lengths = [meter]
        if k in (0, len(raw) - 1):
            # A pickup (first) or its completion (last) is short: any length all staves share.
            lengths += sorted(set().union(*map(_possible_lengths, m.staff_items.values())), reverse=True)
        measures.append(_measure(m.staff_items, m.staff_number, k, lengths))
    return measures
