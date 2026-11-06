"""Divides Polish into a dev half, whose errors may be read, and a test half that stays sealed.

usage: polish_split.py GOLD_DIR OUT.txt

Writes "<dev|test> <page>" lines. The dataset names no pieces and its pages are shuffled, so
pages that may share a piece are found from the gold: those sharing SHARED_GRAMS melodic
fragments stay on one side. Nothing here reads a system's output or score: the split must not
depend on what is hard for anyone.
"""
from __future__ import annotations

import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
from pathlib import Path

GRAM_INTERVALS = 8
SHARED_GRAMS = 2
SEED = "polish-split"
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def part_pitches(part: ET.Element) -> list[int]:
    pitches = []
    for note in part.iter("note"):
        pitch = note.find("pitch")
        if pitch is not None and note.find("chord") is None:
            alter = float(pitch.findtext("alter", "0"))
            pitches.append(12 * int(pitch.findtext("octave")) + STEPS[pitch.findtext("step")] + int(alter))
    return pitches


def melody_grams(gold: Path) -> set[tuple[int, ...]]:
    """Transposition-free fragments of each part; a repeated note says little, so they must move."""
    grams = set()
    for part in ET.parse(gold).getroot().iter("part"):
        pitches = part_pitches(part)
        steps = [b - a for a, b in zip(pitches, pitches[1:])]
        for i in range(len(steps) - GRAM_INTERVALS + 1):
            gram = tuple(steps[i:i + GRAM_INTERVALS])
            if len(set(gram)) >= 3:
                grams.add(gram)
    return grams


def groups_sharing_grams(grams: dict[str, set]) -> list[list[str]]:
    parent = {page: page for page in grams}

    def root(page: str) -> str:
        while parent[page] != page:
            parent[page] = parent[parent[page]]
            page = parent[page]
        return page

    for a, b in combinations(sorted(grams), 2):
        if len(grams[a] & grams[b]) >= SHARED_GRAMS:
            parent[root(a)] = root(b)
    groups = defaultdict(list)
    for page in grams:
        groups[root(page)].append(page)
    return sorted(map(sorted, groups.values()))


def split(groups: list[list[str]]) -> dict[str, str]:
    """Each group, largest first, to the smaller half; ties and order fall to the seed."""
    rng = random.Random(SEED)
    rng.shuffle(groups)
    halves: dict[str, list[str]] = {"dev": [], "test": []}
    for group in sorted(groups, key=len, reverse=True):
        names = sorted(halves, key=lambda half: (len(halves[half]), rng.random()))
        halves[names[0]] += group
    return {page: half for half, pages in halves.items() for page in pages}


def main() -> None:
    gold, out = Path(sys.argv[1]), Path(sys.argv[2])
    grams = {path.stem: melody_grams(path) for path in sorted(gold.glob("*.musicxml"))}
    groups = groups_sharing_grams(grams)
    assignment = split(groups)
    out.write_text("".join(f"{assignment[page]} {page}\n" for page in sorted(assignment)))
    sizes = sorted(map(len, groups), reverse=True)
    print(f"{len(groups)} groups, largest {sizes[:8]}; "
          f"dev {sum(h == 'dev' for h in assignment.values())}, test {sum(h == 'test' for h in assignment.values())}")


if __name__ == "__main__":
    main()
