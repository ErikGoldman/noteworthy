"""Repeat barlines in LMX.

OLiMPiC's LMX has no barlines: its linearizer skips them, so a model trained on it cannot
write the repeat signs a page prints. Our labels add `repeat:start` straight after the
`measure` token of a bar that opens with a forward repeat, and `repeat:end` at the end of a
bar that closes with a backward one. OLiMPiC's delinearizer does not know them, so they are
taken out before it runs and drawn as <barline> elements afterwards.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

REPEAT_START, REPEAT_END = "repeat:start", "repeat:end"
REPEAT_TOKENS = (REPEAT_START, REPEAT_END)


def repeats_of(measure: ET.Element) -> tuple[bool, bool]:
    """Whether the measure opens with a forward repeat and closes with a backward one."""
    def repeat(location: str, direction: str) -> bool:
        return any(barline.get("location", "right") == location and barline.find(f"repeat[@direction='{direction}']") is not None
                   for barline in measure.findall("barline"))
    return repeat("left", "forward"), repeat("right", "backward")


def with_repeat_tokens(tokens: list[str], opens: bool, closes: bool) -> list[str]:
    """A measure's tokens, from its `measure` token on, with its repeat tokens in place."""
    return tokens[:1] + [REPEAT_START] * opens + tokens[1:] + [REPEAT_END] * closes


def without_repeats(lmx: str) -> tuple[str, list[tuple[int, str]]]:
    """The LMX without repeat tokens, and each one's measure (counted from 0) and token."""
    kept, repeats, measure = [], [], -1
    for token in lmx.split():
        if token == "measure":
            measure += 1
        if token in REPEAT_TOKENS:
            repeats.append((max(measure, 0), token))
        else:
            kept.append(token)
    return " ".join(kept), repeats


def repeat_barline(token: str) -> ET.Element:
    start = token == REPEAT_START
    barline = ET.Element("barline", location="left" if start else "right")
    ET.SubElement(barline, "bar-style").text = "heavy-light" if start else "light-heavy"
    ET.SubElement(barline, "repeat", direction="forward" if start else "backward")
    return barline


def add_repeat_barlines(part: ET.Element, repeats: list[tuple[int, str]]) -> None:
    """Draws each repeat token as a barline on its measure: a forward repeat first in the bar,
    a backward one last. A measure the delinearizer did not produce takes none."""
    measures = part.findall("measure")
    for index, token in repeats:
        if index < len(measures):
            if token == REPEAT_START:
                measures[index].insert(0, repeat_barline(token))
            else:
                measures[index].append(repeat_barline(token))
