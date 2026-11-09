"""Barlines in LMX.

OLiMPiC's LMX has no barlines: its linearizer skips them, so a model trained on it cannot
write the repeat signs, double bars and final bars a page prints. Our labels add a bar's
opening sign straight after its `measure` token (`repeat:start`) and its closing sign at its
end (`repeat:end`, `barline:double`, `barline:final`). OLiMPiC's delinearizer does not know
them, so they are taken out before it runs and drawn as <barline> elements afterwards.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

REPEAT_START, REPEAT_END = "repeat:start", "repeat:end"
DOUBLE, FINAL = "barline:double", "barline:final"
# token: (location, bar-style, repeat direction)
BARLINES = {REPEAT_START: ("left", "heavy-light", "forward"), REPEAT_END: ("right", "light-heavy", "backward"),
            DOUBLE: ("right", "light-light", None), FINAL: ("right", "light-heavy", None)}


def token_of(barline: ET.Element) -> str | None:
    """A backward repeat is drawn in the final bar's style, so the repeat decides."""
    location = barline.get("location", "right")
    repeat = barline.find("repeat")
    if repeat is not None:
        direction = repeat.get("direction")
        return {("left", "forward"): REPEAT_START, ("right", "backward"): REPEAT_END}.get((location, direction))
    if location == "right":
        return {"light-light": DOUBLE, "light-heavy": FINAL}.get(barline.findtext("bar-style"))
    return None


def barline_tokens_of(measure: ET.Element) -> tuple[list[str], list[str]]:
    """The measure's opening and closing barline tokens, at most one of each."""
    tokens = [token_of(barline) for barline in measure.findall("barline")]
    opening = [REPEAT_START] if REPEAT_START in tokens else []
    closing = next(([token] for token in (REPEAT_END, DOUBLE, FINAL) if token in tokens), [])
    return opening, closing


def with_barline_tokens(tokens: list[str], opening: list[str], closing: list[str]) -> list[str]:
    """A measure's tokens, from its `measure` token on, with its barline tokens in place."""
    return tokens[:1] + opening + tokens[1:] + closing


def without_barlines(lmx: str) -> tuple[str, list[tuple[int, str]]]:
    """The LMX without barline tokens, and each one's measure (counted from 0) and token."""
    kept, barlines, measure = [], [], -1
    for token in lmx.split():
        if token == "measure":
            measure += 1
        if token in BARLINES:
            barlines.append((max(measure, 0), token))
        else:
            kept.append(token)
    return " ".join(kept), barlines


def barline_element(token: str) -> ET.Element:
    location, style, direction = BARLINES[token]
    barline = ET.Element("barline", location=location)
    ET.SubElement(barline, "bar-style").text = style
    if direction:
        ET.SubElement(barline, "repeat", direction=direction)
    return barline


def add_barlines(part: ET.Element, barlines: list[tuple[int, str]]) -> None:
    """Draws each barline token on its measure: an opening one first in the bar, a closing one
    last. A measure the delinearizer did not produce takes none."""
    measures = part.findall("measure")
    for index, token in barlines:
        if index < len(measures):
            element = barline_element(token)
            if element.get("location") == "left":
                measures[index].insert(0, element)
            else:
                measures[index].append(element)


def kern_barline_tokens(barline: str) -> tuple[list[str], list[str]]:
    """A **kern barline's tokens for the measure it closes and the one it opens: `:|` a backward
    repeat (which decides over the style, as in token_of), `==` a final bar, `||` a double one,
    `|:` a forward repeat."""
    closing = [REPEAT_END] if ":|" in barline else [FINAL] if "==" in barline else [DOUBLE] if "||" in barline else []
    return ([REPEAT_START] if "|:" in barline else []), closing


def kern_measure_barlines(kern: str) -> list[tuple[list[str], list[str]]]:
    """Each measure's opening and closing barline tokens, in the order of the file's measures."""
    measures: list[tuple[list[str], list[str]]] = []
    opening: list[str] = []
    in_measure = False
    for line in kern.split("\n"):
        if line.startswith("="):
            next_opening, closing = kern_barline_tokens(line.split("\t")[0])
            if in_measure:
                measures[-1][1].extend(closing)
            opening, in_measure = next_opening, False
        elif line and not line.startswith(("*", "!")) and not in_measure:
            measures.append((opening, []))
            in_measure = True
    return measures


def with_kern_barline_tokens(lmx: str, kern: str) -> str | None:
    """The LMX with the barline tokens its **kern source draws; None if there are some but the two
    do not have the same number of measures, so they cannot be matched up."""
    barlines = kern_measure_barlines(kern)
    if not any(opening or closing for opening, closing in barlines):
        return lmx
    tokens = lmx.split()
    starts = [i for i, token in enumerate(tokens) if token == "measure"]
    if len(starts) != len(barlines):
        return None
    out = tokens[:starts[0]]
    for (start, end), (opening, closing) in zip(zip(starts, starts[1:] + [len(tokens)]), barlines):
        out += with_barline_tokens(tokens[start:end], opening, closing)
    return " ".join(out)
