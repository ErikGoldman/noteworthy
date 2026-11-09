"""Tuplets whose number the page does not print, in LMX.

OLiMPiC's LMX marks a tuplet's start and stop but not whether its number is drawn, so a model
trained on it writes every 3 shown, though engravers stop printing them once a passage has
settled into triplets. Our labels add `tuplet:hidden` straight after the `tuplet:start` of a
group whose number is not shown. OLiMPiC's delinearizer does not know the token, so it is taken
out before it runs and set on the <tuplet> starts afterwards, which the delinearizer writes one
per `tuplet:start`, in order.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

START, HIDDEN = "tuplet:start", "tuplet:hidden"


def starts_hidden_tuplet(note: ET.Element) -> bool:
    return any(tuplet.get("type") == "start" and tuplet.get("show-number") == "none"
               for tuplet in note.findall("notations/tuplet"))


def without_hidden_numbers(lmx: str) -> tuple[str, list[int]]:
    """The LMX without the token, and which tuplet starts (counted from 0) it followed."""
    kept, hidden, starts = [], [], -1
    for token in lmx.split():
        if token == START:
            starts += 1
        if token == HIDDEN:
            if starts >= 0 and kept[-1] == START:
                hidden.append(starts)
        else:
            kept.append(token)
    return " ".join(kept), hidden


def hide_numbers(part: ET.Element, hidden: list[int]) -> None:
    starts = [tuplet for tuplet in part.iter("tuplet") if tuplet.get("type") == "start"]
    for index in hidden:
        if index < len(starts):
            starts[index].set("show-number", "none")
