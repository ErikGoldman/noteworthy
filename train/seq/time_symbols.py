"""Time signatures drawn as symbols (common time C, cut time ¢) in LMX.

OLiMPiC's LMX spells every time signature as numbers, so a model trained on it cannot say that
a page prints C rather than 4/4. Our labels add `time:common` or `time:cut` straight after the
`time` token, its beats and beat type kept: the bar arithmetic needs them. OLiMPiC's
delinearizer does not know the tokens, so they are taken out before it runs and set on the
<time> elements afterwards.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

COMMON, CUT = "time:common", "time:cut"
SYMBOL_OF_TOKEN = {COMMON: "common", CUT: "cut"}
TOKEN_OF_SYMBOL = {symbol: token for token, symbol in SYMBOL_OF_TOKEN.items()}
KERN_MENSURATION = {"c": COMMON, "c|": CUT}


def token_of(time: ET.Element) -> str | None:
    return TOKEN_OF_SYMBOL.get(time.get("symbol"))


def with_time_symbol(tokens: list[str], token: str | None) -> list[str]:
    """Tokens ending in a time signature, with its symbol token after its `time` token."""
    if token is None:
        return tokens
    at = len(tokens) - tokens[::-1].index("time")
    return tokens[:at] + [token] + tokens[at:]


def without_time_symbols(lmx: str) -> tuple[str, list[tuple[int, str]]]:
    """The LMX without symbol tokens, and each one's time signature (counted from 0) and token."""
    kept, symbols, time = [], [], -1
    for token in lmx.split():
        if token == "time":
            time += 1
        if token in SYMBOL_OF_TOKEN:
            symbols.append((max(time, 0), token))
        else:
            kept.append(token)
    return " ".join(kept), symbols


def add_time_symbols(part: ET.Element, symbols: list[tuple[int, str]]) -> None:
    """Sets each symbol on its time signature. One the delinearizer did not produce takes none."""
    times = part.findall("measure/attributes/time")
    for index, token in symbols:
        if index < len(times):
            times[index].set("symbol", SYMBOL_OF_TOKEN[token])


def kern_time_symbols(kern: str) -> list[str | None]:
    """For each time signature line (`*M`) of a **kern file, the symbol token of the mensuration
    line (`*met(...)`) that follows it before any data, if one does."""
    symbols: list[str | None] = []
    awaiting_mensuration = False
    for line in kern.split("\n"):
        first = line.split("\t")[0]
        if re.fullmatch(r"\*M\d+/\d+", first):
            symbols.append(None)
            awaiting_mensuration = True
        elif first.startswith("*met(") and awaiting_mensuration:
            symbols[-1] = KERN_MENSURATION.get(first[len("*met("):-1])
            awaiting_mensuration = False
        elif not line.startswith(("*", "!")):
            awaiting_mensuration = False
    return symbols


def with_kern_time_symbols(lmx: str, kern: str) -> str | None:
    """The LMX with the symbols its **kern source draws; None if there are symbols but the two do
    not have the same number of time signatures, so the symbols cannot be matched up. (GrandStaff
    leaves out of the LMX many a signature its kern has.)"""
    symbols = kern_time_symbols(kern)
    tokens = lmx.split()
    if not any(symbols):
        return lmx
    if tokens.count("time") != len(symbols):
        return None
    out, time = [], 0
    for token in tokens:
        out.append(token)
        if token == "time":
            if symbols[time]:
                out.append(symbols[time])
            time += 1
    return " ".join(out)
