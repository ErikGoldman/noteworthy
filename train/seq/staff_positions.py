"""A system's staff positions, and how many staves its label writes on: what the crop builders
record and find_staff_positions.py reads, kept free of torch for the builders' environments
(Smashcima's has none)."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Staves:
    centres: tuple[float, ...]  # rows, top to bottom
    interline: float


def staff_count(lmx: str) -> int:
    """The staves an LMX system writes on: its highest staff:N."""
    return max(int(n) for n in re.findall(r"\bstaff:(\d+)", lmx))
