"""Which source scores are held out from training, for validation and the benchmarks.

A score is identified by its file name and size, not its path, so every checkout and
data location (NW_DATA) holds out the same scores. The size also tells apart files that
share a name (every ASAP score is xml_score.musicxml).
"""
from __future__ import annotations

import zlib
from pathlib import Path

HELD_OUT_PER_MILLE = 50


def score_key(source: Path) -> str:
    return f"{source.name}:{source.stat().st_size}"


def is_held_out(source: Path) -> bool:
    return zlib.crc32(score_key(source).encode()) % 1000 < HELD_OUT_PER_MILLE
