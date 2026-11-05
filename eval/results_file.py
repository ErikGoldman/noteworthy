"""The per-item tallies a benchmark leaves for scorecard.py."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path


def save_results(results: dict[str, Counter], path: Path, piece_of: Callable[[str], str]) -> None:
    """Each item's tally with its piece, for scorecard.py: items of one piece are not
    independent, so the scorecard resamples whole pieces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({item: {"piece": piece_of(item), "tally": t} for item, t in results.items()}, indent=0))
