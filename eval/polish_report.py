"""Polish OMR-NED by system and gold standard, as docs/EVALUATION_PROTOCOL.md reports them: the
mean over two-staff pages, three-staff pages and all, and each other system against ours, paired
by page with a 95% range from 5,000 resamples of pages (a difference counts only if its range
excludes zero; lower is better, so a negative change means ours is ahead).

usage: polish_report.py RESULTS_DIR GOLD_DIR

RESULTS_DIR holds <system>.<gold>.<alignment>.json from omr_ned.py, with "ours" among the systems;
GOLD_DIR the split's gold with its **kern header (Corrected or Print), whose spines give each
page's staff count.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

RESAMPLES = 5000


def staff_count(kern: Path) -> int:
    return kern.read_text().split("\n", 1)[0].split("\t").count("**kern")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired_range(differences: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(mean([rng.choice(differences) for _ in differences]) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES) - 1], means[int(0.975 * RESAMPLES) - 1]


def main() -> None:
    results, gold = Path(sys.argv[1]), Path(sys.argv[2])
    staves = {kern.stem: staff_count(kern) for kern in gold.glob("*.krn")}
    groups = {"two staves": [p for p, n in staves.items() if n == 2],
              "three staves": [p for p, n in staves.items() if n == 3],
              "all": sorted(staves)}
    scores: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for path in sorted(results.glob("*.*.*.json")):
        system, standard, alignment = path.stem.split(".")
        scores[standard, alignment][system] = json.loads(path.read_text())["scores"]
    for (standard, alignment), by_system in sorted(scores.items()):
        print(f"== {standard} gold, {alignment} alignment ({len(staves)} pages)")
        print(f"{'system':<14}" + "".join(f"{f'{name} ({len(pages)})':>20}" for name, pages in groups.items()))
        for system, page_scores in sorted(by_system.items(), key=lambda kv: kv[0] != "ours"):
            print(f"{system:<14}" + "".join(f"{mean([page_scores[p] for p in pages]):>20.2f}" if pages else f"{'':>20}"
                                            for pages in groups.values()))
        ours = by_system.get("ours")
        for system, page_scores in by_system.items():
            if ours is None or system == "ours":
                continue
            differences = [ours[p] - page_scores[p] for p in groups["all"]]
            low, high = paired_range(differences)
            print(f"  ours - {system}: {mean(differences):+.2f} [{low:+.2f}, {high:+.2f}]")


if __name__ == "__main__":
    main()
