"""Bars read right, beside OMR-NED (docs/EVAL_SETS.md, spec 3), on Polish: a gold staff-bar is right
when the whole-page alignment charges it nothing.

usage: bars_right.py GOLD_DIR COSTS.jsonl [...] [--out PAGES.json]

COSTS.jsonl is ned_costs.py --whole-page's output for one run against GOLD_DIR's kern gold. The
bars are those musicdiff compares (its "bar" indices and each page's "gold_bars"): it leaves out a
bar with nothing it compares, a staff of hidden rests, which no reading can get right or wrong
(Polish 051 has two such on its upper staff). An edit naming a gold bar charges it. One naming only the reading's bar (a note or bar the reading has
and the gold lacks) charges the gold bar that reading bar is aligned to: the alignment is walked
from the whole-bar edits, a gold bar the reading lacks (insbar, named by its gold index) skipped,
a reading bar the gold lacks (delbar) charged to the gold bar before it. Edits naming both sides
check the walk; disagreements are counted and printed, never silently dropped.

Per run: the share of gold staff-bars right, per page, mean over pages, and OMR-NED from the same
file (so both measures rest on one scoring); then by what the gold bar holds, pooled over pages: a
hidden tuplet, a printed tuplet, two or more voices, a chord of three or more notes, none of
these (a bar holding two counts under both). Run with .venv-musicdiff."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from omr_ned import authors_metric, parsed_as_authors_do

CLASSES = ["hidden tuplet", "printed tuplet", "two voices", "chord of 3+", "none of these"]


def bar_classes(measure) -> set[str]:
    held = set()
    for note in measure.recurse().notes:
        for tuplet in note.duration.tuplets:
            if tuplet.type == "start":
                held.add("hidden tuplet" if tuplet.tupletActualShow is None else "printed tuplet")
        if note.isChord and len(note.pitches) >= 3:
            held.add("chord of 3+")
    if sum(1 for voice in measure.voices if voice.notes) >= 2:
        held.add("two voices")
    return held or {"none of these"}


def gold_bars(metric, gold: Path, compared: list[list[int]]) -> list[list[set[str]]]:
    """Each staff's compared bars (upper first, as the scorer orders parts), each as its classes."""
    score = parsed_as_authors_do(metric, gold, prediction=False)
    measures = [list(part.getElementsByClass("Measure")) for part in score.parts]
    return [[bar_classes(measures[s][m]) for m in staff] for s, staff in enumerate(compared)]


def reading_to_gold(gold_count: int, missing: set[int], extra: set[int]) -> dict[int, int]:
    """A staff's reading bar index -> the gold bar it is charged to."""
    mapping, reading, gold = {}, 0, 0
    while gold < gold_count or reading in extra:
        if reading in extra:
            mapping[reading] = max(gold - 1, 0)
            reading += 1
        elif gold in missing:
            gold += 1
        else:
            mapping[reading] = gold
            reading += 1
            gold += 1
    return mapping


def charged_bars(page: dict, counts: list[int]) -> tuple[set[tuple[int, int]], int]:
    """The gold (staff, bar) the page's edits charge, and how many edits naming both sides the walk
    places elsewhere."""
    missing, extra = defaultdict(set), defaultdict(set)
    for edit in page["edits"]:
        if edit["op"] == "insbar":
            missing[edit["gold"]["part"]].add(edit["gold"]["bar"])
        elif edit["op"] == "delbar":
            extra[edit["pred"]["part"]].add(edit["pred"]["bar"])
    walks = {part: reading_to_gold(count, missing[part], extra[part]) for part, count in enumerate(counts)}
    charged, disagreements = set(), 0
    for edit in page["edits"]:
        gold, pred = edit.get("gold"), edit.get("pred")
        if gold:
            charged.add((gold["part"], gold["bar"]))
            if pred and walks[pred["part"]].get(pred["bar"]) != gold["bar"]:
                disagreements += 1
        else:
            charged.add((pred["part"], walks[pred["part"]].get(pred["bar"], counts[pred["part"]] - 1)))
    return charged, disagreements


def scored(gold_dir: Path, costs: Path, metric) -> dict:
    pages, by_class, disagreements = {}, defaultdict(lambda: [0, 0]), 0
    for page in map(json.loads, costs.read_text().splitlines()):
        if "edits" not in page:
            raise SystemExit(f"{costs.name} {page['page']}: not scored ({page.get('error')})")
        bars = gold_bars(metric, gold_dir / f"{page['page']}.krn", page["gold_bars"])
        charged, wrong_walk = charged_bars(page, [len(staff) for staff in bars])
        disagreements += wrong_walk
        total = sum(len(staff) for staff in bars)
        right = total - len({(s, b) for s, b in charged if b < len(bars[s])})
        omr_ned = 100 * (page["distance"] + page["syntax_fixed"]) / page["size"]  # distance has the refund taken off
        pages[page["page"]] = {"bars right": right / total, "omr_ned": omr_ned}
        for s, staff in enumerate(bars):
            for b, classes in enumerate(staff):
                for name in classes:
                    by_class[name][0] += (s, b) not in charged
                    by_class[name][1] += 1
    return {"pages": pages, "by_class": dict(by_class), "walk_disagreements": disagreements}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path)
    ap.add_argument("costs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    metric = authors_metric()
    results = {costs.stem: scored(args.gold, costs, metric) for costs in args.costs}
    print(f"{'run':48s} {'bars right':>10s} {'OMR-NED':>8s}  " + "  ".join(f"{c:>14s}" for c in CLASSES) + "  walk disagreements")
    for run, result in results.items():
        pages = result["pages"].values()
        mean = lambda key: sum(p[key] for p in pages) / len(pages)
        classes = "  ".join(f"{100 * r / n:13.1f}%" if n else f"{'-':>14s}"
                            for r, n in (result["by_class"].get(c, (0, 0)) for c in CLASSES))
        print(f"{run:48s} {100 * mean('bars right'):9.1f}% {mean('omr_ned'):8.2f}  {classes}  {result['walk_disagreements']}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
