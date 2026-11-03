"""Break recognition errors down by kind, over a directory of benchmark outputs.

usage: diagnose.py BENCH_DIR SCORES_LIST [--count N]

Pairs each BENCH_DIR/NNN.musicxml with its source score (same order and selection as
synthetic_benchmark.py). Measures are aligned by edit distance first, so a missed or
spurious barline costs one measure instead of shifting every later one. Within aligned
measures, notes are matched greedily from the most to the least specific agreement, and
each leftover mismatch is named by what differs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from score_events import NoteEvent, parse_musicxml
from synthetic_benchmark import is_validation


def by_measure(events: list[NoteEvent]) -> list[list[NoteEvent]]:
    n = max((e.measure for e in events), default=-1) + 1
    out: list[list[NoteEvent]] = [[] for _ in range(n)]
    for e in events:
        out[e.measure].append(e)
    return out


def measure_cost(a: list[NoteEvent], b: list[NoteEvent]) -> float:
    """1 - pitch-set F1, a cheap similarity for aligning measures."""
    ca, cb = Counter(e.midi for e in a), Counter(e.midi for e in b)
    if not ca and not cb:
        return 0.0
    return 1 - 2 * sum((ca & cb).values()) / (sum(ca.values()) + sum(cb.values()))


def align_measures(gt: list[list[NoteEvent]], pred: list[list[NoteEvent]]) -> list[tuple[int | None, int | None]]:
    n, m = len(gt), len(pred)
    gap = 0.9
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * gap
    for j in range(1, m + 1):
        cost[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + measure_cost(gt[i - 1], pred[j - 1]),
                             cost[i - 1][j] + gap, cost[i][j - 1] + gap)
    pairs = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + measure_cost(gt[i - 1], pred[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + gap:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


# Leftover pairs are matched by the first rule they satisfy; the label says what differs.
MISMATCH_RULES = [
    ("duration", lambda g, p: g.onset == p.onset and g.midi == p.midi),
    ("onset", lambda g, p: g.midi == p.midi and g.duration == p.duration),
    ("onset+duration", lambda g, p: g.midi == p.midi),
    ("octave", lambda g, p: g.onset == p.onset and abs(g.midi - p.midi) == 12),
    ("semitone", lambda g, p: g.onset == p.onset and abs(g.midi - p.midi) in (1, 2) and g.duration == p.duration),
    ("pitch", lambda g, p: g.onset == p.onset and g.duration == p.duration),
]


def classify_measure(gt: list[NoteEvent], pred: list[NoteEvent], tally: Counter) -> None:
    key = lambda e: (e.onset, e.midi, e.duration)
    exact = Counter(map(key, gt)) & Counter(map(key, pred))
    tally["correct"] += sum(exact.values())
    left_gt, left_pred = [], []
    for events, left in ((gt, left_gt), (pred, left_pred)):
        budget = Counter(exact)
        for e in events:
            if budget[key(e)]:
                budget[key(e)] -= 1
            else:
                left.append(e)
    for label, rule in MISMATCH_RULES:
        for g in list(left_gt):
            match = next((p for p in left_pred if rule(g, p)), None)
            if match is not None:
                left_gt.remove(g)
                left_pred.remove(match)
                tally[label] += 1
    tally["missed"] += len(left_gt)
    tally["extra"] += len(left_pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("scores")
    ap.add_argument("--count", type=int, default=100)
    args = ap.parse_args()

    scores = [Path(l.strip()) for l in Path(args.scores).read_text().splitlines() if l.strip() and is_validation(l.strip())]
    total = Counter()
    worst = []
    for i, score in enumerate(scores[: args.count]):
        xml = Path(args.bench_dir) / f"{i:03d}.musicxml"
        if not xml.exists():
            continue
        pred = by_measure(parse_musicxml(xml))
        gt = by_measure([e for e in parse_musicxml(score) if e.measure < len(pred) + 8])
        # Only the measures shown on the rendered page count; trust the pred's extent,
        # padded so missed trailing measures still register.
        tally = Counter()
        pairs = align_measures(gt, pred)
        last_pred = max((j for _, j in pairs if j is not None), default=-1)
        for gi, pj in pairs:
            if gi is None:
                tally["extra measure"] += 1
                tally["extra"] += len(pred[pj])
            elif pj is None:
                if gi < last_pred + 1:  # beyond the page otherwise
                    tally["missed measure"] += 1
                    tally["missed"] += len(gt[gi])
            else:
                classify_measure(gt[gi], pred[pj], tally)
        total += tally
        n_gt = sum(v for k, v in tally.items() if k not in ("extra", "extra measure"))
        worst.append((tally["correct"] / max(1, n_gt), i, score.name, tally))
    notes = sum(v for k, v in total.items() if "measure" not in k and k != "extra")
    print(f"aligned note accuracy {total['correct'] / max(1, notes):.3f} over {notes} gt notes")
    for k, v in total.most_common():
        print(f"  {k:16s} {v}")
    print("worst scores:")
    for acc, i, name, tally in sorted(worst)[:12]:
        errs = ", ".join(f"{k}={v}" for k, v in tally.most_common() if k != "correct")
        print(f"  {i:03d} acc={acc:.2f} {name}: {errs}")


if __name__ == "__main__":
    main()
