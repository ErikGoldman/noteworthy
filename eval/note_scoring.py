"""Note F1 of a recognized score against ground truth, with the error kinds of diagnose.py.

Measures are aligned first (diagnose.align_measures), so a missed or extra measure costs
only its own notes. Shared by the real-scan and handwritten benchmarks.
"""
from __future__ import annotations

from collections import Counter

from diagnose import align_measures, by_measure, classify_measure
from score_events import NoteEvent

# Mismatch kinds in which the pitch was right: only the timing was wrong.
TIMING_ONLY = ("onset", "duration", "onset+duration")


def score(gt_events: list[NoteEvent], pred_events: list[NoteEvent], unscored: frozenset[int] = frozenset()) -> Counter:
    """Error tally over aligned measures, plus "pred" notes counted. Ground-truth measures
    in `unscored` still take part in the alignment, but neither they nor the predicted
    measure aligned to them are counted."""
    gt, pred = by_measure(gt_events), by_measure(pred_events)
    tally = Counter()
    for g, p in align_measures(gt, pred):
        if g in unscored:
            continue
        if g is not None and p is not None:
            classify_measure(gt[g], pred[p], tally)
            tally["pred"] += len(pred[p])
        elif g is not None:
            tally["missed measure"] += 1
            tally["missed"] += len(gt[g])
        else:
            tally["extra measure"] += 1
            tally["extra"] += len(pred[p])
            tally["pred"] += len(pred[p])
    return tally


def gt_notes(t: Counter) -> int:
    return sum(v for k, v in t.items() if "measure" not in k and k not in ("extra", "pred"))


def f1(t: Counter) -> float:
    return 2 * t["correct"] / max(1, gt_notes(t) + t["pred"])


def report(results: dict[str, Counter], what: str, show_worst: bool) -> None:
    """Totals, plus the worst items when show_worst (never on a held-out test set)."""
    total = sum(results.values(), Counter())
    right_pitch = total["correct"] + sum(total[k] for k in TIMING_ONLY)
    print(f"overall F1 = {f1(total):.3f} over {len(results)} {what}, {gt_notes(total)} gt notes "
          f"(P={total['correct'] / max(1, total['pred']):.3f} R={total['correct'] / max(1, gt_notes(total)):.3f})")
    print(f"right pitch in the right measure, timing aside: {right_pitch / max(1, gt_notes(total)):.3f}")
    for k, v in total.most_common():
        if k != "pred":
            print(f"  {k:16s} {v}")
    if not show_worst:
        return
    print("worst:")
    for name, t in sorted(results.items(), key=lambda kv: f1(kv[1]))[:10]:
        print(f"  {name} F1={f1(t):.3f} " + ", ".join(f"{k}={v}" for k, v in t.most_common() if k not in ("correct", "pred")))
