"""Compare recognized MusicXML against hand-written ground truth.

usage: evaluate.py GROUND_TRUTH.gt PREDICTED.musicxml [--verbose]

A predicted note matches a ground-truth note when measure, onset, pitch and
duration all agree. Staff is reported separately: cross-staff writing makes it
a notational choice rather than a recognition error.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from score_events import NoteEvent, parse_gt, parse_musicxml


def _key(e: NoteEvent):
    return (e.measure, e.onset, e.midi, e.duration)


def _f1(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _matches(gt: list[NoteEvent], pred: list[NoteEvent], key) -> int:
    return sum((Counter(map(key, gt)) & Counter(map(key, pred))).values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth")
    ap.add_argument("predicted")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gt = parse_gt(args.ground_truth).events()
    pred = parse_musicxml(args.predicted)

    print(f"ground truth notes: {len(gt)}   predicted notes: {len(pred)}")
    for label, key in [
        ("full (onset+pitch+duration)", _key),
        ("onset+pitch", lambda e: (e.measure, e.onset, e.midi)),
        ("pitch only, per measure", lambda e: (e.measure, e.midi)),
    ]:
        p, r, f = _f1(_matches(gt, pred, key), len(pred), len(gt))
        print(f"  {label:30s} P={p:.3f} R={r:.3f} F1={f:.3f}")
    staff_ok = _matches(gt, pred, lambda e: (*_key(e), e.staff))
    print(f"  full matches also on the right staff: {staff_ok}/{_matches(gt, pred, _key)}")

    by_measure_gt, by_measure_pred = defaultdict(list), defaultdict(list)
    for e in gt:
        by_measure_gt[e.measure].append(e)
    for e in pred:
        by_measure_pred[e.measure].append(e)
    measures = sorted(set(by_measure_gt) | set(by_measure_pred))
    exact = [m for m in measures if Counter(map(_key, by_measure_gt[m])) == Counter(map(_key, by_measure_pred[m]))]
    print(f"  measures exactly right: {len(exact)}/{len(measures)}")

    if args.verbose:
        for m in measures:
            g, p = Counter(map(_key, by_measure_gt[m])), Counter(map(_key, by_measure_pred[m]))
            missing, extra = g - p, p - g
            if missing or extra:
                fmt = lambda c: " ".join(f"{k[2]}@{k[1]}:{k[3]}" for k in sorted(c.elements()))
                print(f"  m{m}: missing [{fmt(missing)}] extra [{fmt(extra)}]")


if __name__ == "__main__":
    main()
