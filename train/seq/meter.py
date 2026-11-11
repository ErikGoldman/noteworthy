"""The meter in force, read by the model before a system's own tokens.

Most systems print no time signature, so a model reading one alone cannot tell an unmarked
triplet from plain notes by the bar's arithmetic. With meter context the label starts
`meter beats:N beat-type:M` (after <bos>): the bar's length in quarters, spelled with the
time-signature tokens, so 3/4 and 6/8 read alike (only the length matters for the arithmetic).
The model reads it and is never trained to write it (train_seq.py leaves it out of the loss).

In training the meter comes from the label's own bars; some labels get none, as a page's first
system has none yet. In recognition it comes from the time signatures already read on the page.
"""
from __future__ import annotations

from fractions import Fraction

from .lmx_tuplets import likely_bar, measures_of, written_bar

METER = "meter"
PREFIX_LENGTH = 3
BEAT_TYPES = (4, 8, 16, 2)
MAX_BEATS = 16
TRAINED_WITHOUT = 0.15


def meter_tokens(bar: Fraction) -> list[str]:
    """The prefix for a bar of this many quarters, or none if the vocabulary cannot spell it"""
    for beat_type in BEAT_TYPES:
        beats = bar * beat_type / 4
        if beats.denominator == 1 and 1 <= beats <= MAX_BEATS:
            return [METER, f"beats:{beats.numerator}", f"beat-type:{beat_type}"]
    return []


def bar_of(lmx: str) -> Fraction | None:
    """The bar's length a label shows: its time signature, or what its measures add up to"""
    tokens = lmx.split()
    return written_bar(tokens) or likely_bar(measures_of(tokens))


def with_meter(lmx: str, rng=None) -> str:
    """The label with its meter in front; in training (rng given) some go without"""
    bar = bar_of(lmx)
    if bar is None or rng is not None and rng.random() < TRAINED_WITHOUT:
        return lmx
    return " ".join(meter_tokens(bar) + [lmx])


def without_meter(lmx: str) -> str:
    tokens = lmx.split()
    return " ".join(tokens[PREFIX_LENGTH:] if tokens[:1] == [METER] else tokens)
