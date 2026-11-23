"""Where the voice being written stands in its bar, computed by code from the tokens written so far and given
to the decoder as an input (Seq2Seq's beat_clock): the model never has to add lengths itself, and it sees at
every token how much of the bar is left, negative once a voice overfills it.

The clock after each token, as BarClock (seq.label_form) runs: `measure` sets it to 0; each note, rest or
forward that advances moves it on by its length (dots and tuplet ratios, which follow the length token, count
as soon as they are written); a backup moves it back; `backup:bar` returns to 0. A chord's later notes and grace
notes take no time. The bar's length comes from the meter prefix (seq.meter, `meter beats:N beat-type:M`) or a
time signature written in the label (`time beats:N beat-type:M`); before either the bar's length is unknown.

FEATURES per position: whether the bar's length is known; whether every length passed in this bar is (a
whole-bar rest's is not); the time reached, the time left (clipped), whether the voice has overfilled the bar
and whether it stands exactly at its end; and the beat phase (sin and cos of the time in quarters), so a
triplet's thirds look unlike a plain note's halves.
"""
from __future__ import annotations

import math
from fractions import Fraction

import torch

from .label_form import BACKUP_TO_BAR, POSITION, PREFIXES, advances, length_of
from .lmx_tuplets import PITCH

FEATURES = 8
LEFT_CLIP = 4.0  # quarters: beyond a bar of 4/4 either way the time left reads as this
HEADS = {"rest", "rest:measure", "forward", "backup", BACKUP_TO_BAR}


class ClockStepper:
    """The clock of one sequence, token by token (step returns the features after each token)."""

    def __init__(self) -> None:
        self.bar: Fraction | None = None
        self.beats: int | None = None
        self.base_time, self.base_known = Fraction(0), True   # after the elements already finished
        self.element: list[str] = []
        self.previous = ""

    def step(self, token: str) -> list[float]:
        if token == "measure":
            self.base_time, self.base_known, self.element = Fraction(0), True, []
        elif self.starts_element(token):
            self.finish_element()
            self.element = [token]
        else:
            self.element.append(token)
            self.read_meter(token)
        self.previous = token
        return self.features()

    def starts_element(self, token: str) -> bool:
        heads_here = token in PREFIXES or token in HEADS or bool(PITCH.match(token)) or bool(POSITION.match(token))
        return heads_here and self.previous not in PREFIXES

    def read_meter(self, token: str) -> None:
        if token.startswith("beats:"):
            self.beats = int(token.split(":")[1])
        elif token.startswith("beat-type:") and self.beats is not None:
            self.bar = Fraction(4 * self.beats, int(token.split(":")[1]))
            self.beats = None

    def finish_element(self) -> None:
        self.base_time, self.base_known = self.passed(self.element)

    def passed(self, element: list[str]) -> tuple[Fraction, bool]:
        """The clock after the finished elements and element (whole or begun)."""
        if not element or element[0] not in HEADS and not PITCH.match(element[0]) and element[0] not in PREFIXES:
            return self.base_time, self.base_known
        if element[0] == BACKUP_TO_BAR:
            return Fraction(0), self.base_known
        length = length_of(element)
        if length is None:
            return self.base_time, False
        if element[0] == "backup":
            return self.base_time - length, self.base_known
        return (self.base_time + length if advances(element) else self.base_time), self.base_known

    def features(self) -> list[float]:
        time, known = self.passed(self.element)
        t = float(time)
        phase = 2 * math.pi * t
        if self.bar is None or not known:
            return [float(self.bar is not None), float(known), t / 4, 0.0, 0.0, 0.0, math.sin(phase), math.cos(phase)]
        left = float(self.bar - time)
        return [1.0, 1.0, t / 4, max(-LEFT_CLIP, min(LEFT_CLIP, left)) / 4, float(left < 0), float(left == 0),
                math.sin(phase), math.cos(phase)]


def clock_features(tokens: list[str]) -> list[list[float]]:
    """FEATURES after each token of a sequence."""
    stepper = ClockStepper()
    return [stepper.step(token) for token in tokens]


class BeatClock:
    """The clock for batches of token ids (the decoder's input), given the vocabulary's tokens by id."""

    def __init__(self, tokens_by_id: list[str]):
        self.names = tokens_by_id

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        rows = [clock_features([self.names[i] for i in seq]) for seq in ids.tolist()]
        return torch.tensor(rows, dtype=torch.float32, device=ids.device)
