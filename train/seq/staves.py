"""A system crop's staves, found from their lines, and one crop per staff for StaffSeq2Seq.

Staff lines are the rows darker than the rows around them (each row's mean darkness, less its
neighbourhood's): a five-line comb slid down that profile peaks on each staff's middle line. A
match is as strong as the comb's two weakest teeth together, so all five must sit on lines: a
comb two or two and a half times too wide, whose teeth reach the other staff's lines, leaves some
in the gaps, and one shifted a line up or down has a tooth off the staff; a faint or broken line
(Smashcima draws some so) only weakens the match.
The comb's spacing is the interline: the one whose best match is strongest. Blurred lines, as on
GrandStaff's small JPEGs, still darken their rows. Staves are looked for in vertical strips, so
that a page's slight tilt, a camera's warp or a hand's wavering moves a line only a little within
a strip; each strip finds its own interline, and strips that agree on the staff count give each
staff's centre, and the interline, by median.

A crop may hold more staves than its system: a song's voice staff, or a neighbouring system's,
reaching into the margins. system_staves keeps the system's: the adjacent run of its size whose
middle is nearest the crop's, as crops are cut centred on their system.

Each staff's crop is STAFF_INTERLINES tall, centred on it: as tall as half a two-staff system
crop (seq.data.CROP_INTERLINES), so a staff scaled to 256 px keeps the resolution a system has at
512. Neighbouring staves' crops overlap in the gap between them, where ledger-line notes, beams
and voices crossing between the hands sit, so those are seen from both staves.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .data import CROP_INTERLINES

STAFF_INTERLINES = CROP_INTERLINES / 2
STRIPS = 8
# Interlines tried, as shares of the crop's height: a system crop is about CROP_INTERLINES tall,
# more with wide margins or a third staff, less when cut tight.
MIN_INTERLINE_SHARE, MAX_INTERLINE_SHARE = 1 / 80, 1 / 12
STAFF_RESPONSE = 0.5  # a staff's comb response is at least this share of its strip's strongest


@dataclass(frozen=True)
class Staves:
    centres: tuple[float, ...]  # rows, top to bottom
    interline: float


def find_staves(image: np.ndarray) -> Staves | None:
    """The staves of a grayscale system crop (dark ink on light), or None when the strips that
    find staves do not agree on how many there are."""
    found = []
    for strip in np.array_split(image, STRIPS, axis=1):
        profile = _line_profile(strip)
        interline = _interline(profile)
        if centres := _staff_centres(_comb(profile, interline), interline):
            found.append((centres, interline))
    if not found:
        return None
    count, votes = Counter(len(c) for c, _ in found).most_common(1)[0]
    if votes < len(found) / 2:
        return None
    agreeing = [(c, d) for c, d in found if len(c) == count]
    return Staves(tuple(float(np.median([c[k] for c, _ in agreeing])) for k in range(count)),
                  float(np.median([d for _, d in agreeing])))


def system_staves(image: np.ndarray, count: int) -> Staves | None:
    """The crop's system of `count` staves: of the staves found, the adjacent run of that many
    whose middle is nearest the crop's. None when fewer are found."""
    staves = find_staves(image)
    if staves is None or len(staves.centres) < count:
        return None
    runs = [staves.centres[i:i + count] for i in range(len(staves.centres) - count + 1)]
    middle = image.shape[0] / 2
    return Staves(min(runs, key=lambda run: abs((run[0] + run[-1]) / 2 - middle)), staves.interline)


def _line_profile(image: np.ndarray) -> np.ndarray:
    """Each row's darkness, less the mean of the rows around it (a line's neighbourhood is two
    interlines at the most interline tried): staff lines stand out, filled regions do not."""
    darkness = 1 - image.astype(np.float32).mean(axis=1) / 255
    window = max(3, round(2 * MAX_INTERLINE_SHARE * len(darkness)) | 1)
    return np.maximum(darkness - np.convolve(darkness, np.ones(window) / window, mode="same"), 0)


def _comb(profile: np.ndarray, interline: float) -> np.ndarray:
    """At each row, the mean of the two weakest of the profile's values at the five rows a staff
    centred there has its lines on, less their mean halfway between those rows: a comb no wider
    than a thick, blurred line puts all its teeth, and its gaps, on that one line."""
    rows = np.arange(len(profile), dtype=np.float32)
    at = lambda offsets: np.stack([np.interp(rows + k * interline, rows, profile, left=0, right=0) for k in offsets])
    return np.sort(at(range(-2, 3)), axis=0)[:2].mean(axis=0) - at((-1.5, -0.5, 0.5, 1.5)).mean(axis=0)


def _interline(profile: np.ndarray) -> float:
    n = len(profile)
    tried = np.arange(max(2.0, MIN_INTERLINE_SHARE * n), MAX_INTERLINE_SHARE * n, 0.25)
    return float(tried[np.argmax([_comb(profile, d).max() for d in tried])])


def _staff_centres(response: np.ndarray, interline: float) -> list[float]:
    """Rows where the comb response peaks, strongest first, at least a staff's height (four
    interlines) and a line apart; returned top to bottom."""
    threshold, taken = STAFF_RESPONSE * response.max(), []
    for row in np.argsort(response)[::-1]:
        if response[row] < threshold:
            break
        if all(abs(row - t) > 5 * interline for t in taken):
            taken.append(row)
    return sorted(float(t) for t in taken) if response.max() > 0 else []


def staff_crops(image: np.ndarray, staves: Staves) -> list[np.ndarray]:
    """One crop per staff, STAFF_INTERLINES tall and centred on it, white beyond the image's edges."""
    half = round(STAFF_INTERLINES * staves.interline / 2)
    padded = cv2.copyMakeBorder(image, half, half, 0, 0, cv2.BORDER_CONSTANT, value=255)
    return [padded[round(c):round(c) + 2 * half] for c in staves.centres]
