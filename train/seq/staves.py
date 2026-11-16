"""A system crop's staves, found from their lines, and one crop per staff for StaffSeq2Seq.

Staff lines are long, thin, horizontal ink: of the ink running on across much of the crop, the
part no thicker than a staff line (_staff_line_ink), so that ledger lines, noteheads, chord
clumps and beams fall away. Each row's darkness in that ink, less its neighbourhood's, is the
profile a five-line comb slides down, peaking on each staff's middle line. A match is as strong
as the comb's two weakest teeth together, so all five must sit on lines: a comb two or two and a
half times too wide, whose teeth reach the other staff's lines, leaves some in the gaps, and one
shifted a line up or down has a tooth off the staff; a
faint line only weakens the match, and a line broken within a strip (Smashcima draws some so)
leaves that strip to the others. Checked against the
segmenter on 300 crops (EXPERIMENTS.md): it splits where darkness alone could not.
The comb's spacing is the interline: about the one whose best match is strongest (_interline). Blurred lines, as on
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
and voices crossing between the hands sit, so those are seen from both staves. Optionally the
top and bottom staves' crops slide away from the system (outward): beyond the system nothing
else sees what lies there (ottava signs, the highest and lowest notes), while the gap between
staves is seen from both sides.
"""
from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np

from .data import CROP_INTERLINES
from .staff_positions import Staves

STAFF_INTERLINES = CROP_INTERLINES / 2
STRIPS = 8
# Staff-line ink must run across a whole strip, a row up or down allowed (_staff_line_ink): on a
# scan thousands of pixels wide a strip is hundreds wide, and a slight skew moves its lines more
# than that, so taller crops are looked at shrunk to this height, at which a strip is about 160
# px wide. GrandStaff's crops are about 260 px tall; at 256, five of 3,000 PDMX crops' thin lines
# faded too far to be found (EXPERIMENTS.md).
FINDING_HEIGHT = 320
INTERLINE_PLATEAU = 0.03
# Interlines tried, as shares of the crop's height: a system crop is about CROP_INTERLINES tall,
# more with wide margins or a third staff, less when cut tight.
MIN_INTERLINE_SHARE, MAX_INTERLINE_SHARE = 1 / 80, 1 / 12
# A staff's comb response is at least this share of the system's weakest staff's: of a strip's
# strongest, but in a system of two, of the second strongest, since a staff whose lines beams and
# chords cover, or blur fades, may match at a third of the other's (GrandStaff, PDMX, Smashcima).
STAFF_RESPONSE = 0.5


def find_staves(image: np.ndarray, count: int = 1) -> Staves | None:
    """The staves of a grayscale system crop (dark ink on light) of at least `count` staves, or
    None when the strips that find staves do not agree on how many there are. A crop taller than
    FINDING_HEIGHT is looked at shrunk to it."""
    scale = min(1.0, FINDING_HEIGHT / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), round(image.shape[0] * scale))
    found = _find_staves_at_scale(cv2.resize(image, size, interpolation=cv2.INTER_AREA), count)
    return None if found is None else Staves(tuple(c / scale for c in found.centres), found.interline / scale)


def _find_staves_at_scale(image: np.ndarray, count: int) -> Staves | None:
    found, thickness = [], _line_thickness(image)
    for strip in np.array_split(image, STRIPS, axis=1):
        profile = _line_profile(strip, thickness)
        interline = _interline(profile)
        if centres := _staff_centres(_comb(profile, interline), interline, count):
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
    staves = find_staves(image, count)
    if staves is None or len(staves.centres) < count:
        return None
    runs = [staves.centres[i:i + count] for i in range(len(staves.centres) - count + 1)]
    middle = image.shape[0] / 2
    return Staves(min(runs, key=lambda run: abs((run[0] + run[-1]) / 2 - middle)), staves.interline)


def _line_profile(image: np.ndarray, thickness: int) -> np.ndarray:
    """Each row's darkness in staff-line ink (_staff_line_ink), less the mean of the rows around
    it (a line's neighbourhood: two interlines at the most interline tried): staff lines stand
    out, filled regions do not."""
    darkness = _staff_line_ink(image, thickness).mean(axis=1) / 255
    window = max(3, round(2 * MAX_INTERLINE_SHARE * len(darkness)) | 1)
    return np.maximum(darkness - np.convolve(darkness, np.ones(window) / window, mode="same"), 0)


def _staff_line_ink(image: np.ndarray, thickness: int) -> np.ndarray:
    """The strip's darkness (0 white, 255 black) in ink that runs across the whole strip, a row up
    or down allowed for tilt, and is no thicker than a staff line: staff lines run across the
    strip, but a ledger line is a notehead or two wide, noteheads, stems and chord clumps no
    wider, and a beam, as long as it may be, is three or four lines thick. By darkness alone a
    stack of ledger lines, or of beams, an interline apart, was a staff, as strong as a real staff
    whose lines beams and chords cover or blur fades (Audiveris keeps only such thin, long ink for
    its staff lines, too). Ink running a third of the strip was not enough: on GrandStaff's
    blurred, warped copies a beamed group's ledger lines merge into runs that long, and a staff
    found a line up or down on them (EXPERIMENTS.md)."""
    within_a_row = cv2.dilate(255 - image, np.ones((3, 1), np.uint8))
    long = cv2.morphologyEx(within_a_row, cv2.MORPH_OPEN, np.ones((1, image.shape[1]), np.uint8))
    thicker_than_a_line = cv2.morphologyEx(long, cv2.MORPH_OPEN, np.ones((2 * thickness + 3, 1), np.uint8))
    return (long - thicker_than_a_line).astype(np.float32)


def _line_thickness(image: np.ndarray) -> int:
    """A staff line's thickness in pixels: the commonest vertical run of ink (Otsu's threshold),
    since staff lines cross every column and other ink only some."""
    _, ink = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    columns = np.pad(ink.T.astype(np.int8), ((0, 0), (1, 1)))
    edges = np.diff(columns, axis=1)
    runs = np.nonzero(edges == -1)[1] - np.nonzero(edges == 1)[1]
    return int(np.bincount(runs).argmax()) if len(runs) else 1


def _comb(profile: np.ndarray, interline: float) -> np.ndarray:
    """At each row, the mean of the two weakest of the profile's values at the five rows a staff
    centred there has its lines on, less their mean halfway between those rows: a comb no wider
    than a thick, blurred line puts all its teeth, and its gaps, on that one line."""
    rows = np.arange(len(profile), dtype=np.float32)
    at = lambda offsets: np.stack([np.interp(rows + k * interline, rows, profile, left=0, right=0) for k in offsets])
    return np.sort(at(range(-2, 3)), axis=0)[:2].mean(axis=0) - at((-1.5, -0.5, 0.5, 1.5)).mean(axis=0)


def _interline(profile: np.ndarray) -> float:
    """The middle of the run of spacings around the best whose matches are within
    INTERLINE_PLATEAU of its: a line is a few rows thick (more with the tilt _staff_line_ink
    allows), so combs somewhat narrower or wider than the staff still put every tooth on a line,
    and the best alone was the narrowest of them, 2.4% short on PDMX's drawn staves (0.6% so)."""
    n = len(profile)
    tried = np.arange(max(2.0, MIN_INTERLINE_SHARE * n), MAX_INTERLINE_SHARE * n, 0.25)
    response = np.array([_comb(profile, d).max() for d in tried])
    best = int(np.argmax(response))
    near = response >= (1 - INTERLINE_PLATEAU) * response[best]
    first, last = best, best
    while first > 0 and near[first - 1]:
        first -= 1
    while last < len(tried) - 1 and near[last + 1]:
        last += 1
    return float((tried[first] + tried[last]) / 2)


def _staff_centres(response: np.ndarray, interline: float, count: int) -> list[float]:
    """Rows where the comb response peaks, at least a staff's height (four interlines) and a line
    apart, and at least STAFF_RESPONSE of the count-th strongest such peak; top to bottom."""
    peaks = _separated_peaks(response, interline)
    if len(peaks) < count:
        return []
    threshold = STAFF_RESPONSE * response[peaks[count - 1]]
    return sorted(float(p) for p in peaks if response[p] >= threshold)


def _separated_peaks(response: np.ndarray, interline: float) -> list[int]:
    """Rows of positive response, strongest first, each more than five interlines from any stronger."""
    taken: list[int] = []
    for row in np.argsort(response)[::-1]:
        if response[row] <= 0:
            break
        if all(abs(row - t) > 5 * interline for t in taken):
            taken.append(int(row))
    return taken


def staff_crops(image: np.ndarray, staves: Staves, outward: float = 0.0, cut_between: bool = False) -> list[np.ndarray]:
    """One crop per staff, STAFF_INTERLINES tall and centred on it, white beyond the image's edges,
    even for a staff whose centre lies beyond them (augmentation's margins cut into the crop).
    With several staves, the top one's crop is centred `outward` interlines above it and the
    bottom one's as far below; middle staves stay centred. cut_between: each crop white past the
    midpoints to its neighbours, so neighbouring crops share no ink (the gap between two staves
    is split between them, as a whole-system model's nearest-staff bands split it)."""
    half = round(STAFF_INTERLINES * staves.interline / 2)
    centres = [round(c + staves.interline * outward * slide) for c, slide in zip(staves.centres, slides(len(staves.centres)))]
    pad = half + max(0, -min(centres), max(centres) - image.shape[0])
    padded = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)
    crops = [padded[c + pad - half:c + pad + half].copy() for c in centres]
    if cut_between:
        for crop, c, (above, below) in zip(crops, centres, midpoints(staves.centres)):
            top = c - half
            crop[:rows_before(above - top, len(crop))] = 255
            crop[rows_before(below - top, len(crop)):] = 255
    return crops


def rows_before(y: float, height: int) -> int:
    """How many of a crop's rows lie above y (in its rows, possibly infinite)."""
    return int(min(max(y, 0), height) + 0.5)


def midpoints(centres: tuple[float, ...]) -> list[tuple[float, float]]:
    """Each staff's bounds halfway to its neighbours, unbounded (infinite) where it has none."""
    between = [(a + b) / 2 for a, b in zip(centres, centres[1:])]
    return list(zip([-math.inf] + between, between + [math.inf]))


def slides(count: int) -> list[int]:
    """Which way each of `count` staves' crops slides outward: -1 up for the top, +1 down for the
    bottom, 0 for the middle ones and for a staff alone."""
    return [0] * count if count < 2 else [-1] + [0] * (count - 2) + [1]
