"""A tuplet look's drawn part, on Verovio's SVG (seq.tuplet_display): numbers set far from their
notes, drawn small, and brackets drawn as arcs. None of it changes what the label says.

Verovio draws a tuplet as <g class="tuplet"> holding its notes, a <g class="tupletNum"> (the digits,
each a <use> placed by translate(x, y) scale(s, s)) and, when bracketed, a <g class="tupletBracket">
of two polylines, hook end, bar, and the gap for the number between them.

A number set far from its notes can leave the page: Verovio's page holds the music within a margin
of 500 units (about three interlines at unit 9), and the rasterizer draws nothing outside it, so a
label would say printed where nothing is drawn (found by eye, 2026-11-19). The page is grown to
hold every number moved; it must be, before the page is sized for drawing (omr.render.layout_svg).
"""
from __future__ import annotations

import random
import re

from lxml import etree

from omr.render import SVG_NS
from seq.tuplet_display import TupletLook

PLACED = re.compile(r"translate\(([-\d.]+),\s*([-\d.]+)\)\s*scale\(([-\d.]+),\s*([-\d.]+)\)")
FAR_INTERLINES = (1.5, 4.0)
SMALL_SCALE = (0.55, 0.8)
ARC_POINTS = 24
MARGIN = re.compile(r"translate\([-\d.]+,\s*([-\d.]+)\)")


def classed(root: etree._Element, name: str) -> list[etree._Element]:
    return [g for g in root.iter(f"{{{SVG_NS}}}g") if name in g.get("class", "").split()]


def glyph_places(group: etree._Element) -> list[tuple[float, float, float]]:
    """(x, y, scale) of each glyph drawn in the group."""
    places = []
    for use in group.iter(f"{{{SVG_NS}}}use"):
        found = PLACED.search(use.get("transform", ""))
        if found:
            places.append((float(found[1]), float(found[2]), float(found[3])))
    return places


def away_from_notes(tuplet: etree._Element, number: etree._Element) -> int:
    """-1 if the number is drawn above its notes (SVG rows grow downwards), else 1."""
    heads = [y for head in classed(tuplet, "notehead") for _, y, _ in glyph_places(head)]
    digits = [y for _, y, _ in glyph_places(number)]
    if not heads or not digits:
        return -1
    return -1 if sum(digits) / len(digits) < sum(heads) / len(heads) else 1


def moved(group: etree._Element, dy: float) -> None:
    group.set("transform", f"translate(0, {dy:.1f}) " + group.get("transform", ""))


def shrunk(number: etree._Element, factor: float) -> None:
    """Each digit scaled about the middle of the number, so it stays over its group."""
    places = glyph_places(number)
    middle = sum(x for x, _, _ in places) / len(places)
    for use in number.iter(f"{{{SVG_NS}}}use"):
        found = PLACED.search(use.get("transform", ""))
        if found:
            x, y, sx, sy = (float(v) for v in found.groups())
            use.set("transform", PLACED.sub(f"translate({middle + (x - middle) * factor:.1f}, {y:.1f}) "
                                            f"scale({sx * factor:.3f}, {sy * factor:.3f})", use.get("transform")))


def polyline_points(line: etree._Element) -> list[tuple[float, float]]:
    return [tuple(map(float, p.split(","))) for p in line.get("points").split()]


def as_arc(bracket: etree._Element) -> bool:
    """The bracket's two halves redrawn as one shallow arc from hook tip to hook tip, with the same
    gap for the number; False (unchanged) if it is not drawn as two hooked halves."""
    lines = bracket.findall(f"{{{SVG_NS}}}polyline")
    if len(lines) != 2 or any(len(polyline_points(line)) != 3 for line in lines):
        return False
    (left_tip, left_corner, gap_left), (right_tip, _, gap_right) = (polyline_points(line) for line in lines)
    ends_y, bar_y = left_tip[1], left_corner[1]
    control_y = 2 * bar_y - ends_y  # a quadratic's apex lies halfway to its control point
    points = []
    for k in range(ARC_POINTS + 1):
        t = k / ARC_POINTS
        x = (1 - t) ** 2 * left_tip[0] + 2 * (1 - t) * t * (left_tip[0] + right_tip[0]) / 2 + t ** 2 * right_tip[0]
        y = (1 - t) ** 2 * ends_y + 2 * (1 - t) * t * control_y + t ** 2 * ends_y
        points.append((x, y))
    halves = [[p for p in points if p[0] <= gap_left[0]], [p for p in points if p[0] >= gap_right[0]]]
    for line, half in zip(lines, halves):
        line.set("points", " ".join(f"{x:.1f},{y:.1f}" for x, y in half))
        line.set("stroke-linejoin", "round")
    return True


def page_grown_to_hold(root: etree._Element, rows: list[float], interline: float) -> None:
    """The page's viewBox grown to hold glyphs placed at these rows (their baselines) of the page's
    margin group, with two interlines above (one clipped a 3:2's top) and one below."""
    page = root.find(f"{{{SVG_NS}}}svg")
    margin = next((g for g in classed(root, "page-margin")), None)
    found = MARGIN.search(margin.get("transform", "")) if margin is not None else None
    top_offset = float(found[1]) if found else 0.0
    x, y, width, height = (float(v) for v in page.get("viewBox").split())
    top = min(min(rows) + top_offset - 2 * interline, y)
    bottom = max(max(rows) + top_offset + interline, y + height)
    page.set("viewBox", f"{x:g} {top:g} {width:g} {bottom - top:g}")


def restyle_tuplets(svg: bytes, look: TupletLook, interline: float, rng: random.Random) -> bytes:
    """The look's far, small and arc shares drawn on each printed group; interline in SVG units.
    Run on Verovio's own SVG, before layout_svg sizes it."""
    if not (look.far or look.small or look.arc):
        return svg
    root = etree.fromstring(svg)
    moved_rows = []
    for tuplet in classed(root, "tuplet"):
        numbers, brackets = classed(tuplet, "tupletNum"), classed(tuplet, "tupletBracket")
        number, bracket = (numbers or [None])[0], (brackets or [None])[0]
        if number is None or not glyph_places(number):
            continue
        if bracket is not None and rng.random() < look.arc:
            as_arc(bracket)
        if rng.random() < look.small:
            shrunk(number, rng.uniform(*SMALL_SCALE))
        if rng.random() < look.far:
            dy = round(away_from_notes(tuplet, number) * rng.uniform(*FAR_INTERLINES) * interline, 1)
            for group in (number, bracket):
                if group is not None:
                    moved(group, dy)
            moved_rows += [y + dy for _, y, _ in glyph_places(number)]
    if moved_rows:
        page_grown_to_hold(root, moved_rows, interline)
    return etree.tostring(root)
