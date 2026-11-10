"""Render MusicXML pages with Verovio into (image, per-channel mask) pairs.

Two stages, so labels can change without redoing layout:
  1. layout: Verovio lays out a page; the SVG is sized to the target interline and
     annotated with everything labeling needs (grace notes), then kept as is.
  2. labeling (label_page): every visible primitive of that SVG is assigned one channel
     from its semantic context (nearest Verovio class + SMuFL glyph). The page image is
     the render of all primitives; each channel mask is the render of only that
     channel's primitives, so overlapping symbols (stem through notehead) get exact labels.
Datasets store the stage-1 SVG, so relabel_dataset.py can redo stage 2 alone.
"""
from __future__ import annotations

import functools
import random
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import numpy as np
import verovio
from lxml import etree

from .fonts import use_smufl_fonts
from .channels import CHANNEL_INDEX, CHANNELS, GLYPH_CHANNEL, TUPLET_GLYPH_CHANNEL

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
PRIMITIVES = {f"{{{SVG_NS}}}{t}" for t in ("path", "polygon", "polyline", "rect", "ellipse", "use", "text", "circle", "line")}

# Verovio staff space in internal units is 2 * unit * 10.
UNITS_PER_INTERLINE_PER_UNIT = 20.0

FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
HANDWRITTEN_FONTS = {"Petaluma"}

# Mask alpha threshold. Low, so hairline staff lines survive anti-aliasing.
MASK_ALPHA_THRESHOLD = 64

NOTEHEAD_GLYPHS = ("E0A0", "E0A2", "E0A3", "E0A4")
HEAD_CENTER_RADIUS_INTERLINES = 0.25
HEAD_CENTER_ATTR = "data-nw-head-center"
GRACE_ATTR = "data-nw-grace"


@dataclass
class RenderedPage:
    image: np.ndarray        # (H, W) uint8, 255 = paper
    masks: np.ndarray        # packed (H, W, ceil(C/8)) uint8, see channels.pack_masks
    interline_px: float
    font: str
    svg: bytes               # the laid-out page, input to label_page


def random_layout_options(rng: random.Random) -> dict:
    portrait = rng.random() < 0.85
    w, h = rng.choice([(2100, 2970), (2159, 2794), (2300, 3050)])
    if not portrait:
        w, h = h, w
    return {
        "font": rng.choice(FONTS),
        "pageWidth": w,
        "pageHeight": h,
        "pageMarginLeft": rng.randint(50, 250),
        "pageMarginRight": rng.randint(50, 250),
        "pageMarginTop": rng.randint(50, 250),
        "pageMarginBottom": rng.randint(50, 250),
        "unit": rng.uniform(6.0, 10.5),
        "spacingStaff": rng.randint(6, 20),
        "spacingSystem": rng.randint(4, 24),
        "spacingLinear": rng.uniform(0.15, 0.4),
        "spacingNonLinear": rng.uniform(0.45, 0.75),
        "staffLineWidth": rng.uniform(0.1, 0.3),
        "stemWidth": rng.uniform(0.1, 0.35),
        "barLineWidth": rng.uniform(0.15, 0.5),
        "ledgerLineThickness": rng.uniform(0.15, 0.4),
        "beamMaxSlope": rng.randint(5, 20),
        "justifyVertically": rng.random() < 0.5,
        "breaks": "auto",
        "header": rng.choice(["auto", "none"]),
        "footer": "none",
        "transpose": rng.choice(["", "", "", "+M2", "-M2", "+m3", "-m3", "+P4", "-P4", "+m2", "-m2"]),
        "svgBoundingBoxes": False,
        "svgHtml5": False,
        "scale": 100,
    }


def channel_of(prim: etree._Element) -> str:
    """Channel of one visible primitive of a laid-out page SVG."""
    classes: list[str] = []
    node = prim.getparent()
    while node is not None and node.tag == f"{{{SVG_NS}}}g":
        cls = node.get("class", "").split()
        if cls:
            classes.append(cls[0])
            if node.get(GRACE_ATTR):
                return "other"
        node = node.getparent()
    nearest = classes[0] if classes else ""
    glyph = prim.get(XLINK_HREF, "")[1:5] if prim.tag == f"{{{SVG_NS}}}use" else ""

    if nearest == "staff" and prim.tag.endswith("path"):
        return "staff_line"
    if nearest == "ledgerLines":
        return "ledger_line"
    if nearest == "barLine":
        return "barline" if prim.tag.endswith("path") else "other"
    if nearest == "system" and prim.tag.endswith("path"):
        return "barline"  # the system-start line joining the staves
    if nearest == "grpSym":
        return "brace"
    if nearest == "stem":
        return "stem"
    if nearest == "beam" and prim.tag.endswith("polygon"):
        return "beam"
    if nearest == "dots" and prim.tag.endswith("ellipse"):
        return "aug_dot"
    if nearest in ("tie", "slur"):
        return "tie_slur"
    if nearest in ("notehead", "flag", "accid", "keyAccid", "clef", "meterSig", "rest", "mRest"):
        return GLYPH_CHANNEL.get(glyph, "other")
    if nearest == "tupletNum":
        return TUPLET_GLYPH_CHANNEL.get(glyph, "other")
    if nearest == "octave":
        return "ottava"
    return "other"


@functools.cache
def _notehead_metrics(font: str) -> dict[str, tuple[float, float]]:
    """Glyph-space centre (x, y) of each notehead glyph, from Verovio's SMuFL bounding boxes."""
    tree = etree.parse(str(Path(verovio.__file__).parent / "data" / f"{font}.xml"))
    return {g.get("c"): (float(g.get("x")) + float(g.get("w")) / 2, float(g.get("y")) + float(g.get("h")) / 2)
            for g in tree.getroot().iter("g") if g.get("c") in NOTEHEAD_GLYPHS}


_TRANSFORM = re.compile(r"translate\(([-\d.]+),\s*([-\d.]+)\)\s*scale\(([-\d.]+),\s*([-\d.]+)\)")


def _add_head_center_markers(inner: etree._Element, font: str, radius_units: float) -> None:
    """Insert a circle at the centre of every (non-grace) notehead glyph, beside its <use>.

    Being a sibling, the circle inherits exactly the glyph's transform chain. Glyph paths
    are y-flipped in <defs>, hence the minus sign on the glyph-space y offset.
    """
    metrics = _notehead_metrics(font)
    for use in list(inner.iter(f"{{{SVG_NS}}}use")):
        glyph = use.get(XLINK_HREF, "")[1:5]
        if glyph not in metrics or channel_of(use) not in ("head_filled", "head_half", "head_whole"):
            continue
        tx, ty, sx, sy = (float(v) for v in _TRANSFORM.match(use.get("transform")).groups())
        gx, gy = metrics[glyph]
        use.addnext(etree.Element(f"{{{SVG_NS}}}circle", {HEAD_CENTER_ATTR: "1"},
                                  cx=f"{tx + gx * sx:.1f}", cy=f"{ty - gy * sy:.1f}", r=f"{radius_units:.1f}"))


def _strip_invisible(root: etree._Element) -> None:
    for g in root.iter(f"{{{SVG_NS}}}g"):
        cls = g.get("class", "")
        if "bounding-box" in cls or "placeholder" in cls:
            g.getparent().remove(g)


class _ChannelRasterizer:
    """Draws subsets of one page's channels, parsing the SVG only once.

    Each draw prunes the parsed tree's child lists down to the primitives of the
    requested channels, so drawing one channel costs in proportion to its content
    rather than to the whole page. Pixels are read straight from the cairo surface;
    a PNG round trip used to dominate dataset rendering time.
    """

    def __init__(self, svg: bytes, channel_of_prim: dict[int, str], width: int, height: int):
        self._tree = cairosvg.parser.Tree(bytestring=svg)
        self._width, self._height = width, height
        self._children: dict[int, list] = {}
        self._channels: dict[int, frozenset[str]] = {}  # per node: channels drawn in its subtree
        self._index(self._tree, channel_of_prim)

    def _index(self, node, channel_of_prim: dict[int, str]) -> frozenset[str]:
        self._children[id(node)] = list(node.children)
        prim = node.get("data-nw")
        found = {channel_of_prim[int(prim)]} if prim is not None else set()
        for child in node.children:
            found |= self._index(child, channel_of_prim)
        self._channels[id(node)] = frozenset(found)
        return self._channels[id(node)]

    def _prune(self, node, wanted: frozenset[str]) -> None:
        # Nodes holding no labeled primitive (defs, empty groups) are always kept.
        kept = [c for c in self._children[id(node)] if not self._channels[id(c)] or self._channels[id(c)] & wanted]
        node.children = kept
        for child in kept:
            self._prune(child, wanted)

    def alpha(self, channels: frozenset[str]) -> np.ndarray:
        """Alpha coverage (H, W) uint8 of the given channels' primitives."""
        self._prune(self._tree, channels)
        surface = cairosvg.surface.PNGSurface(self._tree, None, 96, output_width=self._width, output_height=self._height)
        surface.cairo.flush()
        stride = surface.cairo.get_stride()
        bgra = np.frombuffer(surface.cairo.get_data(), np.uint8).reshape(surface.height, stride // 4, 4)
        return bgra[: self._height, : self._width, 3].copy()


def seeded_toolkit() -> verovio.toolkit:
    """A Verovio toolkit whose element ids are the same on every run. Unseeded, they are random,
    and where elements overlap the page's pixels and masks depend on them (1 page in 24 of the
    segmenter's training set came out differently on every build)."""
    tk = verovio.toolkit()
    tk.setOptions({"xmlIdSeed": 1})
    return tk


def render_score_pages(musicxml_path: str, options: dict, target_interline_px: float,
                       max_pages: int, rng: random.Random) -> list[RenderedPage]:
    tk = seeded_toolkit()
    tk.setOptions(options)
    if not tk.loadFile(musicxml_path):
        return []
    page_numbers = list(range(1, tk.getPageCount() + 1))
    rng.shuffle(page_numbers)
    px_per_unit = target_interline_px / (UNITS_PER_INTERLINE_PER_UNIT * options["unit"])
    pages = []
    for page_no in page_numbers[:max_pages]:
        svg = layout_svg(tk, tk.renderToSVG(page_no), px_per_unit)
        image, masks = label_page(svg, options["font"], target_interline_px)
        pages.append(RenderedPage(image=image, masks=masks, interline_px=target_interline_px, font=options["font"], svg=svg))
    return pages


def layout_svg(tk: verovio.toolkit, svg: str, px_per_unit: float) -> bytes:
    """The page SVG, sized in pixels and self-describing: labeling needs no toolkit."""
    root = etree.fromstring(svg.encode())
    _strip_invisible(root)
    for g in root.iter(f"{{{SVG_NS}}}g"):
        if g.get("class", "").split()[:1] in (["note"], ["chord"]) and "grace" in tk.getElementAttr(g.get("id", "")):
            g.set(GRACE_ATTR, "1")
    inner = root.find(f"{{{SVG_NS}}}svg")
    vb = [float(v) for v in inner.get("viewBox").split()]
    root.set("width", f"{int(round(vb[2] * px_per_unit))}px")
    root.set("height", f"{int(round(vb[3] * px_per_unit))}px")
    return etree.tostring(root)


# Verovio draws every ottava extension with one dash pattern; other engravers use thinner,
# sparser dashes, which a network trained on Verovio alone does not recognize.
OTTAVA_DASH_UNITS = (15.0, 60.0)
OTTAVA_GAP_RATIO = (1.0, 4.0)
OTTAVA_STROKE_SCALE = (0.5, 1.5)


def _vary_line_styles(inner: etree._Element, rng: random.Random) -> None:
    """Randomizes the drawing style of lines Verovio always draws alike. Seeded from the
    page, so relabeling a stored page reproduces it."""
    dash = rng.uniform(*OTTAVA_DASH_UNITS)
    gap = dash * rng.uniform(*OTTAVA_GAP_RATIO)
    stroke_scale = rng.uniform(*OTTAVA_STROKE_SCALE)
    for g in inner.iter(f"{{{SVG_NS}}}g"):
        if g.get("class", "").split()[:1] != ["octave"]:
            continue
        for line in g.iter(f"{{{SVG_NS}}}path"):
            if line.get("stroke-dasharray"):
                line.set("stroke-dasharray", f"{dash:.0f} {gap:.0f}")
                line.set("stroke-width", f"{float(line.get('stroke-width', '18')) * stroke_scale:.0f}")


def label_page(svg: bytes, font: str, interline_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Page image (H, W) uint8 and packed channel masks of a laid-out page SVG."""
    use_smufl_fonts()
    root = etree.fromstring(svg)
    inner = root.find(f"{{{SVG_NS}}}svg")
    width, height = int(root.get("width").removesuffix("px")), int(root.get("height").removesuffix("px"))
    px_per_unit = width / float(inner.get("viewBox").split()[2])

    _add_head_center_markers(inner, font, HEAD_CENTER_RADIUS_INTERLINES * interline_px / px_per_unit)
    _vary_line_styles(inner, random.Random(zlib.crc32(svg)))

    # Glyph <defs> live in the outer <svg>, so iterating the inner page only visits drawn primitives.
    channel_of_prim: dict[int, str] = {}
    for n, prim in enumerate(inner.iter(*PRIMITIVES)):
        prim.set("data-nw", str(n))
        channel_of_prim[n] = "head_center" if prim.get(HEAD_CENTER_ATTR) else channel_of(prim)

    rasterizer = _ChannelRasterizer(etree.tostring(root), channel_of_prim, width, height)
    present = frozenset(channel_of_prim.values())
    image = 255 - rasterizer.alpha(present - {"head_center"})
    return image, _packed_masks(rasterizer, present, height, width)


def _packed_masks(rasterizer: "_ChannelRasterizer", present: frozenset[str], height: int, width: int) -> np.ndarray:
    """Channel masks packed as channels.pack_masks does (channel c = bit 7 - c % 8 of byte
    c // 8), built directly: filling a full (H, W, C) bool array channel by channel is a
    strided write per channel and dominated labeling time on large pages."""
    planes = np.zeros(((len(CHANNELS) + 7) // 8, height, width), dtype=np.uint8)
    for name in present:
        c = CHANNEL_INDEX[name]
        planes[c // 8] |= (rasterizer.alpha(frozenset({name})) >= MASK_ALPHA_THRESHOLD).view(np.uint8) << (7 - c % 8)
    return np.ascontiguousarray(np.moveaxis(planes, 0, -1))

