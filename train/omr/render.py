"""Render MusicXML pages with Verovio into (image, per-channel mask) pairs.

Every visible SVG primitive is assigned one channel from its semantic context
(nearest Verovio class + SMuFL glyph). The page image is the render of all
primitives; each channel mask is the render of only that channel's primitives,
so overlapping symbols (stem through notehead, beam over stem) get exact labels.
"""
from __future__ import annotations

import copy
import functools
import io
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import numpy as np
import verovio
from lxml import etree
from PIL import Image

from .channels import CHANNEL_INDEX, CHANNELS, GLYPH_CHANNEL, pack_masks

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


@dataclass
class RenderedPage:
    image: np.ndarray        # (H, W) uint8, 255 = paper
    masks: np.ndarray        # packed (H, W, ceil(C/8)) uint8, see channels.pack_masks
    interline_px: float
    font: str


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


class _Labeler:
    """Assigns a channel to every visible primitive of a Verovio page SVG."""

    def __init__(self, toolkit: verovio.toolkit):
        self._tk = toolkit
        self._grace_cache: dict[str, bool] = {}

    def _is_grace(self, element_id: str) -> bool:
        if element_id not in self._grace_cache:
            self._grace_cache[element_id] = "grace" in self._tk.getElementAttr(element_id)
        return self._grace_cache[element_id]

    def channel_of(self, prim: etree._Element) -> str:
        classes: list[str] = []
        node = prim.getparent()
        while node is not None and node.tag == f"{{{SVG_NS}}}g":
            cls = node.get("class", "").split()
            if cls:
                classes.append(cls[0])
                if cls[0] in ("note", "chord") and self._is_grace(node.get("id", "")):
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
        return "other"


@functools.cache
def _notehead_metrics(font: str) -> dict[str, tuple[float, float]]:
    """Glyph-space centre (x, y) of each notehead glyph, from Verovio's SMuFL bounding boxes."""
    tree = etree.parse(str(Path(verovio.__file__).parent / "data" / f"{font}.xml"))
    return {g.get("c"): (float(g.get("x")) + float(g.get("w")) / 2, float(g.get("y")) + float(g.get("h")) / 2)
            for g in tree.getroot().iter("g") if g.get("c") in NOTEHEAD_GLYPHS}


_TRANSFORM = re.compile(r"translate\(([-\d.]+),\s*([-\d.]+)\)\s*scale\(([-\d.]+),\s*([-\d.]+)\)")


def _add_head_center_markers(inner: etree._Element, labeler: "_Labeler", font: str, radius_units: float) -> None:
    """Insert a circle at the centre of every (non-grace) notehead glyph, beside its <use>.

    Being a sibling, the circle inherits exactly the glyph's transform chain. Glyph paths
    are y-flipped in <defs>, hence the minus sign on the glyph-space y offset.
    """
    metrics = _notehead_metrics(font)
    for use in list(inner.iter(f"{{{SVG_NS}}}use")):
        glyph = use.get(XLINK_HREF, "")[1:5]
        if glyph not in metrics or labeler.channel_of(use) not in ("head_filled", "head_half", "head_whole"):
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


def _rasterize(root: etree._Element, width: int, height: int) -> np.ndarray:
    """Returns alpha coverage (H, W) uint8 of the drawing."""
    png = cairosvg.svg2png(bytestring=etree.tostring(root), output_width=width, output_height=height)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    return rgba[..., 3]


def _filtered_copy(root: etree._Element, keep) -> etree._Element:
    clone = copy.deepcopy(root)
    for prim in list(clone.iter(*PRIMITIVES)):
        tag_id = prim.get("data-nw")
        if tag_id is not None and not keep(int(tag_id)):
            prim.getparent().remove(prim)
    return clone


def _remove_except(root: etree._Element, keep_ids: set[int]) -> etree._Element:
    return _filtered_copy(root, lambda n: n in keep_ids)


def _remove_ids(root: etree._Element, drop_ids: set[int]) -> etree._Element:
    return _filtered_copy(root, lambda n: n not in drop_ids)


def render_score_pages(musicxml_path: str, options: dict, target_interline_px: float,
                       max_pages: int, rng: random.Random) -> list[RenderedPage]:
    tk = verovio.toolkit()
    tk.setOptions(options)
    if not tk.loadFile(musicxml_path):
        return []
    page_numbers = list(range(1, tk.getPageCount() + 1))
    rng.shuffle(page_numbers)
    labeler = _Labeler(tk)
    px_per_unit = target_interline_px / (UNITS_PER_INTERLINE_PER_UNIT * options["unit"])
    pages = []
    for page_no in page_numbers[:max_pages]:
        pages.append(_render_page(tk.renderToSVG(page_no), labeler, px_per_unit, target_interline_px, options["font"]))
    return pages


def _render_page(svg: str, labeler: _Labeler, px_per_unit: float, interline_px: float, font: str) -> RenderedPage:
    root = etree.fromstring(svg.encode())
    _strip_invisible(root)
    inner = root.find(f"{{{SVG_NS}}}svg")
    vb = [float(v) for v in inner.get("viewBox").split()]
    width, height = int(round(vb[2] * px_per_unit)), int(round(vb[3] * px_per_unit))
    root.set("width", f"{width}px")
    root.set("height", f"{height}px")

    _add_head_center_markers(inner, labeler, font, HEAD_CENTER_RADIUS_INTERLINES * interline_px / px_per_unit)

    # Glyph <defs> live in the outer <svg>, so iterating the inner page only visits drawn primitives.
    channel_prims: dict[str, set[int]] = {name: set() for name in CHANNELS}
    for n, prim in enumerate(inner.iter(*PRIMITIVES)):
        prim.set("data-nw", str(n))
        channel = "head_center" if prim.get(HEAD_CENTER_ATTR) else labeler.channel_of(prim)
        channel_prims[channel].add(n)

    image = 255 - _rasterize(_remove_ids(root, channel_prims["head_center"]), width, height)
    masks = np.zeros((height, width, len(CHANNELS)), dtype=bool)
    for name, ids in channel_prims.items():
        if ids:
            alpha = _rasterize(_remove_except(root, ids), width, height)
            masks[..., CHANNEL_INDEX[name]] = alpha >= MASK_ALPHA_THRESHOLD
    return RenderedPage(image=image, masks=pack_masks(masks), interline_px=interline_px, font=font)

