"""Convert MUSCIMA++ v2 handwritten pages into the training page format.

usage: convert_muscima.py MUSCIMA_ROOT OUT_DIR

MUSCIMA_ROOT holds v2.0/ (MuNG annotations) and CvcMuscima-Distortions/ (images).
Every annotated object carries a pixel mask, which is painted into the channel its
class maps to. Abstract objects (staff, keySignature, measureSeparator, ...) have no
ink of their own and are skipped. MUSCIMA++ is CC BY-NC-SA 4.0.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from omr.channels import CHANNEL_INDEX, CHANNELS, pack_masks
from omr.render import HEAD_CENTER_RADIUS_INTERLINES

CLASS_CHANNEL = {
    "noteheadFull": "head_filled", "noteheadHalf": "head_half", "noteheadWhole": "head_whole",
    "stem": "stem", "beam": "beam", "legerLine": "ledger_line", "staffLine": "staff_line",
    "barline": "barline", "barlineHeavy": "barline", "brace": "brace", "bracket": "brace",
    "flag8thUp": "flag_8", "flag8thDown": "flag_8", "flag16thUp": "flag_16", "flag16thDown": "flag_16",
    "flag32ndUp": "flag_32", "flag32ndDown": "flag_32", "flag64thUp": "flag_32", "flag64thDown": "flag_32",
    "augmentationDot": "aug_dot",
    "accidentalSharp": "sharp", "accidentalFlat": "flat", "accidentalNatural": "natural",
    "accidentalDoubleSharp": "double_sharp", "accidentalDoubleFlat": "double_flat",
    "gClef": "clef_g", "fClef": "clef_f", "cClef": "clef_c",
    "restWhole": "rest_whole", "restHalf": "rest_half", "restQuarter": "rest_quarter",
    "rest8th": "rest_8", "rest16th": "rest_16", "rest32nd": "rest_32", "rest64th": "rest_32",
    "slur": "tie_slur", "tie": "tie_slur",
    "timeSigCommon": "ts_common", "timeSigCutCommon": "ts_cut",
}
ABSTRACT_CLASSES = {"staff", "staffSpace", "staffGrouping", "keySignature", "timeSignature", "measureSeparator"}
HEAD_CLASSES = {"noteheadFull", "noteheadHalf", "noteheadWhole"}
DOCUMENT_NAME = re.compile(r"CVC-MUSCIMA_W-(\d+)_N-(\d+)_D-ideal")


def decode_mask(rle: str, height: int, width: int) -> np.ndarray:
    values = []
    for token in rle.split():
        bit, count = token.split(":")
        values.extend([bit == "1"] * int(count))
    return np.array(values, dtype=bool).reshape(height, width)


def estimate_interline(nodes) -> float:
    tops = sorted(n["top"] + n["height"] / 2 for n in nodes if n["cls"] == "staffLine")
    gaps = [b - a for a, b in zip(tops, tops[1:]) if b - a < 60]
    return float(np.median(gaps))


def parse_nodes(xml_path: Path) -> list[dict]:
    nodes = []
    for node in etree.parse(str(xml_path)).getroot().iter("Node"):
        nodes.append({
            "id": int(node.findtext("Id")), "cls": node.findtext("ClassName"),
            "top": int(node.findtext("Top")), "left": int(node.findtext("Left")),
            "width": int(node.findtext("Width")), "height": int(node.findtext("Height")),
            "mask": node.findtext("Mask"), "outlinks": [int(v) for v in (node.findtext("Outlinks") or "").split()],
        })
    return nodes


def channel_for(node: dict, time_signature_numerals: set[int]) -> str | None:
    cls = node["cls"]
    if cls in ABSTRACT_CLASSES:
        return None
    if cls in CLASS_CHANNEL:
        return CLASS_CHANNEL[cls]
    if cls.startswith("numeral") and node["id"] in time_signature_numerals:
        return f"ts_{cls[len('numeral'):]}"
    return "other"  # text, dynamics, articulations, grace notes, ...


def convert_page(xml_path: Path, image_path: Path, out_stem: Path) -> None:
    nodes = parse_nodes(xml_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image.mean() < 128:  # stored white-on-black
        image = 255 - image
    interline = estimate_interline(nodes)
    numerals = {o for n in nodes if n["cls"] == "timeSignature" for o in n["outlinks"]}
    masks = np.zeros(image.shape + (len(CHANNELS),), dtype=bool)
    for n in nodes:
        channel = channel_for(n, numerals)
        if channel is None or channel not in CHANNEL_INDEX or not n["mask"]:
            continue
        t, l, h, w = n["top"], n["left"], n["height"], n["width"]
        masks[t:t + h, l:l + w, CHANNEL_INDEX[channel]] |= decode_mask(n["mask"], h, w)
        if n["cls"] in HEAD_CLASSES:
            centre = np.zeros(image.shape, np.uint8)
            cv2.circle(centre, (l + w // 2, t + h // 2), max(1, round(HEAD_CENTER_RADIUS_INTERLINES * interline)), 1, -1)
            masks[..., CHANNEL_INDEX["head_center"]] |= centre.astype(bool)
    cv2.imwrite(str(out_stem.with_suffix(".png")), image)
    np.savez_compressed(out_stem.with_suffix(".npz"), masks=pack_masks(masks))
    out_stem.with_suffix(".json").write_text(json.dumps({
        "source": xml_path.name, "font": "handwritten", "interline_px": interline, "options": {}}))


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for xml_path in sorted((root / "v2.0" / "data" / "annotations").glob("CVC-MUSCIMA_*.xml")):
        m = DOCUMENT_NAME.search(xml_path.name)
        writer, page = int(m.group(1)), int(m.group(2))
        image_path = root / "CvcMuscima-Distortions" / "ideal" / f"w-{writer:02d}" / "image" / f"p{page:03d}.png"
        convert_page(xml_path, image_path, out / f"muscima_w{writer:02d}_p{page:03d}")
    print(f"converted {len(list(out.glob('*.json')))} pages")


if __name__ == "__main__":
    main()
