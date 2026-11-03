"""Segmentation output channels.

This list is the single source of truth for channel order. It is embedded in the
exported ONNX model's metadata, and the C++ side looks channels up by name, so
reordering here never silently breaks inference.

Channels are multi-label (sigmoid per channel): a pixel may belong to a stem and
a beam at the same time.
"""
import numpy as np

CHANNELS = [
    "staff_line",
    "ledger_line",
    "barline",
    "brace",          # system-start brace/bracket; helps group grand staves
    "head_filled",
    "head_half",
    "head_whole",
    "head_center",    # small disk at each notehead centre; splits touching chord heads
    "stem",
    "beam",
    "flag_8",
    "flag_16",
    "flag_32",        # 32nd and shorter
    "aug_dot",
    "sharp",
    "flat",
    "natural",
    "double_sharp",
    "double_flat",
    "clef_g",
    "clef_f",
    "clef_c",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_8",
    "rest_16",
    "rest_32",        # 32nd and shorter
    "tie_slur",       # visually indistinguishable; assembly decides by pitch
    "ts_0", "ts_1", "ts_2", "ts_3", "ts_4", "ts_5", "ts_6", "ts_7", "ts_8", "ts_9",
    "ts_common",
    "ts_cut",
    "tuplet_3",       # tuplet numbers meaning 2/3 time: 3 and 6
    "tuplet_other",   # other tuplet numbers (2, 4, 5, 7, ...)
    "ottava",         # 8va/8vb/15ma number and its dashed extension line
    "other",          # dynamics, text, articulations, ornaments, grace notes...
]

CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}

# SMuFL code points (as used in Verovio's <use xlink:href="#XXXX-...">).
GLYPH_CHANNEL = {
    "E0A4": "head_filled", "E0A3": "head_half", "E0A2": "head_whole", "E0A0": "head_whole",
    "E240": "flag_8", "E241": "flag_8",
    "E242": "flag_16", "E243": "flag_16",
    "E244": "flag_32", "E245": "flag_32", "E246": "flag_32", "E247": "flag_32",
    "E262": "sharp", "E260": "flat", "E261": "natural", "E263": "double_sharp", "E264": "double_flat",
    # Mid-system clef changes use the smaller *ClefChange glyphs (E07A-E07C).
    "E050": "clef_g", "E052": "clef_g", "E053": "clef_g", "E07A": "clef_g",
    "E062": "clef_f", "E064": "clef_f", "E065": "clef_f", "E07C": "clef_f",
    "E05C": "clef_c", "E07B": "clef_c",
    "E4E3": "rest_whole", "E4E4": "rest_half", "E4E5": "rest_quarter",
    "E4E6": "rest_8", "E4E7": "rest_16", "E4E8": "rest_32", "E4E9": "rest_32", "E4EA": "rest_32", "E4EB": "rest_32",
    **{f"E08{d}": f"ts_{d}" for d in range(10)},
    "E08A": "ts_common", "E08B": "ts_cut",
}

# Tuplet number glyphs (tupletN = E880 + N).
TUPLET_GLYPH_CHANNEL = {f"E88{n}": "tuplet_3" if n in (3, 6) else "tuplet_other" for n in range(10)}


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """(H, W, C) bool -> (H, W, ceil(C/8)) uint8."""
    return np.packbits(masks, axis=-1)


def unpack_masks(packed: np.ndarray) -> np.ndarray:
    """(H, W, ceil(C/8)) uint8 -> (H, W, C) bool."""
    return np.unpackbits(packed, axis=-1, count=len(CHANNELS)).astype(bool)


def map_packed(packed: np.ndarray, fn) -> np.ndarray:
    """Apply a cv2 nearest-neighbour geometric op `fn(array) -> array` to packed masks.

    Nearest-neighbour ops only copy whole bytes, so they work on packed bits directly.
    cv2 accepts at most 4 channels per call, hence the chunking.
    """
    chunks = [fn(np.ascontiguousarray(packed[..., c:c + 4])) for c in range(0, packed.shape[-1], 4)]
    return np.dstack([ch.reshape(ch.shape[0], ch.shape[1], -1) for ch in chunks])
