"""Make Verovio's SMuFL music fonts available to cairo as text fonts.

Verovio draws most symbols as glyph paths, but text containing music symbols (the note
of a tempo mark, "♩ = 120") names the SMuFL font as a text font. Without that font
installed, cairo draws an empty box there, so the network never saw such notes in
context and reads real ones, as other engravers print them, as notes.

The fonts ship inside Verovio as WOFF2 embedded in CSS; they are extracted to a cache
directory and exposed through a private fontconfig file, leaving the system untouched.
"""
from __future__ import annotations

import base64
import functools
import io
import os
import re
from pathlib import Path

import verovio
from fontTools.ttLib import TTFont

FONT_DIR = Path(__file__).resolve().parents[2] / "data" / "fonts"
_WOFF2_URL = re.compile(r"url\(data:application/font-woff2;charset=utf-8;base64,([A-Za-z0-9+/=]+)\)")


def _extract_fonts() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for css in (Path(verovio.__file__).parent / "data").glob("*.css"):
        target = FONT_DIR / f"{css.stem}.otf"
        match = _WOFF2_URL.search(css.read_text())
        if target.exists() or not match:
            continue
        font = TTFont(io.BytesIO(base64.b64decode(match.group(1))))
        font.flavor = None
        partial = target.with_suffix(f".{os.getpid()}.tmp")  # parallel workers may race
        font.save(str(partial))
        partial.replace(target)


@functools.cache
def use_smufl_fonts() -> None:
    """Must run before cairo first renders text in this process (fontconfig reads its
    configuration once)."""
    _extract_fonts()
    conf = FONT_DIR / "fonts.conf"
    if not conf.exists():
        partial = conf.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(
            '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
            '  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>\n'
            f"  <dir>{FONT_DIR}</dir>\n  <cachedir>{FONT_DIR / 'cache'}</cachedir>\n</fontconfig>\n")
        partial.replace(conf)
    os.environ["FONTCONFIG_FILE"] = str(conf)
