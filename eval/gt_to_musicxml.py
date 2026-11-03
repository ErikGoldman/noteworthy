"""Write a ground-truth .gt file as MusicXML, for proofreading it against the original.

usage: gt_to_musicxml.py GROUND_TRUTH.gt OUT.musicxml
"""
from __future__ import annotations

import sys
from fractions import Fraction
from xml.sax.saxutils import escape

from score_events import GtNote, GtScore, parse_gt

DIVISIONS = 8  # per quarter; a 32nd note is 1 division
TYPE_NAMES = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd"}
CLEF_SIGNS = {"G2": ("G", 2), "F4": ("F", 4), "C3": ("C", 3)}


def _divs(d: Fraction) -> int:
    v = d * DIVISIONS
    assert v.denominator == 1, d
    return int(v)


def _note_xml(note: GtNote, staff: int, voice: int, open_ties: set) -> list[str]:
    if note.is_space:
        return [f"<forward><duration>{_divs(note.duration)}</duration><voice>{voice}</voice><staff>{staff}</staff></forward>"]
    body_common = (f"<duration>{_divs(note.duration)}</duration><voice>{voice}</voice>"
                   f"<type>{TYPE_NAMES[note.base]}</type>" + "<dot/>" * note.dots + f"<staff>{staff}</staff>")
    if not note.pitches:
        return [f"<note><rest/>{body_common}</note>"]
    out = []
    for i, (step, alter, octave) in enumerate(note.pitches):
        key = (staff, step, alter, octave)
        ties = []
        if key in open_ties:
            ties.append("stop")
            open_ties.discard(key)
        if note.tie:
            ties.append("start")
            open_ties.add(key)
        tie_xml = "".join(f'<tie type="{t}"/>' for t in ties)
        notations = "".join(f'<tied type="{t}"/>' for t in ties)
        pitch = f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") + f"<octave>{octave}</octave></pitch>"
        out.append("<note>" + ("<chord/>" if i else "") + pitch + body_common.replace("<voice>", tie_xml + "<voice>", 1)
                   + (f"<notations>{notations}</notations>" if notations else "") + "</note>")
    return out


def to_musicxml(score: GtScore) -> str:
    beats, beat_type = score.header["time"][0].split("/")
    measure_len = Fraction(int(beats) * 4, int(beat_type))
    clefs = {int(v.split()[0]): CLEF_SIGNS[v.split()[1]] for v in score.header["clef"]}
    open_ties: set = set()
    measures_xml = []
    for m, staves in score.measures.items():
        parts = []
        if m == 0:
            clef_xml = "".join(f'<clef number="{s}"><sign>{sign}</sign><line>{line}</line></clef>' for s, (sign, line) in sorted(clefs.items()))
            parts.append(f"<attributes><divisions>{DIVISIONS}</divisions><key><fifths>{score.header['key'][0]}</fifths></key>"
                         f"<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time><staves>{len(clefs)}</staves>{clef_xml}</attributes>")
        voice_no = 0
        actual_len = Fraction(0)
        for staff, voices in sorted(staves.items()):
            for voice in voices:
                voice_no += 1
                length = sum((n.duration for n in voice), Fraction(0))
                actual_len = max(actual_len, length)
                for note in voice:
                    parts.extend(_note_xml(note, staff, voice_no, open_ties))
                parts.append(f"<backup><duration>{_divs(length)}</duration></backup>")
        implicit = ' implicit="yes"' if actual_len < measure_len and m == 0 else ""
        measures_xml.append(f'<measure number="{m}"{implicit}>' + "".join(parts) + "</measure>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<score-partwise version="3.1">'
            f"<work><work-title>{escape('ground truth')}</work-title></work>"
            '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
            '<part id="P1">' + "".join(measures_xml) + "</part></score-partwise>\n")


if __name__ == "__main__":
    with open(sys.argv[2], "w") as f:
        f.write(to_musicxml(parse_gt(sys.argv[1])))
