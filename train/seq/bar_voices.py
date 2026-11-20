"""How many voices a staff holds in a bar, from the one LMX parser (seq.lmx_tuplets.parse_measure),
for every caller: the voice-count label form, the two-voice repeat of training, and spec 4's
two-voice class. Two questions, kept apart:

  voices_written: stretches between backups that write a voice (a note, rest or whole-bar rest,
      drawn or not), each on the staff of its first; what a label spells out, so what the
      voice-count form commits the decoder to. A stretch of forwards alone is not one: MuseScore
      writes such stretches to carry a clef change or fill, and they draw nothing.
  voices_with_notes: voices with notes on the staff (grace notes and chord tones too; not rests, nor
      anything print-object:no): a staff holds two voices when it has notes from two (the user,
      2026-11-19). A cross-staff note puts its voice on the staff it is printed on (2026-11-16), so a
      figure running from one staff into the other makes two voices there if that staff has another
      voice's notes in the bar.

Staves come from the stretch's own staff:N tokens (parse_measure), never a clef's: the two rules
this replaces read a voice's staff as the first staff token in its stretch (a bar-opening clef's,
`clef:F4 staff:2`) or the first after its first timed note (the next voice's, after a grace note).
Checked against MusicXML's own voices and staves by eval/check_bar_voices.py.
"""
from __future__ import annotations

from collections import Counter

from seq.lmx_tuplets import Voice, parse_measure


def stretches(bar: list[str]) -> list[Voice]:
    return parse_measure(bar, 0, len(bar))


def voices_written(bar: list[str]) -> dict[int, int]:
    counts: Counter = Counter()
    for stretch in stretches(bar):
        heads = [e for e in stretch.events if e.kind != "forward"]
        staff = next((e.staff for e in heads if e.staff is not None), None)
        if staff is not None:
            counts[staff] += 1
    return dict(counts)


def voices_with_notes(bar: list[str]) -> dict[int, int]:
    voices: dict[int, set] = {}
    for k, stretch in enumerate(stretches(bar)):
        for event in stretch.events:
            if event.kind == "note" and event.drawn and event.staff is not None:
                voices.setdefault(event.staff, set()).add(event.voice or f"stretch {k}")
    return {staff: len(names) for staff, names in voices.items()}


def has_two_voices_on_a_staff(bar: list[str]) -> bool:
    return any(n > 1 for n in voices_with_notes(bar).values())
