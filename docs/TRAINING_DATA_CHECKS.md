# Training-data checks: does each crop draw what its label says?

A training crop teaches the model that its image means its label. Where an engraver draws
something other than what the label says, the model learns that wrong reading, and the
mistake shows up later on real pages, where it is much harder to trace. seq514 learned two
such readings (see EXPERIMENTS.md, "seq514"): octaves that do not follow the notehead, and
hidden tuplet numbers over printed 3s.

A whole-crop check such as SER on a sample does not see these: they hit 1-3% of crops. Check
each rare token's crops by eye instead, after any change to a crop builder or a new engraver,
and before a training run.

## How to check

1. Count each token's crops in a set, to find the rare ones:

   ```
   python3 - <<'E'
   from collections import Counter; from pathlib import Path
   root = Path('data/seq514_data/seq2/musescore'); crops = Counter()
   for n in (root / 'samples.train.txt').read_text().split():
       crops.update(set((root / f'{n}.lmx').read_text().split()))
   for t, c in sorted(crops.items(), key=lambda kv: kv[1]): print(c, t)
   E
   ```

   Also compare paired counts (`tied:start` against `tied:stop`, `slur:start` against
   `slur:stop`): a difference means marks open at a system's edge.
2. Make a contact sheet for each rare token, in each crop set:

   ```
   python train/crop_sheets.py data/seq514_data/seq2/pdmx tremolo* --count 4
   python train/crop_sheets.py data/seq514_data/seq2/pdmx tied:start --tail   # system ends
   ```

3. For each crop, compare what is drawn with the label (the `.lmx` beside the `.png`). Where
   the sheet cannot show it (which tuplet group is hidden, which octave a note is in), print
   the label's tokens and look at the image at full size.
4. Where they disagree, find out which side is wrong. Fix the builder so that the label and the
   image agree, then re-render only the scores affected. Both builders skip a score whose
   folder exists: move those folders aside and run the builder again. A fix that uses the
   random generator differently changes every later system of a score: re-render every score
   it could reach.

## What each engraver does (as of 2026-11-09)

| Mark | Verovio crops | MuseScore 3.6 crops | Handling |
|---|---|---|---|
| Ottava sign open at the system's end (runs into the next system) | Draws it; every crop's signs are compared with its label (`ottavas_drawn_as_labelled`) | **Leaves the sign out** and draws the notes where they sound | Each system's signs are redrawn from its label, closed within the system (`with_label_ottavas`) |
| Tuplet number hidden (`show-number="none"`) | Drawn hidden | **Ignored: every number drawn** | MuseScore crops label every number shown (`show_every_tuplet_number`); hidden numbers are taught by the Verovio crops |
| Octave clef (treble with an 8 below, 2.4% of PDMX piano parts) | **Draws the 8; notes an octave from the label** | **Same** | Clefs drawn plain, notes kept at sounding pitch (`without_octave_clefs`, both builders); our clef tokens have no octave |
| Tie or slur open at a system edge | No half arc drawn | No half arc drawn | **Open.** The label keeps the tie/slur token; a print draws a half arc. About 3.4% of crops at each end |
| Mid-score final or double barline | Drawn on both staves | Drawn on the upper staff only | MuseScore systems where a barline differs between staves are skipped |
| 8vb, 15ma, 15mb | Drawn as labelled | Drawn as labelled (after the ottava fix) | |
| Tremolos (single and two-note), arpeggios, slashed graces | Drawn as labelled | Drawn as labelled | |
| C and cut time, repeats and voltas | Drawn as labelled | Drawn as labelled | |
| Hidden rests (`print-object:no`) | Not drawn | Not drawn | |
| C clefs, G1, F3, F5 | Drawn as labelled | Drawn as labelled | |
| Stemless notes (`stem:none`) | Drawn as labelled | Drawn as labelled (a beam may float) | |
| Text-font ottava signs, old-style quarter rests, small naturals | Verovio styles only; drawn as labelled | Not used | |

Sources to look at when adding a set: OLiMPiC synthetic and GrandStaff come as images, so a
mismatch there cannot be re-rendered, only left out. Neither has octave clefs (checked
2026-11-09).

Noise that is not a mismatch, as no token covers it: lyrics, chord symbols, pedal brackets,
copyright and other text, glissando lines.

## Log

- 2026-11-09, before seq515. MuseScore and Verovio (pdmx, marks) checked for every token in the
  table. Fixed: the open ottava sign (728 MuseScore crops), hidden tuplet numbers (642 MuseScore
  crops), octave clefs (about 2% of crops in every PDMX set). Re-rendered: 3,482 + 516 MuseScore
  scores, 1,500 Verovio score folders. Old crops in `data/seq514_data/musescore_ottava_before_fix`
  and `data/seq514_data/before_clef_tuplet_fix`. Left: open ties and slurs.
