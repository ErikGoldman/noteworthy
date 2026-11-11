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
| Tie or slur open at a system edge | Half arc drawn by Verovio from neighbouring measures (`with_neighbours`, from seq516) | Systems left out (from seq516) | The label keeps the tie/slur token, as a print draws a half arc. About 10% of crops have one. Before seq516 no arc was drawn: 5% of OLiMPiC dev's tie/slur marks, 26% of seq515's tie/slur errors |
| Mid-score final or double barline | Drawn on both staves | Drawn on the upper staff only | MuseScore systems where a barline differs between staves are skipped |
| 8vb, 15ma, 15mb | Drawn as labelled | Drawn as labelled (after the ottava fix) | |
| Tremolos (single and two-note), arpeggios, slashed graces | Drawn as labelled | Drawn as labelled | |
| C and cut time, repeats and voltas | Drawn as labelled | Drawn as labelled | |
| Hidden rests (`print-object:no`) | Not drawn | Not drawn | |
| C clefs, G1, F3, F5 | Drawn as labelled | Drawn as labelled | |
| Stemless notes (`stem:none`) | Drawn as labelled | Drawn as labelled (a beam may float) | |
| Text-font ottava signs, old-style quarter rests, small naturals | Verovio styles only; drawn as labelled | Not used | |

Smashcima crops (`train/build_smashcima_crops.py`, our fork at `../Smashcima`) draw only what
the fork knows: notes, rests, accidentals, dots, clefs and clef changes, key and time signatures
(C and cut C too), ties, slurs (both drawn to the system edge when open there), triplet 3s,
staccato, accent, tenuto, single, double, final and repeat barlines, brace. The builder strips
grace notes, fermatas, arpeggios, ornaments, strong accents, ottava signs and non-triplet
tuplet numbers (hidden in the label then), and leaves out a system with 32nd notes or shorter,
breves, double flats or stemless notes (`UNDRAWN_TOKENS`), hidden pitched notes, or a measure
drawing nothing. Hidden rests are drawn as nothing, as in a print. A mark added to the
labels must be drawn by the fork or added to one of those lists. Writers draw quarter rests in
many shapes, some like a 3: not a mismatch, but look twice at tuplet sheets.

MusiCorpus crops (`train/build_musicorpus_crops.py`) are real pages: each piano grand staff,
labelled from its own MusicXML, cut from the page with its symbols. What the label could not say
is left out rather than mended: Dolores' training pages (their transcriptions omit the slurs the
pages draw; its README says expression slurs are not annotated), octave clefs, hidden notes,
hidden whole-measure rests (put where a page draws a measure-repeat sign), and grand staves where
only some tuplets are numbered. The transcriptions do not say which numbers are drawn; the symbol
annotations do (tuplet3 and the like): none drawn, all labelled hidden. Dynamics, lyrics, text,
mordents and turns are drawn without a token, as noise.

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
- 2026-11-10, Smashcima crops (sample of 671, `--limit 450 --systems-per-score 2`). Checked:
  tuplet 3s, accents, repeats (heavy barlines were drawn the height of a whole system: fixed,
  scaled to one staff), double barlines. Not yet: every other rare token, before a run uses them.
- 2026-11-10, Smashcima crops again (fresh samples of 1,100, `--limit 700 --systems-per-score 2`).
  Checked: hidden rests (drawn as nothing), keys of 5-7 accidentals, time signatures (C, cut,
  odd beats), double sharps, tenutos, staccatos, the G1 clef, hidden tuplet numbers, beam
  hooks, measure rests, ties and slurs open at the system's end. Found: every crop cut short
  at the right by up to 4% (Smashcima's 24.4 mm inch: last notes and barline missing, still in
  the label; fixed, all earlier Smashcima crops void) and slurs drawn as thick blocks from
  tiny samples (fixed). All tokens in the table above are now checked for Smashcima crops.
- 2026-11-10, MusiCorpus crops (all 53 training grand staves, 39 handwritten, 14 engraved).
  Checked: tuplets (15 crops, 161 groups: numbers drawn on the first bar only and none after on
  several; labels said every number shown; fixed from the symbol annotations, 10 grand staves with
  some numbers drawn left out), ottava signs (4 crops: drawn where labelled), the cut (the
  dataset's own clipped notes on ledger lines; now cut from the page with every symbol, and the
  voice staff above a song's piano shows in part, as when recognizing), measure-repeat signs
  (labelled as hidden measure rests: left out), Dolores slurs (drawn, never labelled: Dolores not
  trained on).
- 2026-11-11, seq516's Verovio crops (half arcs at system breaks, random edge margins, 60%
  justified). Checked 24 random pdmx crops with a tie or slur open at an edge (303 of 3,000
  sampled): Verovio's split arcs drawn after the key signature (incoming) and to the closing
  barline (outgoing), no courtesy signatures, no ottava from a neighbour; labels unchanged, crop
  counts as seq514's. Outgoing arcs are still shorter than a print's where the line is not
  justified (Verovio caps the margin before the barline).
- 2026-11-11, seq516's `unmarked` Verovio set (3,990 crops, 19,007 tuplet groups, 90% labelled
  hidden). Checked 8 random crops: each draws exactly the numbers its label shows (the kept
  first ones), none where all are hidden.
- 2026-11-11, GrandStaff-LMX clefs, keys and meters (found by a label sweep: seq515 reading the
  training set, its worst-read crops looked at). The release drops every clef, key and meter
  change, and a staff whose spine splits before its signatures loses them (treble, C major, no
  meter: every flat or sharp then written out as an accidental). All 53,882 systems rebuilt by
  OLiMPiC's own build with the source's signatures put back (`train/rebuild_grandstaff_lmx.py`;
  unchanged, the build gives back the release byte for byte): 20,757 mended, 499 left out (notes,
  measures or more than signatures unlike the source). Seven mended systems rendered beside
  their images: clefs, keys, meters and accidentals as drawn. seq516's GrandStaff labels: no
  wrong key or missing meter left (2,878 and 3,383 before), 25% with a clef change. Write-up and
  patch for the dataset's users: `errata/grandstaff-lmx/`. Also found: 29 blank GrandStaff images
(one grey or nearly black, most of them distorted ones; systems using one left out; OLiMPiC
synthetic scanned, none blank), and 68 systems whose images draw empty bars after a merger
of the staves where OLiMPiC cut the label (left out). Grace notes: GrandStaff's images draw
  every one, as a stemless notehead (8 of 8 by eye), but two thirds of the labels lack them;
  all 2,984 training systems with one in the source left out. Not checked yet: reminder
  accidentals (one label has a natural its image does not draw).
