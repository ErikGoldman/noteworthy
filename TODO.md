# To do

Status as of 2026-11-06, late evening. Details and numbers are in `MORNING_NOTES.md` and `EXPERIMENTS.md`.

## Running now

- Retrain `runs/seq512`: resumed from step 40,000 of 192,000 with `bash scripts/train_seq.sh --resume`.
  Log `runs/seq512.log`. When it ends, the script runs `seq_predict` and the OLiMPiC benchmark.
  No other GPU job while it runs; heavy CPU slows it.
- Articulation marks for the last 14 dev pages.

## While training runs (light CPU, no GPU)

1. Per-note "dot or wedge" mark: done. `add_art.py` mark `t`, scored with `omr_ned.py --ambiguous-marks`.
2. Add the ticks on 065 as ambiguous: done: 43 flagged, a few faint ones left unmarked. Clear wedges elsewhere stay wedges: add a wedge option to the
   entry helper and do a wedge pass over finished pages, whose agents were told to ignore wedges.
3. Articulation marks on the 56 sealed pages: done 2026-11-07, 1,131 marks (786 staccato, 333 accent,
   12 wedge). One agent per two pages; each wrote an image-only proof (key, clefs, bar count, first bass
   note) before opening gold, checked against gold by script; all 56 passed 3 of 4 fields. I spot-checked
   crops on 000, 001, 004, 007, 031. Gold questions they raised are in `GOLD_FOLLOWUPS.md`, among them
   099 (gold has 4 bars not on the page) and 084 bar 13 (user call on handwritten marks).
4. Ottava survey: the pages with no reliable result, then 072, 013, 060, 061, 044. Open: 050 bars 3-4
   have printed pitch in gold, so marking needs a pitch change; 082 bar 6 needs a staff fix first;
   076 bar 31's top note is in the left spine.
5. Hand-filed pages still open: 076 ("inner treble voice in left spine", bar not recorded; bar 4
   checked 2026-11-07 and already on the upper staff, since that split is on the upper spine). 030 done with `staff_down` (bar 7) and `move_down` (bar 8).
6. Padding rests (P8), Print standard: `hide_rests` override exists; 096 bars 6-13 done. Other pages
   need the image per range. Moves' own padding is now hidden automatically.
7. Tests for `eval/gold_overrides.py` and `eval/page_musicxml.py`.
8. Commit: the working tree on branch `gold-review` holds output fixes P4-P7, overrides and `TODO.md`.
   Waiting on the user's OK. Branch not merged to `main`.

## When training finishes

Done 2026-11-07: 1-5 (see EXPERIMENTS.md, "Retrain seq512"). Both scoring bugs found then are fixed in
`eval/omr_ned.py` (kern parsed with the authors' header). Next: 049 ottava against the image,
unlocked scores, then 6-10.


1. Check `runs/seq512.log`: finished, no NaN, final dev SER. `best.pt` is the candidate.
2. Polish dev pages with `seq_pages.py`, which needs the GPU free.
3. Ottava token recall: OLiMPiC dev crops with signs, and pages 023 and 110. 083 and 087 too.
4. Standard table against k512: dev SER, OLiMPiC crops, Mutopia, OMR-NED on As published, Corrected
   and Print standard. Paired ranges, all systems on the same gold. Copisteria with the same two
   attribute fixes (P5, P6), and say so.
5. First real check of `<octave-shift>` output on model output: labels only so far.
6. Confirm P4-P7 on real pages: tested on saved output only.
7. Re-rank and note-cost split: done 2026-11-07 (EXPERIMENTS.md, "Note cost split by kind"): mostly
   model reading errors that move a voice's later notes (quintuplets, beam counts, triplets).
8. Then investigate: slurs (model writes fewer than gold; test whether training crops vary them
   enough); barlines in cross-staff runs (083); bars on 087, 093, 051, 065, not re-inspected.
9. Update `docs/EVALUATION_PROTOCOL.md` lines 13 and 17, which name the frozen candidate.
10. Decide the frozen candidate with the user.

## Cost causes still open (seq512, Print standard, unlocked)

- 096: unmarked triplets on a page without a meter (see EXPERIMENTS.md). Not --unmarked-tuplets
  crops: tried as H6, mostly refuted (the ratio is what is missing, and no sign shows it). Open.
- 049 false ottava from a long line: fixed by plain-line negatives in crops (r4 fine-tune: 0 tokens).
- 023 missed ottava: not the crop, the line's style (wavy, redrawn dotted, dashed, with va), its
  distance or its system-wide shape: 0 tokens in every test. The crop cut the sign off; fixed in seq_pages.py (cut at the gap's middle). With
  the sign in view no token yet. 017's false ties (taller crops): each bar opens with a grace C slurred to a C, and the model
  now reads that slur as a tie (6 per system in systems 2-4 at both heights; none there before).
  Output fix P9 (`mend_ties`) removes such ties (they join different pitches): no training change
  needed.
- Bars on 065 (handwriting) and 030 (hand-filed moves still to add), 076 and 110 (ottavas).
- P8 padding rests on the other dev pages: image per range (003, 037 and 088 checked: their rests are printed, as in most of
  this composer's inner-voice writing; P8 looks concentrated in pages like 096).
- Beams (459 unlocked): 062 and 067 are cross-staff beams that kern cannot hold once notes move
  (start/stop against partial); 023's Trio was a gold error (short J), fixed in beams_closed.
- 021: done (bars 21-23 accidentals shown). Bar 13: a natural read as a sharp by the model.
- Accidentals: gold lacks `X` on accidentals the print restates (060 bars 25-26 after sharps); 
  `show_accidental` override exists, 060 done for four notes; other pages need image review (144
  accidentdel, not all gold).
- Slurs: gold leaves out printed slurs (017: grace slurs and long arcs); model misses some (030, 069).

## For the next training run

- Ottava signs: `8` alone, dotted lines, 8va above the lower staff are now in `build_seq_dataset.py
  --marks` (data/seq3/marks); tested by fine-tune q512: fixes 110, not 023 (see EXPERIMENTS.md). Long plain
  lines between the staves as negatives (049 false sign) added too: `add_plain_line`, a solid
  bracket line over a run of notes, no label, in a quarter of --marks scores; checked by eye.
- Gold: 110 Print standard needs its `*8va` passages (last system, both staves).

- No staccatissimo token: the model cannot write wedges.
- Repeat barlines: `repeat:start` / `repeat:end` tokens (train/seq/repeats.py), in labels from our
  linearizer, drawn back as <barline> by seq_predict.delinearize, left out of SER. Round trip
  checked on 5 PDMX scores. In the run started 2026-11-07 (seq513).
- Handwriting: no sequence model has any. Needs MUSCIMA++ as LMX; Smashcima drew no key signature
  on a 5-sharp system, is slow, and must exclude the test writers.

## Held for the user

- Sealed test run. Numbers only, and only after gold and the model are frozen.
- Email to the Transcoda authors: draft at https://claude.ai/artifact/NDK3s41464hRbJP1ybf7Bp. Send
  after the review is finished. Add dot-or-wedge and the sealed results of the review as needed.
- SMB application, which must come from the user.
- Gold reviews: sample of about 20 cases for the user to check; the hardest calls were 067 bars 5-8
  and 12, and 062 bar 6.
- C++ port: off plan until there is a good model.

## Rules that are easy to forget

- Gold must be checked through converter21 (`eval/check_gold.py`), never music21's own kern reader.
- Sealed pages: gold and images only. No system output, no errors, nothing into heuristics or
  training. Corrections to gold are allowed.
- Scoring is plain OMR-NED throughout; only the gold differs between standards.
- Single-page locked scores can swing widely on correct changes (042): check unlocked too.
