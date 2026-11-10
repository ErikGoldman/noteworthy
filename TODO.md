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
5. Hand-filed pages: 076 done 2026-11-08 with `move_up` (bars 15-17, the inner voice under the
   trill). Left: bar 17's bass note (`4F#`) sits in the right spine's second voice but is drawn on
   the lower staff, and no op moves a split column down. 030 done with `staff_down` (bar 7) and
   `move_down` (bar 8). 110's last system is wrong beyond filing (GOLD_FOLLOWUPS.md).
6. Padding rests (P8), Print standard: `hide_rests` override exists; 096 bars 6-13 done. Other pages
   need the image per range. Moves' own padding is now hidden automatically.
7. Tests: `eval/test_page_musicxml.py` (P9-P11, slur pairing) and `eval/test_gold_overrides.py` (line-local
   ops, staff moves, one slur per chord), stdlib unittest, run from eval/: 32 tests.
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
  `show_accidental` override exists, 060 done for four notes, 037 bar 14 (G#4, two voices); 037 bars
  18-25 and other pages need image review at a finer scan than the 1050-px page images (144
  accidentdel, not all gold).
- Tuplet numbers: Print gold could take `*Xtuplet` where the print hides 3s (22 edits); image per page.
- Slurs: gold leaves out printed slurs (017: grace slurs and long arcs); model misses some (030, 069).
- Ties: model read 2 in 5 ties as slurs; P10 `ties_from_slurs` (2026-11-07, not committed) turns a slur
  between events of the same pitches into a tie. Gold's 55 "orphan" ties are mostly
  ties into another voice (096), not gold errors.
- Slur into a chord that holds the same pitch (060 bar 3, +0.13): P10's tie rule turns the model's
  slur from a 16th A♭5 into the next chord into a tie on the chord's A♭5, but the arc goes to the
  chord's top E♭6, so it is a slur (gold is right; the user read it, 2026-11-08). The LMX slur
  token names events, not heads, so no output rule can tell this from 096's real ties. Possible
  fixes: a per-note tie the model writes itself (it often reads ties as slurs now), or the arc's end
  height from the image. Not urgent: the rule is 9 pages better, 1 worse.

## For the next training run

- **`seq514.patch` (2026-11-08, built in a scratch worktree; apply after seq513 is scored; supersedes
  barline-tokens.patch, which it includes).** All tests pass in the worktree; a 40-score crop build
  checked by eye. New tokens at the vocabulary's end: `barline:double`, `barline:final`,
  `time:common`, `time:cut`, `tuplet:hidden`.
  - Time symbols: labels from MusicXML `symbol` (PDMX, OLiMPiC synthetic) and from GrandStaff's
    `*met(c)` (`build_grandstaff_written.py`: 2,189 labels gain a token, 112 systems left out whose
    symbols cannot be matched up); 30% of 4/4 and 2/2 crops drawn as symbols. Output sets
    `<time symbol>`; the old 4/4-to-C page rule is removed. Left out of SER.
  - Hidden tuplet numbers (B): `tuplet:hidden` after `tuplet:start` where the number is not drawn
    (OLiMPiC synthetic has 652 such systems; PDMX about 1 in 10 tuplet scores); 20% of crops with
    tuplets hide them (first kept half the time). Output sets `show-number="none"`, and such a group
    takes no bracket. Left out of SER.
  - Small naturals: 30% of crops draw naturals at 0.6-0.85 size (Polish 055/022/087 misreads).
  - Ottava signs in type (Polish 023): 40% of marks-style signs set in a text font (DejaVu Serif or
    Sans, Ubuntu; bold, italic, 1.3-2.3 interlines; `8`, `8.`, `8va`, `8va.`, `8ve`, `8vb`, `8ba` as
    the glyph allows), with headroom so they are not clipped; 30% of lines a chain of slanted dashes.
  - Old-style quarter rests (restQuarterOld) in 20% of crops: OLiMPiC scans draw them; seq512 read
    them as eighth rests, seq513 as half rests (197 of 8,314 visible rests).
  - GrandStaff's distorted images for half its systems; labels with tuplets, 32nds or 64ths twice
    in the mix (6% of PDMX crops, 24% of GrandStaff's).
  - Training keeps a moving average of the weights (decay 0.9995): dev SER, best.pt and
    average_<step>.pt every 10,000 steps hold the averaged weights; last.pt holds both.
  - D done 2026-11-09: `tuplet_numbers` override (kern's `*Xtuplet` / `*tuplet`), 20 entries on 9
    dev pages, each from the image (EXPERIMENTS.md, "seq514"); 055 added (user: no number on the page).
    022 bar 21 added (user: two unmarked triplet groups after a tied quarter). Every dev page
    with gold tuplets is now checked. Publication note on B still to write.

- **MuseScore crops drew no ottava sign where one runs into the next system (fixed 2026-11-09,
  re-rendered, not yet trained).** MuseScore drops an unclosed `<octave-shift>` and draws the
  notes where they sound, while the label gives printed pitches with ottava tokens: 728 of 53,854
  crops, 516 scores. seq514 learned an octave that does not follow the notehead (Polish 012 bars
  6-8, 105 bar 1: no sign, an octave high) and signs where none is drawn (043; 546 tokens against
  510 on OLiMPiC scans). `with_label_ottavas` now redraws each system's signs from its label,
  each closed within the system. The crops from before the fix are in
  `data/seq514_data/musescore_ottava_before_fix`. Lesson: a renderer check by SER (3.7% on 150
  crops) does not see a 1.4% class; check each rare token's crops by eye.
- **Training-crop check by eye, 2026-11-09 (before seq515), each rare token's crops.** Fixed and
  re-rendered: (1) MuseScore 3.6 ignores MusicXML's show-number="none" and draws every tuplet
  number: 642 MuseScore crops labelled hidden numbers it drew. Its crops now label every number
  shown (`show_every_tuplet_number`); hidden numbers come from the Verovio crops, which draw them
  as labelled. Likely a cause of seq514 hiding 23% of printed 3s. (2) Octave clefs (treble with
  an 8, 2.4% of piano parts): labels give sounding pitch with a plain clef while both engravers
  draw the 8 and the notes an octave off. Now drawn plain (`without_octave_clefs`), both
  pipelines. OLiMPiC synthetic and GrandStaff have none. Re-rendered: 4,622 MuseScore scores
  (tuplets or octave clefs), 1,146 per Verovio set; old crops in
  `data/seq514_data/before_clef_tuplet_fix`.
  Checked and right: ottavas (8va, 8vb, 15ma, 15mb, after the fix), tremolos, arpeggios, slashed
  graces, C and cut time, repeats and voltas, hidden rests, rare clefs, stemless notes.
  Left as is: a tie or slur open at a system's end (or arriving from the last one) is in the
  label but drawn by neither engraver, where a print draws a half arc; about 3.4% of crops each
  way. Fixing it means drawing the half arc; not done.
- seq514's quarter-read-as-half on OLiMPiC scans (152 aligned notes against seq513's 110) sits in
  a few crops (6010916/p2-s4, 6499366/p2-s2, 6189628/p2-s4): chords under a slur with a flagged
  eighth after; no cause found yet.
- After each new model: re-measure each kept page-output rule in eval/lmx_tuplets.py (wrong tuplet
  marks cleared, triplets under a second beam, one beam too many) on its readings, and count the
  bars each makes worse against gold, not only the net. A rule whose gain is not significant comes out.
- 096 is still 8.5 worse on Print after D: dense voices, seq514 drops voice-2 rests and writes a
  voice twice in bars 11-12; not investigated further.

- Ottava signs: `8` alone, dotted lines, 8va above the lower staff are now in `build_seq_dataset.py
  --marks` (data/seq3/marks); tested by fine-tune q512: fixes 110, not 023 (see EXPERIMENTS.md). Long plain
  lines between the staves as negatives (049 false sign) added too: `add_plain_line`, a solid
  bracket line over a run of notes, no label, in a quarter of --marks scores; checked by eye.
- Gold: 110 Print standard needs its `*8va` passages (last system, both staves).

- No staccatissimo token: the model cannot write wedges.
- Double/final barlines: `barline-tokens.patch` (apply after seq513). Repeat barlines: `repeat:start` / `repeat:end` tokens (train/seq/repeats.py), in labels from our
  linearizer, drawn back as <barline> by seq_predict.delinearize, left out of SER. Round trip
  checked on 5 PDMX scores. In the run started 2026-11-07 (seq513).
- Double and final barlines: gold has them (69 edits on dev, Print), our labels have no token for
  them (only repeat:start/end). Add `barline:double` / `barline:final` like repeats.py. Next run.
- Handwriting: no sequence model has any. Needs MUSCIMA++ as LMX; Smashcima drew no key signature
  on a 5-sharp system, is slow, and must exclude the test writers.

## Held for the user

- Sealed test run. Numbers only, and only after gold and the model are frozen.
- Headline metric: authors' OMR-NED or `--whole-page-alignment` (or both). Then the protocol text.
- Email to the Transcoda authors: add the bar-pinning finding (017, 042).
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

## Decided 2026-11-08 (user)

- Next run: `time:common` / `time:cut` tokens. Scoring the non-Print standards (As published,
  Corrected) must accept either the symbol or the numbers (like `--either-ottava`).
- Report the authors' OMR-NED and our whole-page alignment side by side; we steer by whole-page.
- Frozen candidate: after seq513's Polish comparison. Commit after that and the open decisions.
- Transcoda email: hold until results are in.
- Hidden tuplet numbers: before scoring seq513, make the non-Print scoring accept either a shown or a
  hidden number (E). Next run adds a `tuplet:hidden` token (B), and the Print gold marks tuplets whose number isn't
  printed (D).
- For publication: B can cost us on the authors' strict OMR-NED wherever their gold shows a tuplet
  number the page doesn't print, since we output it hidden. Say so next to that number. Our reasoning:
  the target is what's printed, and MusicXML can record the hidden number
  (`<tuplet show-number="none">`), so a hidden 3 is the faithful transcription and the rhythm is
  unchanged. Quantify how much of the gap comes from this when seq514 is scored.
