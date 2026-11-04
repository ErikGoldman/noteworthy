# Experiment log

Benchmarks (all on scores the network never trained on):
- **verovio-100**: page 1 of the first 100 validation-split PDMX piano scores, rendered by Verovio
  (same engraver as training). `eval/synthetic_benchmark.py --count 100`
- **musescore-100**: same scores rendered by MuseScore 4 (different engraver and font).
  `--renderer musescore`
- Metric: note F1 on (measure, onset, pitch, duration). `eval/diagnose.py` aligns measures and
  breaks errors down by kind (onset, pitch, octave, ...).
- Night Album ("Sleepless", `eval/sleepless.gt`) is held out and not used for tuning.

## 2026-11-03

### prelim model (runs/prelim: 8k steps, ~1.7k pages, 42 channels)
| change | verovio-100 F1 | musescore-100 F1 |
|---|---|---|
| baseline | 0.794 | — |
| time-signature digit grouping overflow fix (every signature read as 0 → 4/4) | 0.786* | — |
| time/key signatures read per measure; touching digits split at middle line | 0.799 | 0.802 |

\*Correct signatures exposed scores whose 4/4 fallback had masked mid-system changes.

Findings:
- MuseScore ≈ Verovio: engraver change costs nothing measurable; errors are pipeline/model.
- Error mix (verovio-100, aligned accuracy 0.836): onset 1741, pitch 1213, missed 787,
  extra 506, octave 396.
- Labeling bugs found in training data: mid-system clef changes (SMuFL E07A–E07C, 14% of clefs),
  tuplet numbers and ottava lines were all labeled `other`. Fixed; added channels
  `tuplet_3`, `tuplet_other`, `ottava` (45 channels). v1 run aborted at 4k steps.
- Rests: the network fires rest_8/16/32 together on one eighth rest → duplicate items break
  rhythm chains (score 005).
- 4/2 time signature: prelim net can't tell the 2 (rare as denominator).
- Octave-transposing clefs (G8vb) unsupported (score 037).

Infrastructure:
- Dataset renderer ~6× faster (in-memory cairo surfaces, SVG parsed once per page).
- Datasets store laid-out SVGs; `train/relabel_dataset.py` regenerates masks after
  labeling changes without Verovio.

### pipeline fixes (prelim model, spot checks)
- Rival-kind suppression (rest_8/16/32 etc. firing on one symbol): of 6 spot-checked scores,
  4 improved (005 0.260→0.296, 045 0.426→0.434, 075 0.697→0.728, 084 0.828→0.847),
  063 unchanged, 021 0.368→0.344.
- Barlines accepted when they bridge two staves of the system even if only partly
  predicted (a missed barline shifted every later measure): 005 0.296→0.524.
- Clef changes now recorded at any position (MusicXML inline attributes); opening clefs no
  longer depend on the detected staff start.
- Dataset build: a few scores hang Verovio in C++ (SIGALRM cannot interrupt); the parent now
  stops waiting after 300 s instead of the whole build stalling at 11995/12000.

### v2 training (runs/v2)
- Data: data/synth_v2 (11995 pages, 45 channels, corrected clef-change labels, tuplet/ottava
  channels) + MUSCIMA ×20. 60k steps, batch 16, crop 384, lr 2e-3 one-cycle.
- ~100 crops/s on a quiet machine (vs 22 crops/s when the CPU was shared).
- Baseline after pipeline fixes, prelim model: **verovio-100 0.811, musescore-100 0.811**
  (from 0.799 / 0.802). Onset errors 1741 → 1328.
- v2 @10k steps (verovio-100): 0.803. 8va handled (score 008 first measures now exact,
  0.175 → 0.316); triplets and inline clef changes appear in the output.
- Semitone errors rose to 772 at 10k: stray clef_c specks on time-signature digits counted as
  a clef, which pushed the key-signature search window past the key accidentals of the
  first system. Fix: clef blobs must be ≥ 2 interlines tall (smallest real clef ≈ 2.4).
  Spot checks @10k: 033 0.63→1.0, 042 0.46→1.0, 007 0.47→0.97, 066 0.63→0.92, 047 0.81→1.0.
- Class frequencies (share of pages containing the class): ts_0 0%, tuplet_other 0.7%,
  ts_cut 0.7%, clef_c 0.8%, flag_32 1.7%, ts_common 2.7%, rest_32 3.2%, ottava 4.3%,
  double_flat 5.5%. Added `--balanced-crops` (crops centred on a uniformly chosen class
  present on the page) for the next run; in a smoke test it doubles rare-class crops.
- v2 @20k steps (with clef height filter): **verovio-100 0.872, musescore-100 0.852**.
  Plain pitch errors 1247 → 23; onset errors now dominate (1174 / 1413).
- Tuplets: tuplet_other (val F1 0.00 at every eval; 0.7% of pages) fires on 3s with mass
  equal to tuplet_3, so rival suppression dropped real triplets. Only tuplet_3 is extracted
  now (075 0.587→0.755, 083 0.573→0.787). A triplet beamed together with plain notes is
  the 3 notes centred under the number (075 → 0.79).
- Time signatures decoded under constraints: each glyph keeps per-digit probability mass and a
  number is the allowed value with the highest total (beats 1–32, beat types powers of two).
  063 (4/2, the 2 read as 3): 0.58 → 0.819. Remaining: 6 vs 9 near-ties at 20k (066).
- Found: Verovio sets music symbols inside text (the note of "♩ = 120") in the SMuFL font
  as text; with no such font installed cairo drew an empty box, so training never showed
  a tempo-mark note. MuseScore prints a real note there and the network reads it as one
  (069: spurious first note shifts every onset of the measure). Fix: omr/fonts.py extracts
  Verovio's embedded WOFF2 fonts into data/fonts and points fontconfig at them.
  relabel_dataset.py now rewrites images as well as masks; synth_v2 to be re-rasterized
  after the v2 run. (The verovio-100 benchmark renders still show the box; kept for
  comparability until re-baselined.)
- Rests carry a voice side (raised → upper voice, lowered → lower voice, by engraving
  convention), used with stem direction for polyphony detection and voice chains.
  Spot checks: 003 0.712→0.998, 086 0.697→0.848.
- v2 @20k with tuplet/time-signature/voice-side changes: **verovio-100 0.887 (from 0.872),
  musescore-100 0.861 (from 0.852)**. Onset errors 1174 → 902 (verovio).
- MuseScore draws ottava extensions as thin, sparse dashes; v2 only finds the "8", so the
  shift covered one note (088: 42 octave errors). Rasterization now varies the ottava dash
  pattern and stroke width per page (seeded from the page SVG, so relabeling reproduces
  it); C++ joins dashes across gaps up to 2 interlines.
- Key signatures validated by spelling: accidentals must follow the standard order
  (sharps F C G D A E B, flats B E A D G C F) at the clef's steps. Mid-system changes need an
  exact spelling or naturals cancelling exactly the current key (064 flipped -2/+1 every
  measure from note accidentals after barlines).
- A head in reach of several accidentals takes the nearest (a key-signature flat just before
  the first note used to override the note's own natural).
- Barlines with no note or rest before them (start-repeat after clef/key/time) no longer
  open an empty measure that shifted the whole page: 064 0.30 → 0.90.

### v2 final (runs/v2: 60k steps; val mean F1 0.712 @60k, best 0.721 @30k)
- model_last (60k), before the key fix below: **verovio-100 0.920**, musescore-100 0.856.
  The verovio jump (0.887 → 0.920) is mostly the opening-barline fix (whole-page measure
  shifts); semitone errors on musescore doubled (242 → 521).
- Cause: strict key spelling at system starts. MuseScore's flats (Leland) put the note
  reference a step off for this pipeline, so the ordered run stopped early (4 flats read
  as 2). System starts now count accidentals again; strict spelling is kept for
  mid-system changes only. MuseScore spot checks: 077 0.675→0.795, 099 0.771→0.922,
  059 0.770→0.919, 080 0.307→0.433.
- synth_v2 re-rasterized (fonts + ottava line styles); mask packing 8.3 s → 2.3 s per page.
- v3 started: fine-tune of v2 last.pt, 20k steps, lr 5e-4, --balanced-crops 0.5.
- **Root cause found for rare-class false positives.** v2 @60k vs @20k with the same binary:
  verovio-100 0.920 vs 0.921, onset errors 1346 vs 858. 30% of first time signatures were
  misread (6/8 → 9/8 or 5/8, 4/4 → 9/1 or 5/1). On a "6", ts_5, ts_6, ts_9, ts_common and
  ts_cut all fire with near-equal mass, though the labels never overlap. The soft Dice term
  was computed only for channels present in the batch, so rare channels were almost never
  penalized for false positives (BCE averaged over 45 channels × all pixels is too weak).
  This also explains tuplet_other on 3s and clef_c specks on digits. Fix: Dice over all
  channels (absent → 1 − 1/(mass+1)). v4 = fine-tune of v3 with the corrected loss.

### v3 (runs/v3: v2 + 20k fine-tune, balanced crops 0.5, re-rasterized data with fonts and
### ottava styles; old present-only Dice)
- **verovio-100 0.941, musescore-100 0.926** (v2 final: 0.920 / 0.856). MuseScore onset
  errors 1267 → 501.

### v4 (aborted at 10k): all-channel Dice over-corrected
- The 6 vanished entirely (ts_5/ts_6/ts_9 val F1 0.00; 6/8 read as 4/4): penalizing rare
  false positives made the network give up on confusable classes.

### v5 (running): exclusive-family softmax
- Mutually exclusive glyph families (heads, flags, accidentals, clefs, rests, time-signature
  glyphs, tuplet numbers) share a softmax with an implicit "none" class (omr/heads.py);
  Dice back to present channels only. ONNX now outputs probabilities (C++ no longer applies
  a sigmoid); old checkpoints re-export with plain sigmoids — v3 re-exported, outputs
  unchanged on spot checks. Fine-tune of v3, 20k steps, balanced crops 0.5.
- v5 @5k: val F1 ts_5 0.87, ts_6 0.86, ts_9 0.87 (v3: 0.70 / 0.35 / 0.28), tuplet_other
  0.35 (never above 0.09 before). Score 095 now reads 6/8 (v2/v3: 9/8 or 5/8).
- **v5 final (20k): verovio-100 0.947, musescore-100 0.928** (v3: 0.941 / 0.926); val mean F1
  0.796 (v3 0.740). Onset errors 782 → 625 (verovio), 501 → 466 (musescore).
  First-time-signature misreads 30/100 → 2/100 (verovio), 1/97 (musescore); the rest are
  cut time read as common time, which gives the same measure length.
- Mid-system key changes must cancel (with naturals) every accidental of the old key they
  drop, as engravers print them; a lone note accidental after a barline spelled a 1-sharp
  key (083 flipped 4 → 1 sharp mid-system). 083 0.785 → 0.834. All 5 genuine first-page key
  changes in verovio-100 still read (005's two measures late).
- v5 + key-change rule: **verovio-100 0.948, musescore-100 0.928** (aligned accuracy
  0.951 / 0.954). Remaining onset errors are now case-specific (triplet placement in 083,
  a chord split across two values in 034, column merging in 019), not one systematic cause.

## Summary of the session (2026-11-03/04)
| model + pipeline | verovio-100 | musescore-100 |
|---|---|---|
| prelim, start of session | 0.794 | — |
| prelim, pipeline fixes | 0.811 | 0.811 |
| v2 final | 0.920 | 0.856 |
| v3 | 0.941 | 0.926 |
| **v5 (current)** | **0.948** | **0.928** |
