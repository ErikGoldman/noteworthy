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

## 2026-11-04

### Night Album (held out), v5 + key-change rule
- **Note F1 0.992** (633 GT notes, 633 predicted); onset+pitch 0.998; 57/62 measures exact;
  all matched notes on the right staff. Pages 2-4 read in 45 s (8 threads).
- The 5 wrong measures: m0 one note an octave high (B4 for B3); m26, m41 a quarter read as
  an eighth; m43, m60 a long note given the wrong duration. Not investigated, so the
  score stays held out.
- At 633 notes this score can no longer separate models (5 errors); the PDMX benchmarks
  remain the development signal.

### Reproducible pipeline (`./reproduce.sh`)
- One command from a fresh clone: environment, downloads (md5-checked, ASAP pinned),
  score lists, render, MUSCIMA++ conversion, training, export, C++ build, benchmarks.
  Smoke configuration (64 pages, 300 steps) runs end to end in about 10 minutes.
- Recipe = base 60k steps from scratch + fine-tune 20k (lr 5e-4, balanced crops 0.5)
  with the current code. v5's own lineage differed (v2 sigmoid heads on pre-font-fix
  labels → v3 → v5 softmax heads), so a reproduced model will differ somewhat from v5.
- Regenerated PDMX lists are byte-identical; the ASAP list is now sorted (it was in
  filesystem order), which changes which scores the render samples.

### Held-out rule and deduplication (breaks comparability with earlier results)
- Held-out scores are now chosen by hash of file name + size (`train/omr/held_out.py`),
  not absolute path, so every checkout and data location holds out the same scores.
- Copies of held-out pieces are dropped from training (`train/omr/pieces.py`: equal
  distinctive title words, composers not contradicting): 228 of 20,872 listed scores
  (re-uploads and arrangements, e.g. several "Megalovania", Schubert D 899). Lieder
  benchmark candidates exclude any song with a copy in PDMX or ASAP.
- Consequence: v5 trained on 91 of the new verovio/musescore-100 scores, so v5 cannot be
  benchmarked on them. All numbers above are on the old split; the next reference is a
  fresh `./reproduce.sh` model. `data/synth_v2` was sampled from the old list
  (`data/render_scores_synth_v2.txt`) and should not be used for new held-out comparisons.
- Pipeline stages now check their own outputs instead of stamp files: a second
  `./reproduce.sh` downloads nothing, renders nothing, and `train.py` returns at once on a
  finished checkpoint. MuseScore is found via `$MUSESCORE` or PATH.

### Real scans: OLiMPiC (`eval/olimpic_benchmark.py`)
- OLiMPiC (ÚFAL, CC BY-SA): OpenScore Lieder piano accompaniments with the IMSLP scans
  they were transcribed from. Its release also ships the IMSLP page images, system boxes
  and a sample→page map, so both system crops and full pages are scored, all automated
  (replaces the manual IMSLP download plan; IMSLP blocks scripted downloads).
- Dev split, songs with any copy in PDMX/ASAP left out (780 of 1,438 systems; most
  Lieder songs are on PDMX). Test split held out like Night Album.
- v5, systems (658 crops, 36.5k notes): **F1 0.532**; right pitch in the right measure
  regardless of timing 0.875.
- v5, pages (48 songs, the voice staff and lyrics on the page, only annotated piano staves
  scored): **F1 0.565**; right pitch 0.884; 25 extra measures, none missed, so page
  layout holds up. Carrying the time signature across systems is worth +0.03 over crops.
- Error mix is timing: onset 4756, onset+duration 4049, duration 2744 vs missed 1982,
  extra 2345, semitone 1357. Worst songs are implied triplets (6474359: every pitch
  right, triplet eighths read as plain eighths once the "3" stops being printed) —
  the same top failure the OLiMPiC authors report for their end-to-end model.
- Synthetic-vs-real gap: 0.95 → 0.56. Next: implied tuplets (beam groups whose durations
  overfill the beat), then a scan-like augmentation check.
- Recognizer: several inputs are read as consecutive pages; `--layout` writes where each
  system's staves lie on the input pages (used to score only annotated staves).

### Implied tuplets (C++ `imply_tuplets`)
- An unmarked beam group of 3 or 6 equal plain values becomes a triplet when the staff
  overfills its measure and making all such groups triplets fills it exactly.
- v5, verovio-100: 0.937 → **0.938** (040 0.887 → 0.942, 058 0.901 → 0.956, nothing worse).
  Night Album unchanged (0.992). Worst OLiMPiC song 6474359: 0.104 → 0.18.
- A looser test (accept any fill ≥ the measure) catches more scans, where grace notes read
  as 32nds spoil the exact fill (6474359 system p3-s1 0.04 → 0.57), but verovio-100 drops
  to 0.932: on three interleaved voices (017, 0.715 → 0.338) the column timeline overfills
  with no tuplet and plain 16th groups were turned into triplets. Exact kept. The scan
  cases need grace notes recognized as such.

### Zeus comparison (`scripts/compare_zeus.sh`, `eval/zeus_comparison.py`)
- Zeus (released model, CNN + BiLSTM encoder, attention LSTM decoder, LMX output) runs on
  the same OLiMPiC system crops; 1,438 dev crops in 86 s on CPU (SER 15.8%).
- Dev, 658 crops of songs without a copy in our training sources, our note F1:
  **Zeus 0.767 vs v5 0.532**; right pitch regardless of timing 0.902 vs 0.875. Zeus is
  better on 455 crops, v5 on 95. The gap is timing: implied triplets (Zeus reads
  6474359 perfectly), mid-song crops without a time signature (we assume 4/4 and place
  onsets by x over the wrong measure length), mixed 16th/32nd beaming.
- Zeus trained on synthetic renders of other Lieder songs (in-domain), we on PDMX.
- TEDn (the paper's metric, their code) is wired up for the test split; numbers pending a
  reference model. TEDn also counts stems, beams, slurs and articulations, which we do
  not output, so our note F1 is reported for both as well.

### Handwritten benchmark (`eval/handwritten_benchmark.py`, `eval/muscima_truth.py`)
- Goal: handwritten scores at about their real-world share, 5-10% of training pages;
  print remains the gate. MUSCIMA++ was 19% of the mix (140 pages x 20); now 100 pages x 9,
  about 7%.
- MUSCIMA++ splits (omr.muscima): writer-independent test set (20 pages, 7 writers seen
  nowhere else) held out like Night Album; writer-dependent test set (20 pages) is the
  handwritten dev set; both out of training (previously a 5% hash). All writers copied
  the same 20 pieces, so pieces always leak; only the handwriting is unseen.
- Ground truth from the MuNG graphs: pitch by mung2midi (ties dropped first; it rejects
  ties over three notes), durations from each head's links (its own duration inference
  fails on two-stem notes), measures from measure separators, onsets by chaining each
  staff left to right, or per stem direction. A measure is scored only when every staff
  fills the time signature (or, without one, the page's most common fill; pickups at the
  page's ends). 2,138 of 2,862 measures scored. The 7 copies of each piece agree at
  pairwise F1 0.946 on measures both score; the rest are one-note writer/annotation slips
  and a few pieces (05, 17, 18, 20: few or no separators/legible rhythm).
- v5 (which trained on all 140 pages, dev included, so optimistic): **dev F1 0.296**,
  right pitch timing aside 0.601, over 1,610 scored notes on 20 pages. Print is 0.94.
