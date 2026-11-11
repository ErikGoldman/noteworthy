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
- Exact rule, rest of the A/B (v5): musescore-100 0.925 → 0.925 (tolerant 0.921); OLiMPiC
  dev pages 0.565 → **0.569** (tolerant 0.579: +0.01 on scans, but −0.005/−0.004 on the
  synthetic benchmarks from wrongly tripleted polyphony; exact kept).

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

## 2026-11-04 (afternoon): system-crop transformer (train/seq, train_seq.py)
- Model: ResNet encoder (stride 32 rows x 16 columns, 384 channels) + 2 transformer
  encoder layers over the 2D grid (learned row + sinusoidal column positions) + 6-layer
  transformer decoder (d 384, 8 heads) emitting OLiMPiC's LMX tokens; 26M parameters.
  Height 256, greedy decoding. Augmentation: random vertical margins, rotation ±1°,
  horizontal scale 0.85-1.15, stroke weight, binarization, capture_degradation.
- Data: OLiMPiC synthetic (15k Lieder accompaniment systems, MuseScore renders; Zeus's
  training set), GrandStaff-LMX (41.6k piano excerpts, Verovio renders; images from
  grfia.dlsi.ua.es), and our own PDMX system crops (build_seq_dataset.py: two-staff
  parts cut into 2-6-measure systems with OLiMPiC's split_part_to_systems, rendered alone
  with Verovio under random fonts and engraving options, LMX by OLiMPiC's linearizer).
- Scored on OLiMPiC scanned dev crops (eval/seq_predict.py writes Zeus-layout
  predictions; note F1 by olimpic_benchmark.py --predictions on the 658 crops of songs
  without a PDMX/ASAP copy; SER over all 1,438).

### Smashcima first look (not yet used)
- smashcima 1.1.2 installs, but: its MUSCIMA++ v1.0 download URL (LINDAT) now returns an
  HTML page (the GitHub v1.0 release works); it needs OpenCV 4 contrib only (OpenCV 5
  changed findNonZero's shape; plain + contrib OpenCV together lose cv2.ximgproc).
- Renders a whole page per call (8-18 s on one core, even for one system). Picks a random
  writer from all 50 MUSCIMA++ writers, including our 7 test writers: its writer domain
  must be restricted to training writers.
- On an OLiMPiC system in 5 sharps it drew no key signature, so image and LMX disagree;
  usable as-is only for systems without key signatures. Parked behind the printed
  transformer experiments.

### State when paused (15:26, the user needed the PC)
- PDMX system crops: 67,361 systems from 20,644 training scores (data/seq_pdmx). Three
  scores hung inside Verovio and blocked the pool; the builder was stopped and the list
  written from the finished directories. TODO: a per-score watchdog like build_dataset.py's.
  Forked workers lose Verovio's fonts ("font resources are not available"): spawn.
- Run A (OLiMPiC synthetic x2 + GrandStaff, 80k steps planned): stopped at step 8,500;
  last checkpoint step 5,000, dev SER 46.4% on the first 300 scanned dev crops (Zeus on
  all dev: 15.8%). Resumes from runs/seq_a/last.pt. It ran at 125 samples/s alone and
  60-80 while the crop builder shared the CPU.
- Next: finish run A, score it against Zeus on the 658 clean dev crops (note F1, SER),
  then run B with the PDMX crops (monitored with --dev-samples data/olimpic_dev_clean.txt,
  since PDMX holds copies of other dev songs).

### Resumed (evening): run B first
- The per-score watchdog is in (d4c18f1). Run B started before run A, since it is the
  configuration the pipeline ships (scripts/train_seq.sh → runs/seq): synthetic x2 +
  GrandStaff + PDMX crops, 139,018 training samples, 80k steps, dev = the 658 clean dev
  crops. 123 samples/s, about 4.3 h. Run A (Zeus's training data only) stays the
  ablation, parked at step 5,000.
- Lost ~15 min: training crawled (100 steps in 9+ min) right after a side evaluation of
  run A on the GPU ran out of memory, until restarted from step 5,000. The cause was not
  established: memory spilling to shared system memory was a guess, never measured, and
  training later slowed again (52 s per 100 steps, step ~14,600) with no side job. Resuming
  also failed: the checkpoint's args held a Path, which torch.load's weights_only default
  rejects (fixed, 02ad620). Keep other GPU jobs off this 16 GB card during training.
- Run B step 5,000: dev SER 70.3% on the 658 clean crops (run A, step 5,000: 46.4% on
  the first 300 of all dev crops; not the same crops, so not comparable yet).

### ONNX export of the sequence model (export_seq_onnx.py, 57b632b)
- encoder.onnx + decoder.onnx (next-token logits from the whole prefix, no KV cache),
  contract in metadata (vocabulary, bos/eos, height, width multiple, input encoding).
  The TorchScript exporter froze the encoder's cell count into the attention reshapes;
  the dynamo exporter (onnxscript) keeps it dynamic. Greedy tokens equal PyTorch's on
  real crops (192 and 177 tokens).
- CPU speed without a KV cache: ~6 s per system (while training shared the CPU); a page
  of 6 systems would take ~40 s. A KV-cache decoder is needed before shipping in C++.
- Run B step 10,000: dev SER 28.6% on the 658 clean crops (from 70.3% at 5,000).

### WSL crash, 18:47: a scoring bug, not training
- Run B step 15,000: dev SER 21.0% on the 658 clean crops. Restarted there with a PyTorch
  memory cap (GPU_MEMORY_FRACTION 0.85); it ran at ~104 samples/s to step 18,200. The cap
  is an unproven remedy: any restart also restored speed for thousands of steps.
- Scoring Transcoda's handwritten pages took one Python to 43 GB (kernel log: global OOM,
  anon-rss 43,074,688 kB); WSL shut down at 18:49. Lost: steps 15,000-18,200.
- Cause: align_measures compared costs with == on a border computed as i * gap, which
  rounds differently from (i - 1) * gap + gap from i = 7. With few predicted measures the
  walk back left the table: an endless loop on w05_p011, IndexError on five other pages.
  Fixed (9179997), and those pages now score. The bug could also have mis-paired measures
  silently in earlier benchmarks, wherever 7+ leading ground-truth measures went unmatched.
- Lesson: run new scoring code under ulimit -v and timeout beside long jobs.

### Transcoda against ours (eval/transcoda_predict.py)
- btrkeks/transcoda-59M-zeroshot-v1 at b529f8a, its own venv, CPU, beam 3, no grammar
  constraint (the model card's headline setting): 17-49 s per printed page, ~90 s per
  handwritten page with training beside it. **kern to MusicXML by music21, scored by our
  note F1. Pages keep only what precedes the first break of **kern's spine rules.
- Control, 6 held-out Verovio pages: Transcoda F1 0.968, v5 0.982 (2,019 notes). The
  conversion and scoring work. Small, and Transcoda trained on PDMX and on Verovio renders,
  so this checks the path and ranks nothing.
- Handwritten dev, 20 pages, 1,610 notes: Transcoda **F1 0.031**, pitch aside from timing
  0.104. 7 pages give no scorable notes: all rests, grace-note loops, or invalid **kern.
  It never trained on handwriting; v5's 0.296 is on pages it trained on. Neither number
  says who reads handwriting better on unseen pages.
- Night Album, one run, numbers only (633 notes): v5 F1 0.992 by both scorers. Transcoda
  0.379 with measure alignment, 0.046 by evaluate.py's exact measure index. One piece, and
  its errors stay unread, so how much of the gap is the model and how much is joining
  its four pages into one score is unknown.
- Run B step 20,000: dev SER 18.0% on the 658 clean crops (21.0% at 15,000).
- Run B step 25,000: dev SER 12.7% on the 658 clean crops (18.0% at 20,000). Zeus's 15.8%
  is over all 1,438 dev crops, so not yet comparable.
- Run B step 30,000: dev SER 11.9% on the 658 clean crops (12.7% at 25,000).

### Scorecard (eval/scorecard.py): one row per kind of input, ranges by piece
- Benchmarks write results.json (per-item tallies with their piece); the scorecard gives
  F1 with a 95% range from resampling pieces, and with a baseline the change on the same
  resampled pieces. Held-out PDMX now has a dev (first 100, the pages developed against)
  and a test split (next 100 sharing no piece with them, not yet run).
- v5 on dev: rendered Verovio 0.938 [0.914, 0.958] (100 pieces); real scans, crops 0.537
  [0.464, 0.605] (48 songs); real scans, pages 0.569 [0.494, 0.642] (48 songs);
  handwriting 0.296 [0.214, 0.386] (20 pieces, trained on, so optimistic). Night Album 0.992
  is one piece and has no range. MuseScore 4 is not installed here: that row was not run.
- Crops read 0.537 where 0.532 was logged earlier: not explained yet.
- Zeus re-run on dev, scored the same way: crops F1 0.767 [0.699, 0.828], +0.230
  [+0.155, +0.318] over v5, reproducing the earlier 0.767. Zeus's SER on the 658 clean
  crops: 14.9% (15.8% on all dev), the like-for-like figure for run B's 11.9% at step
  30,000. Caveats: run B's best.pt is picked on these crops, and its SER drops gold tokens
  outside the vocabulary. Note F1 is the deciding number and comes when training ends.

### Mutopia as a source of non-Lieder print: prototype on 5 piano pieces
- Mutopia: at least 515 piano pieces (LilyPond, CC or public domain) with PDF and MIDI. 314
  single-file ones pass omr.pieces, but that filter is too loose for a test set: it
  passed Chopin's Prelude 14 and Beethoven Op. 49 No. 2 on title wording. A test set needs a
  stricter check, by composer and catalogue number.
- Truth from the MIDI (notes, durations, time signatures) plus \partial from the source,
  compared with v5 reading the PDF: bars equal on 4 of 5, off by one on the other; notes
  within 6; pitch in the right measure 0.955-1.000.
- F1 0.993 Chopin, 0.961 Beethoven, 0.879 Kumar, 0.661 Burgmuller, 0.236 Schumann. In the two
  worst, traced by hand, the errors are v5's: unmarked triplets read as eighths
  (Schumann, Kinderscenen 1), and 50 invented 1/6 durations in a piece with no tuplets
  (Burgmuller, Op. 100 No. 18). Not imply_tuplets: with that call disabled Burgmuller stays
  0.661, while Chopin's Prelude 14, all unmarked triplets, falls from 0.993 to 0.005, so the
  rule earns its place. Likely a fingering 3 read as a tuplet numeral: the source has one
  on the first note of such groups (r16 g-3 ( fis e)), 10 in all, against 12 measures with
  invented triplets. Consistent, not proven; no fix tried.
- MIDI repeats were not unrolled in these 5; other sources use \unfoldRepeats (see below).
- Run B step 35,000: dev SER 10.6% on the 658 clean crops (11.9% at 30,000; Zeus 14.9% on the same crops).
- Run B step 40,000: dev SER 10.8% on the 658 clean crops (10.6% at 35,000): first step without improvement.
- Run B step 45,000: dev SER 10.2% on the 658 clean crops (10.8% at 40,000).

### Mutopia benchmark (eval/mutopia_benchmark.py, "print, varied repertoire" row)
- Selection (select_mutopia.py, run once, list committed): composers absent from ASAP and
  GrandStaff, then a melody check against all 20,644 training scores (top-note interval
  8-grams; 10% shared with one score makes a copy). It flagged 15 of 43 candidates that
  title matching had passed, up to exact copies, including Burgmuller's Op. 100 No. 18.
- Two earlier scores (0.623, 0.624) were wrong and are void: the truth failed on unfolded
  repeats, cadenzas, "\partial8" without a space, mid-piece pickups, grace notes (they
  sound as notes and shift their neighbours) and bar-line-free scores. Sources with these
  are now rejected, from the source and MIDI only, plus any piece with onsets off a written
  grid. So the set tests no grace notes, free time or played-out repeats.
- Composers whose results were read stay in dev; test is only composers never looked at.
  Result: 10 dev pieces, 7 test pieces from 6 composers.
- v5 on dev: F1 0.646 [0.585, 0.774], 8,176 notes; pitch in the right measure 0.948. Bars
  agree within 4 on every piece. Timing is the gap: sampled by hand, Sousa's errors are v5's
  (eighth rests between 6/8 chords dropped, so later onsets shift). Rendered print at 0.938
  did not show this.

### Melody check of the OLiMPiC clean dev songs
- Staff by staff against all training scores. One real copy: Haydn's The Mermaid's Song
  (6472842), retitled in PDMX, 60% of its fragments, 32 crops. Now excluded: 626 crops, 47
  songs. Lower shares are mostly chance on repeated accompaniment figures (Brahms's
  Anklange "matched" a Minecraft parody), so 10% does not transfer from Mutopia here.
- v5 crops 0.536 [0.463, 0.607], pages 0.570 [0.493, 0.645]; Zeus crops 0.767 [0.693, 0.831];
  Zeus SER 15.15% on the 626. Run B's logged SER still covers the 658, Haydn included.
- The test split has not had this check.
- Run B step 50,000: dev SER 10.1% on the 658 crops (10.2% at 45,000).
- Run B step 55,000: dev SER 9.0% on the 658 crops (10.1% at 50,000).
- Run B step 60,000: dev SER 8.2% on the 658 crops (9.0% at 55,000).
- Run B step 65,000: dev SER 8.0% on the 658 crops (8.2% at 60,000).
- H3, stemless heads read as quarters: a whole note has no stem, and a LilyPond whole read as
  filled became a quarter. Any stemless head is now a whole. Mutopia dev 0.791 -> 0.799
  (Little Toy Lost 0.653 -> 0.881), no piece worse. Scan regression check pending.
- Real scans: crops are bimodal (80 above 0.9, 99 below 0.2). Songs with 30%+ tuplets, 3
  songs, 3,815 notes: F1 0.195, tuplet share 0.74 in truth, 0.07 in output. Rule misses them:
  a bass note plus five eighths under one beam is not a group of 3 or 6, and the long note
  breaks the timeline. Needs two voices under one beam: left alone, as the transformer
  would replace this code. Kumar's vase (atonal, 208 semitone errors) left alone too.
- Run B step 70,000: dev SER 8.2% on the 658 crops (8.0% at 65,000): levelling off.
- Run B step 75,000: dev SER 8.2% on the 658 crops: flat since 65,000.
- Run B done (80k steps; best.pt step 65,000, dev SER 8.0%, last 8.4%; SER 8.55% over all 1,438
  dev crops, Zeus 15.8%). **Note F1 on the 626 clean crops: 0.878 [0.822, 0.924]**: Zeus 0.767,
  v5 0.536; +0.111 [+0.074, +0.156] over Zeus, +0.343 [+0.275, +0.416] over v5.
- Lead holds on the 34 songs outside the melody check's printed top 14 (2 more at 10%+ were
  not printed): run B 0.880, Zeus 0.764, v5 0.585; +[0.070, 0.171]. Not explained by copies.
- H3 passes the full check: real scans crops +0.007, rendered print +0.003, handwriting -0.006
  (all in noise). H4, beams or flags outrank a head read as hollow (scan song 6499366: sixteenths
  read as halves): crops 0.544, +0.009 [0.000, +0.020]; Mutopia dev 0.803, +0.158 [+0.032, +0.312]
  over v5; rendered 0.941; handwriting 0.290.

### Sequence model on whole pages (eval/seq_pages.py)
- Layout from the recognizer finds systems; last two staves cropped, cut at the emptiest row
  towards each neighbour; transcribed; measures joined. A first 0.545 was void: layout
  coordinates are input pixels, debug pages are normalized, so crops were misplaced.
- Run B on Mutopia dev, no tuning for pages: F1 0.750 [0.549, 0.942], right pitch 0.920; 8 of 10
  pieces at 0.85+, Handel 1.000. Kumar's two pieces (5/4, 8/4, 12/4; a third of the notes):
  0.37, 0.42. +0.059 [-0.024, +0.163] over Transcoda; v5 with its fixes is 0.803, tuned here.
- H5, a model's backup falls short: on Kumar's 5/4 and 12/4 the left hand came after the right.
  99.55% of gold backups return to the measure start (7,708 of 7,743), so delinearize now
  forces that. Run B crops 0.878 -> 0.916 [0.871, 0.953], +0.038 [+0.023, +0.057], no retraining.
  The same rewind lifts Zeus 0.767 -> 0.803; run B's lead stands: +0.113 [+0.080, +0.155].
- Mistake found: build_seq_dataset read render_scores.txt whole, and that list keeps held-out
  scores. 3,325 of 67,392 crops (4.9%, 862 scores) were held out, so run B saw rendered-print
  benchmark pieces: its rendered row cannot be trusted. OLiMPiC and Mutopia unaffected (both
  screened against the full list). Builder fixed; a clean run needs training from scratch.
- Remaining run B error: median crop 0.982, 413 of 626 at 0.95+; four songs hold most of it,
  unmarked tuplets written as plain notes. H6: renders always print the number, so the model
  never learned tuplets without one. Testing: tuplet systems drawn with numbers hidden.
- Run B on OLiMPiC whole pages (layouts from v5's page run, rewind fix): **F1 0.867 [0.821,
  0.909]**, right pitch 0.945; +0.297 [+0.245, +0.345] over v5's 0.570. Crops 0.916, so the layout
  step costs about 0.05.
- Run B on Mutopia dev with the rewind: 0.843 [0.731, 0.946], Kumar 0.64, 0.66; +0.152 [+0.062,
  +0.242] over Transcoda's 0.691; +0.039 [-0.014, +0.083] over tuned v5. The rewind came from
  these pieces, so the Mutopia test split decides.

### Unmarked tuplets (H6, H7)

- Crop builder stalled for two hours: CairoError cannot be unpickled, so one reaching the pool
  killed its result thread. seq_tuplets: 10,491 crops of systems with tuplets, numbers hidden
  (first kept half the time). seq_pdmx lost its 3,325 held-out crops.
- H6 (renders always print the number, so train on crops without): run C, 25k steps from run
  B, tuplets at weight 2, dev SER 7.69%. Mostly refuted: crops 0.916, +0.001 [-0.007, +0.008];
  tuplet-token recall 0.743 to 0.760, precision 0.898 to 0.924. `6in4` stays 0 of 521 although
  the training data holds over 12,000. Crops flip all or nothing between runs (6474359/p8-s1
  went 1.000 to 0.000). Pages +0.024 [-0.001, +0.055], Mutopia +0.010 [+0.001, +0.016] may
  be extra training rather than tuplets: control run (same steps, no seq_tuplets) pending.
- The crops show why: pitches and beams are read perfectly, only the ratio is dropped; no
  number is printed, and 0 of 24 crops of 6801790 show a time signature. Only the bar's
  arithmetic tells.
- H7 (eval/lmx_tuplets.py, restore tuplets where a voice overfills and 3/6-note groups fill
  it exactly): first version F1-neutral, gains cancelled by one song (6982743: rest + three
  16ths, read as a triplet). Added: a tuplet starts on a multiple of its span; the bar is at
  least the longest voice without candidates. On 4,000 OLiMPiC training crops 4 correct ones
  change; 823 of 1,924 measures with stripped tuplets return. Measures where all voices are
  unmarked stay out of reach of a lone crop.
  Run C + H7 vs run C: crops 0.924, +0.008 [-0.000, +0.025]; pages 0.896, +0.005 [-0.008,
  +0.018]; Mutopia 0.851, -0.001 [-0.003, +0.000]. Tuplet recall 0.827, precision 0.925. The
  rules were tuned on dev crops; Mutopia, not tuned on, shows no gain. A narrow fix for one
  engraving habit, not a general one. Page rows rescored from saved LMX.

### Control, crop framing and reading selection (H8, H9)

- Control for H6: 25k steps from run B without seq_tuplets, dev SER 7.57%. Run C (with) beats
  it by +0.002 crops, +0.006 pages, +0.001 Mutopia, all inside noise. H6 refuted outright:
  run C's gains over run B came from more training. Only the tuplet repair (H7) is real.
- H8: page crops were tighter than OLiMPiC's, 1.53 against 2.06 times the system height at
  the median of 643 crops, so staves reached the model about 40% larger. White padding, run
  C + H7 on pages: 1.8 gives 0.909, 2.06 gives 0.915, 2.3 gives 0.914 (from 0.896); none
  clear of noise. Mutopia falls: -0.009, -0.008, -0.038. Padding is not uniformly good: 33
  songs rise and 12 fall, Traviata_08 loses 87 notes, 6011095 loses 248, while Kumar gains 48.
  Staff size at the input does not explain the split (Mutopia 10.7 px, OLiMPiC pages 11.0).
- H9: read each system padded and unpadded, keep the reading whose voices fill their bars
  (lmx_tuplets.bar_fill, no ground truth needed; padded wins ties). Failing reads break bars:
  6011095 falls from 1.00 to 0.38 on several systems when padded.
  vs unpadded: pages 0.925, +0.030 [+0.015, +0.051]; Mutopia 0.860, +0.009 [+0.001, +0.018].
  Four framings: 0.931 and 0.861 at twice the cost; not taken. Reproduced end to end.
  vs run B: pages +0.059 [+0.025, +0.099], Mutopia +0.018 [+0.001, +0.031]; crops +0.009
  [-0.004, +0.026]. Selection doubles inference. Pages now match the crops (0.924).

### OMR-NED on Transcoda's Polish benchmark (eval/omr_ned.py)

- Transcoda's own metric, on its own real-scan benchmark: btrkeks/polish-scores at 5542822,
  112 pages, 105 with two spines, evaluation only. Treated like Night Album: never trained or
  tuned on, numbers only. Their omr_ned.py runs unchanged, versions from their uv.lock.
- Its published gold cannot be scored as it stands: 34 of 112 pages score 100 against
  themselves (mean 30.36). The kern has no header or terminator, _ensure_kern_header sizes the
  terminator from the first line, these pages end with another spine count, and the strict
  gold parse then returns an empty score. Written out correctly, gold against itself is 0.00.
- Transcoda at b529f8a, beam 3, CPU, all 112 pages, no page failed to score:
  gold as published 65.50 (published: 63.97); terminated gold **51.30**. The 78 unaffected
  pages score 51.75 either way. So the published figure very likely carries the bug, and the
  model is about 14 points better than reported. 51.30 is the bar. The 1.53 left over is
  unexplained; the paper speaks of 102 scans, the file holds 112.
- Published gold MusicXML has no beams (357 of 419 differences on a page), so it differs from
  the kern by 9.44 and is not a fair target. Our MusicXML is compared directly with the gold
  kern: the beamed conversion of the gold scores 0.54 that way (median page 0).
- A first scoring of Transcoda gave 74.05 and 62.59: 31 pages had no prediction, after my
  MusicXML export crashed two shards behind an exit code of 0. Void.

- H10 (train on crops cut near the system at empty rows, so framing stops mattering): run D, as
  the control but for that. Dev SER 7.81% (control 7.57%). One read: crops 0.926, pages 0.896,
  Mutopia 0.861, each within noise of the control's. Two reads: pages 0.924, Mutopia 0.860,
  -0.002 [-0.010, +0.007] and -0.001 [-0.007, +0.005]. Two reads still give pages +0.028, so
  the training change does not replace them, and run C, run D and the control tie. Dropped.

- Synthetic plumbing test, 20 two-spine pages of verovio-synth-omr (not a result: its sources
  overlap our training): 8 pages came out empty. estimate_interline's global threshold loses
  pale staff lines, so the scale lands far off (0.26 to 1.0 where 1.9 is right) and the
  segmenter finds no staves. A threshold-free estimate from row autocorrelation did well on
  OLiMPiC's scans (2 of 171 pages off by over 5%) but not on these pale pages. Instead
  main.cpp retries pages with under two staves at fixed interlines, then sets the scale from
  the staves found. Plumbing 45.64 to 13.45, median about 2.5; one page still fails.

- **Run C + H7 + H9 on Polish, one run, numbers only: OMR-NED 37.24** (median 34.88), all 112
  pages output, none failed. Transcoda on the same gold: 51.30 (median 50.80). Paired, ours
  minus Transcoda: -14.06 [-18.51, -9.49], better on 78 of 112 pages. Against the gold as
  published, with its 34 broken pages at 100: 57.29 against 65.50.
  Three-spine pages (7), which we read only two staves of: 62.99 against 57.54.
- Transcoda's kern path gets a wrong terminator on its predictions too, and syntax fixes count
  as errors (102 of 112 pages), while our MusicXML has neither. Correct terminators: 51.28.
  Syntax penalty removed as well, kinder than its own metric: 48.72, still -11.48 [-15.89,
  -6.99], better on 75.
- Not settled: 120M variant unscored (60.7 published on the flawed gold; weights not on
  Hugging Face under the author's account). No check yet that Polish pieces are absent from
  GrandStaff or PDMX, which both systems trained on. 112 pages from one archive.
- Overlap of Polish with our training data: select_mutopia's melody check (8 intervals of the top
  notes, crops pooled by piece) against PDMX/ASAP render list, GrandStaff and OLiMPiC, 92,471
  files, 2,257 unparsed. Best share per page: median 0.03, max 0.20; none at 0.26, where copies
  began. 7 pages reach the 0.10 threshold; on the other 105, ours 37.91, Transcoda 52.02,
  -14.11 [-18.62, -9.28]. Overlap does not explain the lead. Musetrainer and OpenScore
  Quartets, which only Transcoda trained on, were not checked.
- SOTA checks: the 120M variant's weights are not public (no GitHub release, only the 59M
  checkpoint on Hugging Face), so it cannot be rescored. Legato is public but reports 86.73,
  far behind. Legato 2 appears in a January 2026 article; a search found no weights, and no
  OMR-NED figures. "M2-OMR" could not be found. PRAIG/SMB (Sheet Music Benchmark: 685 real
  scans, 469 pianoform, CC BY-NC) would be a far larger test, but is gated: manual approval
  under a person's own account. Its KernScores sources overlap GrandStaff, so it would need
  the melody check first.

### Toward a credible claim

- PDMX flags scores whose public license disagrees with the one in the file, and its authors
  recommend leaving them out. 2,896 of our 20,456 PDMX scores carry the flag, 7,278 of 64,067
  crops (11.4%). The builder now skips them, and the README overstated this. The clean
  retrain restarted on 56,789 PDMX crops. Run C and the U-Net model still contain them.
- Melody check on OLiMPiC test: 100 songs, best share max 0.195, 4 at 0.10, none at 0.26. The
  same script on dev finds the known copy at 0.81, so it can find copies. It also finds
  6654137 at 0.48, which had passed the metadata filter into the clean dev list and scores
  0.975 against 0.924: now excluded, leaving 46 songs. Dev numbers above include it.
- docs/EVALUATION_PROTOCOL.md: the final run's rules, written before it. The final model is
  last.pt, not best.pt: best.pt is selected on real scans, Transcoda claims no such selection.
- The gap to Transcoda's published Polish numbers is a constant: greedy 67.42 against 65.76
  (+1.66), beam 65.50 against 63.97 (+1.53), and beam gains 1.92 here, 1.79 in the paper. So the
  method reproduces, with something fixed differing, probably the page set (102 in the paper,
  112 in the file) or later gold curation. Corrected gold: greedy 54.09, beam 51.30. Beam stays
  the baseline, its best. One greedy shard hung in music21's conversion of a malformed
  prediction, which OMR-NED does not need: --kern-only.

### homr and Audiveris on Polish; copisteria found

Polish is a held-out benchmark: numbers only. Corrected gold, the same scorer and pages as ours.
Paired against run C (37.24), resampling pages:

| System | OMR-NED | Failed | Ours minus it | Ours better on |
|---|---|---|---|---|
| homr 0.7.0, defaults | 43.86 | 0 | -6.62 [-11.34, -2.07] | 70/112 |
| Transcoda 59M, beam 3 | 51.30 | 0 | -14.06 | 78/112 |
| Audiveris 5.11.0 | 63.60 | 4 | -26.35 [-32.10, -20.77] | 94/112 |

- Audiveris rejected the 1050-wide pages as staffless, so it reads them enlarged 2x. It split
  22 pages into movements; keeping only the first gave 64.65, joining them 63.60.
- homr's benchmark passes --no-title, which 0.7.0 does not list.
- homr's run slowed training from 280 to 50 steps a minute, even at nice 19 on separate cores:
  CPU baselines wait for training to end.
- Copisteria (github.com/kobimusic/copisteria at c685f28, copista-28m, defaults), found through
  homr's Benchmark.md, scored the same way: **26.74**, 0 failed, median 20.35. Ours minus it:
  +10.51 [+6.48, +14.60], ours better on 32/112. It beats run C, and the published 26.4 and 31.5
  bracket it. A SOTA claim on Polish is off: the clean model would need to gain over 10 points.
  It trains on rendered PDMX pages, like us, with no real scans or Polish listed.

### Copisteria on dev: is the Polish gap the metric or the data?

Dev, so errors may be read; eval/copisteria_dev.py, copista-28m, defaults, scored by our scorers.
Change is ours minus copisteria, resampling pieces.

| Dev set | Metric | Clean model | Run C | Copisteria | Clean's change |
|---|---|---|---|---|---|
| OLiMPiC crops, 46 songs | note F1 | 0.916 | 0.924 | 0.845 | +0.071 [+0.039, +0.104] |
| OLiMPiC crops, 46 songs | OMR-NED | 9.89 | not run | 25.10 | -15.21 [-20.14, -9.98] |
| Mutopia, 10 pieces | note F1 | 0.857 | 0.860 | 0.783 | +0.074 [-0.005, +0.146] |

- Same metric, same crops, and we lead, so OMR-NED alone does not explain Polish. The gap comes
  with the data or the page pipeline, which Polish's errors would show and may not be read.
- This does not offset Polish. The comparison favours us: we train on OLiMPiC's train split
  and its genre, tuned H5, H7 and H9 on these sets, and our MusicXML shares its gold's
  normalization. Copisteria reads a crop as a one-system page, without the neighbours its
  evidence model uses. Whole OLiMPiC pages were not run: its MusicXML does not locate the piano
  staves. Its right pitch is 0.924 on Mutopia, 0.917 on crops: its losses are largely timing.

### Polish dev half: where the gap is

Dev half only (eval/polish_split.txt; docs/EVALUATION_PROTOCOL.md). 56 pages, run C 36.86,
copisteria 25.21, gap 11.65; note F1 0.747 against 0.866, with 3,214 notes missed against 921.

- Crops were misplaced. seq_pages.page_scales infers a page's scale from staff heights; the
  recognizer's real factor differs (page 012: 2.22, inferred 2.07), and the error grows down
  the page, so lower systems lost their bottom staff. Scratch rerun with the true factor
  (debug page width over input width), no repo change yet:

| Pages | Before | Run C, true scale | Clean, true scale | Copisteria |
|---|---|---|---|---|
| Two staves, 49 | 33.13 | 25.67 | 26.99 | 25.65 |
| Three staves, 7 | 62.99 | 90.76 | 91.07 | 22.20 |
| All 56 | 36.86 | 33.80 | 35.00 | 25.21 |

- Two-staff pages tie copisteria, run C better on 26 of 49. Medians over all 56: 22.98, 19.26.
- Three-staff pages get worse: the crops now find the last two staves, as designed, where the
  drift had landed nearer the top two; the gold's order then favoured that. We read no third
  staff, and these 7 are the rest of the gap. All 7 fell in dev, so the sealed half has none.
- Page 026: one 6-staff system where the page has three, so measures are lost whichever crop.
  Not yet counted across pages.
- The 37.24 on all 112 pages carries this bug, as does anything else read through this path.

### Crop scale fixed in the pipeline

The recognizer now writes each system's true scale to layout.json and seq_pages reads it. A
layout without it fails loudly. Full rerun, replacing the scratch numbers above (the two differ
by up to 0.15 on Polish):

| Dev set | Before | Run C | Clean | Copisteria |
|---|---|---|---|---|
| Polish, two staves, 49 pages, OMR-NED | 33.13 (C) | 25.82 | 27.09 | 25.65 |
| Polish, three staves, 7 pages | 62.99 (C) | 90.76 | 91.07 | 22.20 |
| Polish, all 56 | 36.86 (C) | 33.93 | 35.09 | 25.21 |
| Mutopia, note F1 | 0.860 / 0.857 (C / clean) | 0.867 | 0.861 | 0.783 |
| Rendered, note F1 | 0.964 (clean) | 0.968 | 0.965 | not run |

- Clean against its old self: Mutopia +0.004 [+0.000, +0.015], rendered +0.001 [-0.001, +0.004].
  OLiMPiC pages crop at scale 1 and do not use this path.
- Clean against copisteria on Mutopia: +0.078 [+0.003, +0.148].
- Merged systems are rare: one 6-staff system in 56 pages (026), one 1-staff system (077).
  No special case added. All 26 three-staff systems are on the 7 three-staff pages.
- Sealed half untouched: no reading of those pages since the first look.

### Polish dev, two-staff pages after the crop fix: what remains

Clean model against copisteria, 49 pages. Note F1 over all notes 0.837 against 0.864.

- Semitone errors are the clearest difference: 744 against 251. Missed notes (807, 880), timing
  (about 1,017, 1,032) are level; extra notes 419 against 247.
- Few come from the first key: wrong on 6 of 49 pages, 117 of the 744. On some pages our systems
  disagree on the key (-4 and -3 on 051 and 087, 0 and 5 on 108 and 076), as each is read alone;
  how many semitone errors that explains is not measured. Hypothesis, untested: H11, one key
  per page unless a double bar changes it.
- The tie hides large swings both ways: 304 points of page gaps against us, 233 for us. Pages
  051 and 093 lose over 40 each; on 088, 003 and 037 copisteria loses over 40.

- H11 refuted, scratch only, nothing adopted. Each system's first key set to the page's majority
  (24 of 265 systems changed), clean model, 49 two-staff pages: OMR-NED 27.09 to 27.03, change
  -0.06 [-0.16, +0.02], but semitone errors rose from 744 to 907 and note F1 fell from 0.837 to
  0.827. Why they rose is not known; real key changes between systems would do it.

- Semitone errors with matching onset and duration, clean against copisteria, 49 two-staff
  pages: 519 against 225. No single cause: invented accidental 194 against 116, missed accidental
  162 against 67, both notes in the key (an E/F or B/C neighbour, so the notehead's line) 163
  against 42. Carry within a bar is small: 56 against 34 where the same pitch was right earlier
  in the bar, tie carry 21 against 8.
- We read sharp more often than flat, 336 against 183, most on flat keys: per 1,000 notes, up
  25.5 and down 11.7 on 26 flat-key pages; copisteria 8.2 and 5.4. So we tend to lose flats.
- So the delinearizer's carry rules are not the problem; it is the model's reading of small
  marks and line positions. Polish pages have about 9 pixels between staff lines; whether that
  low resolution is why is untested.

- Resolution, first check: no support, but a weak test. Staff spacing on the 49 pages runs only
  8.6 to 12.0 pixels, median 9.4. Rank correlation with the clean model's semitone rate 0.10
  [-0.21, +0.38]. By thirds, per 1,000 notes: 55.4, 28.7, 53.9, not a trend; copisteria 13.6, 7.7,
  24.4. All these pages are low resolution, so this cannot show that low resolution does not
  matter. Direct test, not run: blur OLiMPiC dev crops down to 9 pixels and compare.
- Direct resolution test: refuted. OLiMPiC dev crops, clean model, shrunk to Polish's scale (median
  height 836 to 310) and enlarged back: note F1 0.916 to 0.925, semitone errors 734 to 710 on 34,544
  notes. Blur alone does not make our accidentals fail. Polish's rate is about twice OLiMPiC's
  (47 against 21 per 1,000), so the cause is something else about those scans or engravings,
  which we did not train on. That the blurred crops scored higher is unexplained and untested.
- The first versions of these Polish F1s counted only outright misses in the denominator; the
  numbers above are corrected. Error counts and OMR-NED were not affected.

### Root cause of the semitone errors: tests so far (Polish dev, two-staff pages, clean model)

No retraining; all scratch. Decided with the user: the claim is two-staff piano, no new training
data until the cause is shown, sealed half and the authors' email wait.

Ruled out:
- Crowding: per 1,000 notes, 31 to 35 across chord sizes, flat across bar density, 37 and 30 by staff.
- Contrast: Polish ink is pale (1st percentile 69 against 0). Stretching it: semitone 744 to 737,
  OMR-NED -0.05 [-1.52, +1.65].
- Spacing: systems are narrower at model input (776 against 964 pixels). Widening 1.24x: 758,
  -0.24 [-1.77, +1.05]. Both together: 721, -0.96 [-2.61, +0.48], note F1 0.837 to 0.827.
- Rare keys: 5 flats or sharps are 7.2% of the mix, and OLiMPiC dev reads every key at 100%,
  including 55 crops at 5 and 35 at 6.
- Crop size and context, for key reading: 23 pages with 3 or more accidentals, 112 systems,
  correct keys nearly unchanged from ratio 1.6 to 2.6, tight, or the first 1 to 2.5 crop-heights only.

Found:
- 46% of matched semitone errors (239 of 519) come in runs of one pitch and direction within four
  bars; only 58 are shared with copisteria. They point at system-wide causes.
- Our key is wrong in 168 of 1,001 measures, copisteria's in 70. Those measures hold 152 of 495
  errors: 66 per 1,000 notes against 28 where the key is right. Key accuracy by accidentals:
  1: 0.89, 2: 0.97, 3: 0.85, 4: 0.92, 5: 0.33; copisteria 0.94, 1.00, 1.00, 1.00, 0.61. Gold checked
  by eye on pages 108 and 110: right. The 5-accidental figures rest on 7 pages.
- With the key right we still have 27.8 per 1,000 against copisteria's 12.6. Unexplained.

By elimination, what is left for keys is how these engravings look. That is not shown directly.
Proposed direct test: render fixed music in each font and read the key, which needs no Polish.

### OMR-NED by edit type, and why key signatures fail (Polish dev, 49 two-staff pages, clean model)

Edit cost summed over pages, before the margin below. Total 33,312 against copisteria's 32,042.
Against us: whole bars unmatched (delbar + insbar) 9,536 against 7,649; printed accidentals to
remove 557 against 124; notes inserted and deleted 4,155 against 3,689; other added marks 1,145
against 770. For us: beams 1,236 against 2,610; extra part 0 against 1,101; articulations 725
against 1,229. Ten pages hold 6,747 of our bar cost; on 051, 065, 026 and 093 copisteria pays
almost none. Why those bars fail to match is not known.

A missed key costs more than one symbol on OMR-NED after all: the model keeps a note's pitch and
writes a printed accidental on it instead. Pages 108 and 076 hold 200 of the 557.

Key signatures:
- Width refuted: squeezing crops to 0.7 leaves 5-accidental reads unchanged; binarizing too.
- The model does read the signature: blanking it gives 0 on 99 of 112 systems, so "0" means it
  saw nothing it recognized. Blanking the first 0.35 crop-heights kills Polish keys and leaves
  OLiMPiC's (75 of 80), so Polish signatures sit nearer the crop's left edge.
- Cause for 1 to 4 accidentals: crops trimmed too tightly at the left. White margin of 0.1
  crop-heights: 149 to 163 of 169 correct, 3 and 4 accidentals 75 of 75. More margin adds nothing.
- 5 accidentals unchanged, 12 of 37: a separate cause, unknown. Read as 0 or 6, stable under
  every transform tried.

End to end, scratch, not in the pipeline: OMR-NED 27.09 to 24.80, change -2.29 [-4.77, +0.30],
better on 26 pages, worse on 23; extra notes 419 to 285, missed 807 to 733, semitone 744 to 758,
note F1 0.837 to 0.848. Against copisteria's 25.65: -0.84 [-5.96, +4.14], medians 21.68 and 20.15.
Not yet checked on Mutopia or rendered pages.

### OMR-NED's bar locking

Why bars go unmatched: on page 042 the scorer reports 2,234 while editing each bar in place costs
644, so it is not finding the cheapest edit. Its bar pairs are not the cause (bar 2: edit 17,
delete and insert 87), and tuplets are not (unmatched at 5.8% of tuplet bars, 6.3% of others).
musicdiff first locks identical bars together with a Myers longest common subsequence, then aligns
only what lies between. Identical bars can lock to the wrong repeat, which forces everything
between to be deleted and inserted, so reading more bars exactly can raise the score.

Rescored with that step off, everything else unchanged, 49 two-staff pages (Audiveris 47):

| System | OMR-NED as published | Without locking | Median |
|---|---|---|---|
| Clean + left margin | 24.80 | 23.26 | 21.53 |
| Clean | 27.09 | 24.64 | 21.55 |
| Copisteria | 25.65 | 24.91 | 20.15 |
| homr | 44.38 | 44.24 | 41.08 |
| Audiveris | 69.14 | 68.62 | 74.88 |

- Margin's effect without locking: -1.38 [-2.67, -0.31], better on 28 of 49. Resolved, where the
  published scorer's noise left -2.29 [-4.77, +0.30].
- Against copisteria without locking: -1.65 [-6.54, +3.07], better on 25 of 49. Still a tie.
- Transcoda not rescored: its predictions are **kern, not yet run through this path.
- Locking is musicdiff's design, not a Transcoda change. The headline stays as published.

### Left margin in the pipeline

seq_pages adds 0.1 crop-heights of white on the left. Clean model, dev:

| Set | Before | After | Change |
|---|---|---|---|
| Mutopia, note F1 | 0.861 | 0.883 | +0.022 [+0.002, +0.047] |
| Rendered, note F1 | 0.965 | 0.970 | +0.005 [+0.000, +0.011] |
| OLiMPiC pages, note F1 | 0.925 | 0.924 | -0.001 [-0.011, +0.007] |
| Polish, two staves, OMR-NED | 27.09 | 24.99 | |

The first OLiMPiC run ended in a traceback that was not kept; a rerun passed. Cause unknown.
Scratch's 24.80 and the pipeline's 24.99 differ by the rounding of margins and a fresh layout pass.

Full edit cost, locking off, clean + margin against copisteria, 49 two-staff pages. Rows sum to
the scorer's totals, 29,002 and 31,178. Earlier note costs in this log are void: a note edit keeps
its cost in the fourth field, then its index in the chord, and I read the index.

| Category | Clean | Copisteria | Difference |
|---|---|---|---|
| Notes and rests missing or extra | 17,227 (59%) | 15,676 (50%) | +1,551 |
| Whole bars | 5,096 (18%) | 6,384 (20%) | -1,288 |
| Other marks | 1,796 | 2,138 | -342 |
| Beams | 1,647 | 2,793 | -1,146 |
| Articulations | 1,204 | 1,237 | -33 |
| Tuplets | 540 | 546 | -6 |
| Accidentals | 454 | 237 | +217 |
| Grace notes | 408 | 492 | -84 |
| Part names | 245 | 258 | -13 |
| Ties | 213 | 125 | +88 |
| Pitch, heads, dots | 111 | 133 | -22 |
| Whole parts | 0 | 1,107 | -1,107 |
| Expressions | 61 | 52 | +9 |

Copisteria collapses on 037, 003, 071, 088: 5,697 against our 1,177. On the other 45 pages it leads,
25,481 against 27,825, and on notes 11,273 against 16,594. Those note edits, grace notes aside:

| Kind, 45 pages | Clean | Copisteria |
|---|---|---|
| Gold note with no counterpart | 5,385 | 5,153 |
| Our note with no counterpart | 3,138 | 2,520 |
| Right pitch, moved onset | 3,676 | 1,689 |
| Wrong pitch, right onset | 3,496 | 1,368 |

- Wrong pitch: 369 of 533 of our cases are one staff step off, against 53 of 228: noteheads a line
  or space away, roughly 2,400 of cost. The largest single difference found.
- Moved onset: mostly a quarter to three-quarters of a beat, 287 shifts against 90. In bars with an
  onset error, later onsets are wrong 44% of the time against 32%: a modest cascade.
- Tilt: median 0.4 degrees against OLiMPiC's 0.2; 64% of crops at 0.3 or more against 34%.
- Why noteheads are a step off is unknown. Encoder grid: 8 rows, 32 pixels each, about 8 pixels
  between staff lines, so a staff fits in about one row.

### Notehead errors: what causes them (Polish dev, clean + margin)

Examples first: left-hand chords on ledger lines far above the bass staff; an identical chord in the
same bar reads right, then a step off. 57 of 68 multi-note events shift together.
- Tilt: deskewing crops, one-step errors 369 to 349, OMR-NED 24.99 to 24.28. Not the main cause.
  Training already rotates by up to 1 degree.
- Ledger lines: step errors per 1,000 notes rise from 14 to 20 in the staff to 40 to 150 outside it;
  copisteria stays near 4 to 27. On OLiMPiC dev too: bass, three ledger lines down, 107 against 12.
  So it is general, not Polish.
- Rarity: refuted, ledger notes are common in every training set.
- Longer training: 12,000 more steps at the same settings, OMR-NED 25.01, step errors 382. No change.
- Tight crops: 391. Weak test, since the model never trained on them and the staves hardly grow.
- Pure rhythm errors are rare: only 51 bar-staves have right pitches at wrong onsets.

### Notehead errors: input pixels are the cause (Polish dev, 49 two-staff pages)

All fine-tuned from clean, 288,000 samples each, except that h640 follows h512.

| Run | Input height, rows | OLiMPiC crops | Mutopia | OMR-NED | One-step | Wrong pitch | Moved |
|---|---|---|---|---|---|---|---|
| ctrl | 256, 8 | 0.923 | 0.886 | 25.01 | 382 | 3,871 | 4,314 |
| r16 | 256, 16 | 0.920 | 0.875 | 25.35 | 352 | 3,668 | 4,048 |
| h384 | 384, 12 | 0.922 | 0.885 | 23.21 | 285 | 3,260 | 3,458 |
| h512 | 512, 16 | 0.932 | 0.882 | 22.87 | 201 | 2,799 | 3,618 |

- More rows on the same pixels do little; more pixels do. Dose response on one-step errors.
- Clean renders read at 256 show no ledger effect: 1.8 per 1,000 outside the staff, 2.2 inside, on
  224 OLiMPiC dev systems; scans of those pieces 32 and 9.8, on all dev systems. The limit is
  resolution on old engravings, not resolution alone.
- Refuted: short ledger lines. Verovio's shortest and thinnest, 4.8 against 2.6 per 1,000 (120 systems).
- Errors concentrate in pieces: 6011095 alone has 160 of them, dense chord stacks on old plates.
- h512 against ctrl: -2.15 [-4.76, -0.07]; against copisteria -2.78 [-7.99, +2.12]; against h384
  -0.35 [-1.96, +0.99]. Without bar locking: 21.93, 21.62 (h384), 24.39 (ctrl), 24.91 (copisteria).
- Note edits, 45 pages, h512 (clean, copisteria): gold unmatched 5,267 (5,385, 5,153); ours unmatched
  2,795 (3,138, 2,520); moved 3,618 (3,676, 1,689); wrong pitch 2,799 (3,496, 1,368).

### Right pitch, wrong time: beam thickness (Polish dev, h512)

93% of this cost sits in bars that do not add up to the page's usual bar, and five pages hold over
half. Page 051 is hand-copied: single beams as thick as a printed double, read as 16ths. Causal test,
60 beam-heavy OLiMPiC dev systems drawn by Verovio with beam polygons 2.2 times thicker: eighths read
as 16ths 0 to 1,317 of 2,016; SER 2.7% to 25.6%. The second-choice duration is no rescue: label
smoothing saturates probabilities, and only about 10% of durations have a contender.

### Tuplet marks that run on, and gold the parser damages (Polish dev, b512)

- `imply_tuplets` is not the cause of the triplet errors: without it, OMR-NED 23.21 against 22.96.
- The model marks plain notes before a triplet passage as triplets, so the voice falls short. Clearing
  marks from a voice's first or last notes where that fills the bar exactly and leaves whole triplets:
  33 voices, 9 pages all better, 22.96 to 22.44 on saved reads.
- 7% of voices still miss the bar afterwards.
- Published gold: 8 merges on 6 open pages join voices of different staves. music21 then loses notes:
  15,534 of 16,149 parsed; 16,045 after moving the merges. Against repaired gold: ours 20.22,
  copisteria 23.49, -3.26 [-8.60, +1.68].

### Thick beams and round heads in training (Polish dev, from h512)

| Run | SER | Crops | Mutopia | OMR-NED | One-step | Wrong pitch | Moved |
|---|---|---|---|---|---|---|---|
| h512 | 0.0778 | 0.932 | 0.882 | 22.87 | 201 | 2,799 | 3,618 |
| b512, thick beams | 0.0744 | 0.932 | 0.887 | 22.96 | 200 | 2,928 | 3,456 |
| b512 with tuplet repair | same | 0.932 | 0.887 | 22.40 | 210 | 2,946 | 2,485 |
| c512, thick beams and round heads, repair | 0.0741 | 0.937 | 0.885 | 22.49 | 222 | 2,836 | 2,950 |

- b512 reads the 2.2-times beams at 2.6% SER, and normal beams at 2.82% against 2.71%.
- Round heads: SER on round-head renders about 6.0% to 2.95% (40 and 60 systems, so a rough figure).
- c512 against b512t: +0.09 [-1.23, +1.30], no overall change. It gains on 093, 051 and 091, and loses
  on 010, 096, 083 and 047. Bar locking on those pages is unchecked, and the rise in moved cost is
  unexplained.
- 32nds rose from 360 to 382 against gold 320, on pages 091, 055 and 057: thick double beams nearly
  touch. A cap on stacked beams is untested.

### What wrong pitch is made of (c512, 29 bar-aligned pages, 9,218 heads; copisteria in brackets)

Exact 7,166 (7,386); onset absent 711 (730); accidental 250 (200); one step 194 extra or lost and 86
moved (199, 89); two steps 95 and 67 (87, 63); octave 152 and 100 (142, 80); beyond an octave 69 and
166 (52, 141). Earlier counts that said most of it was not one-step paired the wrong heads in chords.

- Refuted, musical prior: in chords, ours is more consonant 36 times, gold 37, equal 83.
- Octaves come in runs: 123 of 138 heads sit in 27 bars. On page 023 `emptiest_row` cuts off the `8-----`
  line, and when a crop keeps it the model still ignores it: the dotted style is unlike Verovio's, and
  only 168 of 4,000 OLiMPiC crops have any ottava.

### Open: key signatures are not one symbol

The model writes one `key:fifths` token per key signature, for both staves, with no sense of the single
accidentals. Evidence that it matches a remembered shape: moving one sharp of five (A sharp to the top
line of the bass staff, pages 076 and 108) makes it read three at 0.87, with all five still there.

Cancelling naturals break the idea of one shape per key. What is printed at a key change depends on the
old key as well as the new one, so there are about 15 x 15 blocks, not 15, and styles differ:
- Naturals, then the new key in full: the new key can be read off and the naturals ignored.
- All naturals: the new key is C.
- Going down within sharps or flats (4 sharps to 2), some editions print only the naturals for what is
  dropped. The new key is then the old key minus the naturals, and a crop of that system alone cannot tell
  what it is. It needs the key in force, so it needs state across systems, which a system-by-system
  reader does not have.
- The block can sit at the end of a system, as a courtesy, with the next system printing only the new key.

Unknown: how often each occurs in our pages, whether the real changes we read correctly (012, 017, 022, 041,
065) printed naturals, and whether Verovio draws naturals at key changes in our training crops.

Other variants to check: doubled accidentals (a plain count fails), other octave placements, signature only
on the first line, C clefs. Not representable in LMX: a different key per staff, non-standard signatures.
Out of scope for now, and a limit on any claim of general piano coverage.

Possible directions: read accidentals one by one with the key decoded from them and the key in force, or
pass the key in force from system to system. A per-staff check from the detector could flag disagreement.

### Ottavas, segnos and old key layouts (Polish dev, from k512)

Whole bars an octave off had three causes. A segno read as an `8`: erasing it alone returns pages 049 and 091 to gold. A short `8...` line held to the end of the system (087). A printed sign ignored (023).
OLiMPiC's LMX drops the sign and keeps the sounding pitch, so the model had to add an octave silently and hold it. Labels now give the printed pitch, with a token on each shifted note (seq.ottava).

Bass-clef 5-sharp signatures on 076 and 108 put A sharp on the top line. Each staff alone: treble reads 5 at about 0.9, bass reads 3 at about 0.87, and together a low-confidence 6 or 3. Tight spacing was refuted (041).

o512, 8,000 steps from k512 with the token, marks and old-style sharps: those failures are gone, wrong pitch 2,551 to 2,125. Overall not distinguishable from k512 without page 026: locked +0.74 [-0.84, +2.61], unlocked +97 [-1,520, +1,679].
New failure: token recall is poor after a short fine-tune (291 tokens for 510 shifted notes on OLiMPiC dev crops with signs; none on 023 or 110), and a missed token now means an octave low.

### Vertical positioning, tested without training (k512)

- 16-pixel rows refuted: r512 against k512, one-step 199 / 194, OMR-NED 23.07 / 22.09, and 2.5 times slower.
- Staff geometry as a correction: refuted with today's detector. In 114 one-step bars it is nearer gold 33 times, nearer our wrong reading 59; it misplaces 489 heads in bars we read correctly.
- Straightening staves: 23.28, one-step 193. Warp is not the cause. Five-way voting: 22.06, one-step 190, at 5 times the cost.

### Where OMR-NED's cost comes from (k512, repaired gold, unlocked, 21,373)

OMR-NED compares step, octave and printed accidental: a wrong key is one small edit, and the notes it makes sound wrong are free.
- Notes on the other staff: 1,974 of 6,090 unmatched. A detector check of those heads sides with us 131 to 8. Gold's spines often record hands; published kern and MusicXML agree with each other, not with the print.
- Bars, 2,449: merged systems 377 (fixed, page 026 45.99 to 7.81, piano_staff_pairs); staff blank in print about 620; gold bars not on the page about 230 (038); gold parse damage 260 (108); ottava 182; about 780 unexplained.
- Beams: gold gives 16ths one beam level where print has two. Whether this explains all 1,684 is unproven.

### Polish gold: what the review found (dev, k512 unless stated)

- The scorer reads kern with converter21, registered by the authors' metric, not music21's own
  reader. The two give different notes, beams and ottavas for one file. Checks of gold must load
  the metric first (eval/check_gold.py). Published gold keeps 40,084 of 40,202 notes under it.
- Silent staff swaps: the staff holding two voices changes at a barline with no spine change.
  Six pages: Transcoda 59.02 to 47.47, Copisteria 38.44 to 16.94, k512 37.69 to 16.16.
- Beam groups left open: L with no J, 145 on dev, and the parser drops the group's beams. Closed
  mechanically, 49 pages: k512 19.45 to 18.05, Copisteria 23.01 to 21.92, Transcoda 49.99 to
  49.43. Beam edits 1,577 to 402; note and bar cost fell too, 108 by 21 points, not yet explained.
- Notes filed by hand, drawn on the other staff: Print standard only. Four pages: k512 33.98 to
  24.37, Copisteria 32.38 to 22.70, Transcoda 59.95 to 53.41.
- Ottavas: gold has none. converter21 reads *8va, so inserted lines are enough. On a bar of 076:
  right sign 0, missed sign 2, sounding pitch with no sign 62.

### Retrain seq512 (192,000 steps, ottava tokens, marks, P4-P7 output), 2026-11-07

Dev SER 0.0749 at the last step (best). Polish dev, 49 two-staff pages, all systems on the same gold;
k512 and o512 with P4-P7, Copisteria with P5/P6; paired ranges are seq512 minus the other.

| System | Crops F1 | Mutopia F1 | Corrected | Print standard |
|---|---|---|---|---|
| seq512 | 0.936 | 0.878 | 15.32 | 12.04 |
| o512 | | | 16.17 (-0.85 [-3.00, +1.03]) | 12.26 (-0.21 [-1.38, +1.03]) |
| k512 | 0.933 (+0.003 [-0.010, +0.014]) | 0.887 (-0.009 [-0.048, +0.016]) | 16.35 (-1.03 [-3.06, +0.68]) | 13.71 (-1.67 [-3.71, +0.05]) |
| Copisteria | | | 21.25 (-5.93 [-10.62, -1.83]) | 18.62 (-6.58 [-11.20, -2.39]) |
| Transcoda | | | 49.39 (-34.07) | 49.34 (-37.30) |

- No measurable change against k512 or o512 on any row; most of the gain on k512 is page 026 (-35),
  which o512 already fixed. Locked scores; unlocked not yet run.
- As published, fixed: the cross-format path parsed **kern without the header line the authors add
  (`_ensure_kern_header`); published gold and Transcoda's output have none, and the parser returned an
  empty score that scored 100. Now parsed as theirs: seq512 43.52, o512 44.31, k512 44.45, Copisteria
  45.48, Transcoda 65.63 (kern to kern, unchanged); 14 pages score 100 for every system.
  The same bug made `--ambiguous-marks` score Transcoda 100 everywhere; with the fix, 49.34 with or
  without the flag. Corrected unchanged (15.32): its gold has headers.
- First real `<octave-shift>` output: shifts on 083, 076, 087 (gold has ottavas there) and 049
  (gold has none: to check against the image). None on 023.

### Bars on ottava pages, seq512 (unlocked, Corrected gold)

seq512 against k512 unlocked: 16,946 against 17,262. Bars cost rose (delbar 559 against 346, insbar
1,589 against 1,545), all on 110, 083, 076: the pages with ottavas. Three causes, each tested:
- Scoring convention: Corrected gold folds ottavas into pitch, and a MusicXML `<octave-shift>` is read
  by the scorer as moving its notes, so a correctly drawn sign costs the whole passage (083: 20.9 with
  the sign, 12.3 with the same output's signs removed). Removing all signs: Corrected 15.32 to 15.10,
  Print 12.04 to 12.34. Per page, the better of the two, as Print standard already allows: Corrected
  15.04, Print 12.01. 087 goes the other way under Corrected (22.0 signed, 25.0 folded): its gold
  keeps printed pitch at some signs.
- Missed signs: 110 has `8` with a dotted line, over the lower staff in treble clef as well as the
  upper. Training crops draw only Verovio's `8va` with dashed lines, 8va on the upper staff and 8vb
  under the lower. No tokens on 110 or 023; k512, trained on sounding labels, got 110 right.
- False sign: 049 bar 3 of the last system, from a long plain line between the staves (no sign in
  the print).
Tuplets on 096 (instuplet 34 against 4): the page has no time signature and no printed 3s; gold has
triplets throughout, seq512 writes fewer than k512 (53 groups against 61, gold about 86). Ambiguous
from the print; not pursued.

### Ottava sign styles in training crops (q512: 8,000 steps from seq512)

`build_seq_dataset.py --marks` now also draws `8` alone (Verovio's octaveAlternativeSymbols), dotted
lines (round caps), varied line thickness, and 8va above the lower staff. 29,069 crops in
data/seq3/marks replace data/seq2/marks in the mix; batch 6, lr 1e-4, as the earlier fine-tunes.
- 110: 0 tokens to 25, on the right bars of both staves. Folded and scored on Corrected, 110 is
  16.88 against seq512's 22.75 (k512 about 18). The style was the cause.
- 023: still no tokens. 049: the false sign stays (5 tokens); its score improves (10.2 to 6.9) for
  other reasons.
- Overall not better: dev SER 0.0776 against 0.0749; Polish Corrected 16.37, +1.05 [-0.41, +2.81];
  Print 12.43, +0.39 [-0.78, +1.67]; Mutopia 0.881 against 0.878. 065 (handwritten) +32.9.
  A short fine-tune at a small batch costs more than it buys; the styles belong in the next full run.
- 110's Print gold has no `*8va` yet, so a correct sign there is charged as on Corrected.

### Beams lost by Print standard's staff moves (gold bug)

Print standard ranks beams third (editbeam 690 against 431 on Corrected), all on the pages with
staff moves: 038 0 to 75, 062 13 to 80, 047 5 to 52, 067 6 to 49, 057 0 to 28. `divided` dropped
every moved note's beam marks, meant for beams the move splits across the staves; most moves take
whole beam groups. Now a group keeps its beams when all its fields go the same way, whole
(`beams_kept`). Print standard, 49 pages: seq512 12.04 to 11.81, k512 13.71 to 13.51, Copisteria
18.62 to 18.44, Transcoda 49.34 to 49.23; every system gains on every move page (038: seq512 16.5 to
11.7). check_gold: 0 failed.

### Printed rests that the print does not have (Print standard's staff moves)

After the beam fix, bar insertions led on the move pages (038 insbar 139). A move filled both gaps
with plain rests: the vacated staff and the new voice. The print draws neither (checked on 038, 062,
047, 067; 012 and 058 are recorded as "staff blank"), so the rests are now `ryy`, not drawn, which the
scorer ignores (`hidden_rest_like`). The seven move pages, seq512 12.92 to 7.43 (038 11.7 to 1.8), k512
17.49 to 12.49, Copisteria 16.13 to 11.00. Print standard, 49 pages, with both fixes:
seq512 11.03, o512 11.31 (-0.28 [-1.49, +0.98]), k512 12.80 (-1.77 [-3.80, +0.03]), Copisteria 17.71
(-6.68 [-11.49, -2.57]), Transcoda 48.95. No sealed page has a move. This is P8 for moves only; the
published gold's own padding rests (096) are still unreviewed.

### Published gold's padding rests (P8), first page: 096

New override `hide_rests {"last_line"}` (Print standard): rests in a range become `ryy`. On 096 the
print draws no rest in bars 6-13 (checked on the image); bars 1-5 and 14-16 have printed rests and
stay. 096: seq512 33.45 to 31.27, k512 19.10 to 16.48, Copisteria 16.61 to 13.90, Transcoda 78.57 to
77.78. Other pages not reviewed: each range needs the image. Separately, seq512 is 15 points behind
k512 on 096 (tuplets and ties, see the unlocked ranking); not investigated.

### seq512 against k512 on 096 (31 against 16)

The page continues a piece in 3/4 with triplet eighths throughout, and prints neither a time
signature nor any 3. Read as plain eighths, many bars fill 4/4 exactly (bar 1: rest, three eighths and
a half; quarter and six eighths), so `likely_bar` votes 4 and no tuplet can be implied. Tested: with
the bar forced to 3, no measure of either model's reading fills in every voice, because gold's
triplet groups often start with a rest or are beamed in twos, which `tuplet_candidates` (beam groups
of 3 or 6) does not see. k512 is ahead because it writes 3in2 itself more often (61 groups against
53, gold about 86). Not fixable at the page: the meter is on the previous page. Ideas, untested:
unmarked-tuplet training crops with rests in the groups (seq512's crops have no --unmarked-tuplets
set); a page-level meter carried from the previous page, which the benchmark does not provide.
- 030 bar 8 (print: treble staff blank, all on the bass staff): move_down added. seq512 10.68 to 9.95,
  k512 13.61 to 12.83, Copisteria 17.52 to 17.50. Bar 7 is the same but its upper staff is two
  columns, which move_down cannot take yet.
- 030 bar 7: new override `staff_down {"last_line", "lower_columns"}` moves every upper column to
  new lower voices (the lower staff split once per column, merged after) and leaves hidden rests
  above. 030: seq512 9.95 to 8.25, k512 12.83 to 11.13, Copisteria 17.50 to 17.32. check_gold: 0
  failed. A mid-spine `*staff2` does not do this: converter21 keeps the notes on their spine's staff.

### Crop boundaries between systems (seq_pages.py), from the 023 missed ottava

023's `8.....` line sits about 7 interlines above its staff. `crop_system` cut each system at the
emptiest row nearest the middle of its 7-interline reach, so the sign was cut off from every crop at
both heights: the model never saw it. Now, between two systems, the cut aims at the middle of the
whole gap (falls back to the reach's middle at a page edge). The 023 crop now holds the sign.
seq512, 49 dev pages: Corrected 15.32 to 14.62 (-0.70 [-2.27, +1.00]; 042 +27.8, the page whose
locked score swings), Print 10.98 to 9.33 (-1.65 [-2.87, -0.58]; 32 pages better, 13 worse). Gains:
093 -18, 037 -12, 081 -11. Losses: 017 +7.6 (34 ties written against gold's 3), 110 +4.3 (its sign
now seen and drawn: gold Print has no `*8va` there yet). 023 still gets no ottava token with the sign
in view; it scores 3.11 on Print, so its pitches already sound. q512 with the new crops: 15.05 and
10.49, still behind seq512.

### Ties that join no same pitch (page_musicxml.mend_ties, P9)

017's new false ties were the model tying a grace note to the different main note its slur leads
to (D5 grace to C5, with the slur written as well). Elsewhere it leaves out a tie's stop with the
same pitch in the next chord (093). Rule: each tie joins the same pitch in the voice's next event;
a half-written tie whose partner pitch is there is completed, one with no such note is removed.
seq512 with the new crops, saved output re-finished: 789 tie elements to 584; Corrected 14.62 to
14.57 (-0.05 [-0.08, -0.03]), Print 9.33 to 9.27 (-0.06 [-0.09, -0.03]); 25 pages better, 3 worse
by under 0.09. Small: the scorer charges a tie as one edit.

### One slur per chord (Print standard)

Unlocked, slurs are the largest cost after notes (extra ins 410, del 398, duration 114). One part
is gold's encoding: a slur on every note of a chord (`4E( 4G( 4e(`), which converter21 reads as
three slurs for one printed arc; 931 such extra marks on the 49 pages. Print standard now keeps each
chord's slur marks on its first note (`one_slur_per_chord`). Print, 49 pages: seq512 with new crops
and P9 9.27 to 9.06; seq512 10.98 to 10.80; o512 11.05; k512 12.49; Copisteria 17.71 to 17.53;
Transcoda 48.95 to 48.74. check_gold: 0 failed. The rest of the slur cost is two-sided: gold leaves out
printed slurs (017: 15 in gold, the grace slurs and long arcs printed), and the model misses some
(030, 069).

### Beam groups left open by a short J (beams_closed, both standards)

023's Trio: the print groups (8th, 16th, 16th) (8th, 8th) (8th, 8th); gold ends the 16ths with
`16J` for `16JJ`, so the primary beam stays open and every later group in the bar nests in it
(the model was right, 55 beam edits). `beams_closed` now also closes the open beams when a note
would carry more than its length allows (`beam_count`: 1 for an eighth or triplet eighth, 2 for a
16th). 24 pages change, sealed included; check_gold 0 failed. 49 pages, Corrected / Print:
seq512 (new crops, P9) 14.57 / 9.06 to 14.46 / 8.95; k512 16.35 / 12.49 to 16.25 / 12.39; Copisteria
21.25 / 17.53 to 21.15 / 17.44; Transcoda 49.39 / 48.74 to 49.32 / 48.67.

### Accidentals the print shows and gold does not (looked at, not fixed)

accidentdel 144 on Print: our output shows an accidental where gold shows none. On 060 (key of six
flats) bars 25-26 come straight after a passage in sharps, and the print restates the flats
(checked on the image: gg-, ee-, dd- and the chords). Kern needs `X` to show an accidental the key
already gives (`ee-X`); gold has none, so converter21 hides them and our correct reading is charged.
No rule can tell a printed courtesy accidental from an unprinted one, so it is an override per
note: `show_accidental {"column", "note"}` adds the `X`. 060 bars 25-26, four notes whose flat is
clear on the image (the chords' flats left out: which notes carry them is not clear at this scan):
seq512 7.30 to 7.21, k512 10.64 to 10.54, Copisteria 12.62 to 12.52. Other pages unreviewed.
- Image review, restated accidentals (`show_accidental`), each checked on the image:
  110 (8 notes: bars 6, 8, 18, 20, 21): seq512 25.42 to 25.08, k512 17.19 to 16.87, Copisteria 40.71
  to 40.61. 058 (bars 4, 6, 7: the arpeggiated left-hand chords print an accidental on every
  note): seq512 4.54 to 2.16, k512 8.83 to 6.43, Copisteria 8.87 to 6.49. 066 (6 notes, bars 6, 7,
  14; bar 11 left out, not clear): seq512 8.63 to 8.51, k512 8.29 to 8.17, Copisteria 7.41 to 7.28.
  Tested and refuted: that the parser hides accidentals on arpeggiated chords (`:`); alone it shows
  them, so 058's hidden ones come from the bars around. 017 (5 notes: the flat restated on b in each fourth
  bar after naturals): seq512 14.97 to 14.79, k512 9.29 to 9.11, Copisteria 14.59 to 14.47. 021 bars
  21-23 (the 058 chord again, all three accidentals printed): seq512 0.61 to 0.48, k512 0.93 to
  0.81, Copisteria 4.47 to 6.03 (it does not print them). A first zoom had looked at bars 17-19,
  which print naturals, and suggested a gold pitch error; reading the staff against the clef on
  058 bar 1 (gold F, C natural, A) and finding the bars by system settled it: gold is right.

### Plain-line negatives and system-wide ottava lines (r4: 8,000 steps from seq512)

`--marks` now also draws a solid line with no sign between the staves (a quarter of scores,
`add_plain_line`), and some ottava passages run from the system's first note or to its last with
the end hook taken off, as a sign carried over a system break (023). data/seq4/marks, 29,081 crops.
- 049's false sign: 5 tokens to 0. 110 keeps its 25. 023: still 0, so the system-wide shape was
  not what kept it unread. Tested on 023's own crops, without training, with seq512 and r4: the
  sign's gap to the notes closed (removed 26 rows), and its wavy line redrawn as dots, as dashes,
  and with `va` added. 0 tokens in every case. Not crop, style, distance or line: still open.
- Overall worse, as q512 was: dev SER 0.0781; Polish (newest gold, --either-ottava) Corrected 14.90
  against seq512's 14.22, Print 9.96 against 8.89. Short small-batch fine-tunes keep costing more
  than they buy; the styles are for the next full run.

### Note cost split by kind (seq512 with new crops and P9, newest Print gold, 49 pages, unlocked)

7,549 of note edits: same pitch at a moved onset 2,860 (shifts mostly a 16th or an eighth); gold
note unmatched 1,727; wrong pitch at the same onset 1,174 (most a step of 1); our note unmatched
1,049; only 12 differ in duration, beam or tuplet alone. Moved onsets lead on 096 (506, unprinted
triplets), 087 (272), 042 (264), 056, 093. On 087, bar by bar: a quintuplet read as 16ths (bar
4), 16ths read as 32nds (12-13, one beam too many), a triplet group misplaced (5). Reading errors of
several kinds, each moving the rest of its voice; no common gold or output cause found. Model-side.

### One beam too many (lmx_tuplets.extra_beams_dropped)

From the note-cost split: on 087 and 057 the model writes 16ths as 32nds, and the voice falls
short of its bar. Where exactly one plain beam group (no dots, no tuplets) fills the bar once its
notes have one beam fewer, the duration and one beam token change; applied after imply_tuplets in
seq_predict.delinearize. Polish dev, seq512 with new crops and P9, newest gold, --either-ottava:
4 groups on 2 pages; Corrected 14.22 to 14.08, Print 8.89 to 8.75 (051 26.14 to 22.86, 057 9.97 to
6.48). 057 checked on the image: the group has two beams. OLiMPiC crops: 0.936, +0.000 [+0.000,
+0.001].

### Quintuplets restored from the bar (lmx_tuplets.TUPLET_OF_GROUP)

087 bar 4: an unmarked group of five 16ths read as plain 16ths overfills the voice. Groups of 5
now become `5in4` when that fills the bar exactly, with the same checks as 3 and 6; the
cleared-triplet rule keeps to the 2/3 ratios. Polish dev: only 087 changes (Corrected 20.29 to 18.59,
Print 17.52 to 15.00); all 49 pages Corrected 14.08 to 14.05, Print 8.75 to 8.70. OLiMPiC crops:
0.936, +0.000 [+0.000, +0.000].
- Tried and dropped: the beam rule on runs of like notes within a beam group, and on all such
  runs together (for 042, whose triplet 32nds share a beam with an eighth). Print 8.70 to 8.67, but
  057 6.48 to 7.88 and 069 worse, 042 unchanged. Reverted to the committed one-group rule.

### Ties read as slurs (page_musicxml.ties_from_slurs, P10)

Started from 069 (gold 21 same-pitch ties, ours 5). Over the 46 dev pages whose bar counts agree,
gold has 396 ties that join the same pitch in the voice's next event; seq512 writes 215 of them.
Misses are spread evenly: inside a bar, across a barline, across a system break, single notes and
chords all about half. Of the 181 missed, 142 sit on a note where the output starts a slur instead,
23 on a note with neither, 16 on no such note. Control: in gold 14% of tied notes also start a slur,
in our output 36%, and away from gold's ties we write as many slurs as gold (1,618 vs 1,594). So the
model reads about two in five ties as slurs: in print both are one arc (088 bars 16-18: long flat
ties between half notes). The delinearizer and finish_score keep chord ties (checked on a made-up
chord); P9 is not the cause (counts as low before it).
- Fix: a slur from one event to the voice's next becomes a tie when both events have exactly the
  same pitches, neither has a grace note (017) or articulation (portato). Run after pair_slurs.
- Shared-pitch version (tie whatever the chords share) first: 138 slurs, but 049 false tie (a
  chord moving under a slur, F4 common). Exact-pitch version: 100 slurs; Corrected 14.02 to 13.89,
  Print 8.67 to 8.54, 20-21 pages better, 2 worse (at most +0.12). Kept that one. Ties found
  after: 310 of 396. Gold freshly built from the current overrides; --either-ottava.
- Lesson: the scratch gold copies (table/g_print3, 15:50) predate the later overrides and lack
  one_slur_per_chord; they put both systems about 0.1 to 0.4 worse. Rebuild gold before scoring.
- Left: 23 missed ties with no slur (model saw no arc), and gold's 55 tie starts with no same-pitch
  partner in the same voice. Listed: most are ties into another voice (096 bars 5-10, the bass's
  triplet Ab3 held into the other voice's chord) or onto a rest where voices renumber; not gold
  errors. A few (069 bar 10, E5 to G5) may be; the scan is too coarse there to tell.

### Whole bars inserted or deleted (P10 output, Print, unlocked)

811 of 10,894: 30 gold bars unmatched (521) and 17 of ours (290). 110 holds 389: its last system's
8va passage, which gold has at sounding pitch in triplets and we read at written pitch with no
sign and no triplet mark. That is the missed-ottava cause the seq513 crops address (q512 and r4
fixed 110). 065 (handwriting) holds 234. The other ten pages share about 190. Nothing new to fix
before seq513 is scored.

### Slurs (P10 output, fresh Print gold, unlocked)

Slur edits: 325 of ours with no gold slur, 165 gold slurs we miss, 109 spans differing; repeats
128 (seq513 adds repeat tokens). 017 has 43 of our extras: from system 2 on, every bar opens with
a grace pair slurred to the main C and has a long arc over the bar (image checked, all five
systems). Gold opens neither: no "(" on the grace pair, and the bar's closing ")" has no opening.
New override op `slur` (gold_overrides.py) adds "(" / ")" / ")(" to a note; converter21 pairs them
as drawn whatever their order in the token. 36 entries on 017 (system 1's grace pair at line 14
left alone: too coarse to see). check_gold passes.
- Result: 017 slur edits 45 to 11, but its score 14.2 to 33.0. The authors' bar alignment now
  matches some repeated bars exactly and pairs the copies wrongly: bars 12-17 go from substituted
  to deleted and inserted (whole-bar edits 82 to 673). Both have 29 bars and the same notes. A
  metric artifact on a page of near-identical bars; the gold change is right. Print mean about
  8.54 to 8.92. Left in place (uncommitted); flagged for the user.

### Clef changes before the barline (page_musicxml.clefs_before_barlines, P11)

Clef edits (Print): 55 gold clefs unmatched, 42 of ours, mostly in pairs per page. 108 and 095:
gold puts a clef change at the end of the bar (beat 3 of bar 6), as the print draws it before the
barline; our output opens the next bar with it. Rule: a clef that opens a bar (not the first) moves
to the end of the bar before. 37 clefs moved; 108, 095 and 066 then match gold's positions exactly.
Fresh gold, P10 output, 017 left out (its gold changed meanwhile): Corrected 13.75 to 13.69 (13
better, 0 worse), Print 8.43 to 8.41 (12 better, 066 +1.95). 066's clef now matches gold exactly;
its cost moved to whole bars and notes: the authors' bar alignment again (as 017).
- Other extras: double and final barlines (69) need tokens, next run (TODO.md).

### Whole-page bar alignment (omr_ned.py --whole-page-alignment)

Why 017 scored worse with correct gold: musicdiff (5.0.2, under the authors' metric) first pins
the longest run of identical bars (Myers' diff) and aligns by edit cost only between the pins.
That maximises identical bars, not the lowest cost; on repeated bars it pins a bar to the wrong
copy and the bars between become whole-bar deletes and inserts. The new flag replaces the pinning
with one block, so the authors' own dynamic programme aligns all bars by cost. It is a minimum
under the same costs, so never above the authors' number (checked: no page higher, on any system).
017 (P10+P11 output): authors' 452 to 1,063 with the corrected slurs; whole-page 376 to 352.

49 dev pages, fresh gold, --ambiguous-marks --either-ottava; authors' / whole-page:

| System | Corrected | Print |
|---|---|---|
| seq512 + P10, P11 | 13.84 / 12.44 | 8.91 / 7.85 |
| k512 | 16.25 / 15.09 | 12.31 / 11.26 |
| Copisteria | 21.07 / 20.39 | 17.39 / 16.24 |
| Transcoda | 49.32 / 49.21 | 48.68 / 48.64 |

No page failed or timed out. Biggest drops for us: 042 (-30.3 Corrected, the anomaly noted
earlier), 017 (-22.1 Print), 030, 093, 097, 066. Ranking and gaps unchanged.
- 096's slur extras (28): mostly the same slurs at shifted onsets (gold 1.33-1.67, ours 1.5-2.0)
  because its unprinted triplets are read as plain eighths. Downstream of the 096 triplet cause,
  not a slur problem.

### Tuplet edits (page_musicxml.state_tuplet_brackets, P5 fixed)

139 tuplet edits (Print, P10+P11 output). By kind: 46 ours bracketed ("3B") where gold has "3";
53 ours mid-group where gold starts or stops one (mostly 096, notes shifted by its misread
triplets: downstream); 22 ours with no number (restored unprinted triplets) where gold shows 3.
- Brackets: P5 bracketed a tuplet whose first note has no beam, and a triplet that opens with a
  rest never does (003: rest + two beamed eighths, printed with neither bracket nor number). Now
  any beam in the group counts. Brackets 62 to 18. Corrected 13.84 to 13.81, Print 8.91 to 8.89;
  4 pages better (003 -0.61, 037, 088), none worse.
- Hidden numbers: converter21 reads Humdrum's `*Xtuplet`, so Print gold could hide 3s the print
  does not show. At most 22 edits; needs each page's image. Not done.

### Beam edits (Print, current output)

334 first-level beam edits; 240 are ours beamed where gold has a flag (musicdiff writes an unbeamed
eighth as "partial"). 062 (64) and 067 (37): our move_up overrides split cross-staff beam groups,
and `beams_kept` then drops the group's beams on both staves, since no kern spine holds a beam
that crosses. The print beams them. 065 (63) is handwriting; 076, 026, 069 the rest.
- Possible: beam each staff's part of a split group on its own (L..J per spine, beam levels
  recounted, hidden padding rests skipped). Truer to the print than flags, but many special cases
  in gold_overrides for about 100 edits on two pages. Not built; asked the user.
- Tried after: also tie chords sharing two or more pitches under a slur (069 bar 10). 5 more
  slurs; Corrected 13.81 unchanged, Print 8.89 to 8.88, 3 pages better, 062 worse. Noise; kept
  the exact-pitch rule. Remaining missed ties (no arc read at all, 23) are the model's.

### Articulation edits (Print, current output)

112 marks gold has that we miss (76 staccato, 29 accent, 4 tenuto), 61 ours gold lacks (44
staccato, 13 accent), only 8 edits of one mark to another. 065 (handwriting) holds 26 of the
misses; the rest spread a few per page. Reading misses and false dots, not placement: no output
rule. Model-side, for the next look at seq513's output.

### 076 hand-filed inner voice, Print standard (2026-11-08)

Bars 15-17: the inner voice under the trilled F# is drawn on the upper staff, gold filed it in the
left spine. Two `move_up` overrides, decided from the image. Polish dev Print, 49 pages, fresh gold,
`--ambiguous-marks --either-ottava`, authors' / whole-page (before → after):

| System | Authors' | Whole-page |
|---|---|---|
| Ours (P10, P11, P5 fix) | 8.89 → 8.76 | → 7.69 |
| k512 | 12.31 → 12.19 | 11.26 → 11.14 |
| Copisteria | 17.39 → 17.26 | 16.24 → 16.11 |
| Transcoda | 48.68 → 48.67 | 48.64 → 48.63 |

Every system gains about the same, as it should for a gold fix. Ours on 076: 602 to 432 edits (whole-page).

### Tuplets the bar rejects, and triplets under a second beam (lmx_tuplets, 2026-11-08)

Two causes from bars whose staff ends off the barline (52 staff-bars on dev, half on handwritten
pages):
- The model writes four plain 16ths as a 5in4 quintuplet (018 bar 6, 026 bar 10). The staff ends a
  fifth of a beat short and every later note of the other staff is offset: 49 and 40 edits for one
  mark. `runaway_tuplets` cleared only 2/3 marks; it now clears any XinY mark when that fills the
  bar exactly and leaves whole tuplets, and takes the mark's `tuplet:start/stop` with it (before,
  a cleared triplet left its bracket behind; 064's output had a bracket round plain 16ths).
- Unmarked triplets under a group's second beam: an eighth, then three 16ths (055 bars 17, 19).
  `tuplet_candidates` now also offers a beam group's inner groups when the group itself is not
  one.

Measured without the GPU: seq512's LMX for every crop height of the 49 pages, re-read on CPU
(fp32, so some readings differ from the GPU's bf16 run; rebuilding 003 from them reproduces the
scored output exactly). Old rules and new on the same readings, `--ambiguous-marks --either-ottava`,
paired over pages:

| Gold | Authors' | Whole-page |
|---|---|---|
| Corrected | 13.94 to 13.71, -0.23 [-0.43, -0.05] | 12.53 to 12.30, -0.23 [-0.43, -0.06] |
| Print | 8.87 to 8.65, -0.22 [-0.42, -0.05] | 7.80 to 7.57, -0.22 [-0.42, -0.06] |

Six pages better (026 -2.66, 055 -2.19, 095 -2.14, 018 -1.86, 081 -1.48, 108 -1.42; Print, authors'), two worse: 076 +0.61 and 091 +0.1 to
+0.2. Neither is a wrong tuplet: with correct triplets restored, another crop height's reading of
one system fills every bar and wins `best_readings`, and that reading is worse elsewhere (076 bar
29 loses a second voice). Still to check: OLiMPiC crops F1 (needs the GPU), as for earlier tuplet rules.
Cached readings for later pipeline tests: scratchpad `readings512/`, built by `dump_readings_cpu.py`.

### Stray beam ends in gold (gold_overrides.beams_closed, 2026-11-08)

042 bar 19: gold `24g(L 24a 24bJ 8cc)J` closes a beam nothing opened, and the scorer's parser
then drops every beam in the bar, so correct beams cost 15 edits. 15 such J on 7 dev pages. Where
the column's note just before ended a group (026 x2, 042 x4: two 16ths and an eighth under one
beam, gold closing after the 16ths too) that J goes, so the group runs on; otherwise the stray J
goes (065, 076, 096, and 067's, made by a staff move). Mechanical, both standards, all 112 pages; none left.
Dev, 49 pages, authors' alignment (whole-page moves the same way), ours on the CPU readings with the new tuplet rules:

| System | Corrected | Print |
|---|---|---|
| Ours | 13.71 to 13.66 | 8.65 to 8.58 |
| k512 | 16.25 to 16.17 | 12.19 to 12.01 |
| Copisteria | 21.07 to 21.01 | 17.26 to 17.19 |
| Transcoda | 49.32 to 49.28 | 48.67 to 48.64 |

Every changed page is better for every system; none worse. Protocol updated (Corrected standard).

### Triplets that open with a rest, and choosing among candidates (lmx_tuplets, 2026-11-08)

056 bars 15-16: gold has triplets that open with an eighth rest, read as a rest and two plain
eighths beside a plain group of three. `tuplet_candidates` now also offers a beamed pair after an
unbeamed rest of their length, and `insertions_for` uses all candidates if they fill the bar,
otherwise the one subset that does (none if several do; at most 10 candidates). CPU readings, gold
with stray beam ends mended and 091's slurs, paired over 49 pages:

| Gold | Authors' | Whole-page |
|---|---|---|
| Corrected | 13.66 to 13.57, -0.09 [-0.28, +0.02] | 12.26 to 12.16, -0.09 [-0.27, +0.02] |
| Print | 8.56 to 8.47, -0.10 [-0.28, +0.02] | 7.49 to 7.39, -0.09 [-0.27, +0.02] |

Three pages change: 056 -3.6, 096 -1.3 to -1.4 (rest-led triplets, the pattern noted for 096
earlier), 093 +0.4 (handwritten; a group the model invented is fitted into the bar as a
triplet). The range includes zero, so by the protocol this does not count on its own.

### One beam too many under a second beam (lmx_tuplets.overbeamed_group, 2026-11-08)

042 bars 3, 11, 19, 27: a triplet read as 32nds where the print has 16ths, beamed with an
eighth, leaves the voice a quarter beat short. `overbeamed_group` looked only at whole beam
groups of plain notes; it now also looks at the notes under a group's second beam, and allows a
tuplet when every note carries the same mark (no dots, as before). Still one fitting group or
none. Unlike the run-based rule tried and dropped earlier, 057 does not change. CPU readings,
gold as above, paired over 49 pages:

| Gold | Authors' | Whole-page |
|---|---|---|
| Corrected | 13.57 to 13.53, -0.04 [-0.10, -0.00] | 12.16 to 12.04, -0.12 [-0.32, -0.01] |
| Print | 8.47 to 8.30, -0.17 [-0.46, -0.01] | 7.39 to 7.27, -0.12 [-0.32, -0.01] |

Four pages change, all better: 042 -6.31 Print (-4.02 whole-page; Corrected authors' only -0.10, the bar pinning
again), 093 -0.98, 017 -0.56, 069 -0.43.

### Dots from the bar's arithmetic (lmx_tuplets.dots_mended, 2026-11-08)

030 bar 21 (3/8): a quarter read as dotted, so the voice overfills by half a beat. Gold has 52
dots we lack and we have 19 it lacks, most on handwritten pages. New rule after the tuplet and
beam rules: in a measure where some voice already fills the bar, a voice off by exactly half of
one event's undotted length takes that event's dot away (over) or adds one (short), chord notes
with it; only if exactly one event fits, never tuplets or double dots. CPU readings, gold as above,
against the step before (one beam too many), paired over 49 pages:

| Variant | Print authors' | Pages better / worse |
|---|---|---|
| Any voice off by half a dot | 8.30 to 8.09, -0.21 [-0.43, -0.01] | 10 / 8 (096 +2.0, 069 +0.9) |
| Only where another voice fills the bar (kept) | 8.30 to 8.09, -0.21 [-0.41, -0.02] | 10 / 6 (096 +2.0) |
| Also only with a written meter, never the guessed bar | 8.30 to 8.16, -0.13 [-0.28, -0.02] | 5 / 1 |

Corrected and whole-page move the same way (kept: Corrected 13.53 to 13.31, -0.22 [-0.42, -0.03]).
069 bar 12 is a short bar before a repeat, short in every voice: the filled-voice guard stops it.
096 prints no meter and its guessed bar is wrong (triplets read plain make bars look like 4/4),
so dots fill a wrong bar; the written-meter variant avoids that but loses 076 -2.3, 103 -2.0,
087 -1.4, which also print no meter and whose guesses are right. Kept the filled-voice variant:
the same bar source as the tuplet and beam rules. Better: 051 -2.3, 076 -2.3, 103 -2.0, 093 -2.0,
087 -1.4, 065 -1.0, 066 -0.9, 030 -0.5, 022 -0.3, 088 -0.1.

### OLiMPiC crops check for tonight's lmx_tuplets rules (2026-11-08, no GPU)

seq512's saved crop LMX (runs/seq512/benchmark/olimpic-systems, 1,438 crops) re-delinearized
with main's rules and with the new ones; note F1 by olimpic_benchmark.py on the 615 clean
systems, paired over 46 songs (F1 from summed tallies):

| Build | Benchmark F1 | Change by song | Songs better / worse |
|---|---|---|---|
| main (old rules) | 0.936 | | |
| All new rules | 0.937 | +0.0004 [-0.0011, +0.0025] | 9 / 13 |
| Without candidate subsets | 0.937 | +0.0006 [-0.0009, +0.0027] | 9 / 12 |
| Without subsets or dots | 0.937 | +0.0008 [-0.0006, +0.0030] | 4 / 3 |

No change is significant. The subset search causes the largest single loss (6259566 p3-s3, 24
to 12 correct: plain 16th arpeggios in beamed threes, some fitted as triplets to an overfull
voice). The dot rule makes most of the small per-song losses. Wrong tuplet marks cleared,
triplets under a second beam, and one beam too many under a second beam do no harm here.

### Final pipeline rules of the night, against main (2026-11-08)

Kept in eval/lmx_tuplets.py: (a) any wrong tuplet mark cleared with its bracket, (b) triplets
under a second beam, (d) one beam too many under a second beam, and dots taken away (not added)
where the voice overfills and another voice fills the bar. Dropped: (c) rest-led triplets with
candidate subsets (not significant on Polish, the worst OLiMPiC loss; saved as
rest-led-triplets.patch), and adding dots to short voices (a short second voice is often
legitimate; OLiMPiC worse). Same CPU readings, newest gold (stray beam ends, 076 moves, 091
slurs) on both sides, paired over 49 pages:

| Gold | Authors' | Whole-page |
|---|---|---|
| Corrected | 13.89 to 13.50, -0.39 [-0.61, -0.18] | 12.48 to 12.01, -0.47 [-0.74, -0.23] |
| Print | 8.79 to 8.28, -0.51 [-0.86, -0.23] | 7.71 to 7.25, -0.47 [-0.73, -0.23] |

16 pages better, 3 worse (076 +0.61 and 091 +0.2, reading-choice flips; 022 +0.56 from a dot
removal). OLiMPiC crops (seq512's saved LMX re-delinearized, 46 songs): -0.0001 [-0.0005,
+0.0003], 3 songs better, 4 worse: no change.

## Ties read as slurs: accents and carried accidentals (2026-11-08, CPU readings of seq512)

Missed ties were 133 of 9,522 Print cost on the final pipeline build (asm_v9). On 039/079 the
model slurred an accented A4+C5+A5 chord to the same chord in the next bar. Two causes:
`plain()` rejected any articulation (meant for portato), and the second chord's C came out C#
because the bar's natural lapsed at the barline, which a tie would carry. Fixes in
page_musicxml: only portato marks block the rule; across a barline, same staff places with no
written accidental on the second event become a tie that carries the first's alterations.
Per page, Print: 064 -0.10, 069 -0.04 -0.04, 077 -0.11, 030 -0.05, 037 -0.12, 039 -0.38, 071
-0.18, 079 -0.43; Corrected the same within 0.02. No page worse. All 10 carried-accidental
places checked against gold: ties with the first chord's pitch.

## Accidental on a shared head (2026-11-08, CPU readings of seq512)

accidentins was 74 on Print, 24 of them on 037: F5 natural in voices 1 and 2 at the same onset,
gold showing the natural in both, the model in voice 1 only. 57 such mixed unisons on dev, 19
shown in both, 292 in neither. `shared_heads_shown_alike` copies the accidental to every voice of
the unison. Print -0.025 [-0.057, -0.001] (Corrected the same); 037 -0.60, 088 -0.29, 003 -0.22;
worse: 055 +0.13, 096 +0.09, 110 +0.05 (two are gold inconsistencies, filed).

## Key-restating accidentals (2026-11-08): refuted

Dropping the shown accidental on notes that restate the key with no change at that step in this
or the previous bar (125 notes on 30 pages): Print 8.22 to 8.48, +0.26 [-0.02, +0.68]. 030
+8.1, 058 +3.2, 021 +2.3 (locked-alignment swings); 17 pages slightly better. No rule.

## Hidden tuplet numbers (2026-11-08)

Edits where we and gold disagree on whether a tuplet's number shows: 49 on 055/077/110/076/088/081.
The scorer charges `edittuplet` when the shown numbers differ ('' vs '3'). The Print gold shows
every number. On 077 the print shows none (checked by eye; 095's printed 3s stay visible at
700 px). Test: `*Xtuplet` on all spines at the top of a copy of 077's gold changes 077's
tuplet edits from 11 to 56. Our 49 model-marked triplets show the 3, and only bar 4, which the
model left unmarked, matches. Cause: crop labels keep `tuplet:start`/`stop` when
`hide_tuplet_numbers` hides the number, so the model's tokens don't say whether a 3 is printed.
Proposed fix: labels drop start/stop for hidden numbers and keep `3in2`. Not built: it changes
OLiMPiC comparability and waits for the user.

Follow-up: implied tuplets closed. The 33 grouping edits ("continue" where gold has "stop" or
"start") come from triplets the model wrote with no marks: music21 reads them as one group
running on. `tuplets_closed` in page_musicxml closes each group as it fills its ratio, within a
bar, with no number shown, and skips notes inside a group the model opened (without that skip
it nested groups on 087 bar 5: 087 +0.12, 110 +0.07). Final: −0.036 [−0.061, −0.016] on Print,
Corrected and both whole-page variants; 12 pages better, 0 worse (096 −0.38, 077 −0.25,
055 −0.24, 095 −0.24, 088 −0.18).

## Ties into a chord (2026-11-08)

Slurs (Print, current build): 270 extradel, 169 extrains, 92 extradurationedit, the largest
cost outside notes. 096's left hand has short arcs from a note into the next chord holding that
pitch; gold ties them (into another voice), but the tie rule wanted equal pitch sets. There
were 20 such slurs in the readings. Gold has a tie in the same bar at 19 of them; at the exact
onset, only 7, but our 16ths and triplets sit at shifted offsets, and by eye 096 bar 8 and 055
bar 6 are ties in gold. 060 bar 3 is a slur in gold (16th into a chord). Rule: tie only the shared
pitches when every note of the first event is in the second. −0.016 [−0.030, −0.003] on all four
scorings, 9 better, 1 worse (060).

## Slur starts with no stop (2026-11-08): refuted

84 of 1,565 slur starts in our page output have no stop (cross-staff on 062, 067, 037). Closing an
orphan stop against the other staff's open slur: +0.038 [+0.007, +0.086], worse on 11 pages
(062 +0.97). Dropping the open starts: identical scores (music21 ignores them). The remaining
extra slurs (057, 022) are small slurs on a faint scan; this needs gold review at a finer scan.

## Time signature symbols (2026-11-08)

Time signatures we write that gold lacks: 6 pages, mostly one-off model reads (005, 023, 097
3/4). 079: gold had `*met(c|)` without `*M2/2`, fixed by override. Then 079 cost more (+0.084),
because our plain 2/2 took an info edit against gold's cut symbol. The LMX has no symbol token.
Rule `time_symbols_drawn`: 4/4 → common, 2/2 → cut. −0.009 [−0.024, +0.000]: 079 −0.25,
077 −0.21, no page worse.

## Ties over a system break (2026-11-08)

After the earlier tie rules, Print still had 94 tie insertions (gold ties we lack) and 22 tie
deletions. 22 of the 76 missed ties on the 9 worst pages sit at a system end. A tie over a
break is printed as two half arcs, and the model, reading one system at a time, writes each
half as a slur end with no partner: a slur stop on the next system's first note (070 bar 5,
108 bar 18, 056 bar 12), a slur start on the last note (070 bar 9), or a pair that
pair_slurs joins across the break but a tie stop on the same note blocked (088 bar 17). A stray
stop also closed an unrelated slur still open on its staff (070: B1's stop closed F♯3's slur).
- Rule `ties_across_system_breaks`, run before pair_slurs: `seq_pages.joined_musicxml` now marks
  each system's first bar with `<print new-system="yes"/>`; across that bar line, a voice's last
  event and its next with the same notes (tie_partners) are tied when either shows a half arc
  (slur start before, slur stop after, or a tie end), and the half slurs go.
- First version grouped a voice per staff: 062, 067, 093 worse (+0.03 to +0.06), where a staff-1
  voice puts one note a beat on staff 2 and slurs to it; those notes were taken as neighbours.
  Grouping by voice number alone: Print −0.008 [−0.018, −0.002], 7 better (088 −0.18,
  105 −0.08, 071, 108, 037, 056), 0 worse; Corrected the same; whole-page −0.008, 0 worse.
  Tie insertions 94 → 87, deletions unchanged, distance 9365 → 9355. 7 new tests (78).
- Not covered: the model writing a slur stop where the tie starts (108 bar 5), half arcs it
  left out (076 bar 24), and voices numbered differently in the two systems.

## Naturals read as sharps and flats (2026-11-08): model-side, no rule

Accidental edits on Print (31): 28 are ours sharp or flat where gold has a natural (18 sharp,
8 flat, one each double), on 055, 022, 087, 081, 095. The pipeline keeps the model's signs (055,
022, 087: 90 natural, 55 flat, 19 sharp tokens in the readings, the same in the output). On
OLiMPiC dev scans the model shows no such bias (natural→sharp 10, sharp→natural 6, out of about
3,100 naturals), so it is this engraving at this scan: 055's naturals (bass bars 4-5, the ♮E and
♮D) are small and look like sharps at 1050 px.
- Rule considered: a sharp on a step the key flats (or a flat on one it sharps) is a natural. Our
  output has 34 such sharps: about 20 are misread naturals (022, 055, 081, 087, 093, 095), but 16
  are on 060, which spells a passage in sharps without a key change, and gold has them too. Net
  gain about 8 edits with a known page it would break: not done.
- Fix belongs in training: naturals drawn small or thin in the crops, or the page at a higher
  scale. For the next run, with the other seq514 items.

## Missed ottavas on 023 and 110 (2026-11-08, after seq513)

seq513 reads its own synthetic ottavas perfectly (60 training crops with signs: every one read,
960 tokens against 913), so the model learned the sign; the real pages differ. Two causes.
- 023: the sign's style. A bold upright `8` over a chain of short slanted dashes, about 4-5
  interlines up. No checkpoint reads a token on its crops, and white paper, binarizing, padding,
  darker ink or rescaling change nothing. Pasting 110's italic `8` alone: 0; redrawing the line as
  dots alone: 0; both together: 17 tokens (about the notes under the sign in that system). Fix in
  training (seq514.patch): an upright bold `8` (DejaVu Serif Bold) in 30% of marks-style signs and
  a chain-of-dashes line in 30%, checked by eye against 023.
- 110: the model does read it, but only just. seq513 best.pt: 38 tokens on the last system at one
  crop height, 0 at the other; the two fill their bars equally, and the earlier height wins
  (best_readings is blind to ottavas, which change pitch, not rhythm). last.pt: 10 against 36, the
  10 filling better. 30 px of white above the crop makes both heights read it (36, 38) with no
  false sign on 049. As a pipeline rule (crop-margin.patch, 1.5 interlines above and below every
  crop) it moves 110 the other way too (best.pt 20 tokens at one height and now chosen; last.pt
  about 11): not kept until tested on all dev pages. seq512 reads no token on 110 at either height.
- Tested on all 49 dev pages from saved per-height readings (seq513 best.pt, whole-page OMR-NED,
  rescoring only changed pages). Crop margin (1.5 interlines above and below): worse, Print 7.11 to
  7.39, Corrected 11.69 to 11.77, changes 37 pages: rejected. Selection, equal fill broken by more
  ottava tokens: Print 7.02, Corrected 11.65, from three pages only (110 -8.1, 087 +3.5, 076 +0.2 / +3.2).
  Carrying the other height's ottava marks onto shared notes: Print 7.04, Corrected 11.57 (110 -5.7,
  076 +1.6 / -0.5) and false signs on 064, 065, 093, which have none. Neither kept: each is 110's gain
  less losses elsewhere, too few pages to tell a rule from tuning to one page.
- Real scans, OLiMPiC scanned dev (56 crops with signs, 510 shifted notes; 400 without):
  seq513 reads a token on 41 crops, 460 tokens, false on 2 of 400; seq512 29 crops, 363 tokens, 1 of 400.
  seq513's ottava training works on real scans in general; 023's type and 110's weak read are the
  exceptions. This is the check to run on seq514 (scratchpad olimpic_ottavas.py).

## seq514 (2026-11-09)

Trained 192k steps on seq513's data plus MuseScore crops (53,854), relabelled GrandStaff
(barlines, time symbols) and the seq514 styles; averaged weights (decay 0.9995). Paused once at
140k for the user's GPU use and resumed from last.pt. Scored: `runs/seq514/average_192000.pt`,
the final step's averaged weights (written from last.pt: snapshots were kept only at multiples
of 10,000; fixed in train_seq.py).

- OLiMPiC scanned dev: SER 7.37% (seq513 best 8.79%); note F1 of best.pt 0.943 against 0.934.
  Token edits 27,849 against 33,221; tuplets missed 4,022 against 6,113, durations 4,844 against
  6,118, pitch 3,331 against 3,744. Ottavas read in 43 of 56 crops (seq513 last 39, seq512 29), no
  false sign in 400, but 546 tokens against 510 gold. Old-style quarter rests and 16th-for-32nd
  fixed; quarter-for-half up (152 against 110).
- Polish dev (49 pages, whole-page / authors'), against seq513 last.pt, same page code:
  Print (now with D) 7.73 / 8.39 against 7.68 / 8.58; Corrected 12.00 / 12.58 against
  11.92 / 12.38. No difference significant. Gains: 023 (-14 Corrected, ottava read), 110, 083,
  091, 059, 064. Losses: 012 and 105 (octave misread, below), 096 (dense voices), 017.
- Page output crashed on 065 (a grace eighth read into a 16th's beam); fixed in lmx_tuplets.
  The fix changes seq513's 077 and 079 (+0.009 Print): comparisons use pages rebuilt with it.
- D, Print gold for unprinted tuplet numbers, by page image: no number anywhere on 003, 037, 056,
  077, 088; mixed on 096 (bars 1-13 and 16 hidden), 110 (6-7, 23-27), 060 (31), 095 (1, 3, 13-15);
  printed on 010, 017, 042, 065, 076, 081, 087, 108. On 065 seq514 wrongly hid 5.
- Cause of the octave misreads: MuseScore crops with a sign that runs on into the next system
  (TODO, "MuseScore crops drew no ottava sign"). seq514 at both crop heights reads 012's low
  treble notes an octave up, the F#4 in the bottom space as F#5 on the top line: the octave is
  not read from the staff position.

- Rest-led triplets with candidate subsets (rule c, 2026-11-08): deleted 2026-11-09 with the
  user. Never significant on Polish (Print -0.10 [-0.28, +0.02]), the worst rule on OLiMPiC, and
  the one most likely to rewrite a bar around a single misread note (a subset that happens to
  fill an overfull bar). The rules kept in lmx_tuplets are re-measured on each new model, with
  the bars each makes worse counted, not only the net (TODO).

### Page-output rules re-measured on seq514 (2026-11-09)

Each kept rule in eval/lmx_tuplets switched off in turn on the same readings (49 dev pages,
whole-page OMR-NED), paired against all rules on; "helps / hurts" counts pages:

| Rule | Print: change with the rule off | helps / hurts | Corrected: change |
|---|---|---|---|
| Implied tuplets (incl. triplets under a second beam) | +0.62 [+0.15, +1.22] | 10 / 3 | +0.60 [+0.14, +1.21] |
| Wrong tuplet marks cleared | +0.46 [+0.03, +0.94] | 13 / 2 | +0.46 [+0.04, +0.93] |
| One beam too many dropped | +0.13 [+0.01, +0.29] | 5 / 0 | +0.13 [+0.01, +0.29] |
| A dot too many taken away | +0.04 [+0.01, +0.09] | 6 / 0 | +0.04 [+0.01, +0.09] |

All four still pay on seq514, each range clear of zero; the two tuplet rules make 5 pages worse
between them. Same check for seq515 is in the scratchpad's after515.sh.

## Note errors on Polish dev, seq514 against seq513 (2026-11-10)

OMR-NED gives a page's cost; `eval/note_errors.py` says what kind of reading went wrong. For each
gold note (Print gold, 49 dev pages, sounding pitch, graces left out): right, wrong duration, an
octave off, another wrong pitch, wrong onset, missing; plus the prediction's extra notes. Bars are
paired by an order-keeping alignment on onset and pitch class (difflib paired repeated passages
far apart; exact pitch left bars read an octave off unpaired).

| Fate of gold notes | seq514 | seq513 last.pt |
|---|---|---|
| Right | 14,404 (91.9%) | 14,380 (91.8%) |
| Wrong duration | 147 | 154 |
| An octave off | 95 | 129 |
| Another wrong pitch | 357 | 322 |
| Wrong onset | 186 | 149 |
| Missing | 486 | 541 |
| Extra (in the prediction) | 453 | 463 |

The two models make about as many note errors (1,724 against 1,758), in different places: seq514
fixes 023 and 110's ottavas and loses 012 (octave, the MuseScore ottava bug) and 096.

Conclusions:
- The errors are concentrated: six pages hold half of them (096, 051, 093, 065, 087, 056), twelve
  pages two thirds. A page-level fix, not a general one, is what each of these needs.
- **Handwriting is the largest single cause.** 051 and 093 are manuscripts, second and third worst
  (359 errors, 21% of all). The sequence model has no handwritten training data (TODO: MUSCIMA++
  as LMX; Smashcima's problems). Handwriting is in scope at 5-10% of training (memory).
- **Unmarked triplets missed** are the next: 096's worst bars (8-10, both staves) are triplet bars
  where seq514 drops the 3in2 marks, so every later onset in the voice shifts (onset 51, missing
  and extra 111 on that page). OLiMPiC scans say the same: missed tuplet tokens are seq514's
  largest error kind there (4,022). The over-hidden-number label bug fixed for seq515 may be part
  of it; seq515 will tell.
- Octave errors: 012 (fixed in seq515's data); 087 (17) and 083 (24 at scorer level) are ottava
  pages, not yet looked at.
- Not the cause: durations alone (147, 1%) and barline styles (seq514 writes 19 double and 42 final
  barlines where gold has 19 and 44).

Next, by size: handwritten training data for the sequence model; then whatever of the triplet
misses seq515 leaves; then 087 and 083's ottava extents.

## Smashcima fork: handwritten crops for the sequence model (2026-11-10)

Smashcima (OMR-Research, Apache-2.0) draws MusicXML with MUSCIMA++ writers' symbols. Forked to
`ErikGoldman/Smashcima`, branch `noteworthy`, cloned at `../Smashcima` (pushed). Our builder:
`train/build_smashcima_crops.py`, run in `.venv-smashcima` (project requirements plus the fork,
OpenCV 5; the fork's asset install needs OpenCV 4 contrib, done once into `data/raw/smashcima`).

Speed. A system took about 4 s, 76% of it a pure-Python Dijkstra seam in the paper quilter.
Replaced by the equivalent dynamic-programming seam (same least cost, checked against the old
code on random errors). The seam error was also computed on uint8 arrays (it wrapped around) and
patches came from the unseeded global numpy generator. With a page sized to the system: 0.55 s a
system, about 4 systems/s on 4 workers.

What it drew before, and now. Upstream draws noteheads, stems, beams, flags, rests, accidentals,
dots, clefs and plain barlines only: no key or time signature (so a label's key disagreed with
every image), no tie, slur, tuplet number, articulation, final, double or repeat barline, clef
change or brace. All of these are now drawn, from MUSCIMA++ samples where it has them (ties,
slurs, triplet 3s, accents, tenutos, heavy barlines, repeat dots, 0s and 1s, newly extracted);
9s are half-turned 6s, cut-C a C with a stroke, small clefs scaled clefs. Ties and slurs are the
sample nearest the target's shape, stretched and sheared between the notes, the pen width put
back; ones open at a system edge run to the edge (as a print's half arc). Barlines are scaled to
one staff (MUSCIMA++ heavy barlines can span the system). Loader fixes: a second `<attributes>`
at one onset replaced the first; double sharps and flats crashed it.

Labels. The builder strips what Smashcima still does not draw and the music can go without
(grace notes, fermatas, arpeggios, ornaments, strong accents, ottava signs, non-triplet tuplet
numbers, hidden then in the label) and leaves out a system with a token it cannot draw (32nd and
shorter, breves, double flats, stemless notes), a hidden pitched note, or a measure drawing
nothing. Hidden rests are drawn as nothing, as in a print (the fork reads them as `<forward>`).

Failures, on 600 PDMX scores through the builder (946 systems): 44 failed to draw (4.6%), not the
1% first measured on hand-cut systems. 40 were the page-edge check: the crop's random margin ran
past a 3 mm side padding, and some writers' braces reach 8.3 mm left of the staves (the left edge
was not checked, so braces were cut). Now 12 mm for the brace plus the widest margin, every edge
checked. The other 4 were fork bugs, fixed: a whole-measure rest without a time signature took 4
beats, not its own `<duration>`; a unison of up- and down-stem voices next to a second lost the
side for one stem; a measure of only `<forward>` had no event for its signatures. After, on a
fresh sample (700 scores, 1,111 systems, after the units fix below): 11 failed to draw (1.0%),
all left out rightly: 5 reach past the padding (chords on 6+ ledger lines, a few very dense
systems), 4 give a key for staff 1 only (the label has one key for both), 1 has display-octave
39, 1 a beam open at the system's end.

Crops cut short (found by eye, 2026-11-10). Upstream Smashcima put 24.4 mm in an inch
(`geometry/units.py`), so it rendered 4.1% more pixels per millimetre than the builder assumed:
every crop lost up to 4% of its width at the right (12 mm of a 300 mm system), the last notes
and barline of labels that still held them. Every Smashcima crop before this fix is wrong;
none was used in training. Fixed in the fork (25.4; glyph assets re-extracted as version 3, as
they store millimetres), the builder converts with Smashcima's own units, and the page layout
constants are the old layout in true millimetres. The stem model was trained in true
millimetres, so its stems were 4% long before; now as measured.

Curves. A few slurs were drawn as thick blocks: a tiny sample (17 x 5 px) enlarged 6 times, its
stroke with it. Now the sample is picked by size as well as shape, enlarging interpolates, and a
stroke grown thick is thinned back to the sample's pen width. What is left out, on 5,867 systems of 600 scores: 13% held an undrawn token
(7.3% hidden rests, now drawn; 4.5% 32nd notes or shorter, which MUSCIMA++ has no flags or rests
for; breves 1.2%, double flats 0.9%), but only 6 of 467 scores had no drawable system. Accepted:
fast passages are under-represented within scores, not scores themselves. Every build writes
`outcomes.json` (what became of each system) and `manifest.json` (both commits, uncommitted
changes, the score list's hash, options); a restart with other code stops.

Held-out writers. Glyphs from MUSCIMA++ v2's two test sets (our handwritten dev and test pages,
40 pages) are not extracted: 43 writers remain (asset version 2).

Stem positions, after Shatri, Mayer and Hajič, "Data-Driven Glyph Positioning for Handwritten
Music Synthesis" (arXiv 2610.28899; no code or model released, none on Hugging Face). Their GNN
takes every glyph's position as input, the stem's own too, so it cannot run before the stem is
drawn; their synthesizer used hand-derived means instead. Their "stem length 3.8 mm" is a
bbox-centre offset, about half a stem: as a length it would halve stems. Here
(`train/stem_positions.py`) each chord is a node with only what Smashcima knows before drawing
stems (notehead staff positions, stem direction, beams or flags, beam group, place on the staff),
edges to its neighbours and its beam group; a 4-layer relational GraphSAGE (128 wide) predicts
the length from the notehead nearest the tip to the tip. 14,832 stems from the training pages,
5 folds by writer:

| Stem tip error on unseen writers (RMSE, mm) | All | Single | Beamed |
|---|---|---|---|
| Smashcima, constant 5 mm | 2.93 | 2.26 | 3.30 |
| Global mean | 2.19 | 1.86 | 2.38 |
| Means per direction, beamed, beam count (the paper's kind) | 1.97 | 1.51 | 2.22 |
| GNN | 1.46 | 1.34 | 1.54 |

Real stems are 6.2-7.6 mm by this measure, longer than Smashcima's 5 mm. The fork runs the network
in numpy (matches torch to 2e-6; the feature scale is exported with the weights) and adds noise
in the parts the held-out errors (1.37 mm) split into: a writer's habit, 0.66 mm, drawn once per
system; then a lone stem's own, 1.09 mm, or for beamed stems a beam group's, 0.89 mm, and each
stem's within it, 0.91 mm (one-way random effects). Independent noise per stem had tilted beams
at random and, since the beam is placed so as only to lengthen stems, made beamed stems too
long. Beamed stems are then refitted to the beam line as before. Not yet modelled: beam slope, flag
and dot offsets (the paper finds no gain on dots and accidentals), spacing between notes.

Not yet done: a training run with these crops; their share is meant to be 5-10% (about 19,000
crops at one per score against seq514's 269,000). Crop checks (docs/TRAINING_DATA_CHECKS.md)
done on a 671-crop sample for tuplets, accents, repeats and double barlines.

## MusiCorpus in training, and what zero-shot means (2026-11-10)

The protocol said "all synthetic, no labelled real scans". Amended (user's decision, before any
test run): zero-shot means nothing of a benchmark is trained on, its pages, pieces or editions;
training data is otherwise chosen for what helps. seq516 adds 53 real piano grand staves from
MusiCorpus 1.0's OmniOMR training split (39 handwritten, 14 engraved;
`train/build_musicorpus_crops.py`, checked by eye: docs/TRAINING_DATA_CHECKS.md).

Overlap with the benchmarks, by the protocol's melodic check (`eval/benchmark_overlap.py`: the
largest share of a benchmark item's 8-interval fragments found in one training piece; 0.10 marks
a copy):

| Benchmark | Items | At or over 0.10 | Largest share |
|---|---|---|---|
| Polish | 112 pages | 0 | 0.061 |
| OLiMPiC scanned (dev and test) | 200 songs | 0 | 0.077 |
| Mutopia (dev and test) | 17 pieces | 0 | 0.001 |

No benchmark item shares a piece with them. The check, a scratch script before, is now in the
repo, for every new training source.



## Ties and slurs across system breaks, and spacing (2026-11-11)

On OLiMPiC dev, seq515's ties and slurs open at a system's edge (a print draws a half arc there)
are 5% of the gold's marks but 26% of its tie and slur count errors (error rate 137% against 22%
inside the system): our crops, each system engraved alone, never drew the arc. From seq516:

- Verovio crops of such systems are engraved between the measure before and the one after, each
  on its own line, and only the system's line is kept: Verovio splits the arcs as on a page.
  The neighbours bring only notes (no courtesy signatures, no ottavas reaching in).
- Every Verovio crop gets random room after the clef, key and time signatures and before the
  closing barline (Verovio's margins, up to their maximum), and 60% are justified over a page
  1.05-1.35 times their natural width. Prints leave 1.5-2.5 staff spaces before the first note
  and 2-4 before the barline (12 OLiMPiC dev examples); the arcs fill that room. Applied to every
  crop, so room alone does not signal a tie.
- MuseScore crops leave such systems out (it stretches a page's non-last lines to its width).

Measure after seq516: the same edge count on OLiMPiC dev.
