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

Crops that depended on the worker (found 2026-11-10, rebuilding a sample from a fresh setup:
21 of 106 images differed, labels all equal). Smashcima's service container kept its types in a
set, which orders them by memory address, and the Styler picks each style domain (writer,
paper) from the shared rng in that order: the same seed drew a different writer and paper with
another worker count, or even another run. Fixed in the fork (67e6563, registration order);
now 60 scores build byte-identical at 1, 3 and 6 workers and under any PYTHONHASHSEED. Crops
built before are not reproducible (none was used in training).

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

## The meter as context: A/B fine-tunes (2026-11-11)

Does giving the decoder the system's time signature (`meter beats:N beat-type:M` after `<bos>`,
out of the loss, dropped on 15% of training labels) help it read rhythm? Two fine-tunes of
seq515's final averaged weights, 15,000 steps each on seq515's data, seed 0, identical but for
`--meter-context` (B).

| | A (control) | B (meter) |
|---|---|---|
| OLiMPiC dev SER in training (averaged, 5k / 10k / 15k) | 6.94 / 6.79 / 6.67 | 6.72 / 6.83 / 6.67 |
| OLiMPiC dev, seq_predict (B given each label's meter) | 7.28 | 6.95 |
| Wrong bars of 5,340 | 956 | 889 |
| A voice does not fill the meter | 588 | 533 |
| Fills the meter, still wrong | 368 | 356 |

The seq_predict row hands B the true meter, an upper bound. On Polish dev (56 pages, v5
segmenter, B reading each page twice to find its meter), A - B with 95% intervals:

| Gold, alignment | A - B |
|---|---|
| Corrected, authors | -0.02 [-0.74, +0.91] |
| Corrected, whole page | -0.29 [-0.79, +0.17] |
| Print, authors | +0.41 [-0.35, +1.36] |
| Print, whole page | -0.09 [-0.57, +0.36] |

No measurable difference on real pages, so seq516 trains without it. The code stays (the
prefix is opt-in), for a larger test if rhythm errors lead again.

## A label sweep, and GrandStaff-LMX's clefs, keys and meters (2026-11-11)

seq515 read GrandStaff's training crops in a seeded random order; the crops it reads worst are
the likeliest bad labels. Of the first 773, none of the 474 whose source has no spine split read
above 0.2 SER (mean 0.003); all 23 above it split. Some of those are only hard (cross-staff
sixteenths); the worst, Joplin's Bethena, had a treble clef on a bass staff and bass notes moved
between measures. Following that up over all of GrandStaff-LMX against its **kern sources:

- No label has a clef change; 13,459 systems' sources change clef with notes after the change
  (25%), and their images draw it. In other sets, 13-22% of labels carry one.
- 3,512 systems label a staff treble where its spine splits before its clef: the bass staff of
  Bethena and of every system like it.
- 416 systems' notes differ from the source measure by measure (grace notes aside); two looked
  at move notes between measures.

Sweeping again on clef-mended labels found the same loss for keys and meters: a staff whose spine
splits before its signatures is labelled C major with no meter (sonata 5-1 in five flats and 3/4
labelled `key:fifths:0`, a flat written on every note), 10% of seq516's GrandStaff crops.

The cause is music21's Humdrum parser and MusicXML export in OLiMPiC's build: a change is put at
offset 0 beside the opening signature or outside any measure, and a signature after a split is
stored in the voices; the export writes none of them. OLiMPiC's build, rerun as it is (music21
9.9.1, MuseScore 3.6.2), gives back the release byte for byte, so `train/rebuild_grandstaff_lmx.py`
runs it with one change: after the parse, music21's signatures are replaced by the source's, at
their offsets. 20,757 systems are mended (39%): 13,477 get their clef changes, 4,825 their key,
5,674 their meter; 107,206 accidental tokens no image draws are gone. The rest come out byte for
byte the release's, but for 499 left out as unlike their source (243 whose notes differ, two of
which, looked at, move notes between measures). `build_grandstaff_written.py` takes the mended
labels: 32,156 training systems (34,912 before). Left out, in the order checked: 373 unlike their
source; 50 whose source merges the two staves into one spine with notes after it (68 in all
splits, found by the sweep resumed on the mended labels: Verovio drew no notes after the merger,
only empty measures, and OLiMPiC's build cut the label there on purpose, so the image has bars the
label lacks; `sonata07-3/maj2_down_m-35-40`, one measure of notes, three empty bars, seq515 wrote
notes into them); 2,984 with a grace note in the source (GrandStaff's images draw each as a
stemless notehead, often under a stray dash, drawn in all 8 checked by eye, and 1,652 of the
2,498 left after the tuplet check have none in their label; OLiMPiC synthetic, drawn by MuseScore,
has real ones); 2 with unmatched signs (112 before, the meters now matching); 6,027 with tuplets;
6 with a blank image (29 of GrandStaff's images, mostly distorted ones, are one grey or nearly
black, found by the same sweep). The Polish gold, read by the same parser for OMR-NED, keeps its clef
changes (on each dev page music21 has at least as many clefs as the source's 95 clef lines after
notes; the extra are copies in each voice of a split), so the benchmark should not be touched;
matching them one by one is still to do. For the dataset's other users: `errata/grandstaff-lmx/`
(what is wrong, why, a patch, every system's status).

The sweep after the mend: 11,184 of GrandStaff's mended training crops (mean 0.029 against 0.018
on the release's labels, seq515 having learned no clef changes) and 1,000 of OLiMPiC synthetic's
(mean 0.029, 19 above 0.3). Every crop looked at among the worst of either agrees with its image
but the merger systems above: seq515's misreads (16ths in threes read as triplets, whole notes as
breves, a lost staff in seven flats) and voice order (arpeggiated chords, an empty staff's
invisible rest). No new kind of label error; OLiMPiC synthetic's labels, built by MuseScore from
MusicXML rather than from **kern, need no further sweep.

seq516 trains on `data/seq516f_data`: seq516_data's sets but GrandStaff, rebuilt as above (the
same as a fresh `build_seq_data.sh`, the other builders being unchanged; its rebuild is
seq516d_data's, the merger and grace-note checks needing only the source).

## seq516 against seq515, stopped on Polish dev (2026-11-12)

seq516 (`runs/seq516`, `data/seq516f_data`) is seq515's recipe with four changes to its data: the
cleaned GrandStaff set above, and three sets seq515 did not have: Smashcima's handwritten crops
(`seq2/smashcima`, weight 2), MusiCorpus's 53 real grand staves (`musicorpus`, weight 30) and the
unmarked-tuplet set (`seq2/unmarked`). As an experiment it was stopped once Polish dev flattened (docs/EVALUATION_PROTOCOL.md):
each averaged snapshot from 130,000 steps scored on the 56 dev pages with the current segmenter,
seq515's final weights scored the same way. OMR-NED, two-staff pages (49), authors' alignment:

| | Print | Corrected |
|---|---|---|
| seq515, 192,000 | 8.69 | 12.96 |
| seq516, 130,000 | 6.87 | 10.81 |
| seq516, 140,000 | 6.81 | 10.93 |
| seq516, 150,000 | 6.97 | 10.84 |

seq516 at 130,000 against seq515 at 192,000, paired by page: Print -1.82 [-2.91, -0.79], better
on 37 of 49 pages; Corrected -2.16 [-3.12, -1.25]. Three snapshots within 0.16 of each other: the
run was stopped at 154,100 steps, with the learning rate at 2.8e-05 of 3e-4. OLiMPiC scanned dev
(train_seq.sh's check) was flat from 100,000 too (0.063-0.065; seq515 0.065-0.075). The four data
changes together are worth about 2 points of OMR-NED on real two-staff pages; the run does not
tell them apart, and the analysis below suggests the new sets, not the GrandStaff cleaning, did
most of it.
Scored by a scratch script (each snapshot in its own folder: benchmark_pages.sh reuses pages
already read in its output folder).

### What changed from seq515 to seq516, and what is left (Print gold, 49 two-staff dev pages)

seq515 at 192,000 against seq516 at 150,000. Gains on the pages that improved sum to 113 points,
losses on the pages that got worse to 28. Edits (one-block alignment) 10,220 to 8,121: notes
7,496 to 5,783, tuplets 232 to 101, bars 637 to 533, marks 889 to 816, accidentals 214 to 176.
Gold notes (`eval/note_errors.py`): another wrong pitch 360 to 236 (a semitone off 169 to 110, a
whole tone 96 to 45), an octave off 105 to 80, wrong onset 165 to 134, missing 545 to 430, extra
426 to 349, wrong duration 150 to 128.

- **Where it gained:** 096 (-12.1, the unmarked-triplet page), 091 (-12.1, 31 grace notes), 064,
  093 (-11.2, a manuscript), 055, 065. Each known cause from the seq514 note analysis shrank:
  handwriting (093, 051), unmarked triplets (096), octaves. Not clefs: both runs get every clef
  change of 18 of the 22 pages with clef changes, and pages with clef changes gained no more than
  pages without (-1.65 against -1.78 on average); nor pages in keys of three or more accidentals
  (-1.80 against -1.65). So the gain fits the new sets (handwriting, unmarked tuplets, real
  staves) better than the GrandStaff cleaning, which this run cannot separate out.
- **Where it got worse:** 042 (+5.9) and 017 (+4.2) only under the authors' alignment (bars pinned
  to the wrong copy; whole-page +0.2 and -0.3). Real: 047 (+4.3: a doubled note and an inner voice
  dropped in two bars, eighths read as triplets in one), 083 (+2.8: a quarter-note chord and an
  eighth rest read three times as a half-note chord, every later onset of the bar a beat late),
  069, 005, 103, 076 (+1.6 to +1.9). Reading slips, no common cause found.
- **What is left:** notes are 71% of the cost (5,783 of 8,121 edits), then marks (816) and bars
  (533). Ten pages hold 53%: 096 (950 edits, 12%: 72 notes missing, 62 extra; its bars not yet looked at again), 110 (833, 10%: 449 bar
  edits, the ottava page), 056, 065, 076, 051 (manuscript), 093 (manuscript, 38 octave errors),
  087 (ottava). Handwriting, unmarked triplets and ottavas remain the leading causes.

To value the GrandStaff cleaning alone: seq516's data with the release's GrandStaff labels
(`data/seq516_data`), stopped at the same step.

### Does reading need context from earlier systems? (seq516, Polish dev)

Before building attention across systems: the 6-8 bars with the most note errors on six of the
ten costliest printed pages (096, 110, 056, 087, 076, 065), each checked on the page image for
whether what it needs lies outside its own system. None does. These engravings restate at each
system's start what is in force: clef, key, and an ottava crossing the break (087 prints its 8
again); triplet numbers sit in the same system as the unmarked groups that follow them (096's
last bar); a trill line running on (076) is a mark, not notes. What the bad bars hold instead:
dense tied chords (096 bars 11-13), printed triplets among rests (110 bars 4-5 and 22, 087 bar 5),
hand crossing with notes beamed across the staves (056 bar 18), an ottava's extent and a
"loco" run with grace notes (087, 076 bar 20), a faint, noisy scan (087), a first and second
ending (065 bar 9). The manuscripts 051 and 093 were not checked: handwriting is in the crop
too. Context from earlier systems would not fix these bars; reading within the system would.

## StaffSeq2Seq, a reader that takes a system as its staves, and back to Seq2Seq (2026-11-12 to 11-16)

From here the log follows `StaffSeq2Seq` (each staff its own image, attention across staves;
`train/seq/model.py`, branch `staff-encoder`), raced against `Seq2Seq` and retired at "(l)" below:
with training matched its lead was a head start. Everything after (l) is Seq2Seq, on branch
`rhythm`. This part of the log was `EXPERIMENTS_STAFF_MODEL.md` until 2026-11-16; its sections
follow unchanged, so "the staff model" in them is StaffSeq2Seq and "the main log" is this file.

## Why (2026-11-12)

Today's reader, `Seq2Seq`, takes a whole system crop at 512 px: a ResNet over the crop, two
transformer encoder layers over its 16-row feature grid, a 6-layer decoder writing LMX. Two
problems with that:

- **Three or more staves have to be squeezed.** At 512 px a two-staff system gets about 8 grid
  rows a staff; three staves would get 5 and lose a third of every staff space. Polish dev's seven
  three-staff pages (songs) score about 90 because the pipeline reads only the piano pair.
- **Context across systems.** The idea was to reduce each staff to vectors, attend across staves
  for the system, then across systems for the piece. Checked first (EXPERIMENTS.md, "Does reading
  need context from earlier systems?"): none of the worst bars of six costly Polish pages needs
  anything outside its own system; engravings restate clef, key and ottava at each system. So
  the piece level waits for evidence; the staff level is justified now.

The plan, in stages: (A) per-staff encoding with attention across staves, no piece level; (B) a
piece level only once a benchmark shows errors that need it. Before any training run, small checks
in order: (1) shapes and masks, (2) splitting crops into staves, (3) overfitting 64 systems with a
blank-input control, (4) blanking a staff and the attention across staves on the overfit model,
(5) speed and memory, (6) a 10,000-step race against `Seq2Seq` on identical GPUs.

## The model

Each staff is read as its own image at 256 px (the same pixels per interline as a system at 512),
through `Seq2Seq`'s CNN. Its feature grid (8 rows, width / 16 columns) gets row, column and staff
embeddings. One block is two layers: attention within each staff over all its rows and columns,
then attention across the staves at each column (the same moment in the music). The decoder is
`Seq2Seq`'s and cross-attends to every staff's grid. Blank padding staves are masked, so a batch
can mix staff counts. Input: `StaffImages(pixels (B, S, 1, 256, W), counts (B,))` and widths.
One block has as many encoder layers as `Seq2Seq`: 26.0M parameters, differing by 3,072 (embeddings).

## Step 1: shapes and masks (passed)

- The decoder and generation moved to a base class both models share. `Seq2Seq` is unchanged: its
  seeded initial weights are tensor for tensor and in key order what they were; seq516's
  checkpoint loads strictly; greedy readings of 12 OLiMPiC dev crops match the old code exactly,
  also with start tokens of two lengths. `generate` encodes once.
- `train/test_staff_model.py`, 9 tests: shapes for 2- and 3-staff systems in one batch; padding
  marks blank staves and columns past the width; a blank staff changes nothing, batched or not,
  whatever its pixels; staves do not see each other before the across-staves layer and do after;
  a later token never changes earlier logits; `generate` with start tokens; parameter count. Each
  mask broken on purpose (5 mutations): each time its test fails.
- Real size, batch 10, 1,600 px wide, 600 tokens, bf16, RTX 5080:

| Model | ms a step | Peak memory |
|---|---|---|
| Seq2Seq, 512 px system | 127 | 3.63 GB |
| StaffSeq2Seq, 2 staves | 118 | 3.63 GB |
| StaffSeq2Seq, 3 staves | 164 | 4.70 GB |
| StaffSeq2Seq, 4 staves | 211 | 5.78 GB |

Commit 707d32d.

## Step 2: splitting system crops into staves

Training data are system crops with no staff positions. On real pages the segmenter's layout
gives them (`cpp/src/layout_json.cpp`: each staff's top and bottom line), so only training crops
need them found. `train/seq/staves.py` finds them from pixels; each staff's crop is 15 interlines,
centred on it, so neighbouring staves' crops overlap in the gap between them.

How it got there, on 300 crops a training set (two staves found, of 300):

| Method | OLiMPiC | GrandStaff | PDMX | Smashcima | MusiCorpus (of 53) |
|---|---|---|---|---|---|
| Five evenly spaced ink rows, Otsu threshold, per strip | 300 | 126 | 294 | 110 | 36 |
| Five-line comb on row darkness (sum of teeth) | 292 | 221 | 300 | 300 | 20 |
| Comb scored by its weakest teeth; crop's middle pair kept | 300 | 265 | 298 | 296 | 50 |
| Same, less the darkness between teeth | 300 | 264 | 300 | 293 | 51 |
| Two weakest teeth together (committed) | 300 | 246 | 300 | 293 | 49 |

What each change was for, from looking at failures:
- GrandStaff's crops are small JPEGs (about 260 px, interline 4-7 px) whose blurred lines a
  threshold drops; Smashcima draws some lines faint or broken. The comb on row darkness copes.
- A comb two or two and a half times too wide catches lines of both staves; scoring it by its
  weakest teeth leaves those in the gaps.
- A comb narrower than a blurred line puts every tooth on that line; subtracting the darkness
  between teeth stops it.
- A unit test showed that forgiving one weak tooth let a comb one line off score as the true one
  (a staff crop centred a line off, invisible in the rates above); the two weakest teeth together
  fix it. The staff distances (median 11.6 interlines on OLiMPiC) match the strict method's.
- OLiMPiC and MusiCorpus crops often hold part of a song's voice staff: the adjacent pair whose
  middle is nearest the crop's middle is the system (crops are cut centred on it).
- Most remaining GrandStaff misses are its camera-distorted copies (28 of 35 in one sample).

Its role is a check and a fallback. For training, staff positions will come from the renderer
for the crops we render (PDMX, MuseScore, Smashcima) and from the segmenter, which inference uses,
for OLiMPiC, GrandStaff and MusiCorpus, so training and inference cut staves alike. 6 tests in
`train/test_staves.py`. Commit 2cbceb1. The by-eye check of split crops is deferred to the real
staff positions.

## The segmenter's layout on Polish dev (2026-11-12)

Is the system finding sound? The current segmenter's layouts of the 56 dev pages (from the seq516
scoring): 268 systems; 266 have the page's staff count, and every page's bars come out within one
of gold's but 106. Both exceptions are one bug: a staff found twice, about 10 px apart, in the
page's last system. On 097 the duplicate is a 1-staff system the reader skips (no harm); on 106
the song's voice staff is found twice and the two copies form a 2-staff "system" read as piano (5
bars of nonsense; a three-staff page, outside the two-staff claim). Fix in the C++ layout (merge
staves overlapping by more than half), with the real staff crops.

## Step 3: overfitting 64 systems

Scratch script (`overfit_staff.py`): 16 systems each from OLiMPiC, GrandStaff, PDMX and Smashcima,
split cleanly, labels under 1,000 tokens; no augmentation, no label smoothing; batch 8, AdamW 3e-4;
all 64 decoded greedily at each evaluation. "Exact": the whole reading equals the label; "tokens
right": share of label tokens equal at the same position (harsh: one dropped token shifts the
rest).

Round 1 (constant learning rate, each crop its own width), at 4,000 steps: the staff model 34 of 64
exact, 85% of tokens; a blank-image control **46 of 64**. The control was not blank to the model:
each system's width (its padding mask) still told the 64 apart, enough to memorise them. Not a leak
in real training (width is real information at inference, and 300,000 systems cannot be told apart
by it), but it voids the test: a model with a broken image path could pass by widths.

Round 2: every arm pads all 64 blank to one width (3,712 px for staff crops, 3,904 for whole
systems); cosine decay over 8,000 steps; the blank control's inputs asserted bitwise identical.

| Step | StaffSeq2Seq exact / tokens | Seq2Seq reference exact / tokens |
|---|---|---|
| 1,000 | 3 / 21% | 0 / 17% |
| 2,000 | 26 / 75% | 31 / 84% |
| 3,000 | 36 / 81% | 36 / 78% |
| 4,000 | 46 / 89% | 52 / 91% |
| 5,000 | 59 / 99.9% | 53 / 96% |
| 6,000 | 53 / 90% | 54 / 93% |
| 7,000 | **64 / 100%** | 60 / 99% |
| 8,000 | (stopped at 64) | 56 / 93% |

**Passed.** The staff model reproduces all 64 from their staff images at 7,000 steps; Seq2Seq,
on the same 64 whole crops, peaks at 60 and ends at 56 (the dips are greedy decoding: one early
wrong token derails a system). The blank control, with inputs asserted bitwise identical, must
write one sequence for all 64 however long it trains: after 1,000 steps, 0 of 64 exact and 1
distinct reading. Nothing but the image reaches the model.

## Step 4: blanking a staff, and the attention across staves (inconclusive on an overfit model)

The overfit staff model (rerun to save its weights: 64 of 64 again at 7,000 steps) read the 64
again four ways (`staff_ablations.py`); per staff, the share of gold notes read back (measure,
pitch and duration equal; the parser checked on all 64 labels):

| Reading | Exact | Staff 1 (upper) | Staff 2 (lower) |
|---|---|---|---|
| Intact | 64 | 100% of 1,674 | 100% of 1,458 |
| Upper staff blanked | 10 | 20.4% | 26.5% |
| Lower staff blanked | 46 | 83.2% | 74.3% |
| No attention across staves (each staff's rows attend to their own) | 64 | 100% | 100% |

The prediction (a blanked staff loses its notes, the other keeps its own) holds for a model that
reads, not for one that has memorised 64 labels: any part of the image that tells the systems
apart lets it write a whole label. With the lower staff blanked it still writes 46 systems
exactly, notes it cannot see included; with the upper staff blanked both collapse, the lower staff
it can see too (each bar's upper-staff tokens come first, and greedy decoding derails after
them). Memorising does not need the attention across staves either: the decoder reads both staves'
grids itself. What it shows: the readings depend on both staves' pixels, and the decoder works
without the across-staves layer. The test moves to a model that generalises: the step-6 race model,
on crops it was not trained on, with cross-staff passages compared to the rest.

## Building for the race (2026-11-12)

**Where training's staff positions come from.** The plan was the renderer's positions for our own
crops and the segmenter's for the rest. The segmenter, run through the recognizer on system
crops, takes about 14 CPU-seconds a crop (full recognition, scale retries): some 34 hours for the
training set on 32 cores. On 100 GrandStaff and PDMX crops it finds the system's two staves in 93,
the line finder in 90; where both do (85), their centres agree to a median 0.05 interlines, 83
within half of one. The two disagreements cut both ways: a camera-warped GrandStaff crop whose
staves the segmenter broke into pieces (its lower centre between the staves), and a PDMX crop
whose bass staff the line finder put a line high. So the line finder, about 230 times faster,
gives the positions for every crop, once (`train/find_staff_positions.py`, now in
`scripts/build_seq_data.sh`, 16 minutes); a crop it cannot split is left out. Inference keeps the
layout's positions. Split, of each set's training samples:

| Set | Split | | Set | Split |
|---|---|---|---|---|
| OLiMPiC synthetic | 14,883 of 14,907 | | MuseScore | 52,201 of 52,243 |
| GrandStaff | 25,779 of 32,156 | | unmarked tuplets | 3,981 of 3,990 |
| PDMX | 56,551 of 56,780 | | Smashcima | 13,867 of 14,189 |
| beams | 29,147 of 29,267 | | MusiCorpus | 49 of 53 |
| round | 29,134 of 29,267 | | OLiMPiC scanned dev | 1,429 of 1,438 |
| marks | 28,937 of 29,075 | | | |

GrandStaff loses a fifth, mostly its camera-distorted copies: for a fair race, `Seq2Seq` trains on
the same samples (`train_seq.py --staff-samples-only`): 313,097 in train_seq.sh's mix.

**Augmentation.** `seq.data.augment` splits into `reframe` (margins, rotation, width; it also
returns where each row went: a row moves with the top margin and scales with the height, since
the crop is scaled back to it) and `degrade` (stroke weight, binarizing, capture), drawing in the
same order: 20 seeded outputs bit for bit as before, so `Seq2Seq` training is unchanged. For
staves (`seq.staff_data`): the system scaled to a staff crop's interline, reframed, its centres
moved, degraded (both staves share paper, blur and tilt), then each staff cut 256 px around its
centre. A test cuts 20 augmented samples of a drawn system and finds each staff within an
interline of its crop's middle; with the centres left unmoved it fails. By eye, augmented OLiMPiC
staff crops: centred, the neighbour's nearest lines at the edge, lyrics kept.

**Training and inference.** `train_seq.py --staves DIR` trains `StaffSeq2Seq` on staff crops,
dev check included; samples without positions are left out and counted. A 300-step smoke run on
the full mix trained, checked and saved (21-39 samples/s with a dev check in it; `Seq2Seq` 31).
`seq_pages.py` cuts each system's staves at the layout's positions, mapped into its crop as
training cuts them, and `seq_predict.load_model` builds a staff run's model; a smoke checkpoint
read page 097 end to end. `seq_predict.py`'s own OLiMPiC crop benchmark does not read staff
models yet and says so.

**Duplicate staves.** Not fixed in the C++ layout: one tilted staff read as two chains side by
side, a little apart in height, neither inside the other's extent (its duplicate test) and too far
apart to link (half an interline). Mending the linking changes every page's staves and needs every
page benchmark again. Instead `seq_pages.without_duplicate_staves`: staves overlapping by more than
half their height cannot both be real, so the one in the system with more staves stays. On Polish
dev it changes 097 (a stray 1-staff copy goes; harmless before) and 106 (the voice staff's two
copies no longer form a system read as piano) and nothing else.

All 128 train tests and the eval tests touched pass.

## The crops the line finder could not split (2026-11-12)

**Were the left-out crops the hard ones?** 7,407 training crops (2.8%) had no staff positions,
6,377 of them GrandStaff. By eye (sheets of each set) mostly not: in 7,370 the finder found one
staff of two, the other plain to see: clean Scarlatti, crisp Smashcima, ordinary PDMX and scanned
OLiMPiC. Blurred GrandStaff copies were the hardest, and still readable. The cause: a staff was
kept only if its comb matched at half the strongest staff's, and a staff whose lines beams and
chords cover, or blur fades, matched at a median 0.27 (GrandStaff), 0.44 (PDMX), 0.46 (Smashcima)
of the other. The interline range was not the problem.

**The fix, in three parts, scored against the segmenter.** The reference: the segmenter's staves
on 300 crops, 100 the old finder missed, 100 whose positions moved by more than an interline when
the cutoff was lowered, 100 others (GrandStaff, Smashcima, PDMX, beams, round, marks). Right means
each of the system's staves within half an interline of the segmenter's. Right / wrong / none:

| Finder | Old misses | Moved | Others |
|---|---|---|---|
| Old | 0 / 0 / 100 | 34 / 66 / 0 | 93 / 7 / 0 |
| Cutoff of the system's weakest staff (half the count-th strongest, not the strongest) | 57 / 41 / 2 | 44 / 56 / 0 | 93 / 7 / 0 |
| + rows scored by long ink only (runs a third of the strip wide) | 75 / 23 / 2 | 83 / 17 / 0 | 96 / 4 / 0 |
| + ink no thicker than a staff line (thickness from vertical runs) | 74 / 24 / 2 | 89 / 11 / 0 | 97 / 3 / 0 |
| + the comb penalised for a line a line beyond the staff (tried; within noise, not kept) | 74 / 24 / 2 | 90 / 10 / 0 | 97 / 3 / 0 |

Lowering the cutoff alone brought back the staves and, as many, stacks of ledger lines and chord
clumps taken for staves: averaged darkness cannot tell a row of short thick ink from a long thin
line. Audiveris keeps only thin, long ink for its staff lines; the third and fourth rows borrow that.
The finder kept is the fourth row's.

The segmenter is not always right. Of the 37 disagreements left (with the penalty; 38 without), 17 have the two interlines about
twofold apart; by eye the finder is right in 15 (the segmenter split a warped GrandStaff staff in
two), so the true counts are about 88 / 91 / 97. Of the rest, about a dozen are warped staves a
little over half an interline apart, and about a dozen real errors (a staff a line or two off
beside chords and beams; a 40-line column of tempo marks over two empty bars, PDMX 016467/s2).

The old positions were also about 7% wrong on the crops the old finder split (the "others").

**Found on the way: augmentation erases staff lines.** `degrade`'s stroke thinning (seq.data,
since the first Seq2Seq) dilates the white by `interline * 2 * 0.06` px each way: a 5x5 ellipse at
staff scale, wider than a 2 px staff line. Where its random field leans thin, lines, stems and
ledger lines go. On 60 PDMX crops x 10 seeds, the thinning keeps under 30% of the staff lines'
contrast in 18.5% of crops (applied to half: about 9% of training crops); blur and the rest of
capture's degradation, 1.2%. test_staff_data's augmented-crop test now fails on seed 5 for this
reason (a crop with no staff left, for which the finder rightly finds none). Not yet changed.

## A small end-to-end test on PDMX, staff positions as drawn (2026-11-12)

**Why.** The question is whether reading staff by staff beats reading the system, not how good a
line finder can be. On our own renders the staff positions need not be found: the renderer drew
them. So a first, small test on rendered PDMX only, before any race.

**Positions as drawn.** `build_seq_dataset.py` now rasterizes each system's staff lines alone (the
staff groups' own paths, with the SVG's style and defs) and writes `<sample>.staves.json`
(`[centre, ..., interline]`, in the crop's rows); `find_staff_positions.py` uses that file where it
exists. On 40 scores 155 of 156 crops came out byte-identical to the old build.

**A bug found on the way: crops of several systems.** The one crop dropped, PDMX 000001/s0, was a
system Verovio broke over several lines: a 6870 x 4742 crop of many staves, labelled as one
two-staff system. The builder now drops a crop whose drawn staves outnumber its label's. In
every earlier PDMX set such crops went to training (for Seq2Seq too) with a label for only part of
the image; not yet counted how many.

**The data.** `data/staff_exact/pdmx`, rebuilt from scratch (11 minutes): 56,727 crops, all with
drawn positions, all two-staff. Dev: 1,135 crops of 293 scores (the first 2% of scores after
shuffling with seed 0); race: the other 55,592; 59,596 training samples a pass, with
rhythm-heavy crops repeated.

**How much a staff crop misses.** Each staff crop is 15 interlines tall, about 5.5 interlines of
room beyond the staff's outer lines. On 3,000 crops, with drawn positions, ink lies beyond the
outer staff crops in 18.9% (beyond 1 interline 12.7%, 2: 7.5%, 4: 2.4%), and in the gap between
staves no crop covers in 2.2%. By eye, mostly not music the label holds (tempo-mark columns, chord
symbols, lyrics, fingering); 2 of 16 flagged crops had notes cut, so about 1-2% of crops lose
notes. A handicap for StaffSeq2Seq, small; taller staff crops would remove it.

**A bug found on the way: augmentation's margins.** `reframe` draws margins down to -8% of the
height: it may crop into the system, cutting notes the label keeps (Seq2Seq too, since the
first). For StaffSeq2Seq it can put a staff's centre beyond the crop, and `staff_crops` then cut a
short crop (235 rows): the first run crashed at step ~1,000. `staff_crops` now pads white enough
for any centre; a test with a centre beyond the image fails on the old code. Cutting notes is
not yet changed (nor the stroke thinning above): both arms see it alike.

**The runs.** Both on the local GPU, one after the other, same samples, seed and schedule: batch
10, lr 3e-4, warmup 800, 8,000 steps (about 1.3 passes), dev SER on 500 dev crops every 2,000
(averaged weights). `StaffSeq2Seq` with `--staves`, `Seq2Seq` with `--staff-samples-only` at
height 512 (`runs/staff_e2e2_staff`, `runs/staff_e2e2_system`).

| Step | StaffSeq2Seq loss | dev SER | Seq2Seq loss | dev SER |
|---|---|---|---|---|
| 2,000 | 2.71 | 1.000 | 2.70 | 4.578 |
| 4,000 | 2.29 | 3.521 | 2.31 | 4.514 |
| 6,000 | 2.19 | 1.967 | 2.24 | 2.975 |
| 8,000 | 1.82 | 1.186 | 1.93 | 1.698 |

Neither reads yet: an SER above 1 is output longer than the label, mostly runaway repetition. So
did Seq2Seq trained from scratch before (seq512-516: 3.65-3.93 at 5,000 steps, 0.6-1.8 at 10,000,
0.3-0.4 at 15,000, with other data and batches): 8,000 steps ends before it ever read well. The
averaged weights also lag (EMA decay 0.9995, about 2,000 steps; at 2,000 they still hold a third of
the initial weights). Training losses alike throughout; StaffSeq2Seq's dev SER lower at every check.
Not scored on Polish dev: models this far from reading would say nothing there.

**Is the staff model broken?** Its 8,000-step predictions on 64 dev crops (SER 1.39 there), by eye
beside their labels: the header right (clefs, mostly the key; the meter often wrong), the
two-staff structure and voices right, some right notes and chords, then a measure repeated
until the token limit (12 of 64 ran to 1,023 tokens) or rests where notes are: what an untrained
Seq2Seq does. The staff crops as the model receives them, dev and augmented train: centred on
their staves, at the right scale, the staves in order, the neighbour staff's nearer half
visible as intended. Nothing points to a fault in the staff path; one dev crop (000129/s0) loses
chords high above its upper staff (the margin above). So: train longer, below.

Throughput: the samples/s printed is the average since the start, dev checks included, and a
check of a model that repeats itself decodes 500 crops to the token limit (about 10 minutes):
the apparent halving after the first check is that, not a slowdown.

**Longer: 25,000 steps an arm.** As above but `--steps 25000 --eval-every 5000`
(`runs/staff_e2e3_staff`, `runs/staff_e2e3_system`); the schedule's cosine is stretched to the
new length, so the 8,000-step runs are not a prefix of these.

| Step | StaffSeq2Seq loss | dev SER | Seq2Seq loss | dev SER |
|---|---|---|---|---|
| 5,000 | 1.81 | 2.260 | 1.94 | 3.225 |
| 10,000 | 1.06 | 0.391 | 1.25 | 0.845 |
| 15,000 | 1.00 | 0.137 | 1.10 | 0.263 |
| 20,000 | 1.01 | 0.094 | 1.03 | 0.132 |
| 25,000 | 0.93 | 0.080 | 0.97 | 0.118 |

**On Polish dev.** Both 25,000-step snapshots (averaged weights), read through the segmenter
as `benchmark_pages.sh` does (`seq_pages.py`; StaffSeq2Seq cuts staves at the layout's positions,
not drawn ones). OMR-NED, lower is better, the 49 two-staff pages:

| Gold, alignment | StaffSeq2Seq | Seq2Seq | seq516 (full mix, 150k steps) |
|---|---|---|---|
| corrected, authors' | **31.71** | 45.16 | 10.84 |
| corrected, whole page | **30.73** | 43.76 | |
| print, authors' | **28.84** | 42.95 | 6.97 |
| print, whole page | **27.92** | 41.86 | |
| published, authors' | **53.91** | 63.32 | |

The staff model's lead holds on real prints, and grows: 30-33% lower than Seq2Seq's, against
32% fewer errors on PDMX dev. Trained on drawn positions, it reads at the segmenter's. Three-staff
pages: 90.9 against 91.2 (neither has seen three staves; seq516 90.3). Both are far from seq516,
which trained on the whole mix for six times as long: this says which reader learns faster from
the same PDMX renders, not what either would reach. One seed each.

**The line finder against drawn positions.** On all 56,727 PDMX crops of
`data/staff_exact/pdmx`, the finder (`seq.staves.system_staves`) beside the positions drawn:
56,688 right (99.93%: each staff within half an interline), 14 wrong (11 by more than an
interline, 5 by more than five), 25 not found. Against the segmenter it had looked 97% right
on ordinary crops: most of that gap was the segmenter's. PDMX is the easiest set (clean
renders); GrandStaff's camera-warped copies and the scans are where it is still unmeasured.

**Drawn positions from MuseScore and Smashcima too.** `build_musescore_crops.py` takes each
staff's middle line from the SVG's StaffLines (whose units are the PNG's pixels) and
`build_smashcima_crops.py` from each StaffVisual's coordinate system, its height averaged along
the staff (a hand's lines slope); both follow the crop through its cut, resize and margins
(`build_seq_dataset.row_resized`: cv2.resize scales pixel centres) and write
`<sample>.staves.json` as the Verovio builder does (`write_sample`). `Staves` and `staff_count`
moved to `seq.staff_positions`, free of torch, which Smashcima's environment lacks. On 40
scores each (136 MuseScore crops, 35 Smashcima): every crop byte-identical to
`data/seq516f_data/seq2`'s, and the finder within 0.06 interlines of every drawn staff
(mean offset under 0.01 interlines, interlines within 4%). A test puts 2-px staff lines on a
page and finds each recorded centre on its middle line after the crop (atol 0.6 px); dropping
the half-pixel of `row_resized`, or a 3-row shift, fails it.

## Augmentation fixed: it no longer erases staff lines or cuts notes (2026-11-13)

Both bugs found above, in `seq.data` and so in both arms and in every Seq2Seq since the first.
Every measurement here is on crops at staff scale (an interline of 17 px, as Seq2Seq at 512).

**Stroke thinning.** At this scale PDMX's staff lines are 1-3 px (of 60 crops, 11 at 1 px, 43
at 2, 6 at 3), and the thinning kernel's radius is 2 px: it can only erase them. On 60 PDMX
crops x 10 seeds, `stroke_weight` alone left under 30% of the lines' contrast in 18.5% of
crops. Capping the thinning at half strength fixed that (0%), but through all of `degrade` it
still left gray lines for the binarization to whiten (test_staff_data's seed 5: every line
gone). A kernel limited to the crop's line thickness thins almost nothing, measuring the same
as no thinning at all. So seq crops are now thickened, never thinned (`stroke_weight(thin=False)`;
the segmenter's augmentation is unchanged). Through all of `degrade`, crops left with under 30%
of their line contrast: 12.3% before, 7.7% now. The rest is capture's ink tone, paper and blur,
which fade a 1-px line on its own as a soft scan does (blur alone: 1.2% of crops under 15%),
not erasure.

**Margins.** `reframe` draws each margin from -8% to +12% of the height; a negative one now cuts
only blank rows (`blank_margins`: rows with nothing darker than mid-grey). Draws that cut ink
before, over 300 training crops x 10 draws a set: PDMX 6.2% (by over an interline 0.9%),
MuseScore 5.4% (0.8%), Smashcima 6.0% (1.1%), GrandStaff 5.8% (1.1%), and OLiMPiC 36.3% (11.9%),
whose crops are cut tight to the ink. Random draws are unchanged; only the depth of a cut is
limited.

Tests (`test_seq_data.py`): a crop cut tight to its outer staff lines keeps them through 200
draws, and unthinned stroke weight never lightens a pixel; both fail on the old code.
test_staff_data's seed 5 passes again.

## Staff positions where none are drawn: GrandStaff and MusiCorpus (2026-11-13)

**A check that needs no truth.** In each of 8 vertical strips, the comb (seq.staves) at each
staff's centre, best within half an interline, against the best a line up or down; a staff is
off where the shifted comb wins in most strips. On 400 PDMX crops it flags none of the drawn
positions and all 400 of the same shifted by an interline. But it shares the finder's notion of a
line: on GrandStaff it passed a finder staff placed on a stack of ledger lines. Used to sample
by, not as truth; the verdicts below are by eye.

**MusiCorpus (53 training crops, handwritten scans).** 1 flagged, by eye a bass staff a line low;
the other 52 sit on their lines. Its staff boxes (`subdivisions.image.json`) include the notes
around a staff, so they give no exact positions either. Finder kept.

**GrandStaff (32,156 training crops: 16,130 clean renders, 16,026 distorted).** A first pass
said the finder was wrong on about 5% of clean crops and 20% of distorted, but it read
`seq516f_data/staff_positions_v2`, written on 2026-11-12 by an earlier finder. Today's finder
(4e1f73e) put every strip of the examples looked at on the right lines; rerun on the whole set,
the check flags 0.25% of clean crops and 11.6% of distorted. The segmenter (`runs/finetune`, the
recognizer's layout) ran on 600 crops, 150 each of clean/distorted x flagged/not by the stale
positions; on the 516 where it found exactly two staves, the finder agreed on 435 (a staff within half an
interline, interlines within 10%); on the clean crops the old positions had flagged, 116 of 141,
from 2.

**What the finder still got wrong: merged ledger lines.** By eye on flagged distorted crops, a
staff a line high, its top tooth on the ledger lines of a beamed group. Blur merges neighbouring
notes' ledger lines into runs over a third of a strip, long enough to pass as staff-line ink, and
where the staff's own bottom line is faint (blur, beams over it) the comb shifted up matches
better. Two other suspects measured and found minor: the profile's neighbourhood mean (beams and
chords inside a staff raise it, so its lines nearby fade; without it, 446 agreeing, no change in
flags) and the line thickness estimate on blurred copies (7-9 px; capped at 3, no better).

**Three changes to seq.staves**, each measured on 1,500 clean and 1,500 distorted GrandStaff
crops, the segmenter's 516, and 3,171 crops with drawn positions (3,000 PDMX, the Smashcima and
MuseScore test builds), then on every set the finder serves:

1. Staff-line ink must run across the whole strip, not a third: segmenter agreement 435 -> 459;
   90% of the strip did as well (457), 60% less so (453). Drawn crops all right throughout.
2. But on OLiMPiC's scans, thousands of pixels wide, a strip is 340-640 px and a slight skew
   moves its lines more than the row of tilt allowed: 24 of 1,438 dev scans no longer found.
   Allowing tilt in proportion to the strip's width broke drawn PDMX crops (9 of 3,171 not
   found); looking at crops taller than 320 px shrunk to 320 lost none of the scans, and kept
   every drawn crop (at 256, 5 thin-lined PDMX crops were not found).
3. Shrinking doubled the interline's error on drawn PDMX (median 1.8% -> 2.4%, p99 4.1% -> 6.3%),
   which showed a bias there all along: lines a few rows thick make the comb's best match flat
   over a range of spacings (9.5-10.5 for a true 10), and argmax took the narrowest. Now the
   middle of the spacings matching within 3% of the best: median 0.6%, p99 2.7%; agreement 463.
   Centres, found to a row of the shrunk crop, are 0.05 interlines off at the median (p99 0.14),
   from 0.02: nothing at a staff crop 15 interlines tall.

By eye, where old and new disagree: 8 of 8 GrandStaff crops right with the new (5 of them, both
clean ones among them, the old on a ledger stack); OLiMPiC written, the 2 crops the new finds no
staves in had wrong old positions, the moved ones look right; the scans that moved mostly sit
better. Against the same finder with third-of-strip ink: GrandStaff 1,207 crops moved, OLiMPiC written 1, scans 9,
MusiCorpus 0, none lost. The check still flags 6.9% of distorted GrandStaff crops (103 of 1,500,
with the new finder's own profile; each finder's check favours it, so flags only compared like
for like): by eye mostly tilted or curved copies, which one straight centre per staff cannot fit
in every strip, the positions right in the middle; a staff crop of 15 interlines holds them.
Positions for the rebuilt sets: `seq516g_data/staff_positions_v2` (the build's own
`staff_positions` ran during these edits, with change 1 alone: 24 scans missing).

The segmenter's slowness here (600 crops in about 20 minutes beside the rebuild) is the shipped
CPU pipeline: a full-resolution network pass per crop (1.2-1.4 s on 2 threads), up to 8 more at
other scales when fewer than two staves are found at the first, and parallel workers limited by
memory bandwidth.

## The race, staff arm: 50,000 steps, and what it gets wrong (2026-11-13)

`runs/race_staff`: StaffSeq2Seq on the full `train_seq.sh` mix of `data/seq516g_data`, positions
from `staff_positions_v2`, batch 10, lr 3e-4, warmup 4,800, 50,000 steps, seed 0, local GPU
(about 4.5 h). The Seq2Seq arm (`runs/race_system`, `--staff-samples-only`, `--height 512`, same
samples and schedule) followed it on the same GPU.

| Step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
|---|---|---|---|---|---|---|---|---|---|---|
| OLiMPiC scanned dev SER | 3.914 | 1.203 | 0.261 | 0.185 | 0.145 | 0.127 | 0.117 | 0.100 | 0.100 | 0.098 |

**Polish dev** (50,000-step averaged weights, through the segmenter as before; OMR-NED, the 49
two-staff pages): print gold 11.58 (whole-page 10.82), corrected 15.47 (15.09), published 43.71.
The 25,000-step PDMX-only snapshot was at 28.84 print, seq516 (150,000 steps) at 6.97. The staff model is
ahead of its 25k self on all 49 pages, ahead of seq516 on 3 and behind it on 42; 25 pages are under 10,
none over 60; the worst 10 pages hold 48% of the summed OMR-NED. Three-staff pages 91.2 (never trained on).

**Polish: what it misses.**
- *Note fates* (`eval/note_errors.py`, print gold, 49 pages): 13,695 right; missing 902, extra 780,
  other pitch 472, duration 246, onset 244, octave 116. seq516: 14,667 right; 430, 349, 236, 128,
  134, 80. The staff model makes about twice seq516's errors of every kind. OMR-NED charges
  most for inserted and deleted notes (75% of its cost, as for seq516).
- *Unprinted tuplets are the largest single cause.* A run of triplets or sextuplets shows its
  number for the first group or two only; after that the staff model writes plain eighths
  (page 003: measure 1 read as triplets, measure 2 as eighths), and every onset after them
  shifts, so each note counts as missing and extra at once. Of 1,373 notes in tuplets in the gold
  it reads 1,040 as tuplets, seq516 1,272. Seven pages where it loses them (003, 056, 081, 087,
  088, 096, 110) hold 41% of its gap to seq516; page 003 alone is 29.5 against 1.4. Without
  those pages: 9.2 against 6.0.
- OMR-NED's tuplet edits agree: tuplets edited 182 times, inserted 140 (seq516: 45, 42). Beam,
  accidental and tie edits are near seq516's (beam 265 vs 255, accidental deleted 120 vs 116).

**OLiMPiC dev: what it misses** (615 crops, read as the dev check reads them, SER 0.0982;
token edits by Levenshtein alignment, `scratchpad dev_patterns.py`):
- By kind of token, share of all edits (and per gold token of that kind): durations 18% (6.7%),
  tuplet tokens 17% (43%), pitches 12% (5.3%), slurs 7% (27%), arpeggios 5% (33%), ties 4% (26%),
  staff tokens 4% (8.5%), rests 3.6% (14%). Most are missed tokens, not substituted ones.
  Notes and rhythm alone (pitches, durations, rests, chords, dots, accidentals, staff, measure,
  backup/forward): SER 6.5%.
- Ties and slurs are confused both ways (tied->slur 173, slur->tied 150), and one tuplet ratio
  is written as another that means the same thing (3in2 -> 6in4, 209). Durations go wrong one
  value at a time (quarter->half 142, quarter->eighth 66).
- Pitch substitutions: an octave 133, more than an octave 98, a step 90, two steps 85.
- *Notes far from the staff* (`scratchpad edge_checks.py`; a note's staff carried from the last
  `staff:N`, as LMX writes it only on a change). Pitch tokens on the staff are wrong 3.4% of the
  time. Between the staves it stays about 3% at any distance (engravers move such notes to the
  nearer staff, so none is more than three ledger lines out). Outside the system (above the
  treble, below the bass) it climbs: 4% one step out, 6-10% at three to six, 15% and more past
  eight, with no step where the crop ends (eleven steps). Much of the climb is ottavas:
  OLiMPiC's labels are at sounding pitch, and in the 24 crops with an octave shift, notes three
  or more ledger lines outside are wrong 63% of the time (12% elsewhere). By eye (5989993
  p2-s3): the *8va bassa* below the bass staff is cut off by its staff crop's lower edge, only
  the top of the 8 left, and the lowest notehead clipped; the system crop shows both. Without
  the octave-shift crops the climb is milder (4% near, 6-10% at three to seven steps).
- *Error along the crop is mostly a mix of crops.* By tenth of the crop, header tokens left
  out: long crops (350+ tokens) 7% in the first tenth, then 10-12%; medium 5.2% rising to 8.2%;
  short flat at 5-7%. At the same number of tokens decoded (100-150), long crops are at 12%,
  medium 6.6%, short 5.4%: long crops are denser music, not later in a drifting decode. The
  earlier 5.7%-to-12.5% rise mixed the easy header into the first tenth and long crops into the
  rest. Staff drift across the finder's strips does not predict late errors (rank correlation
  -0.01; 0-0.25 interlines of drift: 7.5% in the last 30%, over 1 interline: 7.6%).
- Bass staff slightly worse than treble (10.7% vs 9.2% edits per gold token; this split takes a
  token's staff from the next `staff:N`, so it is approximate). Measures are dropped in 17 crops and added in 3, of 615.
- Error is concentrated by song: three of 46 songs hold 35% of all edits; without them SER is
  7.7%. By eye: 6801790, sextuplets of 32nds numbered once, read without the tuplet (895 tuplet
  tokens missed); 6581327, two-note half-note tremolos read as beamed 32nds; 6011095, ten-note
  arpeggiated chords several ledger lines below the bass staff, notes dropped from them.
- Things it rarely gets: hidden objects (`print-object:no`, 94% wrong: nothing to see), strong
  accents (65%), measured tremolos (62%), `forward` (71%, invisible rests).

Next, when the Seq2Seq arm finishes: the same analysis of it, so that what belongs to staff
crops and what to this mix and length can be told apart (seq516 had 150,000 steps; the
tuplet loss may simply be training length). If unprinted tuplets stay the staff model's own
weakness, one hypothesis is that a staff crop leaves out the number printed on the other staff
or far above it; that is a guess until the arms are compared.

## Ties and slurs: the labels, the system's edges, and what the staff model misses (2026-11-13)

**Why.** Tie against slur looks like a simple rule (adjacent notes of the same pitches: tie;
otherwise slur), yet ties and slurs were 11% of the staff arm's OLiMPiC dev edits. Three
questions: do the labels follow the rule, how many arcs are open at a system's edge (a half arc
the crop shows only in part), and what the model gets wrong.

**How.** `scratchpad arcs.py`: each start paired with its stop in the same voice, then leftovers
across voices (a tie into another voice, a slur from hand to hand; ties by pitch, slurs in
order); what is still unpaired is open at an edge. A slur is *tie-like* if its two events are
adjacent in the voice and have the same pitches; a tie is *odd* if its events are not adjacent.

**The labels follow the rule.**

| Set | Crops | Tie-like slurs (of slurs) | Odd ties (of ties) | Arcs open at an edge |
|---|---|---|---|---|
| PDMX (Verovio) | 56,727 | 3.9% | 0.5% | 10.2% |
| MuseScore | 52,243 | 3.9% | 0.4% | 0.0% |
| Smashcima | 14,189 | 3.9% | 0.4% | 10.8% |
| OLiMPiC synthetic, train | 14,907 | 2.1% | 0.9% | 10.9% |
| GrandStaff | 32,156 | (no slurs) | 0.1% | 0.0% |
| MusiCorpus | 53 | 0.7% | 4.8% | 10.2% |
| OLiMPiC dev (written labels) | 1,422 | 2.9% | 3.6% | 9.6% |

Relabelling slurs as ties would change little; label noise is not why the model errs.

**Open arcs, set by set.**
- *PDMX (Verovio):* labels keep them, and the images draw the half arcs (the builder draws
  them from the neighbouring measures, `with_neighbours`). Checked by eye on two crops
  (000086/s2, 000087/s2).
- *MuseScore:* the builder leaves out systems with an open arc (`open_at_an_edge`), since
  MuseScore engraving one system alone draws none.
- *GrandStaff: open ties are drawn but not labelled.* Its labels keep only ties that close in
  the crop: on 3,000 crops, 3,642 tie starts in the kern sources against 3,252 in the labels
  (and stops equal to starts), so about 11% of its ties are missing. Its images draw their half
  arcs: by eye, beethoven sonata12-1 maj3_down_m-212-216 has ties running off the right edge in
  both hands and no tie token. 32,156 training crops teach that a half arc at the edge means
  nothing, against PDMX's, which label them. GrandStaff has no slurs in kern, labels or images
  (consistent; it teaches nothing about slurs).
- *Smashcima:* labels keep them (10.8%); not checked by eye whether the fork draws them.
- `TODO.md`'s note (2026-11-09) that neither engraver draws a half arc is out of date for
  Verovio and MuseScore.

**What the staff arm gets wrong** (OLiMPiC scanned dev, 615 crops, 50,000-step snapshot).
Gold arc tokens read wrong (substituted or missed): ties 18.4% of 2,454, slurs 16.6% of 3,786,
tie-like slurs 18.9% of 106; arcs open at an edge 30.4% of 369. Edge arcs are about 6% of arc
tokens and about a tenth of arc errors. Mid-system, most errors are arcs not written at all:

| Gold | Right | Not written | Read as the other | Other |
|---|---|---|---|---|
| Tie (2,454 tokens) | 81.6% | 9.9% | 6.7% | 1.8% |
| Slur (3,892 tokens) | 83.3% | 11.4% | 3.7% | 1.6% |

So the confusion between ties and slurs is the smaller part. Per arc (both ends in the
system), the missing tokens are more often one end than both:

| Gold | Read whole | Both ends missed | Start only missed | Stop only missed |
|---|---|---|---|---|
| Tie (1,280 arcs) | 85.9% | 5.5% | 5.2% | 3.4% |
| Slur (1,893 arcs) | 80.5% | 3.4% | 6.2% | 9.9% |

A slur is rarely overlooked altogether (3.4%); far more often one end is put on another note
(a stop missed here, one written elsewhere: 381 extra slur tokens). Where a slur starts and ends
is the slur problem, not whether it is there.

What the wholly missed arcs are. The six crops with the most (one per song) show stacks of ties
between chords (6816059 p2-s4, 6652787 p8-s3, the latter across a double barline and a change
of meter), long slurs over several bars in the bass (6505088 p4-s4) and slurs weaving between
two voices of sixteenths (6982743 p2-s1). But per arc, chords and barlines are not where it
fails: ties in a chord are wholly missed 4.7% of the time, on a single note 8.1%; ties across a
barline 2.9%, within a bar 6.9%; slurs over three bars or more 11.8% (of 17), otherwise 3-4%.
Crops with many chords simply have many arcs to miss. No single kind of arc explains it.

**Fixes made (2026-11-13).**
- *GrandStaff's open ties.* They are lost in OLiMPiC's build itself: music21 reads the kern's
  `[` and writes `<tie type="start">`, and MuseScore 3.6.2's normalisation drops a tie with no
  note to end on. Rather than put tokens back into the LMX by matching kern notes to it,
  `build_grandstaff_written.py` now leaves such systems out, as `build_musescore_crops.py`
  leaves out its own (`open_at_an_edge`). `seq.kern` counts `ties_at_edges` (ties paired by
  pitch within a staff, `_` both ending one and starting the next; five tests in
  `test_rebuild_grandstaff_lmx.py`). Of GrandStaff-LMX's 41,598 training systems, 3,477 have
  one; after the other filters 3,213 are left out and 29,500 kept (was 32,156). On 3,000 kept
  crops, tie starts in kern and labels now agree (2,818 against 2,811, from 3,642 against
  3,252). Built into `data/seq516g_data/grandstaff_written_v2`, positions in
  `staff_positions_v2/grandstaff_written_v2.json`; for the next runs, not the race's.
- *Smashcima's open arcs,* by eye on six crops (three ties into the next system, three slurs
  from the last): each draws an arc to or from the edge (000777/s0's reads more like a whole
  slur from the first note). Labels and images agree.


## The race, Seq2Seq arm, and the two arms' errors side by side (2026-11-14)

`runs/race_system`: Seq2Seq at height 512 on the samples with staff positions
(`--staff-samples-only`), the staff arm's data, seed and schedule, 50,000 steps, local GPU (about
5.5 h). Checkpoints verified (`verify_ckpt.py`: both arms' averaged, best and resumable last).

| Step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
|---|---|---|---|---|---|---|---|---|---|---|
| Staff arm, OLiMPiC dev SER | 3.914 | 1.203 | 0.261 | 0.185 | 0.145 | 0.127 | 0.117 | 0.100 | 0.100 | **0.098** |
| Seq2Seq arm | | | 0.314 | 0.188 | 0.143 | | | 0.111 | 0.111 | 0.110 |

**Polish dev**, 49 two-staff pages, OMR-NED: print gold 12.28 (whole-page 10.90) against the staff
arm's 11.58 (10.82); corrected 17.08 against 15.47; published 44.51 against 43.71; seq516 6.97
(print). Page by page the staff arm is better on 17 pages, worse on 20 (by more than 1). On Polish
the two designs are level; on OLiMPiC dev the staff arm leads by about 10%.

**Note fates on Polish** (`eval/note_errors.py`, print gold):

| | Right | Missing | Extra | Other pitch | Duration | Onset | Octave |
|---|---|---|---|---|---|---|---|
| Staff arm | 13,695 | 902 | 780 | 472 | 246 | 244 | 116 |
| Seq2Seq arm | 13,763 | 837 | 779 | 515 | 251 | 196 | 113 |
| seq516 (150,000 steps) | 14,667 | 430 | 349 | 236 | 128 | 134 | 80 |

**The same errors, mostly.**
- *Unprinted tuplets are not the staff design's.* Of Polish dev's 1,373 notes in tuplets the
  staff arm reads 1,040 as tuplets, Seq2Seq 1,080, seq516 1,272. On OLiMPiC dev tuplet tokens are
  wrong 43% of the time for the staff arm, 41% for Seq2Seq. Both arms lose them alike at 50,000
  steps: training length or data, not reading staff by staff.
- *Notes beyond the system: the crop is a part, not the cause.* Seq2Seq's crop reaches 7
  interlines beyond the staves (the staff crop 5.5), yet far-outside notes (three ledger lines or
  more) are wrong 30% of the time against 33%. Under ottava signs 51% against 63%: seeing the sign
  helps, but half are still wrong. Elsewhere far notes are worse for Seq2Seq (16% against 12%).
  E1 should help ottava passages somewhat; it will not fix them.
- *Ties and slurs alike:* tie tokens wrong 18.9% against 18.4%, slurs 15.4% against 16.6%, arcs at
  an edge 27.1% against 30.4%; the same tie/slur swaps.
- *The same worst songs:* 6581327 (measured tremolos; Seq2Seq worse, 48.7% against 28.5%),
  6011095 (big arpeggiated chords, 17.0% against 16.7%), 6801790 (sextuplets numbered once, 20.5%
  against 20.8%), 6499366 (49.8% against 40.2%).
- *Polish OMR-NED edits* (`allops6`, print gold): inserted and deleted notes 10,841 against the
  staff arm's 10,354 cost units; beams 282 against 265; tuplets inserted 142 against 140.

**Where they differ.**
- *Pitch by a step:* Seq2Seq reads a pitch a step off 286 times on OLiMPiC dev, the staff arm 90;
  pitch tokens on the staff wrong 5.0% against 3.4%. A staff crop gives each staff its own 256 px:
  the same pixels per interline as a system at 512, but the CNN and attention see one staff at a
  time. On Polish the gap is smaller (other pitch 515 against 472).
- *The bass staff:* Seq2Seq 12.8% of gold tokens against 9.6% on the treble; the staff arm 10.7%
  against 9.2%.
- *Hidden objects:* Seq2Seq writes 326 `print-object:no` (hidden rests) where the label has none;
  the staff arm 23.
- *Late in long crops:* Seq2Seq's error in the last tenth of a crop is 18.3% (the staff arm's 12.5%);
  by tokens decoded in long crops, 16.4% against 14.1% after 350 tokens.
- *Tuplet ratios:* the staff arm writes `6in4` for `3in2` 209 times, Seq2Seq 59 (the same rhythm;
  Polish's tuplet edits 182 against 71). Seq2Seq has more `2in1` read as 32nds (67 against 31,
  measured tremolos).

**What it means.** At 50,000 steps on this mix, reading staff by staff is about as good as reading
the system: a little better on OLiMPiC (pitches, the bass staff, long crops), level on Polish. The
large gap to seq516, about twice its errors of every kind for both arms, belongs to training
length (T7 tests it) or data, not to the design. Unprinted tuplets are the clearest case.

## Why the staff arm leads on OLiMPiC: two hypotheses tested (2026-11-14)

The staff arm's lead over Seq2Seq at 50,000 steps is on OLiMPiC dev (SER 0.098 against 0.110),
not on Polish dev (11.58 against 12.28 print, 17 pages better, 20 worse). It comes from pitches a
step off (90 against 286), the bass staff (10.7% against 12.8% of gold tokens), hidden rests
written where none are (23 against 326) and the end of long crops (12.5% against 18.3% in the
last tenth).

**Hypothesis 1: a fixed frame for pitch.** A staff crop is centred on its staff, so each feature
row always stands at the same height relative to the staff's lines, and the row embedding can
carry the line position; in a system crop the staves sit at heights that vary with margins and
the gap between them (3 to 11 interlines), so Seq2Seq must find the lines first. The bass staff,
whose height depends on the gap above it, varies most. *Test* (`runs/overnight_2026-11-14/
frame_test.py`): on the 615 OLiMPiC dev crops, each staff's centre as a share of the crop's
height (line finder positions), in fifths per staff, against the share of that staff's pitch
tokens read a step off.

| Bass staff, by its height in the crop (fifths) | 1st (highest) | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Seq2Seq: a step off | 2.4% | 1.4% | 0.8% | 0.7% | 1.0% |
| Seq2Seq: wrong at all | 9.9% | 8.3% | 6.2% | 5.6% | 5.7% |
| Staff arm: a step off | 0.3% | 0.4% | 0.1% | 0.1% | 0.4% |
| Staff arm: wrong at all | 8.3% | 5.6% | 3.9% | 4.7% | 4.9% |

Seq2Seq's step errors on the bass staff move with its height, the staff arm's do not: as the
hypothesis predicts. *Weak*: OLiMPiC's crops are cut alike, so the fifths span little (centres at
0.32 and 0.67 of the height, the outer fifths holding the few that differ), and crops whose staves
sit unusually may differ in other ways (denser music); both arms' "wrong at all" also rise in the
first fifth. The treble staff shows little (Seq2Seq 0.5%, 0.1%, 0.4%, 0.4%, 1.0%).

**Hypothesis 2: on Polish the staff arm is held back by the segmenter's staff positions** (a crop
centred a line off loses the frame; Seq2Seq does not care). *Test* (`positions_test.py`): the 239
system crops the staff arm read on the 49 two-staff pages, rebuilt from the layouts tonight's
scoring saved, the segmenter's staff centres against the line finder's on the same crops.
**Refuted**: the largest disagreement on any page is 0.34 interlines (median page 0.14); no crop
is off by half an interline; the finder finds every system. Two independent methods agree, so the
positions are right, and the lead's absence on Polish needs another cause. A guess, untested: the
staff arm's advantage is partly one on familiar layouts (OLiMPiC dev is the same source as
OLiMPiC's synthetic training crops, at weight 2 in the mix); Polish's older prints are not.

**Ways to give a system model the staff model's properties without cutting staves** (ideas, not
run; all keep the whole system crop, so nothing beyond the staves is lost):
1. *Relative vertical positions:* in Seq2Seq's encoder, replace the absolute row embedding with a
   learned attention bias by vertical distance, so heights are read relative to the lines
   wherever the staff sits. Needs nothing at inference; the cheapest and most direct test of
   hypothesis 1.
2. *Staff-relative position embedding:* each feature row embeds its height relative to the
   nearest staff's centre, in interlines, and which staff (needs staff positions).
3. *Straightened crops:* stretch the system crop vertically so each staff's centre lands on a
   fixed row at a fixed interline (piecewise linear; varying along the width would also take out
   a slight warp). Preprocessing only; needs staff positions.
4. *Staff lines as an auxiliary task:* a training-only head predicts the staff lines' positions
   from the encoder's features (labels free from the drawn positions), pushing the encoder to
   localise lines.
5. *Staff lines as input:* a second input channel with the line finder's staff-line ink.
6. *The staff model's attention pattern:* factorise Seq2Seq's encoder attention into within a
   band of rows about a staff tall, then across bands at each column.
Order proposed: 1, then 3 if 1 does not help; the night's T2, T3 and T5 bear on 6.

## The outer-notes set: the staff model writes no ottava at all (2026-11-14)

Phase A's A1 and A2: both race arms (50,000 steps) on `eval/outer_notes.py` (OLiMPiC dev's 1,422
systems, synthetic and scanned, labels with pitches as printed and ottava tokens) and
`eval/tuplet_bars.py`. The baselines for T1 and the tuplet runs.

| | Staff arm | Seq2Seq arm |
|---|---|---|
| SER, all 2,844 crops | 0.089 | 0.097 |
| Notes under a sign, wrong (948) | **100.0%** | 40.2% |
| Notes far outside the system, wrong (3,606) | 10.6% | 13.0% |
| All other notes, wrong (158,426) | 3.0% | 3.6% |
| Ottava tokens written where no sign is | 0 | 298 |
| Ottava tokens written at all | **0** | 1,115 (labels: 948) |
| Bars with tuplets, numbers shown, right rhythm (1,180) | 35.2% | 28.2% |
| Bars with a hidden number, right rhythm (340) | 59.7% | 52.4% |
| Bars without tuplets, right rhythm (9,308) | 74.9% | 75.0% |
| Tuplet-ratio tokens read (23,420) | 52.6% | 47.6% |

**The staff arm writes no ottava token in any of the 2,844 crops.** Not a pipeline fault: the
vocabulary has them, training labels have them (1,264 of the first 3,000 `marks` crops), and
Seq2Seq, trained on the same samples, writes them. Fed the labels (teacher forcing), on 12 `marks`
training crops with 195 `ottava:8va` tokens the staff arm gives the token a median probability of
0.90 where it belongs and ranks it first 91% of the time (Seq2Seq 0.90, 95%): it reads signs it can
see. By eye on OLiMPiC dev (6412767 p8-s4): MuseScore sets the `8` above the highest notes, at the
system crop's top edge; the top staff's crop cuts it off (a speck left), while `marks` draws its
signs near the staff, inside the crop. On OLiMPiC's engraving the staff arm never sees the sign,
and has learned to write no token it cannot see (likely, not checked: OLiMPiC's synthetic
training crops, at weight 2, label ottavas whose signs their staff crops cut off, the same
engraving). So the staff arm reads ottava passages an
octave off on every OLiMPiC-style page; Seq2Seq, whose crop reaches 7 interlines beyond the
staves, reads 60% of their notes right. T1 (the outer staves' crops slid out by an interline)
is judged first on this row.

**Tuplets.** Bars with a hidden tuplet number are read right more often than bars with every
number shown, by both arms: the hidden-number bars are mostly simple triplet runs, the shown ones
include sextuplets, mixed tuplets and the like. Both read about three bars in four without
tuplets right; with tuplets, the staff arm is ahead (35.2% against 28.2% shown, 59.7% against
52.4% hidden).

## The attention across staves switched off at inference (A3, 2026-11-14)

The staff arm's 50,000-step snapshot read with each block's `across_staves` layer skipped (the
grid passes through unchanged; `runs/overnight_2026-11-14/without_across.py`), against itself
intact. Step 4 again, on a model that reads (on the overfit model it was inconclusive).

| | Intact | Without attention across staves |
|---|---|---|
| OLiMPiC scanned dev SER (615) | 0.098 | 0.560 |
| Edits per gold token, upper / lower staff | 9.2% / 10.7% | 41.8% / 75.2% |
| Crops read exactly | 90 | 5 |
| Crops with the measure count right | 595 | 332 (216 with more) |
| Invisible-rest `forward` tokens written where none are | 44 | 8,061 |
| Outer-notes set SER (2,844) | 0.089 | 0.527 |
| Bars without tuplets, right rhythm | 74.9% | 25.6% |
| Bars with tuplets (numbers shown / hidden), right rhythm | 35.2% / 59.7% | 2.2% / 3.5% |

The trained model leans hard on the layer, the lower staff most (a guess, not tested: the
label gives each bar's upper-staff tokens first, so where the reading of the lower staff is in
the bar may come from the upper staff's columns). This says the
layer is used, not that it is needed: a model trained without it may learn the same from the
decoder's view of both grids. T5 (trained with `--across-columns 0`) answers that.

## The overnight runs at 20,000 steps: too short to test what they were for (2026-11-14)

Eleven 20,000-step runs on RTX 5090 pods (`docs/history/EXPERIMENTS_STAFF_MODEL_PLAN.md`; the race's data with
`grandstaff_written_v2`; warmup 4,800, cosine to 20,000; results in R2 and
`runs/overnight_2026-11-14/r2/runs`). Dev SER is OLiMPiC scanned dev (615); the rest is the
outer-notes set (2,844 crops) and its tuplet bars (`tuplet_bars.py`).

| Run | Dev SER | Outer SER | Under a sign | Far outside | Other notes | Tuplet bars, shown / hidden | Bars without tuplets |
|---|---|---|---|---|---|---|---|
| T0 baseline, seed 0 | 0.271 | 0.241 | 100% | 29.0% | 10.7% | 0.0% / 0.6% | 44.9% |
| T0b baseline, seed 1 | 0.287 | 0.262 | 100% | 32.1% | 12.4% | 0.0% / 0.0% | 40.5% |
| T0c baseline, seed 2 | 0.250 | 0.222 | 100% | 28.0% | 10.3% | 0.0% / 0.6% | 44.9% |
| T1 outer crops slid out | 0.278 | 0.251 | 100% | 28.1% | 11.1% | 0.0% / 0.9% | 45.0% |
| T2 two blocks | 0.259 | 0.235 | 100% | 26.9% | 9.8% | 0.0% / 0.3% | 46.2% |
| T3 groups of 5 columns | 0.272 | 0.244 | 100% | 28.5% | 10.6% | 0.0% / 0.0% | 44.9% |
| T5 no attention across staves | 0.285 | 0.271 | 100% | 35.9% | 12.8% | 0.0% / 0.0% | 40.8% |
| T6 `unmarked` at weight 3 | 0.282 | 0.260 | 100% | 32.9% | 12.5% | 0.2% / 0.3% | 40.9% |
| T9 batch 20, lr 4.2e-4 | **0.123** | 0.124 | 100% | 11.6% | 4.5% | 17.6% / 45.9% | 69.1% |
| T4 Seq2Seq baseline | 0.296 | 0.279 | 100% | 46.9% | 16.2% | 0.0% / 0.0% | 38.2% |
| T10 Seq2Seq, relative rows | 0.276 | 0.252 | 100% | 36.8% | 13.3% | 0.2% / 0.3% | 43.2% |

T7 (100,000 steps) stopped at step 75,800 when the account's balance ran out (below); dev SER
0.074 at 75,000, against the race's 0.098 at 50,000. It is being finished from its last.pt
(step 75,000) on the local GPU.

**Conclusions.**
1. *20,000 steps of this schedule is too short.* Every 20,000-step run is at dev SER 0.25-0.30,
   worse than the 50,000-step run's own snapshot at step 20,000 (0.185): with warmup taking 4,800
   steps and the rate decaying to nothing by 20,000, a short schedule learns less per step than the
   first 20,000 of a long one (the plan expected 0.14-0.17; that was wrong). None reads tuplet bars
   (0-1%), and none writes a single ottava token, Seq2Seq included. So T1 (ottavas), T6 (tuplets)
   and T8's question cannot be read off these runs; the 50,000-step race models can (tuplet bars
   35-60%, Seq2Seq reading 60% of notes under signs).
2. *Seeds differ more than the variants do.* The three baselines span 0.250-0.287; T1, T2, T3, T5,
   T6 and T10 all fall inside that span or at its edge. T5 (no attention across staves) and T4
   (Seq2Seq) are at the worse end and T2 (two blocks) at the better, hints at most.
3. *Training amount is the lever.* T9 (twice the samples a step) is at 0.123 against about 0.27
   and is the only 20,000-step run reading tuplets; T7 at 75,000 steps is at 0.074. But T9 took
   about twice as long a step (20 samples a step at about the same samples a second), so per hour
   of GPU it is no faster: what counts is samples seen. Per sample it is about where the race was
   at 40,000 steps (400,000 samples, dev SER 0.100 mid-schedule).
4. *Experiments must run to about 50,000 steps* for tuplets and ottavas to be measurable, with two
   baseline seeds, since the noise at that length is not yet known.

**Operations.** Pods are now removed by their own scripts once their results are verified in R2
(`runs/overnight_2026-11-14/pod_side.sh`); the first night they waited for the session, which ran
on the PC; the PC restarted at about 06:05 UTC, eleven finished pods idled until the balance ran
out at about 09:00, $40.14 spent against about $15 needed, and T7 stopped. Details and the other
lessons (two bad Community hosts, a hung SSH launch, Vast.ai's prices) in `TODO.md`.

## Where the decoder looks when it decides a tie, a slur or a tuplet (2026-11-14)

The question: do ties, slurs and tuplets go wrong because the decoder does not look at what decides
them (an arc's end note; the rest of the bar, the time signature)? `runs/overnight_2026-11-14/
attention_probe.py`: both race arms (50,000 steps), the labels fed (teacher forcing), on 300 of
OLiMPiC dev's written crops (150 synthetic, 150 scanned); the decoder's cross-attention averaged
over heads and summed into image columns; a note's column is where the decoder looks when it
writes that note's pitch. "Read right": the label's token is the decoder's first choice there.
Only arcs whose ends are three or more columns apart.

When writing `tied:start` / `slur:start`, attention within one column of the arc's end note:

| | Arcs | Averaged over layers | Best layer | On the start note | Even spread | Writing the end note's own pitch |
|---|---|---|---|---|---|---|
| Staff arm, ties read right | 201 | 5.6% | 15.7% | 4.0% | 2.2% | 14.6% |
| Staff arm, ties read wrong | 55 | 6.4% | 18.6% | 3.5% | 2.3% | 13.8% |
| Staff arm, slurs read right | 963 | 7.7% | 28.7% | 3.6% | 2.2% | 13.1% |
| Staff arm, slurs read wrong | 35 | 5.5% | 14.9% | 4.1% | 2.3% | 13.2% |
| Seq2Seq, ties read right | 205 | 5.8% | 17.4% | 6.5% | 2.3% | 13.9% |
| Seq2Seq, ties read wrong | 49 | 6.7% | 19.0% | 7.7% | 2.5% | 13.6% |
| Seq2Seq, slurs read right | 902 | 6.9% | 21.8% | 7.5% | 2.3% | 12.6% |
| Seq2Seq, slurs read wrong | 27 | 4.9% | 14.4% | 7.5% | 2.3% | 13.6% |

- *Both models look ahead to the end note when deciding the arc:* about three times an even
  spread on average, and one layer puts as much there as it does writing that note's own pitch
  (15-29%). The model is not blind to the far end.
- *Slurs read wrong are slurs it looked at less* (best layer 14.9% against 28.7% for the staff arm,
  14.4% against 21.8% for Seq2Seq): a slur goes wrong when the decoder does not find its end.
- *Ties read wrong are not:* their end gets as much attention as ties read right, or more. A tie's
  error is a judgement made while looking (a tie taken for a slur, or none), not a failure to look.
- Tuplets: writing a group's first ratio token, about 90% of the attention lies farther than two
  columns from the note itself, for right and wrong alike (staff arm 94.0% / 88.5%, Seq2Seq 90.6% /
  85.8%, 189 and 9, 191 and 7 groups): the decoder looks across the bar. None of the 198 groups was
  in a crop whose label has a time signature, so whether it looks at one is not answered here.

Caveats: attention is not causation, and teacher forcing is not free reading; 300 crops; wrong
cases few (27-55 arcs, 7-9 tuplet groups).

## Key signatures: read well on prints, misread on two handwritten pages (2026-11-14)

The question: as with tuplets and the meter, does a weak reading of the key signature cost
pitches? In LMX a pitch token gives line or space only (`F4`); its sharp or flat comes from the
key token (`key:fifths:N`) and printed accidentals, so a misread key is one token in SER but a
semitone on every note of the affected steps once the reading is a score, each charged by OMR-NED.

**OLiMPiC dev** (outer-notes set, 2,844 crops, written labels): the first key signature is wrong in
3 crops for the staff arm (0.1%), 7 for Seq2Seq (0.2%); key tokens differ anywhere in 22 and 27.
Printed accidentals: 5.3% wrong (staff arm) and 5.6% (Seq2Seq) of 14,184, with 308 and 350
written where none are.

**Polish dev** (`runs/overnight_2026-11-14/polish_keys.py`, print gold, 49 two-staff pages; the key in
force per staff and bar, bars paired in order; notes found at their place, by onset, letter and
octave, whose sharp or flat differs):

| | Staff-bars, key wrong | Pages | Semitone off, key right | Semitone off, key wrong |
|---|---|---|---|---|
| Staff arm (50,000) | 72 of 2,150 (3.3%) | 4 | 155 of 13,420 (1.2%) | 41 of 237 |
| Seq2Seq arm (50,000) | 64 (3.0%) | 4 | 155 of 13,774 (1.1%) | 26 of 206 |
| seq516 (150,000) | 26 (1.2%) | 2 | 79 of 14,712 (0.5%) | 27 of 160 |

- *Key signatures are not a large cost.* About 3% of bars, nearly all on two pages: 051 (32 of 46
  bars, staff arm) and 065 (30 of 82), both handwritten. By eye on 051: four flats, the
  copyist's loosely spaced and crowding the first notes, read as two (system 2) and three
  (system 3) on both staves; each system restates its key, so one misread signature takes its
  system. 093 and 012 have 8 and 2 such bars.
- *Accidentals cost more than keys:* about 155 notes a semitone off in bars whose key is right,
  against 26-41 in bars whose key is wrong. seq516 has half as many (79), as with every other
  error kind.
- *Both are small beside the notes missed and added* (about 1,700 for the staff arm on Polish).
- Handwriting is about 10% of training samples (Smashcima at weight 2, 53 MusiCorpus grand
  staves at weight 30); whether more of it would fix keys like 051's is open.

## T7: the staff model at 100,000 steps, and the ottava conclusion revised (2026-11-14)

T7 (`runs/t7_100k_final`): StaffSeq2Seq, the race's mix with `grandstaff_written_v2`, a
100,000-step schedule; 75,000 steps on a pod, the rest from its last.pt on the local RTX 5080.
OLiMPiC scanned dev SER (averaged weights), against the race's staff arm (50,000-step schedule):

| Step | 20k | 30k | 40k | 50k | 60k | 70k | 75k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| T7 | 0.154 | 0.102 | 0.088 | 0.080 | 0.075 | 0.077 | 0.074 | 0.071 | 0.074 | **0.0715** |
| Race staff arm | 0.185 | 0.127 | 0.100 | 0.098 (end) | | | | | | |

Most of T7's gain came before 50,000 steps; 50,000 to 100,000 took it from 0.080 to 0.0715. That
T7 at 50,000 steps (its rate still high) is ahead of the race at the end of its schedule points to
the data (GrandStaff without open ties), the pod's hardware, or a longer schedule's higher rate; the
50,000-step baselines on the same data will tell which.

Outer-notes set and tuplet bars, against the race's staff arm at 50,000 steps:

| | Race staff arm (50,000) | T7 (100,000) |
|---|---|---|
| SER, 2,844 crops | 0.089 | 0.065 |
| Notes under a sign, wrong | 100.0% | **31.3%** |
| Notes far outside the system, wrong | 10.6% | 6.4% |
| All other notes, wrong | 3.0% | 2.2% |
| Ottava tokens where no sign is | 0 | 527 |
| Tuplet bars, numbers shown, right rhythm | 35.2% | **57.5%** |
| Tuplet bars, a number hidden, right rhythm | 59.7% | **69.1%** |
| Bars without tuplets, right rhythm | 74.9% | 82.4% |
| Tuplet-ratio tokens read | 52.6% | 71.9% |

**Polish dev** (OMR-NED, 49 two-staff pages; T7 through the segmenter as before):

| | Print, authors' | Print, whole page | Corrected, authors' | Published, authors' |
|---|---|---|---|---|
| Staff arm, 50,000 (race) | 11.58 | 10.82 | 15.47 | 43.71 |
| **T7, 100,000** | **7.16** | **6.72** | **11.48** | **41.36** |
| seq516, 50,000 (of its 192,000 schedule) | 9.68 | | 13.63 | |
| seq516, 100,000 (of its 192,000 schedule) | 6.97 | | 10.92 | |
| seq516, 150,000 | 6.97 | | 10.84 | |

Training length closes nearly all of the gap to seq516: T7 at 100,000 steps is within 0.2 of
seq516 on print gold and 0.6 on corrected. seq516's own curve: 9.68 at 50,000 steps (mid-schedule),
6.97 at 100,000 and still 6.97 at 150,000: on Polish both designs stop gaining at about 100,000.
Three-staff pages stay at about 91 for every model (none trained on three staves).

**The ottava conclusion revised.** "The staff arm writes no ottava token: on OLiMPiC's engraving it
never sees the sign" (above) was too strong. At 100,000 steps the staff model writes them and reads
69% of the notes under signs right, more than Seq2Seq at 50,000 (60%) with its taller crop; likely
from what of the sign its crop keeps (the top of the 8, the dashed line running into the crop),
learned late, and with more false signs (527). E1 (the crops slid outward) still tests how much a
fuller view adds; it is not the only way to the sign. **Tuplets, too, come with training**: bars
with a hidden number from 59.7% to 69.1%, all numbers shown from 35.2% to 57.5%.

## Layout probe: Seq2Seq reads by where the staves sit and how large they are (2026-11-14)

Does a model read relative to the staff lines, or by where things sit in its crop?
`runs/overnight_2026-11-14/layout_probe.py`: OLiMPiC scanned dev's 615 crops read as they are and
with white margin added, 25% of the crop's height above, below or both (staff positions moved to
match, so the staff model cuts the same staves). A system crop scaled to a fixed height then shows
its staves smaller (80% for one margin, 67% for both) and moved (down for a margin above). SER:

| | As is | +25% above | +25% below | +25% both |
|---|---|---|---|---|
| Seq2Seq, race (absolute rows, 50,000) | 0.110 | 0.119 | 0.110 | **0.153** |
| Staff arm, race (50,000) | 0.098 | 0.099 | 0.097 | 0.101 |
| Staff baseline, seed 0 (50,000) | 0.092 | 0.087 | 0.087 | 0.087 |

- *The staff models do not move,* as they should (their staff crops are the same); their spread,
  about 0.005, is the noise of reading on a GPU (not bit for bit repeatable, and greedy decoding
  carries a small difference on).
- *Seq2Seq depends on where the staves sit:* a margin above and a margin below shrink the staves
  alike, but only the one above (the staves pushed down from where they usually are) costs, 9%.
- *And on how large they are:* both margins (staves at two thirds of their size, as in a
  three-staff system at a fixed height) cost 39%.
- T4 and T10 (50,000 steps, today's data) through the same probe:

| | As is | +25% above | +25% below | +25% both |
|---|---|---|---|---|
| T4, Seq2Seq (absolute rows) | 0.109 | 0.118 (+9%) | 0.122 (+12%) | 0.160 (+47%) |
| T10, Seq2Seq (relative rows) | 0.098 | 0.126 (+29%) | 0.115 (+18%) | 0.164 (+68%) |

- *The weakness is the staves' size, not where they sit.* T4 loses about as much to a margin below
  as above (the race's Seq2Seq less so); every variant shrinks the staves, and relative rows,
  which take only where the staves sit out of the reading, do not help: T10 is more sensitive,
  not less. A CNN's features depend on the size of what it sees; the staff models, which scale
  each staff to a fixed interline, do not move.
- So interline scaling (the system at a fixed interline, `--interline-scale`) is the fix this
  probe points to, for robustness to layout as for systems of more staves; relative rows come with
  it only because rows have no fixed meaning in crops of varying height.

## The 50,000-step batch: Seq2Seq baseline (T4) and relative rows (T10) (2026-11-14)

Both on the race's mix with `grandstaff_written_v2`, `--staff-samples-only`, height 512, 50,000
steps, RTX 5090 (Secure). Against the staff baselines of the same batch (seeds 0 and 1).

| | T4 Seq2Seq | T10 relative rows | Staff, seed 0 | Staff, seed 1 |
|---|---|---|---|---|
| OLiMPiC dev SER | 0.109 | **0.098** | 0.092 | 0.100 |
| Outer-notes SER | 0.100 | 0.097 | 0.084 | 0.091 |
| Pitch wrong, upper / lower staff | 3.25% / 4.73% | 2.98% / 4.09% | 2.41% / 3.58% | 2.75% / 4.19% |
| Pitch a step off, upper / lower | 0.36% / 0.55% | 0.36% / 0.75% | 0.20% / 0.24% | 0.16% / 0.26% |
| Notes under a sign, wrong | 45.3% | 40.4% | 100% | 52.1% |
| Far outside the system, wrong | 10.4% | 11.6% | 12.8% | 11.3% |
| Tuplet bars right, numbers shown / hidden | 25.2% / 57.4% | 25.3% / 56.2% | 38.6% / 55.6% | 34.6% / 56.2% |
| Bars without tuplets, right | 73.8% | 75.1% | 74.9% | 74.4% |

- *At matched data, schedule and hardware the staff model leads Seq2Seq on OLiMPiC:* dev SER by
  about 15%, pitches a step off by about 2x, tuplet bars with numbers shown by 10-13 points.
- *Relative rows close most of the overall gap* (dev SER 0.109 to 0.098, within the staff seeds'
  range) and lower wrong pitches, *but not pitches a step off* (unchanged above, worse below):
  the fixed-frame account of the staff model's pitch advantage (above) is not borne out; the
  advantage lies elsewhere (a constant scale, staff-anchored rows, or attention within a staff),
  untested. Not resolution as such: a 256 px staff crop has the pixels per interline of a typical
  two-staff system at 512 (seq/staff_data.py), so the two models read at the same scale on average;
  the staff model's scale is constant, Seq2Seq's moves with each system's margins and spacing.
- Tuplet bars do not move with relative rows. Polish dev for both is in the local queue.

## The 50,000-step batch on Polish dev (2026-11-14)

All nine runs of the batch (`docs/history/EXPERIMENTS_STAFF_MODEL_PLAN.md`, 50,000 steps, the race's mix with
`grandstaff_written_v2`, RTX 5090 pods), read through the segmenter as before. OMR-NED, 49
two-staff pages:

| Run | Print, authors' | Print, whole page | Corrected | Published |
|---|---|---|---|---|
| Staff baseline, seed 0 | 11.03 | 9.79 | 14.98 | 42.99 |
| Staff baseline, seed 1 | 11.71 | 10.33 | 15.39 | 43.90 |
| E1, outer crops slid out | 10.99 | 10.38 | 15.33 | 43.25 |
| T8, meter given | 10.75 | 10.06 | 15.03 | 43.63 |
| T2, two encoder blocks | 10.68 | 10.11 | 14.48 | 43.44 |
| T6, `unmarked` at weight 3 | 12.86 | 11.57 | 16.00 | 44.64 |
| T5, no attention across staves | 12.45 | 11.42 | 16.24 | 44.07 |
| T4, Seq2Seq | 12.63 | 11.45 | 16.95 | 44.92 |
| T10, Seq2Seq with relative rows | 12.51 | 11.10 | 16.48 | 44.37 |

Paired by page (print, authors'; the two staff seeds averaged per page; 95% range of 5,000
resamples of pages):
- *Staff model against Seq2Seq:* 1.26 lower than T4 (range -3.13 to +0.43; better on 20 pages,
  worse on 16 by more than 1), 1.14 lower than T10 (-2.35 to -0.05). About 10%, as on OLiMPiC
  (15%); against one Seq2Seq seed, at the edge of what pages can show.
- *Against the staff baselines:* T2 -0.69 (-2.46 to +0.88), T8 -0.62 (-1.95 to +0.66), E1 -0.38
  (-1.59 to +0.82): none clear. T5 +1.08 (+0.23 to +1.96): **the attention across staves helps on
  Polish too**. T6 +1.49 (-0.32 to +3.80): **more crops with hidden tuplet numbers make Polish
  worse**, though they help OLiMPiC's bars whose number is hidden (61.8% against about 56%);
  perhaps they teach tuplets where none are; not looked into.

**For the next runs.** Keep the attention across staves; leave the `unmarked` weight at 1 until
T6's cost on Polish is understood; the meter (T8) points the right way on both benchmarks without
being clear on either; E1 helps notes far outside the system (6.3% against 11-13% wrong) without
moving Polish. Whether the staff model's lead over Seq2Seq is its constant scale is pure (c)'s
question (Seq2Seq with `--interline-scale`, today's attention, running).

## Did T2, T8 and E1 help? Errors by kind of token (2026-11-14)

`runs/overnight_2026-11-14/category_errors.py` (output in `category_errors_50k.txt`) charges each
edit of a run's outer-notes readings (OLiMPiC dev, 1,422 systems, synthetic and scanned) to the
kind of token it concerns, per 100 label tokens of that kind. Each variant against the mean of
the two staff baselines (seeds 0 and 1); 95% range from resampling systems. The last column is
the baselines' own gap (seed 1 minus seed 0): **the resampling range shows how much the systems
read vary, not how much training runs do, and the seeds differ by more than most effects below.**

| Kind (label tokens) | Baselines, seed 0 / 1 | E1, outer crops | T8, meter | T2, two blocks | Seed gap |
|---|---|---|---|---|---|
| Pitch (162,980) | 3.66 / 4.32 | +0.01 | -0.07 | -0.31 | +0.66 |
| Accidental (14,184) | 6.94 / 7.71 | +0.56 | -0.08 | -0.65 | +0.78 |
| Duration (214,790) | 5.21 / 5.79 | -0.16 | -0.55 | -0.19 | +0.59 |
| Tuplet ratio and marks (33,852) | 50.6 / 54.1 | **-5.11** | **-5.44** | -0.41 | +3.49 |
| Tie (13,532) | 19.2 / 18.3 | +1.39 | +0.29 | +0.43 | -0.90 |
| Slur (19,246) | 17.8 / 19.6 | +0.70 | -0.07 | -0.37 | +1.81 |
| Beam (51,526) | 2.73 / 3.44 | +0.23 | -0.07 | -0.30 | +0.72 |
| Voices, chords, backups (113,030) | 5.89 / 6.12 | -0.30 | **-0.77** | -0.30 | +0.23 |
| Rest (17,864) | 12.1 / 12.6 | -0.55 | +0.85 | -0.17 | +0.55 |
| Key, clef, time (10,558) | 1.43 / 1.69 | -0.07 | -0.05 | -0.28 | +0.26 |
| Ottava (948) | 100 / 111 | **-17.3** | -5.5 | -1.5 | +11.0 |
| Articulation (18,826) | 22.3 / 24.1 | +0.18 | +0.74 | -0.54 | +1.72 |

Read against the seed gap, not the resampling range:
- **T8 (meter given): helps rhythm.** Tuplet tokens -5.4 (seed gap 3.5), voices and chords -0.8
  (gap 0.2), durations -0.6 (gap 0.6); tuplet bars with numbers shown 44.5% right against
  34.6-38.6%. Everything it should touch moves its way, and nothing else moves. The one variant
  of the three whose effect is clear on OLiMPiC.
- **E1 (outer crops slid out): helps what sits outside the system, costs a little on ties.**
  Notes far outside 6.3% wrong against 11-13%, ottava tokens -17 (gap 11); tuplet tokens -5.1;
  ties +1.4 (gap 0.9), accidentals +0.6 (gap 0.8): the outer staves' crops lose some of the
  space between the staves to gain what lies beyond them. A small, targeted gain.
- **T2 (two encoder blocks): nothing beyond seed noise.** Slightly lower on most kinds (pitch
  -0.3, accidentals -0.65, key -0.3), each within the seeds' gap; ties and slurs, the reason it
  was run, do not move (+0.4, -0.4). Polish -0.69 is within page noise too. As at 20,000 steps.
- **On Polish none of the three is resolved by one seed.** The two baselines differ by 4.4 OMR-NED
  per page (standard deviation); each variant's page-by-page differences have the same spread, so
  the means (-0.38 to -0.69) are below what 49 pages and one seed can show. The largest moves are
  a few rhythm-heavy pages (003, 017, 030: 25-28 for both baselines, 6-17 for T8 and T2) with the
  same notes and bars written and only their reading different.
- Ottava, grace notes and rests swing between seeds by 10-25 points: no single-seed conclusion
  on them.

So: **keep T8's meter (clear on rhythm), keep E1's outward slide if notes outside the system
matter (small, targeted), drop T2's second block (costs a second encoder for nothing measured).**
A second seed of T8 would turn its effect from likely to established.

## (c): Seq2Seq at a constant interline, attention unchanged (2026-11-14)

The first step of walking Seq2Seq toward the staff model (docs/history/EXPERIMENTS_STAFF_MODEL_PLAN.md, "Why is
the staff model better?"): whole systems scaled so that every staff's interline is a staff crop's
(256/15, about 17 px; `at_interline`, height up to 1,024), absolute row embeddings and attention as
T4's. `--staff-samples-only ... --interline-scale`, 50,000 steps, seed 0, Community RTX PRO 6000
(about 3 h 15 min). Run `runs/c_interline_pure_50k`.

| run | dev SER (scanned, 615) | outer notes SER, both renderings | false ottava tokens | Polish dev, print, authors, two staves |
|---|---|---|---|---|
| staff, seed 0 | 0.0916 | 0.0839 | 0 | 11.03 |
| staff, seed 1 | 0.0999 | 0.0909 | 249 | 11.71 |
| T4 (Seq2Seq, 512 px) | 0.1087 | 0.0996 | 338 | 12.63 |
| T10 (relative rows) | 0.0976 | 0.0966 | 360 | 12.51 |
| **(c) constant interline** | **0.1139** | **0.1052** | **41** | **13.35** |

Tuplet-ratio tokens read: 39.3% (staff seed 1: 50.6%). Layout probe (as is / +25% above / +25%
below / +25% both): (c) 0.114 / 0.117 / 0.117 / 0.123, against T4's 0.109 / 0.118 / 0.122 / 0.160.

- A constant scale alone does not close the gap: (c) is slightly worse than T4 on every measure
  but false ottavas, by less than the two staff seeds differ. One seed.
- It does what it should on the layout probe: added margin no longer shrinks the staves, so the
  model's errors barely move (+0.010 with both margins, T4 +0.052). The size sensitivity Seq2Seq
  showed is gone, and the gap to the staff model stays: size was not what the staff model wins by.
- False ottava tokens fall from 338 to 41, nearer the staff models' (0 and 249); why is not yet
  looked at.
- The staff positions it scales by were checked: after scaling every source's interline is
  17.05-17.07 px; a five-line fit to the ink agrees with the stored positions (median within 0.05
  interline and 1% spacing on prints), and the crops the fit flagged (handwritten, blurred
  GrandStaff, tilted scans) were looked at by eye: lines on their staves. A tilted scan's lines
  drift a fraction of an interline toward one end, for both models alike.
- So what the staff model has beyond scale: the fixed frame (f) and the attention within and across
  staves (g), below.

## (f) and (g): the staff model's fixed frame and its attention, on whole systems (2026-11-14)

Both on (c)'s constant interline, 50,000 steps, seed 0, Secure GPUs (about 5 h each).
- (f) `--staff-anchored`: absolute row embeddings replaced by the staff model's frame. Each row is
  given its offset from the nearest staff's centre (in interlines, sine waves) plus that staff's
  embedding. Run `runs/f_staff_anchored_50k`.
- (g) `--staff-anchored --staff-bands`: also the staff model's two-stage attention. First each
  staff's rows attend among themselves, then each column attends across staves.
  Run `runs/g_staff_bands_50k`.

| run | dev SER (scanned, 615) | outer notes SER, both | false ottava tokens | tuplet-ratio tokens read | Polish dev, print, authors, two staves |
|---|---|---|---|---|---|
| staff, seed 0 / 1 | 0.0916 / 0.0999 | 0.0839 / 0.0909 | 0 / 249 | - / 50.6% | 11.03 / 11.71 |
| T4 (Seq2Seq, 512 px) | 0.1087 | 0.0996 | 338 | - | 12.63 |
| T4b (T4, seed 1) | 0.1135 | 0.1046 | 198 | 43.9% | 12.68 |
| (c) constant interline | 0.1139 | 0.1052 | 41 | 39.3% | 13.35 |
| (f) + fixed frame | 0.1202 | 0.1101 | 295 | 30.1% | 15.94 |
| (g) + frame + staff attention | 0.1051 | 0.1040 | 304 | 44.0% | 13.11 |

- No step of the walk reaches the staff model, and none beats T4 on Polish.
- The fixed frame alone hurts: (f) is worst everywhere, 3.3 behind T4 on Polish, more than the
  two staff seeds differ.
- The staff model's attention recovers what the frame lost and a little more. (g) is (f) plus
  attention, and it is better by 0.015 dev SER, 14 points of tuplets and 2.8 on Polish. It is
  level with T4 at best: ahead on scanned dev by 0.004, behind on Polish by 0.5, both within seed
  noise.
- So the frame is useful only with the attention that keeps each staff to itself. Even both
  together, on a whole system, do not give what cutting the system into staves gives. The likeliest
  remaining difference (not tested) is what (g) still shares with Seq2Seq: the gap between staves is seen once, in one image, and the
  encoder's convolutions see across it. The staff model crops each staff separately (the gap twice,
  and nothing of the next staff).
- One seed each; T4's second seed (running) gives the noise on the Seq2Seq side.

## (h) and (i): what of cutting into staves matters? (2026-11-14)

After (c), (f) and (g) the difference left is the crop itself. Pixels per staff are not it: at
staff scale every source's interline is 17.07 px for both models, no system is shrunk (2 of about
2,100 past 1,024 px) and widths are squeezed alike (0-7%). A staff crop is 15 interlines tall and
staves sit 9-11 apart, so neighbouring crops share 4-6 interlines of ink, each placed by its own
staff, and the convolutions see each staff alone. Two runs pull those apart, 50,000 steps, seed 0,
Community RTX 5090s:
- (h) `--staff-samples-only P --interline-scale --staff-anchored --staff-windows`: (g) with the
  nearest-staff bands replaced by overlapping windows, each the 8 feature rows a staff crop would
  hold, each row embedded by its offset from the window's own staff. StaffSeq2Seq's encoder on the
  whole system's features: the convolutions see across staves. If it reaches the staff model, a
  whole-system model can have its accuracy.
- (i) `--staves P --staff-cut-between`: the staff model with each crop white past the midpoints to
  its neighbours (no shared ink). If it holds, the overlap does not matter; if it falls to (g),
  the overlap is what (h) restores.

Results (staff seeds 11.03 / 11.71 and T4 12.63 on Polish, above):

| run | dev SER (scanned, 615) | outer notes SER, both | false ottava tokens | tuplet-ratio tokens read | Polish dev, print, authors, two staves |
|---|---|---|---|---|---|
| (g) + frame + staff attention | 0.1051 | 0.1040 | 304 | 44.0% | 13.11 |
| (h) + windows | 0.1055 | 0.0975 | 80 | 43.2% | 14.51 |
| (i) staff model, crops cut at midpoints | 0.1059 | 0.0965 | 280 | 48.9% | 12.58 |

Runs `runs/h_staff_windows_50k` and `runs/i_staff_cut_between_50k`.

- (h) is level with (g) on scanned dev and tuplets. It is better on outer notes by 0.007, now
  ahead of T4 (0.0996) though still behind both staff seeds (0.0839 / 0.0909), and it has a quarter
  of (g)'s false ottavas.
- (i) falls below both staff seeds on dev (by 0.014 / 0.006) and outer notes (0.013 / 0.006), to
  where (h) is. The tuplets hold (48.9% against seed 1's 50.6%). Seed 1 is within noise of it, so
  this is a lean, not a result: the neighbours' ink helps the staff model a little.
- Polish agrees about (i): 12.58, behind both staff seeds by 0.9 / 1.6 and level with T4. Without
  the neighbours' ink the staff model is no better than Seq2Seq.
- Polish disagrees about (h): 14.51, behind (g) by 1.4 and T4 by 1.9, despite its better outer
  notes and ottavas. The windows give the system model the staff model's overlap, but not its
  accuracy; what still separates them (untested) is the convolutions seeing each staff alone.
- Layout probe (OLiMPiC scanned dev SER, as is / +25% above / below / both): (h) 0.105 / 0.103 /
  0.107 / 0.103, (i) 0.106 / 0.102 / 0.106 / 0.102, against T4's 0.109 / 0.118 / 0.122 / 0.160.
  Anchored to the staves, (h) is as indifferent to the margins as the staff model, so its Polish
  loss is not the crop's size.
- Open: training time. T7 (staff, 100,000 steps) reached 7.16 on Polish and seq516 (Seq2Seq, other
  mix and schedule) 6.97 at 100,000: the 1-1.6 lead at 50,000 may be the staff model learning the
  layout sooner, not reading better. A Seq2Seq run matched to T7 (data, 100,000-step schedule),
  scored on Polish at 60k, 80k and 100k, tells.

## (j): the staff model with its frame loosened (2026-11-15)

Hypothesis 2 from the staff side: `--staves P --staff-jitter 1.5`, each training crop centred up
to 1.5 interlines off its staff, so the staff no longer sits at the same rows of every crop
(inference unjittered). 50,000 steps, seed 0, local RTX 5080. Run `runs/j_staff_jitter_50k`.

| run | dev SER (scanned, 615) | outer notes SER, both | false ottava tokens | tuplet-ratio tokens read | Polish dev, print, authors, two staves |
|---|---|---|---|---|---|
| staff, seed 0 / 1 | 0.0916 / 0.0999 | 0.0839 / 0.0909 | 0 / 249 | - / 50.6% | 11.03 / 11.71 |
| T4 / T4b (Seq2Seq, seeds 0 / 1) | 0.1087 / 0.1135 | 0.0996 / 0.1046 | 338 / 198 | - / 43.9% | 12.63 / 12.68 |
| (j) staff, jittered 1.5 | 0.1096 | 0.0982 | 0 | 41.0% | 12.85 |

- Loosening the frame costs the staff model its whole dev lead: 0.018 / 0.010 behind the two
  staff seeds, level with T4, and its tuplets fall to Seq2Seq's (41.0% against 50.6%).
- It keeps the staff model's ottava habit (0 false ottava tokens), so that comes from the crops,
  not the frame.
- Polish agrees: 12.85, behind both staff seeds by 1.8 / 1.1 and level with T4 / T4b (12.63 /
  12.68).
- With (f), where the frame alone hurt Seq2Seq, this says the frame is necessary to the staff
  model but not sufficient for a whole-system one. Seq2Seq's two seeds differ by 0.05 on Polish,
  the staff model's by 0.7.

## How each model divides a system between its staves (2026-11-15)

`runs/overnight_2026-11-14/staff_attention.py`, on the 615 OLiMPiC scanned dev crops, labels fed
(teacher forcing), 50,000-step snapshots. (1) The decoder's cross-attention when it writes a pitch,
summed over memory rows by where each row sits: within 2.5 interlines of the pitch's own staff,
the rest of that staff's side of the midpoint, or the other staff's side; against an even spread.
(2) The same crops with the other staff's side whited out: how much the memory on the own staff's
rows changes (relative L2; the staff model's own crop only), and the share of the own staff's
pitches that are the decoder's first choice, intact and blanked. 20,251 treble and 14,662 bass
pitches.

| model | attention on own staff / its side / other side (even 13-16 / 33-37 / 49-52) | memory change, treble / bass | treble pitches right, intact -> blanked | bass pitches right |
|---|---|---|---|---|
| staff, seed 0 | 14 / 47 / 38; 14 / 44 / 42 | 21% / 25% | 97.6 -> 96.0 | 96.5 -> 93.4 |
| staff, seed 1 | 15 / 44 / 41; 16 / 48 / 37 | 21% / 24% | 97.6 -> 95.6 | 96.5 -> 92.8 |
| T4 (Seq2Seq, 512 px) | 19 / 44 / 37; 19 / 40 / 42 | 19% / 32% | 97.2 -> 93.7 | 96.2 -> 85.6 |
| (c) constant interline | 18 / 34 / 48; 18 / 46 / 36 | 20% / 29% | 97.0 -> 92.5 | 95.4 -> 89.8 |
| (g) + frame + staff attention | 16 / 32 / 52; 17 / 50 / 33 | 15% / 24% | 97.2 -> 95.6 | 94.9 -> 91.5 |

- The decoder's attention says little: every model puts only a little more than an even spread on
  the pitch's own staff and its side. Its memory rows have each seen the whole system (or, for the
  staff model, the other staff through the attention across staves), so where a row sits is not
  what it holds.
- Blanking says more. Seq2Seq reads a staff's pitches with more help from the other staff than the
  staff model does: without the treble, T4 loses 10.6 points of bass pitches and (c) 5.6, the staff
  model 3.1 and 3.7. T4's loss also has the layout in it (half its crop now blank, and it reads by
  where things sit); (c)'s does not.
- (g) is as independent as the staff model (1.6 and 3.4), but 1.4-2 behind it on Polish. So the
  staves' independence is not what gives the staff model its lead either.
- Caveats: blanked crops are outside every model's training; teacher forcing; one seed each but
  the staff model's two (the noise: under 0.6 points).

## (k): one staff at a time (2026-11-15, planned)

The staff model leads Seq2Seq by 0.9-1.6 on Polish dev (11.03 / 11.71 against T4's 12.63) and
0.009-0.022 dev SER, and no step of (c)-(i) gave a whole system its accuracy. On one staff the two
models nearly coincide: at staff scale both see 17.07 px per interline, and what differs is the
staff model's fixed 15-interline frame against the crop's own height. If the lead vanishes on one
staff, it is about reading several staves together; if it stays, it is the crop itself.

**The data: every two-staff system taken apart** (`seq.one_staff`, `train/build_one_staff.py`,
`scripts/build_onestaff_data.sh`), into `data/seq516g_onestaff`, laid out as `seq516g_data`.
- Labels: each bar's runs of notes between backups are a voice on one staff. A staff's label keeps
  the bar's key and meter, its own clef and runs, the backups between them recounted over the
  runs left out, and no staff tokens (OLiMPiC's linearizer on a one-staff part; a test checks
  this). LMX does not write a whole-bar rest's length: it is taken as the backup after it where
  that equals the bar's longest written voice (1,922 of 1,961 sampled) or every voice rests.
- Images: between the staves (half an interline clear of their lines) each piece of ink goes to
  the staff it touches, both if it runs right through (barlines, brace), and a piece touching
  neither (slur, dynamic, dot, tuplet number) to the side of the middle its centre lies on. Each
  crop ends at its own last ink in the gap; the other staff's ink there is painted out (with a
  0.1-interline halo, else blur and anti-aliasing leave outlines) in the paper's median tone.
- Left out, the system whole: a voice on both staves (cross-staff beams and chords, voices
  moving between hands), an ottava sign between the staves (8vb under the upper, 8va over the
  lower: alone in the gap nothing says whose; PDMX's sit as near the other staff, by eye), a
  clef change among the other staff's notes, notes and backups that do not add up, a whole-bar
  rest backed up by other than the bar's length.

| set | systems | split | voice on both | ottava between | other |
|---|---|---|---|---|---|
| OLiMPiC written | 14,907 | 12,493 | 2,404 | 2 | 8 |
| GrandStaff v2 | 29,500 | 27,113 | 2,320 | 0 | 67 (66 clef changes) |
| pdmx | 56,727 | 55,892 | 667 | 145 | 23 |
| beams | 29,267 | 28,858 | 333 | 67 | 9 |
| round | 29,266 | 28,857 | 333 | 67 | 9 |
| marks | 29,068 | 24,742 | 317 | 3,997 | 12 |
| musescore | 52,243 | 51,633 | 466 | 135 | 9 |
| unmarked | 3,987 | 3,717 | 217 | 34 | 19 |
| smashcima | 14,189 | 14,062 | 126 | 0 | 1 |
| MusiCorpus | 53 | 47 | 6 | 0 | 0 |
| **training, all** | **259,207** | **247,414 (95.5%)** | 7,189 | 4,447 | 157 |
| OLiMPiC scanned dev, clean 615 | 615 | 514 | 101 | 0 | 0 |
| outer-notes set (synthetic / scanned) | 1,422 | 1,225 | 197 | 0 | 0 |

Each split system gives two crops: 494,828 training crops, 1,028 dev crops. `marks` loses 14%:
its added 8va passages over the lower staff. OLiMPiC loses 16% and the dev 101 of 615 to voices
crossing between the hands, so the one-staff dev is easier than the two-staff dev in exactly what
it leaves out: one-staff numbers compare only with each other.

Checked by eye (contact sheets of systems and their two crops; rare tokens in their crops): notes
far below the upper staff and far above the lower kept whole with their stems and beams on their
own staff, beamed groups reaching past the middle kept, barlines in both, ties and slurs with
their notes, tuplet numbers below an upper staff kept with it, the 8vb sign that made us drop
ottavas between the staves. Seen and left: words in the gap (dynamics, `dim. sempre`, `a tempo`)
split letter by letter between the crops (noise either way); painted patches lighter than a
GrandStaff copy's uneven grey paper; a cross-staff slur drawn as a half arc on each staff with an
open slur in each label, as at a system's edge; on Smashcima's odd layouts a floating beamed pair
in the gap can land in both crops. Also found, not ours: some PDMX crops label `ottava:8vb`
where the source crop draws no sign (`samples/006690/s2`, lower staff, second bar).

**The runs**, 50,000 steps, seed 0, the race's mix and schedule (`staff_experiment.sh`):

```
R=/home/erikg/projects/noteworthy; O=$R/data/seq516g_onestaff
export NW_SEQ_DATA=$O NW_EXPERIMENT_STEPS=50000 NW_EXPERIMENT_DEV=$O/olimpic-1.0-scanned \
  NW_EXPERIMENT_DEV_SAMPLES=$O/olimpic-1.0-scanned/samples.dev.txt NW_EXPERIMENT_OUTER_DEV=$O/olimpic_written_dev
scripts/staff_experiment.sh k_staff_onestaff_50k --staves $O/staff_positions
scripts/staff_experiment.sh k_system_onestaff_50k --staff-samples-only $O/staff_positions --interline-scale
```

Seq2Seq at constant interline, (c)'s arm, not T4's `--height 512`: on one staff 512 px would give
twice the staff model's pixels per interline, a second difference besides the crop. On two staves
T4 led (c) by 0.005 dev SER and 0.7 on Polish, less than the two staff seeds differ. Smoke tests
(150 steps, batch 2, local GPU) train both arms and score the outer-notes set.

A step is 10 staves, not 10 systems: 50,000 steps see half the music the two-staff runs saw, for
both arms alike.

**What each outcome would mean** (dev SER on the 1,028 one-staff crops, outer notes on 2,450):
- Staff arm ahead by about the two-staff lead (0.01-0.02): the crop itself (the fixed frame, each
  staff alone in the convolutions) is what wins, with nothing of several staves.
- Level within seed noise (0.008 between staff seeds on two staves): the lead is about reading
  several staves: what the whole-system model does with the other staff, or the staff model's
  attention across staves.
- Seq2Seq ahead: the fixed frame costs on one staff what it gains on two.
- One seed each; a second seed for whichever way it falls before a conclusion.

Polish is not scored: its pages are two-staff systems. It would take cutting each segmented system
with `seq.one_staff.staff_images`, reading the staves apart and joining the two readings bar by
bar into one LMX (the inverse of `staff_labels`), with cross-staff passages unreadable by
construction and every bar-count mismatch between the two readings an error of the join, not
of either model.

## (l): Seq2Seq matched to T7, 100,000 and 150,000 steps (2026-11-15)

Training time, the hypothesis left after (c) to (j): the staff model may learn the layout sooner
rather than read better. Seq2Seq on T7's data and schedule (`--staff-samples-only P --height 512`,
the race's mix with `grandstaff_written_v2`, T4's settings otherwise), seed 0, Community RTX 5090s:
`runs/l_system_100k` (100,000-step schedule, against T7), `runs/l_system_150k` and the staff model
`runs/l_staff_150k` (150,000-step schedules, both still training).

OLiMPiC scanned dev SER (averaged weights):

| Step | 20k | 30k | 40k | 50k | 60k | 70k | 75k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| T7 (staff) | 0.154 | 0.102 | 0.088 | 0.080 | 0.075 | 0.077 | 0.074 | 0.071 | 0.074 | **0.0715** |
| (l) Seq2Seq, 100k | 0.172 | 0.116 | 0.100 | 0.085 | 0.075 | 0.077 | 0.082 | 0.078 | 0.073 | **0.0717** |

Outer-notes set (2,844 crops) and tuplet bars, both sets, at 100,000 steps:

| | T7 (staff) | (l) Seq2Seq |
|---|---|---|
| SER | 0.065 | 0.065 |
| Notes under a sign, wrong | 31.3% | **14.0%** |
| Notes far outside the system, wrong | 6.4% | 6.0% |
| All other notes, wrong | 2.2% | 2.2% |
| Ottava tokens where no sign is | 527 | 362 |
| Tuplet bars, numbers shown, right rhythm | 57.5% | 51.5% |
| Tuplet bars, a number hidden, right rhythm | 69.1% | 68.5% |
| Bars without tuplets, right rhythm | 82.4% | 82.5% |
| Tuplet-ratio tokens read | 71.9% | 68.2% |

- The staff model's dev lead is a head start: 0.018 at 20,000 steps, 0.005 at 50,000, none from
  60,000 on, and the two end level (0.0715 / 0.0717).
- At 100,000 Seq2Seq reads notes under an ottava sign twice as well (its crop holds the sign) with
  fewer false signs; the staff model keeps a small lead on tuplets with their numbers shown.
150,000-step schedules, OLiMPiC scanned dev SER:

| Step | 20k | 50k | 60k | 80k | 100k | 120k | 130k | 140k | 150k |
|---|---|---|---|---|---|---|---|---|---|
| (l) staff, 150k | 0.161 | 0.086 | 0.075 | 0.068 | 0.065 | 0.063 | 0.060 | 0.062 | **0.0614** |
| (l) Seq2Seq, 150k | 0.182 | 0.091 | 0.078 | 0.073 | 0.070 | 0.067 | 0.067 | 0.068 | **0.0660** |

The staff model at 150,000, outer-notes set (both): SER 0.057, notes under a sign 25.6% wrong,
far outside 4.8%, elsewhere 2.1%, 582 false ottava tokens; tuplet bars right 63.2% (numbers
shown), 70.6% (a number hidden), 85.2% (none); tuplet-ratio tokens read 77.8%.

Seq2Seq at 150,000, the same sets: SER 0.060, notes under a sign 13.6% wrong, far outside 5.6%,
elsewhere 2.1%, 233 false ottava tokens; tuplet bars right 60.1% / 68.5% / 84.2%; tuplet-ratio
tokens read 74.4%.

- On the longer schedule the staff model's lead does not close: 0.005 at 60,000, 0.005 at 80,000,
  0.004 at 100,000, 0.004 at 120,000, 0.007 at 130,000. Steady, but smaller than the staff
  seeds' spread at 50,000 (0.008), and the 100,000-step pair ended level: one seed each, so a lean.

Errors by kind of token at 150,000 (`category_errors.py`, outer-notes readings, per 100 label
tokens of the kind; edits = rate x tokens), Seq2Seq's ranked by edits:

| Kind (label tokens) | staff | Seq2Seq | Seq2Seq's edits |
|---|---|---|---|
| Tuplet ratio and marks (33,852) | 26.3 | **31.2** | 10,600 |
| Duration (214,790) | 3.45 | 3.71 | 8,000 |
| Voices, chords, backups (113,030) | 4.23 | 4.31 | 4,900 |
| Pitch (162,980) | 2.77 | 2.81 | 4,600 |
| Slur (19,246) | 16.3 | 15.8 | 3,000 |
| Articulation (18,826) | 14.1 | 15.2 | 2,900 |
| Tie (13,532) | 18.0 | 16.7 | 2,300 |
| Rest (17,864) | 9.7 | 10.9 | 2,000 |
| Ottava (948) | 106 | **52** | 500 |

- Seq2Seq's weakness is rhythm: tuplets most of all (the one kind it trails by more than the
  seeds' gap at 50,000, 3.5), then durations and rests. It is level on pitch, voices and beams and
  ahead on ottavas, ties and slurs.

Polish dev (print gold, authors' alignment, 49 two-staff pages):

| Step | 50k | 60k | 80k | 100k |
|---|---|---|---|---|
| T7 (staff) | 8.76 | 8.02 | 7.75 | **7.16** |
| (l) Seq2Seq, 100k | 9.79 | 9.13 | 8.22 | **7.44** |

150,000-step schedules:

| Step | 100k | 120k | 150k |
|---|---|---|---|
| (l) staff, 150k | 7.39 | 7.17 | 7.35 |
| (l) Seq2Seq, 150k | 6.82 | **6.73** | 7.29 |

- The staff model flattens on Polish from 100,000 (7.39 / 7.17 / 7.35) while its dev SER still
  falls (0.065 to 0.061): as seq516 did (6.97 at 100,000 and 150,000), and level with T7 (7.16).
- Seq2Seq on the same schedule is ahead on Polish at every snapshot: 6.82 / 6.73 / 7.29, by 0.57 /
  0.44 / 0.06. Its best (6.73 at 120,000) is the best Polish dev score so far, the staff model's
  dev SER lead (0.004-0.005) notwithstanding. Both slip a little by 150,000 (by one seed each,
  within page noise); the published model is not chosen on Polish dev, so its schedule length is a
  question for OLiMPiC or PDMX dev.
- **Decision: Seq2Seq.** With training matched, the staff model's lead was a head start: level by
  100,000 steps on dev, behind on Polish. Seq2Seq also reads ottavas twice as well, takes any number
  of staves, and is the simpler model.

- On Polish the gap narrows with training: 1.03 at 50,000, 1.11 at 60,000, 0.47 at 80,000, 0.28
  at 100,000. At 100,000 it is less than the staff seeds differ at 50,000 (0.7), more than
  Seq2Seq's do (0.05). Seq2Seq was still gaining 0.8 per 20,000 steps at the end, T7 0.6; the
  150,000-step runs tell whether it passes the staff model or both flatten.

## Does the decoder count the bar? (2026-11-15)

Seq2Seq's weakness at 150,000 steps is rhythm, tuplets most of all (above). Three questions: does
the decoder keep count of the bar, does it know at a bar's first note how long the bar will be (a
left-to-right decoder commits the first note before it writes the rest), and does it see tuplets
whose number is not printed? `runs/overnight_2026-11-14/counting_probe.py` (output in
`counting_probe.log`): both 150,000-step runs, OLiMPiC scanned dev (615 crops, 38,406 notes),
labels fed (teacher forcing). Linear probes (ridge) from the decoder's state where it chooses each
note's duration, trained on two thirds of the crops and read on the rest; self-attention by head.
Which groups hide their number comes from each crop's MusicXML (OLiMPiC's LMX has no
`tuplet:hidden`; our training labels do).

| | Seq2Seq | staff |
|---|---|---|
| Onset in the bar, R2 (best layer; embedding alone 0.23) | 0.85 | 0.86 |
| Bar's length at a voice's first note, R2 (embedding -0.01) | 0.83 | 0.85 |
| The same, bars whose length differs from the bar before (54 held out) | 0.49 | 0.63 |
| "Same as the bar before" at first notes, exact | 95.1% | 95.1% |
| Counting head (layer 2): attention on the voice's durations so far in the bar (even 5.2%) | 70.4% | 82.8% |
| The same head, durations read wrong | 37.3% | 70.1% |
| Ratio token read right, numbers shown / hidden | 98.7% / 99.7% | 98.5% / 99.6% |
| The same at a group's first note | 95.9% / 98.6% | 95.4% / 98.2% |
| "A tuplet" read out before the duration, a hidden group's first note (best layer) | 89.8% | 86.2% |
| Durations read right (labels fed) | 99.6% | 99.6% |

- **The decoder counts.** Its state knows where a note falls in the bar, and one head (layer 2,
  head 7 in Seq2Seq) looks almost only at the voice's durations so far. Where that head looks
  away, the duration goes wrong more often (37% against 70%; about 150 wrong durations).
- **It knows the bar ahead mostly from the bar before.** At a voice's first note the bar's length
  reads out well, but "the same as the bar before" is right 95% of the time. Where the length
  changes (about 5% of bars) the readout falls to R2 0.49: there the first note is written without
  knowing where the bar ends. Real but rare.
- **It sees hidden tuplets.** With the labels fed it writes a hidden group's ratio right 99.7% of
  the time, 98.6% at the group's first note, and its state says "a tuplet" before it writes that
  note's duration for 90% of them (by layers 4-6): it reads the unprinted triplet in the image.
  OLiMPiC dev hides more groups (566) than it shows (366).
- **So the rhythm errors are mostly compounding.** Fed the right history, durations are 99.6%
  right; reading on its own, Seq2Seq errs on 3.7 per 100 duration tokens. Some of that gap is notes
  missed or added, which teacher forcing cannot show, but most of the rhythm weakness looks like
  errors after an earlier error, not a failure to count or to see.
- What this changes: an embedding of the time used in the bar would tell the decoder what it
  already knows; drop it. Worth trying instead: reranking whole bars by whether their durations add
  up (no training; it can undo an early error), and training on the model's own readings
  (scheduled sampling) so it learns to recover. Next check, cheap: errors within a bar before and
  after its first error, reading freely.
- Caveats: teacher forcing is not free reading; linear readouts are a floor on what the state
  holds; within hidden groups the fed labels lack the `tuplet:hidden` the model would have written
  (no effect at a group's first note, where the ratio precedes `tuplet:start`).

## When real scores hide a tuplet's number (2026-11-15)

Our synthetic data hides tuplet numbers by a blind rule (`build_seq_dataset.py`): one system in
five with tuplets has every number hidden, half the time keeping the first; the `unmarked` set
hides all. `runs/overnight_2026-11-14/tuplet_hiding.py` (output `tuplet_hiding.log`) reads every
tuplet group (`show-number="none"` or `print-object="no"` on its start) in 40,000 PDMX piano scores
(one track, program 0, best path; 2,646 with tuplets), ASAP's piano scores (158 with tuplets) and
OLiMPiC's training systems (349 scores; a system's file rarely carries the time signature). All
three are MuseScore engravings. A run is back-to-back groups of one ratio and value in one voice,
nothing else between.

**Hiding is first a score's choice.** Scores hiding none / some / all: PDMX 2,583 / 52 / 11, ASAP
50 / 87 / 21, OLiMPiC 275 / 67 / 7. Within the scores that hide some, the share hidden by:

| | PDMX (52 scores) | ASAP (87) | OLiMPiC (67) |
|---|---|---|---|
| All groups | 71% | 74% | 74% |
| Run place 1st / 2nd / 3rd / 4th / 5th+ | 49 / 64 / 68 / 76 / 85% | 51 / 64 / 73 / 77 / 89% | 53 / 66 / 77 / 82 / 88% |
| First tuplet of the voice in a system / later | 55 / 76% | 64 / 77% | 56 / 77% |
| Alone in its bar / with other tuplets | 43 / 77% | 57 / 77% | 40 / 77% |
| 3:2 / 6:4 / 5:4 | 69 / 85% / - | 73 / 86 / 46% | 75 / 49% / - |
| 4/4 / 3/4 / 2/4 | 77 / 56 / 88% | 83 / 78 / 72% | - |
| Beamed / not | 73 / 68% | 75 / 72% | 75 / 72% |
| On the bar's first note / later | 71 / 72% | 78 / 72% | 72 / 75% |

Across every score in PDMX: 3:2 13%, 6:4 41%, 5:4 5%, 9:8 4%, 7:8 and 4:3 none of 288 and 225.

- **The rule engravers follow is the run:** the longer a passage of like groups has gone on, the
  likelier the number is gone, 50% at the first group to 85-89% from the fifth, the same in all
  three sources. In ASAP, 4/4 and 3/4 runs past their fourth group are 99% hidden.
- **They print it again where the reader needs it:** at the voice's first tuplet in a system
  (55-64% hidden against 76-77% later) and for a lone tuplet in a bar (40-57% against 77%).
- **Sextuplets are hidden more than triplets, odd groupings less**: 5:4 5% in PDMX (7:8 and 4:3
  never), but 30% in ASAP (46% within its scores that hide some): classical engravers hide
  quintuplets in running passages too.
- **3/4 is not hidden more than 4/4.** Within scores it is hidden less (PDMX 56% against 77%, ASAP
  78% against 83%); 2/4 is the most hidden in PDMX. Across all PDMX scores, more 3/4 scores hide
  some (by score 2.8% against 1.4%), which may be the impression: more waltz-like pieces with
  running triplets, not a rule of the meter.
- Beaming and where in the bar the group starts make no difference.
- Even the first group of a run is hidden half the time in these scores: a run here ends at any
  ordinary note, while an engraver thinks in passages (a triplet passage with a quarter between
  groups is still settled). Whether a number, once gone, ever comes back mid-run was not measured.
- Against our rule: we hide by system, all or nothing, in 20% of systems with tuplets, odd
  groupings and lone tuplets alike, and never produce the commonest real pattern (printed at the
  start of a passage or system, then dropped). A rule from this table, per score: show all (about
  three in four), hide all (a few), or hide by run place (1st about half, rising to 90%), printed
  at a system's first tuplet and for lone tuplets more often, 5s and 7s shown more often.
- MuseScore users, mostly amateurs in PDMX; ASAP is curated. Real printed editions may differ.

## How much of the rhythm error is compounding? (2026-11-15)

`runs/overnight_2026-11-14/compounding.py` (output `compounding.log`): `l_system_100k` at
100,000 steps on OLiMPiC scanned dev (615 crops), each label bar read twice. Fed the labels, a bar
is right if the model makes no rhythm decision differently in it (a duration, dot or ratio; a
rest, chord, backup or forward where the label has another of those or a barline; the order of
slur and tie marks, where our labels and OLiMPiC's differ, is not counted). On its own (greedy, as
the dev check and pages read), a bar is right if its rhythm matches the bar it aligns to
(tuplet_bars.py), after imply_tuplets. For a wrong bar paired with one reading bar: whether its
voices still add up (seq.lmx_tuplets: grace notes and a chord's later notes take no time, forwards
do, a whole-bar rest fills it), against the reading's own bar (its time signature, else the length
most of its bars have) and against the label's.

| | Bars | Right fed | Right on its own | Wrong bars right fed (the history's doing) | Wrong bars that do not add up, by its own bar / the label's |
|---|---|---|---|---|---|
| Plain bars | 2,006 | 91.7% | 80.1% | 252 of 399 (63%) | 52% / 53% |
| Bars with tuplets | 308 | 70.8% | 53.9% | 65 of 142 (46%) | 52% / 59% |

- **Most rhythm errors are compounding.** Fed the right history, 92% of plain bars are read right;
  on its own, 80%. Nearly two thirds of the bars it gets wrong it would have got right given the
  right tokens before them. For tuplet bars, about half.
- Tuplets are also hard on their own: 29% of tuplet bars go wrong even fed (8% of plain bars).
- **About half the wrong bars do not add up**, by the reading's own bar length, so a check that
  needs no gold could find them; the other half are wrong and still add up (a quarter read as two
  eighths, a dot moved). This is the most a bar check could fix, if a better reading is on offer.
- What follows: worth trying both a decoding-side fix (beam search, reranked by bar fill, and a
  reverse model) and the training-side one ((n), scheduled sampling, on a pod now).

**Beam search** (`generate(beam=K)`, the readings ranked by mean log-probability per token; 39e6ac6),
the same 615 crops, beam 4 (37 minutes against greedy's 29):

| Bars read right | Greedy | Beam 4's best | Beam 4, the one filling most bars | Any of beam 4 (oracle) | Fed |
|---|---|---|---|---|---|
| Plain bars (2,006) | 80.1% | 80.1% | 80.9% | 84.4% | 91.7% |
| Bars with tuplets (308) | 53.9% | 55.5% | 56.8% | 59.4% | 70.8% |

- **Beam 4 alone does nothing for plain bars** and adds 1.6 points on tuplet bars.
- **Reranking its readings by bar fill adds a little** (0.8 and 1.3 points), a fifth to a quarter of
  what a perfect pick among them would (the oracle: 4.3 and 5.5 points).
- **What the alternatives change** (beam 4's 2,036 alternatives against its best, over the 615
  crops; a crop can have more than four when several readings end at one step): durations 19% of
  the differing tokens, tuplet tokens 19%, pitches 13%, voices and chords 12%, slurs, articulations
  and ties 18%. 38% differ by one token and 75% in one bar; 43% change some bar's rhythm; 54 are
  the same once our own tokens are dropped. So they are the best reading with one uncertain
  decision changed: they can mend a single wrong decision, not a run of them, which is what
  compounding leaves. Of the bars the best reading gets wrong, another reading has right about a
  quarter of the plain ones and a tenth to a fifth of the tuplet ones (beam 4 / 8).
- **The ceiling is low because the readings are alike**: even the best of four is 84%, far from the
  92% the model reads fed the right history. A reverse model reranking the same readings is bound
  by the same oracle. So with this beam, most of the compounding is out of a reranker's reach; it
  has to be fixed in training ((n)), or the beam made wider or more varied (beam 8 checked next).
- **Beam 8 against beam 4** (the first 300 crops, so higher rates; 17 minutes each):

  | Bars read right | Beam 4's best | Beam 8's best | Filling most bars, 4 / 8 | Any, 4 / 8 | Fed |
  |---|---|---|---|---|---|
  | Plain bars (944) | 85.1% | 84.9% | 84.7% / 84.4% | 88.1% / 89.3% | 93.9% |
  | Bars with tuplets (154) | 62.3% | 63.0% | 62.3% / 64.3% | 66.2% / 69.5% | 77.9% |

  A wider beam raises the ceiling slowly (1.2 and 3.3 points) and its best reading not at all; bar
  fill picks no better than the beam's own order on plain bars (about half of wrong bars still add
  up, and an alternative that fills is often wrong too). A better judge of the same readings, a
  reverse model, has 4-6 points of room at beam 8; worth training only if (n) leaves a gap.
- **Longer training, greedy**: `l_system_150k` at 120,000 steps (Polish 6.73, the best so far)
  against `l_system_100k` at 100,000 (7.44): plain bars right fed 92.6% / 91.7%, on its own
  83.3% / 80.1%; tuplet bars fed 78.6% / 70.8%, on its own 58.4% / 53.9%. Of its wrong bars 60%
  (plain) and 52% (tuplets) are still the history's doing. Training longer lifts both readings
  about alike; the compounding share stays.
- On Polish dev (two-staff pages, authors' alignment), beam 4 against greedy: print gold 7.41
  against 7.44, corrected 11.55 against 11.89. Within page noise on print; a third of a point on
  corrected. Not worth its cost on its own.

## What goes wrong in a wrong bar (2026-11-15)

Beam 4's best readings of the 615 crops (`compounding_beam4_readings.jsonl`; as greedy, 80.1% of
plain bars and 55.5% of tuplet bars right), each wrong bar against the label bar it pairs with, by
its timed events (notes and rests, a chord's later notes not counted):

| Wrong bars | Plain (388) | Tuplets (135) |
|---|---|---|
| Timed notes or rests missing | 56% | 36% |
| Timed notes or rests added | 9% | 7% |
| The same events; a chord note missing or added | 6% | 6% |
| The same events; lengths, order or voices wrong | 29% | 51% |
| ... of which the same notes, staves and order, only lengths wrong | 11% | 27% |

- **The commonest error is a missing event, not a wrong length.** Of 552 events missing in those
  bars, 243 are notes, 77 rests, and 232 rests the print does not show (`print-object:no`: the
  labels have 290, the readings 80). Counting a hidden rest as a forward, and a voice's trailing
  forwards as nothing, lifts bars right only from 80.1% to 81.4% (plain) and 55.5% to 58.8%
  (tuplets): mostly the reading writes no forward either, so the voice's next note starts early.
  A real timing error, though the missing event is invisible; the charge for hidden rests as such
  is 1-3 points.
- **It is polyphony.** By the number of voices in the label bar (staves count as voices):

  | Voices in the bar | Bars | Right | Events missing | A whole voice gone |
  |---|---|---|---|---|
  | 2 (one per staff) | 1,029 | 87.3% | 4.2% | 1 of 43 |
  | 3 | 984 | 73.3% | 14.3% | 27 of 141 |
  | 4 | 71 | 42.3% | 31.0% | 12 of 22 |
  | 5 or more | 230 | 56.1% | 26.1% | 34 of 60 |

  With one voice per staff the model reads 87% of bars right; a second voice on a staff costs
  14 points, and the missing events come with it: an inner voice's entry after a rest it leaves
  out, or the voice left out whole.
- **For a duration predictor solved per bar** (lengths chosen jointly so each voice fills the
  bar): its ceiling is the bars with the right events and only lengths wrong, 11% of the wrong
  plain bars and 27% of the wrong tuplet bars. Worth it for tuplets at most; the larger target is
  voices: the events a second voice on a staff holds, and when it enters.

## What the best model gets wrong, by root cause (2026-11-15)

`l_system_150k` at 120,000 steps (Polish dev 6.73, the best), greedy, on OLiMPiC scanned dev (615
crops, 2,314 bars). `runs/overnight_2026-11-14/bar_causes.py` (output `bar_causes_best_120k.txt`)
sorts each bar that differs from its label into the first cause that fits, voice by voice; a
hidden rest counts as the forward it is, and a bar with the label's tokens in another order
(chord marks) as right. 973 bars differ (38% of plain bars, 71% of bars with tuplets, marks
counted). Examples checked by eye (`bar_image.py` cuts a bar out of its crop).

| Root cause | Bars | Share of wrong bars | What it looks like |
|---|---|---|---|
| **Arcs: a tie read as a slur, or back** | 129 | 13% | the arc is found, its kind is wrong; often a tie between chords or between voices |
| **Arcs: a slur's ends, or an arc missing or added** | 177 | 18% | a long slur split in two, or ended a note early |
| **Second voices on a staff** | 225 | 23% | an inner voice's entry placed an eighth early (it must line up under the other voice's notes); a voice left out (dev crop 27); notes moved from one voice to another |
| ... the same notes at the same times, voices split otherwise | 50 | 5% | not visible on the page: a cross-staff arpeggio's voice, or an invisible whole-bar rest put on the other staff (dev 88, 97) |
| **Articulations** | 104 | 11% | marks tight against the notehead missed (tenuto dashes on chords, dev 17), accents added |
| **Pitch** | 101 | 10% | a step off (ledger lines; a note beamed across from the other staff, dev 91), an accidental missed or added, an octave |
| **Notes or rests added or missing, one voice** | 101 | 10% | an extra rest in a sparse bar of rests (dev 87: four rests read as five); fast runs cut short |
| **Tuplets** | 30 | 3% | ratio missing or wrong (8 more only spell the same lengths otherwise: 3in2 groups as 6in4, numbers unprinted, dev 94) |
| **Other lengths** | 46 | 5% | a dot missing or added, a value off; a cadenza's small notes (dev 84) |
| **Chord notes, grace notes** | 41 | 4% | a chord note missing or added |
| Other marks, bar count | 70 | 7% | |

On Polish dev (49 two-staff pages, print gold, `note_errors.py`, `note_errors_best_120k.txt`), of
about 16,000 gold notes: missing 421, extra 335, pitch (not an octave) 252, onset 149, duration
139, octave 57.

- **Most errors are real**: what the page cannot decide (voices split otherwise, tuplets spelled
  otherwise, hidden rests) is about 6% of the wrong bars.
- **Three large causes**: arcs (31%, of which a third are ties read as slurs or back), second
  voices on a staff (23%), and small marks (articulations, 11%). Pitch and lengths are 10% and 8%.
- **Checkable by arithmetic or geometry**: a tie joins two notes of the same pitch (a slur rarely
  does); a voice fills its bar, and an inner voice's entry lines up with the other voice's
  onsets; a bar of rests adds up. Durations alone are a smaller target (Lengths and tuplets 8%).
- Not yet ranked by what OMR-NED charges each cause (Polish's own breakdown is by note kind).

## What OMR-NED charges the best model for, by root cause (2026-11-15)

`l_system_150k` at 120,000 steps on Polish dev's 49 two-staff pages, against print gold, as
`polish_at.sh` scores them (authors' alignment, the ambiguous marks' refund; the ottava folding
left out, which moves a few pages). `runs/overnight_2026-11-14/ned_costs.py` records every edit
musicdiff makes, its cost and where it falls (`ned_costs_best_120k.jsonl`; its total, 8,939, is
the pages' distance exactly); `ned_causes.py` sorts them. musicdiff edits the reading into the
gold: a `noteins` is a gold note the reading lacks, a `notedel` a note it added. In each staff's
bar, a deletion and an insertion are read as one note misread: the same note at another onset or
in another voice, the pitch wrong, the length wrong. Of the gold's 2,156 staff-bars, 28% hold two
or more voices, with 38% of the notes.

| Root cause | Share of OMR-NED's cost | Of which in staff-bars with 2+ voices |
|---|---|---|
| **Notes missing** | 21.7% | 15.3% |
| **The right note at the wrong time** | 18.1% | 6.6% |
| **Bars read so badly musicdiff replaces them whole** (58, on few pages: 110, 042, 093 hold 57%) | 16.4% | - |
| **Extra notes** | 11.9% | 7.8% |
| **Pitch wrong** (with or without the length) | 11.6% | 5.6% |
| Slurs and other spanning marks (musicdiff's extras) | 6.6% | |
| Beams | 4.2% | |
| Lengths: notehead, dots, tuplets | 3.1% | |
| Accidentals | 2.1% | |
| Articulations and expressions | 2.0% | |
| Ties, grace notes | 1.7% | |

- **What moves notes in time is mostly a note before them left out**: in the staff-bars where
  notes are charged for the wrong onset, the same bar also lacks a note (26% of that cost) or a
  rest (24%), has a length or dot wrong (20%), or nothing else wrong on that staff (25%: the shift
  comes from elsewhere).
- **So leaving notes out is the largest cause**, about 30% of the cost with the shifts it brings,
  and most of it on staves with two voices: there notes go missing at four times the rate of
  single-voice staves (15.3% of the cost on 38% of the notes, against 6.4% on 62%). Per note,
  two-voice staff-bars cost about twice as much in all.
- Pitch is 12%; lengths as such, with the shifts they bring, about 7%; arcs, beams and marks
  together 16%.
- **Whole bars replaced** are bars read far from the gold (bar counts match the gold on each of
  those pages); not yet looked into page by page.
- On OLiMPiC dev the same model's wrong bars were mostly arcs and marks (above), but OMR-NED charges
  a missing note by all its parts, a slur by one: by cost, notes left out come first.

## Two voices on a staff: what goes wrong (2026-11-15)

The best model (`l_system_150k` at 120,000) on Polish dev's two-staff pages (print gold; musicdiff
skips what is marked invisible, and pairs voices in whichever order costs least, so voice order
and hidden rests are not charged) and OLiMPiC scanned dev; music21 for the Polish bars.

**Findings**
- **Voices are mostly counted right.** OLiMPiC dev: a staff-bar with two voices is read with two
  95% of the time (one: 3.9%); with three, the count is right 44%. The voices dropped are short
  (one or two notes; 24 of 59 open after a silence). On Polish, three voices are rare (0.7% of
  staff-bars; two voices 27%).
- **What goes missing is mostly the second voice's notes and rests, after the bar's start.** Of
  the notes and rests left out on two-voice staves (after pairing misreads): 61% in the second
  voice, 46% rests, 70% after the first beat. Extra notes there are 93% notes, 86% after the first
  beat; not doublings (28 of 452 repeat a note the gold has once), and the notes two voices share
  (one head, two stems: 320 in the gold) are read twice as they should be (285).
- **A second voice's printed rests are left out.** Of the gold's visible rests in two-voice
  staff-bars, the reading has a visible rest at the same onset for 65% (single-voice staves 89%);
  for 27% it has only the other voice's note there, the rest not written; 7% an invisible rest.
- **And its entries are misplaced.** Page 096 bar 4: the gold's lower voice is a half rest, then
  D-flat 4 on beat 3; the reading writes a hidden rest of one beat, so D-flat 4 comes on beat 2.
  Every later note of that voice is then charged as at the wrong onset.
- **Unprinted triplets in one voice read as plain values** (page 096 bars 4 and 14, 056 bar 6:
  triplet eighths or 16ths written as eighths or 16ths), moving every later onset.
- **Training sources disagree on how a second voice's gaps are drawn.** In the later voices of
  two-voice staff-bars (3,000 systems a source), gaps are visible rests in 79-85% of cases in the
  Verovio renders (pdmx, musescore, beams, marks, smashcima) and GrandStaff, 64% in `unmarked`, but
  43% in OLiMPiC (MuseScore users hide them; the images agree), which is weighted 2 in the mix and
  has the most polyphony (25% of staff-bars against 14-16% elsewhere).

**Hypotheses, with a test each**
1. *The second voice is written far from where it was seen.* LMX writes one voice through the bar,
   then backs up for the next, so the decoder must return to the bar's start and re-find what it
   skipped. Test: cross-attention while writing a second voice: does it go back to the bar's
   start, and is it weaker on bars where the voice is dropped or misplaced? (No training.)
2. *Placing a second voice's entry needs the other voice's time grid.* A forward's length is read
   from where the entry lines up under the first voice; the counting head counts only the current
   voice. Test: error rate of forward lengths against whether the entry lines up with a first-voice
   onset; probe whether the decoder state at a forward knows the first voice's onsets.
3. *A second voice's rests are optional to the model* because OLiMPiC, its largest polyphonic
   source, mostly hides them. Test: the same rest check on OLiMPiC synthetic dev against a PDMX
   (Verovio) dev set: if rests are kept on Verovio renders and dropped elsewhere, it is learned
   per source and leans on OLiMPiC's convention.
4. *Real engraving lays two voices out differently from Verovio* (rests displaced up or down,
   stems crossing, voices sharing heads), and most of the polyphony in training is Verovio's.
   Test: two-voice bar accuracy on OLiMPiC scanned (MuseScore engraving) against its synthetic
   twin (same music, MuseScore render) and against Verovio renders of the same scores.
5. *Unprinted triplets in polyphony* are the tuplet problem in another place: the beat is defined
   by the other voice. Test: hidden-tuplet accuracy in two-voice against one-voice bars.
6. *More polyphony on Polish than in training* (27% of staff-bars against 14-25%). Cheapest lever
   whatever the cause: weight two-voice systems up in the mix; test against (m) at 100,000.

**Hypotheses 1, 3 and 5 tested** (the best model; scripts in `runs/overnight_2026-11-14/`:
`voice_events.py` gives each event its staff, voice on that staff, onset and kind):

- **1, the second voice written far from where it was seen: not supported.** `voice_attention.py`
  (`voice_attention_120k.txt`), OLiMPiC scanned dev, labels fed. Averaged over layers the
  cross-attention is close to even, but layer 2's peak follows the notes: for first-voice events
  on two-voice staves it lands within 0.15 of the bar of the event's onset 92% of the time
  (correlation 0.91). At a later voice's first event it goes back just as well: 91% in bars the
  greedy reading gets right, 92% in bars it gets wrong (layer 3: 91% / 80%). Fed the backup, the
  decoder looks back to where the voice begins. What this does not test: whether it decides to
  write the backup at all.
- **3, the model treats a second voice's rests as optional: not supported.** `rest_keep.py`: of a
  later voice's printed rests, the reading keeps a rest on that staff at that onset for 94% on
  Verovio renders (300 polyphonic pdmx training systems), 99% on MuseScore renders (300
  olimpic_written training systems), 93% on OLiMPiC scanned dev; single-voice staves 91%, 97%, 93%.
  The rests drop only on Polish (65% against 89%). Everywhere it also draws a rest in about a fifth
  of the gaps a later voice leaves unprinted (19-23%).
- **5, unprinted tuplets harder in polyphony: not supported on OLiMPiC dev.** `tuplet_voices.py`:
  tuplet bars read right, numbers printed 56% (one voice, 64 bars) / 48% (two or more, 69); a
  number hidden 72% (88) / 78% (27). Too few hidden-and-polyphonic bars to rule it out, and Polish's
  gold does not say which numbers are printed.
- **What the Polish drop is, by eye** (`polish_bar.py` cuts a bar of a page; the page reader's
  layout gives the systems): both kinds. Some gold rests are not printed at all, though the gold
  calls them visible (kern marks an unprinted rest `yy`): page 096 bar 4's lower-voice half rest,
  page 095 bar 1's bass second voice. Others are printed, but where real engraving puts a lower
  voice's rests, displaced below the staff (page 055 bar 1, the bass); Verovio and MuseScore
  renders draw them otherwise. The reading's timing errors stand either way (096 bar 4: the lower
  voice's D-flat on beat 2 for beat 3).

So the multi-voice loss on Polish is not a decoder that fails to look back or a convention learned
from OLiMPiC; it is in part the gold (unprinted rests marked visible, charged by musicdiff), in part
real engraving's layout of two voices (hypothesis 4), and in part second-voice entries timed wrong.
Next: count the gold's unprinted-but-visible rests on two-voice staves (a correction for
polish_overrides, from the image); then test 4 against real-engraving training data.

**The gold's rests on split staves, checked against the print (2026-11-15).** `rest_audit.py` lists
every rest Polish dev gold draws (no `yy`) on a staff split into two or more voices, from the gold
alone (no reading chooses which are looked at): 439 rests on 34 pages, lines already rewritten by
an override left out. Five agents each checked a batch against the page images (`audit_view.py`
boxes the bar on its system; the barline finder is sometimes a bar off, and the agents placed
those bars by the gold's notes). Verdicts (`rest_audit_verdicts_*.jsonl`): 242 printed, 181 not
printed, 16 unsure; of the unsure, 11 are two silent voices under one drawn rest (one of the two
gold rests is not printed) and 5 cross-staff passages on page 056 (left as they are). Three of the
"not printed" checked again by eye (037 bar 18, 055 bar 4, 060 bar 8): all three right.
- 192 `hide_rest` overrides (a new override: the rest of one field, at a line and column, becomes
  a rest not drawn; `hide_rests` hides a whole range) on 28 pages, print standard. Page 108 pads the
  most (21 of its 31). Where earlier overrides insert spine changes (030, 056, 064, 069, 077), the
  list's staff labels can be wrong and a few rests on split staves may have been missed; the
  overrides point at line and column and are not affected.
- Best model (`l_system_150k` at 120,000), print gold, authors' alignment, two-staff pages: **6.73
  to 6.47** (whole-page 5.85). OMR-NED's cost 8,939 to 8,593: missing notes on two-voice staves
  1,368 to 969, extra notes there 698 to 741 (the reading draws a rest in some unprinted gaps,
  which the gold no longer matches). Two-voice missing notes are still 11.3% of the cost.
- Every run we compare, rescored on the corrected gold (`rescore_polish.sh gold_rests`: the same
  readings, print gold, authors' alignment, two-staff pages; T7's pages read again, its readings
  not kept): Seq2Seq 150k schedule at 100k / 120k / 150k 6.82 / 6.73 / 7.29 to 6.56 / **6.47** /
  7.03; staff 150k schedule 7.39 / 7.17 / 7.35 to 7.21 / 6.99 / 7.11; `l_system_100k` at 100k 7.44
  to 7.18; T7 at 100k 7.16 to 6.91. Seq2Seq gains a little more (0.26 against 0.18-0.25); the
  order and the decision stand. Scores from here on are against this gold.
- Not looked at: rests on staves with one voice, and the 7 three-staff pages.

**What "whole bars replaced" is (2026-11-15).** The largest single cause on the corrected gold
(17.1%, 58 bars), on few pages: 110, 042, 093 hold 57% of it. `ned_costs.py` now places a bar's
edit too. Pages 042 and 093: mostly small errors in passages that repeat. 042's bars 7-8 repeat
5-6; the reading drops two grace notes in bars 5 and 7 and misreads bar 6's rhythm (triplet 16ths
as a dotted 16th and 32nds), and has bar 8 right. 093's lower staff repeats bars 9 and 11; the
reading has one wrong pitch in each of bars 9 and 10. The authors' alignment first pairs bars that
match exactly: in a repeat it pairs the reading's right bar with the gold's earlier twin, and what
is left is deleted and inserted whole, at 20-54 a bar for an error or two. Page 110 is its dotted
`8` ottava passage (bars 23-27), which `ned_costs.py` scores without the ottava folding the
scorer allows (26.10 there against the score's 19.04; every other page agrees to 0.05).

With the page's bars compared as one block (`--whole-page`, as omr_ned.py's whole-page
alignment), the cost falls from 8,593 to 7,801 and the whole-bar cost from 1,470 to 569 (32 bars,
part of it page 110's ottava), while every note-level cause stays within a few percent. So about
900 of it was the alignment magnifying small errors. By cause, whole-page alignment:

| Cause | Share | 2+ voices / 1 voice |
|---|---|---|
| Notes at another onset | 20.7% | 7.5% / 13.2% |
| Notes missing | 20.4% | 13.1% / 7.3% |
| Extra notes | 15.0% | 9.6% / 5.4% |
| Pitch wrong (with or without length) | 13.4% | 6.3% / 7.1% |
| Whole bars replaced | 7.3% | |
| Slurs and other extras, beams | 9.7% | |

- **Polyphony is again the largest family**: the note-level causes on two-voice staves are 36.5%
  of the cost on 38% of the notes (one-voice staves 33% on 62%; per note 1.8 times).
- **Second: timing on one-voice staves** (13.2%), the compounding of a missing note or wrong
  length.
- The headline metric keeps the authors' alignment, so a small error in a repeated passage costs
  a whole bar or two there: fixing small errors in repeats is worth more than it looks.

**The four largest note-level kinds, regrouped by root cause (2026-11-15).** `ned_costs.py` now
records each edited note's context (grace, rest, chord size, clef, how far its outer pitches lie
from the middle line, an ottava over it, a tuplet); `note_causes.py` (`note_causes_best_120k.txt`,
whole-page alignment, corrected gold) pairs a bar's deletions and insertions as before and gives
every note-level edit a root cause: for a missing or extra note the first of ottava, other clef,
grace, a staff's later voice, rest, tuplet, chord tone, off the staff; for a misread pitch the
first of ottava or clef, octave, accidental, a step off the staff, a step on it; for a note moved
in time, the root cause of the earliest note missing or added before it on that staff in that bar.
Each is split by whether the staff holds two or more voices in the bar (kern's voice order is
arbitrary, so a dropped voice is as often the gold's first as its second). Note-level edits are
70% of the cost.

| Root cause | Share of the cost | Of which on staves with 2+ voices |
|---|---|---|
| Tuplets (a hidden triplet read as plain values, and what it moves) | 10.7% | 6.5% |
| A staff's later voice dropped, added or misplaced | 10.5% | 10.5% |
| Chord tones left out or added | 9.7% | 3.1% |
| Off the staff (ledger lines): left out, added, a step or two off | 7.1% | 2.4% |
| Notes moved by an error on another staff or in another bar | 6.0% | 1.2% |
| Pitch on the staff (a step, an accidental, an octave) | 5.8% | 2.7% |
| A plain note left out or added | 4.7% | 2.3% |
| A length or dot | 4.7% | 2.1% |
| Clef change or ottava | 4.4% | 2.5% |
| Rests | 3.5% | 1.5% |
| Grace notes | 3.0% | 1.7% |
| All note-level | 69.9% | 36.5% |

Seen by eye:
- *A low second voice below the staff, dropped whole* (page 012 bars 20-21: the bass's B1 half and
  whole notes two ledger lines down, stems down, a half rest drawn below; the reading has the
  upper chords right and no second voice).
- *Oom-pah bass, the voices merged* (022 bars 12-14: a held low note stems down under after-beat
  chords stems up, sharing its head with the first chord; the reading writes one voice and drops a
  chord).
- *Hidden triplets read as plain eighths* (096 bar 15, in the lower staff in treble clef; 042 bar
  6), every later onset moved.
- *Dotted rhythms misread* (051 bar 15, octave pairs: quarter and two 16ths read as a dotted
  quarter and two 32nds).
- *Chord tones left out*, on few pages: of 79 on one-voice staves, 30 on 051 (octave doublings)
  and 16 on 093.

So polyphony runs through it: half the tuplet cost and all the later-voice cost are on two-voice
staves, and every cause is a little worse there. The other large single causes are chord tones
(mostly one-voice, page-bound) and ledger lines.
- Also noticed (not fixed): page 097 bar 3, the gold's `8BBJ` may be printed an octave higher.


## Polyphony: three experiments toward a root cause (2026-11-15)

The best model (`l_system_150k` at 120,000), no training. Scripts in `runs/overnight_2026-11-14/`.

**1. Does it decide to start the second voice?** `backup_probe.py` (`backup_probe_120k.txt`),
OLiMPiC scanned dev, labels fed: at each `backup` that opens a staff's later voice (993), the
decoder's probability for `backup`, split by whether the greedy reading drops that voice (8.5%).
- Fed the right history, it chooses `backup` there for 86% of the voices it drops reading on its
  own (median probability 0.90), 98% of those it keeps. **It notices the voice**; the drop happens
  in free reading.
- What predicts a drop: entering after a silence (20% dropped against 5%), no notes at all (50% of
  16), one note (12%) against four or more (2%). The first voice's length: no effect.
- What becomes of a dropped voice (`voice_metrics.py`, chord notes counted): 36% merged (all its
  notes written in the other voice at their onsets), 22% partly, 42% lost.

**2. Is it the engraving?** `render_dev_verovio.py` draws OLiMPiC dev's 615 systems with Verovio as
our synthetic crops are drawn (`render_system`, random engraving options), with OLiMPiC's labels:
the same music and labels in three engravings. `voice_metrics.py`, bars read right:

| Same 615 systems | Bars with a two-voice staff | Bars without | Voices dropped |
|---|---|---|---|
| MuseScore, printed and scanned | 65.3% | 88.6% | 5.4% |
| Verovio | 66.0% | 94.3% | 5.7% |

(The MuseScore synthetic twin stopped at 130 systems on an unreadable image.) On Verovio's
engraving, the one our training draws most, one-voice bars get easier and two-voice bars do not:
**polyphony is as hard on clean renders as on scans**. The wrong two-voice bars go wrong alike in
both: a forward's length (an entry placed wrong) 20% / 18%, notes added to a voice 20% / 17%, a
voice missing 17% / 17%, added 13% / 6%, notes missing 12% / 12%; voice structure, not notes.
Second-voice rests are kept alike (94% / 93%). Real engraving (Polish) is other music and not
compared here; it may add to this, but it is not the main cause.

**3. Are Polish's textures rare in training?** `textures.py` (`textures.txt`): two-voice
staff-bars flagged held (a note held through two or more of the other voice's onsets: oom-pah
basses, pedal notes), late (a voice entering after a silence), together (shared onsets), else
independent; and whether a voice lies mostly off the staff.

| | Two-voice staff-bars | Held | Late | Together | Independent | A voice off the staff |
|---|---|---|---|---|---|---|
| Polish dev gold | 27.9% | 53% | 18% | 4% | 24% | 49% |
| olimpic_written | 26.0% | 45% | 18% | 11% | 27% | 59% |
| unmarked | 20.0% | 54% | 24% | 6% | 17% | 58% |
| pdmx | 16.6% | 38% | 27% | 16% | 18% | 53% |
| musescore / smashcima | 15.0 / 14.9% | 40 / 39% | 27 / 28% | 17 / 14% | 16 / 19% | 53 / 52% |
| grandstaff_written_v2 | 13.7% | 41% | 17% | 9% | 32% | 54% |
| musicorpus | 9.9% | 28% | 16% | 30% | 25% | 33% |

The textures are not rare: training's two-voice writing has Polish's mix. Only the amount differs
(Polish 28% of staff-bars, most sources 14-20%).

**4. Is the entry gap itself hard?** `entry_probe.py` (`entry_probe_120k.txt`), the same dev, labels
fed, bars with a two-voice staff: the decoder's first choice at each duration token. Right for
99.2% of note and rest durations (7,628), 92.2% of forwards inside a voice (141), 89.0% of the
forward that opens a later voice (109). Fed the right history, a gap is about ten times as likely
to be wrong as a note's length: not only compounding. Of the 12 wrong entry gaps, 10 would bring
the voice in under another of the staff's notes (the right kind of place, the wrong note), 2 at no
onset there. Few cases, a lean; it supports writing a voice's entry, and gaps in general, as a
position in the bar (an onset the decoder already knows, since it lines entries up with the other
voice's notes) rather than a length it must work out.

**So far:** the model sees the second voice; on any engraving it then loses or merges it in free
reading, or times its entry wrong (a forward's length), most of all when the voice enters after a
silence. Not perception, not a texture missing from training. What is left: how a second voice is
written (a whole voice after a backup, its entry as a forward's length, decided far from where
the voice's notes are), compounding in free reading ((n) tests that), and the amount of two-voice
training data. Experiments it points to: (n)'s two-voice bars against (m)'s; two-voice systems
weighted up; a voice's entry written as its position in the bar rather than a forward's length.

## (m) and (n): the meter, and training on its own readings (2026-11-15)

Against (l)'s `l_system_100k` (Seq2Seq, T7's data and 100,000-step schedule, no meter: dev 0.0717,
Polish 7.44), seed 0, Community RTX 5090s, `rhythm` branch at cd6a297, where every model reads
the meter in force (seq.meter; `--meter-context` retired):
- (m) `m_system_meter_100k`: (l) with the meter. Against (l): what the meter does for Seq2Seq
  (on the staff model, T8: tuplet tokens -5.4 per 100, durations -0.6, nothing else moved).
- (n) `n_system_meter_own_100k`: (m) with `--own-readings 0.25`: a share of the decoder's input
  tokens, rising to a quarter by mid-schedule, replaced by tokens sampled from its own reading of
  the batch (scheduled sampling, two passes). Against (m): whether training on its own mistakes
  cuts the compounding the counting probe points to.

Read by dev SER, tuplet bars, errors by kind of token (`category_errors.py`) and Polish dev at
60k, 80k and 100k.

**(m) at 100,000** (pod gone, exit 0), against (l) at 100,000 (no meter): dev SER 0.0676 against
0.0717 (50k / 60k / 80k: 0.0827 / 0.0764 / 0.0670); outer notes, both sets, 0.0635 against 0.0652;
tuplet-ratio tokens read 80.1% against 68.2%; tuplet bars right, numbers shown 62.9% against 51.5%,
a number hidden 69.7% against 68.5%; bars without tuplets 82.3% against 82.5%; false ottava tokens
189 against 362. The meter helps printed tuplets most and hidden ones hardly; nothing else
moves.

Polish dev, corrected gold (print, authors' alignment, two-staff pages), against (l):

| Step | 60k | 80k | 100k |
|---|---|---|---|
| (l) no meter | 8.90 | **7.97** | **7.18** |
| (m) meter | 8.91 | 8.54 | 7.63 |

Level at 60k, behind by 0.57 and 0.45 after: the meter does not help on pages. Where the page prints
a time signature (23 pages) both read it right (22) and (m) is still worse (7.45 against 6.79), so
not a wrong meter; second-voice rests explain little (kept 87% against 89%). Dev hands a meter model
the label's meter (even where none is printed), which pages never get; its dev gain is in part that.
The meter is off by default again (32a8995).

**(n)'s first start was broken** (12:23 PM to 1:50 PM, about 11,000 steps). Its training loss ran
1.4-1.9 above (m)'s and (l)'s from the first thousand steps, when under 1% of its inputs were
replaced. Cause: before torch 2.11, autocast keeps each weight's bf16 cast for the rest of the
step, and a cast first made under no_grad carries no gradient. The pods run torch 2.8 (the local
GPU 2.11, where the same code trained normally), so the second pass reused the sampling pass's
casts and the decoder's weights got no gradient. Fixed at 9acd037 (the sampling pass under its
own uncached autocast; a test that fails on torch 2.8 before the fix and passes after), and (n)
restarted from scratch on the same pod at about 2:00 PM; the broken run is kept as
`runs/n_broken_autocast`. (m) is unaffected: it has no pass under no_grad. Expected end about
11:30 PM.

**(n) at 100,000** (pod gone, exit 0), against (m) at 100,000: dev SER 0.0717 against 0.0676; outer
notes 0.0618 against 0.0635; tuplet-ratio tokens read 80.9% against 80.1%. Polish dev, corrected
gold, two-staff pages (authors' alignment as above; whole-page in brackets):

| Step | 60k | 80k | 100k |
|---|---|---|---|
| (l) no meter | 8.90 (8.06) | **7.97** (7.35) | **7.18** (6.61) |
| (m) meter | 8.91 (7.76) | 8.54 (7.29) | 7.63 (7.03) |
| (n) meter, own readings | 9.19 (8.38) | 8.15 (7.18) | 8.00 (7.11) |

Training on a quarter of its own readings does not help: against (m) it is ahead at 80k and behind
at 60k and 100k, and behind (l) at every step. Not promoted (`--own-readings` stays off by default).

Two voices, OLiMPiC scanned dev's 615 systems holding a two-voice staff, each run at 100,000
(`read_samples.py --polyphonic`, `voice_metrics.py`, `voice_metrics_lmn_100k.txt`):

| Run | Two-voice staff-bars dropped | Of those merged / partly / lost | Bars right, two-voice staff | Bars right, none |
|---|---|---|---|---|
| (l) | 8.6% | 29% / 29% / 42% | **60.7%** | **86.5%** |
| (m) | 8.4% | 30% / 32% / 39% | 59.3% | 85.5% |
| (n) | 9.7% | 34% / 31% / 36% | 58.3% | 85.4% |

Its own readings in training do not keep the second voice: (n) drops more of them, not fewer.
Compounding in free reading is not cured by scheduled sampling at this share. (m)'s better dev SER
does not show in bars read right either (with the label's meter handed to it).

## (o) and (p): how a second voice is written (2026-11-15)

From "Polyphony: three experiments toward a root cause": the model notices a staff's second voice
but drops, merges or mistimes it reading on its own, and an entry gap's length is ten times as
hard as a note's. Two label forms (`seq.label_form`, applied as labels load, taken out of
readings; round trip over 32,040 training labels from every source with no visible note changed),
each against `l_system_100k` (Seq2Seq, T7's data and 100,000-step schedule, no meter: dev 0.0717,
Polish 7.18 on the corrected gold), seed 0, Community RTX 5090s, `rhythm` at 32a8995:
- (o) `o_system_voice_counts_100k`, `--voice-counts`: after each `measure`, a `voices:N` per staff
  (how many voices it holds in the bar), so the decoder commits to them before writing any.
- (p) `p_system_canonical_gaps_100k`, `--canonical-gaps`: a return to the bar's start as one
  token, `backup:bar` (98% of backups; their length spelled in note values, often two tokens, is
  arithmetic the model need not do), and every run of gaps (forwards, rests not drawn) spelled one
  way, as forwards adding up to it.

The meter is off in both (and by default again): on Polish (m) reads worse than (l), 7.63
against 7.18 at 100,000, 7.45 against 6.79 on the 23 pages that print a time signature, where
both read it right (22 of 23); so not a wrong meter but the meter run itself, or one seed's
noise. On OLiMPiC dev a meter model is handed the label's meter, even on systems that print none,
which pages never get: its dev gain is in part that.

**(p) at 100,000** (pod gone, exit 0), against (l) at 100,000: dev SER 0.0692 against 0.0717. Polish
dev, corrected gold, two-staff pages (authors' alignment; whole-page in brackets):

| Step | 60k | 80k | 100k |
|---|---|---|---|
| (l) plain | 8.90 (8.06) | 7.97 (7.35) | 7.18 (6.61) |
| (p) canonical gaps | **7.64** (7.12) | **6.98** (6.31) | **6.85** (6.39) |

Ahead at every step, by 1.26, 0.99 and 0.33. Two voices on OLiMPiC scanned dev (as above, 615
systems, `voice_metrics_p_system_canonical_gaps_100k.txt`): two-voice staff-bars dropped 7.1%
against (l)'s 8.6%, and of those dropped, lost outright 26% against 42% (merged 40%, partly 34%);
bars right with a two-voice staff 60.0% against 60.7%, without 86.1% against 86.5%. Fewer voices
dropped, and those dropped mostly kept as notes in the other voice, which costs less on pages.
(`read_samples.py` now takes readings out of the model's label form; (l), (m), (n) are plain, so
their readings are unchanged.)

**(o) at 100,000** (pod gone, exit 0): dev SER 0.0742 against (l)'s 0.0717. Polish dev as above:

| Step | 60k | 80k | 100k |
|---|---|---|---|
| (l) plain | 8.90 (8.06) | 7.97 (7.35) | 7.18 (6.61) |
| (o) voice counts | 8.89 (7.76) | 7.60 (6.84) | 7.32 (6.48) |
| (p) canonical gaps | **7.64** (7.12) | **6.98** (6.31) | **6.85** (6.39) |

Mixed against (l) (level, ahead 0.37, behind 0.14 by the authors' alignment; ahead at every step
whole-page) and behind (p) at every step. Two voices on OLiMPiC scanned dev: dropped 8.3% (l 8.6%,
p 7.1%), lost outright 45% of those (l 42%, p 26%); bars right with a two-voice staff 61.3%, the
best of the five, without 84.7%, the worst. Counting a staff's voices first barely changes how many
it drops. Writing where a voice enters in one form (p) is what helps: (p) is the label form to
build on.

**(p) against (l): what changes.** Polish dev, OMR-NED cost by cause (`ned_costs.py --whole-page`,
`ned_causes.py`, corrected gold, 49 two-staff pages, at 100,000; `ned_causes_l_vs_p.txt`): 8,851
to 8,403.

| Cause | (l) | (p) | Change |
|---|---|---|---|
| Missing note, 2+ voices | 1,360 | 1,037 | -323 |
| Extra note, 2+ voices | 1,096 | 797 | -299 |
| Pitch and length wrong, 2+ voices | 335 | 215 | -120 |
| Missing note, 1 voice | 651 | 533 | -118 |
| Whole bar replaced | 856 | 806 | -50 |
| Same note, other onset, 2+ voices | 561 | 832 | +271 |
| Same note, other onset, 1 voice | 717 | 853 | +136 |
| Tuplet edits (edittuplet) | 86 | 152 | +66 |

(p) keeps notes on two-voice staves that (l) drops or replaces (-742), and gives back 407 by writing
some of them at the wrong onset, on one-voice staves too, and more tuplet marks wrong.

OLiMPiC scanned dev, the 615 systems holding a two-voice staff, bar by bar (`compare_bars.py`,
`compare_bars_l_p.txt`). Labels and readings are compared with their gaps spelled one way and gaps
that end a voice left out, as both draw nothing: the label's hidden rests are forwards in (p)'s
readings, and (p) leaves out the padding to the bar's end, which a token comparison counts against
it (`voice_metrics.py`'s bars right for (p) above are low for that reason). Of 2,352 bars, both read
1,614 right and 499 wrong; only (l) 106, only (p) 133 (bars right 73.1% against 74.3%).
- Where only (p) is right, (l)'s error: notes at other onsets 63%, lengths, rests or marks 16%, a
  voice dropped 9%.
- Where only (l) is right, (p)'s error: notes at other onsets 56%, lengths, rests or marks 26%
  (tuplet ratios: 3in2 left out or added, 3in2 written 6in4), a voice added 9%, a voice dropped 5%.
  These bars hold a tuplet more often (24% against 15% where (p) wins).

**(p) against (l): wrong onsets, tuplets, and the two-voice bars it newly reads right**
(`p_patterns.py`, `p_patterns_l_p.txt`; OLiMPiC scanned dev as above, compared as `compare_bars.py`;
Polish from the `ned_costs` edits).
- Where a voice's notes go out of place (dev, wrong bars, the first event out of place in a voice
  read with the right voice count): at the voice's entry, (l) 303 voices, (p) 101; after the event
  before it read another value, 159 against 187; after a tuplet's ratio missed or added, 56 against
  73. (p) places voices where they enter; it reads a few more values wrong.
- Two-voice bars only (p) reads right, against those both read right: the label's backup is spelled
  awkwardly (two tokens, dotted, or under a quarter) in 64% against 43% (46% where only (l) is
  right); a voice ends before the bar on 36% of their two-voice staves against 12%; the voices share
  every onset on 24% against 12%. (l)'s error there: notes at other onsets, from the voice's entry
  (later voice 41, first voice 24 of 66). What (p) gains is the backup's arithmetic.
- Polish, measures where only (p) writes notes at the wrong onset (44): there (p) costs 1,297, (l)
  500, so these are real losses, not notes (l) dropped. 39 of the 44 hold no tuplet in the edited
  notes; the worst are values misread (sixteenths as 32nds, a sixteenth rest as an eighth), the rest
  of the voice shifted after them, plus a few triplets left out or added. Whether a lost checksum
  explains it: no; (p)'s wrong dev bars still have every voice fill the bar as often as (l)'s (49.0%
  against 49.1%).
- Tuplets: tuplet bars read right 52.6% against 54.4% of 270 (5 bars); a ratio added in 33 wrong
  bars against 22; Polish tuplet edits +66 cost (0.8% of the total). No pattern beyond noise: by
  printed or hidden numbers and one or two voices, every cell is within 4 bars.

**OMR-NED cost by root cause for (p)** (`note_causes.py` on the whole-page edits, corrected gold,
49 two-staff Polish dev pages, both at 100,000; `note_causes_p_system_canonical_gaps_100k_100000.txt`,
`note_causes_l_system_100k_100000.txt`; the other edits grouped from `ned_causes_l_vs_p.txt`):

| Root cause | (p) | Share | Against (l) |
|---|---|---|---|
| Tuplets, notes (missing 478, extra 172, misread 242, moved 172) | 1,064 | 12.7% | +115 |
| A staff's later voice | 924 | 11.0% | -474 |
| A chord's note | 828 | 9.9% | -8 |
| Whole bar replaced | 806 | 9.6% | -50 |
| Text, dynamics and other marks (extras) | 619 | 7.4% | -31 |
| Off the staff | 459 | 5.5% | -53 |
| Moved, nothing wrong on that staff in the bar | 431 | 5.1% | +140 |
| Clef or ottava | 408 | 4.9% | +83 |
| Pitch on the staff | 388 | 4.6% | -31 |
| Beams | 385 | 4.6% | -8 |
| A rest | 356 | 4.2% | +49 |
| A plain note | 293 | 3.5% | -134 |
| A length or dot | 286 | 3.4% | +29 |
| Tuplet marks (brackets, numbers) | 226 | 2.7% | +72 |
| Noteheads, dots, grace marks | 225 | 2.7% | +37 |
| Articulations and expressions | 218 | 2.6% | +36 |
| Accidentals | 202 | 2.4% | -4 |
| A grace note | 181 | 2.2% | -203 |
| Ties | 104 | 1.2% | -13 |
| Total | 8,403 | | -448 |

Tuplets, notes and marks together, 1,290 (15.4%), are now the largest cause, ahead of a staff's
later voice; next chord notes, then whole bars replaced.

**Tuplet cost by whether the print shows the number.** Polish: the gold's kern marks hidden numbers
(`*Xtuplet`, `*tuplet`; converter21 sets tupletActualShow on a group's first note only), now in
`ned_costs.py`'s rows (`tuplet_numbers`) and split by `note_causes.py`; checked against the page
images on 003 and 037 (hidden, no numbers drawn), 065 (printed) and 096 (mixed, its hidden part).
Tuplet notes and marks, whole-page alignment, corrected gold, at 100,000
(`note_causes_*_tuplets.txt`):

| Polish bars whose tuplets | Bars | (l) | (p) | (p) per bar |
|---|---|---|---|---|
| hide the number | 147 | 583 (6.6%) | 658 (7.8%) | 4.5 |
| print it | 63 | 376 (4.2%) | 374 (4.5%) | 5.9 |
| no tuplet in the gold's bar | | 144 (1.6%) | 258 (3.1%) | |

OLiMPiC dev, all 615 clean systems, scanned (`tuplet_split.py`, the crop's MusicXML
show-number; token edits per bar as compare_bars.py compares them; `tuplet_split_l_p.txt`):

| OLiMPiC scanned bars whose tuplets | Bars | (l) right | (p) right | (l) edits | (p) edits |
|---|---|---|---|---|---|
| hide every number | 104 | 72.1% | **77.9%** | 900 (8.2%) | **614 (5.9%)** |
| print every number | 133 | 52.6% | 50.4% | 1,556 (14.1%) | 1,525 (14.7%) |
| some of each | 11 | 63.6% | 63.6% | 215 | 97 |
| no tuplet | 2,066 | 80.4% | 80.8% | 8,347 | 8,118 |

The synthetic twin stopped at 128 systems on an unreadable image (as before); on those, hidden 10
bars, printed 42: too few to read. Per bar, printed tuplets cost more than hidden ones on both sets
(Polish 5.9 against 4.5; OLiMPiC 0.114 against 0.043 edits per label token for (p)). Polish holds
more hidden-number bars than printed (147 against 63), OLiMPiC fewer (104 against 133), so Polish's
tuplet cost is mostly hidden by count. (p) reads hidden tuplets better than (l) on OLiMPiC and a
little worse on Polish; printed ones are level on both.

**How tuplets are misread** (`tuplet_kinds.py`; Polish from `ned_costs.py` rows that now record each
tuplet edit's group types and what is drawn, `tuplet_kinds_polish.txt`; OLiMPiC scanned dev, all 615
clean systems reread keeping tuplet:hidden, `tuplet_kinds_olimpic.txt`). In OLiMPiC, 21 systems
(60 bars, 242 notes) carry tuplets with no `<tuplet>` element at all: nothing drawn, so hidden.

Polish, cost of tuplet notes' edits and tuplet marks, with the reading's notes written in place of
a tuplet counted as well (wider than note_causes.py's root cause, which charges those to plain
notes):

| Kind | (l) | (p) | (p) hidden / printed / no gold tuplet |
|---|---|---|---|
| Rhythm: the notes read without the tuplet | 688 | 371 | 190 / 181 / 0 |
| Tuplet notes left out | 446 | 416 | 311 / 105 / 0 |
| Tuplet notes added | 281 | 292 | 43 / 17 / 232 |
| Length at the same onset | 322 | 242 | 157 / 43 / 42 |
| Moved | 140 | 171 | 22 / 149 / 0 |
| Number or bracket drawn otherwise | 80 | 146 | 116 / 30 / 0 |
| Pitch | 70 | 145 | 61 / 84 / 0 |
| Grouping | 74 | 80 | 36 / 30 / 14 |
| Another tuplet rhythm | 9 | 9 | 0 / 9 / 0 |
| Total | 2,110 | 1,872 | |

OLiMPiC scanned, label bars with a tuplet read wrong, (l) / (p): numbers hidden or not drawn at
all, 66 / 46 of 164 (ratio left out 37 / 33); printed, 63 / 66 of 133 (ratio left out 34 / 30,
another ratio 6 / 7, other pitches 6 / 8, notes missing or added 3 / 7, other onsets 7 / 6). Every
label group, its number as the print draws it: of 566 hidden, shown 44.0% / 55.1%, as printed
38.7% / 34.5%, group not read 17.3% / 10.4%; of about 360 printed, as printed 70.5% / 70.8%, hidden
4.5% / 6.6%, not read 25.1% / 22.7%.

The commonest misread is the tuplet not read at all, its notes written as plain values. Next, a
hidden number drawn: the models show a hidden group's number about half the time, and rarely hide a
printed one. Pitches are a small share. (p) reads hidden tuplets' rhythm better than (l) (Polish
504 to 190, OLiMPiC 66 to 46 bars wrong) but draws their numbers more often; printed tuplets are
no better. Counted this way, (p)'s tuplet cost is 238 lower than (l)'s, not higher.

**What a tuplet's notes are read as** (`tuplet_values.py`, `tuplet_values_l_p.txt`; OLiMPiC scanned
dev, all 615 clean systems; each label note under a ratio matched to the reading's note in the same
voice by pitch sequence). (l) / (p), of 2,880 / 2,911 notes: kept 77.6% / 87.0%; the same note value
without the ratio 16.5% / 7.8%; the same value under another ratio 4.1% / 2.9%; another value 1.7% /
2.4%. So a tuplet read as plain values is the same note shape written without its ratio (`16th 3in2`
to `16th` 153 / 120 times, `32nd 6in4` to `32nd` 264 / 16, `eighth 3in2` to `eighth` 24 / 46), not a
value chosen to fit the beat. "Another ratio" is mostly `16th 3in2` written `16th 6in4` (119 / 81):
the same lengths grouped as a sextuplet, a grouping and number error, not a rhythm one. `half 2in1`
read as `32nd` or `64th` (30 / 48) is probably two-note tremolos, unchecked.

## Is (p)'s tuplet gain one song? (2026-11-16, TODO step 1a)

Yes, on both sets, so "(p) reads hidden tuplets better" (above) is dropped until step 2's seeds
say otherwise. Saved readings of (l) and (p) at 100,000 steps, corrected gold, whole-page alignment.

- **OLiMPiC** (`tuplet_values.py`, now split by song; `tuplet_values_l_p_by_song.txt`): `32nd 6in4`
  read as plain 32nds, 264 / 16, is one song, 6801790, which (p) reads at 512 of 604 tuplet notes
  kept against (l)'s 248 of 612. Without it, tuplet notes kept are 87.7% / 87.6% (1,988 of 2,268 /
  2,020 of 2,307): no difference. The other common misreadings sit in two to four songs each
  (`16th 3in2` to `16th`: 6801790 and 6581327; to `16th 6in4`: 6259566).
- **Polish** (`tuplet_kinds.py polish`, now split by page; `tuplet_kinds_polish_by_page.txt`):
  (p)'s tuplet cost is 238 lower in all, and page 003 alone is 411 lower (431 to 20; then 088, 153
  lower). Without 003, (p) is 173 higher than (l): 096 +90, 110 +89, 087 +82, 077 +66, 069 +42.
  "Read without a tuplet", 688 to 371, is 003 216 to under 22 and 088 66 to under 18; 081 rose (47
  to 69).

Both of (p)'s large tuplet gains are a single score whose hidden groups it happened to read.
Whether that is the label form or seed luck is what step 2's seeds (running) decide.

## Why `unmarked` at weight 3 hurt Polish (2026-11-16, TODO step 1c)

T6 and the two staff baselines at 50,000 read again on Polish dev (their readings had not been
kept; `reread_1c.sh`, each report identical to the one saved), then `ned_costs.py --whole-page`
against corrected gold and `tuplet_kinds.py polish` (`tuplet_kinds_polish_t6.txt`).

- **One page holds most of it.** Edit cost T6 14,406 against the baselines' 12,100 / 12,948 (mean
  12,524): +1,882. Page 071 alone is +1,102 (72 to 1,174). Without it T6 is +780, inside the two
  baselines' own gap (+848); 18 pages worse and 14 better by more than 20. Next largest: 096 +514
  (more hidden tuplets read as plain notes: 423 against 150 / 118).
- **071 is a misread meter, made into tuplets by a page rule.** It is in 9/8. All three models
  misread the time signature: the baselines as 3/8, T6 as 6/8. T6's raw reading (page re-read
  with `seq_pages.py`, which now keeps `<page>.lmx.json`) has no tuplet ratio at all; its 119 of
  281 notes in tuplets are `imply_tuplets`: nine eighths in a bar taken as 6/8 fit as triplets,
  so the rule writes them over the whole page. In a bar taken as 3/8 nothing fits, and the
  baselines' pages are left alone. Tuplet notes added where the gold has none: T6 642 (071: 490)
  against 112 / 177.

So the `unmarked` crops did not teach the model to write tuplets where none are: T6 wrote no
ratio on 071, and without 071 its loss is seed-sized. What it shows is a weakness of the page rule.
One misread time signature makes `imply_tuplets` rewrite a whole page, and nothing checks whether
the meter the bars imply agrees with the one read. That belongs with 1b (the rules, each switched
off on saved readings). For step 3c: raising hidden numbers is not ruled out by T6. The 096 cost is
the open part.

## (p)'s invented tuplets: the model or the page rules? (2026-11-16, TODO step 1b)

(l) and (p) at 100,000 read again on Polish dev (`reread_1b.sh`, reports identical to the saved
ones), now keeping each system's reading before the page rules (`seq_pages.py` writes
`<page>.lmx.json`). `rule_ablation.py` rebuilds the pages with each rule off in turn (with every
rule on it matches `seq_pages.py`'s MusicXML byte for byte), `ned_costs.py --whole-page` scores
them against corrected gold, and `rule_ablation_report.py` compares them
(`rule_ablation_l_p.txt`). "Added" is the cost of tuplet notes where the gold's bar has none.

| Rule switched off | (l): cost change, pages helped / hurt, added | (p): the same |
|---|---|---|
| None (all on) | 8,851, added 224 | 8,403, added 232 |
| Implied tuplets | +2,277, 15 / 2, 110 | +1,133, 12 / 3, 80 |
| Wrong tuplet marks cleared | +122, 7 / 2, 273 | +514, 9 / 1, 318 |
| One beam too many dropped | +286, 6 / 1, 224 | +118, 3 / 1, 232 |
| A dot too many taken away | +113, 2 / 1, 224 | +138, 8 / 2, 232 |

- **Mostly the rule.** Of (p)'s 232, implied tuplets write 152; the model's own ratios leave 80,
  fewer than (l)'s 110. The rule writes more on (p) than on (l) (152 against 114), on pages it left
  alone before: 051 +30, 069 +27, 087 +20, 049 +12 (093 and 065 on both). That fits the TODO's
  guess that the rule's test (does a voice fill its bar?) misreads (p)'s voices, which leave out
  the padding to the bar's end. Not yet checked bar by bar.
- **Every rule still pays on both**, each by more than it costs: implied tuplets by far the most,
  but half as much on (p) (+1,133 against +2,277), which writes more ratios itself.
- With 1c (071: a misread 9/8 taken as 6/8, the page filled with triplets), the rule's two
  weaknesses are what it trusts: a voice's length in (p)'s form, and a meter read once.

Next: the rule taught (p)'s form (a voice that stops early is not short), and a meter the bars
contradict not used for implying. Both are fixes to the rule, with no training; measured as here.

## Implied tuplets only where another voice fills the bar (2026-11-16, kept)

From 1b and 1c: `imply_tuplets` made tuplets in bars where every voice overflows, which is a
barline missed (Polish 051 bar 22: a dot dropped and the next bar run on, the bass then made
triplets) or a meter misread (071, 9/8 read as 6/8), not a tuplet. Where a tuplet was read as
plain notes, another voice usually fills the bar (087 bar 4: the bass fills 3/4, the upper
staff's quintuplet read as plain 16ths). Now tuplets are implied only in measures where some voice
fills the bar (`filled`, as `dots_mended` already asks). Tried once, against a rule fixed before
(kept if the cost falls on (l) and (p), 003 worsens by no more than 5, and no correct OLiMPiC label
is newly changed). Saved raw readings at 100,000, corrected gold, whole-page alignment:

| | (l) | (p) |
|---|---|---|
| Edit cost | 8,851 to 8,753 (-98) | 8,403 to 8,270 (-133) |
| Pages better / worse by more than 5 | 2 / 0 | 3 / 0 |
| 003 | unchanged | unchanged |
| Tuplet notes added where the gold has none | 224 to 159 | 232 to 167 |
| Largest | 093 -78, 110 -20 | 051 -59, 093 -46, 110 -28 |

OLiMPiC dev's page gold through every rule: mean 0.454 to 0.021, pages changed 5 to 3 (the three
left are the beam rule's, 1.34 at most), none newly. 6259566's two pages are mended only because
their bars hold one voice: the meter bug (TODO 2a) stands. The cost: a bar whose every voice is an
unmarked tuplet (both hands in triplets, no number) is no longer repaired; none such moved on
Polish dev. It is a page rule chosen on Polish dev (the TODO's leakage note), small (about 1.5% of
the cost), and decided by a criterion fixed beforehand. Polish numbers from here on are on it.

## Printed tuplets whose ratio is left out, by eye (2026-11-16, TODO step 1e)

OLiMPiC scanned dev, label bars of kind "printed" read without their ratio: 31 for (p), 34 for
(l), 42 different (a subagent's look at every system, `scratchpad/look_1e/results.tsv`; six bars
taken as repeats of a figure already seen; `bar_image.py` fails on a system's last bar and often
misplaces barlines, so whole crops were used). For (p):

| What the print shows | Bars |
|---|---|
| No number at all: song 6801790 (16th groups, a rest then two notes, no 3 or 6 anywhere; 14 bars over both models) | 9 |
| Number far from its notes or in a slur (6011095's "4" at a long slur's top; 6629763's small "3" under stem-up chords; 6499366's tiny "3", read as grace notes) | 8 |
| Quarter-plus-eighth triplets, unbeamed, the "3" in a small arc (6656152, 6652787); the beamed group beside them read right; no sextuplets | 5 |
| Number printed once for a run (6259566; 6249298 p1-s4, treble only) | 3 |
| Tremolo (6249298 p1-s4 bar 3) | 1 |
| Other: a clear "3" after a beamed grace figure (6011095 p3-s3, 3 bars), a label error, one chord tone | 5 |

**Label errors.** 6801790's tuplets print no number, yet its MusicXML carries 20 `<tuplet>`
elements none marked `show-number="none"` and the as-printed labels no `tuplet:hidden` (checked by
eye here on p3-s4). By the print they are hidden-number bars counted as printed. So the song that
held all of (p)'s OLiMPiC tuplet gain (1a) is a hidden-number song: (p) wrote the ratio of
unnumbered 32nd groups there (512 of 604 notes kept, (l) 248 of 612). Also 6249298 p1-s4 bar 3
(`6in4` on a tremolo chord, nothing printed) and, less sure on a low-resolution scan, 6499366
p1-s1 bar 1 (bass: rest, A1, B1 labelled triplet eighths; the print looks like an eighth rest, a
grace note and an eighth).

Most of the rest is how the number is set (far up a slur, a small arc over unbeamed groups, once
for a run): training data with numbers set so (TODO 3c) rather than a page rule. Whether OLiMPiC's
*training* labels mark unnumbered groups as printed the same way is 1d(i)'s question.

## The whole bars musicdiff replaces, by eye (2026-11-16, TODO step 1g)

(p) at 100,000, corrected gold, whole-page alignment: 54 whole-bar edits costing 806 of 8,403
(9.6%). (The TODO's "58 bars, 110, 042 and 093 holding 57%" was the older best_120k model under
the authors' alignment; for (p) 042 and 093 have none whole-page. Authors' alignment for (p): 74
bars, 1,390, the extra pages being the repeat-pairing artifact already described.) A subagent's
look at every one (`scratchpad/look_1g/results.tsv`):

| What happened | Cost | Bars | Pages |
|---|---|---|---|
| Cross-staff notes: the reading puts them on the staff they are printed on, the gold on the hand's spine | 376 | 18 | 110 (322), 051, 047, 067 |
| A clef change missed (treble right after the key signature on the lower staff; every note a 10th low) | 153 | 6 | 081 bars 13-15 |
| Read badly (083 a repeated bar under 8va written once; 065 a grace merged; 064 a bar read as a whole rest) | 143 | 8 | 083, 065, 064 |
| A barline missed | 58 | 2 | 051 (handwritten) |
| Empty staff: the reading's `rest:measure` without its `rest` head (about 22 on dev), which converts to a pitchless zero-length note | 46 | 19 | 038, 047, 057, 062, 076 |
| Gold error | 30 | 1 | 076 |
| Segmenter | 0 | 0 | |

110: arpeggio triplets beamed across the staves (bars 23-27); the reading's tokens are right
(`staff:2` on each triplet's first note), the gold keeps the group in the upper spine. Suspected
gold errors, not yet checked here: 110 bar 25 (upper `12eeee-`, `12gggg-`, `12eeee-` printed on
the lower staff outside the 8va: an octave high), 076 bars 14 and 17 (chords in the wrong spine
for where they are printed), 065 bars 39-40 (printed two-note chords encoded as single notes),
083 bar 15 (low confidence: GG# may be F#2).

## Do tuplet-number labels match the training images? (2026-11-16, TODO step 1d(i))

A subagent's look at 20 crops a source, 10 with a `tuplet:hidden` group and 10 without
(`scratchpad/look_1d/results.tsv`, every group judged, none unclear):

| Source | Groups (labelled hidden / printed) | Hidden, number drawn | Printed, no number drawn |
|---|---|---|---|
| olimpic_written | 225 (116 / 109) | 0 | **73** (in 6 of the 10 crops without hidden groups) |
| pdmx, beams, round, marks, unmarked, smashcima | 672 | 0 | 0 |
| musescore | 123 (all printed, as built) | 0 | 1 (a group with no ratio token) |
| musicorpus (all 5 crops) | 54 | 0 | 0 |
| grandstaff_written_v2 | no tuplet in 29,500 labels | | |

Checked here by eye: `olimpic_written` `6012375/p2-s3` draws its right hand's 20 triplet groups
with no number; the label says all 20 are printed. OLiMPiC's training labels call 74% of their
20,901 groups printed; if the sample holds, a large share of those draw none, which teaches that
an unnumbered triplet group has a printed number: a likely cause of the model drawing hidden
numbers about half the time. Likely origin, to confirm: OpenScore's MusicXML exports do not
record a hidden number, while the renders (from the scores' own MuseScore files) and the scans
follow the edition. Song 6801790 in dev is the same (1e). Other oddities: tuplet groups with no
ratio token (`musescore 000018/s3`, `pdmx 012769/s0`, `smashcima 000443/s0`); Smashcima's
handwritten quarter rests that look like a 3.

## Two fixes to how readings become pages (2026-11-16)

- **The meter in force, per bar.** `lmx_tuplets.written_bar` gave a system's first time signature
  and every rule took it for the whole system; the meter carried into the next system was the
  first, not the last (OLiMPiC 6259566 alternates 6/8 and 2/4). Now `bars_in_force` gives each
  measure the signature written in it or before it (else carried, else `likely_bar`), and
  `bar_at_end` carries the last on (`seq_pages`). The three rules and `bar_fill` (which picks a
  crop's height) judge each measure by its own bar. `meters_in_force` keeps a system's own first
  signature, as `seq.meter` gives it in training, and carries the last. Tests: a change inside a
  system, a carried meter, and triplets after 6/8 to 2/4 (none by the old code, six by the new).
  Correct OLiMPiC labels through every rule: unchanged elsewhere (mean 0.021, the beam rule's three
  pages). Training: only `--meter-context` runs read the meter ((m), T8, (n)), and it is unchanged.
- **Hidden whole-bar rests in canonical gaps.** `with_canonical_gaps` took a hidden whole-bar rest
  (`print-object:no rest voice:N rest:measure`) for a gap and left `rest:measure voice:N` without its
  head; (p) learned that and writes it (20 times on Polish dev), and without a head it converts to
  a note with neither pitch nor length (1g's "empty staff", 46 of (p)'s cost). Reading now writes it
  back whole (`with_hidden_bar_rests_restored`; a drawn whole-bar rest keeps its `rest`, so a
  headless one is always hidden): an exact inverse, tested on both kinds. Every reading of a
  canonical-gaps model from here on is affected, OLiMPiC dev SER included.

## OLiMPiC's tuplet labels against the scores' own MuseScore files (2026-11-17, TODO 1h item 1)

Every OLiMPiC tuplet group (train and dev) paired with its group in the OpenScore Lieder `.mscx`
(commit 38c5db51; by measure number as MuseScore 3 exports it, then by order in the measure over
both staves; a subagent's measurement, `scratchpad/tuplet_size/`: script, `groups.tsv`, `unpaired.tsv`).

| Split | Groups | Paired | Labelled printed, score hides | Labelled printed, score shows | Labelled hidden, score hides | Labelled hidden, score shows |
|---|---|---|---|---|---|---|
| train | 21,089 | 20,912 | 9,771 | 5,908 | 5,233 | 0 |
| dev (synthetic = scanned labels) | 2,219 | 2,219 | 850 | 569 | 800 | 0 |

**The cause, confirmed:** the MusicXML export (which the labels come from) records a hidden number
only when MuseScore's `numberType` is 2 (on the tuplet or as the score's style default); a tuplet
made invisible (`<visible>0</visible>`, which hides number and bracket) exports as printed, often with
`bracket="yes"` (2,304 train groups). No exception in the paired groups. So 46% of all groups
(10,622 of 23,308) and about 60% of those labelled printed are wrong, in 152 of 351 train songs with
tuplets (5911190 alone 732) and 15 of 39 dev songs (4985990, 348 of 350). Unpaired, listed: 177
train groups pair by place with another ratio (nested or odd tuplets, 7 songs); one score group
with no OLiMPiC counterpart (6766045, edited since). The repo is newer than OLiMPiC: 37 songs
moved directory, 4 are MuseScore 4 files only (one with tuplets).

This is the label fault 1d(i) sampled (73 of 109), now sized: the model was taught, on its one
real-scan source, that most unnumbered groups have a printed number.

### The relabel, and the songs it leaves out (2026-11-17, decided with the user)

Built: `train/olimpic_tuplet_display.py` (pairing as above, the score deciding every paired group:
hidden means `show-number="none"` and `bracket="no"`, as `hide_tuplet_numbers` writes them), called
by `build_olimpic_written.py`, which now takes the Lieder repository (pinned in
`scripts/fetch_training_data.sh`, `$RAW/olimpic/lieder`, commit 38c5db51, the one OLiMPiC was made
from). New sets, the old ones kept: `seq516g_data/olimpic_written_v2` (train) and
`olimpic_written_dev_v2/{synthetic,scanned}`. Training keeps `olimpic_written` by default;
`NW_EXPERIMENT_OLIMPIC` points `staff_experiment.sh` at v2 for step 3c (i), so the baseline and the
relabel run differ by this one change.

Result: train 9,658 groups relabelled, 14,758 systems kept (14,907 before); dev 850 relabelled,
none left out, all 1,422 kept. Dev's labels differ from the old ones only by the 850 added
`tuplet:hidden` tokens (checked token by token).

**Why some songs are left out, not handled.** Asked how much work it was to handle every case
(the user wanted as much hidden-tuplet data as possible, since we read hidden tuplets badly):
the renamed directories and section-break numbering every song needs anyway, and they are built
in; the rest, a parser for MuseScore 4 files (one song with tuplets, 202 groups) and checking
nested or odd tuplets whose ratios differ by eye, was 2-3 hours for about 380 groups, under 2% of
the gain (hidden-number groups in train go from 5,233 to about 15,000 with the relabel itself).
So a song with any group that does not pair is left out whole, never relabelled in part, and named
with its reason in `left_out.<split>.tsv` and the build's output. Left out of train: 9 songs, 153
systems (1%):

| Song | Why |
|---|---|
| 5016466 | MuseScore 4 file only (Der Lindenbaum) |
| 5001945, 5052823, 5639135 | ratios differ (nested or odd tuplets: 35in16 against 5in4, 16in8 against 8in8, 9in4 against 3in2) |
| 5988112, 5989459, 6199572, 6451773, 6587611 | a group of n in n (4in4, 2in2, 3in3): it changes no durations, so its notes carry no time-modification and the label's ratio reads as none |

The last five could be kept by one rule (a label group without a ratio matches a score group of
n in n); not done, as it was not part of what was decided. Worth it only if the songs are missed.

By eye (docs/TRAINING_DATA_CHECKS.md; a subagent, seed 20261117, `scratchpad/look_relabel/results.tsv`):
30 crops, 318 groups, every one as the image draws it, none unclear: 20 train crops that gained
`tuplet:hidden` (20 songs, 261 groups, 13 of them still drawn), 5 train crops whose groups all
claim a drawn number (11), 5 scanned dev crops that gained tokens (46). Crops that mix the two put
the drawn number exactly where the new label does (6573780 p7-s1: the first triplet of each bar;
dev 6259566 p2-s2: the first three of twelve). Staff positions for v2 are those of the old set,
sample for sample. Ready for step 3c (i).

## Step 2: (l) and (p) over three seeds, on two sets (2026-11-17)

Every reading on the code of 2026-11-16 (meter in force per bar, implied tuplets only where another
voice fills the bar, hidden whole-bar rests restored): Polish dev read again overnight
(`runs/overnight_2026-11-14/overnight_fixed.sh`, 26 readings), scored on the Print standard as
corrected on 2026-11-17 (gold commit of that day: 110's cross-staff triplets and an octave in bar 25,
051, 047 bar 15, 076 bars 14 and 17, 065 bars 39-40, 083 bar 15; `rescore_polish.sh
gold_2026-11-17`); OLiMPiC's scanned dev pages (spec 2, first readings, `olimpic_at.sh`,
`--either-display` until dev is relabelled). Every Polish number before 2026-11-16 is on the old
code; 1a, 1b and 1c's pages hold no change of meter, so they stand.

Two-staff mean OMR-NED, whole-page alignment (authors' in brackets for OLiMPiC):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| (l) Polish, 100k | 6.48 | 7.12 | 6.33 | 6.64 |
| (p) Polish, 100k | 6.05 | 6.94 | 6.55 | 6.51 |
| (l) OLiMPiC pages, 100k | 5.51 (6.22) | 5.16 (5.70) | 4.67 (5.36) | 5.11 |
| (p) OLiMPiC pages, 100k | 5.11 (5.65) | 5.57 (6.62) | 5.27 (5.58) | 5.32 |

(p) − (l), seeds averaged per page, 95% range by resampling (Polish by page, OLiMPiC by song):
Polish -0.13 [-0.62, +0.36] at 100k, -0.28 [-0.83, +0.30] at 80k, +0.24 [-0.33, +0.78] at 60k;
OLiMPiC +0.20 [-0.08, +0.51] at 100k, +0.02 [-0.27, +0.30] at 80k, -0.17 [-0.72, +0.28] at 60k
(seed means: (l) 6.52, 5.37, 5.11 at 60k, 80k, 100k; (p) 6.35, 5.39, 5.32). On the 150,000-step schedule (seed 0) (p) is 5.87 at 120k against
(l)'s 5.75, and 6.05 against 6.28 at 150k.

**Verdict: (p) does not stand.** The rule (TODO step 2) asked for (p) below (l) at 100k with the
range clear of zero on both sets; it is inside the noise on both, and the other way on OLiMPiC.
Step 3 builds on (l). (p)'s single-seed leads of last week (1.26, 0.99, 0.33) were seed 0's: with
seeds 1 and 2 it is behind at 60k (+0.93 [+0.36, +1.53]). The tuplet gain was already one song and
one page (1a). OLiMPiC SER (crops) favoured (p) on every seed; page OMR-NED does not.

**The noise (spec 1, `eval/seed_noise.py`).** One seed's mean scatters, on Polish (49 pages, 100k),
by 0.42 (0.61 at 80k, 0.68 at 60k), more than page-by-page variation explains (variance ratio 1.85
pooled, chance 0.12; at 60k 3.25, 0.01: some runs are better on every page early on); on OLiMPiC
(158 pages) by 0.34, about what page-by-page variation explains (ratio 1.23; 0.26 at 80k, ratio
0.71; 0.65 at 60k, ratio 3.26, chance 0.01, nearly all of it (p)'s seeds, 5.59 to 7.26). A single-seed A/B
must clear about 0.6 on Polish (two seeds' difference) and 0.5 on OLiMPiC to mean anything at
100k, more earlier; three seeds a side bring a mean's noise to 0.24 and 0.20. Polish's noise sits
in 003 (36% of (l)'s), 110, 088, 093, 056; OLiMPiC's in 6801790 (whose tuplet labels are wrong)
and 6581327. Seed 0 ran on an older commit and on Runpod: at 60k and 80k (p)'s seed 0 is 1.6
better than its seeds 1 and 2, which agree with each other.

**Spec 2's checks** (OLiMPiC scanned dev pages: 158 pages of the 46 clean songs, five left out for
systems with no as-printed label): gold against itself 0; gold through the page rules 0.021 (the
beam rule's three pages); 20 pages by eye (a subagent, 15 songs, seed 20261117), each page's piano
systems exactly, in order; page OMR-NED against crop SER over songs, Spearman 0.87 ((l)) and 0.82
((p)). The songs out of rank are page effects, not gold: 6218737 p2-s2 read from the page at 20%
SER against about 3% for the song's crops; 6010916 p97, 5% SER on every system, costs 12.7. A
reading takes 9 minutes on the local GPU, not 50: only the clean songs need layouts.

## Baselines for step 3c (i), and the cost shares on the Print standard (2026-11-17, 1h item 4)

**OLiMPiC pages against the relabelled gold.** `eval/olimpic_pages_gold.py` now builds from
`olimpic_written_dev_v2` (tuplet numbers drawn or hidden as each song's score draws them);
`runs/overnight_2026-11-14/olimpic_pages_gold_v2`, scored with no `--either-display`
(`olimpic_rescore_v2.sh`, `olimpic_steps/RUN_100000/scores_v2`). Only 12 of the 158 pages change,
6 of them 6801790's: the clean songs hold few hidden groups (the relabel's dev groups sit mostly in
songs with a copy in a training source). Whole-page, 100k: (l) 5.55, 5.22, 4.73 (mean 5.17, was
5.11); (p) 5.16, 5.61, 5.32 (5.36, was 5.32). So OLiMPiC's pages barely measure a hidden number;
the relabel run is judged on it by headline, and on hidden numbers by Polish and by OLiMPiC's crops
(`tuplet_kinds.py`'s hidden column).

**OMR-NED cost by root cause, (l), three seeds pooled** (`ned_costs.py --whole-page`,
`note_causes.py`, `ned_causes.py`, `tuplet_kinds.py`; readings on the code of step 2a, Print gold
as corrected 2026-11-17, 49 two-staff pages a seed; `runs/overnight_2026-11-14/cost_shares_print/`).
Total 26,158 over 147 page readings (8,633, 9,216, 8,309 a seed).

| Root cause | Cost | Share |
|---|---|---|
| Tuplets, the wider count (`tuplet_kinds.py`) | 5,576 | 21.3% |
| Notes at another onset ("same note, other onset", 1 voice 2,933, 2+ voices 2,065) | 4,998 | 19.1% |
| A staff's later voice (`note_causes.py`) | 3,418 | 13.1% |
| A chord's note | 2,511 | 9.6% |
| Whole bar replaced | 1,712 | 6.5% |
| Off the staff | 1,593 | 6.1% |

(The rows overlap: a tuplet note at another onset counts in both.) Tuplets: hidden numbers carry
2,966 of the 5,576 (read without a tuplet 942, left out 925, length 599); 096, 003, 056 and 087
hold most of it. Whole bars replaced fell from 856 to 761 on seed 0 (corrected to Print gold), less
than expected: page 110 still costs about 350 a seed in whole bars, bars 22-27, the cross-staff
triplets the gold now places as printed. The readings do not, so it is now the model's cost, not the
gold's. **Step 3's order holds:** tuplets lead (3c, the relabel running), then notes at another
onset (3a, positions in the bar), then a staff's later voice (3b).

## Step 3a's checks with no GPU: positions in the bar (2026-11-17)

`seq.label_form.with_positions` / `with_gaps_from_positions` (tests in `train/test_label_form.py`):
every run of backups and gaps (forwards, rests not drawn, dotted or not) becomes one `at:P`, the
position in quarters where the next event starts; read back, a backup when P is behind and the next
event names another voice, a forward when ahead. A run is left as written where it cannot be read
back for sure: the bar's time unknown (after a whole-bar rest) or before its start, a backup
carrying more than its length, a voice backing up into itself (some GrandStaff labels, e.g.
Beethoven op. 27/2 ii bars 20-24), no way to spell the move. A position that cannot be placed in a
reading is read as no gap and returned (`UnplacedPosition`: bar, voice, position, where the voice
stood) for the caller to log and count. Over every training label of the mix (259,058, (l)'s form;
`runs/overnight_2026-11-14/positions_check.py`, `positions_check3/report.txt`):

- **Round trip:** every drawn event (bar, onset, voice, staff, tokens) the same in 259,057 labels;
  the one other is a PDMX hidden whole-bar rest (011186/s2), not looked into further. No position
  in a label goes unplaced.
- **Length:** 4.2% fewer tokens (59.1M to 56.6M); labels over 1,023 tokens 8 before, 6 after.
- **Vocabulary: 389 distinct positions, above the 200 agreed (so the user decides).** `at:0` is
  1,262,000 of the 1,299,000 (backups to the bar's start). Of the 37,000 others, on a grid of
  sixteenths and twelfths of a quarter under 16 quarters, 125 positions take 97%; the 256
  positions off it (1,008 occurrences) are PDMX's rounded durations (`at:2051/768`), and would stay
  written as backups and forwards. Eighths and twelfths: 94 positions, 96.4%.
- 127,000 backup and forward tokens stay as written.

## How OMR-NED charges errors, and what we steer by (2026-11-17, decided with the user)

**What the metric does.** OMR-NED, as the Sheet Music Benchmark paper defines it and as Transcoda's
code runs it (musicdiff 5.0.2, default detail, our installed version), pools each bar's notes and
pairs a reading's note with a gold note only at the same pitch and the same onset ("MusicDiff only
directly compares notes/rests that have the same pitch and are on the same exact offset within the
measure"). A note at another onset is deleted and inserted at full size. So one early rhythm error
is charged again for every later note it shifts. Built bars, through the authors' code
(`scratchpad` tests, 2026-11-17): a missing quarter rest before six eighths costs 38; the first
eighth read as a quarter 31; an isolated wrong pitch at the end 6. With musicdiff's Voicing detail
(each voice compared as a sequence) the same cost 2, 1 and 1. One quirk the other way: a bar whose
notes match one for one except in onset compares as equal before onsets are looked at, so a voice
entering a beat early, nothing else wrong, costs 0.

**Decision: the cost of an early error's aftermath is not treated as a model problem.** Edit
distance should mean "repair this one note and everything falls into place: cost 1". Notes that are
only shifted by an earlier misread say nothing about the model beyond that first misread, which is
bad luck of where in the bar it fell. So "notes at another onset" (19% of (l)'s cost) is not a
target in itself, and step 3a's positions form, chosen to stop that spread, is set aside: its
small form would mostly be (p)'s `backup:bar` again (`at:0` is 97% of its positions) and would not
touch shifts inside a running voice anyway. The code stays in `seq.label_form` (`with_positions`)
with its checks above. (Separate from the metric: `compounding.py` found the model also makes more
first errors after its own earlier ones, 92% of plain bars right fed against 80% reading on its own;
that is model behaviour, and is not what this decision sets aside.)

**The headline stays OMR-NED** (default detail, Print standard, whole-page and authors' alignment),
to stay comparable with published numbers. Paired page differences are read knowing that where in
a bar errors fall moves it. Diagnostics count first errors: `ned_causes.py`'s pairing, and a
Voicing column.

**Voices: a note in the wrong voice is an error, even where it renders the same** (the user; the
print decides what is drawn, such as hidden tuplet numbers, not how notes are grouped into voices).
OMR-NED charges none of it: a note swapped between voices 0, two voices in one rhythm read as
chords 0 (Voicing: 2 and 6). `eval/omr_ned.py --voicing` scores with musicdiff's Voicing detail
(not OMR-NED; the print-standard options checked to work with it). (l) at 100k, 49 two-staff Polish
pages, whole-page: 4.19, 4.31, 3.98 against OMR-NED 6.48, 7.12, 6.33
(`polish_steps/RUN_100k_fixed_100000/scores_voicing`).

**Not yet decisive: what "the right voice" is.** Bars where reading and gold hold exactly the same
notes but group them otherwise (`runs/overnight_2026-11-14/voice_disagreements.py`,
`voice_disagreements.tsv`): 98 of 4,650 staff-bars over the three seeds, many in all three. Ten
looked at (`voice_audit/`): the Polish gold groups by musical line (110 bar 8 splits E-flat 2 and 3
on one stem into two voices; 026 bar 7 puts up-stemmed D2-G2 with the down-stemmed D3 of the bass
line; a notehead two voices share is written in both), the readings by stem and layout; 003 and 067
are reading errors (a voice written twice). Until the definition is settled the Voicing column
is shown, not used to decide runs.

**How much voice writing the print leaves open, by page and by source**
(`runs/overnight_2026-11-14/voice_ambiguity.py`; `voice_ambiguity_polish.tsv`/`.txt`,
`voice_ambiguity_training.tsv`, training sources sampled 3,000 labels each). Of the notes in
staff-bars with 2+ voices: an "optional split" is a note sharing onset and length with a note of
another voice (drawable as one chord: 110 bar 8's kind); "doubled", one pitch at one onset in two
voices. Polish gold: optional 12.3%, doubled 6.1% (kern records no stems, so whether the print
draws one stem or two is not known from the gold). Most open pages: 051 (4 bars), 067, 083 (45-67%),
108, 060, 087, 110, 076, 022, 062, 096 (19-29%); none on 13 pages with multi-voice bars (105, 095,
070, 066, 005, 010, 064...), and 7 have no multi-voice bars. Training: optional 16-20% in every
source (OLiMPiC 20.0, GrandStaff 17.8, MuseScore 17.8, PDMX 16.7, Smashcima 15.5), 10-13% with both
notes' stems the same way; doubled 0.9-3.1%. So every source splits chordable notes across voices
about as often as the Polish gold; what the Polish gold does more is write a shared note in both
voices (2-6 times as often). Not measured: grouping by line where rhythms differ (026 bar 7), which
needs stems against the image.

**Decision (with the user): four Polish measures reported, voices revisited later.** Each Polish
reading is scored, Print standard, both alignments, by
`runs/overnight_2026-11-14/rescore_polish.sh` into `scores_GOLD/ours.MEASURE.ALIGNMENT.json`
(report `report_GOLD_measures.txt`):

| Measure | What it is |
|---|---|
| `print` | OMR-NED as published (the headline) |
| `print_voicing` | voice by voice (`--voicing`): wrong voices charged, an early error once; not OMR-NED |
| `print_either-voices` | OMR-NED with the voice choices the print leaves open allowed (`--either-voices`) |
| `print_voicing_either-voices` | voice by voice, open choices allowed: what the print decides about voices, charged |

`--either-voices`: wherever either page splits notes of one onset and length across voices, both
pages move all their notes there into one pooled voice, a chord with each pitch once. Neither the
split (110 bar 8) nor a shared note in both voices (060) is then charged, nor which voice such a
chord continues; voices elsewhere are. Tests: `eval/test_omr_ned.py` (110's bar and 060's cost 0;
a note filed in the wrong one of two voices of different rhythms still costs 4; a dropped voice 3).
Known undercharge: two voices the print draws with opposite stems in one rhythm (the kern gold has
no stems). Short-term these are reported, not steered by; the headline and `voice_metrics.py` still
decide runs. To revisit: which convention a voice follows (print or line), the training sources'
disagreement on writing shared notes, and an unpenalised statistic of agreement on open splits.

## What (l) and (p) get wrong, on the voice-aware measure (2026-11-17)

`runs/overnight_2026-11-14/causes_voicing/` (`ned_costs.py --whole-page --voicing --either-voices`,
then `categorize.py`; `categories.txt`): the 49 two-staff Polish dev pages, Print gold of 2026-11-17,
both forms' three seeds at 100k pooled, every edit by family. Totals 13,077 (l) and 13,011 (p): the
measure in `print_voicing_either-voices`, so an early error is charged once and the voice choices
the print leaves open are not charged.

| Family | (l) | Share | (p) | (p) − (l) |
|---|---|---|---|---|
| Notes missing (grace 457, in a tuplet 368, 2+ voices 332, rests 327, off the staff 228) | 1,935 | 14.8% | 2,030 | +95 |
| Markings (slurs 1,391; clefs, ottavas, key and time signatures, barlines) | 1,780 | 13.6% | 1,807 | +27 |
| Beams | 1,228 | 9.4% | 1,184 | −44 |
| A whole voice missing | 1,035 | 7.9% | 948 | −87 |
| Notes extra (rests 310, 2+ voices 293) | 909 | 7.0% | 1,017 | +108 |
| Tuplet missing on notes read otherwise right (hidden-number bars 522 of 756), or extra | 836 | 6.4% | 796 | −40 |
| Pitch wrong | 754 | 5.8% | 764 | +10 |
| Accidentals (402 written where the gold has none) | 718 | 5.5% | 724 | +6 |
| A whole voice extra | 692 | 5.3% | 590 | −102 |
| Chord tones missing or extra | 649 | 5.0% | 672 | +23 |
| Articulations and ornaments | 642 | 4.9% | 656 | +14 |
| Tuplet drawn otherwise (number shown against hidden 305) | 363 | 2.8% | 331 | −32 |
| Grace note kind (graceedit, slash) | 344 | 2.6% | 285 | −59 |
| Note value (head or dot) | 341 | 2.6% | 328 | −13 |
| Ties | 310 | 2.4% | 317 | +7 |
| Voice timing (the gap before a note) | 306 | 2.3% | 300 | −6 |
| Whole bar replaced | 235 | 1.8% | 262 | +27 |

- Charged once, rhythm proper is small: note values 2.6%, voice timing 2.3%. Most of what OMR-NED
  calls "notes at another onset" was aftermath.
- The largest model problems are notes left out (graces, tuplet notes, notes on two-voice staves,
  rests), voices left out or invented whole (13.2% together; 077, 056, 030, 051, 096, 069 most),
  and tuplets (9.2% with the drawn ones; the missing ones mostly where the number is hidden,
  096 and 003 most: the relabel run's target).
- Not notes: slurs alone 10.6%, beams 9.4%, accidentals the gold lacks 3.1%.
- Grace notes together (missing, extra, kind, pitch) about 7%.
- (p) against (l): fewer whole voices dropped or invented (−189), more single notes missing or extra
  (+203); the totals are the same.
- Caveat: rests are compared without visibility (Style is off), so a hidden rest against a drawn
  one, or a hidden rest the reading writes as a gap, may be charged as a rest missing or extra; not
  checked.

**The same on OLiMPiC's scanned dev pages** (158 pages, gold `olimpic_pages_gold_v2`;
`runs/overnight_2026-11-14/causes_voicing_olimpic/categories.txt`; totals (l) 19,450, (p) 20,862).
Agree with Polish (share of (l)'s cost, Polish then OLiMPiC): notes missing 14.8/13.4%, markings
13.6/11.2% (slurs most of both), a whole voice missing 7.9/8.2%, notes extra 7.0/8.4%, chord tones
5.0/6.1%, tuplets drawn otherwise 2.8/3.4%, note values 2.6/3.3%, voice timing 2.3/2.3%. Disagree:
tuplets missing or extra 6.4/11.7% (the lieder hide many triplets), articulations 4.9/9.0%, beams
9.4/3.9%, accidentals 5.5/2.7%, whole bars replaced 1.8/5.6%, grace note kind 2.6/1.2%. (p) against
(l): fewer whole voices dropped on both (−87, −191), but on OLiMPiC more of nearly everything else
(notes missing +517, tuplets +226, beams +201, pitch +145, note values +130), 7% more in all.
Polish's larger beam and accidental shares may be its gold's conventions (kern beams, accidentals
the print shows that the gold omits); not checked.

**Polish's beam and accidental edits, by eye** (`runs/overnight_2026-11-14/beam_acc_audit/`, (l)
seed 0; `show.py` prints a bar's notes with displayed accidentals and beams, gold and reading). Both
are spread thin (135 bars with beam edits, 156 with accidental edits), and both mix gold and model:
- Beams. 065 bar 0: the print beams the first three chords (two beams); the gold has no beam on them,
  and the same edit recurs in bars 4, 10, 11, 18: gold. 076 bar 19: one beam runs over the whole
  32nd run; the gold beams its first half only: gold. 093 bar 4 (a manuscript page): the reading
  writes 32nds where the print has 16ths: model, and a rhythm error, since musicdiff tells a
  beamed note's value by its beams, not its head. So "note value (head or dot)" undercounts misread
  values: for beamed notes they are charged as beams.
- Accidentals. 012 bar 0 (sharps printed on F#, C# and A# in a key that has F# and C#) and 003 bar 10
  (a natural printed on B2): the gold leaves out accidentals the print shows: gold. 060 bar 24 (no
  accidentals printed on the bass chords) and 087 bar 10 (a natural read as a sharp): model.
Not yet counted; a sample of each by eye would size the gold's share.

**Sized: 50 bars by eye** (a subagent; `beam_acc_audit/sample/verdicts.tsv`, 25 bars with beam
edits and 25 with accidental edits drawn at random, seed 20261117, from (l) seed 0; two verdicts
checked by eye here, 069 bar 23 and 105 bar 13, both right). Gold wrong (reading right) 13 of 25 in
each family; model wrong 8 of 25 in each; both wrong 2 and 3; no real difference 2 and 1.
- Beams, gold: mostly beams across the staves left out (062 bars 1, 4, 5, 11, 12; 110 bars 23, 26):
  kern holds no beam across spines, and the move_up/move_down corrections drop such beams on
  purpose. Also groups the print breaks run together under one primary beam (062/5, 069/23,
  064/5), and runs left unbeamed (065/4, 076/19). Beams, model: mostly a different note value
  (017/24, 017/25, 065/7, 110/7, 081/13, 042/5), one real grouping error (093/4), one group dropped.
- Accidentals, gold: 9 of 13 are courtesy accidentals the print shows and the gold leaves hidden
  (062/11, 110/20, 25, 26, 103/3, 105/13, 039/22, 049/5, 037/17); others on the wrong note or chord
  (022/8, 030/20), one printed natural missing (066/4), one sign on a grace note with none (103/7).
  Accidentals, model: mostly naturals read as sharps (022 three times, 042/3, 055/18, 093/4).
- So about half of Polish's beam and accidental cost (1,946 of (l)'s 13,077 over three seeds) is
  the gold's: about 7% of the voice-aware total. This is why Polish carries twice OLiMPiC's share.
- Tooling: `polish_bar.py` cut the wrong bar for 6 of the 50 (062 bars 11, 12; 081/7; 039/22;
  055/18; 022/22); the subagent found them on the full page. The voice audit used the same tool.

## Polish dev gold reviewed whole, for beams and accidentals (2026-11-17)

The 50-bar sample (above) put about half of Polish's beam and accidental cost on the gold. Every
earlier correction started from a bar where a reading disagreed, so gold errors a reading shares
were never looked at. This pass reviewed every bar of the 56 dev pages against the print, for those
two families only (eight subagents, about 155 bars each; brief, tools, every finding and proposal in
`runs/overnight_2026-11-14/gold_review_2026-11-17/`; three fixes checked by eye here: 064 bar 5,
083 bar 5, 095 bar 9, all right).

Tooling:
- `eval/gold_listing.py`: a page bar by bar, with the kern lines as `polish_overrides.jsonl` numbers
  them, then what the scorer reads (accidentals drawn or hidden, beams per level).
- `gold_overrides.py`:
  - New `beams` op: sets one field's beam marks.
  - `show_accidental` on a note written without an accidental now writes kern's `n`. `X` alone is
    ignored by the scorer's parser, which before this silently did nothing. No existing override
    hit that case: the print gold is byte-identical.
  - No op can hide an accidental. The parser ignores kern's `y` and draws whatever the key and bar
    call for. music21 also reads `k` and `K` both as a right-pointing hook. Both are recorded in
    findings, not fixed.
- `polish_bar.py` now cuts the whole system rather than finding barlines from dark columns, which
  failed on faded or skewed scans. It warns when the reading's bar count differs from the gold's.

Corrections added: 185 for beams and accidentals, plus 12 from a pass over the other errors found:
- **Accidentals:** 94 `show_accidental`, mostly courtesy accidentals printed and left hidden.
- **Beams:** 90 `beams`. Groups run together (25 bar-staffs) and runs left unbeamed (9).
- **054:** one `end_after`. Gold's last bar opens the next page.
- **Wrong pitches, fixed with `replace`:** 038/10, 065/1, 065/25, 083/10, 088/5, 108/11 and 076/2,
  all on the corrected standard.
- **Rhythm:** 095/9, as in bar 5.
- **Grace notes:** 070/16 added; 071/6 as a grace, not a 16th (print standard).

Left alone: 098/14, where the claimed dotted rhythm isn't on the print, and pitch claims too
blurred to read (066/15, 069/24, 093/1, 108/5, 110/19).

Not fixable in kern: 38 beam groups that join notes on both staves, on 11 pages (062: 11, 110: 7,
067: 6). See below.

Two-staff mean, whole-page alignment, (l) and (p) over three seeds at 100k (`compare_golds.py`).
The gold tags are 2026-11-17 (before), b (beams and accidentals) and c (plus the pitch fixes):

| gold | model | OMR-NED | voice by voice | either voices | voice by voice, either voices |
|---|---|---|---|---|---|
| before | (l) | 6.64 | 4.16 | 6.46 | 3.94 |
| before | (p) | 6.51 | 4.11 | 6.37 | 3.94 |
| c | (l) | 6.50 | 3.99 | 6.32 | 3.77 |
| c | (p) | 6.37 | 3.94 | 6.22 | 3.77 |

Every measure falls by 0.14 to 0.17 for both models, and (l) against (p) does not move. On the
voice-aware measure that is 4% of the total, short of the 7% the sample suggested.

Recall: the whole-page pass caught only about 7 of the sample's 13 accidental gold errors. It missed
110 bars 20, 25 and 26 (courtesy flats on bass chords under an 8va), 030/20 and 066/4. So a second
pass looks only at staff-bars where a seed of (l) disagrees with the gold on beams or accidentals and
the first pass recorded nothing: 194 staff-bars (`runs/overnight_2026-11-14/second_pass/`).

**Beams across staves need more than kern.** music21 keeps such a beam as per-note beam types on
each staff (lower: start, continue; upper: continue, stop), so a gold would need exactly those. Kern
can't write them: a lone J without its L makes the parser drop the bar's beams. A MusicXML stage
(kern, then music21, then MusicXML with the beams set) fails on its first test. Writing the
corrected gold to MusicXML and scoring it against the kern costs 0.07 on average, nonzero on 19
of 56 pages. music21's writer adds key, time and clef signs (51 of 96 cost units) and changes some
slurs, beams and arpeggios. So the stage would shift scores on pages with no cross-staff beam. The
other way is a scorer flag that patches beams into the parsed gold. Set aside for the user to
decide (TODO).

**Second pass, and the gold as it now stands (2026-11-17d).** Four subagents checked the 194
staff-bars (`second_pass/bars.tsv`, `gold_review_2026-11-17/pass2/`). The gold was right in 179;
the reading there was wrong.

Fixed:
- 12 courtesy accidentals.
- 3 beam runs (two grace pairs, among them 081/7; one unbeamed).
- 3 pitches: 066/13, where the sharp under the turn is the turn's own, so the note is B-flat by
  the key; 083/21; 103/7.
- Here, by eye: 110 bars 25-26, the 8va bass chords' G-flats and A-flat, which pass one had logged
  only for their cross-staff beams.

Left as recorded:
- 030/20: which note the sharp belongs to.
- 022/8, second chord.
- Beam levels and hook directions music21 sets from note lengths: 030/19, 087/13, 065/7, 065/12,
  065/13. A `beams` op cannot remove a hook a 16th's length implies.
- 096/14: a beam across the barline.

polish_overrides.jsonl now holds 3,459 lines. Two-staff whole-page means on 2026-11-17d:

| Run | OMR-NED | Voice by voice | Either voices | Voice by voice, either voices |
|---|---|---|---|---|
| (l), seeds 6.32 / 6.97 / 6.17 | 6.49 | 3.97 | 6.31 | 3.75 |
| (p), seeds 5.88 / 6.77 / 6.40 | 6.35 | 3.92 | 6.21 | 3.76 |

## Step 3c (i): OLiMPiC's tuplets relabelled from the scores (2026-11-17)

`l_system_relabel_100k`: (l), seed 0, 100k on Vast, with OLiMPiC train labelled v2, so tuplet
numbers are drawn or hidden as each song's score draws them. Judged against (l)'s three seeds. The
seeds were read again on `olimpic_written_dev_v2` (`tuplet_baseline_v2.sh`, `runs/l_system*_100k/dev_v2/`),
so both sides use the same labels.

**Headline, inside the seeds' noise.**

| Measure | Relabel | (l) mean | (l) seeds |
|---|---|---|---|
| Polish, gold d, OMR-NED | 6.71 | 6.49 | 6.17-6.97 |
| Polish, gold d, voice by voice | 3.95 | 3.97 | 3.79-4.12 |
| OLiMPiC pages against v2 gold, whole-page | 5.34 | 5.17 | 4.73-5.55 |

**Tuplets, scanned OLiMPiC dev crops** (`tuplet_bars.py`):

| Bars, read with the right rhythm | Relabel | (l) seeds |
|---|---|---|
| A number hidden | 44.5% | 53.0, 51.1, 52.0 |
| Numbers shown | 43.8% | 45.9, 43.8, 45.0 |
| No tuplets | 77.7% | 79.9, 78.1, 78.7 |
| Tuplet-ratio tokens read | 59.6% | 65.3, 65.9, 65.7 |

On the synthetic crops, hidden-number bars are 53.4% against 57.3, 53.9 and 48.2. The run marks
hidden numbers far better: `tuplet:hidden` is written for 1,176 of 1,650 hidden groups, against
503 for seed 0. But it writes fewer tuplets at all: 8,104 ratio tokens against 8,828, out of
11,710.

On Polish (`tuplets_polish_d/`, `tuplet_kinds.py`), tuplet cost is 2,215, against 2,077, 1,997 and
1,526. The hidden-number share is 1,243, against 1,451, 993 and 929. The "drawn" cost for hidden
numbers falls to 0, against 36, 107 and 33. Printed groups read without a tuplet rise to 365,
against 184, 213 and 162.

**Verdict: the relabel does not stand.** It teaches the display and loses the tuplets: on the
real-scan set it reads 7 to 9 points fewer hidden-number bars right, outside the seeds' 2-point
spread. One seed only, so the drop could be smaller, but it is not a gain. v2 stays out of the
default.

A guess at why, not tested: v1's labels called many unnumbered groups printed, which told the
model a tuplet is there whatever the image shows. With v2 an unnumbered group looks like plain
notes plus a hidden marker, and the model writes no tuplet more often. So 3c (ii), hiding numbers
in the synthetic sets as engravers do, risks the same, and should be measured on this
tuplets-written rate, not on display.

## Print fixes: what the print draws and kern cannot say (2026-11-17, decided with the user)

Kern cannot hold beams across the staves, and music21's kern reader ignores `y`, so the gold cannot
leave off a sign the print leaves off. A MusicXML stage would change scores elsewhere (above), so
the user chose a scorer flag, `omr_ned.py --gold-print-fixes FILE`. It sets corrections on the
parsed gold before comparing. The corrections are in `eval/polish_print_fixes.jsonl`, each naming
its notes by bar, staff, onset, pitch and, where needed, length.

There are two kinds:
- `beams`: each note's beam types per level.
- `hide_accidental`: the sign set to not drawn.

A note the file names that is not found, or found twice, fails the page. `ned_costs.py` and
`gold_listing.py` apply the same function. In `rescore_polish.sh`, each gold tag keeps the copy of
the file it was built with, so scores on older tags (17d and before) do not move.

Tested (`test_omr_ned.py`):
- A reading with a beam across the staves costs 0 against beamless gold with its fix applied.
- A reading that leaves off a sign the gold would draw costs 0 once the sign is hidden. musicdiff
  charges a sign only where its display is "drawn", and music21 reads a MusicXML note with no
  `<accidental>` as not drawn.

What the file holds:
- 71 beam groups on 12 pages (062: 26, 067: 14, 110: 15, and others), transcribed by two subagents
  from the print (`gold_review_2026-11-17/print_fixes_work/`).
- 12 hidden signs that the reviewers had already checked by eye.
- 12 kern staff moves that put the beamed notes on their printed staff (051, 060, 076, 081, 083,
  093, 110).

Not yet expressible:
- 030 bars 29 and 37, and 056 bar 17: `moved_down` cannot move notes out of an upper staff already
  split in two voices.
- 076 bar 18: one notehead with two stems.

Two-staff whole-page means, gold 2026-11-17e (17d plus the print fixes and moves):

| | OMR-NED | voice by voice | either voices | both |
|---|---|---|---|---|
| (l), seeds 6.02 / 6.70 / 5.91 | 6.21 | 3.70 | 6.04 | 3.51 |
| (p), seeds 5.64 / 6.50 / 6.15 | 6.10 | 3.72 | 5.96 | 3.57 |
| relabel run (3c i) | 6.49 | 3.74 | 6.36 | 3.66 |

Every measure falls by about 0.27, more than my estimate of under 2%. The models already read
these beams: (l) seed 0's 062 goes from 6.66 to 2.81. Every changed page falls except (p)'s 093
(+0.37). Neither (l) against (p) nor the relabel verdict moves. From here, runs are judged on 17e.

**`move_down` out of a two-voice upper staff (2026-11-17, with the user).** `gold_overrides.moved_down`
assumed the upper staff was one column, the last. It split the column left of that and took notes
from the last. With the upper staff split in two voices, the new voice opened inside the upper
staff and nothing moved. It now:
- finds each column's staff from the header through every split and merge (`staves_of_columns`);
- opens the new voice after the lower staff's last column;
- takes the notes from a column the override names (`"column"`), the last by default.

Interpretation lines in the range now leave an upper-staff clef on the upper staff. Before, the
new column took it; no existing override hit that, and the print gold is byte-identical. It fails
loudly on a voice split inside the range, and on a note held across the split, which makes the
scorer's parser read no parts at all (a merge across a held note it reads: 051).

Tests (`test_gold_overrides.py`) cover:
- staff tracking through splits and merges;
- a named and a default source column;
- a lower staff already in two voices;
- clef and barline lines in the range;
- both failures;
- an end-to-end parse with the moved notes on the lower staff.

With it, 030 bars 29 and 37 get their staff move and their six-note beam across the staves:
(l) seed 0's page 030 goes 7.91 to 6.75. 056 bar 17 is left: its gold files bass notes in an
upper-staff voice and treble notes in a lower one, and doubles notes across voices, so it wants a
fresh transcription, not moves. Gold for later runs: tag `gold_2026-11-17f` (17e plus 030).

## Bars read right (spec 3): it fails its check on Polish and is dropped (2026-11-18)

`eval/bars_right.py`, on (l) and (p)'s three seeds at 60k, 80k and 100k and on `l_system_150k`,
against Print gold 2026-11-17e with its print fixes (`runs/overnight_2026-11-14/bars_right2/`:
`table.txt`, `results.json`, `signal_noise.txt`). A gold staff-bar is right when the whole-page
alignment charges it nothing. An edit naming only the reading's bar charges the gold bar it is
aligned to; the alignment is walked from the whole-bar edits and checked against every edit that
names both sides (0 disagreements). OMR-NED comes from the same cost files, so both measures rest
on one scoring.

What had to be fixed first:
- **musicdiff leaves out bars with nothing it compares** (a staff of hidden rests: 051's upper
  staff has two). Its bar indices and music21's then drift apart, and the first walk disagreed 26
  times. `ned_costs.py` now records musicdiff's own bar index ("bar") on each edit and each page's
  compared gold bars ("gold_bars"). Its totals do not move.
- **Edits that name no bar** (the spec's 7.4%) are now all placed: `ned_costs.py` already placed
  extras at their element's bar.

**The check: signal against noise.** Signal is each seed's change from 60k to 100k, a real
difference, averaged over six runs. Noise is the spread of a form's three seeds at 100k, pooled.

| Measure | Signal | Noise | Ratio |
|---|---|---|---|
| Bars right | +1.83 points | 1.65 (per form 2.29, 0.45) | 1.11 |
| OMR-NED | −1.19 | 0.41 (per form 0.40, 0.42) | **2.89** |

The seed spread rests on two degrees of freedom a form, but both measures share it, so the
comparison holds. Bars right follows training: (l) seed 0 goes 60.1 → 61.5 → 61.8%, and
`l_system_150k` reaches 63.8-64.4%. Its noise, though, is about four times its share of the
signal: (l) seed 1 reads 58.4% against 61.8 and 62.7. Most bars hold a cost somewhere, and about
40% are charged, so one misread note flips a whole bar.

**Verdict, as the spec required: not used to decide A/Bs on Polish.** OMR-NED stays the decider.
The per-class table still shows where bars go wrong (100k means: hidden tuplet bars 24-41% right,
printed 11-21%, two voices 37-47%, plain 68-73%), so `bars_right.py` stays as a diagnostic.

Not done: OLiMPiC's half. It needs the seeds' readings at 60k on OLiMPiC dev, about 2.5 GPU hours.

## Step 3b's first run trained without its change (2026-11-18)

`l_system_two_voice_100k` finished with 318,975 training samples, exactly (l)'s. `--two-voice-repeat 4`
never reached the mix: with staff positions (every system run), `train_seq.py` rebuilt the items
with `dataset_items(spec)` and no repeat. The unit test covered `dataset_items` alone, not the path
the runs take. Now one function, `training_items`, builds the mix for every path. Its tests
(`train/test_two_voice_repeat.py`) cover the plain path, the staff-position path and a set weight
on top; the training suite passes (251 tests). The runs that set weights in the `--data` specs
(unmarked, Smashcima and MusiCorpus left out) were not affected: the old path read the spec's
weight.

Rerun as `l_system_two_voice_100k_r2`. The first run is kept as what it is: (l), seed 0, the same
code and data on another Vast host, one more look at run-to-run noise at a fixed seed.

**The repeat of (l) seed 0, scored (2026-11-18).** `l_system_two_voice_100k` is (l) seed 0 again,
on another Vast host (above). Whole-page, two staves, on the 17d gold:

| | Polish OMR-NED | OLiMPiC pages, v2 gold |
|---|---|---|
| Original seed 0 | 6.32 | 5.55 |
| Repeat | 6.34 | 5.34 |

At a fixed seed Polish barely moves. The run-to-run spread between seeds (0.5-0.6) is the seed's:
its initial weights, batches and augmentation. OLiMPiC pages move by 0.2 even here, so they are the
noisier set.

## Correction: step 3c (i)'s verdict was noise, not the relabel (2026-11-18)

The repeat of (l) seed 0 (`l_system_two_voice_100k`, the same code, data and seed as the original,
on another host, without the relabel) was read on `olimpic_written_dev_v2` as the seeds were
(`runs/l_system_two_voice_100k/dev_v2/`). Scanned crops:

| Run | Hidden-number bars right | Shown-number bars | Tuplet ratios read | `tuplet:hidden` written |
|---|---|---|---|---|
| (l) seeds 0 / 1 / 2 (trained 11-15/16) | 53.0 / 51.1 / 52.0% | 45.9 / 43.8 / 45.0% | 65.3 / 65.9 / 65.7% | 503 (seed 0) |
| (l) seed 0 repeated (trained 11-18) | **43.9%** | 44.7% | **59.4%** | 464 |
| Relabel (3c i) | 44.5% | 43.8% | 59.6% | **1,176** |

The repeat loses what I blamed on the relabel. The mix is identical to the seeds' (sample counts
match line by line), but the losses differ from the first logged step: training is not
reproducible across hosts at a fixed seed. The three seeds' tight spread (51-53%) understated
this measure's run-to-run noise; at least 9 points separate two runs of one configuration. On
Polish the repeat landed close to its original (6.34 against 6.32), which was luck.

**3c (i), revised: undecided on rhythm, and better on display.** Against the repeat, the
relabel reads tuplets alike: 44.5% against 43.9% hidden-number bars, 59.6% against 59.4% ratios.
It writes `tuplet:hidden` for 1,176 of 1,650 hidden groups, against 464. Its Polish "drawn" cost
for hidden numbers is 0. The earlier verdict ("does not stand") is withdrawn. Hidden-tuplet rhythm
on OLiMPiC crops needs several runs a side before any single-run claim.

The guess about why v2 makes the model write fewer tuplets is withdrawn with it.

**Also: 3d, `unmarked` left out** (`l_system_no_unmarked_100k`, 311,001 samples, no unmarked):

| Measure | 3d | (l) seeds | Repeat |
|---|---|---|---|
| Polish, 17f, OMR-NED | 6.16 | 6.00 / 6.68 / 5.89 (mean 6.19) | 6.00 |
| Polish, voice by voice | 3.59 | 3.68 mean | 3.49 |
| OLiMPiC pages (v2 gold) | 5.19 | 5.17 mean | 5.34 |
| Polish tuplet cost (17d) | 2,349 | 2,077 / 1,997 / 1,526 | - |

No headline effect. Polish's tuplet cost is above every seed, mostly hidden tuplets read without
one (685, against 504 / 168 / 272), which fits what `unmarked` teaches. Given the noise above,
this is not a verdict. Its v2 crop reading is queued.
3d on `olimpic_written_dev_v2` (scanned crops; `runs/l_system_no_unmarked_100k/dev_v2/`): hidden-number
bars 51.8%, shown 42.2%, ratios read 63.0%. Inside the seeds' range and above the repeat. Leaving
`unmarked` out shows no measurable effect on any measure here. Its Polish tuplet cost is the only
sign, and it sits inside the noise this section measured. **Verdict: no effect seen.**
`unmarked` costs 2.5% of the mix and stays until a cheaper test can see it, or the user drops it.

## Leave-outs of Smashcima and MusiCorpus, and the tuplet drop since 11-17 (2026-11-18)

`l_system_no_smashcima_100k` (289,217 samples, no Smashcima) and `l_system_no_musicorpus_100k`
(317,085, no MusiCorpus): (l), seed 0, 100k, Vast. Polish whole-page, two staves, gold 2026-11-17f:

| Run | OMR-NED | Voice by voice | Either voices | Both | OLiMPiC pages | Manuscripts 051 / 093 (17d) |
|---|---|---|---|---|---|---|
| (l) seeds 0/1/2 | 6.00 / 6.68 / 5.89 | 3.71 / 3.83 / 3.49 | | | 5.55 / 5.22 / 4.73 | 13.7, 13.5 / 18.8, 19.4 / 13.2, 15.3 |
| Smashcima left out | 6.17 | 3.85 | 5.99 | 3.66 | 5.11 | 20.5, 18.3 |
| MusiCorpus left out | 6.00 | 3.61 | 5.84 | 3.42 | 4.74 | 19.3, 15.3 |

**Neither moves the headline.** Without Smashcima, the two handwritten pages read worse than
two of the three seeds. Seed 1 is as bad, so this is a hint at most. Under the noise below it is
no verdict.

**Hidden-number tuplet bars right, scanned OLiMPiC crops, v2 labels, every (l)-form run at 100k:**

| Run | Trained | Hidden-number bars | Ratios read |
|---|---|---|---|
| (l) seeds 0 / 1 / 2 | 11-15, 11-16 | 53.0 / 51.1 / 52.0% | 65.3 / 65.9 / 65.7% |
| relabel | 11-17 | 44.5% | 59.6% |
| seed 0 repeated | 11-18 | 43.9% | 59.4% |
| Smashcima left out | 11-18 | 43.6% | 58.1% |
| MusiCorpus left out | 11-18 | 38.2% | 58.2% |
| `unmarked` left out | 11-18 | 51.8% | 63.0% |

Four of the five runs trained since 11-17 sit 7-14 points below all three seeds. That looks less
like chance than the correction above assumed. Nothing in the training path differs for (l)
(`git diff 4513248`: label_form's changes touch only canonical gaps and the unwired positions;
`lmx_tuplets` is not on (l)'s training path). The training data on disk is unchanged since 11-13.
The pods pin their packages and take torch from the image, and seed 2 ran on the same offer as
the seed-0 repeat. Either hidden-tuplet rhythm on these crops swings this much by chance (440
bars, most from a few songs), or the pod environment changed between 11-16 and 11-17.

**Test, launched 2026-11-18:** `l_system_seed1_100k_repeat`, seed 1 run again exactly. Near 51%,
the low runs were chance. Near 44%, something changed, and every run since 11-17 is compared
against a fresh baseline, not the old seeds. Until then, no tuplet verdict on 3c (i), 3d or the
leave-outs.

## Spec 4 on OLiMPiC: two voices and chords, per song (2026-11-18)

`runs/overnight_2026-11-14/cause_bars.py` scores label bars read right, by class, per song, mean
over songs, on the 46 clean scanned songs. A bar is right by compare_bars.py's test. Songs with
fewer than 5 bars of a class are left out; a class needs 20 songs. Readings are on
`olimpic_written_dev_v2`.

**The check (spec 4): signal against noise**, as spec 3 measured it. Signal is each seed's change
from 60k to 100k; noise is the three seeds' spread at 100k.

| Class (songs) | 60k, three seeds | 100k | Signal | Noise | Ratio |
|---|---|---|---|---|---|
| Two voices on a staff (38) | 59.0 / 55.4 / 55.6% | 62.4 / 59.9 / 60.1% | +4.1 | 1.4 | **2.97** |
| Chord of 3+ (39) | 77.5 / 76.5 / 74.0% | 79.6 / 78.6 / 78.3% | +2.9 | 0.7 | **4.16** |
| None of these (28) | | 88.6 / 86.5 / 87.7% | +0.2 | 1.1 | 0.21 |
| Tuplet, shown / hidden | | | | | too few (5 songs each) |

Both cause classes are at least as reliable as Polish OMR-NED (2.89) and aim at one cause. They
are used for changes aimed at voices or chords. Plain bars barely move with training, so they
cannot judge anything. Tuplets need spec 5.

The step-3 runs on these classes, all read on v2 at 100k:

| Run | Two voices | Chords |
|---|---|---|
| seeds | 62.4 / 59.9 / 60.1% | 79.6 / 78.6 / 78.3% |
| seed 0 repeated | 61.7% | 78.0% |
| relabel | 61.6% | 79.8% |
| no `unmarked` | 63.4% | 80.3% |
| no Smashcima | 61.8% | 80.1% |
| no MusiCorpus | 62.3% | 78.7% |

No run stands out. 3b's 90k reading on these classes is still to do.

**Step 3b at 90k, by class** (`l_system_two_voice_100k_r3`, read on v2 at 90k; its pod stopped at 95k).
Two voices 63.6%, chords 80.7%. On two voices that is +2.8 over the seeds' mean (62.4 / 59.9 /
60.1), about two seed standard deviations, and the highest of nine runs. But `unmarked` left out,
which changes nothing about voices, reads 63.4%, and the per-page two-voice rows earlier showed no
gap. **A hint, not a verdict:** two more seeds of 3b would decide it.

## Spec 5's first 50 gold pages through the scorer (2026-11-18)

The 50 pages with checked gold (pairs 00–23 and the two pilots; the two Op. 30 drafts from pairs 25
and 26 left out), in `runs/spec5/score_check_2026-11-18/`. No print fixes yet; `--either-ottava`.

- **Gold against itself: 0.00 on every page.** All parse as the scorer reads kern.
- **The "meter change one bar late" flags were the checker's, not the scorer's.** music21's own
  kern reader drops every clef, key and meter change after an unnumbered barline (`=`; `=9` keeps
  them). converter21, which OMR-NED uses, keeps them all in the right bar. `runs/spec5/tools/check_gold.py`
  now checks bar lengths with the scorer's parse (`bar_lengths.py`, musicdiff venv).
- **The best model (`l_system_150k` at 120k): 9.13 authors' alignment, 8.16 whole page**; Polish dev
  print is 6.73. Median page about 3. Missed and extra notes are 74% of the cost, beams 6%, tuplets 7%.
- **The four worst pages are model errors; gold held up on every bar checked against the print:**
  Czerny 038 (67; cadenza runs of 11–23, and bars running over system breaks without a barline,
  where the model ends a bar), Czerny 046 (55; three voices on one staff in sextuplet 16ths),
  Op. 53 p. 017 (39; hands alternating 16ths under cross-staff beams, read as a chord on every 16th),
  Op. 53 p. 007 (31; hidden-number triplets read as plain eighths, shifting every onset).
  Op. 53 p. 017's 346 beam edits are mostly the cross-staff beams the print-fixes step will set.

## The seed-1 repeat: the drop since 11-17 was chance (2026-11-18)

`l_system_seed1_100k_repeat2` (seed 1 run again exactly, on Vast; stopped near 90k by the credit)
against the original seed 1, both at 90k on `olimpic_written_dev_v2`'s scanned crops
(`dev_v2_90k/tuplet_bars.txt`):

| Run at 90k | Hidden-number bars right | Shown-number bars | Tuplet ratios read |
|---|---|---|---|
| seed 1 (trained 11-15/16) | 52.0% | 44.4% | 65.9% |
| seed 1 repeated (11-18) | **56.4%** | 41.2% | 67.8% |

Near 51% or above, so the low runs since 11-17 (43.9%, 44.5%) were chance, not a changed
environment: (l)'s three seeds stay the baseline. But one configuration has now read 43.9% to
56.4% on hidden-number bars: **this measure moves about 12 points between identical runs**, so
no single run decides a tuplet question; two or three seeds a side at least. The repeat and 3b's
seed 0 are finished locally from their last checkpoints (torch 2.11 for their last 15k and 5k
steps, the pods' 2.8 before), and 3b's seeds 1 and 2 run on Vast.

**Both finished at 100k, read on `olimpic_written_dev_v2`'s scanned crops (`dev_v2/tuplet_bars.txt`):**

| Run at 100k | Tuplet ratios read | Shown-number bars | Hidden-number bars | No-tuplet bars |
|---|---|---|---|---|
| (l) seed 0 | 65.3% | 45.9% | 53.0% | 79.9% |
| (l) seed 1 | 65.9% | 43.8% | 51.1% | 78.1% |
| (l) seed 2 | 65.7% | 45.0% | 52.0% | 78.7% |
| (l) seed 1 repeated | 68.1% | 41.2% | 57.0% | 78.1% |
| 3b seed 0 (`l_system_two_voice_100k_r3`) | **71.0%** | 45.0% | **62.7%** | 79.4% |

The repeat again lands above its original on hidden-number bars (+5.9) and below on shown ones:
the same noise as at 90k. 3b's seed 0 is ahead on hidden-number bars by about 10 points over
(l)'s mean, within reach of the 12-point spread seen between identical runs; seeds 1 and 2 decide.

## The user's spot check of the Polish gold corrections (2026-11-19)

Twenty of the 2026-11-17 review's corrections, checked by the user against the print
(`runs/overnight_2026-11-14/gold_spot_check_2026-11-18/`). **19 right, 1 wrong.** The wrong one
was mine, not the transcriber's: page 071's left-hand D4 (Print, the measure after the first
line's end) is a real 16th ending a slur from the line before, as the original gold had it; my
override had made it a grace note. Its three overrides are removed. Cases 1, 4, 5 and 8 were first
marked unsure or wrong over my descriptions (a missing 32nd chord, which gold had right; bar
numbers), and are right on the gold.

The case notes' bar numbers were one behind the print for 11 of the 20: the override reasons in
`eval/polish_overrides.jsonl` count bars from 0 in some places and from 1 in others. The overrides
themselves are tied to kern lines, so gold is unaffected; the page now gives places as measures
counted in each crop.
