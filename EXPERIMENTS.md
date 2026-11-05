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
