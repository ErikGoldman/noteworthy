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
