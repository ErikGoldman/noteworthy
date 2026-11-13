# StaffSeq2Seq: a reader that takes a system as its staves

The experiment log for `StaffSeq2Seq` (`train/seq/model.py`, branch `staff-encoder`), newest last.
The main log is `EXPERIMENTS.md`; seq516's results, which this work starts from, are there.

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

## Leakage to watch

- In the overfit test: width (fixed above). Staff count (2 for all), batch statistics (blank
  batches identical), the target's length (padding comes after the label; decoding never sees it):
  none tells the systems apart.
- In real training: benchmark music in new data (3-staff PDMX renders go through the held-out and
  overlap filters); tuning on Polish dev (stopping, architecture: its numbers grow optimistic, the
  sealed half decides); the staff count taken from the label (inference takes it from the layout:
  staff positions for training must come from renderer or segmenter); copies across splits
  (GrandStaff transpositions and distorted copies, OLiMPiC songs in PDMX); the segmenter's own
  training pages, to be checked before its staff positions are used.

## Next

1. Step 6: the 10,000-step race on two identical Runpod GPUs, StaffSeq2Seq (`--staves`) against
   Seq2Seq (`--staff-samples-only`), same samples, seed and schedule; dev checks every 5,000, then
   both final snapshots on Polish dev.
2. Step 4 again, on the race model and crops it did not train on: each staff blanked, the
   attention across staves off, cross-staff passages against the rest.
3. Later: the C++ staff linking (a tilted staff's chains across a gap); seq_predict.py for staff
   models; three-staff training data.
