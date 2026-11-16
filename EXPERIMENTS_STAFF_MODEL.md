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

Eleven 20,000-step runs on RTX 5090 pods (`EXPERIMENTS_STAFF_MODEL_PLAN.md`; the race's data with
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
lessons (two bad Community hosts, a hung SSH launch, Vast.ai's prices) in `TODO_STAFF_MODEL.md`.

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

All nine runs of the batch (`EXPERIMENTS_STAFF_MODEL_PLAN.md`, 50,000 steps, the race's mix with
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

The first step of walking Seq2Seq toward the staff model (EXPERIMENTS_STAFF_MODEL_PLAN.md, "Why is
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

## (m) and (n): the meter, and training on its own readings (2026-11-15, running)

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

## (o) and (p): how a second voice is written (2026-11-15, running)

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
