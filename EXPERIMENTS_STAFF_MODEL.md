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

1. Step 4, on a rerun of the staff arm that saves its weights: blank each staff and read again (that
   staff's notes must break, the other's survive); turn off attention across staves (cross-staff
   passages must break).
2. Before the race: staff positions from renderer and segmenter; the duplicate-staff fix;
   augmentation that moves staff positions with the image (geometry on the whole system, centres
   moved by the margin shift and the rotation, staves cut, ink and paper degradation shared by the
   staves); one crop per staff in `seq_pages.py`; `train_seq.py` training `StaffSeq2Seq`.
3. Step 6: the 10,000-step race on two identical Runpod GPUs.
