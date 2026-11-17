# Dev sets and measures for deciding between runs

Written 2026-11-16. Specs for a sharper measure of an experiment run: what to build, from what,
and what must hold before it is used. Nothing here changes `docs/EVALUATION_PROTOCOL.md`: the
claim is still made on the frozen system, the sealed halves and the protocol's benchmarks. These
sets decide A/Bs between experiment runs, and Polish dev stays the number reported for them.

## Why

Every verdict of the week of 2026-11-09 that rested on one seed and a gap under about 0.7 turned
out to be undecided, and three of them came apart into a single page or song when split:

- (p) against (l) at 100,000, Print standard, whole-page alignment, 49 two-staff pages: 6.64
  against 6.87. On page 003 alone (l) scores 19.8 and (p) 2.0. Without it (p) is 0.14 behind, and
  better on 22 pages of 49. On the other 48 pages (p) led by 0.40 and 0.58 at 60,000 and 80,000.
- (p)'s tuplet gain: OLiMPiC song 6801790 and Polish page 003 (EXPERIMENTS.md, "Is (p)'s tuplet
  gain one song?").
- T6's loss: page 071, +1,102 of +1,882 (EXPERIMENTS.md, "Why `unmarked` at weight 3 hurt
  Polish"). And 096 is +514 for T6 where the two baselines differ by at most 197 on any page.

Two things make the measure coarse, and they need different cures.

- **Few pages, heavy tails.** 49 pages, a total cost near 8,400, and single pages that move by
  400 to 1,100 between runs. More pages cure this, by about the square root of their number, but
  only the part of the run-to-run spread that is page by page (spec 1 measures that part).
- **Few pieces per cause.** Hidden tuplets are 15 to 22% of the cost and rest on pages 003 and
  096. A change aimed at one cause is judged on two passages. More pages of the same mix do not
  cure this: a set chosen by cause does (spec 4).

## Rules for every set here

- Sealed data stays sealed: no page of Polish's test half, MusiCorpus' test split or OLiMPiC's
  test split enters a dev set.
- Nothing of a dev set is trained on, by page or by piece. A set drawn from data already held
  out (OLiMPiC dev's clean songs, `omr.held_out` scores) costs nothing. A set that takes pieces
  out of training means a rebuild and new baselines, and says so in its spec.
- The unit that is resampled is the piece (song, score), or the page where pieces are not known
  (Polish), as in the protocol's Metric section.
- A set is used for verdicts only after its own noise is measured (spec 1, on that set).
- Each set is built by a script from the repository and the fetched data, with no hand steps.
- Choices made looking at a set are disclosed in the protocol, as OLiMPiC scanned dev's are.

## 1. The noise floor, from step 2's seeds

**For:** replacing "about 0.7" with a measured number per set, per metric and per step, and
saying how much of it more pages would remove.

**From:** (l) and (p), seeds 0, 1 and 2, at 60,000, 80,000 and 100,000 (TODO step 2): the
per-page scores in `polish_steps/<run>_<step>/scores/`, and spec 2's once it exists.

**Computes,** for each set, metric, form and step:

- *Null differences.* The difference of means between two seeds of the same form, three pairs a
  form, listed for each step. They are shown, not pooled: a pair's values at 60,000, 80,000 and
  100,000 are the same two runs, and three pairs share three seeds, so a form gives two
  independent differences, not nine. The standard deviation of a form's three seed means is
  known only within a factor of about 0.5 to 6 (two degrees of freedom), and the table says so.
- *Does page resampling understate the noise?* Not by counting how often a range excludes zero:
  two independent trials a form cannot show 1 in 20. By two figures that should agree if seeds
  differ only page by page. (i) The standard error the protocol's page resampling gives a
  difference between two seeds of one form. (ii) The same quantity built from the pages: each
  page's variance over the three seeds, summed over pages, which rests on two degrees of freedom
  a page and so on about a hundred in all. Then both against the spread of the seed means
  actually seen. Seen clearly above (i) and (ii): one run is better on every page, page
  resampling understates the noise, and later A/Bs need seeds. Seen about equal: the noise is
  page by page, and more pages cut it.
- *Where the spread sits.* Pages ranked by their share of (ii), and the share of the five
  largest.
- *What more pages would buy,* stated from the above: the page-by-page part falls with the
  square root of the pages (a set four times the size halves it); the every-page part does not.
- *What three seeds buy:* the spread of a three-seed mean, for the rule TODO step 2 applies.

**Seed 0 is not quite a seed.** (l)'s and (p)'s seed 0 ran on an earlier commit and on Runpod;
seeds 1 and 2 on the current code and on Vast, with the same image and training path. Seeds 1
against 2 is the clean pair. If seed 0 stands apart from both in either form, the table says the
setup may be the cause, and the floor is taken from 1 against 2 until a seed 0 is run again.

**Deliverable:** `eval/seed_noise.py SCORES...` (per-unit scores of runs grouped by form), its
table in EXPERIMENTS.md, and the 0.7 in TODO.md's "How every run is judged" replaced by it.

**Decides:** the sizes of specs 4 and 5, and whether later A/Bs need two seeds a side.

## 2. OLiMPiC scanned dev pages, by OMR-NED

**For:** a second set, on real scans, about three times Polish dev's size, from data already
held out. It is in the training distribution (OLiMPiC's other songs are trained on), so it cuts
noise and says less than Polish about unseen editions.

**Pages:** OLiMPiC scanned dev is 100 songs, 374 pages, 1,422 systems
(`data/olimpic_written_dev/scanned`). Only songs with no copy in a training source are scored
(`data/olimpic_dev_clean.txt`): 46 songs and 615 systems, of which the as-printed labels hold
610 (five open inside a sign and were left out of `olimpic_written_dev`: 6011095 p3-s2 and
p8-s4, 6581327 p3-s2, p6-s1 and p7-s3), on 163 pages. The full IMSLP pages are in
`data/raw/olimpic/sources/imslp_pngs`.

**Gold:** one MusicXML per page, the page's systems in order, from the labels with pitches as
printed and ottava signs (`olimpic_written_dev`, so it is to the print, as Polish's Print
standard is). The labels are joined by the code that joins a reading's systems, with the page
rules off (`imply_tuplets`, `extra_beams_dropped`, `dots_mended`): gold is what the label says.
That needs the rules to be a parameter of `seq_predict.delinearize` and `joined_musicxml` (which
rules to apply, all by default); today they can only be switched off by replacing attributes, as
`rule_ablation.py` does. The parameter comes first: `rule_ablation.py` and the mending of
`imply_tuplets` (TODO 1b, 1c) use it too.
A page holding one of the five systems left out is left out, and the count stated.

**Reading:** as Polish dev is read: `seq_pages.py CHECKPOINT olimpic --layouts`, the segmenter's
layout once per song (it does not change between runs), the systems inside OLiMPiC's annotated
boxes kept, written per page and not per song. Per page because musicdiff's time and memory grow
with the score, a page is the protocol's unit on Polish, and 163 units resample better than 46;
the ranges are still resampled by song.

**Score:** `omr_ned.py --gold-format musicxml`, authors' and whole-page alignment, mean over
pages. No `--either-display` once the labels are right: the gold then says whether a tuplet's
number is drawn. They are not yet: OLiMPiC's labels call many unnumbered groups printed
(EXPERIMENTS.md, "Do tuplet-number labels match the training images?"; dev song 6801790), so
until dev is relabelled (TODO 1h) the set is scored with `--either-display`, and its numbers
before and after the relabel are not compared.

**Checks before it is used:**

- Gold scored against itself is 0 on every page.
- Gold through the page rules against gold, each rule alone and all together: what a rule does
  to a correct reading, which should be nothing. This is a standing test of the rules as well as
  of the gold. It covers both weaknesses found on 2026-11-16 (`imply_tuplets` on (p)'s short
  voices, 1b; on bars that contradict the meter, 1c), given labels in (p)'s form and a label
  with its time signature changed, and the mended rule is checked against it.
- 20 pages by eye, gold rendered beside the scan: the systems are the page's piano systems, in
  order, none missing.
- One checkpoint's page scores against its crop SER on the same systems: they should agree in
  rank over songs. Songs where they do not are looked at before the set is trusted.

**Cost:** a Polish checkpoint (56 pages) reads in about 17 minutes on the local GPU, so 163
pages are about 50, before musicdiff. Step 2's 18 checkpoints are about 15 hours, behind the
Polish queue on the one GPU.

**Then,** in this order: (l) and (p), every seed, at 100,000 (6 checkpoints, about 5 hours), and
spec 1 on this set; 60,000 and 80,000 after, as the GPU is free.

**Deliverable:** `eval/olimpic_pages_gold.py` (gold per page), a per-page mode for `seq_pages.py
olimpic`, a script beside `polish_at.sh` that scores one checkpoint, and a line in the protocol's
"What was developed on real scans".

## 3. Bars read right, beside OMR-NED

**For:** a measure that one collapsed page cannot move by 1,000. OMR-NED charges every note of a
misaligned or re-metered page (071); a bar is right or it is not. OMR-NED stays the headline and
what problems are ranked by; this decides A/Bs alongside it.

**Definition:**

- *Polish:* a gold bar is right when the whole-page alignment charges it nothing (no edit in
  `ned_costs.py --whole-page` names its measure; an inserted bar counts against the gold bar
  before it). Reported as the share of gold bars right, per page, mean over pages.
- *Edits that name no bar.* musicdiff's `extra*` edits (`extradel`, `extrains`,
  `extradurationedit`, `extrainfoedit`, `extrasymboledit`) carry no place in `ned_costs.py`'s
  output: 569 edits and 619 of 8,403 cost for (p) at 100,000, 7.4%. `bars_right.py` first tries
  to place them, from musicdiff's own objects, where `ned_costs.py` drops the place; that is a
  change to `ned_costs.py`, and its totals must not move. Those still unplaced count against no
  bar and are reported beside the share, as cost per page, for every run: a run may not gain
  bars by moving cost into them unseen. If more than a few per cent stay unplaced, the Polish
  half of this metric is not used.
- *Bar numbering.* An edit's `measure` is taken as musicdiff's index into the part's measures,
  and that must be shown to be the gold's bar on pages with a pickup, a repeat, or a bar split
  across a system break: for every page, the largest index named against the gold's bar count,
  and five such pages by eye with the charged bars marked (`bar_image.py`). Until that holds,
  nothing is reported by bar on Polish.
- *OLiMPiC:* a label's bar is right when the reading's bar aligned to it is equal with gaps
  spelled one way (`compare_bars.py`'s test, `seq.label_form`), over spec 2's songs, per song,
  mean over songs.
- *By what the bar holds,* from the gold or label alone: a hidden tuplet, a printed tuplet, a
  staff with two voices, a chord of three or more notes, none of these. A bar holding two counts
  under both.

**Checks before it is used,** all from saved readings and cost files, no GPU:

- It follows training where OMR-NED does: `l_system_100k` at 50,000 to 100,000 and
  `l_system_150k` to 150,000.
- Its signal against its noise is better than OMR-NED's. Signal: one run's change from 60,000 to
  100,000, a difference known to be real. Noise: spec 1's null differences at 100,000. The ratio,
  for both metrics, on both sets. If bars-right's ratio is not clearly the higher, it is dropped
  and this section says so.
- (m), (n), (o), (p) and T6 re-read with it: where its verdict differs from OMR-NED's, the pages
  are looked at.

**Deliverable:** `eval/bars_right.py` (both definitions, the classes), its table for the runs
above in EXPERIMENTS.md.

## 4. Sets by cause

**For:** judging a change aimed at one cause on many pieces that hold it, where its effect is
largest, before it is asked to show in a whole-page mean.

**Measure:** spec 3's bars right within the class, per piece, mean over pieces: a piece with
300 such bars counts as much as one with 10, so no one song holds the number. Pieces with fewer
than 5 bars of the class are left out. A class with fewer than 20 pieces is reported as too few
to decide, not as a result.

**What OLiMPiC dev can supply** (labels of `olimpic_written_dev/scanned`, counted 2026-11-16 by
tokens, so approximate):

| bars holding | all 100 songs | the 46 clean songs |
|---|---|---|
| any tuplet | 760 bars, 40 songs, 24 with 10 or more | 299 bars, 18 songs, 8 with 10 or more |
| a hidden number | 238 bars, 14 songs, three hold 63% | 169 bars, 8 songs, three hold 85% |
| a second voice or staff (`backup`) | every song | every song |

So two voices and chords can be judged on OLiMPiC's clean songs as they are (46 pieces), and
tuplets cannot: 8 songs, and for hidden numbers effectively 3.

**The sets:**

- *Two voices on a staff; chords:* spec 2's songs, classes from spec 3. No new data.
- *Tuplets, printed and hidden:* rendered systems of held-out PDMX scores (`omr.held_out`, 5% of
  the scores, never trained on, their copies kept out too). 40 or more scores for each of printed
  and hidden, at most 5 systems a score, chosen by the label alone (the first systems holding the
  class, in score order). Rendered by MuseScore, the engraver training has less of, with each
  tuplet's number shown or hidden as the score's author set it, and the label taken from that
  setting. No hiding rule of ours touches the set: step 3c may train on such a rule
  (`build_seq_dataset.hide_tuplet_numbers`, or its rewrite in TODO, "Hide tuplet numbers as
  engravers do"), and a set drawn with the rule would favour the run trained on it. The rates
  of 2026-11-15 were measured on these same authored settings (EXPERIMENTS.md, "When real scores
  hide a tuplet's number"); about three scores in four show every number, so the hidden class
  is the smaller, and it is counted first. Degraded as `synthetic_benchmark.py --degrade` does.
  Read with `seq_pages.py rendered`.
  *How MuseScore is made to hide them.* MuseScore 3.6.2 ignores MusicXML's `show-number` on
  import and draws every number (`build_musescore_crops.show_every_tuplet_number`), and PDMX
  has no `.mscz` to render from instead. So each score is converted to MuseScore's own `.mscx`
  first, the tuplets the MusicXML hides are given `numberType` and `bracketType` 2 (no number,
  no bracket) there, and the `.mscx` is rendered; MuseScore honours its own file. Tuplets are
  paired by staff, bar, voice and order, and a score is left out unless every pair agrees in
  ratio and in its notes' values (and pitches, where the tuplet has notes): two 3:2 groups
  swapped between voices would pass a ratio check alone. Tested 2026-11-16 on one score
  (PDMX `QmaQRqWuUZdCkt6L65HGgPr3Yxjtd2CYtHHRps5uedxunn`, 23 tuplets, 11 hidden): exactly the
  author's hidden groups lose number and bracket, checked by eye in bar 12. It costs one more
  MuseScore call per score. The same step would let `build_musescore_crops.py` draw numbers as
  authored rather than all of them; that is a training change, judged apart.
  The 20 systems a class by eye (below) include, for the hidden class, that no hidden group
  shows a number or bracket.
  This measures whether the rhythm is read, not whether a scan is. It is used with OLiMPiC's 8
  clean tuplet songs and Polish's tuplet pages shown beside it, and a change that helps only the
  renders is not believed.
- *Real scans with hidden tuplets* are what is missing, and spec 5 is where they come from.

**Checks before it is used:** the count of held-out scores holding each class (if under 40, the
set is as large as there are and the table says so); 20 systems a class by eye against their
labels (docs/TRAINING_DATA_CHECKS.md); spec 1's null differences on each class.

**Deliverable:** `eval/build_cause_sets.py` (deterministic: the list of scores and systems
written to `eval/cause_sets.txt` and committed, as `polish_split.txt` is), a mode of
`bars_right.py` that reports by class and piece.

## 5. More real print pages

**For:** the cause that has no real-scan evidence beyond two pages (hidden tuplets), and a larger
set of unseen editions. The most useful set for the claim and the most costly: Polish's gold
needed a review of every page against the print.

**Held until specs 1 to 4 are in,** because they size it: spec 1 says whether pages or seeds are
the limit, and spec 4's renders say which causes still lack evidence.

**What a page must be:**

- An engraved two-staff piano page from an edition no training source or benchmark holds
  (`eval/benchmark_overlap.py`, 0.10 as the threshold, both ways).
- With a transcription under an open licence to start the gold from (the project is open and
  non-commercial; NC-SA is fine).
- Gold made to the Print standard and reviewed against the image before any system reads it,
  the review recorded as `polish_overrides.jsonl` records Polish's.
- Chosen by the gold alone, for the causes spec 4 lacks: pages are picked before any system's
  output on them exists.

**The nearest sources are already trained on.** KernScores' piano music (Beethoven, Mozart,
Haydn, Chopin, Joplin, Scarlatti, Hummel) is GrandStaff's source, and ASAP's scores overlap it
heavily; the overlap check would turn most of it away. So either the source lies outside
GrandStaff's and ASAP's repertoire, or the chosen pieces are taken out of training, every copy
and transposition of them, with the rebuild and new baselines the rules above require. The
second is the costlier and is chosen only if the first finds too little.

**Size:** from spec 1. As a first figure, 20 pieces holding hidden tuplets, 2 pages each.

**Open, for the user:** the source (which collection pairs scans with open transcriptions of
piano music outside OpenScore Lieder, Polish and MusiCorpus), and who reviews the gold. Half of
whatever is built is sealed at once, by a split made from the gold alone, as Polish's was.
