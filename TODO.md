# To do

Status as of 2026-11-17. The reader is Seq2Seq; StaffSeq2Seq was retired on 2026-11-15
(EXPERIMENTS.md, "(l): Seq2Seq matched to T7"), and its own experiment list is in git history
(`TODO_STAFF_MODEL.md` before this commit). Results and evidence are in `EXPERIMENTS.md`. The
list from before the staff model (seq512 to seq516, 2026-11-06 onward) is
`docs/history/TODO_2026-11-06.md`: its standing rules and decisions are below; its other open
items have not been re-checked since.

## Where the cost is (Polish dev, 2026-11-16)

(p), canonical gaps, at 100,000 steps, corrected gold, whole-page alignment, 49 two-staff pages
(EXPERIMENTS.md, "OMR-NED cost by root cause for (p)"): tuplets 15.4% (notes and marks; 22% by
the wider count of "How tuplets are misread", and they led for (l) too), a staff's later voice
11.0% (a third less than (l)'s), chord notes 9.9%, whole bars replaced 9.6%. Later: spanning
marks and beams, ledger lines. The order of 2026-11-15 (polyphony, hidden tuplets, chord tones)
is in git history; what follows replaces it.

These shares are to be made again (step 1h): on the Print standard, since nearly half of "whole
bars replaced" is cross-staff notes the Corrected gold files by hand and the reading by the print
(1g); and for (l), if step 2 drops (p). At 60,000 page 003 is seed luck ((l) seeds 28.4, 4.3,
12.9; (p) seeds 0 and 2: 2.2, 17.7; Print standard, whole-page), and (p)'s seed 2 is the worst
of the five, so "a third less than (l)'s" is one seed's figure.

## How every run is judged

- **Two sets, one metric.** Polish dev and, once step 1f has it, OLiMPiC's scanned pages, both by
  OMR-NED; a change counts when both move the same way. OLiMPiC's SER is a training check only
  ((m) and T6 were better on it and worse on Polish).
- **Against the seeds' spread** (step 2), paired by page with the baseline's seeds averaged per
  page, as for the staff baselines. A single-seed gap decides nothing under about 0.6 on Polish
  and 0.5 on OLiMPiC's pages at 100k, more at 60k-80k (measured 2026-11-17, EXPERIMENTS.md,
  "Step 2"); three seeds a side bring a mean's noise to 0.24 and 0.20.
- **With pages 003 and 096 shown apart,** and 071 for any staff-model run: between them they
  move by 400 to 1,100 from run to run and have decided three comparisons (docs/EVAL_SETS.md,
  "Why").
- **At 60k, 80k and 100k,** not the end alone: a lead that shrinks with training is a head start
  (the staff model's; perhaps (p)'s).
- **One change a run,** and by cause as well as in total, with one accounting named: tuplets by
  the wider count (`tuplet_kinds.py`), voices by `voice_metrics.py`, the rest by `note_causes.py`.
- **By page and by song** before a gain is believed: one song or page often holds it.
- The gold's commit and the alignment stated with every number; a new training path's tests run
  on the pods' torch first.

## Next steps, in order

### 1. Before any training run (saved readings and checkpoints; no pod)

These decide what step 3 trains, and 1a and 1b whether (p)'s tuplet results stand.

a. **Done 2026-11-16: yes, one song and one page** (OLiMPiC 6801790, Polish 003; EXPERIMENTS.md,
   "Is (p)'s tuplet gain one song?"); the claim is dropped until step 2's seeds. Was: **Is (p)'s tuplet gain one song?** `tuplet_values.py` and `tuplet_kinds.py` split by song
   (OLiMPiC) and by page (Polish). `32nd 6in4` read as plain 32nds falls from 264 to 16, nearly
   all of (p)'s gain in notes kept (77.6% to 87.0%), and looks like 6801790 alone; Polish's 504 to
   190 may be 096 or 003. If one song and one page hold it, "(p) reads hidden tuplets better" is
   dropped until step 2's seeds say otherwise.
b. **Done 2026-11-16: mostly the rule** (implied tuplets write 152 of (p)'s 232, on pages it left
   alone for (l); the model's own 80 are fewer than (l)'s 110; every rule still pays; EXPERIMENTS.md,
   "(p)'s invented tuplets"). Next: mend `imply_tuplets` for (p)'s short voices and for a meter the bars
   contradict (1c, 071). Was: **(p)'s invented tuplets: the model or the page rules?** Tuplet notes added where the gold has
   none cost 232 ((l) less: that row is 144 against 258). (p) leaves out the padding to a bar's
   end, so `lmx_tuplets`' rules, which go by whether a voice fills its bar, may see short voices.
   Each rule switched off in turn on (l)'s and (p)'s saved readings (as "Page-output rules
   re-measured on seq514"), and the 232 split by whether the ratio is in the raw reading. The
   rules: mend how they read (p)'s form, no training. The model: it joins T6's question (1c).
c. **Done 2026-11-16: one page, a misread 9/8 that `imply_tuplets` filled with triplets** (071, +1,102 of
   +1,882; T6 wrote no ratio there; without 071, seed-sized; EXPERIMENTS.md, "Why `unmarked` at weight 3
   hurt Polish"). Hidden numbers are not shown to teach false tuplets; the rule's trust in a read meter
   goes to 1b. Was: **Why `unmarked` at weight 3 hurt Polish** (T6, +1.49 [-0.32, +3.80]): its saved Polish
   readings through `tuplet_kinds.py`; tuplets added where the gold has none, by page, against the
   baseline seeds. Decides whether step 3's tuplet data may raise hidden numbers at all.
d. **(i) done 2026-11-16: OLiMPiC's labels** (EXPERIMENTS.md, "Do tuplet-number labels match the
   training images?"): 73 of 109 sampled groups labelled printed draw no number; every other
   source agrees. Next, before 3c: confirm the cause (OpenScore MusicXML not recording hidden numbers;
   the scores' own MuseScore files do), relabel OLiMPiC train and dev tuplets from those files (the
   `.mscx` route of spec 4), check a sample by eye, rebuild, one run. (ii) the probe is still open.
   Was: **A hidden number drawn about half the time** (44% / 55% of OLiMPiC's hidden groups; a printed
   one hidden 5-7%). Two checks tell labels from reading. (i) Per training source, tuplet groups
   whose image draws no number against what the label says, 20 crops a source by eye
   (docs/TRAINING_DATA_CHECKS.md); OLiMPiC's 21 systems with no `<tuplet>` element show sources
   differ. (ii) Labels fed, the decoder's first choice at each hidden group's number token
   (as `backup_probe.py`). Wrong when fed: labels or perception, and (i) says which. Right when
   fed: a slip of free reading, and no label fix will cure it.
e. **By eye done 2026-11-16** (EXPERIMENTS.md, "Printed tuplets whose ratio is left out, by eye"):
   a third are 6801790, whose print shows no numbers (labels wrong: hidden, not printed); most of
   the rest is how the number is set (far up a slur, small arc over unbeamed groups, once for a run):
   training data, not a rule. The probe (labels fed) is still open. Was: **Printed tuplets whose ratio is left out** (OLiMPiC, 34 / 30 bars; printed-number bars are
   right about half the time against 80% of plain bars). The 30 by eye (`bar_image.py`), sorted
   as looked at: number far from its notes or inside a slur, sextuplets and mixed groups,
   tremolos (`half 2in1`), a number printed once for a run. Labels fed, the ratio token's rank at
   those groups: right when fed means compounding, wrong means the number is not read. And
   printed against hidden bars matched by notes and voices in the bar, since printed bars are
   denser music and that alone may be the difference.
f. **Done 2026-11-17: built, checked, read at 100k** (EXPERIMENTS.md, "Step 2"; `olimpic_at.sh`,
   9 minutes a checkpoint). 60k and 80k still to read. Was: **The second set:** OLiMPiC's scanned dev pages through `seq_pages.py`, scored by OMR-NED
   against their MusicXML. Then (l) and (p) at 60k, 80k and 100k on it: a second opinion on (p)
   from checkpoints already here. Spec: docs/EVAL_SETS.md, "2. OLiMPiC scanned dev pages, by
   OMR-NED" (the 46 clean songs, 163 pages; gold per page from the as-printed labels; four checks
   before it is used). The dev labels share 1d(i)'s fault (unnumbered groups called printed:
   6801790), so until they are relabelled (1h) this set is scored with `--either-display`, and the
   spec's "no `--either-display`" holds from the relabel on.
g. **By eye done 2026-11-16** (EXPERIMENTS.md, "The whole bars musicdiff replaces, by eye"): for (p)
   54 bars, 806; 47% cross-staff notes (gold by hand, reading by print), 19% a missed clef change,
   6% `rest:measure` without its head; four suspected gold errors. Was: **The whole bars musicdiff replaces, by eye:** 9.6% of the cost, 58 bars for the best model,
   110, 042 and 093 holding 57%. What each needs (a rule, gold, training) is not known.

h. **What 1a to 1g leave to do** (2026-11-16), in this order; none needs a pod.
   1. **Done 2026-11-17: 46% of all groups** (10,622 of 23,308; EXPERIMENTS.md, "OLiMPiC's tuplet
      labels against the scores' own MuseScore files"). Was: *Size OLiMPiC's tuplet-label fault.* The 73 of 109 is 6 crops. Over the whole source, train
      and dev: each group's label against the score's own MuseScore file, counted by song. Confirms
      the cause (OpenScore's MusicXML not recording a hidden number) and says how much of the
      20,901 groups the relabel moves.
   2. **Built 2026-11-17** (`olimpic_written_v2`, `olimpic_written_dev_v2`; 9 train songs left out,
      decided with the user; 30 crops by eye, all right). Next: the run (step 3c (i)), and the
      re-measures below. Was: *Relabel OLiMPiC's tuplets,* train and dev, from those files (1d); 20 crops by eye
      (docs/TRAINING_DATA_CHECKS.md); rebuild. Then re-measure what rested on the old labels:
      "a hidden number drawn 44% / 55%" (1d), "printed bars right half the time" (1e), the 98.6%
      with labels fed ("Hide tuplet numbers", below), and `tuplet_kinds.py`'s printed and hidden
      columns on OLiMPiC. Until then each is read as unsure which way.
   3. *`rest:measure` without its `rest` head* (1g: about 22 on dev, 19 bars, 46 of (p)'s cost)
      converts to a pitchless zero-length note. Mend the conversion (`seq_predict.delinearize`),
      with a test; measured on saved raw readings, as the implied-tuplets rule was.
   4. *The cost shares again, on the Print standard* (`ned_costs.py --whole-page`,
      `note_causes.py`, `tuplet_kinds.py`), for the form step 2 keeps, on the code of step 2a.
      Expect the cross-staff bars (376 of 806, page 110 holding 322) to go; "Where the cost is"
      is rewritten from it, and step 3's order checked against it.
   5. **Done 2026-11-17** (076 bar 14, 065 bars 39-40, 083 bar 15 corrected or moved; 067 was right;
      110's cross-staff triplets and 051 moved too; gold check 0 failures). Was: *Suspected gold errors, from the image alone,* no reading consulted: 110 bar 25 (upper
      notes printed on the lower staff outside the 8va, an octave high), 076 bars 14 and 17
      (chords in the wrong spine for where they are printed), 065 bars 39-40 (two-note chords
      encoded as single notes), 083 bar 15 (GG# or F#2; low confidence). Each kept or dropped as
      an entry in `eval/polish_overrides.jsonl` with its reason.
   6. *Label oddities seen in 1d:* tuplet groups with no ratio token (`musescore 000018/s3`,
      `pdmx 012769/s0`, `smashcima 000443/s0`): how many in each source, and whether the builder
      or the score is at fault; Smashcima's quarter rests that look like a 3: how often, against
      real handwriting.
   7. *The two probes still open,* labels fed: 1d(ii), the first choice at a hidden group's number
      token, and 1e's, the ratio token's rank where the ratio was left out. After the relabel
      (item 2), or their groups are sorted by wrong labels.

### 2. The noise, and whether (p) stands (5 runs)

**Done 2026-11-17: (p) does not stand; step 3 builds on (l)** (EXPERIMENTS.md, "Step 2"). (p) − (l)
at 100k, three seeds: Polish -0.13 [-0.62, +0.36], OLiMPiC pages +0.20 [-0.08, +0.51].

(p)'s lead over (l) shrinks: 1.26, 0.99, 0.33 at 60k, 80k, 100k (authors' alignment).
- (l) and (p), seeds 1 and 2, 100,000 steps, same data and schedule as seed 0 (4 runs). The
  spread of each form's three seeds, on both sets at 60k, 80k and 100k, is the noise every later
  run is read against.
- (p) on the 150,000-step schedule, seed 0, against `l_system_150k` (6.73 at 120,000).
- (p) is the form to build on if its three seeds average below (l)'s at 100,000 with the paired
  range clear of zero, on both sets, and it is not behind at 150,000. Otherwise step 3 builds on
  (l) and (p)'s gain is logged as a head start.

### 2a. Overnight (local GPU): fix the meter in force, then score step 2 on the fixed code

**Done 2026-11-17** (26 readings; no rebuild was needed: the hidden-rest fix was in the code
before the canonical-gaps readings started).

Found 2026-11-16 by spec 2's gold check (correct labels through the page rules: song 6259566
pages 2 and 4 cost 38 and 30). `lmx_tuplets.written_bar` gives a system's *first* time signature,
and every caller takes it as the bar for the whole system: bars after a change of meter inside a
system are judged by the old one, and the meter carried into the next system is the first of the
last system, not its last (6259566 alternates 6/8 and 2/4: 2/4 is carried while 6/8 is in force,
and `imply_tuplets` makes plain 16ths triplets). `seq_pages.meters_in_force` has the same flaw.
Training: only the meter context reads it (`seq.meter.with_meter`, runs with `--meter-context`:
(m), T8, (n)), on systems that open without a signature and change inside; (l), (p), (o) and
step 2's runs are untouched. Evaluation: every page's rules, `bar_fill`'s choice of crop height,
and (m)/(n)'s meter given when reading. Polish dev's two-staff gold has one page with two meters
(043, 6/8 and 2/4); OLiMPiC's has at least 6259566.

1. The fix: one bar length per bar (each time signature from its bar on), the last carried to the
   next system; in `lmx_tuplets` (`imply_tuplets`, `extra_beams_dropped`, `dots_mended`, `bar_fill`,
   which take a bar per measure) and `seq_pages.meters_in_force`. Tests: a system changing meter
   inside; a carried meter after a change. Check: spec 2's rules-on-gold check goes to 0 on
   6259566, and the other pages do not move.
   With it, the implied-tuplets check of 2026-11-16 (only where another voice fills the bar; kept,
   EXPERIMENTS.md), already in the code: both change the numbers, so both are on the code these
   readings use.
2. Score on the fixed code, in fresh `polish_steps` folders (new `*_fixed` run links, as the
   `*_raw` ones; nothing overwritten): seed 0 of (l) and (p) at 60k, 80k, 100k, and every step-2
   checkpoint at 60k, 80k, 100k (and 120k, 150k for `p_system_canonical_gaps_150k`), each also on
   the corrected gold (`rescore_polish.sh gold_rests`). About 24 readings of about 17 minutes on
   the local GPU. A crop's reading can change with the fix (`bar_fill` picks the crop height), so
   these are read again, not rebuilt from saved `.lmx.json`.
   Started 2026-11-16 21:33 (`runs/overnight_2026-11-14/overnight_fixed.sh`, log beside it), on the
   meter fix. Afterwards, on the CPU: every page rebuilt from its `.lmx.json` with the final code
   (for canonical-gaps runs, `label_form.with_hidden_bar_rests_restored` first: added after the
   queue started, idempotent) and scored against the final gold (the cross-staff and other gold
   corrections of 1g), then the corrected-gold reports.
3. Then step 2's comparison and spec 1's table (`eval/seed_noise.py`), all on the fixed code.
4. EXPERIMENTS.md: the fix, its date, and that every Polish number before it is on the old code.
   1a, 1b and 1c rest on the old code; none of their pages holds a change of meter (003, 071, 096
   and (p)'s rule pages each have one), so they should stand; say so.

The step-2 Polish queue was stopped 2026-11-16 at about 7:45 PM (the GPU needed); what it had
scored on the old code: `p_system_canonical_gaps_150k` at 60k and 80k, `l_system_seed1_100k` at
60k and 80k, `l_system_seed2_100k` and `p_system_canonical_gaps_seed2_100k` at 60k (corrected-gold
reports in each folder). `p_system_canonical_gaps_seed1_100k_60000` was stopped half-read. The
Vast runs go on to their ends and come home on their own (`vast_launch.sh`).

### 2b. A sharper measure, before the training batch (docs/EVAL_SETS.md)

The week's single-seed verdicts were undecided, and three came apart into one page or song. Step
3's runs are worth about what (p)'s was, so they wait for these, in this order. None needs a pod.

- **Done 2026-11-17 (Polish and OLiMPiC pages, 100k).** **The noise floor** (spec 1), as each of step 2's seeds is scored: the differences between
  seeds of one form, the page resampling's error against the spread seen, where the spread sits
  by page. `eval/seed_noise.py`; its figure replaces the 0.7 above. Seed 0 ran on another commit
  and host: seeds 1 against 2 is the clean pair.
- **The page rules as a parameter** of `seq_predict.delinearize` and `joined_musicxml` (spec 2,
  "Gold"): step 1f's gold, `rule_ablation.py` and the mending of `imply_tuplets` (1b, 1c) all
  need it. Then 1f, at 100,000 first (6 checkpoints, about 5 hours of the local GPU).
- **Bars read right, beside OMR-NED** (spec 3): a measure one collapsed page cannot move, by what
  the bar holds. From saved readings and cost files. First the edits that name no bar (7.4% of
  (p)'s cost) placed or reported apart, and the bar numbering shown on pickups and repeats. Kept
  only if its signal against the seeds' noise beats OMR-NED's; `eval/bars_right.py`.
- **Sets by cause** (spec 4): two voices and chords on OLiMPiC's 46 clean songs, averaged by
  song; tuplets, printed and hidden, on rendered held-out PDMX scores (40 or more a class),
  numbers shown or hidden as each score's author set them and never by our rule, since OLiMPiC's
  clean songs hold hidden numbers in 8 songs and three of those hold 85%.
  `eval/build_cause_sets.py`, the lists committed.
- **More real print pages** (spec 5): held until the three above size it. The source (outside
  GrandStaff's and ASAP's repertoire, or pieces taken out of training) and the gold's reviewer
  are the user's to decide (below, "Held for the user").

### 3. The training batch (each against step 2's baselines, 100,000 steps)

Each run is judged first on its own cause's set (step 2b), then on the two whole-page sets.

In this order if pods are short; (a) and (b) can run with step 2.

a. **Defaults decided 2026-11-16 (user):** one `at:` token per position if the labels hold under
   about 200 (else ask); a position earlier than its voice has reached is read as no gap for
   scoring, but never silently: each one is logged where it happens (page, system, bar, voice) and
   counted in every report beside the score; the small form first, kern-like interleaving only if it
   helps. **A voice's gaps as positions in the bar.** Compounding is the main rhythm cause (92% of plain
   bars right when fed), beam search and (n) did not cure it, and what (p) gave back is notes
   shifted after a misread value. `seq.label_form`, on the form step 2 keeps ((l), as the seeds
   stand at 60,000: (p)'s seed 2 is the worst run; then the positions replace (l)'s backups and
   forwards, and the check below is made on (l)'s labels): each voice's entry and every gap inside
   it written as where the next event starts (`at:` a position in the bar), not as forwards to
   add up. First, with no GPU: the distinct positions in the training labels (the vocabulary it
   needs), label lengths against the 1,023-token limit, and the round trip over the 32,040 labels.
   Measures: `p_patterns.py`'s first event out of place (after a misread value: 159 / 187), the
   moved-onset rows of `ned_causes.py`, `entry_probe.py`. If it helps, the larger form follows:
   a staff's voices written together in time order, as kern does (Copisteria had half our
   moved-onset cost early on, 1,689 against 3,676).
b. **Two-voice systems weighted up,** to Polish's share (27.9% of staff-bars; training 14-20%,
   the same textures). One run; `voice_metrics.py` and the two-voice rows.
c. **Tuplet data: step 1 has decided,** three runs, one change each, in this order.
   (i) **Running 2026-11-17** (`l_system_relabel_100k`, Vast, seed 0, (l), 100k; baseline (l)'s three
   seeds): OLiMPiC's tuplets relabelled (1h items 1 and 2): the only source whose labels disagree
   with its images, and the real-scan one. (ii) "Hide tuplet numbers as engravers do" (below)
   across the synthetic sets, `unmarked` at weight 1: 1c cleared hidden numbers of teaching false
   tuplets (071 was the page rule). (iii) Numbers set as in 1e's bars: far up a slur, in a small
   arc over unbeamed quarter-plus-eighth groups, once for a run. Measures: tuplet bars printed
   and hidden on the tuplet set of step 2b, a hidden number drawn, tuplets added where none are;
   Polish 003 and 096 shown apart.
d. **seq516's data changes, each left out once** (Smashcima, MusiCorpus, `unmarked`, the
   GrandStaff cleaning; 4 runs): they went in together, worth about 2 points, and the handwriting
   share and `unmarked`'s weight rest on guesses until they are apart. `unmarked` first.

### 4. Later

Chord tones left out or added (9.9%, few pages, 051's octave doublings: gold or reading, by eye
first; the share to be made again on the Print standard, 1h item 4); systems of more than two
staves (below); spanning marks and beams; ledger lines.

## Hide tuplet numbers as engravers do

From EXPERIMENTS.md, "When real scores hide a tuplet's number".
`build_seq_dataset.hide_tuplet_numbers` today: one system in five with tuplets, every number hidden (half the time keeping the first), whatever the tuplet. Real
MuseScore scores choose per score (about three in four show all, a few hide all), and the
rest hide by place in a run of like groups (about half at the first, 85-89% from the fifth),
print again at a voice's first tuplet in a system and for a lone tuplet in a bar, and mostly
keep 5s and 7s printed. Rewrite the rule from those rates and rebuild the synthetic sets;
`unmarked` stays as the stress set. Metric: tuplet bars with a number hidden, false tuplet
tokens on plain notes; Polish 003. Once judged low priority (Seq2Seq reads OLiMPiC's real hidden
triplets at a group's first note 98.6% of the time, labels fed), but reading on its own (p) draws
the number on 55% of the groups whose print hides it (EXPERIMENTS.md, "How tuplets are misread").
Both figures rest on OLiMPiC's labels, which call many unnumbered groups printed (1d(i)): to be
measured again after the relabel (1h item 2). This is step 3c's second run, after the relabel.

## Next: Seq2Seq for systems of any number of staves (decided 2026-11-14)

Seq2Seq was chosen (2026-11-15); attention is to be explored so that systems of more than two
staves can be read. The meter stays off by default: on Polish it read worse ((m), 2026-11-15).

- **Built (2026-11-14, tests pass, not yet trained):**
  - `--interline-scale` (Seq2Seq, needs `--relative-rows` and `--staff-samples-only` for the
    positions): each system at staff scale (`seq.staff_data.at_interline`, about 17 px an
    interline), its height following its staves, batched by `collate_systems` (white below,
    unmasked); read the same way by `seq_pages.py` (the segmenter's interline) and through
    `seq_predict.reading_dataset` (one place now chooses every reader's dataset).
  - `--row-windows W` (with `--relative-rows`): the encoder attends within bands of W feature rows
    (8 rows is about a staff at staff scale), within the bands shifted by half, then down each
    column: three rotary-row layers (27.8M parameters against 26.0M). On the CPU at 512-1,024 px
    the CNN dominates and windows are not faster yet; their case at 3-4 staves is the structure,
    at 6 or more the cost.
- **Needs: training data of three staves.** The builders take two-staff piano parts only. Source:
  PDMX songs and duos (a voice or instrument part above a piano part), rendered as one system;
  check by eye (docs/TRAINING_DATA_CHECKS.md). Polish dev's 7 three-staff pages (about 91 for
  every model today) are the measure.
- **Then, at 50,000 steps, all Seq2Seq with `--meter-context` (`unmarked` stays at weight 1: at 3 it
  cost Polish 1.5, T6):**
  (a) today's architecture; (b) relative rows (T10); (c) `--interline-scale` on two-staff data;
  (d) (c) with `--row-windows 8`; (e) (c) or (d) with the three-staff data. Then the long run
  (about 100,000 steps: T7 and seq516 both stop gaining there on Polish).
- **Layout probe** (`runs/overnight_2026-11-14/layout_probe.py`): dev crops read with 25% white
  margin added above, below or both; a model reading by where things sit in its crop suffers.
  Run on the race arms and baseline seed 0; T4 and T10 when they land.

## Pods: lessons of the first night (2026-11-14)

- **A pod now removes itself when its run ends** (`runs/overnight_2026-11-14/pod_side.sh`): exit
  status into the run, pushed to R2 (three tries), the push pulled back and checked, then
  `runpodctl remove pod`. The first night it waited for the session to delete it, and the session
  ran on the PC, which restarted at about 06:05 UTC: eleven finished pods idled until the account's
  balance ran out (about 09:00), $40.14 spent against about $15 needed, and T7 stopped at step
  75,800. If no push gets through, the pod stays up until its deadline so the run can be fetched.
- `pod_watch.sh` reads a run's end from R2 (it no longer needs the pod); `pod_run.sh` launches with
  `setsid` and a timeout (the SSH launch hung on the background run's open descriptors).
- Two Community hosts were bad (an SSH key not installed; a GPU PyTorch could not use): the
  environment check catches the second before any data is pulled.
- **Vast.ai (set up 2026-11-16):** `vast_launch.sh OFFER NAME [options]` rents (`vast_up.sh`),
  runs (`pod_run.sh`) and destroys the instance from the PC when the watcher returns. A Vast
  instance cannot remove itself: the key Vast injects is refused (401) for stop and destroy, the
  account key must not go to a stranger's host, and the account key may not mint a per-instance
  key. So, unlike Runpod, a PC down mid-batch leaves finished instances billing until it is back.
  Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the pods' own: plain PyTorch images
  have Python 3.11, the pins need 3.12). Same speed as Runpod (43 samples/s). The CLI exits 0 on
  refused calls; check the instance list. Bandwidth is billed apart ($0-0.05/GB by host, about
  40 GB a run); reliable 5090s were $0.45 in the morning and $0.65-0.76 by evening, when Runpod
  Community had none.

- **A fixed deadline cannot tell slow from hung** (2026-11-16): `pod_run.sh`'s deadline assumes
  4 steps a second; a Vast host at 3 (Thailand) passed it at step 95,000 of 100,000, and
  `pod_watch.sh` gave the run up and `vast_launch.sh` destroyed the instance. Resumed from R2 on
  another host (`NW_RESUME=1`, `NW_POD_DEADLINE=14400`). To do: call a run hung when its train.log
  has not moved for 30 minutes, not when a clock built from an assumed speed runs out; keep a
  deadline only as a far backstop. **Done 2026-11-17:** `pod_watch.sh` calls a run hung after 30
  minutes with nothing written (by the pod's clock), and the deadline is twice the expected training.
- **Pods run torch 2.8, the local GPU 2.11** (2026-11-15): a training change that passes locally
  can still break on a pod. `--own-readings` did (autocast's cast cache under no_grad). Run a new
  training path's tests on a pod's torch before a long run, or pin the local torch to the pods'.

## The outer-notes dev set (built 2026-11-13)

`data/olimpic_written_dev/{synthetic,scanned}`: OLiMPiC's 1,422 dev systems (of 1,438; systems
opening inside a sign left out), synthetic and scanned, relabelled with pitches as printed and
ottava tokens (`train/build_olimpic_written.py ROOT OUT dev`), staff positions found by
`train/find_staff_positions.py` (all 2,844 split). Read with
`eval/outer_notes.py CHECKPOINT --out READINGS.jsonl` (GPU, about 10 min; `--report` reprints
from saved readings). Built by `scripts/build_seq_data.sh` with the rest (always under
`$NW_DATA`, where `outer_notes.py` looks).

## Open questions

- Whether the worst Polish pages (096, 093, 110, 056) have staff positions from the segmenter
  that are off: a bad position reads the whole page badly, and would look like a crop problem.
- Tie and slur confusion (about 320 swaps on OLiMPiC dev, both ways) and arpeggios, tremolos,
  strong accents: frequent in SER, cheap in OMR-NED.
- Errors rise mildly along medium-length crops (5.2% to 8.2%): decoder drift or the right side
  of the page; not told apart.
- A clef change right after the key signature at a system's start (081 bars 13-15: treble on the
  lower staff missed, every note a 10th low, 153 of (p)'s cost; 1g). How often training crops
  hold a clef change in that place, by source, before it is called a reading problem.
- Why page 003's hidden tuplets are read by some seeds and not others (five runs at 60,000: 28.4,
  4.3, 12.9, 2.2, 17.7), and 096's by none: what differs between the two passages.

## Leakage to watch

- Benchmark music in new data (3-staff PDMX renders go through the held-out and overlap filters).
- Tuning on Polish dev (stopping, architecture, label form): its numbers grow optimistic; the
  sealed half decides.
- The staff count taken from the label: inference takes it from the layout, so staff positions
  for training must come from the renderer or the segmenter.
- Copies across splits (GrandStaff transpositions and distorted copies, OLiMPiC songs in PDMX);
  the segmenter's own training pages, to be checked before its staff positions are used.
- **Held-out PDMX has copies in other training sets (found 2026-11-16, to fix).**
  `list_scores.drop_copies_of_held_out` drops a held-out score's copies only among the render
  list's own scores (PDMX and ASAP), matched by `omr.pieces` metadata; GrandStaff, OLiMPiC,
  MusiCorpus (and the Lieder) are never compared. A melodic check (`melody_grams`, 0.10) found
  training copies of held-out scores: Mozart K279 in GrandStaff (0.20), Chopin's Sonata 2 in ASAP
  (0.37), the Moonlight sonata and Für Elise as PDMX copies under other titles (0.56, 0.26;
  sizing subagent, `scratchpad/spec5_sources/`, not yet re-run here). Anything that takes held-out
  PDMX as unseen is affected: `synthetic_benchmark.py`'s dev and test, spec 4's printed-tuplet
  set. To do: a filter that compares every held-out score with every training set by melody, both
  ways, at 0.10; the held-out scores with a copy leave the held-out list (they are not moved into
  training: their copies are already there). Then the count of what remains, and whether any past
  synthetic-benchmark number moves when they are left out.

## Held for the user

- Sealed test run. Numbers only, and only after gold and the model are frozen.
- Headline metric: authors' OMR-NED or `--whole-page-alignment` (or both). Then the protocol text.
- Email to the Transcoda authors: add the bar-pinning finding (017, 042).
- Email to the Transcoda authors: draft at https://claude.ai/artifact/NDK3s41464hRbJP1ybf7Bp. Send
  after the review is finished. Add dot-or-wedge and the sealed results of the review as needed.
- SMB application, which must come from the user.
- Gold reviews: sample of about 20 cases for the user to check; the hardest calls were 067 bars 5-8
  and 12, and 062 bar 6.
- C++ port: off plan until there is a good model.
- More real print pages for dev (docs/EVAL_SETS.md, spec 5): which collection pairs scans with
  open transcriptions of piano music outside OpenScore Lieder, Polish, MusiCorpus and the
  repertoire of GrandStaff and ASAP (KernScores' piano music is trained on), and who
  reviews the gold. A first figure: 20 pieces holding hidden tuplets, 2 pages each, half sealed.

## Rules that are easy to forget

- Gold must be checked through converter21 (`eval/check_gold.py`), never music21's own kern reader.
- Sealed pages: gold and images only. No system output, no errors, nothing into heuristics or
  training. Corrections to gold are allowed.
- Scoring is plain OMR-NED throughout; only the gold differs between standards.
- Single-page locked scores can swing widely on correct changes (042): check unlocked too.

## Decided 2026-11-16 (user)

- **Cross-staff notes go on the staff they are printed on**, in gold as in readings (the print
  decides). Polish dev's whole-bar cost is 47% this (1g, page 110 most); gold corrected through
  `polish_overrides.jsonl` after checking how kern and converter21 carry a note printed on the other
  staff.
- **Spec 4's hidden-tuplet class: skipped for now, to revisit** (held-out PDMX has 4 scores that
  hide numbers). Its printed class stands.
- **Step 3a's defaults** (one `at:` token per position; small form first) agreed. A position behind
  its voice: no gap for scoring, and always reported (logged and counted), never dropped silently.
- **Spec 5's gold: Claude reviews every page against the image** and sends the user a few pages to
  spot-check the review's accuracy.

## Decided 2026-11-08 (user)

- Next run: `time:common` / `time:cut` tokens. Scoring the non-Print standards (As published,
  Corrected) must accept either the symbol or the numbers (like `--either-ottava`).
- Report the authors' OMR-NED and our whole-page alignment side by side; we steer by whole-page.
- Frozen candidate: after seq513's Polish comparison. Commit after that and the open decisions.
- Transcoda email: hold until results are in.
- Hidden tuplet numbers: before scoring seq513, make the non-Print scoring accept either a shown or a
  hidden number (E). Next run adds a `tuplet:hidden` token (B), and the Print gold marks tuplets whose number isn't
  printed (D).
- For publication: B can cost us on the authors' strict OMR-NED wherever their gold shows a tuplet
  number the page doesn't print, since we output it hidden. Say so next to that number. Our reasoning:
  the target is what's printed, and MusicXML can record the hidden number
  (`<tuplet show-number="none">`), so a hidden 3 is the faithful transcription and the rhythm is
  unchanged. Quantify how much of the gap comes from this when seq514 is scored.
