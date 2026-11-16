# To do

Status as of 2026-11-16. The reader is Seq2Seq; StaffSeq2Seq was retired on 2026-11-15
(EXPERIMENTS.md, "(l): Seq2Seq matched to T7"), and its own experiment list is in git history
(`TODO_STAFF_MODEL.md` before this commit). Results and evidence are in `EXPERIMENTS.md`. The
list from before the staff model (seq512 to seq516, 2026-11-06 onward) is
`docs/history/TODO_2026-11-06.md`: its standing rules and decisions are below; its other open
items have not been re-checked since.

## Priorities by root cause (decided 2026-11-15)

From what OMR-NED charges the best model on Polish dev (EXPERIMENTS.md, "What OMR-NED charges the
best model for, by root cause"; corrected gold, whole-page alignment). In
this order; the big causes before the small ones (spanning marks and beams wait).

1. **Polyphony: a staff's second voice.** Two-voice staves are 36.5% of the cost on 38% of the
   notes; were they as accurate as one-voice staves, the cost would fall about 16%. What goes wrong:
   a second voice off the staff dropped whole (012 bars 20-21), oom-pah basses with the voices
   merged (022 bars 12-14), inner-voice entries placed early (096 bar 4). Not the decoder failing
   to look back, nor a rest convention learned from OLiMPiC (tested). Open: real engraving's
   layout of two voices against our renders (hypothesis 4), whether the model decides to start the
   second voice at all (the backup token, labels fed). First steps: a metric of dropped, merged and
   mistimed voices; then training data with more real-engraving polyphony, or two-voice systems
   weighted up.
2. **Hidden tuplets.** 10.7% of the cost, the largest single cause, two thirds of it on two-voice
   staves (triplets in one voice against plain notes in the other). Read with (n), scheduled
   sampling, since a misread triplet moves every later note; then the engraver-like hiding rule
   below ("Hide tuplet numbers as engravers do"). Measure apart from polyphony: they overlap.
3. **Chord tones left out or added.** 9.7% now, the largest note-level cause once polyphony is
   fixed (12.6%); mostly one-voice staves, on few pages (051's octave doublings hold 30 of 79).
   First: check by eye whether the gold or the reading is wrong, as for the rests.

Later: spanning marks and beams (9.7%; what musicdiff's extras are, slurs or dynamics and pedal the
vocabulary cannot write, not yet known), ledger lines (7.1%).

**Update 2026-11-16.** Canonical gaps, (p), cut the second voice's cost by a third (1,398 to 924 on
Polish dev); tuplets, notes and marks, are now the largest cause (15.4% of (p)'s cost), then a
staff's later voice (11.0%), chord notes (9.9%) and whole bars replaced (9.6%) (EXPERIMENTS.md,
"OMR-NED cost by root cause for (p)"). Most tuplet misreads are the ratio left out, each note
written as its printed value; next, a hidden number drawn ("How tuplets are misread", "What a
tuplet's notes are read as").

## To test next (from the review of both logs, 2026-11-16)

In this order: 1 first, since 2-5 are judged by differences of its size.

1. **Confirm (p) before building on it, and measure the noise.** (p)'s lead over (l) on Polish
   dev shrinks with training: 1.26, 0.99, 0.33 at 60k, 80k, 100k (authors' alignment), the shape
   the staff model's lead had before it proved a head start. And two staff baselines at 50,000
   steps differ by 0.68 (11.03 against 11.71), more than the gaps (m), (o) and (p) were judged on
   (0.45, 0.14, 0.33). Runs: (p) on the 150,000-step schedule against `l_system_150k` (6.73 at
   120,000); one more seed each of (l) and (p) at 100,000. The seeds' spread is the noise floor
   for every later comparison; until it is known, a single-seed gap under about 0.7 decides
   nothing.
2. **Labels in time order.** Compounding is the main rhythm cause (fed the right history, 92% of
   plain bars are read right), and neither beam search nor training on its own readings (n) cured
   it; (p) fixed where a voice enters, and what it gave back is notes shifted after a misread
   value. A label form in which one misread length cannot move the rest: events written in time
   order across a staff's voices (as kern does), or each event with its position in the bar
   rather than only its length. Copisteria writes kern and had half our moved-onset cost early on
   (1,689 against 3,676, clean model, 45 pages). Through `seq.label_form`, against (l) and (p),
   at the length and seeds item 1 settles.
3. **Tuplets, now the largest cause for (p)** (15.4% of the cost, notes and marks). First why T6
   (`unmarked` at weight 3) made Polish worse (+1.49 [-0.32, +3.80]) while helping OLiMPiC's
   hidden-number bars: whether it writes tuplets where none are, by page. Then "Hide tuplet
   numbers as engravers do" (below) across every set's crops, the first group or two numbered and
   the rest not, rather than a separate set weighted up.
4. **Two-voice systems weighted up.** Polish dev has 27.9% two-voice staff-bars, most training
   sources 14-20%, with the same mix of textures. Cheap, and already named under priority 1;
   measured with `voice_metrics.py` and the two-voice rows of `ned_causes.py`.
5. **seq516's data changes, one at a time.** Smashcima, MusiCorpus, `unmarked` and the GrandStaff
   cleaning went in together and were worth about 2 points; the GrandStaff control planned then
   was never run. One run leaving each out (four runs), so the next data decisions, the handwriting
   share and T6's question among them, rest on what each set does.
6. **The whole bars musicdiff replaces, by eye.** 9.6% of (p)'s cost, 58 bars for the best model,
   110, 042 and 093 holding 57%; "not yet looked into page by page". No GPU.
7. **A second set of real scans scored by OMR-NED.** Polish dev's 49 two-staff pages now decide
   post-rules, gold overrides, stopping, architecture and label form, ten pages holding half the
   cost; and OLiMPiC's SER disagrees with it ((m) and T6 better on dev, worse on Polish: SER
   weighs arcs, OMR-NED notes). OLiMPiC's scanned dev pages through `seq_pages.py`, scored by
   OMR-NED against their MusicXML, reported beside Polish dev for every run; a change counts when
   both move the same way.

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
- Vast.ai lists RTX 5090s at about $0.20-0.45 an hour against Runpod Community's $0.69; worth
  setting up for the next batch (needs its CLI and an API key; the pod scripts are plain SSH).

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

## Leakage to watch

- Benchmark music in new data (3-staff PDMX renders go through the held-out and overlap filters).
- Tuning on Polish dev (stopping, architecture, label form): its numbers grow optimistic; the
  sealed half decides.
- The staff count taken from the label: inference takes it from the layout, so staff positions
  for training must come from the renderer or the segmenter.
- Copies across splits (GrandStaff transpositions and distorted copies, OLiMPiC songs in PDMX);
  the segmenter's own training pages, to be checked before its staff positions are used.

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

## Rules that are easy to forget

- Gold must be checked through converter21 (`eval/check_gold.py`), never music21's own kern reader.
- Sealed pages: gold and images only. No system output, no errors, nothing into heuristics or
  training. Corrections to gold are allowed.
- Scoring is plain OMR-NED throughout; only the gold differs between standards.
- Single-page locked scores can swing widely on correct changes (042): check unlocked too.

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
