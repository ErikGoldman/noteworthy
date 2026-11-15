# To do: the staff model (StaffSeq2Seq)

Status as of 2026-11-13, late evening (UTC). Results and evidence are in
`EXPERIMENTS_STAFF_MODEL.md`; the older pipeline's list is `TODO.md`.

## Running now (night of 2026-11-14)

The overnight experiments of `EXPERIMENTS_STAFF_MODEL_PLAN.md`:
- Local queue (`runs/overnight_2026-11-14/overnight_local.sh`, log `overnight_local.log` there): waits for the race's
  Seq2Seq arm (done about 04:15 UTC), then Phase A (A1-A4), then T0, T0b, T1, then Polish dev for
  every 20,000-step run home.
- Runpod (RTX 5090, Secure Cloud, $1.39/h each): T2, T3, T4, T5, one pod each, driven by
  `runs/overnight_2026-11-14/pod_run.sh` and `pod_side.sh` (code over SSH, data from R2, run copied home); pods created once the
  R2 uploads end (about 03:45 UTC) and deleted by the session as each finishes.
- R2: `data/seq516g_data` (19.2 GB) and, separately, `grandstaff_written_v2`, its positions file,
  `olimpic_written_dev` and OLiMPiC's code.

## When the Seq2Seq arm finishes

1. Score it on Polish dev as the staff arm was (`runs/overnight_2026-11-14/polish_at.sh race_system 50000`) and
   its OLiMPiC dev readings (`dev_predictions.py`, `dev_patterns.py`, `edge_checks.py`), then the
   same error analysis: what belongs to staff crops and what to this mix and length.
   In particular: does the system model also lose unprinted tuplets, and does it read notes
   under ottava signs better?
2. Both arms on `eval/outer_notes.py` (the outer-notes set below): the baseline numbers for the
   crop experiment, and the first answer on whether a system crop reads those notes better.
3. **Ablation: is the attention across staves used?** (Step 4 again, on a model that reads; on
   the overfit model it was inconclusive.) Read the staff arm's 50,000-step snapshot with each
   block's `across_staves` layer skipped (the grid passes through unchanged), against the intact
   model, on the OLiMPiC scanned dev crops (`dev_predictions.py`, `dev_patterns.py`) and
   `outer_notes.py`. Compare overall SER, notes and rhythm per staff, and bars with tuplets
   (bar fill, onsets); Polish dev too if the crops show a difference. If nothing moves, the model
   does not use the layer, and E3/E4 ask whether it would with more of it. An inference-only
   switch (no retraining): about 10-15 minutes of GPU reading.
4. Log both arms and the ablation in EXPERIMENTS_STAFF_MODEL.md.

## Next: Seq2Seq for systems of any number of staves (decided 2026-11-14)

The user leans to Seq2Seq unless the staff model shows compelling gains soon; the meter goes in;
attention is to be explored so that systems of more than two staves can be read.

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

## Experiments to run

Each is a short run against one shared baseline, judged on a metric aimed at what it changes;
overall SER only as a check that nothing else got worse (seed noise swamps changes under 1% of
the edits). Short runs: 20,000 steps with the whole schedule (warmup 4,800, cosine to 20,000),
the race's data and seed, about 1 h 45 min each on the local GPU. A 20,000-step snapshot of the
50,000-step run was at dev SER 0.185 and reads properly; a schedule that ends there anneals and
should do a little better. Winners go into the 150,000-step staff run (about 13 h locally).

- **Baseline**: StaffSeq2Seq as raced, 20,000 steps. Shared by every experiment below.

- **E1: outer staves' crops reach outward.**
  - *Why.* A staff crop is STAFF_INTERLINES (15) tall, centred on its staff: 5.5 interlines of
    room each side. Outside the system that cuts off ottava signs and the highest and lowest
    notes. In OLiMPiC dev crops with an octave shift, notes three or more ledger lines outside
    are wrong 63% of the time (12% elsewhere); by eye, 5989993 p2-s3's *8va bassa* is cut off by
    the bass staff's crop. Between the staves no dev note is more than three ledger lines out,
    and both staves' crops see that gap.
  - *Change.* Keep 15 interlines and the 256-px staff scale; slide the window outward for the
    top and bottom staves, about 7 interlines beyond the outer line and 4 on the inner side (7 =
    seq_pages.py's MARGIN_INTERLINES: the staff crops then see what a system crop sees). Middle
    staves of three-staff systems stay centred. In `seq.staves.staff_crops` /
    `seq.staff_data.cut_staves`, as a training option recorded in the checkpoint's args, so
    `seq_pages.py`, `outer_notes.py` and the dev check cut each model's crops as it trained.
  - *First.* Measure the gap between a system's staves (inner line to inner line) over the
    training sets' staff positions, so the inner sides still overlap for dynamics, hairpins and
    cross-staff beams; choose the split from that, not 7/4 blind.
  - *Tests.* A drawn system with marks 6.5 interlines above the top staff and below the bottom
    one shows them in the outer crops; the inner sides still overlap at the measured gap.
  - *Metric.* `eval/outer_notes.py`: notes under signs (948 notes in 102 crops) and notes three
    or more ledger lines outside the system (3,606 in 1,022 crops) read wrong; false ottava
    tokens; SER over all 2,844 crops, and all other notes, unchanged.
  - *Expected.* Small overall (far-outside notes are under 1% of OLiMPiC dev edits; Polish
    octave errors 116 against seq516's 80), large on pages with ottava passages.

- **E2 (to design): unprinted tuplets.** The largest single cause found: a run of triplets or
  sextuplets numbered only on its first group or two is read as plain notes after that, every
  later onset shifting (Polish 003: 29.5 against seq516's 1.4). 41% of the staff model's gap to
  seq516 on Polish dev. Wait for the Seq2Seq arm: if it loses them too, it is training length or
  data (the `unmarked` crops), not staff crops. Metric to build: bar-fill or onset errors in
  bars with tuplets, on a dev set of systems whose later tuplets are unnumbered.

- **E3: two encoder blocks (`blocks=2`).** Today one block: attention within each staff, then
  across the staves at each column, then the decoder. What crosses between the staves at one
  column is never spread along the other staff. A second block mixes again (within, across,
  within, across), so an alignment seen at one column reaches the notes around it. Costs two
  more encoder layers (about +3.5M parameters, slower steps; measure). Metric: tuplet bars
  (E2's), notes and rhythm SER per staff, cross-staff passages; overall SER as the check.
  Worth running whatever the ablation shows, but first if the across-staves layer matters.

- **E4: attention across the staves over neighbouring columns.** Today a position attends only
  to the other staves' rows at exactly its own column (about one interline wide), while
  Seq2Seq's encoder attends across the whole grid. Widen the across-staves attention to a window
  of columns either side (e.g. +-2, about four interlines: a beat's worth of notes at ordinary
  spacing), so three notes in one hand can be weighed against one beat in the other a little to
  the side. A windowed mask over the (staves x rows x columns) layout, or the across-staves
  layer run on groups of 2k+1 columns. Choose the window from the spacing of a beat in
  the training crops. Same metrics as E3; E3 and E4 can also be combined if both help.

- **Data for the next runs, every arm alike:** `grandstaff_written_v2` in place of
  `grandstaff_written` (systems with ties across an edge left out; EXPERIMENTS_STAFF_MODEL.md,
  "Ties and slurs"). The baseline must use it too, so experiments compare like with like.

- **Hide tuplet numbers as engravers do** (EXPERIMENTS_STAFF_MODEL.md, "When real scores hide a
  tuplet's number"). `build_seq_dataset.hide_tuplet_numbers` today: one system in five with
  tuplets, every number hidden (half the time keeping the first), whatever the tuplet. Real
  MuseScore scores choose per score (about three in four show all, a few hide all), and the
  rest hide by place in a run of like groups (about half at the first, 85-89% from the fifth),
  print again at a voice's first tuplet in a system and for a lone tuplet in a bar, and mostly
  keep 5s and 7s printed. Rewrite the rule from those rates and rebuild the synthetic sets;
  `unmarked` stays as the stress set. Metric: tuplet bars with a number hidden, false tuplet
  tokens on plain notes; Polish 003. Low priority: Seq2Seq already reads OLiMPiC's real hidden
  triplets (98.6% at a group's first note, labels fed).

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

## Done this session

- Line finder fixed and committed (63739b1); positions for every crop in `staff_positions_v2`.
- Staff arm, 50,000 steps: OLiMPiC dev SER 0.098, Polish dev 11.58 print (seq516: 6.97).
- `scripts/r2_data.py`: push and pull through R2, round-trip tested (not committed).
