# Staff model: the overnight experiments (2026-11-14)

What runs the night after the race, and why. Results go into `EXPERIMENTS_STAFF_MODEL.md` as
they come in; this file says what each experiment is for and how it is judged. Open work and
later experiments are in `TODO_STAFF_MODEL.md`.

## The questions

1. **Is reading staff by staff better than reading the system?** On PDMX alone StaffSeq2Seq led
   Seq2Seq by about 30%; on the full mix their OLiMPiC dev SER was level at 25,000 steps
   (0.145 against 0.143).
2. **Is the staff arm's gap to seq516 mostly training time?** Polish dev print OMR-NED 11.58 at
   50,000 steps against seq516's 6.97 at 150,000; about twice seq516's errors of every kind.
3. **Are unprinted tuplets a weakness of the staff design, or of training?** The staff arm reads
   1,040 of Polish dev's 1,373 tuplet notes as tuplets, seq516 1,272; 41% of the gap to seq516.
4. **Does the model use its attention across staves, and would more of it help?**
5. **Do crops that reach outward fix the notes beyond the system?** Notes under ottava signs
   are wrong 63% of the time on OLiMPiC dev; by eye, a staff crop cuts off an *8va bassa*.
6. **How large is the noise between seeds?** Without it no 20,000-step comparison can be read.
7. **Does a system model read pitches like the staff model if its rows are read relative to
   one another?** (EXPERIMENTS_STAFF_MODEL.md, "a fixed frame for pitch".)

## Common to every training run

- Data: the race's mix (`train_seq.sh`'s sets of `data/seq516g_data`), with
  `grandstaff_written_v2` in place of `grandstaff_written` (systems with a tie across an edge left
  out). Positions from `staff_positions_v2`.
- 20,000 steps, the whole schedule (warmup 4,800, cosine to 20,000), batch 10, lr 3e-4, seed 0
  unless stated. A 20,000-step snapshot of the 50,000-step run read properly (dev SER 0.185).
- Scored the same way after training: OLiMPiC scanned dev SER (615 crops); `eval/outer_notes.py`
  (notes under signs, far outside, false signs, SER of all 2,844 crops); the tuplet metric
  (below); Polish dev print OMR-NED (locally: it needs the C++ segmenter) and its tuplet notes.
- Local runs on the RTX 5080; others on Runpod pods (one run each, data and code from R2,
  results and checkpoints back to R2, each pod removed when done).

## Phase A: reading what exists (local GPU, about 1.5 h)

**A1. The race's Seq2Seq arm at 50,000 steps** (`runs/race_system`). Polish dev, OLiMPiC dev
readings and their error analysis (`dev_patterns.py`, `edge_checks.py`, `arcs.py`),
`outer_notes.py`. *Answers 1 and 3*: the design comparison at equal data, seed and steps;
whether a system crop also loses unprinted tuplets, and whether it reads notes under signs
better (it reaches 7 interlines beyond the staves, a staff crop 5.5).

**A2. The staff arm on `outer_notes.py`**, the baseline numbers for 5.

**A3. Attention across staves switched off at inference** in the staff arm (each block's
`across_staves` layer skipped). *Answers half of 4*: whether the trained model relies on the
layer. If errors rise, it does; if not, the decoder may do the aligning itself (it reads every
staff's grid), and only training without the layer (T5) says whether it is worth having.

**A4. seq516's 50,000 and 100,000-step snapshots on Polish dev.** *Answers 2*: if seq516 at
50,000 steps was near 11.6, the staff arm sits on the same learning curve, earlier. (Rough:
seq516's schedule was 192,000 steps long, so at 50,000 its rate was still high.)

## Phase B: 20,000-step training runs

| Run | What | Where | Answers |
|---|---|---|---|
| T0 | StaffSeq2Seq baseline, seed 0 | Runpod | the reference for all |
| T0b, T0c | StaffSeq2Seq baseline, seeds 1 and 2 | Runpod | 6 |
| T1 | E1: outer staves' crops reach outward | Runpod | 5 |
| T2 | E3: two encoder blocks | Runpod (Secure) | 4 |
| T3 | E4: attention across staves over neighbouring columns | Runpod | 4 |
| T4 | Seq2Seq baseline, seed 0 (dropped, then back as T10's baseline) | Runpod | 7 |
| T5 | StaffSeq2Seq without attention across staves | Runpod | 4 |
| T6 | StaffSeq2Seq with the `unmarked` crops at three times their weight | Runpod | 3 |
| T7 | StaffSeq2Seq, 100,000 steps (the whole schedule) | Runpod | 2 |
| T8 | StaffSeq2Seq given the meter in force (`--meter-context`) | local | 3 |
| T9 | StaffSeq2Seq at batch 20, lr 4.2e-4 (twice the samples a step) | Runpod | 2 |
| T10 | Seq2Seq with relative rows (`--relative-rows`) | Runpod | 7 |

Runpod runs on RTX 5090s (Community Cloud unless noted), one run a pod, about 1.25 h each (58
samples/s on the smoke test, against 33 on the local RTX 5080). T8 on the local GPU after Phase A,
a fourth seed (T0d) in its place if its code were not ready; GPU training is not bit for bit
repeatable even on one card, so one run on the 5080 beside 5090 baselines is read with the seeds'
spread in mind. Then Polish dev for each run as it comes home (local: the C++ segmenter).

**T0, T0b, T0c: the baseline three times.** Their spread is the noise any other run must
beat. Targeted metrics (notes under signs, tuplet bars) should move by more than it; overall SER
may not.

**T1 (E1): crops that reach outward** (`--staff-outward 1`). A staff crop stays 15 interlines
tall at 256 px, but the top staff's window slides up one interline and the bottom staff's down:
6.5 interlines beyond the outer line (was 5.5), 4.5 inside. The split comes from the gap between
a system's staves, inner line to inner line, over the training positions: median 4.7 (GrandStaff)
to 7.6 (OLiMPiC synthetic) interlines, 95th percentile 6.7 to 11.3; two inner sides of 4.5 cover
a gap of 9, the 95th percentile of most sets. Middle staves stay centred. *Judged by* `outer_notes.py`: notes under signs and notes
far outside, wrong, against T0/T0b; false ottava tokens and SER of all crops not worse.
*Expected*: a large fall on those notes, nothing elsewhere.

**T2 (E3): two encoder blocks** (`--blocks 2`, 29.6M parameters against 26.0M). Within each staff, across the staves, within, across: what
crosses at one column spreads along the other staff. About two more encoder layers' worth of
parameters. *Judged by* tuplet bars and per-staff note and rhythm errors, then SER.

**T3 (E4): attention across staves over neighbouring columns** (`--across-columns 5`). The
attention across staves takes groups of 5 neighbouring columns together (a column is about an
interline at 256 px, so a group is about five interlines, a beat at ordinary spacing), not one
column alone, so three notes in one hand can be weighed against a beat in the other a little to
the side. Groups side by side, not a window around each column: the layer stays a plain one over
a reshaped grid, at the cost of group edges. *Judged as* T2.

**T4: Seq2Seq at 20,000 steps.** The race's other arm on this night's data and schedule.
*Answers 1 and 3* at the length every other run has: does the design comparison hold, and does
a system model lose unprinted tuplets as early in training.

**T5: no attention across staves** (`--across-columns 0`, 24.2M parameters). Each staff encoded
alone; only the decoder sees both.
With T0 and T2/T3, a dose-response: no mixing, one layer at one column, one layer over several
columns, two layers. *Answers 4*: if T5 matches T0, the decoder already aligns the staves and
more mixing in the encoder is unlikely to help.

**T6: more crops with unnumbered tuplets.** The `unmarked` set (systems with tuplets, numbers
hidden after the first group) at weight 3 in the mix. *Answers 3* by data alone: if hidden-number
bars read better (tuplet metric) with the rest unchanged, the weakness is what the model sees,
not how it is built.

**T7: 100,000 steps.** The staff model's whole schedule stretched to 100,000 steps. With T0
(20,000) and the race (50,000, GrandStaff before the tie fix), a learning curve of one design:
*answers 2* more directly than seq516's mid-schedule snapshots.

**T8: the meter in force.** As Seq2Seq's `--meter-context`: the label starts
`meter beats:N beat-type:M`, read and never written; in training from the label's bars (15%
without, as a page's first system), on dev from the label, on Polish from the time signatures
read earlier on the page (seq_pages.py reads twice). *Answers 3*: if a system's bar length is
what the model lacks to tell unnumbered triplets from plain notes, hidden-number bars improve.

**T9: batch 20.** Twice the samples a step at the same 20,000 steps, the learning rate up by
the square root of two. With T7: whether more training helps by samples seen or by steps, and so
whether the long run could reach its quality in fewer hours.

**Launched** 04:47 UTC, one RTX 5090 pod a run (`runs/overnight_2026-11-14/pods.tsv`): nine on
Community Cloud at $0.69/h, T2 on Secure at $1.39/h; about $7.60/h while all run. T7 first (100,000
steps, deadline 7 h). Each pod pushes its run to R2 every 20 minutes and removes itself at its
deadline; the session deletes each when its run is home. Incidents: T0c's first pod rejected the
SSH key and T4's first two (one host) had no usable CUDA, all deleted and replaced (T4 on Secure);
and the drivers' launch over SSH hung, waiting on the background run's stdin (fixed with `ssh -n`),
so at 05:32 eleven drivers were stopped, their runs untouched (started under nohup: training went
on), and `pod_watch.sh` took over the watching.

**T10: relative rows, against T4.** Seq2Seq without its rows' own embedding (where a row is in
the crop); its encoder's attention gets a learned bias by the vertical distance between two rows
instead, the same in every column and head. Heights are then read relative to one another, so a
staff means the same wherever it sits, as in a staff crop. First built as a learned bias added to
the attention scores: out of memory at step 500 (a bias over every pair of a wide crop's
positions, about 5 GB a layer, and PyTorch's fused attention given up: 27 samples/s against T4's
56). Rebuilt as rotary position embedding on the row axis (`RowRotaryEncoderLayer`: queries and
keys rotated by their row, so a score depends on rows apart only, with fused attention kept) and
restarted on the same pod at about 05:45 UTC. *Answers 7*, judged against T4 on
pitches a step off and the bass staff (OLiMPiC dev), then SER and Polish. T4 was dropped when the
race answered its first purpose and is back as T10's baseline. Launched 05:05 UTC.

## The tuplet metric

On OLiMPiC dev's written labels (`data/olimpic_written_dev`, synthetic and scanned): the bars
whose label has a tuplet, and of those the share read with the right rhythm (each note's
duration and tuplet ratio as labelled, by bar, aligned per bar). Reported apart: bars whose
tuplets are numbered in the image and bars where a number is hidden (`tuplet:hidden` in our
labels), since the second is the failure seen on Polish. On Polish dev: notes in tuplets read as
tuplets (`tuplet_notes.py`).

## Not tonight

The 150,000-step staff run (about 13 h locally): it is what these answer for. Unprinted tuplets
as an experiment of their own (E2) waits for A1 and T4: if the system model loses them too, the
cause is training length or data, not the staff crops.

## Why is the staff model better? Walking Seq2Seq toward it (2026-11-14, afternoon)

At 50,000 steps the staff model leads Seq2Seq by about 15% on OLiMPiC dev and 10% on Polish,
pitches a step off by 2x. Not resolution as such: a 256 px staff crop has the pixels per
interline of a typical two-staff system at 512. What it has and Seq2Seq lacks: (1) a constant
scale, (2) a fixed frame (each staff at the same rows of its crop, a staff embedding), (3)
attention within a staff, then across staves by column, (4) staves through the CNN apart, the gap
between them seen twice. Each run adds one of these to Seq2Seq (T4's settings otherwise: the
race's mix, `--staff-samples-only`, 50,000 steps, no meter); where the gap closes says which
matters. The last (g), if it matches the staff model, is a system model for any number of staves.

| Run | Change | Tests | Where |
|---|---|---|---|
| (c) `c_interline_pure_50k` | `--interline-scale` (absolute rows) | 1 | pod, RTX PRO 6000 |
| (f) `f_staff_anchored_50k` | (c) plus `--staff-anchored`: each row embedded by its nearest staff and its offset from that staff's centre, in interlines | 2 | pod |
| (g) `g_staff_bands_50k` | (f) plus `--staff-bands`: one layer within each staff's rows, one down each column | 3 | pod |
| `t4b_system_seed1_50k` | T4 at seed 1 | the noise floor of Seq2Seq comparisons | local |
| `j_staff_jitter_50k` | staff model, `--staff-jitter 1.5`: training crops centred up to 1.5 interlines off their staff | 2, from the staff side | local, after T4b |

Read by: dev SER, `category_errors.py` (pitch, a step off via `targeted_metrics.py`), tuplet bars,
the layout probe (does (c) flatten it?), Polish dev.
