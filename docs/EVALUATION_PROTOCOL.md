# Evaluation protocol

Written before the final evaluation. Changing anything here after that run is a new
experiment, to be reported as one alongside the original result.

## The claim under test

Zero-shot transcription of real scans of engraved piano music, by OMR-NED, against systems
with public weights. Orchestral, quartet and handwritten music are outside it.

## The system

- Sequence model: the clean retrain. `runs/seq_clean/last.pt`, the final step of its fixed
  schedule, with no checkpoint selection: `best.pt` is chosen on real scans, so it is not used.
- Training data: OLiMPiC synthetic, GrandStaff, PDMX crops without held-out scores or
  `license_conflict` scores. All synthetic. No labelled real scans.
- Pipeline: `eval/seq_pages.py` at the commit that adds this file's frozen tag, reading each
  system at `CROP_HEIGHT_RATIOS` and restoring tuplets (`eval/lmx_tuplets.py`).
- Layout: the v5 U-Net in `cpp/`, used only to find systems.

## What was developed on real scans (disclosed)

No model weights were fitted to real scans, but these choices were made looking at OLiMPiC's
scanned dev split and Mutopia dev: `CROP_HEIGHT_RATIOS`, the rules in `lmx_tuplets.py`,
`rewind_to_measure_start`, and the decision to read twice. The U-Net model was trained with
PDMX scores flagged `license_conflict`, and on handwritten pages (MUSCIMA++).

Earlier looks at held-out data: Polish was run once with run C (OMR-NED 37.24). The final
Polish run is therefore a second look, and is reported as such. Night Album has been scored
for v5 and Transcoda.

## Benchmarks

Each runs once, on the frozen system. Numbers only; errors are not read.

| benchmark | pages | gold | notes |
|---|---|---|---|
| Polish (`btrkeks/polish-scores`) | 112 | corrected `**kern` | also as published and to the print standard (see Polish split) |
| OLiMPiC scanned test | 100 songs | LMX and MusicXML | crops and pages; Zeus is the baseline |
| Mutopia test | 7 pieces | MIDI | note F1, not OMR-NED |
| SMB pianoform, if access is granted | 469 | `**kern` | pages sharing 0.10 or more of their melody fragments with one training piece reported apart |

The synthetic Verovio benchmark is not used for any claim: its sources overlap our training.

## Polish split

Copisteria leads run C on all 112 pages by 10.51, and the cause cannot be found without reading
errors, so Polish is divided by `eval/polish_split.py` into `eval/polish_split.txt`: 56 dev, 56
test, from the gold alone, before any error was read, committed first. Pages sharing melodic
fragments stay on one side; the dataset names no pieces, so this may miss some.

- Dev: errors may be read and fixes tested. It never enters training: the dataset's card says
  evaluation only, and no Polish page or anything derived from one is trained on.
- Test: numbers only, and any Polish claim is made on these 56 pages, for every system.
- Gold changes: `eval/polish_overrides.jsonl`, applied by `eval/gold_overrides.py`, may cover both
  halves, because a wrong page misstates every system's score. Each is decided from the printed
  page, without any system's read, and is kept whichever system it helps. On test pages gold is
  compared with the image only: no system's output or errors are read, and nothing seen there
  feeds a heuristic, a rule or training. Reports say that test gold was reviewed.
- Three scores, always for every system, all OMR-NED, differing only in the gold:
  - As published: the authors' setup, to reproduce their figures.
  - Corrected, the headline: terminators, beam groups left open, and errors under any convention, such as silent staff
    swaps. Gold's conventions stay: notes filed by hand, ottavas folded into pitch.
  - Print standard: Corrected, with notes on the staff they are drawn on and the print's ottavas
    as `*8va` lines, which the scorer's parser (converter21) reads for gold and kern predictions
    alike. The rule: a note belongs to the staff it is drawn on.
- Print standard also gets a column with each page's better score of signed and sounding
  ottavas, so a system that folds a sign into its pitches can be seen fairly. A missed sign with
  the printed pitches costs 2 however long the passage, though it sounds an octave off: reports
  say so. These rules were fixed after dev results and before any test run.
- A number over all 112 pages is reported only with the note that half were developed against.
- The card also says Transcoda used no Polish for selection. Ours does, for its fixes, and
  every report says so.

## Metric

OMR-NED from the authors' `src/evaluation/omr_ned.py` at Transcoda commit d4e2e68, unchanged,
with musicdiff 5.0.2, music21 9.9.1 and converter21 3.5.0. A page that fails to parse, compare
or finish in 900 s, or that has no prediction, scores 100. Scores are means over pages.
MusicXML predictions are compared directly with the `**kern` gold (floor 0.54 on Polish).

Comparisons are paired, with a 95% range from 5,000 resamples of pages (of songs or pieces
where those are the unit). A difference counts only if its range excludes zero. Every page
stays in the mean, including pages with more staves than the system reads.

## Baselines

Each runs with its authors' recommended settings, through the same scorer, and is reported
whether or not it helps our case: Transcoda 59M at beam 3, and every other public system we
can run. A published number is quoted only as published, never as a comparison on corrected
gold. Models fine-tuned on a benchmark are not zero-shot baselines.

## Overlap

Before scoring, each benchmark is checked for copies of training pieces with
`select_mutopia.melody_grams`: the share of a page's or song's fragments found in one training
piece, 0.10 as the threshold, on the known copy in OLiMPiC dev (share 0.81) as positive control.
Results are given with and without flagged pages. Polish: 7 pages flagged, none at 0.26.
OLiMPiC test: 4 songs flagged, none at 0.26.

## Reporting

All rows are published, including those we lose. Predictions and per-page scores go with them.
