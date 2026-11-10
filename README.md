# noteworthy

Optical music recognition: sheet-music PDFs or images in, MusicXML out. A U-Net
(PyTorch) segments symbols; a C++ pipeline reads staves, symbols, rhythm and pitch.
Current scope: notes and rhythm of engraved piano music.

## Reproduce everything

```
git clone --recursive <this repository> noteworthy && cd noteworthy
./reproduce.sh
```

This sets up the Python environments, downloads the sources (about 22 GB, each checked against
a checksum or pinned commit), builds the training data, trains the networks, builds the C++
recognizer, runs the benchmarks and compares with Zeus, the OLiMPiC authors' end-to-end
recognizer (in its own Python 3.11 environment, `.venv-zeus`). Every stage skips work already
done, so it is safe to rerun at any time; an interrupted run picks up where it stopped. Settings
(page and step counts, data locations, worker counts) are environment variables listed in
`scripts/common.sh`; `reproduce.sh` shows a minutes-long smoke configuration.

Reproducible by design: the score lists and every training crop come out byte for byte the same
on any machine and with any worker count (each crop is seeded by its score, never by its
worker), and training is seeded (`train_seq.py --seed`). What still varies between runs is the
GPU's own arithmetic order, so two trainings agree to within run-to-run noise, not bit for bit.
A training resumed with `--resume` draws a new batch order from its step.

Requirements: Linux, an NVIDIA GPU, about 90 GB of disk, and
`git curl unzip cmake g++ libopencv-dev` plus [uv](https://docs.astral.sh/uv/); without a
display, also `xvfb` (MuseScore 3.6.2, which engraves some training crops and is downloaded
by `fetch_data.sh`, opens a window). The MuseScore benchmark also needs MuseScore 4, on PATH
or named by `$MUSESCORE` (under WSL, the Windows install's `MuseScore4.exe` works).

## Recognize a score

```
cpp/build/noteworthy score.pdf --model runs/finetune/model.onnx -o score.musicxml
```

## Layout

- `train/` dataset building (Verovio, MuseScore and Smashcima crops, MUSCIMA++ conversion) and training
- `third_party/smashcima` our fork of [Smashcima](https://github.com/OMR-Research/Smashcima) (Apache-2.0), which draws the handwritten crops; a submodule
- `cpp/` the recognizer
- `eval/` benchmarks and error analysis; `EXPERIMENTS.md` is the experiment log
- `scripts/` the stages `reproduce.sh` runs

## Data and licenses

`reproduce.sh` downloads these; none are redistributed here.

| data | used for | license |
|---|---|---|
| [PDMX](https://zenodo.org/records/15571083) (MusicXML from MuseScore.com) | training renders, benchmarks | public domain or CC0 on MuseScore.com; the sequence model leaves out scores flagged `license_conflict`, as PDMX's authors recommend, while the U-Net model still has them |
| [ASAP](https://github.com/fosfrancesco/asap-dataset) (classical piano MusicXML) | training renders | CC BY-NC-SA 4.0 |
| [MUSCIMA++ 2.0](https://github.com/OMR-Research/muscima-pp) and [CVC-MUSCIMA](http://www.cvc.uab.es/cvcmuscima/) (handwritten pages) | training, handwritten benchmark (its two test sets) | CC BY-NC-SA 4.0 |
| [OLiMPiC](https://github.com/ufal/olimpic-icdar24) (IMSLP scans of [OpenScore Lieder](https://github.com/OpenScore/Lieder) songs, with piano transcriptions) | benchmarks only | CC BY-SA 4.0; the Lieder transcriptions themselves are CC0 |
| [OLiMPiC synthetic](https://github.com/ufal/olimpic-icdar24) (MuseScore renders of Lieder accompaniment systems) | sequence-model training | CC BY-SA 4.0 |
| [GrandStaff](https://github.com/multiscore/e2e-pianoform) images with [GrandStaff-LMX](https://hdl.handle.net/11234/1-5423) transcriptions (piano excerpts) | sequence-model training | LMX: CC BY-SA 4.0; images: no license stated, published for research replication |
| [MusiCorpus 1.0](http://hdl.handle.net/20.500.12800/1-6147) (historical manuscripts, mostly handwritten, with MusicXML transcriptions) | training (its train split); its test split is a sealed benchmark | CC BY-NC-SA 4.0 |
| [MUSCIMA++ 1.0](https://hdl.handle.net/11372/LRT-2372) glyphs and [Moravian Library](https://www.mzk.cz/) paper scans, as Smashcima downloads them | handwritten training crops | MUSCIMA++: CC BY-NC-SA 4.0; paper: not stated by Smashcima (patches of scans from the library's public digital collection) |
| [MuseScore 3.6.2](https://github.com/musescore/MuseScore/releases/tag/v3.6.2) | engraving training crops | GPL-3.0 (the program; its output is ours) |
| [OLiMPiC code and Zeus model](https://github.com/ufal/olimpic-icdar24) (the authors' TEDn metric and end-to-end recognizer) | comparison only (`scripts/compare_zeus.sh`) | code MIT; Zeus model CC BY-SA 4.0 |

ASAP, MUSCIMA++ and MusiCorpus are non-commercial (NC), so models trained by this pipeline
are for non-commercial use. OLiMPiC and GrandStaff-LMX are share-alike (SA): a redistributed
sequence model trained on them must carry the same license. Its scans come from
IMSLP, where works are public domain in Canada and in most countries for these 19th- and
early 20th-century editions, but check local copyright before redistributing images.

Night Album, used as a held-out test, is copyrighted and not included.
