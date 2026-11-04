# noteworthy

Optical music recognition: sheet-music PDFs or images in, MusicXML out. A U-Net
(PyTorch) segments symbols; a C++ pipeline reads staves, symbols, rhythm and pitch.
Current scope: notes and rhythm of engraved piano music.

## Reproduce everything

```
./reproduce.sh
```

This sets up the Python environment, downloads the training sources (PDMX, ASAP,
MUSCIMA++; about 6 GB), renders the synthetic training pages, trains the network,
builds the C++ recognizer and runs the benchmarks. Every stage skips work already
done, so it is safe to rerun at any time; an interrupted run picks up where it stopped. Settings (page and
step counts, data locations) are environment variables listed in `scripts/common.sh`;
`reproduce.sh` shows a minutes-long smoke configuration.

Requirements: Linux, an NVIDIA GPU, about 40 GB of disk, and
`git curl unzip cmake g++ libopencv-dev` plus [uv](https://docs.astral.sh/uv/).
The MuseScore benchmark also needs MuseScore 4, on PATH or named by `$MUSESCORE` (under
WSL, the Windows install's `MuseScore4.exe` works).

## Recognize a score

```
cpp/build/noteworthy score.pdf --model runs/finetune/model.onnx -o score.musicxml
```

## Layout

- `train/` dataset building (Verovio renders, MUSCIMA++ conversion) and training
- `cpp/` the recognizer
- `eval/` benchmarks and error analysis; `EXPERIMENTS.md` is the experiment log
- `scripts/` the stages `reproduce.sh` runs

## Data and licenses

`reproduce.sh` downloads these; none are redistributed here.

| data | used for | license |
|---|---|---|
| [PDMX](https://zenodo.org/records/15571083) (MusicXML from MuseScore.com) | training renders, benchmarks | the scores used are public domain or CC0 |
| [ASAP](https://github.com/fosfrancesco/asap-dataset) (classical piano MusicXML) | training renders | CC BY-NC-SA 4.0 |
| [MUSCIMA++ 2.0](https://github.com/OMR-Research/muscima-pp) and [CVC-MUSCIMA](http://www.cvc.uab.es/cvcmuscima/) (handwritten pages) | training | CC BY-NC-SA 4.0 |
| [OLiMPiC](https://github.com/ufal/olimpic-icdar24) (IMSLP scans of [OpenScore Lieder](https://github.com/OpenScore/Lieder) songs, with piano transcriptions) | benchmarks only | CC BY-SA 4.0; the Lieder transcriptions themselves are CC0 |

ASAP and MUSCIMA++ are non-commercial (NC), so models trained by this pipeline are for
non-commercial use. OLiMPiC is share-alike (SA): anything redistributed that is derived
from it must carry the same license; it is only evaluated on here. Its scans come from
IMSLP, where works are public domain in Canada and in most countries for these 19th- and
early 20th-century editions, but check local copyright before redistributing images.

Night Album, used as a held-out test, is copyrighted and not included.
