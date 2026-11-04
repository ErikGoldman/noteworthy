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
builds the C++ recognizer and runs the benchmarks. Finished steps are recorded in
`data/.stamps`, so an interrupted run picks up where it stopped. Settings (page and
step counts, data locations) are environment variables listed in `scripts/common.sh`;
`reproduce.sh` shows a minutes-long smoke configuration.

Requirements: Linux, an NVIDIA GPU, about 40 GB of disk, and
`git curl unzip cmake g++ libopencv-dev` plus [uv](https://docs.astral.sh/uv/).
The MuseScore benchmark also needs MuseScore 4 (under WSL, its Windows install).

## Recognize a score

```
cpp/build/noteworthy score.pdf --model runs/finetune/model.onnx -o score.musicxml
```

## Layout

- `train/` dataset building (Verovio renders, MUSCIMA++ conversion) and training
- `cpp/` the recognizer
- `eval/` benchmarks and error analysis; `EXPERIMENTS.md` is the experiment log
- `scripts/` the stages `reproduce.sh` runs

MUSCIMA++ is CC BY-NC-SA 4.0, so models trained with it are for non-commercial use.
