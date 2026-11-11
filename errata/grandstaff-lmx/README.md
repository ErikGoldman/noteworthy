# GrandStaff-LMX errata: lost clefs, keys and meters

GrandStaff-LMX ([Mayer, Straka, Hajič jr., Pecina 2024](https://hdl.handle.net/11234/1-5423),
release 2024-02-12) gives LMX and MusicXML labels for the 53,882 systems of
[GrandStaff](https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz). In 20,757 of them (39%) the labels'
clefs, keys or meters disagree with the `**kern` source the system's image was engraved from, and
with the image. The notes' pitches are right (LMX spells them absolutely), but:

| Systems whose labels | train | dev | test | all |
|---|---|---|---|---|
| lack a clef change the source and image have | 10,369 | 1,181 | 1,927 | 13,477 |
| have a staff's opening clef wrong | 2,660 | 284 | 492 | 3,436 |
| have the opening key wrong (mostly C major) | 3,688 | 419 | 718 | 4,825 |
| lack a key change | 833 | 103 | 154 | 1,090 |
| lack the opening meter, or have a later one in its place | 4,357 | 482 | 835 | 5,674 |
| lack a meter change | 202 | 21 | 41 | 264 |
| draw accidentals the image does not (or miss ones it does) | 5,851 | 645 | 1,077 | 7,573 |
| any of these: mended by this patch | 15,980 | 1,797 | 2,980 | 20,757 |
| (all systems) | 41,598 | 4,623 | 7,661 | 53,882 |

The accidentals follow from the keys: MuseScore, normalising the MusicXML, wrote out every note's
accidental against the key it was given. The mended labels have 107,206 fewer accidental tokens
(and 1,632 more: naturals the real key needs). A model trained on the release learns to read past
clefs, keys and meters it can see, and to write accidentals no image draws.

## What is wrong

1. **No clef, key or meter change survives** inside a system, though the images draw them.
2. **When a staff's spine splits into voices (`*^`) before its signatures**, that staff loses them:
   its clef becomes treble, its key C major, and the meter is lost. The system's key and meter
   tokens come from the first staff, so a split on the upper staff (the right spine) loses the
   system's key and meter, and one on the lower staff gives that staff C major: its every
   flat or sharp is written out as an accidental.

For example, `beethoven/piano-sonatas/sonata05-1/maj2_down_m-97-101` is in five flats and 3/4
(image and source); its label has `key:fifths:0`, no time signature, and a flat on every B, E, A,
D and G. `beethoven/piano-sonatas/sonata01-2/maj2_down_m-10-15`: the lower staff opens in bass clef
and changes to treble a beat into measure 3; the label has `clef:G2 staff:2` and no change.

## Why

GrandStaff-LMX was made by reading the `**kern` with music21's Humdrum parser, exporting a crude
MusicXML with music21, normalising it with MuseScore 3.6.2 and linearizing it
([`app/datasets/grandstaff/build.py`](https://github.com/ufal/olimpic-icdar24/blob/f4f45ac523ebe09832342270a68b1b4368f63da2/app/datasets/grandstaff/build.py)).
The loss is in music21 (reproduced with 9.9.1; the version used in 2024 is not recorded):

- The parser puts a change at offset 0 of the first measure, beside the opening signature, or
  outside any measure, instead of where it falls. The MusicXML export writes neither.
- A signature after a spine split is stored in each of the split's voices rather than in the
  measure, and the export does not find it there.

Any other use of `ConverterHumdrum` followed by a MusicXML export may lose them the same way.

## The patch

`lmx.patch.gz` mends the `.lmx` of the 20,757 systems, as a unified diff of the release's files;
`musicxml.patch.gz` (28 MB, so not kept here: MuseScore lays each mended system out again, which
moves every coordinate) mends their `.musicxml`, and `make_patch.sh` makes it. To apply them, in
the folder holding GrandStaff-LMX:

```
zcat lmx.patch.gz | patch -p1
zcat musicxml.patch.gz | patch -p1
```

The mended files come from OLiMPiC's own build, run through its own functions with one change
(`train/rebuild_grandstaff_lmx.py`): after music21's parse, every clef, key and meter music21 made
is taken out and the source's put in each staff's measures, at the offset the source has them.
The source is read by `train/seq/kern.py`: spines followed through splits and merges, a data line
starting when the shortest note sounding on the line before it ends, grace notes taking no time;
a signature with no notes after it (a courtesy clef at the system's end) is not a change.
MuseScore 3.6.2 and OLiMPiC's Linearizer then run as in the original build.

Checks:

- **The build reproduces the release.** Run unchanged (music21 9.9.1, MuseScore 3.6.2 with default
  settings), OLiMPiC's build gives back the release's `.lmx` and `.musicxml` byte for byte (40 of
  40 systems tried). With the change, the 32,626 systems it does not mend come out byte for byte
  the release's.
- **Nothing but signatures changes.** In every mended system the LMX differs from the release's
  only in clef, key and meter tokens, accidentals, stem directions (MuseScore turns a stem by where
  the note sits, which a clef moves), and the backups and forwards MuseScore writes to reach a
  clef's beat when no note of its staff starts there. 54 systems that changed more are not
  mended (the 10 looked at: voice and staff bookkeeping).
- **The notes match the source,** measure by measure and staff by staff (grace notes aside).
- **By eye:** seven mended systems drawn by Verovio beside GrandStaff's images: clefs, keys,
  meters and accidentals as in the image, mid-measure clef changes too, but for one reminder
  natural (below), which the release has as well.
- **The same output twice:** two runs give byte-identical mended files.
- The patches applied to a copy of the release give the mended files exactly (41,514 of 41,514).

`make_patch.sh` makes both patches and `systems.tsv.gz` again from the two downloads.

## Other problems found, not mended

`systems.tsv.gz` lists every system with its split and one of these:

| Status | Systems | |
|---|---|---|
| correct | 32,626 | the build with the change gives back the release |
| mended | 20,757 | in the patch |
| notes differ from the source | 243 | pitches by measure and staff (grace notes aside) are not the source's |
| measure count differs from the source | 136 | not looked into |
| the rebuild changes more than signatures, accidentals and stems | 54 | voice and staff bookkeeping, in the 10 looked at |
| a staff's first clef, key or meter comes after notes in the source | 54 | the opening signature is not in the source |
| the source's spines cannot be followed | 12 | a voice of one staff merges into the other's spine |

We leave all but the first two out of training. Of the systems whose notes differ, two were checked
against their images, and in both the label moves notes between measures:

- `joplin/joplin/bethena/maj3_down_m-88-91`: the label's first measure on the lower staff holds
  the last measure's notes.
- `beethoven/piano-sonatas/sonata12-3/maj3_up_m-12-16`: three notes of measure 3's lower staff are
  in measure 2.

Both have spines that split and merge within the system, as do most of the 243.

Three more things we noticed and did not mend:

- **Reminder accidentals.** `beethoven/piano-sonatas/sonata18-1/maj2_up_m-232-235` has an A♭ in
  measure 1 and an A in measure 2; the label (release and mended alike) gives the A a natural
  the image does not draw. How often the labels have reminder accidentals the images lack is open.
- **Grace notes.** 3,029 training systems have grace notes in their source; 2,040 of their labels
  have none, and 491 fewer than the source. The images draw them: in all 8 checked by eye, each
  grace note is a stemless notehead (often under a short stray dash, a broken beam), whether the
  label has it or not, e.g. `beethoven/piano-sonatas/sonata06-1/min3_up_m-126-129` (five drawn,
  none labelled). We leave every system with a grace note in its source out of training.
- **Tuplet numbers.** Labels say a tuplet's number is shown where images often draw none (the
  later groups of a run, or all of them) and draw a sextuplet as one 6 where the label has two
  triplets: about a third of 16 systems with tuplets, checked by eye.

## Blank images

Not a label problem, but found the same way: 29 of GrandStaff's images are blank, one flat grey
or nearly all black (25 of them distorted ones, the camera simulation's), though their labels
have notes. We leave a system out of training when the image we use is one of them:

- `beethoven/piano-sonatas/sonata02-4/maj3_up_m-76-79_distorted.jpg`
- `beethoven/piano-sonatas/sonata04-3/original_m-103-106_distorted.jpg`
- `beethoven/piano-sonatas/sonata08-1/original_m-46-49_distorted.jpg`
- `beethoven/piano-sonatas/sonata09-1/original_m-157-161_distorted.jpg`
- `beethoven/piano-sonatas/sonata11-2/maj3_down_m-1-6.jpg`
- `beethoven/piano-sonatas/sonata11-2/maj3_down_m-1-6_distorted.jpg`
- `beethoven/piano-sonatas/sonata11-2/min3_down_m-1-6.jpg`
- `beethoven/piano-sonatas/sonata11-2/min3_down_m-1-6_distorted.jpg`
- `beethoven/piano-sonatas/sonata11-2/original_m-1-6.jpg`
- `beethoven/piano-sonatas/sonata11-2/original_m-1-6_distorted.jpg`
- `beethoven/piano-sonatas/sonata13-1/maj3_up_m-25-28_distorted.jpg`
- `beethoven/piano-sonatas/sonata13-2/maj2_down_m-52-55_distorted.jpg`
- `beethoven/piano-sonatas/sonata18-1/maj3_down_m-247-250_distorted.jpg`
- `beethoven/piano-sonatas/sonata21-3/maj3_down_m-676-679_distorted.jpg`
- `beethoven/piano-sonatas/sonata23-1/maj3_up_m-165-168_distorted.jpg`
- `beethoven/piano-sonatas/sonata23-1/min3_down_m-136-140_distorted.jpg`
- `beethoven/piano-sonatas/sonata25-1/original_m-190-193_distorted.jpg`
- `beethoven/piano-sonatas/sonata25-3/maj2_up_m-49-53_distorted.jpg`
- `beethoven/piano-sonatas/sonata32-2/original_m-52-56_distorted.jpg`
- `chopin/mazurkas/mazurka07-4/min3_up_m-36-40_distorted.jpg`
- `hummel/preludes/prelude67-06/original_m-1-4_distorted.jpg`
- `joplin/joplin/harmony/maj3_down_m-103-106_distorted.jpg`
- `mozart/piano-sonatas/sonata01-2/min3_down_m-60-63_distorted.jpg`
- `mozart/piano-sonatas/sonata02-3/min3_down_m-75-78_distorted.jpg`
- `mozart/piano-sonatas/sonata06-1/min3_up_m-1-5_distorted.jpg`
- `mozart/piano-sonatas/sonata10-1/original_m-43-46_distorted.jpg`
- `mozart/piano-sonatas/sonata17-3/maj3_down_m-21-26_distorted.jpg`
- `scarlatti-d/keyboard-sonatas/L319K442/min3_down_m-61-64_distorted.jpg`
- `scarlatti-d/keyboard-sonatas/L342K220/min3_up_m-5-9.jpg`

## Empty bars after a merger of the staves

Also an image problem. In 68 systems (50 train, 11 dev, 7 test) the source merges the two staves'
spines into one (a `*v\t*v` line) with notes after it. Verovio drew those images with no notes
after the merger, only empty measures (no rests either), and OLiMPiC's build cuts the source at
the merger on purpose (`_load_kern_file`), so the label's notes are the image's, but the image
draws bars the label lacks. `beethoven/piano-sonatas/sonata07-3/maj2_down_m-35-40`: a measure of
notes, then three empty bars. `systems.tsv.gz` has them as correct (the label is what the build
gives); we leave them out of training. They are the systems with a `*v\t*v` line followed by a data line.

## Licence

GrandStaff-LMX is CC BY-SA 4.0; this patch and `systems.tsv.gz`, derived from it, are too.
