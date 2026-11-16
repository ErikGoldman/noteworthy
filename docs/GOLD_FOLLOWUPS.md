# Sealed-mark follow-ups (gold questions raised by mark agents; image-only fixes allowed)
- 002 bars 3-5: merge at line 39 (`*v *v *`) joins lower staff with upper voice 1; col 1 of bar 5 holds lower-staff notes. Print shows plain eighths where kern has triplets (no 3 printed). Check crossing merge / swap.
- 004 bar 1: agent saw `16r` + sixteenth in RH where print shows even eighth chords. My look: bar 1 RH chords also carry staccato dots (agent left out as unclear), under an 8va dashed line ending "loco" before bar 2. Add dots + check ottava + rhythm.
- 006 bar 21: middle RH chord has a dot right of the top note; gold has an undotted quarter. Possible missing augmentation dot (check bar fill).
- 009 bar 9: LH GG prints as an eighth; gold has 16GG + 16r.
- 009 bar 9: LH GG prints as an eighth; gold has 16GG + 16r.
- 008: gold has no *clef lines at all (print: F4/G2). Clefs score as edits; check other pages for missing clefs.
- 014 bar 33: staccato dot placed on LH B (line 300 col 1); could belong to the ledger f# in col 0. Bar 17 line 155 slur ends mid-phrase.
- 025 bar 12 on: print beams RH in eighth pairs, gold has triplet eighths.
- 027 bar 14: accented RH octave looks like F in print, gold g/gg. Bar 17: small > between staves left unmarked.
- 020 bar 16: gold ties the RH chord; print shows a slur with staccato on the 2nd chord (tie may be a slur).
- 036 bar 19 line 237: final GGG written twice (8GGG lower col, 4GGG upper col); print shows one eighth notehead marked 'r.' Dot put on col 0.
- 046 bar 3 LH: print 8E- 8r 4r 4b- (like bar 7); gold 4E- 4r 4b-.
- 031 bars 3, 17, 19, 21: gold gives single notes where print shows two-note chords. Bar 33: accent under beat 1 hidden by ink blot, unmarked.
- 089 bars 17-20: gold triplet eighths where print shows plain eighths, no 3; LH voicing differs from print. (Same pattern as 002, 025: spurious triplets.)
- 084 bar 13 (handwritten, lines 132-135): three small open '∠' marks above RH 1st quarter and both LH quarters; not '>'. Copyist's accent or tiny hairpins? Left unmarked: DONE 2026-11-07: the user says accents; three entries added.
- 074 bar 7: opening chord tied from bar 6 in gold, but print puts a staccato on it (tie may be a slur). Bar 10 last RH chord: dot right of lowest note, possibly a missing augmentation dot.
- 101 bars 18, 20 (lines 140, 156): print splits the RH octave into voices (upper half note, 8th rest, lower half); gold has one chord field.
- 099 FIXED 2026-11-07 (end_after at line 266, corrected standard). Was: gold has 20 bars, print 16. Gold bars 17-20 (lines 267-339) look like the next page: candidate end_after (print standard? or corrected). Bar 15 accented chord on ledger lines below treble; gold BB E B may be an octave low.
- 094 bar 5: gold ties the two dotted chords; print reads as slur with portato dots.
- 099: four short 8va passages (bars 2, 4, 6, 8: dotted `8`) not yet in Print standard gold.
- 110 last system (bars 25-28, gold_terminated lines 170-225): gold does not match the print, so adding `*8va` alone would not help. Print: upper `8` from bar 25 to the end of bar 27, lower `8` from bar 26 through bar 28's first chord. Each triplet is a lower-staff chord (ledger lines, natural in bar 25) beamed across to two upper-staff notes (bar 25: D♮5 then F5, written). Gold: the lower staff has one note (`8cc-`) with `8r`, the chord's top note is filed as the first upper triplet (`12ee-`), and the octaves are inconsistent (bar 25 at written pitch under the `8`; bar 26 `12eeee-`, two octaves above written). The `replace` op now exists (2026-11-08); what is missing is a reliable chord read from this 1050 px scan. Not done. Affects Corrected and Print alike, and it is most of 110's 389 bar cost, not only our missed ottava.
- Grace slurs gold leaves out (Print standard: printed slurs are added). Found by counting slurs that start on a grace note: ours 138, gold 86 on dev. 091 done 2026-11-08 (17 entries, each checked on the image). Still open: 055 (ours 11, gold 3; grace pairs into main notes in bars 1-3, and a phrase arc over bar 5 gold lacks), 095 (10 against 3), 022 (15 against 7), 047 (16 against 11). Each changes a page by about 0.5 to 0.7 for every system, so about 0.01 on the dev mean.
- 110 last system, attempted read at 4x (2026-11-08): upper staff bar 25 is clear (D♮5 F5, E♮5 G♮5, D♮5 F5 per beat, under the `8`). The lower-staff chords sit on 2-3 ledger lines above a treble staff, with touching heads; my best read of the first is about C6/E6 with naturals (gold has a single C♭5, an octave lower), but the second chord's spacing (second or third) is not readable at 1050 px. Not written into gold; needs a better scan or your eye.
  User's read (2026-11-08): the first chord is C♮6+E6, natural on the C, certainly not a C5 at the bottom;
  agrees with mine. Still unread: the chords after it. Fix the system in one `replace` once those are read,
  not this chord alone: gold also files its top note in the upper staff.
- 055 bar 4 (index 3), lower staff, E2 natural at the bar's start, and 110 bar 2 (index 1), lower
  staff, A3 natural on beat 2: two voices share the head, and gold shows the accidental in one
  voice only. Elsewhere (037, 088, 096) gold shows it in both. Check the image and make it
  consistent. (2026-11-08)
  - 055 checked on the page image: bar 4's lower staff has one low E head with one natural, shared
    by two gold voices; gold should show the natural in both, as elsewhere. 110 not checked
    (chords there unreadable at 1050 px).
- 095 bars 12-16 (index 11-15): the eighth-note grace notes before each figure. Gold writes them
  unslashed (`qq`), bar 1's as slashed (`q`); seq512 reads all as slashed. Slash or not is not
  visible at 1050 px; check on a finer scan. Cost 12. (2026-11-08)
  Resolved (user, 2026-11-08): the stems are slashed. Five `slash_grace` overrides (Corrected, so
  Print too) on lines 94, 109, 123, 137, 151. The 16th grace pairs (lines 56-57, 166-177) are
  unchanged. Current build's 095 on Print: 5.86 to 5.00. Golds now_print4/now_corrected3 updated in
  place (old 095 in $S/gold_bak).
- 060 bar 3 (index 2), right hand: the 16th A-flat 5 has a short arc into the next chord's A-flat 5,
  with a longer arc above both. Gold writes a slur from the 16th and a tie onward from the chord.
  The short arc may be a tie (16th tied into the chord); this needs a finer scan than 1050 px.
  It's the one page the "held note joined by a chord" tie rule made worse (+0.13).
  Resolved (user, 2026-11-08; confirmed at 4x): two separate arcs that seem to touch. The short one
  goes from the 16th A♭5 up to the chord's top E♭6, so it is a slur; the long one ties E♭6 to the
  next chord's E♭6. Gold is right. The +0.13 is our tie rule turning the model's 16th-to-chord slur
  into a tie on the shared A♭5; the LMX slur does not say which head an arc reaches, so the rule
  cannot tell the two apart.
- Staff moves checked note by note on the image (2026-11-08), all 14 threshold entries (047, 057,
  062, 067, 076). Two notes were on the wrong side of the cut-off:
  - 067 bar 8: the last eighth's E-flat 4 is drawn on the upper staff (chord with G4, beamed to the
    bass B-flat 3). Fixed: cut-off F4 to E-flat 4. 067 Print, current build: 10.88 to 10.57.
  - 062 bar 6, line 53: the second chord's B-flat 3 hangs below the ledger line on the upper staff;
    the bar's other B-flat 3s are on the bass staff, so no cut-off works. Not fixed. Splitting the
    range breaks the kern (062 scores 100). Fixed (user: "make gold reproduce print properly"):
    `move_up` now lists the notes drawn on the upper staff, by gold line and place in the field,
    instead of a pitch cut-off. All 16 entries converted from the checked cut-offs; rebuilt gold is
    identical on every page but 062, which gains the B-flat 3 and the beam between the two chords.
    062 Print, current build: 8.99 to 8.43.
