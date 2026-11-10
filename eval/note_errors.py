"""Each gold note's fate in a prediction, by kind, per page: right, wrong duration, an octave
off, another wrong pitch, wrong onset, missing; and the prediction's extra notes. Staves are
matched top to bottom, bars by an order-keeping alignment, notes within a bar by onset, pitch and
duration in turn, all at sounding pitch. Graces left out. OMR-NED says what a page costs; this
says what kind of reading went wrong.
usage (repo root, .venv-musicdiff): eval/note_errors.py GOLD_DIR PRED_DIR [PRED_DIR...] -- PAGE...
(gold .krn, e.g. Print gold from gold_overrides.py; predictions .musicxml from seq_pages.py)."""
import sys
from collections import Counter
import converter21; converter21.register()
import music21 as m21

KINDS = ['ok', 'duration', 'pitch_octave', 'pitch_other', 'onset', 'missing', 'extra']


def staves(path):
    """Each staff's bars as lists of (onset, midi, length), staves top to bottom."""
    # Sounding pitch on both sides: gold may write the printed pitch under an 8va sign.
    score = m21.converter.parse(path, forceSource=True).toSoundingPitch()
    out = []
    for part in score.parts:
        bars = []
        for measure in part.getElementsByClass('Measure'):
            notes = []
            for n in measure.recurse().notes:
                if n.duration.isGrace:
                    continue
                onset = float(n.getOffsetInHierarchy(measure))
                for p in (n.pitches if n.isChord else [n.pitch]):
                    notes.append((round(onset, 3), p.midi, round(float(n.quarterLength), 3)))
            bars.append(notes)
        out.append(bars)
    return out


def take(pool, key):
    for i, item in enumerate(pool):
        if key(item):
            return pool.pop(i)
    return None


def bar_fates(gold, pred, counts):
    gold, pred = list(gold), list(pred)
    for g in list(gold):
        if g in pred:
            pred.remove(g); gold.remove(g); counts['ok'] += 1
    for kind, same in (('duration', lambda g, p: g[:2] == p[:2]),
                       ('pitch_octave', lambda g, p: g[0] == p[0] and g[2] == p[2] and abs(g[1] - p[1]) == 12),
                       ('pitch_other', lambda g, p: g[0] == p[0] and g[2] == p[2]),
                       ('onset', lambda g, p: g[1:] == p[1:])):
        for g in list(gold):
            if take(pred, lambda p: same(g, p)):
                gold.remove(g); counts[kind] += 1
    counts['missing'] += len(gold); counts['extra'] += len(pred)


def likeness(gold_bar, pred_bar):
    """Notes the two bars share by onset and pitch class, over the larger: 0 to 1. Pitch class,
    so that a bar read an octave off still pairs with its gold and counts as an octave error."""
    g, p = Counter((o, m % 12) for o, m, _ in gold_bar), Counter((o, m % 12) for o, m, _ in pred_bar)
    return sum((g & p).values()) / max(sum(g.values()), sum(p.values()), 1)


def aligned_bars(g_staff, p_staff):
    """Pairs of bar indices, in order, maximizing summed likeness (a bar may go unpaired).
    Not difflib: repeated passages made it pair bars far apart and leave the rest unmatched."""
    n, m = len(g_staff), len(p_staff)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1], likeness(g_staff[i], p_staff[j]) + best[i + 1][j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if best[i][j] == likeness(g_staff[i], p_staff[j]) + best[i + 1][j + 1]:
            pairs.append((i, j)); i += 1; j += 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return pairs


def page_fates(gold_path, pred_path):
    counts = Counter()
    for g_staff, p_staff in zip(staves(gold_path), staves(pred_path)):
        pairs = aligned_bars(g_staff, p_staff)
        for a, b in pairs:
            bar_fates(g_staff[a], p_staff[b], counts)
        paired_g, paired_p = {a for a, _ in pairs}, {b for _, b in pairs}
        counts['missing'] += sum(len(bar) for k, bar in enumerate(g_staff) if k not in paired_g)
        counts['extra'] += sum(len(bar) for k, bar in enumerate(p_staff) if k not in paired_p)
    return counts


if __name__ == '__main__':
    args = sys.argv[1:]; cut = args.index('--')
    gold_dir, pred_dirs, pages = args[0], args[1:cut], args[cut + 1:]
    totals = {d: Counter() for d in pred_dirs}
    print('page ' + '  '.join(f'{k:>12s}' for k in KINDS))
    for p in pages:
        for d in pred_dirs:
            c = page_fates(f'{gold_dir}/{p}.krn', f'{d}/{p}.musicxml')
            totals[d].update(c)
            print(f'{p} {d.rstrip("/").split("/")[-1][-8:]:>8s} ' + '  '.join(f'{c[k]:>12d}' for k in KINDS), flush=True)
    for d, c in totals.items():
        print('TOTAL', d.rstrip('/').split('/')[-1], '  '.join(f'{k} {c[k]}' for k in KINDS))
