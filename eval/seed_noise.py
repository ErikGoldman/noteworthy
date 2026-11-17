"""How far apart runs that differ only in their seed score on a dev set: the noise a single-seed
A/B must clear (docs/EVAL_SETS.md, spec 1). Runs are given as FORM:SCORES, SCORES an omr_ned.py
results file ({"scores": {unit: score}}); runs of one form are its seeds. One step at a time.

For each form, and pooled over forms:
  - each seed's mean, and the difference of every two seeds' means (the null differences);
  - (i) the page-resampled standard error of a two-seed difference, as the protocol's paired
    range would give it (analytic: the standard deviation of the per-unit differences over the
    root of their number, which the bootstrap tends to);
  - (ii) the variance a seed mean would have if seeds differed only unit by unit, independently:
    each unit's variance over the seeds, summed over units, over the units' number squared;
  - the variance of the seed means as seen, against (ii): the ratio, its degrees of freedom and
    the chance of a ratio at least this large were the noise unit by unit only. A ratio well
    above 1 means some runs are better on every unit: page resampling understates the noise
    and more units will not remove that part, only more seeds.
  - the units holding the most of (ii), and what a three-seed mean and four times the units
    would bring the noise of a mean to.

usage: seed_noise.py [--units FILE] FORM:SCORES.json FORM:SCORES.json ...
  --units FILE  only these units (one a line; e.g. Polish dev's two-staff pages)"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev, variance

from scipy.stats import chi2

LARGEST = 5


@dataclass(frozen=True)
class Form:
    name: str
    seeds: list[str]  # each run's file, as given
    scores: list[dict[str, float]]  # each run's score per unit, on the units shared

    @property
    def units(self) -> list[str]:
        return sorted(self.scores[0])

    def seed_means(self) -> list[float]:
        return [mean(s.values()) for s in self.scores]

    def unit_variances(self) -> dict[str, float]:
        """Each unit's variance over the seeds."""
        return {u: variance([s[u] for s in self.scores]) for u in self.units}

    def page_by_page_variance(self) -> float:
        """(ii): the variance of a seed mean if the seeds differed unit by unit only."""
        return sum(self.unit_variances().values()) / len(self.units) ** 2

    def resampled_standard_error(self, a: int, b: int) -> float:
        """(i): the standard error page resampling gives the difference of two seeds' means."""
        differences = [self.scores[a][u] - self.scores[b][u] for u in self.units]
        return stdev(differences) / math.sqrt(len(differences))


@dataclass(frozen=True)
class Excess:
    """Seed means' variance as seen, against what unit-by-unit noise alone would give."""
    seen: float
    page_by_page: float
    degrees: int

    @property
    def ratio(self) -> float:
        return self.seen / self.page_by_page

    @property
    def chance(self) -> float:
        """The chance of a ratio at least this large were the noise unit by unit only."""
        return float(chi2.sf(self.ratio * self.degrees, self.degrees))


def excess_of(forms: list[Form]) -> Excess:
    """Pooled over forms, each weighted by its degrees of freedom (seeds less one)."""
    degrees = sum(len(f.scores) - 1 for f in forms)
    seen = sum((len(f.scores) - 1) * variance(f.seed_means()) for f in forms) / degrees
    page_by_page = sum((len(f.scores) - 1) * f.page_by_page_variance() for f in forms) / degrees
    return Excess(seen, page_by_page, degrees)


def read_forms(runs: list[str], units: set[str] | None) -> list[Form]:
    """Runs grouped by form, each restricted to the units every run of every form scores."""
    files: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        name, path = run.split(":", 1)
        files[name].append(path)
    scores = {name: [json.loads(Path(p).read_text())["scores"] for p in paths] for name, paths in files.items()}
    shared = set.intersection(*(set(s) for runs_of in scores.values() for s in runs_of))
    if units is not None:
        missing = units - shared
        if missing:
            raise SystemExit(f"units not scored by every run: {sorted(missing)[:10]}")
        shared = units
    for name, runs_of in scores.items():
        if len(runs_of) < 2:
            raise SystemExit(f"form {name}: one run; a seed's noise needs two or more")
    return [Form(name, files[name], [{u: s[u] for u in shared} for s in runs_of]) for name, runs_of in scores.items()]


def report(forms: list[Form]) -> None:
    for form in forms:
        means = form.seed_means()
        print(f"== {form.name}: {len(means)} seeds, {len(form.units)} units")
        for path, m in zip(form.seeds, means):
            print(f"   {m:7.3f}  {path}")
        print("   null differences (first - second), with (i) page resampling's standard error:")
        for a, b in combinations(range(len(means)), 2):
            print(f"   seeds {a + 1} - {b + 1}: {means[a] - means[b]:+7.3f}  (i) {form.resampled_standard_error(a, b):.3f}")
        print_excess(excess_of([form]))
        shares = sorted(form.unit_variances().items(), key=lambda uv: -uv[1])
        total = sum(v for _, v in shares)
        largest = shares[:LARGEST]
        print(f"   units holding the most of (ii): " + ", ".join(f"{u} {v / total:.0%}" for u, v in largest)
              + f"; the {LARGEST} together {sum(v for _, v in largest) / total:.0%}")
    if len(forms) > 1:
        print(f"== pooled over {len(forms)} forms")
        print_excess(excess_of(forms))


def print_excess(excess: Excess) -> None:
    seen, page = math.sqrt(excess.seen), math.sqrt(excess.page_by_page)
    every_unit = math.sqrt(max(excess.seen - excess.page_by_page, 0.0))
    print(f"   seed means' standard deviation as seen {seen:.3f}; (ii) unit by unit alone {page:.3f}")
    print(f"   variance ratio {excess.ratio:.2f} on {excess.degrees} degrees of freedom; chance of one this large "
          f"were the noise unit by unit only: {excess.chance:.3f}")
    print(f"   a mean's noise: one seed {seen:.3f}, three seeds {seen / math.sqrt(3):.3f}; "
          f"four times the units {math.sqrt(page ** 2 / 4 + every_unit ** 2):.3f} "
          f"(the every-unit part, {every_unit:.3f}, does not fall with more units)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="FORM:SCORES.json, two or more runs a form")
    ap.add_argument("--units", type=Path, help="only these units, one a line")
    args = ap.parse_args()
    units = set(args.units.read_text().split()) if args.units else None
    report(read_forms(args.runs, units))


if __name__ == "__main__":
    main()
