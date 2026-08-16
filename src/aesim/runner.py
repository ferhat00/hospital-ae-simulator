"""Run orchestration: single runs, replication sets, scenario sweeps, paired
comparisons under common random numbers.

Replication i of every scenario derives its seeds from the same master
SeedSequence child, so ``paired_diff`` computes per-replication deltas whose
variance is far lower than independent sampling would give.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aesim.model import AEModel
from aesim.params import Scenario
from aesim.results import ResultsBundle, compute_kpis, event_log_frame, summarise_replications


@dataclass
class ReplicationResults:
    scenario: Scenario
    bundles: list[ResultsBundle] = field(default_factory=list)

    @property
    def kpi_frame(self) -> pd.DataFrame:
        return pd.DataFrame([b.kpis for b in self.bundles])

    @property
    def summary(self) -> pd.DataFrame:
        return summarise_replications(self.bundles)


def single_run(scenario: Scenario, rep: int = 0, keep_event_log: bool = False) -> ResultsBundle:
    model = AEModel(scenario, rep=rep)
    raw = model.run()
    bundle = compute_kpis(raw)
    if keep_event_log:
        bundle.event_log = event_log_frame(raw)
    return bundle


def multiple_replications(
    scenario: Scenario,
    n_reps: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ReplicationResults:
    n = n_reps if n_reps is not None else scenario.run.default_reps
    out = ReplicationResults(scenario)
    for rep in range(n):
        out.bundles.append(single_run(scenario, rep=rep))
        if progress is not None:
            progress(rep + 1, n)
    return out


def scenario_analysis(
    scenarios: dict[str, Scenario],
    n_reps: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Run each named scenario and return a tidy summary frame."""
    frames = []
    for name, scenario in scenarios.items():
        cb = (lambda i, n, _name=name: progress(_name, i, n)) if progress else None
        reps = multiple_replications(scenario, n_reps, progress=cb)
        summary = reps.summary.reset_index()
        summary.insert(0, "scenario", name)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def paired_diff(a: ReplicationResults, b: ReplicationResults) -> pd.DataFrame:
    """Per-replication KPI differences (b - a) with paired-t 95% CIs.

    Valid because replication i of both scenarios shares seeds (CRN). Rows where
    either side is missing a value are dropped pairwise.
    """
    from scipy import stats

    fa, fb = a.kpi_frame, b.kpi_frame
    n_pairs = min(len(fa), len(fb))
    rows = []
    for kpi in sorted(set(fa.columns) & set(fb.columns)):
        xa = fa[kpi].to_numpy(dtype=float)[:n_pairs]
        xb = fb[kpi].to_numpy(dtype=float)[:n_pairs]
        mask = ~(np.isnan(xa) | np.isnan(xb))
        d = xb[mask] - xa[mask]
        n = len(d)
        if n == 0:
            continue
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if n > 1 else 0.0
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 and sd > 0 else 0.0
        rows.append(
            {
                "kpi": kpi,
                "baseline_mean": float(xa[mask].mean()),
                "scenario_mean": float(xb[mask].mean()),
                "diff_mean": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "significant": bool(half > 0 and (mean - half > 0 or mean + half < 0)),
                "n_pairs": n,
            }
        )
    return pd.DataFrame(rows).set_index("kpi")
