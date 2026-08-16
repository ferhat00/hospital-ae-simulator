"""Calibration acceptance: baseline KPI means must sit inside the NHS actual
bands declared in the config's `targets:` section.

Slow (10 reps x 42 days ~ 40s) — marked `calibration`, excluded from the default
run; invoke with `pytest -m calibration`.
"""

from __future__ import annotations

import pytest

from aesim.params import Scenario
from aesim.runner import multiple_replications


@pytest.mark.calibration
def test_baseline_kpis_inside_nhs_bands(baseline: Scenario):
    reps = multiple_replications(baseline, n_reps=10)
    means = reps.kpi_frame.mean(numeric_only=True)

    failures = []
    for kpi, (lo, hi) in baseline.targets.items():
        if kpi not in means:
            failures.append(f"{kpi}: not computed")
            continue
        val = means[kpi]
        if not (lo <= val <= hi):
            failures.append(f"{kpi}: {val:.3f} outside [{lo}, {hi}]")
    assert not failures, "; ".join(failures)


@pytest.mark.calibration
def test_face_validity_signatures(baseline: Scenario):
    """Qualitative signatures a UK ED model must show (Bowers 2011; RCEM)."""
    reps = multiple_replications(baseline, n_reps=5)
    means = reps.kpi_frame.mean(numeric_only=True)

    # admitted patients fare far worse than discharged — the exit-block gap
    assert means["four_hour_discharged"] - means["four_hour_admitted"] > 0.25
    # boarders occupy a material share of majors cubicles (RCEM warns at 10%)
    assert means["boarders_share_of_majors"] > 0.10
    # admitted median LoS well above discharged median
    assert means["median_time_in_dept_admitted"] > means["median_time_in_dept_discharged"] + 60

    # morning bed starvation: mean boarding for DTA in 00:00-08:00 exceeds 16:00-20:00
    df = reps.bundles[0].patients
    done = df[(~df["in_warmup"]) & df["t_bed"].notna() & df["t_dta"].notna()].copy()
    dta_hour = (done["t_dta"] % 1440.0) // 60.0
    boarding = done["t_bed"] - done["t_dta"]
    overnight = boarding[(dta_hour >= 0) & (dta_hour < 8)]
    afternoon = boarding[(dta_hour >= 14) & (dta_hour < 20)]
    if len(overnight) > 10 and len(afternoon) > 10:
        assert overnight.mean() > afternoon.mean()
