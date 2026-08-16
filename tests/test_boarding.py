"""Exit block mechanics: admitted patients hold their ED space until a bed is
granted, so bed scarcity must saturate the ED — without deadlocking it."""

from __future__ import annotations

import pytest

from aesim.params import Scenario
from aesim.runner import single_run


def test_bed_scarcity_saturates_ed_without_deadlock(baseline: Scenario):
    """One bed with week-long stays: boarding must pile up, admitted patients'
    4-hour performance must collapse, but the run completes and pure-discharge
    UTC patients still flow."""
    s = baseline.with_overrides(
        **{
            "run.run_length_days": 4,
            "run.warm_up_days": 1,
            "beds.n_beds": 1,
            "beds.initial_occupancy": 0.0,
            "beds.los_days": {"kind": "fixed", "mean": 7.0},
            "beds.elective_demand_per_day": 0,
        }
    )
    b = single_run(s, rep=0)  # completing at all proves no deadlock
    done = b.patients[(~b.patients["in_warmup"]) & b.patients["t_departure"].notna()]
    admitted_done = done[done["admitted"]]
    # almost nobody admitted can leave (1 bed, 7-day LoS)
    assert len(admitted_done) <= 3
    # boarding is consuming the ED: boarders occupy a large share of majors
    assert b.kpis["boarders_share_of_majors"] > 0.3
    # UTC (no bed dependency) keeps discharging
    utc_done = done[done["stream"] == "utc"]
    assert len(utc_done) > 50
    assert b.kpis["four_hour_utc"] > 0.9


def test_infinite_beds_no_boarding(baseline: Scenario):
    s = baseline.with_overrides(
        **{
            "run.run_length_days": 3,
            "run.warm_up_days": 1,
            "beds.enabled": False,
        }
    )
    b = single_run(s, rep=0)
    assert b.kpis["mean_boarding_min"] == 0.0
    assert b.kpis["twelve_hour_dta"] == 0.0


def test_boarding_time_recorded_and_positive_when_beds_tight(baseline: Scenario):
    s = baseline.with_overrides(
        **{
            "run.run_length_days": 4,
            "run.warm_up_days": 1,
            "beds.n_beds": 5,
            "beds.initial_occupancy": 1.0,
            "beds.los_days": {"kind": "fixed", "mean": 2.0},
            "beds.elective_demand_per_day": 0,
        }
    )
    b = single_run(s, rep=0)
    done = b.patients[b.patients["t_departure"].notna()]
    boarded = done["boarding_time"].dropna()
    if len(boarded):
        assert (boarded >= 0).all()
        assert boarded.max() > 60  # someone waited over an hour for a bed
    # bed occupancy pinned at ~100%
    assert b.kpis["bed_occupancy"] == pytest.approx(1.0, abs=0.05)
