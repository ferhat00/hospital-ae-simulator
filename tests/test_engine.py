"""Whole-engine correctness: conservation, degenerate cases, reproducibility,
and common-random-numbers synchronisation across scenarios."""

from __future__ import annotations

import pandas as pd
import pytest

from aesim.params import Scenario
from aesim.runner import single_run


def _mini(baseline: Scenario, **over) -> Scenario:
    base = {
        "run.run_length_days": 4,
        "run.warm_up_days": 1,
    }
    base.update(over)
    return baseline.with_overrides(**base)


@pytest.fixture(scope="module")
def mini_bundle(baseline: Scenario):
    return single_run(_mini(baseline), rep=0, keep_event_log=True)


def test_conservation_arrivals_equal_departures_plus_in_flight(mini_bundle):
    log = mini_bundle.event_log
    arrivals = int((log["event"] == "arrival").sum())
    departures = int((log["event"] == "depart").sum())
    assert arrivals == departures + mini_bundle.extras["n_in_flight"]
    assert arrivals == len(mini_bundle.patients)


def test_every_resource_use_has_matching_end(mini_bundle):
    log = mini_bundle.event_log
    uses = log[log["event_type"] == "resource_use"].groupby("event").size()
    ends = log[log["event_type"] == "resource_use_end"].groupby("event").size()
    for event, n in uses.items():
        # ends can lag uses only for patients still in the department at cutoff
        assert ends.get(event, 0) <= n
        assert n - ends.get(event, 0) <= mini_bundle.extras["n_in_flight"] + 1


def test_timestamps_monotone(mini_bundle):
    df = mini_bundle.patients
    order = [
        "t_arrival",
        "t_triage_start",
        "t_triage_end",
        "t_first_clinician",
        "t_dta",
        "t_bed",
        "t_departure",
    ]
    for _, row in df.iterrows():
        times = [row[c] for c in order if pd.notna(row[c])]
        assert times == sorted(times), f"non-monotone journey for pid {row['pid']}"


def test_departures_never_before_arrival(mini_bundle):
    df = mini_bundle.patients.dropna(subset=["t_departure"])
    assert (df["t_departure"] >= df["t_arrival"]).all()
    assert (df.dropna(subset=["time_in_department"])["time_in_department"] >= 0).all()


def test_degenerate_infinite_resources_zero_waits(baseline: Scenario):
    s = _mini(
        baseline,
        **{
            "streams.resus.spaces": 500,
            "streams.majors.spaces": 500,
            "streams.minors.spaces": 500,
            "streams.utc.spaces": 500,
            "streams.sdec.spaces": 500,
            "spaces.triage_rooms": 500,
            "staff_pools.ed_doctors.shifts": [{"start": 0, "end": 24, "count": 400}],
            "staff_pools.ed_doctors.availability_factor": 1.0,
            "staff_pools.ed_nurses.shifts": [{"start": 0, "end": 24, "count": 400}],
            "staff_pools.ed_nurses.availability_factor": 1.0,
            "staff_pools.triage_nurses.shifts": [{"start": 0, "end": 24, "count": 400}],
            "staff_pools.utc_enps.shifts": [{"start": 0, "end": 24, "count": 400}],
            "staff_pools.sdec_clinicians.shifts": [{"start": 0, "end": 24, "count": 400}],
            "beds.enabled": False,
            "routing.hot_clinic_frac": 0.0,
        },
    )
    b = single_run(s, rep=0)
    completed = b.patients[(~b.patients["in_warmup"]) & b.patients["t_departure"].notna()]
    # with unlimited resources nobody waits for a space or clinician
    waits = completed["time_to_first_clinician"].dropna()
    triage_dur = (completed["t_triage_end"] - completed["t_triage_start"]).dropna()
    assert b.kpis["lwbs"] == 0.0
    assert b.kpis["four_hour_utc"] == pytest.approx(1.0, abs=0.01)
    # time to clinician = registration + triage service only (no queueing)
    assert waits.median() < 30.0
    assert (triage_dur >= 0).all()
    # crews are still occupied ~26 min (stretcher assessment + clinical handover)
    # even with infinite capacity: Little's law gives ~1.2 crews present on average
    assert b.kpis["mean_ambulances_waiting"] < 2.0
    assert b.kpis["handover_median_min"] < 35.0  # no queueing component


def test_degenerate_zero_arrivals(baseline: Scenario):
    s = _mini(baseline, **{"arrivals.mean_daily_attendances": 1e-9,
                          "arrivals.booked_utc_per_day": 0,
                          "beds.elective_demand_per_day": 0})
    b = single_run(s, rep=0)
    assert b.kpis["attendances"] == 0.0


def test_same_rep_reproducible(baseline: Scenario):
    s = _mini(baseline)
    a = single_run(s, rep=0, keep_event_log=True)
    b = single_run(s, rep=0, keep_event_log=True)
    pd.testing.assert_frame_equal(a.event_log, b.event_log)
    assert a.kpis == b.kpis


def test_different_reps_differ(baseline: Scenario):
    s = _mini(baseline)
    a = single_run(s, rep=0)
    b = single_run(s, rep=1)
    assert a.kpis["attendances"] != b.kpis["attendances"] or (
        a.patients["t_arrival"].tolist() != b.patients["t_arrival"].tolist()
    )


def test_crn_same_arrivals_across_capacity_scenarios(baseline: Scenario):
    """Changing majors capacity must not perturb arrival instants, modes or
    triage categories — the CRN guarantee that makes paired comparison valid."""
    a = single_run(_mini(baseline), rep=3)
    b = single_run(_mini(baseline, **{"streams.majors.spaces": 40}), rep=3)
    fa = a.patients[["pid", "mode", "t_arrival", "cat"]].set_index("pid").sort_index()
    fb = b.patients[["pid", "mode", "t_arrival", "cat"]].set_index("pid").sort_index()
    n = min(len(fa), len(fb))
    pd.testing.assert_frame_equal(fa.iloc[:n], fb.iloc[:n])
