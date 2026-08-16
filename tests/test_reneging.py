"""LWBS reneging: impatient GREEN/BLUE patients abandon the minors/UTC queue."""

from __future__ import annotations

import pytest

from aesim.params import Scenario
from aesim.runner import single_run


def test_near_zero_patience_all_queued_greens_lwbs(baseline: Scenario):
    """One minors room with very long service and ~zero patience: every GREEN or
    BLUE who cannot get a room instantly must leave without being seen."""
    s = baseline.with_overrides(
        **{
            "run.run_length_days": 3,
            "run.warm_up_days": 1,
            "streams.minors.spaces": 1,
            "streams.minors.treatment_time": {"kind": "fixed", "mean": 600.0},
            "streams.utc.spaces": 1,
            "streams.utc.treatment_time": {"kind": "fixed", "mean": 600.0},
            "triage.patience_min": {"GREEN": 0.01, "BLUE": 0.01},
        }
    )
    b = single_run(s, rep=0)
    done = b.patients[(~b.patients["in_warmup"]) & b.patients["t_departure"].notna()]
    greens_minors = done[done["stream"].isin(["minors", "utc"]) & done["cat"].isin(["GREEN", "BLUE"])]
    assert len(greens_minors) > 20
    # everyone except the handful who found the room free immediately reneged
    lwbs_share = (greens_minors["disposal"] == "lwbs").mean()
    assert lwbs_share > 0.9
    # reneging is prompt: once in the space queue (triage complete), a reneged
    # walk-in departs within ~patience minutes — any earlier time is triage queueing
    lwbs = greens_minors[greens_minors["disposal"] == "lwbs"]
    walkin_lwbs = lwbs[lwbs["mode"] == "walk_in"].dropna(subset=["t_triage_end"])
    assert len(walkin_lwbs) > 10
    post_triage = walkin_lwbs["t_departure"] - walkin_lwbs["t_triage_end"]
    assert (post_triage < 5.0).all()


def test_generous_patience_low_lwbs(baseline: Scenario):
    s = baseline.with_overrides(
        **{
            "run.run_length_days": 3,
            "run.warm_up_days": 1,
            "triage.patience_min": {"GREEN": 100000, "BLUE": 100000},
        }
    )
    b = single_run(s, rep=0)
    assert b.kpis["lwbs"] == pytest.approx(0.0, abs=0.001)


def test_urgent_categories_never_renege(baseline: Scenario):
    s = baseline.with_overrides(
        **{"run.run_length_days": 3, "run.warm_up_days": 1}
    )
    b = single_run(s, rep=0)
    done = b.patients[b.patients["t_departure"].notna()]
    lwbs = done[done["disposal"] == "lwbs"]
    assert set(lwbs["cat"].unique()) <= {"GREEN", "BLUE"}
