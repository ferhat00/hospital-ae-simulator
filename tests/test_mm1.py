"""M/M/1 sanity check: strip the model down to one doctor, Poisson arrivals and
exponential service, and compare the mean queue wait to the analytic result.

rho = 0.8, service mean 20 => Wq = rho/(1-rho) * m = 80 min.
"""

from __future__ import annotations

import pytest

from aesim.params import Scenario
from aesim.runner import multiple_replications

SERVICE_MEAN = 20.0
RHO = 0.8
# lambda = rho / m per minute -> daily arrivals
DAILY = RHO / SERVICE_MEAN * 1440.0  # 57.6


@pytest.fixture(scope="module")
def mm1_scenario(baseline: Scenario) -> Scenario:
    flat24 = [1.0] * 24
    return baseline.with_overrides(
        **{
            "run.run_length_days": 30,
            "run.warm_up_days": 3,
            "arrivals.mean_daily_attendances": DAILY,
            "arrivals.hourly_profile": flat24,
            "arrivals.dow_multipliers": [1.0] * 7,
            "arrivals.ambulance_fraction_by_hour": [0.0] * 24,
            "arrivals.booked_utc_per_day": 0,
            "arrivals.paeds_fraction": 0.0,
            "arrivals.daily_cv": 0.0,  # stationary Poisson — no day effects
            # everyone is a YELLOW walk-in headed to majors
            "triage.cat_mix_walkin": {"RED": 0, "ORANGE": 0, "YELLOW": 1.0, "GREEN": 0, "BLUE": 0},
            "triage.registration_time": {"kind": "fixed", "mean": 0.0},
            "triage.triage_time_walkin": {"kind": "fixed", "mean": 0.0},
            # majors is the M/M/1: one doctor, unbounded waiting room
            "streams.majors.spaces": 2000,
            "streams.majors.assessment_time": {"kind": "exponential", "mean": SERVICE_MEAN},
            "streams.majors.treatment_time": {"kind": "fixed", "mean": 0.0},
            "staff_pools.ed_doctors.shifts": [{"start": 0, "end": 24, "count": 1}],
            "staff_pools.ed_doctors.availability_factor": 1.0,
            "staff_pools.triage_nurses.shifts": [{"start": 0, "end": 24, "count": 50}],
            "spaces.triage_rooms": 50,
            "diagnostics.bloods.prob": 0.0,
            "diagnostics.xray.prob": 0.0,
            "diagnostics.ct.prob": 0.0,
            "routing.admission_prob_by_cat": {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0, "BLUE": 0},
            "routing.utc_stream_prob": {},
            "routing.sdec_eligible_frac": {},
            "routing.hot_clinic_frac": 0.0,
            "beds.enabled": False,
        }
    )


def test_mm1_mean_queue_wait(mm1_scenario: Scenario):
    reps = multiple_replications(mm1_scenario, n_reps=10)
    frame = reps.kpi_frame
    # time_to_first_clinician == queue wait (registration and triage take 0 time)
    waits = []
    for b in reps.bundles:
        done = b.patients[(~b.patients["in_warmup"]) & b.patients["t_departure"].notna()]
        waits.append(done["time_to_first_clinician"].dropna().mean())
    mean_wait = sum(waits) / len(waits)
    analytic_wq = RHO / (1 - RHO) * SERVICE_MEAN  # 80 min
    assert mean_wait == pytest.approx(analytic_wq, rel=0.15)
    # server utilisation ~ rho
    util = frame["util_ed_doctors"].mean()
    assert util == pytest.approx(RHO, abs=0.03)


def test_mm1_throughput_conserves(mm1_scenario: Scenario):
    reps = multiple_replications(mm1_scenario, n_reps=3)
    per_day = reps.kpi_frame["attendances_per_day"].mean()
    assert per_day == pytest.approx(DAILY, rel=0.05)
