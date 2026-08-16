"""NSPP thinning correctness: empirical hourly rates must match the profile, and
the interarrival / accept-reject streams must be independent children."""

from __future__ import annotations

import numpy as np
import pytest

from aesim.arrivals import MIN_PER_DAY, ArrivalRateProfile, NSPPThinning, booked_slot_times
from aesim.params import Scenario


def _collect_arrivals(profile: ArrivalRateProfile, seed: int, horizon_days: int) -> np.ndarray:
    ss = np.random.SeedSequence(seed)
    iat, thin = ss.spawn(2)
    gen = NSPPThinning(profile, iat, thin)
    out, t = [], 0.0
    horizon = horizon_days * MIN_PER_DAY
    while True:
        t = gen.next_arrival(t)
        if t >= horizon:
            break
        out.append(t)
    return np.asarray(out)


def test_total_volume_matches_mean_daily(baseline: Scenario):
    days = 200
    walk = _collect_arrivals(ArrivalRateProfile(baseline.arrivals, "walk_in"), 1, days)
    amb = _collect_arrivals(ArrivalRateProfile(baseline.arrivals, "ambulance"), 2, days)
    per_day = (len(walk) + len(amb)) / days
    assert per_day == pytest.approx(baseline.arrivals.mean_daily_attendances, rel=0.03)


def test_hourly_shape_matches_profile(baseline: Scenario):
    days = 400
    profile = ArrivalRateProfile(baseline.arrivals, "walk_in")
    arr = _collect_arrivals(profile, 3, days)
    hours = ((arr % MIN_PER_DAY) // 60).astype(int)
    counts = np.bincount(hours, minlength=24).astype(float)
    empirical = counts / counts.sum()
    # expected hourly share of walk-ins (profile x (1 - amb fraction), renormalised)
    weights = np.asarray(baseline.arrivals.hourly_profile, dtype=float)
    weights = weights / weights.sum()
    amb = np.asarray(baseline.arrivals.ambulance_fraction_by_hour)
    expected = weights * (1 - amb)
    expected = expected / expected.sum()
    np.testing.assert_allclose(empirical, expected, atol=0.006)


def test_ambulance_share_higher_overnight(baseline: Scenario):
    days = 400
    walk = _collect_arrivals(ArrivalRateProfile(baseline.arrivals, "walk_in"), 4, days)
    amb = _collect_arrivals(ArrivalRateProfile(baseline.arrivals, "ambulance"), 5, days)

    def share_in_window(lo_h: int, hi_h: int) -> float:
        def count(a: np.ndarray) -> int:
            h = (a % MIN_PER_DAY) // 60
            return int(((h >= lo_h) & (h < hi_h)).sum())

        w, a = count(walk), count(amb)
        return a / (a + w)

    assert share_in_window(1, 6) > share_in_window(10, 15) + 0.1


def test_dow_multiplier_monday_uplift(baseline: Scenario):
    days = 700
    arr = _collect_arrivals(ArrivalRateProfile(baseline.arrivals, "walk_in"), 6, days)
    dow = ((arr // MIN_PER_DAY) % 7).astype(int)
    counts = np.bincount(dow, minlength=7).astype(float)
    assert counts[0] == counts.max()  # Monday (day 0) is the busiest


def test_rate_never_exceeds_thinning_bound(baseline: Scenario):
    profile = ArrivalRateProfile(baseline.arrivals, "ambulance")
    ts = np.arange(0, 7 * MIN_PER_DAY, 7.0)
    assert all(profile.rate_at(t) <= profile.lambda_max + 1e-12 for t in ts)


def test_reproducible_and_stream_sensitive(baseline: Scenario):
    profile = ArrivalRateProfile(baseline.arrivals, "walk_in")
    a = _collect_arrivals(profile, 9, 5)
    b = _collect_arrivals(profile, 9, 5)
    c = _collect_arrivals(profile, 10, 5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_zero_rate_returns_infinity():
    from aesim.params import ArrivalParams

    p = ArrivalParams(
        mean_daily_attendances=1e-9,
        hourly_profile=tuple([1.0] * 24),
        dow_multipliers=tuple([1.0] * 7),
        ambulance_fraction_by_hour=tuple([1.0] * 24),  # all ambulance -> walk-in rate 0
    )
    profile = ArrivalRateProfile(p, "walk_in")
    gen = NSPPThinning(profile, *np.random.SeedSequence(1).spawn(2))
    assert gen.next_arrival(0.0) == float("inf")


def test_booked_slots_inside_open_hours(baseline: Scenario):
    slots = booked_slot_times(baseline.arrivals, baseline.streams["utc"].open_hours)
    assert len(slots) == round(baseline.arrivals.booked_utc_per_day)
    for s in slots:
        hour = s / 60.0
        assert 8 <= hour < 24
