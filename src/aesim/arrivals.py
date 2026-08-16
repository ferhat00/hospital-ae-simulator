"""Arrival processes.

Walk-in and ambulance arrivals follow a non-stationary Poisson process (NSPP)
sampled by the thinning algorithm (Lewis & Shedler): candidates are drawn from a
homogeneous Poisson process at the peak rate ``lambda_max`` and accepted with
probability ``lambda(t) / lambda_max``. Two independent RNG streams — one for
candidate inter-arrival times, one for accept/reject — keep common random
numbers synchronised across scenarios.

The rate function is piecewise-constant on hours: hourly profile x day-of-week
multiplier, split into walk-in and ambulance components by the hour-varying
ambulance fraction. Day 0 is a Monday. Booked UTC appointments are a separate
scheduled (non-Poisson) stream with a no-show probability.
"""

from __future__ import annotations

import numpy as np

from aesim.params import ArrivalParams

MIN_PER_DAY = 1440.0
MIN_PER_WEEK = 7 * MIN_PER_DAY


class ArrivalRateProfile:
    """Piecewise-constant lambda(t) per minute for one arrival mode."""

    def __init__(self, params: ArrivalParams, mode: str):
        if mode not in ("walk_in", "ambulance"):
            raise ValueError(mode)
        profile = np.asarray(params.hourly_profile, dtype=float)
        profile = profile / profile.sum()  # share of a day's arrivals per hour
        dow = np.asarray(params.dow_multipliers, dtype=float)
        dow = dow / dow.mean()  # preserve the weekly mean
        amb = np.asarray(params.ambulance_fraction_by_hour, dtype=float)
        mode_frac = amb if mode == "ambulance" else 1.0 - amb

        # 7 x 24 matrix of arrivals-per-minute
        per_hour_all = params.mean_daily_attendances * profile  # arrivals in each hour
        self.rates = np.outer(dow, per_hour_all * mode_frac) / 60.0
        self.lambda_max = float(self.rates.max())

    def rate_at(self, t_min: float) -> float:
        t = t_min % MIN_PER_WEEK
        day = int(t // MIN_PER_DAY)
        hour = int((t % MIN_PER_DAY) // 60.0)
        return float(self.rates[day, hour])


class NSPPThinning:
    """Generator of successive arrival instants for one mode.

    ``day_factors`` (optional) scales the whole profile per simulation day —
    shared between walk-in and ambulance streams so that a flu-surge day lifts
    both. The thinning bound covers the worst day.
    """

    def __init__(
        self,
        profile: ArrivalRateProfile,
        iat_seed: np.random.SeedSequence,
        thin_seed: np.random.SeedSequence,
        day_factors: np.ndarray | None = None,
    ):
        self.profile = profile
        self.day_factors = day_factors
        self.iat_rng = np.random.default_rng(iat_seed)
        self.thin_rng = np.random.default_rng(thin_seed)
        peak = 1.0 if day_factors is None else float(np.max(day_factors))
        self.lambda_max = profile.lambda_max * peak

    def _rate_at(self, t: float) -> float:
        rate = self.profile.rate_at(t)
        if self.day_factors is not None:
            day = min(int(t // MIN_PER_DAY), len(self.day_factors) - 1)
            rate *= float(self.day_factors[day])
        return rate

    def next_arrival(self, t_now: float) -> float:
        """Next accepted arrival strictly after ``t_now`` (simulation minutes)."""
        lam_max = self.lambda_max
        if lam_max <= 0:
            return float("inf")
        t = t_now
        while True:
            t += float(self.iat_rng.exponential(1.0 / lam_max))
            if self.thin_rng.uniform() * lam_max <= self._rate_at(t):
                return t


def booked_slot_times(params: ArrivalParams, open_hours: tuple[float, float]) -> list[float]:
    """Evenly spaced booked-appointment slot offsets (minutes from midnight) within
    the UTC's open hours. No-shows are sampled per-day by the model."""
    n = round(params.booked_utc_per_day)
    if n <= 0:
        return []
    lo, hi = open_hours
    span_hours = (hi - lo) if hi > lo else (24.0 - lo + hi)
    # slots on the half-step grid so none sit exactly at open/close
    step = span_hours * 60.0 / n
    return [(lo * 60.0 + (i + 0.5) * step) % MIN_PER_DAY for i in range(n)]
