"""Random sampling primitives.

Every sampler owns its own ``numpy.random.Generator`` seeded from a dedicated
``SeedSequence`` child. Giving each sampling *purpose* its own stream keeps the
draws synchronised between scenarios (common random numbers): changing majors
capacity in scenario B does not perturb the triage-category draws it shares with
scenario A, so paired comparisons have far lower variance.

Distributions are parameterised on the natural scale (mean, cv) and converted
internally — "about 20 minutes, sometimes an hour" elicits directly.
"""

from __future__ import annotations

import math

import numpy as np

from aesim.params import DistConfig


class Sampler:
    """A duration distribution bound to its own random stream."""

    def __init__(self, cfg: DistConfig, seed: np.random.SeedSequence):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def sample(self) -> float:
        c = self.cfg
        match c.kind:
            case "fixed":
                return c.mean
            case "exponential":
                return float(self.rng.exponential(c.mean)) if c.mean > 0 else 0.0
            case "lognormal":
                mu, sigma = _lognormal_params(c.mean, c.cv)
                return float(self.rng.lognormal(mu, sigma))
            case "gamma":
                shape = 1.0 / (c.cv**2)
                scale = c.mean * c.cv**2
                return float(self.rng.gamma(shape, scale))
            case "triangular":
                mode = c.mode if c.mode is not None else (c.low + c.high) / 2
                return float(self.rng.triangular(c.low, mode, c.high))
            case "uniform":
                return float(self.rng.uniform(c.low, c.high))
        raise ValueError(f"unknown kind {c.kind!r}")

    def residual_sample(self) -> float:
        """Stationary residual (forward-recurrence) time — 'how much longer will an
        in-progress activity of this kind last, observed at a random instant'.

        Exact for lognormal (length-biased lognormal is LN(mu + sigma^2, sigma))
        and exponential (memoryless). For other kinds we approximate with
        U * sample(), which is adequate for warm-start bed prefill.
        """
        c = self.cfg
        u = float(self.rng.uniform())
        if c.kind == "exponential":
            return self.sample()
        if c.kind == "lognormal":
            mu, sigma = _lognormal_params(c.mean, c.cv)
            length_biased = float(self.rng.lognormal(mu + sigma**2, sigma))
            return u * length_biased
        return u * self.sample()


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    sigma2 = math.log(1.0 + cv**2)
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


class Bernoulli:
    def __init__(self, p: float, seed: np.random.SeedSequence):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def sample(self) -> bool:
        return bool(self.rng.uniform() < self.p)


class Categorical:
    """Draws one of ``labels`` with the given probabilities (need not be pre-normalised)."""

    def __init__(self, labels: list[str], probs: list[float], seed: np.random.SeedSequence):
        total = sum(probs)
        if total <= 0:
            raise ValueError("Categorical needs positive total probability")
        self.labels = list(labels)
        self.cum = np.cumsum(np.asarray(probs, dtype=float) / total)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> str:
        u = self.rng.uniform()
        idx = int(np.searchsorted(self.cum, u, side="right"))
        return self.labels[min(idx, len(self.labels) - 1)]


class UniformDraw:
    """Bare U(0,1) stream for routing decisions that share one purpose."""

    def __init__(self, seed: np.random.SeedSequence):
        self.rng = np.random.default_rng(seed)

    def sample(self) -> float:
        return float(self.rng.uniform())


class DischargeHourSampler:
    """Snaps an inpatient's nominal discharge instant to a realistic clock hour.

    UK ward discharges cluster in the afternoon (~14:00-18:00); that afternoon
    weighting is exactly why bed availability is worst in the morning. Given a
    nominal discharge time (admission + sampled LoS), we draw a clock hour from
    ``weights`` and move the discharge to that hour on whichever adjacent day
    lands closest to the nominal instant, never earlier than one hour after
    admission.
    """

    def __init__(self, weights: tuple[float, ...], seed: np.random.SeedSequence):
        if len(weights) != 24:
            raise ValueError("discharge hour weights must have 24 entries")
        self.hours = Categorical([str(h) for h in range(24)], list(weights), seed)
        self.rng = self.hours.rng  # share: jitter within the hour uses the same stream

    def snap(self, t_admit: float, nominal_departure: float) -> float:
        hour = int(self.hours.sample())
        jitter = float(self.rng.uniform(0.0, 60.0))
        day = math.floor(nominal_departure / 1440.0)
        candidates = [d * 1440.0 + hour * 60.0 + jitter for d in (day - 1, day, day + 1)]
        feasible = [c for c in candidates if c >= t_admit + 60.0]
        if not feasible:
            return t_admit + 60.0
        return min(feasible, key=lambda c: abs(c - nominal_departure))
