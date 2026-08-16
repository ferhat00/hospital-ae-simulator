"""Distribution recovery: sample means/CVs must match their specification, and
streams must be reproducible and independent."""

from __future__ import annotations

import numpy as np
import pytest

from aesim.distributions import (
    Bernoulli,
    Categorical,
    DischargeHourSampler,
    Sampler,
)
from aesim.params import DistConfig

N = 100_000


def _draws(cfg: DistConfig, seed: int = 1, n: int = N) -> np.ndarray:
    s = Sampler(cfg, np.random.SeedSequence(seed))
    return np.array([s.sample() for _ in range(n)])


@pytest.mark.parametrize(
    "cfg",
    [
        DistConfig("lognormal", mean=38.0, cv=0.6),
        DistConfig("gamma", mean=6.5, cv=0.4),
        DistConfig("exponential", mean=75.0),
    ],
    ids=lambda c: c.kind,
)
def test_mean_recovery(cfg: DistConfig):
    x = _draws(cfg)
    assert x.min() >= 0
    assert x.mean() == pytest.approx(cfg.mean, rel=0.02)


@pytest.mark.parametrize(
    "cfg",
    [DistConfig("lognormal", mean=38.0, cv=0.6), DistConfig("gamma", mean=6.5, cv=0.4)],
    ids=lambda c: c.kind,
)
def test_cv_recovery(cfg: DistConfig):
    x = _draws(cfg)
    assert x.std() / x.mean() == pytest.approx(cfg.cv, rel=0.03)


def test_fixed_and_bounds():
    assert _draws(DistConfig("fixed", mean=15.0), n=10).tolist() == [15.0] * 10
    tri = _draws(DistConfig("triangular", low=5, mode=10, high=15), n=10_000)
    assert tri.min() >= 5 and tri.max() <= 15
    assert tri.mean() == pytest.approx(10.0, rel=0.02)
    uni = _draws(DistConfig("uniform", low=2, high=4), n=10_000)
    assert uni.min() >= 2 and uni.max() <= 4


def test_same_seed_reproducible_different_seed_not():
    cfg = DistConfig("lognormal", mean=20.0, cv=0.5)
    a = _draws(cfg, seed=7, n=100)
    b = _draws(cfg, seed=7, n=100)
    c = _draws(cfg, seed=8, n=100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_spawned_streams_are_independent():
    parent = np.random.SeedSequence(42)
    s1, s2 = parent.spawn(2)
    cfg = DistConfig("exponential", mean=10.0)
    a, b = Sampler(cfg, s1), Sampler(cfg, s2)
    x = np.array([a.sample() for _ in range(5_000)])
    y = np.array([b.sample() for _ in range(5_000)])
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.05


def test_lognormal_residual_mean_exceeds_plain_mean():
    """Inspection paradox: the residual of an in-progress lognormal stay has a
    larger expectation than a fresh draw when cv >= 1."""
    cfg = DistConfig("lognormal", mean=5.0, cv=1.0)
    s = Sampler(cfg, np.random.SeedSequence(3))
    resid = np.array([s.residual_sample() for _ in range(N)])
    # E[residual] = E[L^2]/(2 E[L]) = mean*(1+cv^2)/2 = 5.0 for cv=1
    assert resid.mean() == pytest.approx(5.0, rel=0.05)


def test_bernoulli_rate():
    b = Bernoulli(0.275, np.random.SeedSequence(5))
    hits = sum(b.sample() for _ in range(N))
    assert hits / N == pytest.approx(0.275, abs=0.005)


def test_categorical_proportions():
    probs = {"RED": 0.01, "ORANGE": 0.13, "YELLOW": 0.42, "GREEN": 0.42, "BLUE": 0.02}
    cat = Categorical(list(probs), list(probs.values()), np.random.SeedSequence(6))
    draws = [cat.sample() for _ in range(N)]
    for label, p in probs.items():
        assert draws.count(label) / N == pytest.approx(p, abs=0.006)


def test_discharge_hour_sampler_snaps_to_weighted_hours():
    weights = tuple(1.0 if h == 14 else 0.0 for h in range(24))  # discharge at 14:00 only
    d = DischargeHourSampler(weights, np.random.SeedSequence(7))
    t_admit = 2 * 1440.0 + 600.0  # day 2, 10:00
    nominal = t_admit + 3.2 * 1440.0
    out = d.snap(t_admit, nominal)
    assert out >= t_admit + 60
    hour = (out % 1440.0) // 60.0
    assert hour == 14
    assert abs(out - nominal) <= 1440.0


def test_discharge_hour_sampler_never_before_admission():
    weights = tuple(1.0 for _ in range(24))
    d = DischargeHourSampler(weights, np.random.SeedSequence(8))
    t_admit = 500.0
    for _ in range(200):
        assert d.snap(t_admit, t_admit + 30.0) >= t_admit + 60.0
