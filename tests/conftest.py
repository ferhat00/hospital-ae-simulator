"""Shared fixtures: the baseline scenario and small/fast variants for unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aesim.params import Scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_YAML = REPO_ROOT / "config" / "baseline_dgh.yaml"


@pytest.fixture(scope="session")
def baseline() -> Scenario:
    return Scenario.from_yaml(BASELINE_YAML)


@pytest.fixture()
def fast_scenario(baseline: Scenario) -> Scenario:
    """Short two-day run with a one-day warm-up — for engine unit tests."""
    return baseline.with_overrides(
        **{
            "run.run_length_days": 2,
            "run.warm_up_days": 1,
            "run.default_reps": 2,
        }
    )
