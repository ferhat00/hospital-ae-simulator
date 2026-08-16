"""Example scenario override files must load, validate and differ from baseline."""

from __future__ import annotations

import pytest

from aesim.params import Scenario
from tests.conftest import BASELINE_YAML, REPO_ROOT

SCENARIO_DIR = REPO_ROOT / "config" / "scenarios"


@pytest.mark.parametrize("path", sorted(SCENARIO_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_example_scenario_loads_and_differs(path, baseline: Scenario):
    s = Scenario.from_yaml(BASELINE_YAML, overrides=path)
    assert s.content_hash() != baseline.content_hash()
    assert s.meta["name"] == path.stem


def test_winter_surge_raises_demand(baseline: Scenario):
    s = Scenario.from_yaml(BASELINE_YAML, overrides=SCENARIO_DIR / "winter_surge.yaml")
    assert s.arrivals.mean_daily_attendances > baseline.arrivals.mean_daily_attendances
    # untouched sections inherit from baseline
    assert s.streams["majors"].spaces == baseline.streams["majors"].spaces


def test_extra_sdec_keeps_unmentioned_stream_fields(baseline: Scenario):
    s = Scenario.from_yaml(BASELINE_YAML, overrides=SCENARIO_DIR / "extra_sdec.yaml")
    assert s.streams["sdec"].spaces == 14
    # deep merge must preserve fields the override does not mention
    assert s.streams["sdec"].clinician_pool == baseline.streams["sdec"].clinician_pool
    assert s.streams["sdec"].treatment_time == baseline.streams["sdec"].treatment_time
