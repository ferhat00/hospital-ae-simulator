"""Scenario loading, validation, overrides, hashing, YAML round-trip."""

from __future__ import annotations

import pytest
import yaml

from aesim.params import DistConfig, Scenario, ShiftBlock, StaffPoolParams
from tests.conftest import BASELINE_YAML


def test_baseline_loads_and_validates(baseline: Scenario):
    assert baseline.meta["name"] == "baseline_dgh"
    assert baseline.arrivals.mean_daily_attendances == 250
    assert set(baseline.streams) == {"resus", "majors", "minors", "utc", "sdec"}
    assert baseline.run.run_length_min == 42 * 1440


def test_cat_mixes_sum_to_one(baseline: Scenario):
    assert abs(sum(baseline.triage.cat_mix_walkin.values()) - 1) < 1e-9
    assert abs(sum(baseline.triage.cat_mix_ambulance.values()) - 1) < 1e-9


def test_with_overrides_dotted_path(baseline: Scenario):
    s2 = baseline.with_overrides(**{"streams.majors.spaces": 30, "beds.n_beds": 420})
    assert s2.streams["majors"].spaces == 30
    assert s2.beds.n_beds == 420
    # original untouched
    assert baseline.streams["majors"].spaces != 30
    assert baseline.beds.n_beds != 420


def test_with_overrides_bad_path_raises(baseline: Scenario):
    with pytest.raises(KeyError):
        baseline.with_overrides(**{"streams.majors.nonexistent": 1})


def test_content_hash_stable_and_sensitive(baseline: Scenario):
    assert baseline.content_hash() == baseline.content_hash()
    changed = baseline.with_overrides(**{"streams.majors.spaces": 25})
    assert changed.content_hash() != baseline.content_hash()


def test_round_trip_through_dict(baseline: Scenario):
    rebuilt = Scenario.from_dict(baseline.to_dict())
    assert rebuilt.content_hash() == baseline.content_hash()


def test_yaml_deep_merge_override(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("streams:\n  majors:\n    spaces: 99\n", encoding="utf-8")
    s = Scenario.from_yaml(BASELINE_YAML, overrides=override)
    assert s.streams["majors"].spaces == 99
    assert s.streams["minors"].spaces == 8  # untouched by partial override


def test_validation_rejects_bad_pool_reference(baseline: Scenario):
    with pytest.raises(ValueError, match="clinician_pool"):
        baseline.with_overrides(**{"streams.majors.clinician_pool": "no_such_pool"})


def test_validation_rejects_bad_cat_mix(baseline: Scenario):
    d = baseline.to_dict()
    d["triage"]["cat_mix_walkin"]["RED"] = 0.5  # breaks sum-to-one
    with pytest.raises(ValueError, match="sum to 1"):
        Scenario.from_dict(d)


def test_validation_rejects_wrong_profile_length(baseline: Scenario):
    d = baseline.to_dict()
    d["arrivals"]["hourly_profile"] = [1.0] * 23
    with pytest.raises(ValueError, match="24 entries"):
        Scenario.from_dict(d)


def test_validation_rejects_warmup_ge_runlength(baseline: Scenario):
    with pytest.raises(ValueError, match="warm_up"):
        baseline.with_overrides(**{"run.warm_up_days": 42})


def test_unknown_yaml_key_rejected(baseline: Scenario):
    d = baseline.to_dict()
    d["beds"]["n_bedz"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        Scenario.from_dict(d)


def test_dist_config_validation():
    with pytest.raises(ValueError):
        DistConfig("lognormal", mean=10).validate("x")  # missing cv
    with pytest.raises(ValueError):
        DistConfig("nope", mean=1).validate("x")
    DistConfig("triangular", low=1, mode=2, high=3).validate("x")


def test_shift_block_midnight_wrap():
    night = ShiftBlock(start=22, end=8, count=4)
    assert night.covers(23)
    assert night.covers(2)
    assert not night.covers(12)
    pool = StaffPoolParams(shifts=(night, ShiftBlock(start=8, end=22, count=6)))
    assert pool.rostered_at(3) == 4
    assert pool.rostered_at(12) == 6
    assert pool.max_rostered() == 6


def test_stream_open_hours(baseline: Scenario):
    sdec = baseline.streams["sdec"]
    assert sdec.is_open(12)
    assert not sdec.is_open(21)
    assert sdec.accepts_at(17.5)
    assert not sdec.accepts_at(19)  # open but past last_accept_hour


def test_baseline_yaml_is_valid_yaml_with_no_tabs():
    text = BASELINE_YAML.read_text(encoding="utf-8")
    assert "\t" not in text
    yaml.safe_load(text)
