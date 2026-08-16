"""Scenario parameters: the sole input surface of the simulation.

Every number the model uses lives in a frozen dataclass here, populated from
``config/baseline_dgh.yaml`` (optionally deep-merged with a partial override
file), and overridable programmatically with dotted paths::

    s = Scenario.from_yaml("config/baseline_dgh.yaml")
    s2 = s.with_overrides(**{"streams.majors.spaces": 28, "beds.n_beds": 420})

Triage-category-keyed mappings use string keys ("RED".."BLUE") throughout so the
whole tree round-trips through YAML/JSON losslessly; the model converts to
:class:`aesim.entities.TriageCat` at the point of use.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CAT_NAMES = ("RED", "ORANGE", "YELLOW", "GREEN", "BLUE")

_DIST_KINDS = ("lognormal", "gamma", "exponential", "triangular", "uniform", "fixed")


@dataclass(frozen=True)
class DistConfig:
    """A service-time / duration distribution, parameterised on the natural scale.

    ``mean`` and ``cv`` (coefficient of variation, sd/mean) apply to lognormal and
    gamma. Exponential and fixed use ``mean`` only. Triangular/uniform use
    ``low``/``mode``/``high`` (mode ignored for uniform).
    """

    kind: str
    mean: float = 0.0
    cv: float | None = None
    low: float | None = None
    mode: float | None = None
    high: float | None = None

    def validate(self, where: str) -> None:
        if self.kind not in _DIST_KINDS:
            raise ValueError(f"{where}: unknown distribution kind {self.kind!r}")
        if self.kind in ("lognormal", "gamma"):
            if self.mean <= 0 or self.cv is None or self.cv <= 0:
                raise ValueError(f"{where}: {self.kind} needs mean>0 and cv>0")
        elif self.kind in ("exponential", "fixed"):
            if self.mean < 0:
                raise ValueError(f"{where}: {self.kind} needs mean>=0")
        elif self.kind in ("triangular", "uniform"):
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError(f"{where}: {self.kind} needs low<=high")
            if self.kind == "triangular":
                m = self.mode if self.mode is not None else (self.low + self.high) / 2
                if not (self.low <= m <= self.high):
                    raise ValueError(f"{where}: triangular mode outside [low, high]")


@dataclass(frozen=True)
class ShiftBlock:
    """One roster line: ``count`` staff on duty from ``start`` to ``end`` (clock hours).

    A block wraps midnight when ``end < start`` (e.g. 22 -> 8).
    """

    start: float
    end: float
    count: int

    def covers(self, hour_of_day: float) -> bool:
        if self.start <= self.end:
            return self.start <= hour_of_day < self.end
        return hour_of_day >= self.start or hour_of_day < self.end


@dataclass(frozen=True)
class StaffPoolParams:
    shifts: tuple[ShiftBlock, ...]
    availability_factor: float = 1.0  # fraction of rostered hours actually worked

    def rostered_at(self, hour_of_day: float) -> int:
        """Scheduled headcount at a clock hour (before the availability factor)."""
        return sum(b.count for b in self.shifts if b.covers(hour_of_day))

    def max_rostered(self) -> int:
        return max(self.rostered_at(h / 2) for h in range(48))

    def validate(self, where: str) -> None:
        if not self.shifts:
            raise ValueError(f"{where}: staff pool has no shifts")
        if not (0 < self.availability_factor <= 1):
            raise ValueError(f"{where}: availability_factor must be in (0, 1]")
        for b in self.shifts:
            if not (0 <= b.start < 24 and 0 <= b.end <= 24):
                raise ValueError(f"{where}: shift hours must be within 0-24")
            if b.count < 0:
                raise ValueError(f"{where}: negative shift count")


@dataclass(frozen=True)
class StreamParams:
    """One care stream (resus / majors / minors / utc / sdec)."""

    spaces: int
    clinician_pool: str
    nurse_pool: str | None = None
    assessment_time: DistConfig | None = None
    treatment_time: DistConfig | None = None
    nursing_time: DistConfig | None = None  # obs/cannula/meds by the nurse pool
    open_hours: tuple[float, float] | None = None  # None => 24/7
    last_accept_hour: float | None = None  # SDEC-style intake cutoff
    stepdown_to_majors_prob: float = 0.0  # resus: survivors moved on to majors

    def is_open(self, hour_of_day: float) -> bool:
        if self.open_hours is None:
            return True
        lo, hi = self.open_hours
        if lo <= hi:
            return lo <= hour_of_day < hi
        return hour_of_day >= lo or hour_of_day < hi

    def accepts_at(self, hour_of_day: float) -> bool:
        if not self.is_open(hour_of_day):
            return False
        if self.last_accept_hour is not None:
            lo = self.open_hours[0] if self.open_hours else 0.0
            if lo <= self.last_accept_hour:
                return lo <= hour_of_day < self.last_accept_hour
            return hour_of_day >= lo or hour_of_day < self.last_accept_hour
        return True


@dataclass(frozen=True)
class ArrivalParams:
    mean_daily_attendances: float
    hourly_profile: tuple[float, ...]  # 24 relative weights, normalised internally
    dow_multipliers: tuple[float, ...]  # 7 values, Monday first
    ambulance_fraction_by_hour: tuple[float, ...]  # 24 values in [0, 1]
    booked_utc_per_day: float = 0.0
    booked_no_show_prob: float = 0.0
    paeds_fraction: float = 0.0  # routed through minors/majors, reported separately
    daily_cv: float = 0.0  # lognormal day-to-day demand volatility (flu, weather, events)


@dataclass(frozen=True)
class TriageParams:
    cat_mix_walkin: dict[str, float]
    cat_mix_ambulance: dict[str, float]
    registration_time: DistConfig = field(
        default_factory=lambda: DistConfig("lognormal", mean=4.0, cv=0.5)
    )
    triage_time_walkin: DistConfig = field(
        default_factory=lambda: DistConfig("gamma", mean=6.5, cv=0.4)
    )
    triage_time_ambulance: DistConfig = field(
        default_factory=lambda: DistConfig("gamma", mean=11.0, cv=0.4)
    )
    patience_min: dict[str, float] = field(default_factory=dict)  # LWBS reneging, by cat


@dataclass(frozen=True)
class DiagnosticTest:
    prob: float
    turnaround: DistConfig


@dataclass(frozen=True)
class RoutingParams:
    admission_prob_by_cat: dict[str, float]
    specialty_review_time: DistConfig
    utc_stream_prob: dict[str, float] = field(default_factory=dict)  # walk-ins, by cat
    utc_transfer_to_majors: float = 0.04
    sdec_eligible_frac: dict[str, float] = field(default_factory=dict)  # by cat
    sdec_conversion_to_admission: float = 0.25
    hot_clinic_frac: float = 0.0  # of would-be admissions, discharged to next-day clinic
    handover_clinical_min: float = 15.0  # fixed clinical handover once a space is free
    # Corridor/escalation care: a boarder is decanted from their cubicle to a
    # corridor space when others are queueing for the cubicle. Protects flow at
    # the cost of corridor care (itself a harm KPI). 0 corridor spaces = never.
    corridor_trigger_queue: int = 1  # decant when > this many wait for the space


@dataclass(frozen=True)
class BedPoolParams:
    n_beds: int
    los_days: DistConfig
    discharge_hour_weights: tuple[float, ...]  # 24 weights, afternoon-heavy
    turnaround: DistConfig = field(
        default_factory=lambda: DistConfig("fixed", mean=0.0)
    )  # clean/ward-acceptance friction between discharge and next occupant (minutes)
    weekend_discharge_slip: float = 0.0  # prob a Sat/Sun discharge slips to Monday
    elective_demand_per_day: float = 0.0
    elective_cancel_hour: float = 18.0
    initial_occupancy: float = 0.93
    enabled: bool = True  # False => infinite beds (no exit block); used by tests/M1


@dataclass(frozen=True)
class RunParams:
    run_length_days: int = 42
    warm_up_days: int = 14
    audit_interval_min: float = 15.0
    random_seed: int = 42
    default_reps: int = 30

    @property
    def run_length_min(self) -> float:
        return self.run_length_days * 1440.0

    @property
    def warm_up_min(self) -> float:
        return self.warm_up_days * 1440.0


@dataclass(frozen=True)
class Scenario:
    meta: dict[str, Any]
    run: RunParams
    arrivals: ArrivalParams
    triage: TriageParams
    streams: dict[str, StreamParams]
    diagnostics: dict[str, DiagnosticTest]
    staff_pools: dict[str, StaffPoolParams]
    spaces: dict[str, int]  # triage_rooms, offload_spaces
    routing: RoutingParams
    beds: BedPoolParams
    targets: dict[str, tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: str | Path | None = None) -> Scenario:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if overrides is not None:
            with open(overrides, encoding="utf-8") as f:
                data = _deep_merge(data, yaml.safe_load(f) or {})
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        d = copy.deepcopy(data)
        scenario = cls(
            meta=d.get("meta", {}),
            run=_build(RunParams, d.get("run", {})),
            arrivals=_build(ArrivalParams, d["arrivals"]),
            triage=_build(TriageParams, d["triage"]),
            streams={k: _build(StreamParams, v) for k, v in d["streams"].items()},
            diagnostics={k: _build(DiagnosticTest, v) for k, v in d.get("diagnostics", {}).items()},
            staff_pools={k: _build(StaffPoolParams, v) for k, v in d["staff_pools"].items()},
            spaces=dict(d.get("spaces", {})),
            routing=_build(RoutingParams, d["routing"]),
            beds=_build(BedPoolParams, d["beds"]),
            targets={k: (float(v[0]), float(v[1])) for k, v in d.get("targets", {}).items()},
        )
        scenario.validate()
        return scenario

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **dotted: Any) -> Scenario:
        """Return a copy with dotted-path overrides applied, e.g.
        ``with_overrides(**{"streams.majors.spaces": 28})``."""
        d = self.to_dict()
        for path, value in dotted.items():
            node = d
            *parents, leaf = path.split(".")
            for key in parents:
                if key not in node or not isinstance(node[key], dict):
                    raise KeyError(f"override path {path!r}: {key!r} not found")
                node = node[key]
            if leaf not in node:
                raise KeyError(f"override path {path!r}: {leaf!r} not found")
            node[leaf] = value
        return Scenario.from_dict(d)

    def content_hash(self) -> str:
        """Stable hash of every parameter; used for caching and CRN pairing."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # --------------------------------------------------------------- validation

    def validate(self) -> None:
        a = self.arrivals
        _expect_len("arrivals.hourly_profile", a.hourly_profile, 24)
        _expect_len("arrivals.dow_multipliers", a.dow_multipliers, 7)
        _expect_len("arrivals.ambulance_fraction_by_hour", a.ambulance_fraction_by_hour, 24)
        if a.mean_daily_attendances <= 0:
            raise ValueError("arrivals.mean_daily_attendances must be > 0")
        if any(w < 0 for w in a.hourly_profile) or sum(a.hourly_profile) <= 0:
            raise ValueError("arrivals.hourly_profile weights must be >= 0 and sum > 0")
        if any(not (0 <= f <= 1) for f in a.ambulance_fraction_by_hour):
            raise ValueError("arrivals.ambulance_fraction_by_hour values must be in [0, 1]")
        if not (0 <= a.paeds_fraction < 1):
            raise ValueError("arrivals.paeds_fraction must be in [0, 1)")
        if a.daily_cv < 0:
            raise ValueError("arrivals.daily_cv must be >= 0")

        for name, mix in (
            ("cat_mix_walkin", self.triage.cat_mix_walkin),
            ("cat_mix_ambulance", self.triage.cat_mix_ambulance),
        ):
            _expect_cat_probs(f"triage.{name}", mix, must_sum_to_one=True)
        for dist_name in ("registration_time", "triage_time_walkin", "triage_time_ambulance"):
            getattr(self.triage, dist_name).validate(f"triage.{dist_name}")

        for sname, sp in self.streams.items():
            if sp.spaces < 0:
                raise ValueError(f"streams.{sname}.spaces must be >= 0")
            if sp.clinician_pool not in self.staff_pools:
                raise ValueError(
                    f"streams.{sname}.clinician_pool {sp.clinician_pool!r} not in staff_pools"
                )
            if sp.nurse_pool is not None and sp.nurse_pool not in self.staff_pools:
                raise ValueError(f"streams.{sname}.nurse_pool {sp.nurse_pool!r} not in staff_pools")
            for dname in ("assessment_time", "treatment_time", "nursing_time"):
                dist = getattr(sp, dname)
                if dist is not None:
                    dist.validate(f"streams.{sname}.{dname}")
            if sp.nursing_time is not None and sp.nurse_pool is None:
                raise ValueError(f"streams.{sname}: nursing_time requires a nurse_pool")
            if not (0 <= sp.stepdown_to_majors_prob <= 1):
                raise ValueError(f"streams.{sname}.stepdown_to_majors_prob must be in [0, 1]")

        for pname, pool in self.staff_pools.items():
            pool.validate(f"staff_pools.{pname}")

        for tname, test in self.diagnostics.items():
            if not (0 <= test.prob <= 1):
                raise ValueError(f"diagnostics.{tname}.prob must be in [0, 1]")
            test.turnaround.validate(f"diagnostics.{tname}.turnaround")

        r = self.routing
        _expect_cat_probs("routing.admission_prob_by_cat", r.admission_prob_by_cat)
        _expect_cat_probs("routing.utc_stream_prob", r.utc_stream_prob, allow_missing=True)
        _expect_cat_probs("routing.sdec_eligible_frac", r.sdec_eligible_frac, allow_missing=True)
        r.specialty_review_time.validate("routing.specialty_review_time")
        for pname_, val in (
            ("utc_transfer_to_majors", r.utc_transfer_to_majors),
            ("sdec_conversion_to_admission", r.sdec_conversion_to_admission),
            ("hot_clinic_frac", r.hot_clinic_frac),
        ):
            if not (0 <= val <= 1):
                raise ValueError(f"routing.{pname_} must be in [0, 1]")

        b = self.beds
        if b.enabled:
            if b.n_beds < 1:
                raise ValueError(
                    "beds.n_beds must be >= 1 when beds.enabled; "
                    "set beds.enabled: false for an unconstrained (infinite) bed supply"
                )
            _expect_len("beds.discharge_hour_weights", b.discharge_hour_weights, 24)
            if sum(b.discharge_hour_weights) <= 0:
                raise ValueError("beds.discharge_hour_weights must sum > 0")
            if not (0 <= b.initial_occupancy <= 1):
                raise ValueError("beds.initial_occupancy must be in [0, 1]")
            b.los_days.validate("beds.los_days")
            b.turnaround.validate("beds.turnaround")
            if not (0 <= b.weekend_discharge_slip <= 1):
                raise ValueError("beds.weekend_discharge_slip must be in [0, 1]")

        if self.run.warm_up_days >= self.run.run_length_days:
            raise ValueError("run.warm_up_days must be < run.run_length_days")

        for key in ("triage_rooms",):
            if key not in self.spaces:
                raise ValueError(f"spaces.{key} is required")


# ------------------------------------------------------------------ helpers


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct a dataclass from a plain dict, converting nested
    DistConfig / ShiftBlock / tuple fields as declared by type hints."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        kwargs[f.name] = _convert(f.type, value)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**kwargs)


def _convert(type_hint: Any, value: Any) -> Any:
    hint = str(type_hint)
    if value is None:
        return None
    if "DistConfig" in hint and isinstance(value, dict):
        cfg = _build(DistConfig, value)
        return cfg
    if "ShiftBlock" in hint and isinstance(value, (list, tuple)):
        return tuple(_build(ShiftBlock, v) if isinstance(v, dict) else v for v in value)
    if "DiagnosticTest" in hint and isinstance(value, dict) and "prob" in value:
        return _build(DiagnosticTest, value)
    if "tuple" in hint and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expect_len(name: str, seq: tuple, n: int) -> None:
    if len(seq) != n:
        raise ValueError(f"{name} must have exactly {n} entries, got {len(seq)}")


def _expect_cat_probs(
    name: str,
    mapping: dict[str, float],
    must_sum_to_one: bool = False,
    allow_missing: bool = False,
) -> None:
    unknown = set(mapping) - set(CAT_NAMES)
    if unknown:
        raise ValueError(f"{name}: unknown triage categories {sorted(unknown)}")
    if not allow_missing and set(mapping) != set(CAT_NAMES):
        raise ValueError(f"{name}: must define all of {CAT_NAMES}")
    for cat, p in mapping.items():
        if not (0 <= p <= 1):
            raise ValueError(f"{name}.{cat} must be in [0, 1]")
    if must_sum_to_one and abs(sum(mapping.values()) - 1.0) > 1e-6:
        raise ValueError(f"{name}: probabilities must sum to 1, got {sum(mapping.values()):.4f}")
