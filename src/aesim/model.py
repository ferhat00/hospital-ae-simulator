"""AEModel: wires environment, resources, samplers and generators together.

Random-number discipline: one master ``SeedSequence`` per replication is split
into one child per *sampling purpose* in a fixed registration order
(``SEED_PURPOSES`` + sorted diagnostics + sorted staff pools). Replication i of
every scenario therefore sees identical arrival instants, triage categories and
service draws (common random numbers), so scenario comparisons are paired.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

from aesim.arrivals import MIN_PER_DAY, ArrivalRateProfile, NSPPThinning, booked_slot_times
from aesim.distributions import (
    Bernoulli,
    Categorical,
    DischargeHourSampler,
    Sampler,
    UniformDraw,
)
from aesim.entities import ArrivalMode, Patient, Stream
from aesim.params import CAT_NAMES, Scenario
from aesim.pathways import PathwaysMixin
from aesim.resources import (
    BED_PRIORITY_ED,
    BED_PRIORITY_ELECTIVE,
    BedPool,
    ShiftScheduledResource,
    SpaceStore,
)

SEED_PURPOSES = (
    "walkin_iat",
    "walkin_thin",
    "amb_iat",
    "amb_thin",
    "cat_walkin",
    "cat_ambulance",
    "paeds",
    "reg_time",
    "triage_time_walkin",
    "triage_time_ambulance",
    "route_utc",
    "route_admit",
    "route_sdec_elig",
    "route_sdec_convert",
    "route_hot_clinic",
    "route_stepdown",
    "route_utc_transfer",
    "booked_noshow",
    "svc_resus",
    "svc_majors_assess",
    "svc_majors_treat",
    "svc_minors",
    "svc_utc",
    "svc_sdec",
    "specialty_review",
    "bed_los",
    "bed_discharge_hour",
    "bed_prefill",
    "elective_arrivals",
    "bed_turnaround",
    "day_effects",
    "weekend_slip",
)


@dataclass
class RawResults:
    scenario: Scenario
    rep: int
    patients: list[Patient] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    elective_cancellations: int = 0
    corridor_minutes: float = 0.0


class AEModel(PathwaysMixin):
    def __init__(self, scenario: Scenario, rep: int = 0):
        self.scenario = scenario
        self.rep = rep
        self.env = simpy.Environment()
        self.results = _ResultsCollector()
        self._held_space: dict[int, tuple] = {}
        self._all_patients: dict[int, Patient] = {}
        self._pid = 0
        self.n_in_dept = 0
        self.n_boarding = 0
        self.n_ambulances_waiting = 0
        self.event_log: list[dict[str, Any]] = []
        self.audit_rows: list[dict[str, Any]] = []

        self._build_streams_and_seeds(rep)
        self._build_resources()

    # ------------------------------------------------------------------- seeds

    def _build_streams_and_seeds(self, rep: int) -> None:
        s = self.scenario
        master = np.random.SeedSequence(s.run.random_seed)
        rep_seed = master.spawn(rep + 1)[rep]

        purposes = list(SEED_PURPOSES)
        for test in sorted(s.diagnostics):
            purposes += [f"dx_{test}_p", f"dx_{test}_t"]
        nursing_streams = sorted(n for n, sp in s.streams.items() if sp.nursing_time is not None)
        purposes += [f"svc_nursing_{n}" for n in nursing_streams]
        pool_names = sorted(s.staff_pools)
        purposes += [f"roster_{name}" for name in pool_names]
        children = rep_seed.spawn(len(purposes))
        seed_of = dict(zip(purposes, children, strict=True))
        self._roster_seeds = {name: seed_of[f"roster_{name}"] for name in pool_names}

        d: dict[str, Any] = {}
        d["cat_walkin"] = Categorical(
            list(CAT_NAMES),
            [s.triage.cat_mix_walkin[c] for c in CAT_NAMES],
            seed_of["cat_walkin"],
        )
        d["cat_ambulance"] = Categorical(
            list(CAT_NAMES),
            [s.triage.cat_mix_ambulance[c] for c in CAT_NAMES],
            seed_of["cat_ambulance"],
        )
        d["paeds"] = Bernoulli(s.arrivals.paeds_fraction, seed_of["paeds"])
        d["booked_noshow"] = Bernoulli(s.arrivals.booked_no_show_prob, seed_of["booked_noshow"])
        d["reg_time"] = Sampler(s.triage.registration_time, seed_of["reg_time"])
        d["triage_time_walkin"] = Sampler(s.triage.triage_time_walkin, seed_of["triage_time_walkin"])
        d["triage_time_ambulance"] = Sampler(
            s.triage.triage_time_ambulance, seed_of["triage_time_ambulance"]
        )
        for name in (
            "route_utc",
            "route_admit",
            "route_sdec_elig",
            "route_sdec_convert",
            "route_hot_clinic",
            "route_stepdown",
            "route_utc_transfer",
        ):
            d[name] = UniformDraw(seed_of[name])

        d["svc_resus"] = Sampler(s.streams["resus"].treatment_time, seed_of["svc_resus"])
        d["svc_majors_assess"] = Sampler(
            s.streams["majors"].assessment_time, seed_of["svc_majors_assess"]
        )
        d["svc_majors_treat"] = Sampler(
            s.streams["majors"].treatment_time, seed_of["svc_majors_treat"]
        )
        d["svc_minors"] = Sampler(s.streams["minors"].treatment_time, seed_of["svc_minors"])
        d["svc_utc"] = Sampler(s.streams["utc"].treatment_time, seed_of["svc_utc"])
        d["svc_sdec"] = Sampler(s.streams["sdec"].treatment_time, seed_of["svc_sdec"])
        d["specialty_review"] = Sampler(s.routing.specialty_review_time, seed_of["specialty_review"])

        for test, cfg in s.diagnostics.items():
            d[f"dx_{test}_p"] = UniformDraw(seed_of[f"dx_{test}_p"])
            d[f"dx_{test}_t"] = Sampler(cfg.turnaround, seed_of[f"dx_{test}_t"])
        for name in nursing_streams:
            d[f"svc_nursing_{name}"] = Sampler(
                s.streams[name].nursing_time, seed_of[f"svc_nursing_{name}"]
            )

        d["bed_los"] = Sampler(s.beds.los_days, seed_of["bed_los"])
        d["bed_prefill"] = Sampler(s.beds.los_days, seed_of["bed_prefill"])
        d["bed_turnaround"] = Sampler(s.beds.turnaround, seed_of["bed_turnaround"])
        self.dists = d

        self.discharge_hours = DischargeHourSampler(
            s.beds.discharge_hour_weights, seed_of["bed_discharge_hour"]
        )
        self._elective_rng = np.random.default_rng(seed_of["elective_arrivals"])

        # day-to-day demand volatility, shared across arrival modes
        day_factors = None
        if s.arrivals.daily_cv > 0:
            import math as _math

            cv = s.arrivals.daily_cv
            sigma2 = _math.log(1 + cv**2)
            day_rng = np.random.default_rng(seed_of["day_effects"])
            day_factors = day_rng.lognormal(
                -sigma2 / 2, _math.sqrt(sigma2), size=s.run.run_length_days + 1
            )
        self._weekend_slip_rng = np.random.default_rng(seed_of["weekend_slip"])

        self._walkin_gen = NSPPThinning(
            ArrivalRateProfile(s.arrivals, "walk_in"),
            seed_of["walkin_iat"],
            seed_of["walkin_thin"],
            day_factors=day_factors,
        )
        self._amb_gen = NSPPThinning(
            ArrivalRateProfile(s.arrivals, "ambulance"),
            seed_of["amb_iat"],
            seed_of["amb_thin"],
            day_factors=day_factors,
        )

    # --------------------------------------------------------------- resources

    def _build_resources(self) -> None:
        s = self.scenario
        resus_pool = s.streams["resus"].clinician_pool
        self.staff: dict[str, ShiftScheduledResource] = {}
        for name, pool in s.staff_pools.items():
            self.staff[name] = ShiftScheduledResource(
                self.env,
                name,
                pool,
                self._roster_seeds[name],
                preemptive=(name == resus_pool),
            )
        self.triage_nurses = self.staff["triage_nurses"]

        self.spaces: dict[Stream, SpaceStore] = {}
        for sname, sp in s.streams.items():
            stream = Stream(sname)
            self.spaces[stream] = SpaceStore(self.env, f"space_{sname}", sp.spaces)
        self.triage_rooms = SpaceStore(self.env, "triage_room", s.spaces.get("triage_rooms", 0))
        self.offload_spaces = SpaceStore(self.env, "offload", s.spaces.get("offload_spaces", 0))
        self.corridor = SpaceStore(self.env, "corridor", s.spaces.get("corridor_spaces", 0))
        self.corridor_minutes = 0.0  # total boarder-minutes spent in corridor care

        self.beds = BedPool(self.env, s.beds)
        if self.beds.enabled:
            self.env.process(self._prefill_beds())

    # ------------------------------------------------------------------- events

    def log_event(self, p: Patient, event_type: str, event: str, resource_id=None) -> None:
        if event_type == "arrival_departure":
            if event == "arrival":
                self.n_in_dept += 1
                if p.mode is ArrivalMode.AMBULANCE:
                    self.n_ambulances_waiting += 1
            else:
                self.n_in_dept -= 1
        self.event_log.append(
            {
                "entity_id": p.pid,
                "event_type": event_type,
                "event": event,
                "time": self.env.now,
                "resource_id": resource_id,
                "run": self.rep,
                "pathway": p.stream.value if p.stream else None,
                "cat": p.cat.name if p.cat else None,
                "mode": p.mode.value,
            }
        )

    # --------------------------------------------------------------- processes

    def _new_patient(self, mode: ArrivalMode) -> Patient:
        self._pid += 1
        p = Patient(
            pid=self._pid,
            mode=mode,
            t_arrival=self.env.now,
            in_warmup=self.env.now < self.scenario.run.warm_up_min,
        )
        self._all_patients[p.pid] = p
        return p

    def _walkin_arrivals(self) -> Generator:
        while True:
            t = self._walkin_gen.next_arrival(self.env.now)
            if t == float("inf"):
                return
            yield self.env.timeout(t - self.env.now)
            self.env.process(self.patient_journey(self._new_patient(ArrivalMode.WALK_IN)))

    def _ambulance_arrivals(self) -> Generator:
        while True:
            t = self._amb_gen.next_arrival(self.env.now)
            if t == float("inf"):
                return
            yield self.env.timeout(t - self.env.now)
            self.env.process(self.patient_journey(self._new_patient(ArrivalMode.AMBULANCE)))

    def note_offload(self) -> None:
        """Called by the pathway the moment a crew is released."""
        self.n_ambulances_waiting -= 1

    def _booked_arrivals(self) -> Generator:
        utc = self.scenario.streams.get("utc")
        if utc is None or utc.open_hours is None:
            return
        slots = booked_slot_times(self.scenario.arrivals, utc.open_hours)
        if not slots:
            return
        day = 0
        while True:
            day_start = day * MIN_PER_DAY
            for slot in sorted(slots):
                t = day_start + slot
                if t < self.env.now:
                    continue
                yield self.env.timeout(t - self.env.now)
                if not self.dists["booked_noshow"].sample():
                    self.env.process(self.patient_journey(self._new_patient(ArrivalMode.BOOKED_UTC)))
            day += 1
            if day * MIN_PER_DAY > self.scenario.run.run_length_min:
                return

    def apply_weekend_slip(self, departure: float) -> float:
        """Reduced weekend discharging: a Sat/Sun ward discharge may slip to
        Monday (same clock hour) — the driver of Monday-morning gridlock."""
        slip = self.scenario.beds.weekend_discharge_slip
        if slip <= 0:
            return departure
        dow = int(departure // MIN_PER_DAY) % 7  # day 0 = Monday
        if dow >= 5 and self._weekend_slip_rng.uniform() < slip:
            return departure + (7 - dow) * MIN_PER_DAY
        return departure

    def _prefill_beds(self) -> Generator:
        """Seed t=0 ward occupancy with synthetic patients on residual stays."""
        n = round(self.scenario.beds.initial_occupancy * self.scenario.beds.n_beds)
        for _ in range(n):
            req = self.beds.request(priority=BED_PRIORITY_ED)
            yield req
            self.env.process(self._prefill_stay(req))

    def _prefill_stay(self, req) -> Generator:
        residual_min = self.dists["bed_prefill"].residual_sample() * 1440.0
        departure = self.apply_weekend_slip(self.discharge_hours.snap(0.0, residual_min))
        yield self.env.timeout(max(departure, 1.0))
        yield self.env.timeout(self.dists["bed_turnaround"].sample())
        self.beds.release(req)

    def _elective_admissions(self) -> Generator:
        """Non-ED bed demand: N ~ Poisson(demand/day) electives arrive 08:00-16:00;
        an elective still bedless at the cancel hour is cancelled (a KPI)."""
        rate = self.scenario.beds.elective_demand_per_day
        if rate <= 0 or not self.beds.enabled:
            return
        day = 0
        while day * MIN_PER_DAY <= self.scenario.run.run_length_min:
            day_start = day * MIN_PER_DAY
            n = int(self._elective_rng.poisson(rate))
            times = sorted(self._elective_rng.uniform(8 * 60.0, 16 * 60.0, size=n))
            for offset in times:
                t = day_start + offset
                if t < self.env.now:
                    continue
                yield self.env.timeout(t - self.env.now)
                self.env.process(self._elective_journey(day_start))
            next_day = (day + 1) * MIN_PER_DAY
            if next_day > self.env.now:
                yield self.env.timeout(next_day - self.env.now)
            day += 1

    def _elective_journey(self, day_start: float) -> Generator:
        cancel_at = day_start + self.scenario.beds.elective_cancel_hour * 60.0
        req = self.beds.request(priority=BED_PRIORITY_ELECTIVE)
        result = yield req | self.env.timeout(max(cancel_at - self.env.now, 0.0))
        if req not in result:
            req.cancel()
            self.beds.elective_cancellations += 1
            return
        los_min = self.dists["bed_los"].sample() * 1440.0
        departure = self.apply_weekend_slip(
            self.discharge_hours.snap(self.env.now, self.env.now + los_min)
        )
        yield self.env.timeout(max(departure - self.env.now, 1.0))
        yield self.env.timeout(self.dists["bed_turnaround"].sample())
        self.beds.release(req)

    # --------------------------------------------------------------------- run

    def run(self) -> RawResults:
        from aesim.auditor import audit_process

        self.env.process(self._walkin_arrivals())
        self.env.process(self._ambulance_arrivals())
        self.env.process(self._booked_arrivals())
        self.env.process(self._elective_admissions())
        self.env.process(audit_process(self))
        self.env.run(until=self.scenario.run.run_length_min)
        return RawResults(
            scenario=self.scenario,
            rep=self.rep,
            patients=list(self.results.completed) + self._in_flight_patients(),
            event_log=self.event_log,
            audit=self.audit_rows,
            elective_cancellations=self.beds.elective_cancellations,
            corridor_minutes=self.corridor_minutes,
        )

    def _in_flight_patients(self) -> list[Patient]:
        """Patients still in the department when the run ends (no departure)."""
        done = {p.pid for p in self.results.completed}
        return [p for p in self._all_patients.values() if p.pid not in done]


class _ResultsCollector:
    def __init__(self) -> None:
        self.completed: list[Patient] = []
