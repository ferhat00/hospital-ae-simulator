"""Streamlit-side plumbing: cached runs, widget <-> Scenario mapping, KPI cards."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from aesim.params import Scenario
from aesim.runner import ReplicationResults, multiple_replications

REPO = Path(__file__).resolve().parents[1]
BASELINE_YAML = REPO / "config" / "baseline_dgh.yaml"

#: KPI display metadata: label, format, matching targets key (or None)
KPI_CARDS = [
    ("four_hour_type1", "4h — Type 1 ED", "{:.1%}", "four_hour_type1"),
    ("four_hour_admitted", "4h — admitted", "{:.1%}", "four_hour_admitted"),
    ("four_hour_discharged", "4h — discharged", "{:.1%}", "four_hour_discharged"),
    ("four_hour_utc", "4h — UTC", "{:.1%}", "four_hour_utc"),
    ("twelve_hour_from_arrival", "12h+ from arrival", "{:.1%}", "twelve_hour_from_arrival"),
    ("admission_rate_type1", "Admission rate (Type 1)", "{:.1%}", "admission_rate_type1"),
    ("lwbs", "Left without being seen", "{:.1%}", "lwbs"),
    ("bed_occupancy", "Bed occupancy (08:00)", "{:.1%}", "bed_occupancy"),
    ("median_boarding_min", "Median boarding", "{:.0f} min", None),
    ("handover_over_30", "Handover > 30 min", "{:.1%}", None),
    ("mean_corridor_occupied", "Corridor patients (mean)", "{:.1f}", None),
    ("excess_deaths_rcem_estimate", "RCEM excess-death est.", "{:.1f}", None),
]


@st.cache_resource(show_spinner=False)
def load_baseline() -> Scenario:
    return Scenario.from_yaml(BASELINE_YAML)


@st.cache_data(show_spinner=False, max_entries=12)
def run_cached(scenario_json: str, n_reps: int) -> ReplicationResults:
    """Cache keyed on the full scenario dict + rep count. Runs sequentially
    (never multiprocessing inside Streamlit on Windows)."""
    scenario = Scenario.from_dict(json.loads(scenario_json))
    progress = st.progress(0.0, text="Running replications...")

    def cb(i: int, n: int) -> None:
        progress.progress(i / n, text=f"Replication {i}/{n}")

    try:
        return multiple_replications(scenario, n_reps=n_reps, progress=cb)
    finally:
        progress.empty()


def run_scenario(scenario: Scenario, n_reps: int) -> ReplicationResults:
    return run_cached(json.dumps(scenario.to_dict(), sort_keys=True), n_reps)


def kpi_cards(summary, targets: dict[str, tuple[float, float]], columns: int = 4) -> None:
    """Metric cards with the NHS actual band under each value and an in/out flag."""
    cards = [c for c in KPI_CARDS if c[0] in summary.index]
    for row_start in range(0, len(cards), columns):
        cols = st.columns(columns)
        for col, (kpi, label, fmt, target_key) in zip(
            cols, cards[row_start : row_start + columns], strict=False
        ):
            mean = summary.loc[kpi, "mean"]
            lo, hi = summary.loc[kpi, "ci_low"], summary.loc[kpi, "ci_high"]
            help_text = f"95% CI [{fmt.format(lo)}, {fmt.format(hi)}]"
            delta = None
            if target_key and target_key in targets:
                t_lo, t_hi = targets[target_key]
                band = f"NHS band {fmt.format(t_lo)}–{fmt.format(t_hi)}"
                help_text += f" · {band}"
                if mean < t_lo:
                    delta, delta_color = f"below band ({band})", "off"
                elif mean > t_hi:
                    delta, delta_color = f"above band ({band})", "off"
                else:
                    delta, delta_color = "in NHS band", "off"
            with col:
                st.metric(label, fmt.format(mean), delta=delta, delta_color="off",
                          help=help_text)


def scenario_sidebar(base: Scenario) -> tuple[Scenario, int]:
    """Sidebar controls -> (Scenario, n_reps). Only overrides that differ from
    the baseline are applied, so the cache key stays stable."""
    over: dict[str, object] = {}
    st.sidebar.header("Scenario")

    with st.sidebar.expander("Demand", expanded=True):
        daily = st.slider("Attendances/day", 150, 400, int(base.arrivals.mean_daily_attendances), 5)
        _maybe(over, "arrivals.mean_daily_attendances", daily, base.arrivals.mean_daily_attendances)
        surge = st.slider("Winter surge multiplier", 0.8, 1.4, 1.0, 0.05)
        if surge != 1.0:
            over["arrivals.mean_daily_attendances"] = round(daily * surge, 1)
        cv = st.slider("Day-to-day volatility (cv)", 0.0, 0.3, float(base.arrivals.daily_cv), 0.01)
        _maybe(over, "arrivals.daily_cv", cv, base.arrivals.daily_cv)

    with st.sidebar.expander("Spaces"):
        for stream, label in [
            ("resus", "Resus bays"), ("majors", "Majors cubicles"), ("minors", "Minors rooms"),
            ("utc", "UTC rooms"), ("sdec", "SDEC chairs"),
        ]:
            cur = base.streams[stream].spaces
            val = st.slider(label, 1, max(3 * cur, 12), cur, 1, key=f"sp_{stream}")
            _maybe(over, f"streams.{stream}.spaces", val, cur)
        cur = base.spaces.get("corridor_spaces", 0)
        val = st.slider("Corridor/escalation spaces", 0, 24, cur, 1)
        _maybe(over, "spaces.corridor_spaces", val, cur)
        cur = base.spaces.get("offload_spaces", 0)
        val = st.slider("Ambulance offload (HALO) spaces", 0, 12, cur, 1)
        _maybe(over, "spaces.offload_spaces", val, cur)

    with st.sidebar.expander("Staffing"):
        doc_shifts = list(base.staff_pools["ed_doctors"].shifts)
        labels = ["ED doctors — day", "ED doctors — twilight", "ED doctors — night"]
        new_counts = []
        for shift, label in zip(doc_shifts, labels, strict=False):
            new_counts.append(st.slider(label, 1, 20, shift.count, 1, key=label))
        if [s.count for s in doc_shifts] != new_counts:
            over["staff_pools.ed_doctors.shifts"] = [
                {"start": s.start, "end": s.end, "count": c}
                for s, c in zip(doc_shifts, new_counts, strict=False)
            ]
        nurse_shifts = list(base.staff_pools["ed_nurses"].shifts)
        n_labels = ["ED nurses — day", "ED nurses — night"]
        n_counts = []
        for shift, label in zip(nurse_shifts, n_labels, strict=False):
            n_counts.append(st.slider(label, 2, 24, shift.count, 1, key=label))
        if [s.count for s in nurse_shifts] != n_counts:
            over["staff_pools.ed_nurses.shifts"] = [
                {"start": s.start, "end": s.end, "count": c}
                for s, c in zip(nurse_shifts, n_counts, strict=False)
            ]
        avail = st.slider(
            "Availability factor (breaks etc.)", 0.7, 1.0,
            float(base.staff_pools["ed_doctors"].availability_factor), 0.01,
        )
        _maybe(over, "staff_pools.ed_doctors.availability_factor", avail,
               base.staff_pools["ed_doctors"].availability_factor)

    with st.sidebar.expander("Process & streaming"):
        utc_scale = st.slider("UTC streaming multiplier", 0.0, 1.5, 1.0, 0.05,
                              help="Scales the probability that eligible walk-ins go to the UTC")
        if utc_scale != 1.0:
            over["routing.utc_stream_prob"] = {
                k: min(v * utc_scale, 1.0) for k, v in base.routing.utc_stream_prob.items()
            }
        sdec_scale = st.slider("SDEC eligibility multiplier", 0.0, 2.0, 1.0, 0.05)
        if sdec_scale != 1.0:
            over["routing.sdec_eligible_frac"] = {
                k: min(v * sdec_scale, 1.0) for k, v in base.routing.sdec_eligible_frac.items()
            }
        hot = st.slider("Hot-clinic diversion of admissions", 0.0, 0.3,
                        float(base.routing.hot_clinic_frac), 0.01)
        _maybe(over, "routing.hot_clinic_frac", hot, base.routing.hot_clinic_frac)

    with st.sidebar.expander("Beds (exit block)"):
        cur = base.beds.n_beds
        val = st.slider("Inpatient beds", max(cur - 80, 50), cur + 120, cur, 1)
        _maybe(over, "beds.n_beds", val, cur)
        cur_f = float(base.beds.los_days.mean)
        los = st.slider("Mean inpatient LoS (days)", 3.0, 8.0, cur_f, 0.1)
        if los != cur_f:
            over["beds.los_days"] = {"kind": base.beds.los_days.kind, "mean": los,
                                     "cv": base.beds.los_days.cv}
        cur_f = float(base.beds.turnaround.mean)
        turn = st.slider("Bed turnaround (min)", 0.0, 240.0, cur_f, 5.0)
        if turn != cur_f:
            over["beds.turnaround"] = {"kind": base.beds.turnaround.kind, "mean": turn,
                                       "cv": base.beds.turnaround.cv}
        cur_f = float(base.beds.elective_demand_per_day)
        el = st.slider("Elective bed demand/day", 0.0, 40.0, cur_f, 1.0)
        _maybe(over, "beds.elective_demand_per_day", el, cur_f)
        cur_f = float(base.beds.weekend_discharge_slip)
        slip = st.slider("Weekend discharge slippage", 0.0, 0.8, cur_f, 0.05)
        _maybe(over, "beds.weekend_discharge_slip", slip, cur_f)

    with st.sidebar.expander("Run settings"):
        reps = st.slider("Replications", 2, 50, 10, 1)
        days = st.slider("Run length (days)", 21, 91, base.run.run_length_days, 7)
        _maybe(over, "run.run_length_days", days, base.run.run_length_days)
        seed = st.number_input("Master seed", 0, 10_000, base.run.random_seed)
        _maybe(over, "run.random_seed", int(seed), base.run.random_seed)

    scenario = base.with_overrides(**over) if over else base
    return scenario, int(reps)


def _maybe(over: dict, path: str, value, current) -> None:
    if value != current:
        over[path] = value
