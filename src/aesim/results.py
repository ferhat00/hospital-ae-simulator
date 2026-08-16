"""Turn raw simulation output into patient-level DataFrames and NHS-comparable KPIs.

Conventions:
- KPIs are computed over patients who ARRIVED after the warm-up and DEPARTED
  before the run ended (plus LWBS, who count as attendances and departures, as
  they do in NHS statistics).
- "Type 1" = resus/majors/minors/SDEC streams; "UTC" = the Type 3 stream.
- The 4-hour measure is time from arrival to departure (admission = bed granted,
  discharge = leaving) <= 240 min; 12-hour measures use 720 min.
- The RCEM harm estimate (1 excess death per 72 patients waiting 8-12h before
  admission) is a published estimate applied to model output, not a model
  prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aesim.entities import MTS_TARGET_MIN, Disposal, Patient, Stream, TriageCat
from aesim.model import RawResults

FOUR_HOURS = 240.0
TWELVE_HOURS = 720.0
TYPE1_STREAMS = {Stream.RESUS, Stream.MAJORS, Stream.MINORS, Stream.SDEC}


@dataclass
class ResultsBundle:
    scenario_name: str
    rep: int
    patients: pd.DataFrame
    audit: pd.DataFrame
    kpis: dict[str, float]
    event_log: pd.DataFrame | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def patients_to_frame(patients: list[Patient]) -> pd.DataFrame:
    rows = []
    for p in patients:
        rows.append(
            {
                "pid": p.pid,
                "mode": p.mode.value,
                "cat": p.cat.name if p.cat else None,
                "stream": p.stream.value if p.stream else None,
                "disposal": p.disposal.value if p.disposal else None,
                "is_paeds": p.is_paeds,
                "in_warmup": p.in_warmup,
                "admitted": p.admitted,
                "t_arrival": p.t_arrival,
                "t_offload": p.t_offload,
                "t_triage_start": p.t_triage_start,
                "t_triage_end": p.t_triage_end,
                "t_space": p.t_space,
                "t_first_clinician": p.t_first_clinician,
                "t_dta": p.t_dta,
                "t_bed": p.t_bed,
                "t_departure": p.t_departure,
                "time_in_department": p.time_in_department,
                "time_to_triage": p.time_to_triage,
                "time_to_first_clinician": p.time_to_first_clinician,
                "boarding_time": p.boarding_time,
                "handover_delay": p.handover_delay,
            }
        )
    return pd.DataFrame(rows)


def compute_kpis(raw: RawResults) -> ResultsBundle:
    df = patients_to_frame(raw.patients)
    audit = pd.DataFrame(raw.audit)
    scenario = raw.scenario

    kpis: dict[str, float] = {"attendances": 0.0, "attendances_per_day": 0.0}
    if df.empty:
        return ResultsBundle(scenario.meta.get("name", "unnamed"), raw.rep, df, audit, kpis)

    completed = df[(~df["in_warmup"]) & df["t_departure"].notna()].copy()
    kpis["attendances"] = float(len(completed))
    kpis["attendances_per_day"] = float(
        len(completed) / max(scenario.run.run_length_days - scenario.run.warm_up_days, 1)
    )

    if completed.empty:
        return ResultsBundle(scenario.meta.get("name", "unnamed"), raw.rep, df, audit, kpis)

    is_type1 = completed["stream"].isin([s.value for s in TYPE1_STREAMS])
    is_utc = completed["stream"] == Stream.UTC.value
    lwbs = completed["disposal"] == Disposal.LWBS.value
    within4 = completed["time_in_department"] <= FOUR_HOURS
    over12 = completed["time_in_department"] > TWELVE_HOURS
    admitted = completed["admitted"]

    kpis["four_hour_all"] = float(within4.mean())
    kpis["four_hour_type1"] = float(within4[is_type1].mean()) if is_type1.any() else math.nan
    kpis["four_hour_utc"] = float(within4[is_utc].mean()) if is_utc.any() else math.nan
    # admitted/discharged split follows the ECDS publication: Type 1 & 2 only
    t1_adm = is_type1 & admitted
    t1_dis = is_type1 & ~admitted & ~lwbs
    kpis["four_hour_admitted"] = float(within4[t1_adm].mean()) if t1_adm.any() else math.nan
    kpis["four_hour_discharged"] = float(within4[t1_dis].mean()) if t1_dis.any() else math.nan
    kpis["twelve_hour_from_arrival"] = float(over12.mean())
    kpis["twelve_hour_type1"] = float(over12[is_type1].mean()) if is_type1.any() else math.nan

    boarding = completed["boarding_time"].dropna()
    kpis["twelve_hour_dta"] = float((boarding > TWELVE_HOURS).sum())
    kpis["four_hour_dta"] = float((boarding > FOUR_HOURS).sum())
    kpis["mean_boarding_min"] = float(boarding.mean()) if len(boarding) else 0.0
    kpis["median_boarding_min"] = float(boarding.median()) if len(boarding) else 0.0

    kpis["admission_rate_type1"] = float(admitted[is_type1].mean()) if is_type1.any() else math.nan
    kpis["lwbs"] = float(lwbs.mean())
    kpis["elective_cancellations"] = float(raw.elective_cancellations)
    kpis["corridor_care_hours"] = float(raw.corridor_minutes / 60.0)

    kpis["median_time_in_dept_admitted"] = _q(completed, t1_adm, "time_in_department", 0.5)
    kpis["median_time_in_dept_discharged"] = _q(completed, t1_dis, "time_in_department", 0.5)
    kpis["p90_time_in_dept"] = float(completed["time_in_department"].quantile(0.9))
    kpis["median_time_to_triage"] = float(completed["time_to_triage"].dropna().median())
    kpis["median_time_to_first_clinician"] = float(
        completed["time_to_first_clinician"].dropna().median()
    )

    # MTS: share of each category seen within its target time
    for cat in TriageCat:
        sel = completed["cat"] == cat.name
        if sel.any():
            ttc = completed.loc[sel, "time_to_first_clinician"].dropna()
            target = MTS_TARGET_MIN[cat] if MTS_TARGET_MIN[cat] > 0 else 5.0  # RED: within 5 min
            kpis[f"mts_within_target_{cat.name.lower()}"] = (
                float((ttc <= target).mean()) if len(ttc) else math.nan
            )

    # ambulance handover
    handover = completed["handover_delay"].dropna()
    if len(handover):
        kpis["handover_median_min"] = float(handover.median())
        kpis["handover_over_15"] = float((handover > 15).mean())
        kpis["handover_over_30"] = float((handover > 30).mean())
        kpis["handover_over_60"] = float((handover > 60).mean())
        kpis["handover_crew_hours_lost"] = float(handover.clip(lower=15).sub(15).sum() / 60.0)

    # RCEM harm estimate: patients waiting 8-12h+ from arrival before admission
    arr_to_bed = (completed.loc[admitted, "t_bed"] - completed.loc[admitted, "t_arrival"]).dropna()
    n_8_to_12 = int(((arr_to_bed >= 480) & (arr_to_bed < 720)).sum())
    kpis["excess_deaths_rcem_estimate"] = n_8_to_12 / 72.0

    # audit-derived KPIs (post-warm-up)
    if not audit.empty:
        post = audit[~audit["in_warmup"]]
        if not post.empty:
            n_beds = scenario.beds.n_beds if scenario.beds.enabled else 0
            if n_beds:
                # NHS sitrep occupancy is the 08:00 morning census (the daily peak),
                # not a 24-hour average — measure the same way for comparability
                hour = (post["time"] % 1440.0) // 60.0
                morning = post[hour == 8]
                src = morning if len(morning) else post
                kpis["bed_occupancy"] = float(src["beds_occupied"].mean() / n_beds)
                kpis["bed_occupancy_24h"] = float(post["beds_occupied"].mean() / n_beds)
            majors_spaces = scenario.streams["majors"].spaces
            if majors_spaces:
                kpis["boarders_share_of_majors"] = float(
                    (post["n_boarding"] / majors_spaces).mean()
                )
            kpis["mean_ambulances_waiting"] = float(post["ambulances_waiting"].mean())
            kpis["mean_n_in_dept"] = float(post["n_in_dept"].mean())
            if "corridor_in_use" in post:
                kpis["mean_corridor_occupied"] = float(post["corridor_in_use"].mean())
            for name in scenario.staff_pools:
                busy, duty = post[f"staff_{name}_busy"], post[f"staff_{name}_on_duty"]
                on_duty_time = duty[duty > 0]
                if len(on_duty_time):
                    kpis[f"util_{name}"] = float(
                        busy[duty > 0].sum() / on_duty_time.sum()
                    )
            for stream_name, sp in scenario.streams.items():
                if sp.spaces:
                    kpis[f"occ_{stream_name}"] = float(
                        post[f"{stream_name}_in_use"].mean() / sp.spaces
                    )

    return ResultsBundle(
        scenario_name=scenario.meta.get("name", "unnamed"),
        rep=raw.rep,
        patients=df,
        audit=audit,
        kpis=kpis,
        extras={"n_in_flight": int(df["t_departure"].isna().sum())},
    )


def event_log_frame(raw: RawResults) -> pd.DataFrame:
    return pd.DataFrame(raw.event_log)


def _q(df: pd.DataFrame, mask: pd.Series, col: str, q: float) -> float:
    vals = df.loc[mask, col].dropna()
    return float(vals.quantile(q)) if len(vals) else math.nan


def summarise_replications(bundles: list[ResultsBundle]) -> pd.DataFrame:
    """Mean, sd and t-based 95% CI across replications for every KPI."""
    from scipy import stats  # dev extra; only needed for summaries

    rows = pd.DataFrame([b.kpis for b in bundles])
    out = []
    for col in rows.columns:
        x = rows[col].dropna().to_numpy(dtype=float)
        n = len(x)
        mean = float(np.mean(x)) if n else math.nan
        sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
        if n > 1 and sd > 0:
            half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        else:
            half = 0.0
        out.append(
            {"kpi": col, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half, "n": n}
        )
    return pd.DataFrame(out).set_index("kpi")
