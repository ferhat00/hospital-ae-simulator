"""CLI smoke run: python scripts/run_baseline.py [--reps N] [--days D]

Prints a KPI table for the baseline scenario with the NHS actual bands from the
config's `targets:` section alongside, so calibration quality is visible at a
glance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aesim.params import Scenario
from aesim.runner import multiple_replications

REPO = Path(__file__).resolve().parents[1]

HEADLINE = [
    "attendances_per_day",
    "four_hour_all",
    "four_hour_type1",
    "four_hour_utc",
    "four_hour_admitted",
    "four_hour_discharged",
    "twelve_hour_from_arrival",
    "admission_rate_type1",
    "lwbs",
    "bed_occupancy",
    "boarders_share_of_majors",
    "median_boarding_min",
    "median_time_in_dept_admitted",
    "median_time_in_dept_discharged",
    "median_time_to_triage",
    "median_time_to_first_clinician",
    "handover_median_min",
    "handover_over_30",
    "mean_ambulances_waiting",
    "excess_deaths_rcem_estimate",
    "elective_cancellations",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--days", type=int, default=None, help="override run length (days)")
    ap.add_argument("--config", type=Path, default=REPO / "config" / "baseline_dgh.yaml")
    args = ap.parse_args()

    scenario = Scenario.from_yaml(args.config)
    if args.days is not None:
        scenario = scenario.with_overrides(
            **{
                "run.run_length_days": args.days,
                "run.warm_up_days": min(scenario.run.warm_up_days, max(args.days - 1, 0)),
            }
        )

    t0 = time.perf_counter()
    reps = multiple_replications(scenario, n_reps=args.reps, progress=_progress)
    elapsed = time.perf_counter() - t0
    print(f"\n{args.reps} reps x {scenario.run.run_length_days} days in {elapsed:.1f}s\n")

    summary = reps.summary
    targets = scenario.targets
    print(f"{'KPI':38s} {'mean':>10s} {'95% CI':>21s} {'NHS band':>14s}")
    print("-" * 88)
    for kpi in HEADLINE:
        if kpi not in summary.index:
            continue
        row = summary.loc[kpi]
        band = ""
        for tname, (lo, hi) in targets.items():
            if tname == kpi or (tname + "_type1") == kpi or tname == kpi.replace("_type1", ""):
                band = f"[{lo:.2f}, {hi:.2f}]"
                break
        ci = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        print(f"{kpi:38s} {row['mean']:10.3f} {ci:>21s} {band:>14s}")

    for name in scenario.staff_pools:
        kpi = f"util_{name}"
        if kpi in summary.index:
            row = summary.loc[kpi]
            print(f"{kpi:38s} {row['mean']:10.3f}")


def _progress(i: int, n: int) -> None:
    print(f"  rep {i}/{n}", end="\r", flush=True)


if __name__ == "__main__":
    main()
