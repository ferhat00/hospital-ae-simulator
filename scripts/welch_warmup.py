"""Welch's method warm-up check: plot smoothed hourly series (bed occupancy,
number in department) averaged over replications, and eyeball where the
transient dies out. The bed pool has the slowest transient, so the warm-up is
sized on it.

Usage: python scripts/welch_warmup.py [--reps 10] [--days 42] [--window 24]
Writes welch_warmup.png next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aesim.model import AEModel
from aesim.params import Scenario

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--days", type=int, default=42)
    ap.add_argument("--window", type=int, default=24, help="moving-average window (hours)")
    args = ap.parse_args()

    scenario = Scenario.from_yaml(REPO / "config" / "baseline_dgh.yaml").with_overrides(
        **{"run.run_length_days": args.days, "run.warm_up_days": 0, "run.audit_interval_min": 60}
    )

    series: dict[str, list[np.ndarray]] = {"beds_occupied": [], "n_in_dept": []}
    for rep in range(args.reps):
        raw = AEModel(scenario, rep=rep).run()
        audit = pd.DataFrame(raw.audit)
        for key in series:
            series[key].append(audit[key].to_numpy())
        print(f"  rep {rep + 1}/{args.reps}", end="\r", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (key, arrs) in zip(axes, series.items(), strict=True):
        n = min(len(a) for a in arrs)
        mean_series = np.mean([a[:n] for a in arrs], axis=0)
        smooth = pd.Series(mean_series).rolling(args.window, center=True).mean()
        hours = np.arange(n)
        ax.plot(hours / 24.0, mean_series, alpha=0.35, label="mean over reps")
        ax.plot(hours / 24.0, smooth, lw=2, label=f"{args.window}h moving average")
        ax.axvline(14, color="red", ls="--", label="warm-up = 14 days")
        ax.set_ylabel(key)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("day")
    fig.suptitle("Welch warm-up check — transient must be flat before the red line")
    out = Path(__file__).with_name("welch_warmup.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
