# hospital-ae-simulator

A discrete-event simulation of a **UK (English) Type 1 A&E department with a
co-located Urgent Treatment Centre**, built to identify bottlenecks and test demand,
capacity, staffing and process scenarios — including the one that dominates UK
emergency care performance: **inpatient bed exit block**.

Built with Python 3.11 + [SimPy](https://simpy.readthedocs.io/), following the
Monks/PenCHORD STARS pattern for reusable healthcare simulations, reported per
[STRESS-DES](docs/stress_des.md), and calibrated against NHS England FY2025-26
statistics ([validation](docs/validation.md), [model card](docs/model_card.md)).

## What it models

- **Arrivals**: non-stationary Poisson walk-ins and ambulances (hourly profile,
  Monday uplift, higher overnight ambulance share, day-to-day flu/weather
  volatility), plus NHS 111 booked UTC slots
- **Front door**: registration, Manchester Triage (5 categories with MTS target
  times), ambulance handover with crews held until offload, "fit to sit" streaming
- **Streams**: resus (preemptive for Cat-1), majors, minors, co-located UTC (Type 3),
  SDEC referral, hot-clinic diversion of would-be admissions
- **Resources**: physical spaces AND clinician pools (doctors/nurses/ENPs) on
  shift-varying rotas — capacity changes never interrupt an in-progress treatment
- **Exit block**: admitted patients hold their cubicle until a ward bed is granted;
  the bed pool runs in chronic scarcity with afternoon-weighted discharges, weekend
  slippage, turnaround friction, elective competition/cancellation and corridor
  escalation — reproducing the national admitted-vs-discharged 4-hour gap (~26% vs ~66%)
- **KPIs**: 4-hour and 12-hour performance (by stream and admitted/discharged),
  trolley waits, MTS target compliance, handover delays vs 15/30/60-min standards,
  LWBS, utilisation, boarding, corridor-care hours, elective cancellations, and the
  RCEM excess-death estimate

## Quickstart

```bash
# from the repo root (use a venv OUTSIDE any OneDrive-synced folder)
pip install -e ".[dev,app]"

# fast test suite (~20 s), then the calibration checks (~60 s)
pytest
pytest -m calibration

# CLI: baseline KPI table vs NHS actual bands
python scripts/run_baseline.py --reps 10

# interactive dashboard
streamlit run app/Home.py
```

The dashboard has five pages: **Home** (baseline vs NHS actuals), **Scenario
Builder** (sliders for demand/spaces/staffing/streaming/beds with tabbed
diagnostics), **Compare Scenarios** (paired comparison under common random numbers —
CIs on the *difference*), **Flow Animation** (vidigi patient-flow animation) and
**Model Card**.

## Scenarios from the command line

```python
from aesim.params import Scenario
from aesim.runner import multiple_replications, paired_diff

base = Scenario.from_yaml("config/baseline_dgh.yaml")
winter = Scenario.from_yaml("config/baseline_dgh.yaml",
                            overrides="config/scenarios/winter_surge.yaml")

a = multiple_replications(base, n_reps=20)
b = multiple_replications(winter, n_reps=20)
print(paired_diff(a, b).loc["four_hour_type1"])
```

Example scenario overrides in `config/scenarios/`: `winter_surge`, `extra_sdec`,
`ward_flow_fix`. Any parameter can also be overridden inline:
`base.with_overrides(**{"streams.majors.spaces": 32, "beds.n_beds": 350})`.

## Repository layout

```
src/aesim/        simulation package (importable without Streamlit)
app/              Streamlit dashboard
config/           baseline + example scenario YAMLs (single source of all defaults)
docs/             model card · STRESS-DES report · validation
scripts/          run_baseline.py · welch_warmup.py
tests/            63 fast tests + calibration acceptance (pytest -m calibration)
```

## Notes

- Windows users: keep your virtualenv outside OneDrive-synced folders (file locking
  interferes with pytest/pip/Streamlit hot-reload).
- The model is a *generic representative DGH* (~250 attendances/day). Recalibrate
  `config/baseline_dgh.yaml` to your trust's data before drawing local conclusions.
- The RCEM excess-death figure is a published estimate applied to model output, not a
  model prediction; see the model card for all limitations.
