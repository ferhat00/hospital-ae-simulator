# Validation

## Approach

Three layers, per the UK ED simulation literature (Mohiuddin et al. 2017) and
STRESS-DES:

1. **White-box verification** — 63 fast unit/property tests: distribution mean/CV
   recovery, NSPP hourly-rate correctness and thinning-bound safety, event-log
   conservation (arrivals = departures + in-system; matched resource use/end pairs;
   monotone timestamps), degenerate cases (infinite resources ⇒ no queueing waits;
   zero arrivals ⇒ empty results; near-zero patience ⇒ all queued Green/Blue LWBS),
   an M/M/1 analytic check (ρ=0.8: mean queue wait within 15% of ρ/(1−ρ)·m and
   utilisation within 0.03 of ρ), ghost-slot roster behaviour (capacity tracks the
   schedule; services never interrupted at shift change; ghosts never preempt resus),
   boarding saturation without deadlock at n_beds=1, byte-identical reproducibility at
   equal seeds, and CRN synchronisation across capacity scenarios.
   Run: `pytest`.

2. **Black-box calibration** — baseline KPI means over 10 × 42-day replications must
   sit inside the NHS actual bands declared in `config/baseline_dgh.yaml` `targets:`.
   Run: `pytest -m calibration`.

3. **Face validity** — signatures asserted by test and visible in the app: the
   admitted-vs-discharged 4-hour gap (>25 points), boarders exceeding the RCEM 10%
   cubicle threshold, overnight DTAs boarding longer than afternoon DTAs (morning bed
   starvation), UTC ~97% performance, and Welch-verified warm-up
   (`scripts/welch_warmup.png`).

## Result (baseline, 10 replications × 42 days, seed 42)

| KPI | Model mean | 95% CI | NHS band | In band |
|---|---|---|---|---|
| 4h — Type 1 | 0.549 | [0.500, 0.598] | 0.54–0.60 | ✅ |
| 4h — admitted (T1) | 0.259 | [0.210, 0.308] | 0.24–0.32 | ✅ |
| 4h — discharged (T1) | 0.659 | [0.608, 0.709] | 0.62–0.68 | ✅ |
| 4h — UTC | 0.976 | [0.964, 0.987] | 0.95–0.995 | ✅ |
| 12h+ from arrival | 0.063 | [0.046, 0.079] | 0.06–0.13 | ✅ |
| Admission rate (T1) | 0.262 | [0.256, 0.269] | 0.26–0.29 | ✅ |
| LWBS | 0.025 | [0.017, 0.033] | 0.02–0.05 | ✅ |
| Bed occupancy (08:00) | 0.985 | [0.973, 0.998] | 0.94–0.995 | ✅ |

Supporting values: median time in department 5.5 h admitted vs 3.2 h discharged;
median time to triage 8 min; median time to first clinician 40 min; handover median
30 min with 50% > 30 min (winter-consistent); ~3.8 elective cancellations/day;
corridor care ~1.6 patients on average.

## What was tuned vs held

**Tuned (loose priors):** admission probability by MTS category (to the national
admission-rate band), ED doctor rota (day 11 / twilight 9 / night 4), majors cubicles
(28), bed count (313), elective demand (14/day), discharge-hour weights (peak
13:00–16:00, none after 20:00), demand volatility (cv 0.15), weekend discharge
slippage (0.35), bed turnaround (LN 90 min).

**Held at guideline/literature values:** MTS category mix and target times, triage/
registration/assessment/treatment service-time priors, UTC streaming shares and
specification, SDEC specification (open hours, transfer window, 20–30% conversion),
handover clinical time, staffing availability factor (0.89), inpatient LoS mean
(5 days), RCEM constants.

**Not tuned to targets:** the 4-hour and 12-hour performance figures themselves are
*outputs*; no parameter directly encodes them.

## Known tensions

- The monthly sitrep's Type 1 4-hour figure (60–64% in the best months of FY2025-26)
  cannot be reconciled arithmetically with ECDS's admitted/discharged splits at the
  sitrep admission rate; this model follows the ECDS-first reading (see the note in
  `config/baseline_dgh.yaml`).
- Handover >30 min (50%) sits above the best published months (~23%) and near winter
  peaks (~37–40%); reproducing both while boarding stays in band was not achievable
  with a single static configuration.
- The 12h tail (6.3%) sits at the summer end of the band; winter values (10–13%) are
  reachable with the `winter_surge` example scenario.
