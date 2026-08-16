# STRESS-DES Report

Reporting per Monks, Currie, Onggo, Robinson, Kunc & Taylor (2019), *Strengthening the
reporting of empirical simulation studies: Introducing the STRESS guidelines*,
Journal of Simulation 13(1). 20 items.

## 1. Objectives

**1.1 Purpose of the model.** Identify bottlenecks and test demand/capacity/staffing/
process scenarios in a typical English Type 1 A&E department with a co-located UTC,
with the inpatient bed pool (exit block) modelled explicitly. Generic representative
DGH calibrated to national FY2025-26 statistics; not a site-specific twin.

**1.2 Model outputs.** Per patient: timestamps at each pathway step (arrival, offload,
registration, triage start/end, space granted, first clinician, decision-to-admit, bed
granted, departure). KPIs computed over patients arriving after warm-up and departing
before run end (LWBS count as attendances): 4-hour performance (all / Type 1 / UTC /
admitted / discharged — the last two over Type 1 only, per ECDS), 12h-from-arrival
share, 4h+/12h+ DTA (trolley-wait) counts, admission rate, LWBS rate, time-to-triage
and time-to-first-clinician medians, MTS within-target shares by category, handover
delay distribution vs 15/30/60-min standards and crew-hours lost, boarding
mean/median, boarders' share of majors cubicles, bed occupancy at the 08:00 census,
staff utilisation (busy/on-duty from 15-min audit snapshots), corridor-care hours,
elective cancellations, and the RCEM excess-death estimate (patients waiting 8-12h
arrival-to-bed ÷ 72 — a published estimate applied to output, not a prediction).
Means with t-based 95% CIs over replications; paired-t CIs for scenario differences.

**1.3 Experimentation aims.** Scenario analysis via the Streamlit app or
`runner.scenario_analysis`: demand growth/volatility, spaces, rotas, UTC/SDEC
streaming, hot-clinic diversion, bed count/LoS/turnaround/discharge timing, corridor
and ambulance-offload policies. Baseline vs modified comparisons run under common
random numbers.

## 2. Logic

**2.1 Base model overview.**

```
Walk-in ─ registration ─ triage(room+nurse) ─┐
Ambulance ─ stretcher assess(nurse) ─ [fit-to-sit → minors] else wait space+handover ─┤
Booked(111) ──────────────────────────────────┤
        streaming: RESUS | MAJORS | MINORS | UTC | SDEC
        stream: space → clinician(assess) → nursing → diagnostics(delay) → clinician(treat)
        disposition: discharge | hot clinic | LWBS | admit
        admit: specialty review → DTA → request bed WHILE HOLDING space
               [corridor decant if others queue] → bed granted → depart ED → ward stay
        ward: LoS → discharge-hour snap (afternoon-weighted, weekend slippage) →
              turnaround → bed free.  Electives compete for beds; cancel at 18:00.
```

**2.2 Base model logic.** See `src/aesim/pathways.py` docstring (lock order:
space → clinician → bed; beds never require ED resources).

**2.3 Scenario logic.** Scenarios are parameter overrides only; no structural changes.

**2.4 Algorithms.** NSPP thinning with independent interarrival/accept-reject streams
(`arrivals.py`); ghost-slot shift scheduling (`resources.py`); catch-and-resume
preemption (`pathways.py:_clinician_service`); discharge-hour snapping
(`distributions.py:DischargeHourSampler`); length-biased residual sampling for bed
prefill (exact for lognormal).

**2.5 Entities.** Patients (attributes: arrival mode, MTS category, stream, paeds
flag, SDEC eligibility, admission flag, timestamps); synthetic ward occupants
(prefill); elective admissions.

**2.6 Activities.** Registration, triage, stretcher assessment, clinical handover,
assessment, nursing care, diagnostics (sampled delay), treatment, specialty review,
SDEC workup, ward stay, bed turnaround. Distributions and parameters: `config/baseline_dgh.yaml`.

**2.7 Resources.** Triage rooms; resus bays; majors cubicles; minors rooms; UTC rooms;
SDEC chairs; corridor spaces; optional offload spaces; staff pools (ED doctors —
preemptive for resus, ED nurses, triage nurses, UTC ENPs, SDEC clinicians) with
shift-varying capacity × 0.89 availability; inpatient beds.

**2.8 Queues.** Priority queues keyed on MTS category (lower = more urgent); FIFO
within priority. Reneging: Green/Blue abandon space queues after patience (180/120
min). Preemption: resus doctor requests may interrupt lower-priority consultations;
interrupted service resumes its remainder. Bed queue: ED (priority 0) over electives
(priority 1); electives cancel at 18:00 if bedless.

**2.9 Entry/exit points.** Entry: NSPP walk-ins and ambulances (hourly × day-of-week ×
lognormal day-effect), scheduled UTC bookings with no-shows. Exit: discharge, LWBS,
hot clinic, admission (departure = bed granted; ward stay continues to bed release).
Boundary: single site; no Type 2 activity; no regional ambulance fleet feedback.

## 3. Data

**3.1 Data sources.** NHS England A&E monthly sitreps and ECDS supplementary analyses
(FY2025-26); NHS England Model ED / UTC / SDEC specifications; RCEM clinical standards
(2024-25); MTS validity literature; published UK ED simulation studies (Squires 2023;
Bowers 2011; Mohiuddin 2017 review). No patient-level data used.

**3.2 Pre-processing.** None (aggregate published statistics only).

**3.3 Input parameters.** All in `config/baseline_dgh.yaml` with inline source
comments; full table rendered on the app's Model Card page.

**3.4 Assumptions.** Diagnostics as delays not resources; paediatrics via minors/majors
with a flag; stylised elective logic; corridor decant triggered by queue-for-space > 1;
ambulance Green/Blue offload immediately ("fit to sit"); admission probabilities by
MTS category tuned to the national admission rate. Full list: `docs/model_card.md`.

## 4. Experimentation

**4.1 Initialisation.** Non-terminating system. Warm-up 14 days (sized on the bed
pool, the slowest transient; verified by Welch's method — `scripts/welch_warmup.py`,
see `scripts/welch_warmup.png`). Warm-up implemented as a flag on patients/audit rows,
never truncation. Beds pre-filled to 93% with residual (length-biased) stays.

**4.2 Run length.** 42 days = 14-day warm-up + 4 whole result weeks (whole weeks avoid
day-of-week bias).

**4.3 Estimation approach.** Default 30 independent replications (10 for interactive
app runs); replication *i* of every scenario shares seeds (CRN) via
`SeedSequence(master).spawn(rep)` then one child per named sampling purpose in fixed
registration order. Means ± t-based 95% CIs; paired-t CIs for scenario deltas.

## 5. Implementation

**5.1 Software.** Python 3.11, SimPy 4.1.2 (DES engine), NumPy Generator (PCG64)
random sampling, pandas results, sim-tools 1.3.0 pinned (available for replication
sizing), vidigi 1.3.1 pinned (animation), Streamlit UI. Windows 11 development;
pure-Python, cross-platform.

**5.2 Random sampling.** One `numpy.random.Generator` per sampling purpose, seeded
from a per-replication `SeedSequence`; ~45 named streams (see `model.SEED_PURPOSES`).

**5.3 Model execution.** Sequential replications (a 42-day rep ≈ 3-4 s); CLI scripts
in `scripts/`; app in `app/`. No multiprocessing inside Streamlit (Windows spawn).

**5.4 System specification.** Developed on Windows 11, Python 3.11.9; any modern
machine suffices (single-core, <1 GB RAM).

**5.5 Code access.** Repository `hospital-ae-simulator` (MIT licence): simulation
package `src/aesim/`, tests `tests/` (63 fast + 2 calibration), configuration
`config/`, app `app/`.
