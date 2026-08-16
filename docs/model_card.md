# Model Card — UK A&E Department Simulator

## Purpose

Discrete-event simulation of a **typical English Type 1 emergency department with a
co-located Urgent Treatment Centre** (~250 combined attendances/day), built to identify
bottlenecks and test demand, capacity, staffing and process scenarios. It is a *generic
representative DGH*, calibrated to national statistics — not a digital twin of any
specific trust.

## Scope and structure

**Streams** (following NHS England's *Model Emergency Department*, Feb 2026):
ambulance handover → triage/streaming → Resus / Majors / Minors ("fit to sit") /
co-located UTC (Type 3, incl. NHS 111 booked slots) / SDEC referral, with hot-clinic
diversion of would-be admissions and an **inpatient bed pool as the exit-block
constraint** (boarding in cubicles, corridor escalation, afternoon-weighted discharges,
weekend discharge slippage, bed turnaround friction, elective competition and
cancellation).

**Resources:** physical spaces (resus bays, majors cubicles, minors/UTC rooms, SDEC
chairs, triage rooms, corridor/offload spaces) *and* clinician pools (doctors, nurses,
ENPs, SDEC clinicians) with shift-varying capacity (ghost-slot rostering — capacity
drops never interrupt an in-progress treatment) and an 89% availability factor
(Squires et al. 2023).

**Arrivals:** non-stationary Poisson (thinning), hourly profile with a 10:00–12:00 peak,
secondary evening peak and 04:00–07:00 trough; Monday uplift; hour-varying ambulance
share (higher overnight); lognormal day-to-day volatility (cv 0.15) representing flu
waves/weather; scheduled UTC bookings (~3.5%) with no-shows.

## Key mechanisms

1. **Exit block**: an admitted patient holds their ED space until an inpatient bed is
   granted; the bed pool runs in **chronic scarcity** (demand × LoS ≥ supply), the
   observed UK regime, with elective cancellations and corridor boarding as relief valves.
2. **Priority queueing** by Manchester Triage category; resus doctors are preemptible
   by RED arrivals (interrupted consultations resume their remaining time).
3. **Reneging**: Green/Blue patients abandon (LWBS) after a patience threshold.
4. **Discharge timing**: ward discharges peak 13:00–16:00 and stop by 20:00, so evening
   decision-to-admit patients board overnight — the mechanism behind 12h+ waits.

## Calibration (FY2025-26 England actuals)

| KPI | Model (42-day, 10 reps) | NHS band | Source |
|---|---|---|---|
| 4h % — Type 1 | 0.55 | 0.54–0.60 | ECDS-coherent (see note) |
| 4h % — admitted (T1) | 0.26 | 0.24–0.32 | ECDS |
| 4h % — discharged (T1) | 0.66 | 0.62–0.68 | ECDS |
| 4h % — UTC | 0.98 | 0.95–0.995 | NHS England sitrep |
| 12h+ from arrival | 0.06 | 0.06–0.13 | ECDS |
| Admission rate (Type 1) | 0.26 | 0.26–0.29 | NHS England sitrep |
| LWBS | 0.025 | 0.02–0.05 | literature |
| Bed occupancy (08:00) | 0.985 | 0.94–0.995 | UEC daily sitrep (winter) |

**Data-consistency note:** the monthly sitrep's Type 1 4-hour figure (60–64% in good
months) is mathematically incompatible with ECDS's admitted (26.7%) / discharged (65.1%)
splits at a 26–29% admission share (these blend to ~55–58%). The two NHS collections
differ in coverage and month. This model follows the **ECDS-first reading** so the
admitted/discharged gap — the exit-block signature — is reproduced exactly.

**What was tuned vs held:** admission probabilities by triage category, doctor rota,
majors cubicles, bed count, elective demand, discharge-hour weights and demand
volatility were tuned; triage mix, MTS targets, service-time priors, UTC/SDEC
specifications and all standards were held at literature/guideline values.

## Face-validity signatures (verified by test)

- Admitted patients' 4-hour performance ~40 points worse than discharged — exit block.
- Boarders occupy >10% of majors cubicles (the RCEM crowding threshold).
- Overnight decision-to-admit patients board longer than afternoon ones (morning bed
  starvation from afternoon-weighted discharges).
- UTC essentially never breaches 4 hours (Type 3 reality, ~97%).

## Limitations

- Generic DGH; calibrate `config/baseline_dgh.yaml` to a specific trust's data before
  drawing site-specific conclusions.
- Diagnostics (bloods/X-ray/CT) are sampled delays, not capacity-constrained resources.
- Paediatrics is a reported flag routed through minors/majors, not a separate PED.
- No regional ambulance fleet feedback (handover delay is measured, not fed back into
  response times); no Type 2 (single-specialty) activity; no mental-health pathway.
- The RCEM excess-death figure (1 additional death per 72 patients waiting 8–12h before
  admission) is a **published estimate applied to model output**, not a model prediction.
- Elective cancellation logic is stylised (cancel if no bed by 18:00).

## Key sources

- NHS England, *A&E Attendances and Emergency Admissions* monthly statistics & ECDS
  supplementary analyses (FY2025-26)
- NHS England, *The Model Emergency Department* (Feb 2026); UTC principles & standards;
  SDEC service specification
- RCEM clinical standards: Initial Assessment (2025), ED Crowding (2025), Metrics in
  Emergency Medicine (2025)
- Manchester Triage System validity studies (category mix, admission gradients)
- Squires et al. 2023 (UK ED DES, staffing availability); Bowers et al. 2011 (4h target
  behaviour); Mohiuddin et al. 2017 (UK ED simulation systematic review)
- Architecture: Monks/PenCHORD STARS pattern (`stars-treat-sim`, `pydesrap_mms`),
  HSMA *Little Book of DES*; reporting per STRESS-DES (Monks et al. 2019)
