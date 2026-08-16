"""Home page: what the model is, and headline baseline KPIs vs NHS actuals."""

import streamlit as st

from app_setup import ensure_package  # noqa: F401  (sys.path shim, must be first)
from components import kpi_cards, load_baseline, run_scenario

st.set_page_config(page_title="UK A&E Simulator", page_icon="🏥", layout="wide")

st.title("🏥 UK A&E Department Simulator")
st.markdown(
    """
A discrete-event simulation of a **typical English Type 1 emergency department with a
co-located Urgent Treatment Centre** (~250 attendances/day), built to find bottlenecks
and test capacity/staffing scenarios.

**Pathways modelled:** ambulance handover (with "fit to sit" streaming) · registration &
Manchester Triage · resus (preemptive) · majors · minors · UTC (Type 3, incl. NHS 111
booked slots) · SDEC referral · hot-clinic diversion · **inpatient bed exit block** with
boarding, corridor escalation, afternoon-weighted discharges and elective competition.

Parameters follow NHS England statistics, RCEM standards and the academic ED-simulation
literature — every value is overridable in the **Scenario Builder**. The baseline is
calibrated so the KPI cards below sit inside the national actual bands (FY2025-26).
"""
)

baseline = load_baseline()

col1, col2 = st.columns([1, 3])
with col1:
    reps = st.number_input("Replications", 5, 30, 10, help="More = tighter CIs, slower")
    run = st.button("Run baseline", type="primary", use_container_width=True)

if run or "baseline_runs" in st.session_state:
    if run or st.session_state.get("baseline_reps") != reps:
        st.session_state["baseline_runs"] = run_scenario(baseline, int(reps))
        st.session_state["baseline_reps"] = reps
    results = st.session_state["baseline_runs"]
    st.subheader("Baseline vs NHS England actuals")
    kpi_cards(results.summary, baseline.targets)
    st.caption(
        "Cards show the mean over replications; hover for 95% CIs and the NHS actual band. "
        "The admitted vs discharged 4-hour gap **is** exit block — the model's central mechanism."
    )
    with st.expander("Full KPI table"):
        st.dataframe(results.summary, use_container_width=True)
else:
    st.info("Press **Run baseline** to simulate 42 days and compare against NHS actuals.")

st.divider()
st.markdown(
    """
##### Pages
- **Scenario Builder** — change demand, spaces, staffing, streaming and beds; see the impact
- **Compare Scenarios** — paired comparison against baseline under common random numbers
- **Flow Animation** — watch patients move through the department
- **Model Card** — assumptions, data sources, validation and limitations
"""
)
