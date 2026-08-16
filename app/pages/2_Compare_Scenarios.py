"""Paired scenario comparison under common random numbers.

Replication i of both scenarios shares identical arrival instants, triage
categories and service draws, so the per-replication differences isolate the
effect of the lever you changed — paired-t CIs are far tighter than comparing
independent runs.
"""

import pandas as pd
import streamlit as st

from app_setup import ensure_package  # noqa: F401
import charts
from components import load_baseline, run_scenario, scenario_sidebar
from aesim.runner import paired_diff

st.set_page_config(page_title="Compare Scenarios", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare: baseline vs modified scenario")

baseline = load_baseline()
scenario, n_reps = scenario_sidebar(baseline)

if scenario.content_hash() == baseline.content_hash():
    st.info("Change at least one lever in the sidebar — the comparison runs your modified "
            "scenario against the untouched baseline with the same random numbers.")
    st.stop()

if st.button("Run paired comparison", type="primary"):
    with st.status("Running baseline...", expanded=False):
        base_runs = run_scenario(baseline, n_reps)
    with st.status("Running modified scenario...", expanded=False):
        mod_runs = run_scenario(scenario, n_reps)
    st.session_state["compare"] = (base_runs, mod_runs)

if "compare" not in st.session_state:
    st.stop()

base_runs, mod_runs = st.session_state["compare"]
diff = paired_diff(base_runs, mod_runs)

HEADLINE = [
    "four_hour_type1", "four_hour_admitted", "four_hour_discharged",
    "twelve_hour_from_arrival", "lwbs", "median_boarding_min",
    "handover_over_30", "mean_ambulances_waiting", "corridor_care_hours",
    "elective_cancellations", "excess_deaths_rcem_estimate",
]
show = diff.loc[[k for k in HEADLINE if k in diff.index]].copy()

st.subheader("Headline differences (modified − baseline)")
pretty = show.reset_index()
pretty["95% CI"] = pretty.apply(lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1)
pretty["significant"] = pretty["significant"].map({True: "✅", False: "—"})
st.dataframe(
    pretty[["kpi", "baseline_mean", "scenario_mean", "diff_mean", "95% CI", "significant"]],
    use_container_width=True,
    hide_index=True,
)
st.caption("✅ = the paired 95% CI excludes zero: the change is distinguishable from noise "
           f"with {len(base_runs.bundles)} common-random-number replications.")

with st.expander("All KPIs"):
    st.dataframe(diff, use_container_width=True)

st.subheader("Time-in-department distributions")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Baseline**")
    st.plotly_chart(charts.los_histogram(base_runs.bundles[0].patients),
                    use_container_width=True)
with col2:
    st.markdown("**Modified**")
    st.plotly_chart(charts.los_histogram(mod_runs.bundles[0].patients),
                    use_container_width=True)
