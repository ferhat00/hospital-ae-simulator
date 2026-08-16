"""Scenario Builder: sidebar levers -> run -> tabbed diagnostics."""

import streamlit as st
import yaml

from app_setup import ensure_package  # noqa: F401
import charts
from components import kpi_cards, load_baseline, run_scenario, scenario_sidebar

st.set_page_config(page_title="Scenario Builder", page_icon="🛠️", layout="wide")
st.title("🛠️ Scenario Builder")

baseline = load_baseline()
scenario, n_reps = scenario_sidebar(baseline)

changed = scenario.content_hash() != baseline.content_hash()
st.caption(
    ("**Modified scenario** — differs from baseline." if changed else "Baseline settings.")
    + f" Scenario hash `{scenario.content_hash()}`."
)

if st.button("Run scenario", type="primary"):
    st.session_state["builder_results"] = run_scenario(scenario, n_reps)
    st.session_state["builder_scenario"] = scenario

results = st.session_state.get("builder_results")
if results is None:
    st.info("Set your levers in the sidebar, then press **Run scenario**.")
    st.stop()

ran_scenario = st.session_state["builder_scenario"]
if ran_scenario.content_hash() != scenario.content_hash():
    st.warning("Sidebar has changed since the last run — press **Run scenario** to refresh.")

summary = results.summary
first = results.bundles[0]

tabs = st.tabs(
    ["KPIs", "Time in department", "Occupancy", "Utilisation", "Queues",
     "Flow", "Triage", "Handover", "Harm"]
)

with tabs[0]:
    kpi_cards(summary, baseline.targets)
    st.dataframe(summary, use_container_width=True)

with tabs[1]:
    st.plotly_chart(charts.los_histogram(first.patients), use_container_width=True)
    st.caption(
        "One replication shown. Look for the admitted (red) long tail — patients waiting "
        "for beds — versus the discharged distribution."
    )

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            charts.occupancy_heatmap(
                first.audit, "majors_in_use", ran_scenario.streams["majors"].spaces,
                "Majors cubicles in use (hour x day)",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.occupancy_heatmap(
                first.audit, "n_boarding", None, "Boarders awaiting beds (hour x day)"
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            charts.occupancy_heatmap(
                first.audit, "beds_occupied", ran_scenario.beds.n_beds,
                "Inpatient beds occupied (hour x day)",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.occupancy_heatmap(
                first.audit, "utc_in_use", ran_scenario.streams["utc"].spaces,
                "UTC rooms in use (hour x day)",
            ),
            use_container_width=True,
        )

with tabs[3]:
    st.plotly_chart(charts.utilisation_bars(first.kpis), use_container_width=True)
    st.caption(
        "Staff utilisation is busy-time over on-duty time; spaces are mean occupancy. "
        "RCEM: sustained running above ~85% removes all surge headroom."
    )

with tabs[4]:
    st.plotly_chart(charts.queue_timeseries(first.audit), use_container_width=True)
    st.caption("One replication. Multi-day congestion waves come from demand volatility "
               "meeting a near-capacity bed pool.")

with tabs[5]:
    st.plotly_chart(charts.sankey_flow(first.patients), use_container_width=True)

with tabs[6]:
    st.plotly_chart(charts.mts_target_bars(first.kpis), use_container_width=True)
    st.caption("Share of each Manchester Triage category reaching a clinician inside its "
               "target (Red 0/immediate, Orange 10, Yellow 60, Green 120, Blue 240 min).")

with tabs[7]:
    st.plotly_chart(charts.handover_ecdf(first.patients), use_container_width=True)
    hand_hours = first.kpis.get("handover_crew_hours_lost", 0.0)
    st.metric("Ambulance crew-hours lost to handover (per run)", f"{hand_hours:,.0f} h")

with tabs[8]:
    st.plotly_chart(charts.boarding_by_dta_hour(first.patients), use_container_width=True)
    deaths = summary.loc["excess_deaths_rcem_estimate", "mean"] if "excess_deaths_rcem_estimate" in summary.index else 0.0
    corridor = summary.loc["corridor_care_hours", "mean"] if "corridor_care_hours" in summary.index else 0.0
    cancels = summary.loc["elective_cancellations", "mean"] if "elective_cancellations" in summary.index else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated excess deaths (per 4-week period)", f"{deaths:.1f}",
              help="RCEM/SMR published estimate: one additional death per 72 patients "
                   "waiting 8-12h from arrival before admission. A published estimate "
                   "applied to model output — not a model prediction.")
    c2.metric("Corridor care", f"{corridor:,.0f} h")
    c3.metric("Elective cancellations", f"{cancels:.0f}")

st.divider()
st.download_button(
    "Download scenario as YAML",
    data=yaml.safe_dump(ran_scenario.to_dict(), sort_keys=False),
    file_name="scenario.yaml",
    mime="text/yaml",
)
