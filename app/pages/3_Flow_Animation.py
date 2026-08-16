"""Animated patient flow (vidigi) for one short illustrative replication."""

import streamlit as st

from app_setup import ensure_package  # noqa: F401
import charts
from components import load_baseline, scenario_sidebar
from aesim.runner import single_run

st.set_page_config(page_title="Flow Animation", page_icon="🎬", layout="wide")
st.title("🎬 Patient flow animation")
st.caption(
    "One short single replication for illustration — statistics come from the multi-rep "
    "pages. Icons are patients; boxes are physical areas; the boarding queue at the bottom "
    "is exit block made visible."
)

baseline = load_baseline()
scenario, _ = scenario_sidebar(baseline)

col1, col2, col3 = st.columns(3)
days = col1.slider("Days to animate", 1, 4, 2)
step = col2.slider("Snapshot every (min)", 10, 60, 15, 5)
seed = col3.number_input("Seed", 0, 999, 7)

if st.button("Generate animation", type="primary"):
    anim_scenario = scenario.with_overrides(
        **{
            "run.run_length_days": days + 1,
            "run.warm_up_days": 0,
            "run.random_seed": int(seed),
        }
    )
    with st.status("Simulating and rendering...", expanded=False):
        bundle = single_run(anim_scenario, rep=0, keep_event_log=True)
        try:
            from aesim.animation import build_animation

            fig = build_animation(
                bundle.event_log, anim_scenario, every_x_minutes=int(step), limit_days=days
            )
            st.session_state["anim_fig"] = ("vidigi", fig)
        except RuntimeError as exc:
            st.session_state["anim_fig"] = ("fallback", bundle)
            st.session_state["anim_error"] = str(exc)

if "anim_fig" in st.session_state:
    kind, obj = st.session_state["anim_fig"]
    if kind == "vidigi":
        st.plotly_chart(obj, use_container_width=True)
    else:
        st.warning(
            "Animation unavailable ("
            + st.session_state.get("anim_error", "unknown error")
            + ") — showing static queue time series instead."
        )
        st.plotly_chart(charts.queue_timeseries(obj.audit), use_container_width=True)
