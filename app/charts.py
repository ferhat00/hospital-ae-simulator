"""Pure plotly figure builders: DataFrame in, Figure out. No Streamlit imports,
so everything here is unit-testable and reusable from notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FOUR_H = 240
TWELVE_H = 720
ACCENT = "#1f77b4"
WARN = "#d62728"


def los_histogram(patients: pd.DataFrame) -> go.Figure:
    """Time-in-department distribution, admitted vs discharged, with the 4h and
    12h standards marked. The pre-4h pile-up and the admitted tail are the two
    signatures to look for."""
    done = patients.dropna(subset=["time_in_department"]).copy()
    done = done[done["disposal"] != "lwbs"]
    done["group"] = np.where(done["admitted"], "Admitted", "Discharged")
    fig = px.histogram(
        done,
        x="time_in_department",
        color="group",
        barmode="overlay",
        nbins=80,
        opacity=0.65,
        color_discrete_map={"Admitted": WARN, "Discharged": ACCENT},
        labels={"time_in_department": "time in department (minutes)"},
    )
    fig.add_vline(x=FOUR_H, line_dash="dash", line_color="black",
                  annotation_text="4h standard", annotation_position="top")
    fig.add_vline(x=TWELVE_H, line_dash="dot", line_color=WARN,
                  annotation_text="12h", annotation_position="top")
    fig.update_layout(legend_title=None, margin=dict(t=40))
    fig.update_xaxes(range=[0, done["time_in_department"].quantile(0.99)])
    return fig


def occupancy_heatmap(audit: pd.DataFrame, column: str, capacity: int | None, title: str) -> go.Figure:
    post = audit[~audit["in_warmup"]].copy()
    post["hour"] = (post["time"] % 1440) // 60
    post["day"] = ((post["time"] // 1440) % 7).astype(int)
    pivot = post.pivot_table(index="day", columns="hour", values=column, aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pivot)]
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels={"x": "hour of day", "y": "", "color": column},
        title=title,
    )
    if capacity:
        fig.update_coloraxes(cmax=capacity, cmin=0)
    fig.update_layout(margin=dict(t=50))
    return fig


def queue_timeseries(audit: pd.DataFrame) -> go.Figure:
    post = audit[~audit["in_warmup"]].copy()
    post["day"] = post["time"] / 1440
    fig = go.Figure()
    series = {
        "majors_queue": "Waiting for majors cubicle",
        "bed_dta_queue": "DTA-to-bed queue (boarders)",
        "ambulances_waiting": "Ambulances at hospital",
        "corridor_in_use": "Patients in corridor care",
    }
    for col, label in series.items():
        if col in post:
            fig.add_trace(go.Scatter(x=post["day"], y=post[col], name=label, mode="lines"))
    fig.update_layout(
        xaxis_title="day",
        yaxis_title="count",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30),
    )
    return fig


def utilisation_bars(kpis: dict[str, float]) -> go.Figure:
    rows = []
    for key, val in sorted(kpis.items()):
        if key.startswith("util_"):
            rows.append({"resource": key.removeprefix("util_"), "value": val, "kind": "staff"})
        elif key.startswith("occ_"):
            rows.append({"resource": key.removeprefix("occ_"), "value": val, "kind": "spaces"})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="value",
        y="resource",
        color="kind",
        orientation="h",
        labels={"value": "utilisation / occupancy"},
    )
    fig.add_vline(x=0.85, line_dash="dash", line_color=WARN,
                  annotation_text="RCEM 85%", annotation_position="top")
    fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(margin=dict(t=30), legend_title=None)
    return fig


def sankey_flow(patients: pd.DataFrame) -> go.Figure:
    done = patients.dropna(subset=["disposal"]).copy()
    mode_lab = {"walk_in": "Walk-in", "ambulance": "Ambulance", "booked_utc": "Booked (111)"}
    stream_lab = {
        "resus": "Resus", "majors": "Majors", "minors": "Minors", "utc": "UTC", "sdec": "SDEC",
    }
    disp_lab = {
        "discharged": "Discharged",
        "admitted": "Admitted",
        "sdec_discharged": "Discharged (SDEC)",
        "sdec_admitted": "Admitted (SDEC)",
        "hot_clinic": "Hot clinic",
        "lwbs": "Left without being seen",
    }
    done["mode_l"] = done["mode"].map(mode_lab)
    done["stream_l"] = done["stream"].map(stream_lab).fillna("(no stream)")
    done["disp_l"] = done["disposal"].map(disp_lab)

    nodes = (
        list(mode_lab.values())
        + sorted(done["stream_l"].unique())
        + sorted(done["disp_l"].unique())
    )
    idx = {n: i for i, n in enumerate(nodes)}
    links: dict[tuple[int, int], int] = {}
    for (a, b), n in done.groupby(["mode_l", "stream_l"]).size().items():
        links[(idx[a], idx[b])] = links.get((idx[a], idx[b]), 0) + int(n)
    for (a, b), n in done.groupby(["stream_l", "disp_l"]).size().items():
        links[(idx[a], idx[b])] = links.get((idx[a], idx[b]), 0) + int(n)

    fig = go.Figure(
        go.Sankey(
            node=dict(label=nodes, pad=18, thickness=14),
            link=dict(
                source=[s for s, _ in links],
                target=[t for _, t in links],
                value=list(links.values()),
            ),
        )
    )
    fig.update_layout(margin=dict(t=20, b=20))
    return fig


def handover_ecdf(patients: pd.DataFrame) -> go.Figure:
    hand = patients["handover_delay"].dropna()
    fig = px.ecdf(hand, labels={"value": "handover delay (minutes)"})
    for x, label in ((15, "15 min"), (30, "30 min"), (60, "60 min")):
        fig.add_vline(x=x, line_dash="dash", line_color="grey",
                      annotation_text=label, annotation_position="top")
    fig.update_layout(showlegend=False, yaxis_title="share of ambulance arrivals",
                      margin=dict(t=30))
    return fig


def boarding_by_dta_hour(patients: pd.DataFrame) -> go.Figure:
    done = patients.dropna(subset=["boarding_time", "t_dta"]).copy()
    done["dta_hour"] = (done["t_dta"] % 1440) // 60
    agg = done.groupby("dta_hour")["boarding_time"].mean().div(60.0).reset_index()
    fig = px.bar(
        agg,
        x="dta_hour",
        y="boarding_time",
        labels={"dta_hour": "hour of decision-to-admit", "boarding_time": "mean boarding (hours)"},
    )
    fig.update_layout(margin=dict(t=30))
    return fig


def mts_target_bars(kpis: dict[str, float]) -> go.Figure:
    labels = {
        "mts_within_target_red": "Red (immediate)",
        "mts_within_target_orange": "Orange (10 min)",
        "mts_within_target_yellow": "Yellow (60 min)",
        "mts_within_target_green": "Green (120 min)",
        "mts_within_target_blue": "Blue (240 min)",
    }
    rows = [
        {"category": lab, "within_target": kpis[k]}
        for k, lab in labels.items()
        if k in kpis and not pd.isna(kpis[k])
    ]
    fig = px.bar(
        pd.DataFrame(rows),
        x="category",
        y="within_target",
        labels={"within_target": "seen within MTS target"},
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(margin=dict(t=30))
    return fig
