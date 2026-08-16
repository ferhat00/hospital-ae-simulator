"""vidigi adapter: turn the simulation event log into an animated patient-flow
figure. All vidigi contact is confined here so upstream API churn stays local;
`build_animation` raises RuntimeError with a readable message if vidigi is
unavailable or incompatible, and callers fall back to static charts.

The animation tells the *space-level* story (front door -> triage -> stream
space -> bed queue -> departure); clinician micro-moves and diagnostics are
filtered out for legibility.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aesim.params import Scenario

_KEEP_EVENTS = {
    "arrival",
    "depart",
    "wait_triage",
    "triage",
    "wait_handover_space",
    "wait_offload",
    "wait_space_resus",
    "space_resus",
    "wait_space_majors",
    "space_majors",
    "wait_space_minors",
    "space_minors",
    "wait_space_utc",
    "space_utc",
    "wait_space_sdec",
    "space_sdec",
    "wait_bed",
    "corridor",
}

_X_QUEUE, _X_USE = 30, 46


def _positions(scenario: Scenario) -> pd.DataFrame:
    rows = [
        {"event": "wait_triage", "x": _X_QUEUE, "y": 640, "label": "Queueing for triage"},
        {"event": "triage", "x": _X_USE, "y": 640, "label": "Triage", "resource": "triage_rooms"},
        {"event": "wait_handover_space", "x": _X_QUEUE, "y": 550, "label": "Ambulance handover queue"},
        {"event": "wait_space_resus", "x": _X_QUEUE + 60, "y": 550, "label": "Waiting: resus bay"},
        {"event": "space_resus", "x": _X_USE + 60, "y": 550, "label": "Resus", "resource": "resus"},
        {"event": "wait_space_majors", "x": _X_QUEUE + 60, "y": 460, "label": "Waiting: majors cubicle"},
        {"event": "space_majors", "x": _X_USE + 60, "y": 460, "label": "Majors", "resource": "majors"},
        {"event": "wait_space_minors", "x": _X_QUEUE + 60, "y": 370, "label": "Waiting: minors"},
        {"event": "space_minors", "x": _X_USE + 60, "y": 370, "label": "Minors", "resource": "minors"},
        {"event": "wait_space_utc", "x": _X_QUEUE + 60, "y": 280, "label": "Waiting: UTC"},
        {"event": "space_utc", "x": _X_USE + 60, "y": 280, "label": "UTC", "resource": "utc"},
        {"event": "wait_space_sdec", "x": _X_QUEUE + 60, "y": 190, "label": "Waiting: SDEC"},
        {"event": "space_sdec", "x": _X_USE + 60, "y": 190, "label": "SDEC", "resource": "sdec"},
        {"event": "corridor", "x": _X_USE + 130, "y": 460, "label": "Corridor care", "resource": "corridor_spaces"},
        {"event": "wait_bed", "x": _X_QUEUE + 130, "y": 100, "label": "Boarding: awaiting ward bed"},
    ]
    return pd.DataFrame(rows)


def _resource_counts(scenario: Scenario) -> SimpleNamespace:
    counts = {name: sp.spaces for name, sp in scenario.streams.items()}
    counts["triage_rooms"] = scenario.spaces.get("triage_rooms", 0)
    counts["corridor_spaces"] = scenario.spaces.get("corridor_spaces", 0)
    return SimpleNamespace(**{k: v for k, v in counts.items() if v > 0})


def build_animation(
    event_log: pd.DataFrame,
    scenario: Scenario,
    every_x_minutes: int = 15,
    limit_days: int | None = 2,
    height: int = 750,
):
    """Return a plotly Figure animating the event log, or raise RuntimeError."""
    try:
        from vidigi.animation import animate_activity_log
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"vidigi is not available: {exc}") from exc

    log = event_log[event_log["event"].isin(_KEEP_EVENTS)].copy()
    if log.empty:
        raise RuntimeError("event log holds no animatable events")
    positions = _positions(scenario)
    used = set(log["event"])
    positions = positions[positions["event"].isin(used | {"arrival", "depart"})]

    try:
        return animate_activity_log(
            event_log=log,
            event_position_df=positions,
            scenario=_resource_counts(scenario),
            every_x_time_units=every_x_minutes,
            limit_duration=limit_days * 1440 if limit_days else None,
            plotly_height=height,
            wrap_queues_at=15,
            step_snapshot_max=45,
            time_display_units="dhm",
            display_stage_labels=True,
        )
    except Exception as exc:
        raise RuntimeError(f"vidigi could not render the animation: {exc}") from exc
