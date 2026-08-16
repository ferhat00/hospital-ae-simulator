"""Periodic state snapshots: queue lengths, occupancy, staffing and beds.

These time series feed the occupancy heatmaps, utilisation calculations,
Welch warm-up analysis, and the boarders-vs-RCEM-10%-rule KPI.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from aesim.model import AEModel


def audit_process(model: AEModel) -> Generator:
    interval = model.scenario.run.audit_interval_min
    while True:
        row: dict[str, float] = {
            "time": model.env.now,
            "in_warmup": model.env.now < model.scenario.run.warm_up_min,
            "n_in_dept": model.n_in_dept,
            "n_boarding": model.n_boarding,
            "ambulances_waiting": model.n_ambulances_waiting,
            "beds_occupied": model.beds.occupied,
            "bed_dta_queue": model.beds.dta_queue,
        }
        for stream, store in model.spaces.items():
            row[f"{stream.value}_in_use"] = store.in_use
            row[f"{stream.value}_queue"] = store.queue_length
        row["triage_in_use"] = model.triage_rooms.in_use
        row["triage_queue"] = model.triage_rooms.queue_length
        row["corridor_in_use"] = model.corridor.in_use
        for name, pool in model.staff.items():
            row[f"staff_{name}_busy"] = pool.busy
            row[f"staff_{name}_on_duty"] = pool.on_duty
        model.audit_rows.append(row)
        yield model.env.timeout(interval)
