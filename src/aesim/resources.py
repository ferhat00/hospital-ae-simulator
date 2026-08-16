"""Resource layer: shift-varying staff pools, physical spaces, and the bed pool.

Shift-varying capacity uses the ghost-slot pattern: the underlying SimPy
resource is sized to the roster's maximum simultaneous headcount, and a roster
process holds ``max - on_duty`` "ghost" requests at an ultra-high priority.
Reducing capacity queues a ghost that jumps every patient but still waits for a
natural release — a clinician finishes their current patient and then goes
home; no service is ever interrupted by a shift change. Increasing capacity
releases ghosts. Ghosts never preempt, so an in-progress resus is safe even on
a preemptive resource.
"""

from __future__ import annotations

import heapq
from collections.abc import Generator

import numpy as np
import simpy

from aesim.params import BedPoolParams, StaffPoolParams

GHOST_PRIORITY = -1_000_000
#: ED decision-to-admit patients outrank elective admissions for beds by default.
BED_PRIORITY_ED = 0
BED_PRIORITY_ELECTIVE = 1


class ShiftScheduledResource:
    """A staff pool whose effective capacity follows a shift roster.

    ``request(priority, preempt)`` proxies to the underlying SimPy resource;
    patients use ``priority=int(triage_cat)`` (lower = more urgent).
    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        pool: StaffPoolParams,
        seed: np.random.SeedSequence,
        preemptive: bool = False,
    ):
        self.env = env
        self.name = name
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.capacity = max(pool.max_rostered(), 1)
        cls = simpy.PreemptiveResource if preemptive else simpy.PriorityResource
        self.resource = cls(env, capacity=self.capacity)
        self._ghosts: list[simpy.resources.resource.PriorityRequest] = []
        self._daily_round_up: bool = False
        env.process(self._roster_process())

    # ------------------------------------------------------------------ requests

    def request(self, priority: int, preempt: bool = False):
        if isinstance(self.resource, simpy.PreemptiveResource):
            return self.resource.request(priority=priority, preempt=preempt)
        return self.resource.request(priority=priority)

    def release(self, req) -> None:
        self.resource.release(req)

    # ------------------------------------------------------------------- roster

    def _boundaries(self) -> list[float]:
        """Unique roster-change clock times (hours) within a day."""
        times = {b.start % 24 for b in self.pool.shifts} | {b.end % 24 for b in self.pool.shifts}
        return sorted(times)

    def _roster_process(self) -> Generator:
        boundaries = self._boundaries()
        self._daily_u = 1.0  # deterministic round-down until the first daily draw
        while True:
            day_start = self.env.now - (self.env.now % 1440.0)
            # One availability-rounding draw per day, applied at every boundary that
            # day: 0.89 x 9 rostered => 8 on most days, 9 on ~1 day a week — without
            # jittering capacity at every shift change.
            self._daily_u = float(self.rng.uniform())
            self._apply_target()
            for hours in boundaries:
                t_next = day_start + hours * 60.0
                if t_next <= self.env.now:
                    continue
                yield self.env.timeout(t_next - self.env.now)
                self._apply_target()
            next_midnight = day_start + 1440.0
            if next_midnight > self.env.now:
                yield self.env.timeout(next_midnight - self.env.now)

    def _effective_target(self) -> int:
        rostered = self.pool.rostered_at((self.env.now % 1440.0) / 60.0)
        exact = rostered * self.pool.availability_factor
        base = int(exact)
        frac = exact - base
        target = base + (1 if self._daily_u < frac else 0)
        return min(target, rostered)

    def _apply_target(self) -> None:
        target = self._effective_target()
        ghosts_needed = self.capacity - target
        while len(self._ghosts) < ghosts_needed:
            if isinstance(self.resource, simpy.PreemptiveResource):
                req = self.resource.request(priority=GHOST_PRIORITY, preempt=False)
            else:
                req = self.resource.request(priority=GHOST_PRIORITY)
            self._ghosts.append(req)
        while len(self._ghosts) > ghosts_needed:
            self._drop_one_ghost()

    def _drop_one_ghost(self) -> None:
        # prefer cancelling a ghost still queueing (it never took a slot)
        for i, req in enumerate(self._ghosts):
            if not req.triggered:
                req.cancel()
                del self._ghosts[i]
                return
        req = self._ghosts.pop()
        self.resource.release(req)

    # ---------------------------------------------------------------- telemetry

    @property
    def on_duty(self) -> int:
        """Current effective capacity (slots not held or claimed by ghosts)."""
        return self.capacity - len(self._ghosts)

    @property
    def busy(self) -> int:
        """Staff currently with a patient."""
        ghost_users = sum(1 for g in self._ghosts if g.triggered)
        return self.resource.count - ghost_users

    @property
    def queue_length(self) -> int:
        ghost_queued = sum(1 for g in self._ghosts if not g.triggered)
        return len(self.resource.queue) - ghost_queued


class SpaceStore:
    """A pool of identical physical spaces (bays/cubicles/rooms/chairs) with
    priority queueing and stable per-slot ids for the animation event log."""

    def __init__(self, env: simpy.Environment, name: str, n_spaces: int):
        self.env = env
        self.name = name
        self.n_spaces = n_spaces
        self.enabled = n_spaces > 0
        self.resource = (
            simpy.PriorityResource(env, capacity=n_spaces) if self.enabled else None
        )
        self._free_ids: list[int] = list(range(1, n_spaces + 1))
        heapq.heapify(self._free_ids)
        self._slot_of: dict[object, int] = {}

    def request(self, priority: int):
        if not self.enabled:
            raise RuntimeError(f"space pool {self.name!r} has no capacity")
        return self.resource.request(priority=priority)

    def claim_slot(self, req) -> int:
        """Call once the request has been granted; returns the slot id."""
        slot = heapq.heappop(self._free_ids)
        self._slot_of[req] = slot
        return slot

    def release(self, req) -> int:
        slot = self._slot_of.pop(req)
        heapq.heappush(self._free_ids, slot)
        self.resource.release(req)
        return slot

    @property
    def in_use(self) -> int:
        return self.resource.count if self.enabled else 0

    @property
    def queue_length(self) -> int:
        return len(self.resource.queue) if self.enabled else 0


class BedPool:
    """Inpatient beds — the downstream exit-block constraint.

    ED decision-to-admit patients and elective admissions compete for the same
    pool; the ward stay is (sampled LoS snapped to an afternoon-weighted
    discharge hour). ``prefill`` seeds t=0 occupancy with synthetic patients
    carrying stationary residual stays so the model starts near steady state.
    """

    def __init__(self, env: simpy.Environment, params: BedPoolParams):
        self.env = env
        self.params = params
        self.enabled = params.enabled and params.n_beds > 0
        self.resource = (
            simpy.PriorityResource(env, capacity=params.n_beds) if self.enabled else None
        )
        self.elective_cancellations = 0

    def request(self, priority: int = BED_PRIORITY_ED):
        return self.resource.request(priority=priority)

    def release(self, req) -> None:
        self.resource.release(req)

    @property
    def occupied(self) -> int:
        return self.resource.count if self.enabled else 0

    @property
    def dta_queue(self) -> int:
        return len(self.resource.queue) if self.enabled else 0
