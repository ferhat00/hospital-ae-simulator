"""Ghost-slot roster behaviour: capacity follows the schedule, nobody's service
is interrupted at shift change, and preemptive pools never let ghosts preempt."""

from __future__ import annotations

import numpy as np
import simpy

from aesim.params import BedPoolParams, DistConfig, ShiftBlock, StaffPoolParams
from aesim.resources import BedPool, ShiftScheduledResource, SpaceStore


def _pool(shifts, availability=1.0):
    return StaffPoolParams(
        shifts=tuple(ShiftBlock(**s) for s in shifts), availability_factor=availability
    )


def test_capacity_tracks_schedule():
    env = simpy.Environment()
    pool = _pool([{"start": 8, "end": 17, "count": 2}, {"start": 17, "end": 8, "count": 1}])
    res = ShiftScheduledResource(env, "docs", pool, np.random.SeedSequence(1))
    trace = {}

    def probe():
        for t in (60, 9 * 60, 16 * 60, 18 * 60, 25 * 60, 33 * 60):
            yield env.timeout(t - env.now)
            trace[t] = res.on_duty

    env.process(probe())
    env.run(until=36 * 60)
    assert trace[60] == 1  # 01:00 night
    assert trace[9 * 60] == 2  # 09:00 day
    assert trace[16 * 60] == 2
    assert trace[18 * 60] == 1  # evening
    assert trace[25 * 60] == 1  # 01:00 next day
    assert trace[33 * 60] == 2  # 09:00 next day


def test_no_service_interruption_at_shift_change():
    """Two patients in service at 16:30; capacity drops to 1 at 17:00. Both must
    finish their 60-min services; only then does the ghost claim a slot."""
    env = simpy.Environment()
    pool = _pool([{"start": 8, "end": 17, "count": 2}, {"start": 17, "end": 8, "count": 1}])
    res = ShiftScheduledResource(env, "docs", pool, np.random.SeedSequence(2))
    finished = []

    def patient(name, start, service):
        yield env.timeout(start)
        req = res.request(priority=3)
        yield req
        yield env.timeout(service)
        res.release(req)
        finished.append((name, env.now))

    env.process(patient("a", 16 * 60 + 30, 60))
    env.process(patient("b", 16 * 60 + 30, 60))
    env.process(patient("c", 17 * 60 + 10, 30))  # arrives after the drop
    env.run(until=24 * 60)

    times = dict(finished)
    assert times["a"] == 17 * 60 + 30  # both ran in parallel across the boundary
    assert times["b"] == 17 * 60 + 30
    # c had to wait for the ghost-reduced single slot: starts when a/b finish
    assert times["c"] == 18 * 60


def test_availability_factor_rounds_stochastically_per_day():
    env = simpy.Environment()
    pool = _pool([{"start": 0, "end": 24, "count": 9}], availability=0.89)  # 8.01 eff
    res = ShiftScheduledResource(env, "docs", pool, np.random.SeedSequence(3))
    daily = []

    def probe():
        while True:
            yield env.timeout(1440)
            daily.append(res.on_duty)

    env.process(probe())
    env.run(until=200 * 1440)
    assert set(daily) <= {8, 9}
    share_9 = daily.count(9) / len(daily)
    assert 0.005 <= share_9 <= 0.05  # ~1% of days round up (frac = .01)


def test_ghosts_never_preempt_on_preemptive_pool():
    """A capacity drop must not interrupt an in-progress service even on a
    PreemptiveResource, and a RED preempt must still work against patients."""
    env = simpy.Environment()
    pool = _pool([{"start": 8, "end": 17, "count": 1}, {"start": 17, "end": 8, "count": 1}])
    res = ShiftScheduledResource(env, "docs", pool, np.random.SeedSequence(4), preemptive=True)
    events = []

    def routine(start):
        yield env.timeout(start)
        req = res.request(priority=4, preempt=False)
        yield req
        try:
            yield env.timeout(120)
            res.release(req)
            events.append(("routine_done", env.now))
        except simpy.Interrupt:
            events.append(("routine_preempted", env.now))

    def red(start):
        yield env.timeout(start)
        req = res.request(priority=1, preempt=True)
        yield req
        yield env.timeout(30)
        res.release(req)
        events.append(("red_done", env.now))

    # routine starts 16:30, runs across the 17:00 boundary unharmed
    env.process(routine(16 * 60 + 30))
    env.run(until=19 * 60)
    assert ("routine_done", 18 * 60 + 30) in events

    # fresh env: RED arrival preempts routine service
    env2 = simpy.Environment()
    res2 = ShiftScheduledResource(env2, "docs", pool, np.random.SeedSequence(5), preemptive=True)
    events.clear()

    def routine2():
        yield env2.timeout(9 * 60)
        req = res2.request(priority=4, preempt=False)
        yield req
        try:
            yield env2.timeout(120)
            res2.release(req)
            events.append(("routine_done", env2.now))
        except simpy.Interrupt:
            events.append(("routine_preempted", env2.now))

    def red2():
        yield env2.timeout(9 * 60 + 10)
        req = res2.request(priority=1, preempt=True)
        yield req
        yield env2.timeout(30)
        res2.release(req)
        events.append(("red_done", env2.now))

    env2.process(routine2())
    env2.process(red2())
    env2.run(until=12 * 60)
    assert ("routine_preempted", 9 * 60 + 10) in events
    assert ("red_done", 9 * 60 + 40) in events


def test_space_store_slot_ids_recycle():
    env = simpy.Environment()
    store = SpaceStore(env, "cubicles", 2)

    def use(expect_slot):
        req = store.request(priority=3)
        yield req
        slot = store.claim_slot(req)
        assert slot == expect_slot
        yield env.timeout(10)
        store.release(req)

    def scenario():
        yield env.process(use(1))
        yield env.process(use(1))  # lowest free id is reused

    env.process(scenario())
    env.run()


def test_bed_pool_disabled_and_enabled():
    env = simpy.Environment()
    disabled = BedPool(
        env,
        BedPoolParams(
            n_beds=10,
            los_days=DistConfig("lognormal", mean=5, cv=1.0),
            discharge_hour_weights=tuple([1.0] * 24),
            enabled=False,
        ),
    )
    assert not disabled.enabled
    enabled = BedPool(
        env,
        BedPoolParams(
            n_beds=2,
            los_days=DistConfig("lognormal", mean=5, cv=1.0),
            discharge_hour_weights=tuple([1.0] * 24),
        ),
    )
    assert enabled.enabled and enabled.occupied == 0


def test_pool_with_zero_availability_never_negative():
    env = simpy.Environment()
    pool = _pool([{"start": 8, "end": 10, "count": 1}])  # tiny window
    res = ShiftScheduledResource(env, "x", pool, np.random.SeedSequence(6))
    env.run(until=3 * 1440)
    assert res.on_duty >= 0
    assert res.busy >= 0
