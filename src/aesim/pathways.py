"""Patient-journey SimPy processes, mixed into :class:`aesim.model.AEModel`.

LOCK ORDER (deadlock prevention — do not violate):
    1. physical space (bay / cubicle / room / chair)
    2. clinician (from the stream's staff pool)
    3. inpatient bed (requested while STILL HOLDING the space; the space is
       released only after the bed is granted — this nested hold is exit block)
Beds never require any ED resource. A stream-to-stream transfer (resus step-down,
UTC escalation, SDEC referral) acquires the destination space *before* releasing
the origin space only where the patient physically cannot wait unattended
(resus step-down); otherwise the origin space is released first.

Clinician service uses catch-and-resume preemption: a RED resus arrival may
preempt a doctor from a lower-priority consultation; the victim re-requests at
their own priority and completes the remaining service time.
"""

from __future__ import annotations

from collections.abc import Generator

import simpy

from aesim.entities import ArrivalMode, Disposal, Patient, Stream, TriageCat


class PathwaysMixin:
    """Requires the attributes wired up by AEModel.__init__ (env, resources,
    samplers, log helpers)."""

    # ------------------------------------------------------------------ arrival

    def patient_journey(self, p: Patient) -> Generator:
        self.log_event(p, "arrival_departure", "arrival")
        try:
            if p.mode is ArrivalMode.AMBULANCE:
                yield from self._ambulance_front_door(p)
            elif p.mode is ArrivalMode.BOOKED_UTC:
                p.cat = TriageCat.GREEN
                p.stream = Stream.UTC
                yield from self._utc(p)
                return
            else:
                yield from self._walkin_front_door(p)

            if p.disposal is Disposal.LWBS:
                return

            match p.stream:
                case Stream.RESUS:
                    yield from self._resus(p)
                case Stream.MAJORS:
                    yield from self._majors(p)
                case Stream.MINORS:
                    yield from self._minors(p)
                case Stream.UTC:
                    yield from self._utc(p)
                case Stream.SDEC:
                    yield from self._sdec(p)
        finally:
            if p.t_departure is not None:
                self.log_event(p, "arrival_departure", "depart")

    # -------------------------------------------------------------- front doors

    def _walkin_front_door(self, p: Patient) -> Generator:
        # registration (clerk not modelled as a scarce resource; RCEM <=5 min)
        p.t_reg_start = self.env.now
        yield self.env.timeout(self.dists["reg_time"].sample())
        p.t_reg_end = self.env.now

        # triage: room then nurse (lock order: space -> clinician)
        self.log_event(p, "queue", "wait_triage")
        room_req = self.triage_rooms.request(priority=int(TriageCat.YELLOW))
        yield room_req
        slot = self.triage_rooms.claim_slot(room_req)
        nurse_req = self.triage_nurses.request(priority=int(TriageCat.YELLOW))
        yield nurse_req
        p.t_triage_start = self.env.now
        self.log_event(p, "resource_use", "triage", resource_id=slot)
        yield self.env.timeout(self.dists["triage_time_walkin"].sample())
        p.t_triage_end = self.env.now
        self.log_event(p, "resource_use_end", "triage", resource_id=slot)
        self.triage_nurses.release(nurse_req)
        self.triage_rooms.release(room_req)

        p.cat = TriageCat[self.dists["cat_walkin"].sample()]
        self._assign_attributes(p)
        p.stream = self._stream_walkin(p)

    def _ambulance_front_door(self, p: Patient) -> Generator:
        """Stretcher assessment at the door, then wait for a receiving space.
        The crew is released only once the patient is off the stretcher."""
        nurse_req = self.triage_nurses.request(priority=int(TriageCat.ORANGE))
        yield nurse_req
        p.t_triage_start = self.env.now
        yield self.env.timeout(self.dists["triage_time_ambulance"].sample())
        p.t_triage_end = self.env.now
        self.triage_nurses.release(nurse_req)

        p.cat = TriageCat[self.dists["cat_ambulance"].sample()]
        self._assign_attributes(p)

        # "fit to sit": ambulant low-acuity arrivals are handed over at the door
        # and wait in the seated area — the crew is not held for a cubicle
        if p.cat in (TriageCat.GREEN, TriageCat.BLUE):
            p.stream = Stream.MINORS
            yield from self._handover_with_nurse(p)
            p.t_offload = self.env.now
            self.note_offload()
            return

        p.stream = Stream.RESUS if p.cat is TriageCat.RED else Stream.MAJORS

        # optional HALO/cohort offload area frees the crew before a cubicle exists
        if self.offload_spaces.enabled and p.stream is Stream.MAJORS:
            self.log_event(p, "queue", "wait_offload")
            off_req = self.offload_spaces.request(priority=int(p.cat))
            yield off_req
            self.offload_spaces.claim_slot(off_req)
            yield from self._handover_with_nurse(p)
            p.t_offload = self.env.now  # crew released; patient waits in offload area
            self.note_offload()
            yield from self._enter_majors_space(p)
            self.offload_spaces.release(off_req)
        else:
            # crew waits with the patient until the receiving space is granted
            self.log_event(p, "queue", "wait_handover_space")
            yield from self._acquire_stream_space(p)
            yield from self._handover_with_nurse(p)
            p.t_offload = self.env.now
            self.note_offload()

    def _assign_attributes(self, p: Patient) -> None:
        p.is_paeds = self.dists["paeds"].sample()
        p.will_admit = self.dists["route_admit"].sample() < self.scenario.routing.admission_prob_by_cat[p.cat.name]
        elig = self.scenario.routing.sdec_eligible_frac.get(p.cat.name, 0.0)
        p.sdec_eligible = self.dists["route_sdec_elig"].sample() < elig

    def _stream_walkin(self, p: Patient) -> Stream:
        if p.cat is TriageCat.RED:
            return Stream.RESUS
        # co-located UTC streaming (Type 3), walk-ins only
        utc = self.scenario.streams.get("utc")
        if utc is not None and utc.spaces > 0 and utc.accepts_at(self._hour_now()):
            p_stream = self.scenario.routing.utc_stream_prob.get(p.cat.name, 0.0)
            if self.dists["route_utc"].sample() < p_stream:
                return Stream.UTC
        # SDEC direct streaming for eligible patients while it accepts
        if p.sdec_eligible and self._sdec_accepts():
            return Stream.SDEC
        if p.cat in (TriageCat.ORANGE, TriageCat.YELLOW):
            return Stream.MAJORS
        return Stream.MINORS

    def _hour_now(self) -> float:
        return (self.env.now % 1440.0) / 60.0

    def _sdec_accepts(self) -> bool:
        sp = self.scenario.streams.get("sdec")
        if sp is None or sp.spaces <= 0 or not sp.accepts_at(self._hour_now()):
            return False
        store = self.spaces[Stream.SDEC]
        return store.in_use + store.queue_length < sp.spaces

    # ------------------------------------------------------------ space helpers

    def _acquire_stream_space(self, p: Patient) -> Generator:
        """Acquire the receiving space for an ambulance patient (resus bay or
        majors cubicle) and remember the request for the stream process."""
        if p.stream is Stream.RESUS:
            yield from self._enter_resus_space(p)
        else:
            yield from self._enter_majors_space(p)

    def _enter_majors_space(self, p: Patient) -> Generator:
        store = self.spaces[Stream.MAJORS]
        self.log_event(p, "queue", "wait_space_majors")
        req = store.request(priority=int(p.cat))
        yield req
        slot = store.claim_slot(req)
        p.t_space = self.env.now
        self.log_event(p, "resource_use", "space_majors", resource_id=slot)
        self._held_space[p.pid] = (store, req, "space_majors", slot)

    def _enter_resus_space(self, p: Patient) -> Generator:
        store = self.spaces[Stream.RESUS]
        self.log_event(p, "queue", "wait_space_resus")
        req = store.request(priority=int(p.cat))
        yield req
        slot = store.claim_slot(req)
        p.t_space = self.env.now
        self.log_event(p, "resource_use", "space_resus", resource_id=slot)
        self._held_space[p.pid] = (store, req, "space_resus", slot)

    def _release_held_space(self, p: Patient) -> None:
        store, req, event, slot = self._held_space.pop(p.pid)
        store.release(req)
        self.log_event(p, "resource_use_end", event, resource_id=slot)

    # ---------------------------------------------------------------- streams

    def _resus(self, p: Patient) -> Generator:
        if p.pid not in self._held_space:  # walk-in REDs
            yield from self._enter_resus_space(p)

        yield from self._clinician_service(
            p, pool=self.staff[self.scenario.streams["resus"].clinician_pool],
            duration=self.dists["svc_resus"].sample(), preempt=True, first_contact=True,
        )
        yield from self._nursing_care(p, "resus")
        yield from self._diagnostics_delay(p)

        if self.dists["route_stepdown"].sample() < self.scenario.streams["resus"].stepdown_to_majors_prob:
            # step-down: secure a majors cubicle BEFORE giving up the bay
            resus_hold = self._held_space.pop(p.pid)
            self.log_event(p, "queue", "wait_space_majors")
            store = self.spaces[Stream.MAJORS]
            req = store.request(priority=int(p.cat))
            yield req
            slot = store.claim_slot(req)
            self.log_event(p, "resource_use", "space_majors", resource_id=slot)
            self._held_space[p.pid] = (store, req, "space_majors", slot)
            r_store, r_req, r_event, r_slot = resus_hold
            r_store.release(r_req)
            self.log_event(p, "resource_use_end", r_event, resource_id=r_slot)
            yield from self._majors_after_assessment(p, assessed=True)
            return

        if p.will_admit:
            yield from self._admit(p)
        else:
            yield from self._depart(p, Disposal.DISCHARGED)

    def _majors(self, p: Patient) -> Generator:
        if p.pid not in self._held_space:  # walk-ins; ambulance already holds a cubicle
            yield from self._enter_majors_space(p)

        yield from self._clinician_service(
            p, pool=self.staff[self.scenario.streams["majors"].clinician_pool],
            duration=self.dists["svc_majors_assess"].sample(), first_contact=True,
        )
        yield from self._nursing_care(p, "majors")
        yield from self._diagnostics_delay(p)

        # secondary SDEC referral after workup: frees the cubicle
        if p.sdec_eligible and not p.will_admit and self._sdec_accepts():
            self._release_held_space(p)
            yield from self._sdec(p)
            return

        yield from self._majors_after_assessment(p, assessed=False)

    def _majors_after_assessment(self, p: Patient, assessed: bool) -> Generator:
        if not assessed:
            yield from self._clinician_service(
                p, pool=self.staff[self.scenario.streams["majors"].clinician_pool],
                duration=self.dists["svc_majors_treat"].sample(),
            )
        if p.will_admit:
            yield from self._admit(p)
        else:
            yield from self._depart(p, Disposal.DISCHARGED)

    def _minors(self, p: Patient) -> Generator:
        store = self.spaces[Stream.MINORS]
        self.log_event(p, "queue", "wait_space_minors")
        req = store.request(priority=int(p.cat))
        patience = self.scenario.triage.patience_min.get(p.cat.name)
        if patience is not None:
            result = yield req | self.env.timeout(patience)
            if req not in result:
                req.cancel()
                yield from self._depart(p, Disposal.LWBS)
                return
        else:
            yield req
        slot = store.claim_slot(req)
        p.t_space = self.env.now
        self.log_event(p, "resource_use", "space_minors", resource_id=slot)
        self._held_space[p.pid] = (store, req, "space_minors", slot)

        yield from self._clinician_service(
            p, pool=self.staff[self.scenario.streams["minors"].clinician_pool],
            duration=self.dists["svc_minors"].sample(), first_contact=True,
        )
        yield from self._nursing_care(p, "minors")
        if p.will_admit:
            yield from self._admit(p)
        else:
            yield from self._depart(p, Disposal.DISCHARGED)

    def _utc(self, p: Patient) -> Generator:
        store = self.spaces[Stream.UTC]
        self.log_event(p, "queue", "wait_space_utc")
        req = store.request(priority=int(p.cat))
        patience = self.scenario.triage.patience_min.get(p.cat.name)
        if patience is not None:
            result = yield req | self.env.timeout(patience)
            if req not in result:
                req.cancel()
                yield from self._depart(p, Disposal.LWBS)
                return
        else:
            yield req
        slot = store.claim_slot(req)
        p.t_space = self.env.now
        self.log_event(p, "resource_use", "space_utc", resource_id=slot)
        self._held_space[p.pid] = (store, req, "space_utc", slot)

        yield from self._clinician_service(
            p, pool=self.staff[self.scenario.streams["utc"].clinician_pool],
            duration=self.dists["svc_utc"].sample(), first_contact=True,
        )

        if self.dists["route_utc_transfer"].sample() < self.scenario.routing.utc_transfer_to_majors:
            # escalation into the ED: release the UTC room, join majors queue
            self._release_held_space(p)
            p.stream = Stream.MAJORS
            yield from self._majors(p)
            return
        yield from self._depart(p, Disposal.DISCHARGED)

    def _sdec(self, p: Patient) -> Generator:
        p.stream = Stream.SDEC
        store = self.spaces[Stream.SDEC]
        self.log_event(p, "queue", "wait_space_sdec")
        req = store.request(priority=int(p.cat))
        yield req
        slot = store.claim_slot(req)
        p.t_space = self.env.now
        self.log_event(p, "resource_use", "space_sdec", resource_id=slot)
        self._held_space[p.pid] = (store, req, "space_sdec", slot)

        yield from self._clinician_service(
            p, pool=self.staff[self.scenario.streams["sdec"].clinician_pool],
            duration=self.dists["svc_sdec"].sample(), first_contact=p.t_first_clinician is None,
        )

        if self.dists["route_sdec_convert"].sample() < self.scenario.routing.sdec_conversion_to_admission:
            yield from self._admit(p, sdec=True)
        else:
            yield from self._depart(p, Disposal.SDEC_DISCHARGED)

    # ------------------------------------------------------------- disposition

    def _admit(self, p: Patient, sdec: bool = False) -> Generator:
        # hot clinic diverts a fraction of would-be admissions to a next-day slot
        if self.dists["route_hot_clinic"].sample() < self.scenario.routing.hot_clinic_frac:
            yield from self._depart(p, Disposal.HOT_CLINIC)
            return

        # specialty referral -> review -> decision to admit (holds the space)
        yield self.env.timeout(self.dists["specialty_review"].sample())
        p.t_dta = self.env.now
        self.log_event(p, "queue", "wait_bed")

        if not self.beds.enabled:
            p.t_bed = self.env.now
        else:
            self.n_boarding += 1
            bed_req = self.beds.request()
            # corridor escalation: if others queue for this space and a corridor
            # slot is free, decant the boarder so the cubicle can turn over
            corridor_req = None
            if (
                self.corridor.enabled
                and p.pid in self._held_space
                and self._held_space[p.pid][0].queue_length
                > self.scenario.routing.corridor_trigger_queue
                and self.corridor.in_use < self.corridor.n_spaces
            ):
                corridor_req = self.corridor.request(priority=int(p.cat))
                yield corridor_req  # free slot -> immediate
                slot = self.corridor.claim_slot(corridor_req)
                self._release_held_space(p)
                self.log_event(p, "resource_use", "corridor", resource_id=slot)
                t_corridor = self.env.now
            yield bed_req
            self.n_boarding -= 1
            p.t_bed = self.env.now
            if corridor_req is not None:
                self.corridor_minutes += self.env.now - t_corridor
                self.log_event(p, "resource_use_end", "corridor")
                self.corridor.release(corridor_req)
            self.env.process(self._bed_stay(p, bed_req))

        # the ED space is released only now — boarding consumed it until the bed
        yield from self._depart(p, Disposal.SDEC_ADMITTED if sdec else Disposal.ADMITTED)

    def _bed_stay(self, p: Patient, bed_req) -> Generator:
        los_min = self.dists["bed_los"].sample() * 1440.0
        departure = self.apply_weekend_slip(
            self.discharge_hours.snap(p.t_bed, p.t_bed + los_min)
        )
        yield self.env.timeout(max(departure - self.env.now, 1.0))
        # clean/ward-acceptance friction before the bed can take its next occupant
        yield self.env.timeout(self.dists["bed_turnaround"].sample())
        self.beds.release(bed_req)

    def _depart(self, p: Patient, disposal: Disposal) -> Generator:
        p.disposal = disposal
        p.t_departure = self.env.now
        if p.pid in self._held_space:
            self._release_held_space(p)
        self.results.completed.append(p)
        return
        yield  # pragma: no cover — makes this a generator for uniform `yield from`

    # -------------------------------------------------------- clinician service

    def _clinician_service(
        self,
        p: Patient,
        pool,
        duration: float,
        preempt: bool = False,
        first_contact: bool = False,
    ) -> Generator:
        """Acquire a clinician and deliver `duration` minutes of service,
        surviving preemption by re-requesting and completing the remainder."""
        remaining = duration
        while remaining > 1e-9:
            self.log_event(p, "queue", f"wait_clinician_{pool.name}")
            req = pool.request(priority=int(p.cat), preempt=preempt)
            start: float | None = None
            try:
                yield req
                if first_contact and p.t_first_clinician is None:
                    p.t_first_clinician = self.env.now
                    first_contact = False
                self.log_event(p, "resource_use", f"clinician_{pool.name}")
                start = self.env.now
                yield self.env.timeout(remaining)
                remaining = 0.0
                self.log_event(p, "resource_use_end", f"clinician_{pool.name}")
                pool.release(req)
            except simpy.Interrupt:
                # Preempted mid-service: SimPy has already evicted our request from
                # the resource (no release needed). Bank the minutes served and
                # re-queue at our own priority for the remainder.
                if start is not None:
                    remaining = max(remaining - (self.env.now - start), 0.0)
                    self.log_event(p, "resource_use_end", f"clinician_{pool.name}")

    # ------------------------------------------------------------ nursing care

    def _nursing_care(self, p: Patient, stream_name: str) -> Generator:
        """Obs/cannulation/medication by the stream's nurse pool (holds the space)."""
        sp = self.scenario.streams[stream_name]
        if sp.nursing_time is None or sp.nurse_pool is None:
            return
        yield from self._clinician_service(
            p,
            pool=self.staff[sp.nurse_pool],
            duration=self.dists[f"svc_nursing_{stream_name}"].sample(),
        )

    def _handover_with_nurse(self, p: Patient) -> Generator:
        """The 15-minute clinical handover needs a receiving nurse."""
        nurse_pool = self.scenario.streams["majors"].nurse_pool
        if nurse_pool is None:
            yield self.env.timeout(self.scenario.routing.handover_clinical_min)
            return
        pool = self.staff[nurse_pool]
        req = pool.request(priority=int(p.cat))
        yield req
        yield self.env.timeout(self.scenario.routing.handover_clinical_min)
        pool.release(req)

    # ------------------------------------------------------------- diagnostics

    def _diagnostics_delay(self, p: Patient) -> Generator:
        """Investigations in majors/resus are sampled delays taken in parallel
        (the patient holds their space, not a clinician)."""
        longest = 0.0
        for test, cfg in self.scenario.diagnostics.items():
            if self.dists[f"dx_{test}_p"].sample() < cfg.prob:
                longest = max(longest, self.dists[f"dx_{test}_t"].sample())
        if longest > 0:
            yield self.env.timeout(longest)
