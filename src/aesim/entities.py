"""Core entity types: the Patient record and the enums that describe its journey.

Times are simulation minutes from t=0 (midnight on day 0, a Monday).
A timestamp of None means the patient never reached that step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class TriageCat(IntEnum):
    """Manchester Triage System category. Lower value = more urgent = higher queue priority."""

    RED = 1  # immediate, target 0 min to clinician
    ORANGE = 2  # very urgent, 10 min
    YELLOW = 3  # urgent, 60 min
    GREEN = 4  # standard, 120 min
    BLUE = 5  # non-urgent, 240 min


#: MTS maximum time to first clinician contact, minutes.
MTS_TARGET_MIN: dict[TriageCat, float] = {
    TriageCat.RED: 0.0,
    TriageCat.ORANGE: 10.0,
    TriageCat.YELLOW: 60.0,
    TriageCat.GREEN: 120.0,
    TriageCat.BLUE: 240.0,
}


class Stream(StrEnum):
    """Where the patient is cared for after streaming at triage."""

    RESUS = "resus"
    MAJORS = "majors"
    MINORS = "minors"
    UTC = "utc"
    SDEC = "sdec"


class ArrivalMode(StrEnum):
    WALK_IN = "walk_in"
    AMBULANCE = "ambulance"
    BOOKED_UTC = "booked_utc"


class Disposal(StrEnum):
    DISCHARGED = "discharged"
    ADMITTED = "admitted"
    SDEC_DISCHARGED = "sdec_discharged"
    SDEC_ADMITTED = "sdec_admitted"
    HOT_CLINIC = "hot_clinic"  # admission avoided via next-day rapid-access clinic
    LWBS = "lwbs"  # left without being seen (reneged)


@dataclass
class Patient:
    """One attendance. Timestamp fields are filled in as the journey progresses."""

    pid: int
    mode: ArrivalMode
    t_arrival: float
    cat: TriageCat | None = None
    stream: Stream | None = None
    disposal: Disposal | None = None
    is_paeds: bool = False
    sdec_eligible: bool = False
    will_admit: bool = False  # sampled at triage; realised at disposition

    # journey timestamps (sim minutes)
    t_offload: float | None = None  # ambulance only: patient off stretcher, crew released
    t_reg_start: float | None = None
    t_reg_end: float | None = None
    t_triage_start: float | None = None
    t_triage_end: float | None = None
    t_space: float | None = None  # granted a physical space in their stream
    t_first_clinician: float | None = None  # start of first assessment/treatment contact
    t_dta: float | None = None  # decision to admit
    t_bed: float | None = None  # inpatient bed granted (boarding ends)
    t_departure: float | None = None

    in_warmup: bool = field(default=False)

    # -- derived measures (None where the step never happened) ------------------

    @property
    def time_in_department(self) -> float | None:
        if self.t_departure is None:
            return None
        return self.t_departure - self.t_arrival

    @property
    def time_to_triage(self) -> float | None:
        if self.t_triage_start is None:
            return None
        return self.t_triage_start - self.t_arrival

    @property
    def time_to_first_clinician(self) -> float | None:
        if self.t_first_clinician is None:
            return None
        return self.t_first_clinician - self.t_arrival

    @property
    def boarding_time(self) -> float | None:
        """Decision-to-admit to bed granted ('trolley wait')."""
        if self.t_dta is None or self.t_bed is None:
            return None
        return self.t_bed - self.t_dta

    @property
    def handover_delay(self) -> float | None:
        """Ambulance arrival to crew released."""
        if self.mode is not ArrivalMode.AMBULANCE or self.t_offload is None:
            return None
        return self.t_offload - self.t_arrival

    @property
    def admitted(self) -> bool:
        return self.disposal in (Disposal.ADMITTED, Disposal.SDEC_ADMITTED)
