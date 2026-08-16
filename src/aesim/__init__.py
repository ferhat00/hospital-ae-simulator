"""aesim — UK A&E department discrete-event simulator.

Type 1 ED + co-located UTC, SDEC/hot-clinic routes, and an inpatient bed pool
as the downstream exit-block constraint. See docs/model_card.md for scope,
assumptions and data sources.
"""

from aesim.entities import ArrivalMode, Disposal, Patient, Stream, TriageCat
from aesim.params import Scenario

__version__ = "0.1.0"

__all__ = [
    "ArrivalMode",
    "Disposal",
    "Patient",
    "Scenario",
    "Stream",
    "TriageCat",
    "__version__",
]
