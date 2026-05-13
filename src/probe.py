from dataclasses import dataclass

from bound_interval import BoundInterval
from type_aliases import VarIndex


@dataclass(frozen=True)
class Probe:
    var_index: VarIndex
    bound_interval: BoundInterval
