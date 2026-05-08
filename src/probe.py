from dataclasses import dataclass
from types import VarIndex

from bound_interval import BoundInterval


@dataclass(frozen=True)
class Probe:
    def __init__(self, var_index: VarIndex, bound_interval: BoundInterval):
        self.var_index = var_index
        self.bound_interval = bound_interval
