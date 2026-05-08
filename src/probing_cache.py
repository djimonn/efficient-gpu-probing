from typing import Tuple

from bound_interval import BoundInterval
from types import VarIndex


class ProbingCache:
    def __init__(self):
        self.cache: dict[
            Tuple[VarIndex, BoundInterval], dict[VarIndex, BoundInterval]
        ] = {}
