from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundInterval:
    def __init__(
        self, lower_bound: float = float("-inf"), upper_bound: float = float("inf")
    ):
        assert not math.isinf(lower_bound) or not math.isinf(
            upper_bound
        ), "At least one bound must be finite"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
