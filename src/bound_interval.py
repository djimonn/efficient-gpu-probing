from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundInterval:
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")

    def __post_init__(self):
        assert not math.isinf(self.lower_bound) or not math.isinf(
            self.upper_bound
        ), "At least one bound must be finite"
