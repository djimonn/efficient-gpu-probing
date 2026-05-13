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

    def to_string(self) -> str:
        if self.lower_bound == self.upper_bound:
            return f"{{{self.lower_bound}}}"
        return f"[{self.lower_bound}, {self.upper_bound}]"

    @classmethod
    def from_single_value(cls, value: float) -> "BoundInterval":
        return cls(lower_bound=value, upper_bound=value)
