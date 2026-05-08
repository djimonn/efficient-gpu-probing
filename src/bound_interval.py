from dataclasses import dataclass


@dataclass(frozen=True)
class BoundInterval:
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
