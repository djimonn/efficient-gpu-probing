import math
from typing import Tuple

from bound_interval import BoundInterval
from types import VarIndex

from milp_problem import MILPProblem


class ProbingCache:
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.cache: dict[
            Tuple[VarIndex, BoundInterval], dict[VarIndex, BoundInterval]
        ] = {}

    def probe(self, var_index: VarIndex) -> None:
        default_interval = BoundInterval(
            self.problem.lb[var_index], self.problem.ub[var_index]
        )
        for probe_interval in self._split_interval(default_interval):
            pass

    def _split_interval(
        self, interval: BoundInterval
    ) -> Tuple[BoundInterval, BoundInterval]:
        if interval.upper_bound - interval.lower_bound == 1:
            return [
                BoundInterval(interval.lower_bound, interval.lower_bound),
                BoundInterval(interval.upper_bound, interval.upper_bound),
            ]
        elif math.isfinite(interval.lower_bound) and math.isfinite(
            interval.upper_bound
        ):
            mid = (interval.lower_bound + interval.upper_bound) / 2
            return [
                BoundInterval(interval.lower_bound, mid),
                BoundInterval(mid + 1, interval.upper_bound),
            ]
        elif math.isfinite(interval.lower_bound):
            return [
                BoundInterval(interval.lower_bound, interval.lower_bound),
                BoundInterval(interval.lower_bound + 1, math.inf),
            ]
        elif math.isfinite(interval.upper_bound):
            return [
                BoundInterval(-math.inf, interval.upper_bound - 1),
                BoundInterval(interval.upper_bound, interval.upper_bound),
            ]
        else:
            raise ValueError("Cannot split an unbounded interval")
