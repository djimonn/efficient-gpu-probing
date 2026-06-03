# pyright: basic
from abc import ABC, abstractmethod
import math
from typing import Any, Generator, Tuple, cast
import numpy as np
import numpy.typing as npt

from bound_interval import BoundInterval
from cache_entry import CacheEntry
from probe_metrics import ProbeMetrics
from propagation_result import PropagationResult
from propagation_state import PropagationState
from type_aliases import VarIndex

from milp_problem import MILPProblem


class ProbingCache(ABC):
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.probe_results: dict[Tuple[VarIndex, BoundInterval], CacheEntry] = {}

    @abstractmethod
    def probe(self, var_index: VarIndex) -> list[ProbeMetrics]:
        pass

    @abstractmethod
    def propagate_until_fixpoint(
        self, additional_constraint: tuple[VarIndex, BoundInterval]
    ) -> Any:
        pass

    def build_cache_entry_by_host_scan(self, result: PropagationResult) -> CacheEntry:
        if not result.is_feasible:
            return CacheEntry(False)
        var_bounds: dict[VarIndex, BoundInterval] = {}
        for k, (lb, ub, orig_lb, orig_ub) in enumerate(
            zip(
                cast(npt.NDArray[np.float64], result.lb),
                cast(npt.NDArray[np.float64], result.ub),
                self.problem.original_lb,
                self.problem.original_ub,
            )
        ):
            if lb > orig_lb or ub < orig_ub:
                var_bounds[k] = BoundInterval(lower_bound=lb, upper_bound=ub)
        return CacheEntry(True, var_bounds=var_bounds)

    def build_cache_entry_from_compacted_bounds(
        self,
        is_feasible: bool,
        changed_indices: npt.NDArray[np.int64],
        changed_lb: npt.NDArray[np.float64],
        changed_ub: npt.NDArray[np.float64],
    ) -> CacheEntry:
        if not is_feasible:
            return CacheEntry(False)

        var_bounds = {
            int(k): BoundInterval(float(lb), float(ub))
            for k, lb, ub in zip(changed_indices, changed_lb, changed_ub)
        }
        return CacheEntry(True, var_bounds=var_bounds)

    def split_interval(
        self, interval: BoundInterval
    ) -> Tuple[BoundInterval, BoundInterval]:
        if interval.upper_bound - interval.lower_bound == 1:
            return (
                BoundInterval.from_single_value(interval.lower_bound),
                BoundInterval.from_single_value(interval.upper_bound),
            )
        elif math.isfinite(interval.lower_bound) and math.isfinite(
            interval.upper_bound
        ):
            mid = math.floor((interval.lower_bound + interval.upper_bound) / 2)
            return (
                BoundInterval(interval.lower_bound, mid),
                BoundInterval(mid + 1, interval.upper_bound),
            )
        elif math.isfinite(interval.lower_bound):
            return (
                BoundInterval(interval.lower_bound, interval.lower_bound),
                BoundInterval(interval.lower_bound + 1, math.inf),
            )
        elif math.isfinite(interval.upper_bound):
            return (
                BoundInterval(-math.inf, interval.upper_bound - 1),
                BoundInterval(interval.upper_bound, interval.upper_bound),
            )
        else:
            raise ValueError("Cannot split an unbounded interval")

    def compute_tight_bounds(
        self, problem: MILPProblem, i: int, k: VarIndex, state: PropagationState
    ) -> BoundInterval:
        tight_upper_bound = state.ub[k]
        try:
            tight_upper_bound = problem.get_tight_upper_bound(i, k, state)
        except ValueError:
            pass
        tight_lower_bound = state.lb[k]
        try:
            tight_lower_bound = problem.get_tight_lower_bound(i, k, state)
        except ValueError:
            pass
        return BoundInterval(
            lower_bound=(
                math.ceil(tight_lower_bound)
                if self.problem.is_integer[k] and not math.isinf(tight_lower_bound)
                else tight_lower_bound
            ),
            upper_bound=(
                math.floor(tight_upper_bound)
                if self.problem.is_integer[k] and not math.isinf(tight_upper_bound)
                else tight_upper_bound
            ),
        )


class ProbingCache2:
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.probe_results: dict[Tuple[VarIndex, BoundInterval], CacheEntry] = {}

    def _propagate_until_fixpoint_advanced(self, problem: MILPProblem) -> tuple[
        bool,  # is_feasible
        npt.NDArray[np.int64],  # changed_indices
        npt.NDArray[np.float64],  # changed_lb
        npt.NDArray[np.float64],  # changed_ub
    ]:

        def row_nonzeros(i: int) -> Generator[VarIndex, None, None]:
            start = problem.A.indptr[i]  # type: ignore
            end = problem.A.indptr[i + 1]  # type: ignore
            for ptr in range(start, end):  # type: ignore
                yield problem.A.indices[ptr]  # type: ignore

        state = PropagationState(problem.original_lb.copy(), problem.original_ub.copy())
        changed_mask = np.zeros(problem.num_variables, dtype=np.bool_)

        changed = True
        while changed:
            changed = False
            for i in range(problem.num_constraints):
                if problem.constraint_is_infeasible(i, state):
                    return (
                        False,
                        np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64),
                    )
                for k in row_nonzeros(i):
                    new_bounds = self._compute_tight_bounds(problem, i, k, state)  # type: ignore
                    if new_bounds.lower_bound > state.lb[k]:
                        state.lb[k] = new_bounds.lower_bound
                        changed = True
                        changed_mask[k] = True
                    if new_bounds.upper_bound < state.ub[k]:
                        state.ub[k] = new_bounds.upper_bound
                        changed = True
                        changed_mask[k] = True
                    if state.lb[k] > state.ub[k]:
                        return (
                            False,
                            np.empty(0, dtype=np.int64),
                            np.empty(0, dtype=np.float64),
                            np.empty(0, dtype=np.float64),
                        )

        changed_indices: npt.NDArray[np.int64] = np.where(changed_mask)[0]
        changed_lb = state.lb[changed_indices]
        changed_ub = state.ub[changed_indices]
        return True, changed_indices, changed_lb, changed_ub
