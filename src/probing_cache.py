import math
from typing import Generator, Tuple, cast

from bound_interval import BoundInterval
from cache_entry import CacheEntry
from propagation_state import PropagationState
from type_aliases import VarIndex

from milp_problem import MILPProblem


class ProbingCache:
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.probe_results: dict[Tuple[VarIndex, BoundInterval], CacheEntry] = {}

    def probe(self, var_index: VarIndex) -> None:
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        for probe_interval in self._split_interval(default_interval):
            # Basic probing techniques (see "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh)
            extended_problem = self.problem.extend_with_constraint(
                var_index, probe_interval
            )
            self.probe_results[(var_index, probe_interval)] = (
                self._propagate_until_fixpoint(extended_problem)
            )

    def _compute_tight_bounds(
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

    def _propagate_until_fixpoint(self, problem: MILPProblem) -> CacheEntry:
        state = PropagationState(problem.original_lb.copy(), problem.original_ub.copy())
        cache_entry = CacheEntry(True)

        def row_nonzeros(i: int) -> Generator[VarIndex, None, None]:
            start = problem.A.indptr[i]  # type: ignore
            end = problem.A.indptr[i + 1]  # type: ignore
            for ptr in range(start, end):  # type: ignore
                yield problem.A.indices[ptr]  # type: ignore

        changed = True
        while changed:
            changed = False
            for i in range(problem.num_constraints):
                if problem.constraint_is_infeasible(i, state):
                    return CacheEntry(False)
                for k in row_nonzeros(i):
                    # for k in range(problem.num_variables):
                    new_bounds = self._compute_tight_bounds(problem, i, k, state)  # type: ignore
                    if new_bounds.lower_bound > new_bounds.upper_bound:
                        return CacheEntry(False)
                    if new_bounds.lower_bound > state.lb[k]:
                        state.lb[k] = new_bounds.lower_bound
                        changed = True
                        cache_entry.var_bounds[k] = BoundInterval(
                            state.lb[k], state.ub[k]
                        )
                    if new_bounds.upper_bound < state.ub[k]:
                        state.ub[k] = new_bounds.upper_bound
                        changed = True
                        cache_entry.var_bounds[k] = BoundInterval(
                            state.lb[k], state.ub[k]
                        )
        return cache_entry

    def _split_interval(
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
