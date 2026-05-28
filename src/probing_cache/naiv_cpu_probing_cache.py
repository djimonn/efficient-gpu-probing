from typing import Any, Generator

from bound_interval import BoundInterval
from probe_metrics import ProbeMetrics
from probing_cache.probing_cache import ProbingCache
from propagation_result import PropagationResult
from propagation_state import PropagationState
from type_aliases import VarIndex
import time


class NaivCPUProbingCache(ProbingCache):

    def propagate_until_fixpoint(
        self, additional_constraint: tuple[int, BoundInterval]
    ) -> Any:
        def row_nonzeros(i: int) -> Generator[VarIndex, None, None]:
            start = self.problem.A.indptr[i]  # type: ignore
            end = self.problem.A.indptr[i + 1]  # type: ignore
            for ptr in range(start, end):  # type: ignore
                yield self.problem.A.indices[ptr]  # type: ignore

        state = PropagationState(
            self.problem.original_lb.copy(), self.problem.original_ub.copy()
        )
        add_constr_idx, add_constr_bound = additional_constraint
        state.lb[add_constr_idx] = max(
            state.lb[add_constr_idx], add_constr_bound.lower_bound
        )
        state.ub[add_constr_idx] = min(
            state.ub[add_constr_idx], add_constr_bound.upper_bound
        )

        changed = True
        while changed:
            changed = False
            for i in range(self.problem.num_constraints):
                if self.problem.constraint_is_infeasible(i, state):
                    return PropagationResult(is_feasible=False)
                for k in row_nonzeros(i):
                    new_bounds = self.compute_tight_bounds(self.problem, i, k, state)  # type: ignore
                    if new_bounds.lower_bound > new_bounds.upper_bound:
                        return PropagationResult(is_feasible=False)
                    if new_bounds.lower_bound > state.lb[k]:
                        state.lb[k] = new_bounds.lower_bound
                        changed = True
                    if new_bounds.upper_bound < state.ub[k]:
                        state.ub[k] = new_bounds.upper_bound
                        changed = True
        return PropagationResult(is_feasible=True, lb=state.lb, ub=state.ub)

    def probe(self, var_index: int) -> ProbeMetrics:
        start = time.perf_counter()
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        for probe_interval in self.split_interval(default_interval):
            # Basic probing techniques (see "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh)
            # naive approach: scan all bounds and check if any bound has changed/improved.
            propagation_result = self.propagate_until_fixpoint(
                (var_index, probe_interval)
            )
            naive_cache_entry = self.build_cache_entry_by_host_scan(propagation_result)

            self.probe_results[(var_index, probe_interval)] = naive_cache_entry

        return ProbeMetrics(
            problem=self.problem,
            duration_ms=(time.perf_counter() - start) * 1000,
            full_copy=True,
            num_changed_bounds=(len(naive_cache_entry.var_bounds)),  # type: ignore
        )
