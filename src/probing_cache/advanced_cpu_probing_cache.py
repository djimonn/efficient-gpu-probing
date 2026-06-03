from typing import Any, Generator

from bound_interval import BoundInterval
from probe_metrics import ProbeMetrics
from probing_cache.probing_cache import ProbingCache
from propagation_state import PropagationState
from type_aliases import VarIndex
import numpy as np
import numpy.typing as npt
import time


class AdvancedCPUProbingCache(ProbingCache):

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

        changed_mask = np.zeros(self.problem.num_variables, dtype=np.bool_)

        changed = True
        while changed:
            changed = False
            for i in range(self.problem.num_constraints):
                if self.problem.constraint_is_infeasible(i, state):
                    return (
                        False,
                        np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64),
                    )
                for k in row_nonzeros(i):
                    new_bounds = self.compute_tight_bounds(self.problem, i, k, state)  # type: ignore
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

    def probe(self, var_index: int) -> list[ProbeMetrics]:
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        metrics: list[ProbeMetrics] = []
        for probe_interval in self.split_interval(default_interval):
            start = time.perf_counter()
            (
                is_feasible,
                changed_indices,
                changed_lb,
                changed_ub,
            ) = self.propagate_until_fixpoint((var_index, probe_interval))
            advanced_cache_entry = self.build_cache_entry_from_compacted_bounds(
                is_feasible, changed_indices, changed_lb, changed_ub
            )

            self.probe_results[(var_index, probe_interval)] = advanced_cache_entry
            metrics.append(
                ProbeMetrics(
                    instance_name=self.problem.name,
                    num_vars=self.problem.num_variables,
                    num_integer_vars=self.problem.num_integer_vars,
                    var_index=var_index,
                    probe_lower_bound=probe_interval.lower_bound,
                    probe_upper_bound=probe_interval.upper_bound,
                    is_feasible=is_feasible,
                    implementation="advanced",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    num_changed_bounds=len(changed_indices),  # type: ignore
                    result_copied_bytes=0,  # this is a CPU implementation, so we don't have GPU memory copies
                )
            )
        return metrics
