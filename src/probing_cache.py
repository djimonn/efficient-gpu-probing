import math
from typing import Tuple

from bound_interval import BoundInterval
from cache_entry import CacheEntry
from type_aliases import VarIndex

from milp_problem import MILPProblem


class ProbingCache:
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.probe_results: dict[Tuple[VarIndex, BoundInterval], CacheEntry] = {}

    def probe(self, var_index: VarIndex) -> None:
        default_interval = BoundInterval(
            self.problem.lb[var_index], self.problem.ub[var_index]
        )
        for probe_interval in self._split_interval(default_interval):
            # Basic probing techniques (see "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh)
            extended_problem = self.problem.extend_with_constraint(
                var_index, probe_interval
            )
            changed = True
            while (
                changed
            ):  # keep applying probing techniques until no more changes can be made
                changed = False
                for i in range(
                    extended_problem.num_constraints
                ):  # iterate through all constraints
                    if extended_problem.constraint_is_infeasible(
                        i
                    ):  # if any constraint is infeasible, then the probe interval is infeasible
                        self.probe_results[(var_index, probe_interval)] = CacheEntry(
                            False
                        )
                        break

                    else:
                        if (var_index, probe_interval) not in self.probe_results:
                            self.probe_results[(var_index, probe_interval)] = (
                                CacheEntry(True)
                            )

                        for k in range(
                            extended_problem.num_variables
                        ):  # iterate through all variables
                            if k == var_index:
                                continue
                            tight_upper_bound = extended_problem.ub[k]
                            try:
                                tight_upper_bound = (
                                    extended_problem.get_tight_upper_bound(i, k=k)
                                )
                            except (
                                ValueError
                            ):  # get_tight_lower_bound may raise an exception if the sign of the coefficient of variable k in constraint i is 'wrong'
                                pass
                            tight_lower_bound = extended_problem.lb[k]
                            try:
                                tight_lower_bound = (
                                    extended_problem.get_tight_lower_bound(i, k=k)
                                )
                            except (
                                ValueError
                            ):  # get_tight_upper_bound may raise an exception if the sign of the coefficient of variable k in constraint i is 'wrong'
                                pass

                            k_new_bounds = BoundInterval(
                                lower_bound=(
                                    math.ceil(tight_lower_bound)
                                    if self.problem.is_integer[k]
                                    and not math.isinf(tight_lower_bound)
                                    else tight_lower_bound
                                ),
                                upper_bound=(
                                    math.floor(tight_upper_bound)
                                    if self.problem.is_integer[k]
                                    and not math.isinf(tight_upper_bound)
                                    else tight_upper_bound
                                ),
                            )
                            update_bound_result = self.probe_results[
                                (var_index, probe_interval)
                            ].update_bounds(
                                var_index=k,
                                new_bounds=k_new_bounds,
                                initial_bounds=BoundInterval(
                                    self.problem.lb[k], self.problem.ub[k]
                                ),
                            )
                            if update_bound_result:
                                extended_problem = (
                                    extended_problem.extend_with_constraint(
                                        var_index=k, bound=k_new_bounds
                                    )
                                )
                            changed = changed or update_bound_result

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
            mid = math.floor(interval.lower_bound + interval.upper_bound) / 2
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
