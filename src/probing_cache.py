# pyright: basic
import math
from typing import Callable, Generator, Tuple, cast
import numpy as np
import numpy.typing as npt

from bound_interval import BoundInterval
from cache_entry import CacheEntry
from propagation_result import PropagationResult
from propagation_state import PropagationState
from type_aliases import VarIndex

from milp_problem import MILPProblem

_NAIVE_GPU_KERNELS = None


def _get_naive_gpu_kernels() -> Tuple[Callable[..., None], Callable[..., None]]:
    global _NAIVE_GPU_KERNELS
    if _NAIVE_GPU_KERNELS is not None:
        return _NAIVE_GPU_KERNELS

    try:
        from numba import cuda  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The naive GPU propagator requires numba with CUDA support. "
            "Install numba in the environment used to run this project."
        ) from exc

    if not cuda.is_available():
        raise RuntimeError(
            "Numba is installed, but no CUDA-capable GPU is available to this process."
        )

    @cuda.jit
    def check_row_infeasibility_kernel(
        csr_indptr,
        csr_indices,
        csr_data,
        b,
        lb,
        ub,
        is_infeasible,
    ):
        i = cuda.grid(1)
        if i >= b.size:
            return

        L_min = 0.0
        for ptr in range(csr_indptr[i], csr_indptr[i + 1]):  # type: ignore
            k = csr_indices[ptr]
            a_ik = csr_data[ptr]
            if a_ik == 0:
                continue
            elif a_ik > 0:
                L_min += a_ik * lb[k]
            else:
                L_min += a_ik * ub[k]

        if L_min > b[i]:
            cuda.atomic.max(is_infeasible, 0, 1)  # type: ignore

    @cuda.jit
    def propagate_variables_kernel(
        csr_indptr,
        csr_indices,
        csr_data,
        csc_indptr,
        csc_indices,
        csc_data,
        b,
        lb,
        ub,
        is_integer,
        lb_next,
        ub_next,
        changed,
        infeasible,
    ):
        k = cuda.grid(1)
        if k >= lb.size:
            return

        old_lb = lb[k]
        old_ub = ub[k]
        new_lb = old_lb
        new_ub = old_ub

        for col_ptr in range(csc_indptr[k], csc_indptr[k + 1]):  # type: ignore
            i = csc_indices[col_ptr]
            a_ik = csc_data[col_ptr]

            l_min_except_k = 0.0
            for row_ptr in range(csr_indptr[i], csr_indptr[i + 1]):
                j = csr_indices[row_ptr]
                if j == k:
                    continue

                a_ij = csr_data[row_ptr]
                if a_ij == 0:
                    continue
                elif a_ij > 0:
                    l_min_except_k += a_ij * lb[j]
                else:
                    l_min_except_k += a_ij * ub[j]

            candidate = (b[i] - l_min_except_k) / a_ik
            if a_ik > 0.0:
                if candidate < new_ub:
                    new_ub = candidate
            elif a_ik < 0.0:
                if candidate > new_lb:
                    new_lb = candidate

        if is_integer[k]:
            if not math.isinf(new_lb):
                new_lb = math.ceil(new_lb)
            if not math.isinf(new_ub):
                new_ub = math.floor(new_ub)

        if new_lb > new_ub:
            cuda.atomic.max(infeasible, 0, 1)  # type: ignore

        lb_next[k] = new_lb
        ub_next[k] = new_ub

        if new_lb > old_lb or new_ub < old_ub:
            cuda.atomic.max(changed, 0, 1)  # type: ignore

    _NAIVE_GPU_KERNELS = (check_row_infeasibility_kernel, propagate_variables_kernel)
    return _NAIVE_GPU_KERNELS


class ProbingCache:
    def __init__(self, problem: MILPProblem):
        self.problem = problem
        self.probe_results: dict[Tuple[VarIndex, BoundInterval], CacheEntry] = {}

    def probe_gpu(self, var_index: VarIndex) -> None:
        default_interval = BoundInterval(self.problem.original_lb[var_index], self.problem.original_ub[var_index])
        for probe_interval in self._split_interval(default_interval):
            extended_problem = self.problem.extend_with_constraint(var_index, probe_interval)
            propagation_result = self._propagate_until_fixpoint_naiv_GPU(extended_problem)
            print(f"moin meister: {propagation_result}")
            _naive_cache_entry = self._build_cache_entry_by_host_scan(
                propagation_result
            )
            self.probe_results[(var_index, probe_interval)] = _naive_cache_entry
            

    def probe(self, var_index: VarIndex) -> None:
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        for probe_interval in self._split_interval(default_interval):
            # Basic probing techniques (see "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh)
            extended_problem = self.problem.extend_with_constraint(
                var_index, probe_interval
            )

            # naive approach: scan all bounds and check if any bound has changed/improved.
            propagation_result = self._propagate_until_fixpoint_naiv(extended_problem)
            _naive_cache_entry = self._build_cache_entry_by_host_scan(
                propagation_result
            )

            # advanced approach: only return the bounds that have changed/improved
            (
                is_feasible,
                changed_indices,
                changed_lb,
                changed_ub,
            ) = self._propagate_until_fixpoint_advanced(extended_problem)
            advanced_cache_entry = self._build_cache_entry_from_compacted_bounds(
                is_feasible, changed_indices, changed_lb, changed_ub
            )

            self.probe_results[(var_index, probe_interval)] = advanced_cache_entry

    def _build_cache_entry_from_compacted_bounds(
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

    def _build_cache_entry_by_host_scan(self, result: PropagationResult) -> CacheEntry:
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

    def _propagate_until_fixpoint_advanced(self, problem: MILPProblem) -> tuple[
        bool,
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
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
                    # for k in range(problem.num_variables):
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

    def _propagate_until_fixpoint_naiv_GPU(
        self, problem: MILPProblem
    ) -> PropagationResult:
        (
            check_row_infeasibility_kernel,
            propagate_variables_kernel,
        ) = _get_naive_gpu_kernels()
        from numba import cuda

        A_csc = problem.A.tocsc()

        csr_indptr = np.ascontiguousarray(problem.A.indptr, dtype=np.int64)
        csr_indices = np.ascontiguousarray(problem.A.indices, dtype=np.int64)
        csr_data = np.ascontiguousarray(problem.A.data, dtype=np.float64)
        csc_indptr = np.ascontiguousarray(A_csc.indptr, dtype=np.int64)
        csc_indices = np.ascontiguousarray(A_csc.indices, dtype=np.int64)
        csc_data = np.ascontiguousarray(A_csc.data, dtype=np.float64)
        rhs = np.ascontiguousarray(problem.b, dtype=np.float64)
        initial_lb = np.ascontiguousarray(problem.original_lb, dtype=np.float64)
        initial_ub = np.ascontiguousarray(problem.original_ub, dtype=np.float64)
        integer_flags = np.ascontiguousarray(problem.is_integer, dtype=np.bool_)

        num_constraints = rhs.size
        num_variables = initial_lb.size

        d_csr_indptr = cuda.to_device(csr_indptr)
        d_csr_indices = cuda.to_device(csr_indices)
        d_csr_data = cuda.to_device(csr_data)
        d_csc_indptr = cuda.to_device(csc_indptr)
        d_csc_indices = cuda.to_device(csc_indices)
        d_csc_data = cuda.to_device(csc_data)
        d_b = cuda.to_device(rhs)
        d_is_integer = cuda.to_device(integer_flags)

        d_lb = cuda.to_device(initial_lb)
        d_ub = cuda.to_device(initial_ub)
        d_lb_next = cuda.device_array_like(d_lb)
        d_ub_next = cuda.device_array_like(d_ub)

        zero_flag = np.zeros(1, dtype=np.int32)
        d_changed = cuda.to_device(zero_flag)
        d_infeasible = cuda.to_device(zero_flag)

        threads_per_block = 128
        row_blocks = max(1, math.ceil(num_constraints / threads_per_block))
        var_blocks = max(1, math.ceil(num_variables / threads_per_block))

        # The host controls the fixpoint loop. Each kernel launch computes one
        # synchronous propagation round from current bounds into next bounds.
        max_iterations = max(1000, 2 * (num_constraints + num_variables))
        for _ in range(max_iterations):
            d_changed.copy_to_device(zero_flag)
            d_infeasible.copy_to_device(zero_flag)

            check_row_infeasibility_kernel[row_blocks, threads_per_block](  # type: ignore
                d_csr_indptr,
                d_csr_indices,
                d_csr_data,
                d_b,
                d_lb,
                d_ub,
                d_infeasible,
            )
            cuda.synchronize()
            if int(d_infeasible.copy_to_host()[0]) != 0:
                return (
                    False,
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )  # type: ignore

            propagate_variables_kernel[var_blocks, threads_per_block](  # type: ignore
                d_csr_indptr,
                d_csr_indices,
                d_csr_data,
                d_csc_indptr,
                d_csc_indices,
                d_csc_data,
                d_b,
                d_lb,
                d_ub,
                d_is_integer,
                d_lb_next,
                d_ub_next,
                d_changed,
                d_infeasible,
            )
            cuda.synchronize()

            if int(d_infeasible.copy_to_host()[0]) != 0:
                return PropagationResult(is_feasible=False)
                # return (
                #     False,
                #     np.empty(0, dtype=np.float64),
                #     np.empty(0, dtype=np.float64),
                # )  # type: ignore

            changed = int(d_changed.copy_to_host()[0]) != 0
            d_lb, d_lb_next = d_lb_next, d_lb
            d_ub, d_ub_next = d_ub_next, d_ub
            if not changed:
                return PropagationResult(True, d_lb.copy_to_host(), d_ub.copy_to_host()) # type: ignore
                # return True, d_lb.copy_to_host(), d_ub.copy_to_host()  # type: ignore

        raise RuntimeError(
            f"GPU propagation did not converge after {max_iterations} iterations."
        )

    def _propagate_until_fixpoint_naiv(self, problem: MILPProblem) -> PropagationResult:

        def row_nonzeros(i: int) -> Generator[VarIndex, None, None]:
            start = problem.A.indptr[i]  # type: ignore
            end = problem.A.indptr[i + 1]  # type: ignore
            for ptr in range(start, end):  # type: ignore
                yield problem.A.indices[ptr]  # type: ignore

        state = PropagationState(problem.original_lb.copy(), problem.original_ub.copy())

        changed = True
        while changed:
            changed = False
            for i in range(problem.num_constraints):
                if problem.constraint_is_infeasible(i, state):
                    return PropagationResult(is_feasible=False)
                for k in row_nonzeros(i):
                    # for k in range(problem.num_variables):
                    new_bounds = self._compute_tight_bounds(problem, i, k, state)  # type: ignore
                    if new_bounds.lower_bound > new_bounds.upper_bound:
                        return PropagationResult(is_feasible=False)
                    if new_bounds.lower_bound > state.lb[k]:
                        state.lb[k] = new_bounds.lower_bound
                        changed = True
                        # cache_entry.var_bounds[k] = BoundInterval(
                        #     state.lb[k], state.ub[k]
                        # )
                    if new_bounds.upper_bound < state.ub[k]:
                        state.ub[k] = new_bounds.upper_bound
                        changed = True
                        # cache_entry.var_bounds[k] = BoundInterval(
                        #     state.lb[k], state.ub[k]
                        # )
        return PropagationResult(is_feasible=True, lb=state.lb, ub=state.ub)

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
