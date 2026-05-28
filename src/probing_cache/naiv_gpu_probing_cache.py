# pyright: basic
import math
import time
from typing import Any, Callable, Tuple
import numpy as np

from bound_interval import BoundInterval
from probe_metrics import ProbeMetrics
from probing_cache.probing_cache import ProbingCache
from propagation_result import PropagationResult
from type_aliases import VarIndex

_NAIVE_GPU_KERNELS = None


def get_naive_gpu_kernels() -> Tuple[Callable[..., None], Callable[..., None]]:
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


class NaivGPUProbingCache(ProbingCache):

    def propagate_until_fixpoint(
        self, additional_constraint: Tuple[VarIndex, BoundInterval]
    ) -> Any:
        (
            check_row_infeasibility_kernel,
            propagate_variables_kernel,
        ) = get_naive_gpu_kernels()
        from numba import cuda  # type: ignore

        A_csc = self.problem.A.tocsc()

        csr_indptr = np.ascontiguousarray(self.problem.A.indptr, dtype=np.int64)
        csr_indices = np.ascontiguousarray(self.problem.A.indices, dtype=np.int64)
        csr_data = np.ascontiguousarray(self.problem.A.data, dtype=np.float64)
        csc_indptr = np.ascontiguousarray(A_csc.indptr, dtype=np.int64)
        csc_indices = np.ascontiguousarray(A_csc.indices, dtype=np.int64)
        csc_data = np.ascontiguousarray(A_csc.data, dtype=np.float64)
        rhs = np.ascontiguousarray(self.problem.b, dtype=np.float64)
        # add additional constraint
        add_constraint_idx, add_constraint_interval = additional_constraint
        initial_lb = np.ascontiguousarray(
            self.problem.original_lb, dtype=np.float64
        ).copy()
        initial_lb[add_constraint_idx] = max(
            add_constraint_interval.lower_bound, initial_lb[add_constraint_idx]
        )
        initial_ub = np.ascontiguousarray(
            self.problem.original_ub, dtype=np.float64
        ).copy()
        initial_ub[add_constraint_idx] = min(
            add_constraint_interval.upper_bound, initial_ub[add_constraint_idx]
        )
        integer_flags = np.ascontiguousarray(self.problem.is_integer, dtype=np.bool_)

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
                return PropagationResult(False)

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

            changed = int(d_changed.copy_to_host()[0]) != 0
            d_lb, d_lb_next = d_lb_next, d_lb
            d_ub, d_ub_next = d_ub_next, d_ub
            if not changed:
                return PropagationResult(True, d_lb.copy_to_host(), d_ub.copy_to_host())  # type: ignore

        raise RuntimeError(
            f"GPU propagation did not converge after {max_iterations} iterations."
        )

    def probe(self, var_index: VarIndex) -> ProbeMetrics:
        start = time.perf_counter()
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        for probe_interval in self.split_interval(default_interval):
            propagation_result = self.propagate_until_fixpoint(
                (var_index, probe_interval)
            )
            _naive_cache_entry = self.build_cache_entry_by_host_scan(propagation_result)
            self.probe_results[(var_index, probe_interval)] = _naive_cache_entry
        return ProbeMetrics(
            problem=self.problem,
            duration_ms=(time.perf_counter() - start) * 1000,
            num_changed_bounds=(
                len(propagation_result.lb) if propagation_result.lb is not None else 0
            ),
            full_copy=True,
        )
