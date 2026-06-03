# pyright: basic
import math
import time
from typing import Any, Callable, Tuple
import numpy as np

from bound_interval import BoundInterval
from compacted_propagation_result import CompactedPropagationResult
from probe_metrics import ProbeMetrics
from probing_cache.naiv_gpu_probing_cache import get_naive_gpu_kernels
from probing_cache.probing_cache import ProbingCache
from type_aliases import VarIndex

_ADVANCED_GPU_KERNELS = None


def _get_advanced_gpu_kernels() -> (
    Tuple[Callable[..., None], Callable[..., None], Callable[..., None]]
):
    global _ADVANCED_GPU_KERNELS
    if _ADVANCED_GPU_KERNELS is not None:
        return _ADVANCED_GPU_KERNELS

    check_row_infeasibility_kernel, _ = get_naive_gpu_kernels()
    from numba import cuda  # type: ignore

    @cuda.jit
    def propagate_variables_with_change_tracking_kernel(
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
        changed_mask,
        changed_count,
        changed_indices,
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

            # There is one thread per variable, so no other thread writes
            # changed_mask[k]. The atomic is only needed for the shared count.
            if changed_mask[k] == 0:
                changed_mask[k] = 1
                pos = cuda.atomic.add(changed_count, 0, 1)  # type: ignore
                changed_indices[pos] = k

    @cuda.jit
    def gather_changed_bounds_kernel(
        changed_indices,
        lb,
        ub,
        compact_indices,
        compact_lb,
        compact_ub,
    ):
        pos = cuda.grid(1)
        if pos >= compact_indices.size:
            return

        k = changed_indices[pos]
        compact_indices[pos] = k
        compact_lb[pos] = lb[k]
        compact_ub[pos] = ub[k]

    _ADVANCED_GPU_KERNELS = (
        check_row_infeasibility_kernel,
        propagate_variables_with_change_tracking_kernel,
        gather_changed_bounds_kernel,
    )
    return _ADVANCED_GPU_KERNELS


class AdvancedGPUProbingCache(ProbingCache):

    def propagate_until_fixpoint(
        self, additional_constraint: Tuple[VarIndex, BoundInterval]
    ) -> CompactedPropagationResult:
        (
            check_row_infeasibility_kernel,
            propagate_variables_with_change_tracking_kernel,
            gather_changed_bounds_kernel,
        ) = _get_advanced_gpu_kernels()
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

        d_changed_mask = cuda.to_device(np.zeros(num_variables, dtype=np.int32))
        d_changed_count = cuda.to_device(zero_flag)
        d_changed_indices = cuda.device_array(num_variables, dtype=np.int64)  # type: ignore

        threads_per_block = 128
        row_blocks = max(1, math.ceil(num_constraints / threads_per_block))
        var_blocks = max(1, math.ceil(num_variables / threads_per_block))

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
                return CompactedPropagationResult(
                    is_feasible=False,
                    changed_indices=np.empty(0, dtype=np.int64),
                    changed_lb=np.empty(0, dtype=np.float64),
                    changed_ub=np.empty(0, dtype=np.float64),
                    result_copied_bytes=0,
                )

            propagate_variables_with_change_tracking_kernel[var_blocks, threads_per_block](  # type: ignore
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
                d_changed_mask,
                d_changed_count,
                d_changed_indices,
            )
            cuda.synchronize()

            if int(d_infeasible.copy_to_host()[0]) != 0:
                return CompactedPropagationResult(
                    is_feasible=False,
                    changed_indices=np.empty(0, dtype=np.int64),
                    changed_lb=np.empty(0, dtype=np.float64),
                    changed_ub=np.empty(0, dtype=np.float64),
                    result_copied_bytes=0,
                )

            changed = int(d_changed.copy_to_host()[0]) != 0
            d_lb, d_lb_next = d_lb_next, d_lb
            d_ub, d_ub_next = d_ub_next, d_ub
            if not changed:
                num_changed = int(d_changed_count.copy_to_host()[0])
                if num_changed == 0:
                    return CompactedPropagationResult(
                        is_feasible=True,
                        changed_indices=np.empty(0, dtype=np.int64),
                        changed_lb=np.empty(0, dtype=np.float64),
                        changed_ub=np.empty(0, dtype=np.float64),
                        result_copied_bytes=0,
                    )

                d_compact_indices = cuda.device_array(num_changed, dtype=np.int64)  # type: ignore
                d_compact_lb = cuda.device_array(num_changed, dtype=np.float64)
                d_compact_ub = cuda.device_array(num_changed, dtype=np.float64)

                compact_blocks = max(1, math.ceil(num_changed / threads_per_block))
                gather_changed_bounds_kernel[compact_blocks, threads_per_block](  # type: ignore
                    d_changed_indices,
                    d_lb,
                    d_ub,
                    d_compact_indices,
                    d_compact_lb,
                    d_compact_ub,
                )
                cuda.synchronize()

                compact_indices_host = d_compact_indices.copy_to_host()
                compact_lb_host = d_compact_lb.copy_to_host()
                compact_ub_host = d_compact_ub.copy_to_host()

                return CompactedPropagationResult(
                    is_feasible=True,
                    changed_indices=compact_indices_host,
                    changed_lb=compact_lb_host,
                    changed_ub=compact_ub_host,
                    result_copied_bytes=compact_indices_host.nbytes
                    + compact_lb_host.nbytes
                    + compact_ub_host.nbytes,
                )  # type: ignore

        raise RuntimeError(
            f"Advanced GPU propagation did not converge after {max_iterations} iterations."
        )

    def probe(self, var_index: VarIndex) -> list[ProbeMetrics]:
        default_interval = BoundInterval(
            self.problem.original_lb[var_index], self.problem.original_ub[var_index]
        )
        metrics: list[ProbeMetrics] = []
        for probe_interval in self.split_interval(default_interval):
            start = time.perf_counter()
            # advanced approach: only return the bounds that have changed/improved
            compact_propagation_result = self.propagate_until_fixpoint(
                (var_index, probe_interval)
            )
            advanced_cache_entry = self.build_cache_entry_from_compacted_bounds(
                compact_propagation_result.is_feasible,
                compact_propagation_result.changed_indices,
                compact_propagation_result.changed_lb,
                compact_propagation_result.changed_ub,
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
                    is_feasible=compact_propagation_result.is_feasible,
                    implementation="advanced",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    num_changed_bounds=len(compact_propagation_result.changed_indices),
                    result_copied_bytes=compact_propagation_result.result_copied_bytes,
                )
            )
        return metrics
