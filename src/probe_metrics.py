from dataclasses import dataclass

from type_aliases import ImplementationType


@dataclass(frozen=True)
class ProbeMetrics:
    instance_name: str
    num_vars: int
    num_integer_vars: int
    var_index: int
    probe_lower_bound: float
    probe_upper_bound: float
    is_feasible: bool
    implementation: ImplementationType
    duration_ms: float
    num_changed_bounds: int  # number of bounds that were improved
    result_copied_bytes: int
