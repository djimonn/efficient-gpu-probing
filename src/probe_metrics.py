from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeMetrics:
    duration_ms: float
    # num_fixpoint_iterations: int
    num_changed_bounds: int  # number of bounds that were improved
    # num_bound_improvements: int  # how often any bound was improved
    full_copy: bool  # whether or not full copy of lb/ub arrays from host-device was done (advanced vs naiv)
