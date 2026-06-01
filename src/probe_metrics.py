from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeMetrics:
    instance_name: str
    num_vars: int
    duration_ms: float
    num_changed_bounds: int  # number of bounds that were improved
    full_copy: bool  # whether or not full copy of lb/ub arrays from host-device was done (advanced vs naiv)
