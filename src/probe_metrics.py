from dataclasses import dataclass

from milp_problem import MILPProblem


@dataclass(frozen=True)
class ProbeMetrics:
    problem: MILPProblem
    duration_ms: float
    num_changed_bounds: int  # number of bounds that were improved
    full_copy: bool  # whether or not full copy of lb/ub arrays from host-device was done (advanced vs naiv)
