from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompactedPropagationResult:
    is_feasible: bool
    changed_indices: np.ndarray
    changed_lb: np.ndarray
    changed_ub: np.ndarray
    result_copied_bytes: int
