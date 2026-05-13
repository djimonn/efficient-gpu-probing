from typing import Optional

from bound_interval import BoundInterval
from type_aliases import VarIndex


class CacheEntry:
    def __init__(self, is_feasible: bool):
        self.is_feasible = is_feasible
        self.bounds: Optional[dict[VarIndex, BoundInterval]] = (
            {} if is_feasible else None
        )

    def update_bounds(self, var_index: VarIndex, new_bounds: BoundInterval) -> None:
        if not self.is_feasible:
            return
        assert self.bounds is not None, "Feasible entries must have bounds"
        if var_index in self.bounds:
            current_bounds = self.bounds[var_index]
            updated_bounds = BoundInterval(
                max(current_bounds.lower_bound, new_bounds.lower_bound),
                min(current_bounds.upper_bound, new_bounds.upper_bound),
            )
            if updated_bounds.lower_bound > updated_bounds.upper_bound:
                self.is_feasible = False
                self.bounds = None
            else:
                self.bounds[var_index] = updated_bounds
        return
