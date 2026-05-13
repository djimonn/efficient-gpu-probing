from bound_interval import BoundInterval
from type_aliases import VarIndex


class CacheEntry:
    def __init__(self, is_feasible: bool):
        self.is_feasible = is_feasible
        self.var_bounds: dict[VarIndex, BoundInterval] = {}

    def to_string(self) -> str:
        if not self.is_feasible:
            return "Infeasible"
        else:
            bounds_str = ", ".join(
                f"x_{var_index}: [{bound.lower_bound}, {bound.upper_bound}]"
                for var_index, bound in self.var_bounds.items()  # type: ignore
            )
            return f"Feasible with bounds: {bounds_str}"

    def update_bounds(self, var_index: VarIndex, new_bounds: BoundInterval) -> None:
        if not self.is_feasible:
            print("not feasible bra")
            return
        assert self.var_bounds is not None, "Feasible entries must have bounds"

        if var_index not in self.var_bounds:
            self.var_bounds[var_index] = new_bounds
        current_bounds = self.var_bounds[var_index]
        updated_bounds = BoundInterval(
            max(current_bounds.lower_bound, new_bounds.lower_bound),
            min(current_bounds.upper_bound, new_bounds.upper_bound),
        )
        if updated_bounds.lower_bound > updated_bounds.upper_bound:
            self.is_feasible = False
            self.var_bounds = {}
        else:
            self.var_bounds[var_index] = updated_bounds
