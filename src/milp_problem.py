from dataclasses import dataclass
from typing import Optional, cast
import scipy.sparse as sp  # type: ignore
import numpy.typing as npt
import numpy as np
import gurobipy as gp
from pathlib import Path
from bound_interval import BoundInterval
from type_aliases import VarIndex

# See "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh for more details on the techniques we will implement in this project.


# For this project, we are only interested in probing, so we don't need the objective function.
# The (non-bound) constraints are stored in the form Ax <= b, where A is a sparse matrix and b is a dense vector.
# Additionally, we store the variable bounds (lb / ub) and whether the variables are integer or continuous (is_integer).
@dataclass
class MILPProblem:
    def __init__(
        self,
        name: str,
        A: sp.csr_matrix,
        b: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
        is_integer: npt.NDArray[np.bool_],
    ):
        self.A = A
        assert (
            self.num_constraints == b.shape[0]
        ), "Number of rows in A must match the length of b"
        self.name = name
        self.b = b
        self.lb = lb
        self.ub = ub
        self.is_integer = is_integer

    @property
    def num_constraints(self) -> int:
        shape = cast(tuple[int, int], self.A.shape)
        return shape[0]

    @property
    def num_variables(self) -> int:
        shape = cast(tuple[int, int], self.A.shape)
        return shape[1]

    def extend_with_constraint(
        self, var_index: VarIndex, bound: BoundInterval
    ) -> "MILPProblem":
        lb = self.lb.copy()
        ub = self.ub.copy()
        lb[var_index] = max(lb[var_index], bound.lower_bound)
        ub[var_index] = min(ub[var_index], bound.upper_bound)
        return MILPProblem(self.name, self.A, self.b, lb, ub, self.is_integer)

    # i is the index of the constraint to operate on.
    def _L_min(self, i: int, except_k: Optional[VarIndex] = None) -> float:
        """
        Computes the minimum value of the left-hand side of constraint i, given the current variable bounds.
        """
        assert 0 <= i < self.num_constraints, "Constraint index out of bounds"
        if except_k:
            assert 0 <= except_k < self.num_variables, "Variable index out of bounds"
        res = 0.0
        for j in range(self.num_variables):
            if j == except_k:
                continue
            a_ij = cast(float, self.A[i, j])
            if self.A[i, j] == 0:
                continue
            elif self.A[i, j] > 0:
                res += a_ij * self.lb[j]
            else:
                res += a_ij * self.ub[j]
        return res

    # i is the index of the constraint to operate on.
    def _L_max(self, i: int, except_k: Optional[VarIndex] = None) -> float:
        """
        Computes the maximum value of the left-hand side of constraint i, given the current variable bounds.
        """
        assert 0 <= i < self.num_constraints, "Constraint index out of bounds"
        if except_k:
            assert 0 <= except_k < self.num_variables, "Variable index out of bounds"
        res = 0.0
        for j in range(self.num_variables):
            if j == except_k:
                continue
            a_ij = cast(float, self.A[i, j])
            if a_ij == 0:
                continue
            elif a_ij > 0:
                res += a_ij * self.ub[j]
            else:
                res += a_ij * self.lb[j]
        return res

    # i is the index of the constraint to check.
    def constraint_is_infeasible(self, i: int) -> bool:
        return self._L_min(i) > self.b[i]

    # i is the index of the constraint to check.
    def constraint_is_redundant(self, i: int) -> bool:
        return self._L_max(i) <= self.b[i]

    # i is the index of the constraint to check
    # k is the index of the variable to check
    def get_tight_upper_bound(self, i: int, k: VarIndex) -> float:
        assert self.A[i, k] > 0, "Coefficient must be positive for this to work"
        a_ik = cast(float, self.A[i, k])
        return min(self.ub[k], (self.b[i] - self._L_min(i, except_k=k)) / a_ik)

    def get_tight_lower_bound(self, i: int, k: VarIndex) -> float:
        assert self.A[i, k] < 0, "Coefficient must be negative for this to work"
        a_ik = cast(float, self.A[i, k])
        return max(self.lb[k], (self._L_min(i, except_k=k) - self.b[i]) / a_ik)

    @classmethod
    def from_mps_file(cls, name: str, path: str) -> "MILPProblem":
        # check if the instance is in the MIPLIB 2017 library
        instance_path = Path(path)
        if not instance_path.exists():
            raise ValueError(f"Path {path} not found")

        model = gp.read(path)
        # print(model.getA())
        A = model.getA()
        constrs = model.getConstrs()
        rhs = np.array([c.RHS for c in constrs], dtype=np.float64)
        sense = np.array([c.Sense for c in constrs])
        rows: list[sp.csr_matrix] = []
        bs: list[np.float64] = []
        for i, s in enumerate(sense):
            row = A.getrow(i)
            if s == "<":
                rows.append(row)
                bs.append(rhs[i])
            elif s == ">":
                rows.append(-row)
                bs.append(-rhs[i])
            elif s == "=":
                rows.append(row)
                bs.append(rhs[i])
                rows.append(-row)
                bs.append(-rhs[i])
            else:
                raise ValueError(f"Unknown constraint sense: {s}")
        A_leq: sp.csr_matrix = sp.vstack(rows, format="csr")  # type: ignore
        b_leq = np.array(bs, dtype=np.float64)
        vars_ = model.getVars()
        lb = np.array([v.LB for v in vars_], dtype=np.float64)
        ub = np.array([v.UB for v in vars_], dtype=np.float64)

        return MILPProblem(
            name,
            A_leq,
            b_leq,
            lb,
            ub,
            np.array([v.VType == gp.GRB.INTEGER for v in vars_], dtype=np.bool_),
        )
