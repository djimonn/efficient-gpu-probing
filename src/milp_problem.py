from dataclasses import dataclass
from typing import Optional
import scipy.sparse as sp
import numpy.typing as npt
import numpy as np
import gurobipy as gp
from pathlib import Path
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
        assert (
            A.shape[0] == b.shape[0]
        ), "Number of rows in A must match the length of b"
        self.name = name
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.is_integer = is_integer

    # see paper
    def L_min(self, i: int, except_k: Optional[VarIndex] = None) -> float:
        assert 0 <= i < self.A.shape[0], "Constraint index out of bounds"
        if except_k:
            assert 0 <= except_k < self.A.shape[1], "Variable index out of bounds"
        res = 0.0
        for j in range(self.A.shape[1]):
            if j == except_k:
                continue
            if self.A[i, j] > 0:
                res += self.A[i, j] * self.lb[j]
            else:
                res += self.A[i, j] * self.ub[j]
        return res

    def L_max(self, i: int, except_k: Optional[VarIndex] = None) -> float:
        assert 0 <= i < self.A.shape[0], "Constraint index out of bounds"
        if except_k:
            assert 0 <= except_k < self.A.shape[1], "Variable index out of bounds"
        res = 0.0
        for j in range(self.A.shape[1]):
            if j == except_k:
                continue
            if self.A[i, j] > 0:
                res += self.A[i, j] * self.ub[j]
            else:
                res += self.A[i, j] * self.lb[j]
        return res

    # i is the index of the constraint to check.
    def constraint_is_infeasible(self, i: int) -> bool:
        return self.L_min(i) > self.b[i]

    # i is the index of the constraint to check.
    def constraint_is_redundant(self, i: int) -> bool:
        return self.L_max(i) <= self.b[i]

    # i is the index of the constraint to check
    # k is the index of the variable to check
    def get_tight_upper_bound(self, i: int, k: VarIndex) -> float:
        assert self.A[i, k] > 0, "Coefficient must be positive for this to work"
        return min(self.ub[k], (self.b[i] - self.L_min(i, except_k=k)) / self.A[i, k])

    def get_tight_lower_bound(self, i: int, k: VarIndex) -> float:
        assert self.A[i, k] < 0, "Coefficient must be negative for this to work"
        return max(self.lb[k], (self.L_min(i, except_k=k) - self.b[i]) / self.A[i, k])

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
        rows = []
        bs = []
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
        A_leq = sp.vstack(rows, format="csr")
        b_leq = np.array(bs, dtype=np.float64)
        vars_ = model.getVars()
        lb = np.array([v.LB for v in vars_], dtype=np.float64)
        ub = np.array([v.UB for v in vars_], dtype=np.float64)
        vtypes = np.array([v.VType for v in vars_])

        return MILPProblem(
            name,
            A_leq,
            b_leq,
            lb,
            ub,
            [t == gp.GRB.INTEGER for t in vtypes],
        )
