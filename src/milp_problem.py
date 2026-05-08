from dataclasses import dataclass
import scipy.sparse as sp
import numpy.typing as npt
import numpy as np
import gurobipy as gp
from pathlib import Path


# For this project, we are only interested in probing, so we don't need the objective function.
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
        self.name = name
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.is_integer = is_integer

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
