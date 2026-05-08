from dataclasses import dataclass
import scipy.sparse as sp
import numpy.typing as npt
import numpy as np


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
    def from_miplib_instance(cls, instance: str) -> "MILPProblem":
        pass
