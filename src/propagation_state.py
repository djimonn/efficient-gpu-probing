import numpy.typing as npt
import numpy as np


class PropagationState:
    def __init__(self, lb: npt.NDArray[np.float64], ub: npt.NDArray[np.float64]):
        self.lb = lb
        self.ub = ub
