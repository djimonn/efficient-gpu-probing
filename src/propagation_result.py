from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class PropagationResult:
    is_feasible: bool
    lb: Optional[npt.NDArray[np.float64]] = None
    ub: Optional[npt.NDArray[np.float64]] = None
