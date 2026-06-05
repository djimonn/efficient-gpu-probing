from pathlib import Path
from typing import Optional
import numpy as np
import sys

from milp_problem import MILPProblem
from probing_cache.advanced_gpu_probing_cache import AdvancedGPUProbingCache
from probing_cache.naiv_gpu_probing_cache import NaivGPUProbingCache
from probing_cache.probing_cache import ProbingCache

# call like this: python benchmark_single_probe ['n' (naiv) / 'a' (advanced)]


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Arguments missing")
    path = Path("data/MIPLIB2017_benchmark_set/50v-10.mps.gz")
    problem = MILPProblem.from_mps_file(name="50v-10", path=str(path))
    probing_cache: Optional[ProbingCache] = None
    if sys.argv[1] == "n":
        probing_cache = NaivGPUProbingCache(problem=problem)
    elif sys.argv[1] == "a":
        probing_cache = AdvancedGPUProbingCache(problem=problem)
    else:
        raise ValueError("Unknown argument passed")
    first_int_index = int(np.flatnonzero(problem.is_integer)[0])
    probing_cache.probe(var_index=first_int_index)


if __name__ == "__main__":
    main()
