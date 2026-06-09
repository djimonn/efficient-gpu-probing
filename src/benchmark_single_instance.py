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
    best_adv_path = Path("data/MIPLIB2017_benchmark_set/neos-873061.mps.gz")
    best_naiv_path = Path("data/MIPLIB2017_benchmark_set/neos-4387871-tavua.mps.gz")
    if sys.argv[1] not in ["n", "a"]:
        raise ValueError("Unknown argument passed")
    problem = MILPProblem.from_mps_file(name="biggest speedup", path=str(best_adv_path if sys.argv[1] == "a" else best_naiv_path))
    probing_cache: Optional[ProbingCache] = None
    if sys.argv[1] == "n":
        probing_cache = NaivGPUProbingCache(problem=problem)
    elif sys.argv[1] == "a":
        probing_cache = AdvancedGPUProbingCache(problem=problem)
    for var_index in range(problem.num_variables):
        if problem.is_integer[var_index]:
            probing_cache.probe(var_index=var_index)


if __name__ == "__main__":
    main()
