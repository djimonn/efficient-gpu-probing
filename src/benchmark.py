from milp_problem import MILPProblem
from probe_metrics import ProbeMetrics
from probing_cache.probing_cache import ProbingCache
from pathlib import Path


def main():
    directory = Path("data/MIPLIB2017_benchmark_set")
    _metrics: list[ProbeMetrics] = []
    for file in directory.iterdir():
        problem = MILPProblem.from_mps_file(name=file.stem, path=str(file))
        # probing_cache_naiv = ProbingCache(problem)
        # probing_cache_advanced = ProbingCache(problem)
        # for var_index in range(problem.num_variables):
        #     if problem.is_integer[var_index]:
        #         metrics.append(probing_cache_naiv.probe_gpu_naiv(var_index))
        #         metrics.append(probing_cache_advanced.probe_gpu_advanced(var_index))
    return

    problem = MILPProblem.from_mps_file(name="test", path="data/test.mps")
    print(problem.is_integer)
    probing_cache = ProbingCache(problem)
    for var_index in range(problem.num_variables):
        if problem.is_integer[var_index] or problem.name == "test":
            probing_cache.probe(var_index)
    print(probing_cache.probe_results)


if __name__ == "__main__":
    main()
