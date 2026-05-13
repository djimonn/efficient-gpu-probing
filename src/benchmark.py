from milp_problem import MILPProblem
from probing_cache import ProbingCache


def main():
    problem = MILPProblem.from_mps_file(name="test", path="data/test.mps")
    probing_cache = ProbingCache(problem)
    for var_index in range(problem.num_variables):
        probing_cache.probe(var_index)
    print(probing_cache.probe_results)


if __name__ == "__main__":
    main()
