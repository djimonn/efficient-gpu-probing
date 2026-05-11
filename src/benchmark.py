from milp_problem import MILPProblem


def main():
    problem = MILPProblem.from_mps_file(name="test", path="data/test.mps")
    # print(problem.lb)
    # print(problem.ub)
    print(problem.L_min(0))


if __name__ == "__main__":
    main()
