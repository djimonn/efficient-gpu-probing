# pyright: reportPrivateUsage=false
import numpy as np
import pytest
import scipy.sparse as sp  # type: ignore

from milp_problem import MILPProblem


def test_from_mps_file():
    # unknown instance should raise ValueError
    with pytest.raises(ValueError):
        problem = MILPProblem.from_mps_file("gibtsnicht", "nonexistent.mps.gz")

    # MIPLIB2017 instance should be loaded successfully
    air05 = "air05"
    air05_path = f"data/MIPLIB2017_benchmark_set/{air05}.mps.gz"
    problem = MILPProblem.from_mps_file(air05, air05_path)
    assert problem is not None
    assert problem.name == air05

    test_name = "test"
    problem = MILPProblem.from_mps_file(test_name, "data/test.mps")
    assert problem is not None
    assert problem.name == test_name
    assert problem.A.shape == (4, 3)
    exptected_A = sp.csr_matrix([[1, 1, 0], [-1, 0, -1], [0, -1, 1], [0, 1, -1]])
    assert (problem.A != exptected_A).nnz == 0  # type: ignore
    assert problem.b.shape == (4,)
    assert (problem.b == [5, -10, 7, -7]).all()
    assert problem.lb.shape == (3,)
    assert problem.ub.shape == (3,)
    assert (problem.lb == [0, -1, 0]).all()
    assert (problem.ub == [4, 1, np.inf]).all()


def test_L_min():
    # example problem:
    # max ... (objective function doesn't matter for this test / us)
    # s.t.  2x0 + 3x1 - x2 <= 8  (0)
    #       -x0 + 2x1 + x2 <= -1 (1)
    #       x0  +  x1      <= 20 (2)
    #       0 <= x0 <= 1
    #       0 <= x1 <= 5
    #       1 <= x2 <= 4
    A = sp.csr_matrix(
        [
            [2, 3, -1],
            [-1, 2, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    b = np.array([8, -1, 20], dtype=float)
    lb = np.array([0, 0, 1], dtype=float)
    ub = np.array([1, 5, 4], dtype=float)
    is_integer = np.array([False, False, False], dtype=bool)
    problem = MILPProblem("test", A, b, lb, ub, is_integer)

    ### no except_k ###
    # L_min for constraint 0 should be 2*0 + 3*0 - 1*4 = -4
    assert problem._L_min(0) == -4
    # L_min for constraint 1 should be -1*1 + 2*0 + 1*1 = 0
    assert problem._L_min(1) == 0
    # L_min for constraint 2 should be 1*0 + 1*0 = 0
    assert problem._L_min(2) == 0

    ### with except_k ###
    # L_min for constraint 0 except k=0 should be 3*0 - 1*4 = -4
    assert problem._L_min(0, except_k=0) == -4
    # L_min for constraint 0 except k=1 should be 2*0 - 1*4 = -4
    assert problem._L_min(0, except_k=1) == -4
    # L_min for constraint 0 except k=2 should be 2*0 + 3*0 = 0
    assert problem._L_min(0, except_k=2) == 0

    # L_min for constraint 1 except k=0 should be 2*0 + 1*1 = 1
    assert problem._L_min(1, except_k=0) == 1
    # L_min for constraint 1 except k=1 should be -1*1 + 1*1 = 0
    assert problem._L_min(1, except_k=1) == 0
    # L_min for constraint 1 except k=2 should be -1*1 + 2*0 = -1
    assert problem._L_min(1, except_k=2) == -1

    # L_min for constraint 2 except k=0 should be 0*1 = 0
    assert problem._L_min(2, except_k=0) == 0
    # L_min for constraint 2 except k=1 should be 1*0 = 0
    assert problem._L_min(2, except_k=1) == 0
    # L_min for constraint 2 except k=2 should be 1*0 + 1*0 = 0
    assert problem._L_min(2, except_k=2) == 0


def test_L_max():
    # example problem:
    # max ... (objective function doesn't matter for this test / us)
    # s.t.  2x0 + 3x1 - x2 <= 8  (0)
    #       -x0 + 2x1 + x2 <= -1 (1)
    #       x0  +  x1      <= 20 (2)
    #       0 <= x0 <= 1
    #       0 <= x1 <= 5
    #       1 <= x2 <= 4
    A = sp.csr_matrix(
        [
            [2, 3, -1],
            [-1, 2, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    b = np.array([8, -1, 20], dtype=float)
    lb = np.array([0, 0, 1], dtype=float)
    ub = np.array([1, 5, 4], dtype=float)
    is_integer = np.array([False, False, False], dtype=bool)
    problem = MILPProblem("test", A, b, lb, ub, is_integer)

    ### no except_k ###
    # L_max for constraint 0 should be 2*1 + 3*5 - 1*1 = 16
    assert problem._L_max(0) == 16
    # L_max for constraint 1 should be -1*0 + 2*5 + 1*4 = 14
    assert problem._L_max(1) == 14
    # L_max for constraint 2 should be 1*1 + 1*5 = 6
    assert problem._L_max(2) == 6

    ### with except_k ###
    # L_max for constraint 0 except k=0 should be 3*5 - 1*1 = 14
    assert problem._L_max(0, except_k=0) == 14
    # L_max for constraint 0 except k=1 should be 2*1 - 1*1 = 1
    assert problem._L_max(0, except_k=1) == 1
    # L_max for constraint 0 except k=2 should be 2*1 + 3*5 = 17
    assert problem._L_max(0, except_k=2) == 17

    # L_max for constraint 1 except k=0 should be 2*5 + 1*4 = 14
    assert problem._L_max(1, except_k=0) == 14
    # L_max for constraint 1 except k=1 should be -1*0 + 1*4 = 4
    assert problem._L_max(1, except_k=1) == 4
    # L_max for constraint 1 except k=2 should be -1*0 + 2*5 = 10
    assert problem._L_max(1, except_k=2) == 10

    # L_max for constraint 2 except k=0 should be 1*5 = 5
    assert problem._L_max(2, except_k=0) == 5
    # L_max for constraint 2 except k=1 should be 1*1 = 1
    assert problem._L_max(2, except_k=1) == 1
    # L_max for constraint 2 except k=2 should be 1*1 + 1*5 = 6
    assert problem._L_max(2, except_k=2) == 6


def test_constraint_is_infeasible():
    # example problem:
    # max ... (objective function doesn't matter for this test / us)
    # s.t.  2x0 + 3x1 - x2 <= 8  (0)
    #       -x0 + 2x1 + x2 <= -1 (1)
    #       x0  +  x1      <= 20 (2)
    #       0 <= x0 <= 1
    #       0 <= x1 <= 5
    #       1 <= x2 <= 4
    A = sp.csr_matrix(
        [
            [2, 3, -1],
            [-1, 2, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    b = np.array([8, -1, 20], dtype=float)
    lb = np.array([0, 0, 1], dtype=float)
    ub = np.array([1, 5, 4], dtype=float)
    is_integer = np.array([False, False, False], dtype=bool)
    problem = MILPProblem("test", A, b, lb, ub, is_integer)
    assert not problem.constraint_is_infeasible(0)
    assert problem.constraint_is_infeasible(
        1
    )  # L_min for constraint 1 is 0, which is > b[1] = -1, so constraint 1 is infeasible
    assert not problem.constraint_is_infeasible(2)


def test_constraint_is_redundant():
    # example problem:
    # max ... (objective function doesn't matter for this test / us)
    # s.t.  2x0 + 3x1 - x2 <= 8  (0)
    #       -x0 + 2x1 + x2 <= -1 (1)
    #       x0  +  x1      <= 20 (2)
    #       0 <= x0 <= 1
    #       0 <= x1 <= 5
    #       1 <= x2 <= 4
    A = sp.csr_matrix(
        [
            [2, 3, -1],
            [-1, 2, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    b = np.array([8, -1, 20], dtype=float)
    lb = np.array([0, 0, 1], dtype=float)
    ub = np.array([1, 5, 4], dtype=float)
    is_integer = np.array([False, False, False], dtype=bool)
    problem = MILPProblem("test", A, b, lb, ub, is_integer)
    assert not problem.constraint_is_redundant(0)
    assert not problem.constraint_is_redundant(1)
    assert problem.constraint_is_redundant(
        2
    )  # L_max for constraint 2 is 6, which is <= b


def test_tight_bounds():
    # example problem:
    # max ... (objective function doesn't matter for this test / us)
    # s.t.  2x0 + 3x1 - x2 <= 8  (0)
    #       -x0 + 2x1 + x2 <= -1 (1)
    #       x0  +  x1      <= 20 (2)
    #       0 <= x0 <= 1
    #       0 <= x1 <= 5
    #       1 <= x2 <= 4
    A = sp.csr_matrix(
        [
            [2, 3, -1],
            [-1, 2, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    b = np.array([8, -1, 20], dtype=float)
    lb = np.array([0, 0, 1], dtype=float)
    ub = np.array([1, 5, 4], dtype=float)
    is_integer = np.array([False, False, False], dtype=bool)
    problem = MILPProblem("test", A, b, lb, ub, is_integer)
    assert problem.get_tight_upper_bound(0, 1) == min(
        problem.ub[1], (8 - (2 * 0 - 1 * 4)) / 3
    )
    assert problem.get_tight_lower_bound(0, 2) == max(
        problem.lb[2], (2 * 0 + 3 * 0 - 8) / -1
    )
