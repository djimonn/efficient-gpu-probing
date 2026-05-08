import numpy as np
import pytest
import scipy.sparse as sp

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
    assert (problem.A != exptected_A).nnz == 0
    assert problem.b.shape == (4,)
    assert (problem.b == [5, -10, 7, -7]).all()
    assert problem.lb.shape == (3,)
    assert problem.ub.shape == (3,)
    assert (problem.lb == [0, -1, 0]).all()
    assert (problem.ub == [4, 1, np.inf]).all()
