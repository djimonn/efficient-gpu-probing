import pytest

from bound_interval import BoundInterval
from milp_problem import MILPProblem
from probing_cache import ProbingCache
from numba import cuda  # type: ignore


@pytest.mark.skipif(not cuda.is_available(), reason="Requires GPU")
def test__propagate_until_fixpoint_advanced_GPU():
    problem = MILPProblem.from_mps_file(
        name="iterative example", path="data/example_p13.mps"
    )
    probing_cache = ProbingCache(problem)
    ### x_1 ###
    probing_cache.probe_gpu(0)
    ## x_1 = 0 ##
    probe_res = probing_cache.probe_results[(0, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_1 = 0
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(0.0)
    # y_2 = 20
    print(f"y_2 = {probe_res.var_bounds[4]}")
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(20.0)
    # y_3 = 5
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(5.0)
    # x_2 = 1
    assert probe_res.var_bounds[1] == BoundInterval.from_single_value(1.0)
    # x_3 = 1
    assert probe_res.var_bounds[2] == BoundInterval.from_single_value(1.0)

    ### x_2 ###
    probing_cache.probe_gpu(1)
    ## x_2 = 0 ##
    probe_res = probing_cache.probe_results[(1, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_2 = 0
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(0.0)
    # y_1 = 15
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(15.0)
    # x_1 = 1
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)

    ### x_3 ###
    probing_cache.probe_gpu(2)
    ## x_3 = 0 ##
    probe_res = probing_cache.probe_results[(2, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_3 = 0
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(0.0)
    # x_1 = 1
    print(f"x_1 = {probe_res.var_bounds[0].to_string()}")
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)


@pytest.mark.skipif(not cuda.is_available(), reason="Requires GPU")
def test_propagate_until_fixpoint_naiv_GPU():
    problem = MILPProblem.from_mps_file(
        name="iterative example", path="data/example_p13.mps"
    )
    probing_cache = ProbingCache(problem)
    ### x_1 ###
    probing_cache.probe_gpu_advanced(0)
    ## x_1 = 0 ##
    probe_res = probing_cache.probe_results[(0, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_1 = 0
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(0.0)
    # y_2 = 20
    print(f"y_2 = {probe_res.var_bounds[4]}")
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(20.0)
    # y_3 = 5
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(5.0)
    # x_2 = 1
    assert probe_res.var_bounds[1] == BoundInterval.from_single_value(1.0)
    # x_3 = 1
    assert probe_res.var_bounds[2] == BoundInterval.from_single_value(1.0)

    ### x_2 ###
    probing_cache.probe_gpu(1)
    ## x_2 = 0 ##
    probe_res = probing_cache.probe_results[(1, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_2 = 0
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(0.0)
    # y_1 = 15
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(15.0)
    # x_1 = 1
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)

    ### x_3 ###
    probing_cache.probe_gpu(2)
    ## x_3 = 0 ##
    probe_res = probing_cache.probe_results[(2, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_3 = 0
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(0.0)
    # x_1 = 1
    print(f"x_1 = {probe_res.var_bounds[0].to_string()}")
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)


def test_iterative_probing():
    # See p.13 of "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh for the example problem
    problem = MILPProblem.from_mps_file(
        name="iterative example", path="data/example_p13.mps"
    )
    probing_cache = ProbingCache(problem)

    ### x_1 ###
    probing_cache.probe(0)
    ## x_1 = 0 ##
    probe_res = probing_cache.probe_results[(0, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_1 = 0
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(0.0)
    # y_2 = 20
    print(f"y_2 = {probe_res.var_bounds[4]}")
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(20.0)
    # y_3 = 5
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(5.0)
    # x_2 = 1
    assert probe_res.var_bounds[1] == BoundInterval.from_single_value(1.0)
    # x_3 = 1
    assert probe_res.var_bounds[2] == BoundInterval.from_single_value(1.0)

    ### x_2 ###
    probing_cache.probe(1)
    ## x_2 = 0 ##
    probe_res = probing_cache.probe_results[(1, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_2 = 0
    assert probe_res.var_bounds[4] == BoundInterval.from_single_value(0.0)
    # y_1 = 15
    assert probe_res.var_bounds[3] == BoundInterval.from_single_value(15.0)
    # x_1 = 1
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)

    ### x_3 ###
    probing_cache.probe(2)
    ## x_3 = 0 ##
    probe_res = probing_cache.probe_results[(2, BoundInterval.from_single_value(0.0))]
    assert probe_res.is_feasible
    # y_3 = 0
    assert probe_res.var_bounds[5] == BoundInterval.from_single_value(0.0)
    # x_1 = 1
    print(f"x_1 = {probe_res.var_bounds[0].to_string()}")
    assert probe_res.var_bounds[0] == BoundInterval.from_single_value(1.0)


def test_probe_trivial():
    problem = MILPProblem.from_mps_file(
        name="trivial test", path="data/trivial_test.mps"
    )
    probing_cache = ProbingCache(problem)

    ## var 0 ##
    probing_cache.probe(0)
    assert len(probing_cache.probe_results) == 2
    # x_0 = 0.0
    assert probing_cache.probe_results[
        (0, BoundInterval.from_single_value(0.0))
    ].is_feasible
    assert (0, BoundInterval.from_single_value(0.0)) in probing_cache.probe_results
    assert (
        1
        not in probing_cache.probe_results[
            (0, BoundInterval.from_single_value(0.0))
        ].var_bounds
    )  # setting x_0 = 0 does not constrain x_1 further, so it should not be in the cache

    # x_0 = 1.0
    assert probing_cache.probe_results[
        (0, BoundInterval.from_single_value(1.0))
    ].is_feasible
    assert (0, BoundInterval.from_single_value(1.0)) in probing_cache.probe_results
    assert (
        probing_cache.probe_results[(0, BoundInterval.from_single_value(1.0))]
    ).var_bounds[1] == BoundInterval.from_single_value(0.0)

    ## var 1 ##
    probing_cache.probe(1)
    assert len(probing_cache.probe_results) == 4
    # x_1 = 0.0
    assert probing_cache.probe_results[
        (1, BoundInterval.from_single_value(0.0))
    ].is_feasible
    assert (1, BoundInterval.from_single_value(0.0)) in probing_cache.probe_results
    assert (
        0
        not in probing_cache.probe_results[
            (1, BoundInterval.from_single_value(0.0))
        ].var_bounds
    )  # setting x_1 = 0 does not constraint x_0 further, so it should not be in the cache
    # x_1 = 1.0
    assert probing_cache.probe_results[
        (1, BoundInterval.from_single_value(1.0))
    ].is_feasible
    assert (1, BoundInterval.from_single_value(1.0)) in probing_cache.probe_results
    assert (
        probing_cache.probe_results[(1, BoundInterval.from_single_value(1.0))]
    ).var_bounds[0] == BoundInterval.from_single_value(0.0)
