from bound_interval import BoundInterval
from milp_problem import MILPProblem
from probing_cache import ProbingCache


def test_probe():
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
    print(
        probing_cache.probe_results[
            (0, BoundInterval.from_single_value(0.0))
        ].to_string()
    )
    assert (
        probing_cache.probe_results[(0, BoundInterval.from_single_value(0.0))]
    ).var_bounds[1] == BoundInterval(0.0, 1.0)
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
        probing_cache.probe_results[(1, BoundInterval.from_single_value(0.0))]
    ).var_bounds[0] == BoundInterval(0.0, 1.0)
    # x_1 = 1.0
    assert probing_cache.probe_results[
        (1, BoundInterval.from_single_value(1.0))
    ].is_feasible
    assert (1, BoundInterval.from_single_value(1.0)) in probing_cache.probe_results
    assert (
        probing_cache.probe_results[(1, BoundInterval.from_single_value(1.0))]
    ).var_bounds[0] == BoundInterval.from_single_value(0.0)
