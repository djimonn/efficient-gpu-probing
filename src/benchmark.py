from milp_problem import MILPProblem
from probe_metrics import ProbeMetrics
from probing_cache.advanced_cpu_probing_cache import AdvancedCPUProbingCache
from probing_cache.advanced_gpu_probing_cache import AdvancedGPUProbingCache
from probing_cache.naiv_cpu_probing_cache import NaivCPUProbingCache
from probing_cache.naiv_gpu_probing_cache import NaivGPUProbingCache
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import sys
from numba import cuda  # type: ignore


def get_run_id() -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    return (
        f"slurm_{slurm_job_id}"
        if slurm_job_id is not None
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )


def metrics_to_dataframe(metrics: list[ProbeMetrics]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_name": [metric.instance_name for metric in metrics],
            "num_vars": [metric.num_vars for metric in metrics],
            "num_integer_vars": [metric.num_integer_vars for metric in metrics],
            "var_index": [metric.var_index for metric in metrics],
            "probe_lower_bound": [metric.probe_lower_bound for metric in metrics],
            "probe_upper_bound": [metric.probe_upper_bound for metric in metrics],
            "is_feasible": [metric.is_feasible for metric in metrics],
            "implementation": [metric.implementation for metric in metrics],
            "duration_ms": [metric.duration_ms for metric in metrics],
            "num_changed_bounds": [metric.num_changed_bounds for metric in metrics],
            "result_copied_bytes": [metric.result_copied_bytes for metric in metrics],
        }
    )


def write_instance_metrics(metrics: list[ProbeMetrics]) -> None:
    if not metrics:
        return

    output_dir = Path("output") / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_name = metrics[0].instance_name
    metrics_to_dataframe(metrics).to_csv(
        output_dir / f"{instance_name}.txt", index=False
    )


def write_instance_error(instance_name: str, error: Exception) -> None:
    output_dir = Path("output") / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{instance_name}.error.txt").open("w") as file:
        file.write(f"{type(error).__name__}: {error}\n")


def benchmark_instance(file: Path) -> list[ProbeMetrics]:
    instance_metrics: list[ProbeMetrics] = []
    problem = MILPProblem.from_mps_file(name=file.stem, path=str(file))
    probing_cache_naiv = (
        NaivGPUProbingCache(problem)
        if cuda.is_available()
        else NaivCPUProbingCache(problem)
    )
    probing_cache_advanced = (
        AdvancedGPUProbingCache(problem)
        if cuda.is_available()
        else AdvancedCPUProbingCache(problem)
    )
    for var_index in range(problem.num_variables):
        if problem.is_integer[var_index]:
            instance_metrics.extend(probing_cache_naiv.probe(var_index))
            instance_metrics.extend(probing_cache_advanced.probe(var_index))
    return instance_metrics


def main():
    _run_id = get_run_id()
    metrics: list[ProbeMetrics] = []
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        directory = (
            Path("data/MIPLIB2017_benchmark_set")
            if cuda.is_available()
            else Path("data")
        )
        files = sorted(directory.iterdir())

    for file in files:
        if not file.is_file():
            continue

        instance_metrics: list[ProbeMetrics] = []
        try:
            instance_metrics = benchmark_instance(file)
        except Exception as error:
            write_instance_metrics(instance_metrics)
            write_instance_error(file.stem, error)
            print(f"Skipping {file.name} after error: {error}")
            continue

        write_instance_metrics(instance_metrics)
        metrics.extend(instance_metrics)

    # plot_stuff(metrics, run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
