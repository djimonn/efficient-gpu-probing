from milp_problem import MILPProblem
from probe_metrics import ProbeMetrics
from probing_cache.advanced_cpu_probing_cache import AdvancedCPUProbingCache
from probing_cache.advanced_gpu_probing_cache import AdvancedGPUProbingCache
from probing_cache.naiv_cpu_probing_cache import NaivCPUProbingCache
from probing_cache.naiv_gpu_probing_cache import NaivGPUProbingCache
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
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
            "instance": [metric.problem.name for metric in metrics],
            "implementation": [
                "GPU naive" if metric.full_copy else "GPU advanced"
                for metric in metrics
            ],
            "duration_ms": [metric.duration_ms for metric in metrics],
            "num_changed_bounds": [metric.num_changed_bounds for metric in metrics],
            "full_copy": [metric.full_copy for metric in metrics],
        }
    )


def write_instance_metrics(metrics: list[ProbeMetrics], run_id: str) -> None:
    if not metrics:
        return

    output_dir = Path("output") / "metrics" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_name = metrics[0].problem.name
    metrics_to_dataframe(metrics).to_csv(output_dir / f"{instance_name}.txt", index=False)


def write_instance_error(instance_name: str, run_id: str, error: Exception) -> None:
    output_dir = Path("output") / "metrics" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{instance_name}.error.txt").open("w") as file:
        file.write(f"{type(error).__name__}: {error}\n")


def plot_stuff(metrics: list[ProbeMetrics], run_id: str) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(
        {
            "instance": [metric.problem.name for metric in metrics],
            "implementation": [
                "GPU naive" if metric.full_copy else "GPU advanced"
                for metric in metrics
            ],
            "duration_ms": [metric.duration_ms for metric in metrics],
            "num_changed_bounds": [metric.num_changed_bounds for metric in metrics],
        }
    )

    if df.empty:
        print("No metrics collected; skipping plots.")
        return

    # total runtime per instance + implementation
    summary = df.groupby(["instance", "implementation"], as_index=False)[
        "duration_ms"
    ].sum()

    # sort by naive runtime
    order = summary[summary["implementation"] == "GPU naive"].sort_values(
        "duration_ms", ascending=False
    )["instance"]

    plt.figure(figsize=(14, 6))  # type: ignore
    sns.barplot(
        data=summary,  # type: ignore
        x="instance",
        y="duration_ms",
        hue="implementation",
        order=order,
    )
    plt.yscale("log")  # type: ignore
    plt.xticks(rotation=70, ha="right")  # type: ignore
    plt.ylabel("Total probing cache construction time [ms]")  # type: ignore
    plt.xlabel("MIPLIB2017 instance")  # type: ignore
    plt.title("Total probing cache construction time per instance")  # type: ignore
    plt.tight_layout()
    plt.savefig(output_dir / f"total_runtime_per_instance_{run_id}.png", dpi=200)  # type: ignore
    plt.close()

    pivot = summary.pivot(
        index="instance", columns="implementation", values="duration_ms"
    )
    if {"GPU naive", "GPU advanced"}.issubset(pivot.columns):
        speedup = (
            pivot.assign(speedup=pivot["GPU naive"] / pivot["GPU advanced"])
            .reset_index()
            .sort_values("speedup", ascending=False)
        )

        plt.figure(figsize=(14, 6))  # type: ignore
        sns.barplot(data=speedup, x="instance", y="speedup")  # type: ignore
        plt.axhline(1.0, color="black", linewidth=1)  # type: ignore
        plt.xticks(rotation=70, ha="right")  # type: ignore
        plt.ylabel("Speedup: naive / advanced")  # type: ignore
        plt.xlabel("MIPLIB2017 instance")  # type: ignore
        plt.title("Advanced GPU probing speedup per instance")  # type: ignore
        plt.tight_layout()
        plt.savefig(output_dir / f"speedup_per_instance_{run_id}.png", dpi=200)  # type: ignore


def main():
    run_id = get_run_id()
    metrics: list[ProbeMetrics] = []
    directory = (
        Path("data/MIPLIB2017_benchmark_set") if cuda.is_available() else Path("data")
    )
    for file in directory.iterdir():
        if not file.is_file():
            continue

        instance_metrics: list[ProbeMetrics] = []
        try:
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
                    instance_metrics.append(probing_cache_naiv.probe(var_index))
                    instance_metrics.append(probing_cache_advanced.probe(var_index))
        except Exception as error:
            write_instance_metrics(instance_metrics, run_id)
            write_instance_error(file.stem, run_id, error)
            print(f"Skipping {file.name} after error: {error}")
            continue

        write_instance_metrics(instance_metrics, run_id)
        metrics.extend(instance_metrics)

    plot_stuff(metrics, run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
