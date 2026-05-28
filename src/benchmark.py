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
from numba import cuda  # type: ignore


def plot_stuff(metrics: list[ProbeMetrics]) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    plt.savefig(output_dir / f"total_runtime_per_instance_{timestamp}.png", dpi=200)  # type: ignore
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
        plt.savefig(output_dir / f"speedup_per_instance_{timestamp}.png", dpi=200)  # type: ignore


def main():
    metrics: list[ProbeMetrics] = []
    directory = (
        Path("data/MIPLIB2017_benchmark_set") if cuda.is_available() else Path("data")
    )
    for file in directory.iterdir():
        if not file.is_file():
            continue
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
                metrics.append(probing_cache_naiv.probe(var_index))
                metrics.append(probing_cache_advanced.probe(var_index))
    plot_stuff(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
