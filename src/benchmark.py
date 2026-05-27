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


def plot_stuff(metrics: list[ProbeMetrics]) -> None:
    df = pd.DataFrame(
        {
            "duration_ms": [m.duration_ms for m in metrics],
            "num_changed_bounds": [m.num_changed_bounds for m in metrics],
            "implementation": [
                "naive/full copy" if m.full_copy else "advanced/compact"
                for m in metrics
            ],
        }
    )

    sns.boxplot(data=df, x="implementation", y="duration_ms")
    plt.yscale("log")  # type: ignore
    plt.ylabel("Duration [ms]")  # type: ignore
    plt.xlabel("")  # type: ignore
    plt.title("Probe runtime: full bound copy vs compact changed bounds")  # type: ignore
    plt.show()  # type: ignore


def main():
    directory = Path("data/MIPLIB2017_benchmark_set")
    metrics: list[ProbeMetrics] = []
    i = 0
    for file in directory.iterdir():
        if i > 2:
            break
        i += 1
        problem = MILPProblem.from_mps_file(name=file.stem, path=str(file))
        probing_cache_naiv = NaivCPUProbingCache(problem)
        probing_cache_advanced = AdvancedCPUProbingCache(problem)
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
