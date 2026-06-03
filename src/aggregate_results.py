from seaborn.objects import Path

from probe_metrics import ProbeMetrics
import numpy as np
import matplotlib.pyplot as plt
from probe_metrics import ProbeMetrics
from pathlib import Path
import matplotlib.pyplot as plt


def parse_file(content: str) -> list[ProbeMetrics]:
    # The content is expected to be a single line CSV with the following columns:
    # instance_name, num_vars,duration_ms,num_changed_bounds,full_copy
    # The first line is the header, so we skip it.
    lines = content.strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    if len(header) != 5:
        raise ValueError("File header does not have the expected number of columns.")
    probe_metrics: list[ProbeMetrics] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split(",")
        if len(values) != 5:
            raise ValueError(
                f"Line does not have the expected number of columns: {line}"
            )
        probe_metrics.append(
            ProbeMetrics(
                instance_name=values[0],
                num_vars=int(values[1]),
                duration_ms=float(values[2]),
                num_changed_bounds=int(values[3]),
                full_copy=values[4].lower() == "true",
            )
        )
    return probe_metrics


def parse_results() -> list[ProbeMetrics]:
    directory = Path("output/metrics")
    files = sorted(directory.iterdir())
    probe_metrics: list[ProbeMetrics] = []

    for file in files:
        if not file.is_file() or "error" in file.name.lower():
            continue
        try:
            probe_metrics.extend(parse_file(file.read_text()))
        except ValueError as e:
            print(f"Error parsing {file.name}: {e}")
            continue
    return probe_metrics


def aggregate_probes(
    probes: list[ProbeMetrics],
) -> dict[str, tuple[ProbeMetrics, ProbeMetrics]]:
    instance_metrics: dict[str, tuple[ProbeMetrics, ProbeMetrics]] = {}
    for probe in probes:
        if probe.instance_name not in instance_metrics:
            instance_metrics[probe.instance_name] = (None, None)  # type: ignore
        if probe.full_copy:
            instance_metrics[probe.instance_name] = (probe, instance_metrics[probe.instance_name][1])  # type: ignore
        else:
            instance_metrics[probe.instance_name] = (instance_metrics[probe.instance_name][0], probe)  # type: ignore
    return instance_metrics

def scatter_and_regression(naiv_x, naiv_y, advanced_x, advanced_y, x_label) -> None:
    # Linear regression lines
    naiv_coeffs = np.polyfit(naiv_x, naiv_y, deg=1)
    advanced_coeffs = np.polyfit(advanced_x, advanced_y, deg=1)

    naiv_line = np.poly1d(naiv_coeffs)
    advanced_line = np.poly1d(advanced_coeffs)

    x_line = np.linspace(min(min(naiv_x), min(advanced_x)), max(max(naiv_x), max(advanced_x)), 100)  # type: ignore
    plt.figure()
    plt.scatter(naiv_x, naiv_y, label="GPU naive")  # type: ignore
    plt.scatter(advanced_x, advanced_y, label="GPU advanced")  # type: ignore
    plt.plot(x_line, naiv_line(x_line))  # type: ignore
    plt.plot(x_line, advanced_line(x_line))  # type: ignore
    plt.xlabel(x_label)  # type: ignore
    plt.ylabel("Runtime (ms)")  # type: ignore
    plt.legend()  # type: ignore
    plt.savefig(f"output/graphics/{x_label.replace(' ', '_')}_plot.png")  # type: ignore
    plt.close()


def main():
    all_probes: list[ProbeMetrics] = parse_results()
    instance_metrics: dict[str, tuple[ProbeMetrics, ProbeMetrics]] = aggregate_probes(
        all_probes
    )

    if len([probe for probe in all_probes if probe.full_copy]) == 0:
        raise ValueError("No GPU naive probes found in the data.")
    if len([probe for probe in all_probes if not probe.full_copy]) == 0:
        raise ValueError("No GPU advanced probes found in the data.")

    # Avg times
    total_time_naiv = sum(probe.duration_ms for probe in all_probes if probe.full_copy)
    total_time_advanced = sum(
        probe.duration_ms for probe in all_probes if not probe.full_copy
    )
    avg_time_naiv = total_time_naiv / len(
        [probe for probe in all_probes if probe.full_copy]
    )
    avg_time_advanced = total_time_advanced / len(
        [probe for probe in all_probes if not probe.full_copy]
    )
    print(f"Average time for GPU naive: {avg_time_naiv:.2f} ms")
    print(f"Average time for GPU advanced: {avg_time_advanced:.2f} ms")

    # naiv/advanced schneller in % Fälle
    num_naiv_faster = sum(
        1
        for (naiv_probe, advanced_probe) in instance_metrics.values()
        if naiv_probe.duration_ms <= advanced_probe.duration_ms
    )
    num_advanced_faster = len(instance_metrics) - num_naiv_faster
    print(
        f"GPU naive faster in {num_naiv_faster / len(instance_metrics) * 100:.2f}% of cases"
    )
    print(
        f"GPU advanced faster in {num_advanced_faster / len(instance_metrics) * 100:.2f}% of cases"
    )


    # Korrelation Anzahl Bounds Changes und Laufzeit vergleich
    instances_sorted_by_bound_changes = sorted(
        instance_metrics.items(),
        key=lambda x: x[1][0].num_changed_bounds,
    )
    naiv_x = np.array(
        [naiv_probe.num_changed_bounds for _, (naiv_probe, _) in instances_sorted_by_bound_changes]
    )
    advanced_x = np.array([advanced_probe.num_changed_bounds for _, (_, advanced_probe) in instances_sorted_by_bound_changes])
    naiv_y = np.array(
        [naiv_probe.duration_ms for _, (naiv_probe, _) in instances_sorted_by_bound_changes]
    )
    advanced_y = np.array(
        [
            advanced_probe.duration_ms
            for _, (_, advanced_probe) in instances_sorted_by_bound_changes
        ]
    )
    scatter_and_regression(naiv_x, naiv_y, advanced_x, advanced_y, "Number of bound changes")


    # Korrelation Anzahl Variablen und Laufzeit vergleich
    instances_sorted_by_num_vars = sorted(
        instance_metrics.items(),
        key=lambda x: x[1][0].num_vars,
    )
    naiv_x = np.array(
        [naiv_probe.num_vars for _, (naiv_probe, _) in instances_sorted_by_num_vars]
    )
    advanced_x = np.array([advanced_probe.num_vars for _, (_, advanced_probe) in instances_sorted_by_num_vars])
    naiv_y = np.array(
        [naiv_probe.duration_ms for _, (naiv_probe, _) in instances_sorted_by_num_vars]
    )
    advanced_y = np.array(
        [
            advanced_probe.duration_ms
            for _, (_, advanced_probe) in instances_sorted_by_num_vars
        ]
    )
    scatter_and_regression(naiv_x, naiv_y, advanced_x, advanced_y, "Number of variables")


    # Korrelation num_bounds_change/num_vars und Laufzeit vergleich
    instances_sorted_by_num_vars = sorted(
        instance_metrics.items(),
        key=lambda x: x[1][0].num_vars,
    )
    naiv_x = np.array(
        [naiv_probe.num_changed_bounds/naiv_probe.num_vars for _, (naiv_probe, _) in instances_sorted_by_num_vars]
    )
    advanced_x = np.array([advanced_probe.num_changed_bounds/advanced_probe.num_vars for _, (_, advanced_probe) in instances_sorted_by_num_vars])
    naiv_y = np.array(
        [naiv_probe.duration_ms for _, (naiv_probe, _) in instances_sorted_by_num_vars]
    )
    advanced_y = np.array(
        [
            advanced_probe.duration_ms
            for _, (_, advanced_probe) in instances_sorted_by_num_vars
        ]
    )
    scatter_and_regression(naiv_x, naiv_y, advanced_x, advanced_y, "Number of bound changes per variable")


if __name__ == "__main__":
    main()
