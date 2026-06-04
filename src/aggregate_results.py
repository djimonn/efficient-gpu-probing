from dataclasses import dataclass
from typing import Optional, cast

from seaborn.objects import Path

from probe_metrics import ProbeMetrics
import numpy as np
import matplotlib.pyplot as plt
from probe_metrics import ProbeMetrics
from pathlib import Path
import matplotlib.pyplot as plt

from type_aliases import ImplementationType


@dataclass(frozen=True)
class ProbeKey:
    instance_name: str
    var_index: int
    probe_lower_bound: float
    probe_upper_bound: float
    implementation: ImplementationType


def match_probes(
    probes: list[ProbeMetrics],
) -> dict[ProbeKey, tuple[ProbeMetrics, ProbeMetrics]]:
    probe_dict: dict[
        ProbeKey, tuple[Optional[ProbeMetrics], Optional[ProbeMetrics]]
    ] = {}
    for probe in probes:
        key = ProbeKey(
            instance_name=probe.instance_name,
            var_index=probe.var_index,
            probe_lower_bound=probe.probe_lower_bound,
            probe_upper_bound=probe.probe_upper_bound,
            implementation=probe.implementation,
        )
        if key not in probe_dict:
            probe_dict[key] = (None, None)
        if probe.implementation == "naiv":
            probe_dict[key] = (probe, probe_dict[key][1])
        else:
            probe_dict[key] = (probe_dict[key][0], probe)
    # Filter out keys where we don't have both implementations
    matched_probes: dict[ProbeKey, tuple[ProbeMetrics, ProbeMetrics]] = {}
    for key, (naiv_probe, advanced_probe) in probe_dict.items():
        if naiv_probe is not None and advanced_probe is not None:
            matched_probes[key] = (naiv_probe, advanced_probe)
        else:
            raise ValueError(
                f"Missing probe for key {key}: naiv={naiv_probe}, advanced={advanced_probe}"
            )
    return matched_probes


def parse_file(content: str) -> list[ProbeMetrics]:
    # The content is expected to be a single line CSV with the following columns:
    # instance_name,num_vars,num_integer_vars,var_index,probe_lower_bound,probe_upper_bound,is_feasible,implementation,duration_ms,num_changed_bounds,result_copied_bytes
    # The first line is the header, so we skip it.
    lines = content.strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    if len(header) != 11:
        raise ValueError("File header does not have the expected number of columns.")
    probe_metrics: list[ProbeMetrics] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split(",")
        if len(values) != 11:
            raise ValueError(
                f"Line does not have the expected number of columns: {line}"
            )

        def parse_implementation(s: str) -> ImplementationType:
            if s not in ["naiv", "advanced"]:
                raise ValueError(f"Invalid implementation: {s}")
            return cast(ImplementationType, s)

        probe_metrics.append(
            ProbeMetrics(
                instance_name=values[0],
                num_vars=int(values[1]),
                num_integer_vars=int(values[2]),
                var_index=int(values[3]),
                probe_lower_bound=float(values[4]),
                probe_upper_bound=float(values[5]),
                is_feasible=values[6].lower() == "true",
                implementation=parse_implementation(values[7]),
                duration_ms=float(values[8]),
                num_changed_bounds=int(values[9]),
                result_copied_bytes=int(values[10]),
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
) -> dict[str, tuple[list[ProbeMetrics], list[ProbeMetrics]]]:
    instance_metrics: dict[str, tuple[list[ProbeMetrics], list[ProbeMetrics]]] = {}
    for probe in probes:
        if probe.instance_name not in instance_metrics:
            instance_metrics[probe.instance_name] = ([], [])
        if probe.implementation == "naiv":
            instance_metrics[probe.instance_name][0].append(probe)
        else:
            instance_metrics[probe.instance_name][1].append(probe)
    return instance_metrics


def scatter_and_regression(naiv_x, naiv_y, advanced_x, advanced_y, x_label) -> None:  # type: ignore
    # Linear regression lines
    naiv_coeffs = np.polyfit(naiv_x, naiv_y, deg=1)  # type: ignore
    advanced_coeffs = np.polyfit(advanced_x, advanced_y, deg=1)  # type: ignore

    naiv_line = np.poly1d(naiv_coeffs)  # type: ignore
    advanced_line = np.poly1d(advanced_coeffs)  # type: ignore

    x_line = np.linspace(min(min(naiv_x), min(advanced_x)), max(max(naiv_x), max(advanced_x)), 100)  # type: ignore
    plt.figure()  # type: ignore
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

    matched_probes = match_probes(all_probes)

    print(matched_probes)
    return

    instance_metrics: dict[str, tuple[list[ProbeMetrics], list[ProbeMetrics]]] = (
        aggregate_probes(all_probes)
    )  # instance : ([naiv_probes], [advanced_probes])

    if len([probe for probe in all_probes if probe.implementation == "naiv"]) == 0:
        raise ValueError("No GPU naive probes found in the data.")
    if len([probe for probe in all_probes if probe.implementation == "advanced"]) == 0:
        raise ValueError("No GPU advanced probes found in the data.")

    # Avg times
    total_time_naiv = sum(
        probe.duration_ms for probe in all_probes if probe.implementation == "naiv"
    )
    total_time_advanced = sum(
        probe.duration_ms for probe in all_probes if probe.implementation == "advanced"
    )
    avg_time_naiv = total_time_naiv / len(
        [probe for probe in all_probes if probe.implementation == "naiv"]
    )
    avg_time_advanced = total_time_advanced / len(
        [probe for probe in all_probes if probe.implementation == "advanced"]
    )
    print(f"Average time for GPU naive: {avg_time_naiv:.2f} ms")
    print(f"Average time for GPU advanced: {avg_time_advanced:.2f} ms")

    # naiv/advanced schneller in % Instanzen
    instance_avg_times: dict[str, tuple[float, float]] = {}
    for instance_name, (naiv_probes, advanced_probes) in instance_metrics.items():
        avg_naiv = sum(probe.duration_ms for probe in naiv_probes) / len(naiv_probes)
        avg_advanced = sum(probe.duration_ms for probe in advanced_probes) / len(
            advanced_probes
        )
        instance_avg_times[instance_name] = (avg_naiv, avg_advanced)
    naiv_faster_count = sum(
        1
        for avg_naiv, avg_advanced in instance_avg_times.values()
        if avg_naiv < avg_advanced
    )
    advanced_faster_count = sum(
        1
        for avg_naiv, avg_advanced in instance_avg_times.values()
        if avg_advanced < avg_naiv
    )
    total_instances = len(instance_avg_times)
    print(
        f"GPU naive faster in {naiv_faster_count} out of {total_instances} instances ({(naiv_faster_count/total_instances)*100:.2f}%)"
    )
    print(
        f"GPU advanced faster in {advanced_faster_count} out of {total_instances} instances ({(advanced_faster_count/total_instances)*100:.2f}%)"
    )

    # Korrelation result_copied_bytes und Laufzeit
    naiv_x = np.array(
        [
            naiv_probe.result_copied_bytes
            for naiv_probe in all_probes
            if naiv_probe.implementation == "naiv"
        ]
    )
    advanced_x = np.array(
        [
            advanced_probe.result_copied_bytes
            for advanced_probe in all_probes
            if advanced_probe.implementation == "advanced"
        ]
    )
    naiv_y = np.array(
        [
            naiv_probe.duration_ms
            for naiv_probe in all_probes
            if naiv_probe.implementation == "naiv"
        ]
    )
    advanced_y = np.array(
        [
            advanced_probe.duration_ms
            for advanced_probe in all_probes
            if advanced_probe.implementation == "advanced"
        ]
    )
    scatter_and_regression(
        naiv_x, naiv_y, advanced_x, advanced_y, "Result copied bytes"
    )


if __name__ == "__main__":
    main()
