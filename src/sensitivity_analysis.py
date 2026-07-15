"""Run the FTHA Otto-cycle sensitivity study described in the notebook.

The analysis evaluates six ignition timings and twenty engine speeds, writes a
tidy CSV file to ``reports`` and saves the five study charts to ``img``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .FTHA import evaluate_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
IMAGES_DIRECTORY = PROJECT_ROOT / "img"
RESULTS_PATH = REPORTS_DIRECTORY / "sensitivity_analysis.csv"

ENGINE_SPEEDS_RPM = np.linspace(500.0, 10_000.0, 20)
IGNITION_TIMINGS_DEGREES = np.linspace(-120.0, 0.0, 6)


def calculate_sensitivity_results() -> pd.DataFrame:
    """Evaluate all 120 operating points from the notebook study."""
    records: list[dict[str, float]] = []
    for ignition_timing_degrees in IGNITION_TIMINGS_DEGREES:
        for engine_speed_rpm in ENGINE_SPEEDS_RPM:
            metrics = evaluate_operating_point(
                engine_speed_rpm=engine_speed_rpm,
                ignition_timing_degrees=ignition_timing_degrees,
            )
            records.append(
                {
                    "engine_speed_rpm": engine_speed_rpm,
                    "ignition_timing_degrees": ignition_timing_degrees,
                    "thermal_efficiency_percent": (
                        100.0 * metrics.thermal_efficiency
                    ),
                    "net_specific_power_kw_per_kg": (
                        metrics.net_specific_power_kw_per_kg
                    ),
                    "work_consumption_ratio": metrics.work_consumption_ratio,
                    "maximum_pressure_kpa": metrics.maximum_pressure_kpa,
                    "maximum_temperature_k": metrics.maximum_temperature_k,
                }
            )
    return pd.DataFrame.from_records(records)


def _save_chart(
    results: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    vertical_axis_label: str,
    filename: str,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.4), dpi=160)
    for ignition_timing_degrees, angle_results in results.groupby(
        "ignition_timing_degrees",
        sort=True,
    ):
        axis.plot(
            angle_results["engine_speed_rpm"],
            angle_results[value_column],
            linewidth=1.6,
            marker="o",
            markersize=3.0,
            label=f"Ignição = {ignition_timing_degrees:.0f}°",
        )

    axis.set_title(title)
    axis.set_xlabel("Rotação do motor [rpm]")
    axis.set_ylabel(vertical_axis_label)
    axis.set_xlim(ENGINE_SPEEDS_RPM[0], ENGINE_SPEEDS_RPM[-1])
    axis.set_xticks(np.linspace(ENGINE_SPEEDS_RPM[0], ENGINE_SPEEDS_RPM[-1], 6))
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(fontsize="small", ncols=2)
    figure.tight_layout()
    figure.savefig(IMAGES_DIRECTORY / filename, bbox_inches="tight")
    plt.close(figure)


def save_sensitivity_artifacts(results: pd.DataFrame) -> None:
    """Write the tabular results and all charts to the project directories."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.10g")

    chart_definitions = (
        (
            "thermal_efficiency_percent",
            "Eficiência térmica × rotação",
            "Eficiência térmica [%]",
            "thermal_efficiency_vs_engine_speed.png",
        ),
        (
            "net_specific_power_kw_per_kg",
            "Potência líquida específica × rotação",
            "Potência líquida específica [kW/kg]",
            "net_specific_power_vs_engine_speed.png",
        ),
        (
            "work_consumption_ratio",
            "Razão de consumo de trabalho × rotação",
            "Razão de consumo de trabalho [-]",
            "work_consumption_ratio_vs_engine_speed.png",
        ),
        (
            "maximum_pressure_kpa",
            "Pressão máxima × rotação",
            "Pressão máxima [kPa]",
            "maximum_pressure_vs_engine_speed.png",
        ),
        (
            "maximum_temperature_k",
            "Temperatura máxima × rotação",
            "Temperatura máxima [K]",
            "maximum_temperature_vs_engine_speed.png",
        ),
    )
    for value_column, title, vertical_axis_label, filename in chart_definitions:
        _save_chart(
            results,
            value_column=value_column,
            title=title,
            vertical_axis_label=vertical_axis_label,
            filename=filename,
        )


def run_sensitivity_analysis() -> pd.DataFrame:
    """Calculate the case study, persist its artifacts, and return the data."""
    results = calculate_sensitivity_results()
    save_sensitivity_artifacts(results)
    return results


if __name__ == "__main__":
    generated_results = run_sensitivity_analysis()
    print(f"Saved {len(generated_results)} operating points to {RESULTS_PATH}")
    print(f"Saved charts to {IMAGES_DIRECTORY}")
