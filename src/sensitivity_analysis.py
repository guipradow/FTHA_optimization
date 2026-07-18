"""Reproduce and persist the complete ``e2.2_case_study.ipynb`` study.

The script first simulates a reference point with the article parameters and
saves its state history, metrics, and five thermodynamic diagrams. It then
evaluates six ignition timings and twenty engine speeds, writes all 120
operating points and a summary
of their extrema, and saves the five sensitivity charts. Every chart uses a
black-and-white-safe visual identity distinct from the article validation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .base_case_analysis import (
    ARTICLE_BASE_PARAMETERS,
    ARTICLE_HEAT_ADDITION_STEP_DEGREES,
    ARTICLE_STROKE_INTERVALS,
)
from .FTHA import (
    CycleResult,
    ModelParameters,
    simulate_cycle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
IMAGES_DIRECTORY = PROJECT_ROOT / "img"
RESULTS_PATH = REPORTS_DIRECTORY / "sensitivity_analysis.csv"
SUMMARY_PATH = REPORTS_DIRECTORY / "sensitivity_analysis_summary.csv"
BASE_CASE_STATES_PATH = REPORTS_DIRECTORY / "case_study_base_case_states.csv"
BASE_CASE_SUMMARY_PATH = REPORTS_DIRECTORY / "case_study_base_case_summary.csv"

BASE_CASE_LOG_PRESSURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY
    / "case_study_base_log_pressure_vs_log_specific_volume.png"
)
BASE_CASE_PRESSURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY / "case_study_base_pressure_vs_specific_volume.png"
)
BASE_CASE_PRESSURE_ANGLE_FIGURE = (
    IMAGES_DIRECTORY / "case_study_base_pressure_vs_crank_angle.png"
)
BASE_CASE_TEMPERATURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY / "case_study_base_temperature_vs_specific_volume.png"
)
BASE_CASE_POLYTROPIC_EXPONENT_FIGURE = (
    IMAGES_DIRECTORY / "case_study_base_polytropic_exponent_vs_crank_angle.png"
)

ENGINE_SPEEDS_RPM = np.linspace(500.0, 10_000.0, 20)
IGNITION_TIMINGS_DEGREES = np.linspace(-120.0, 0.0, 6)

# All physical and thermodynamic values come from the article. The combustion
# time is retained from the case-study definition because engine speed converts
# this fixed duration to the heat-addition angle swept by the model.
CASE_STUDY_PARAMETERS = replace(
    ARTICLE_BASE_PARAMETERS,
    heat_addition_duration_s=2_500e-6,
)
BASE_CASE_ENGINE_SPEED_RPM = 4_800.0
BASE_CASE_IGNITION_TIMING_DEGREES = -15.0

RESULT_METRICS = (
    ("thermal_efficiency_percent", "%"),
    ("net_specific_power_kw_per_kg", "kW/kg"),
    ("work_consumption_ratio", "-"),
    ("maximum_pressure_kpa", "kPa"),
    ("maximum_temperature_k", "K"),
)

# Redundant encodings (dash and marker) preserve every series in grayscale.
BLACK_AND_WHITE_SERIES_STYLES = (
    {"linestyle": "-", "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": "-.", "marker": "^"},
    {"linestyle": ":", "marker": "D"},
    {"linestyle": (0, (5, 1)), "marker": "v"},
    {"linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
)

CHART_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}


def case_study_crank_angle_grid_rad(
    engine_speed_rpm: float,
    ignition_timing_degrees: float,
    parameters: ModelParameters = CASE_STUDY_PARAMETERS,
) -> np.ndarray:
    """Build the article-resolution grid for one case-study operating point."""
    heat_addition_angle_degrees = (
        6.0 * engine_speed_rpm * parameters.heat_addition_duration_s
    )
    heat_addition_intervals = round(
        heat_addition_angle_degrees / ARTICLE_HEAT_ADDITION_STEP_DEGREES
    )
    ignition_start_rad = float(np.deg2rad(ignition_timing_degrees))
    ignition_end_rad = ignition_start_rad + float(
        np.deg2rad(heat_addition_angle_degrees)
    )
    if ignition_start_rad < -np.pi or ignition_end_rad > np.pi:
        raise ValueError("Heat addition must remain inside the simulated cycle.")

    compression_grid = np.linspace(
        -np.pi,
        ignition_start_rad,
        ARTICLE_STROKE_INTERVALS + 1,
    )
    heat_addition_grid = np.linspace(
        ignition_start_rad,
        ignition_end_rad,
        heat_addition_intervals + 1,
    )[1:]
    expansion_grid = np.linspace(
        ignition_end_rad,
        np.pi,
        ARTICLE_STROKE_INTERVALS + 1,
    )[1:]
    return np.concatenate(
        (compression_grid, heat_addition_grid, expansion_grid)
    )


def calculate_case_study_base_case(
    parameters: ModelParameters = CASE_STUDY_PARAMETERS,
) -> CycleResult:
    """Simulate the ``N=4800 rpm`` and ``theta=-15 degrees`` reference point."""
    return simulate_cycle(
        engine_speed_rpm=BASE_CASE_ENGINE_SPEED_RPM,
        ignition_timing_degrees=BASE_CASE_IGNITION_TIMING_DEGREES,
        parameters=parameters,
        crank_angle_grid_rad=case_study_crank_angle_grid_rad(
            BASE_CASE_ENGINE_SPEED_RPM,
            BASE_CASE_IGNITION_TIMING_DEGREES,
            parameters,
        ),
    )


def base_case_state_history(result: CycleResult) -> pd.DataFrame:
    """Convert the complete base-case state and interval history to a table."""
    final_state_padding = np.array([np.nan])
    return pd.DataFrame(
        {
            "crank_angle_rad": result.crank_angle_rad,
            "total_volume_m3": result.total_volume_m3,
            "specific_volume_m3_per_kg": (
                result.specific_volume_m3_per_kg
            ),
            "pressure_kpa": result.pressure_kpa,
            "temperature_k": result.temperature_k,
            "specific_internal_energy_kj_per_kg": (
                result.specific_internal_energy_kj_per_kg
            ),
            "heat_added_to_next_state_kj_per_kg": np.concatenate(
                (result.heat_added_kj_per_kg, final_state_padding)
            ),
            "work_on_gas_to_next_state_kj_per_kg": np.concatenate(
                (result.work_on_gas_kj_per_kg, final_state_padding)
            ),
            "polytropic_exponent_to_next_state": np.concatenate(
                (result.polytropic_exponent, final_state_padding)
            ),
        }
    )


def base_case_summary(result: CycleResult) -> pd.DataFrame:
    """Return the reference inputs and performance indicators in one row."""
    metrics = result.metrics
    heat_addition_angle_degrees = (
        6.0
        * BASE_CASE_ENGINE_SPEED_RPM
        * CASE_STUDY_PARAMETERS.heat_addition_duration_s
    )
    return pd.DataFrame.from_records(
        [
            {
                "engine_speed_rpm": BASE_CASE_ENGINE_SPEED_RPM,
                "ignition_timing_degrees": (
                    BASE_CASE_IGNITION_TIMING_DEGREES
                ),
                "heat_addition_angle_degrees": heat_addition_angle_degrees,
                "thermal_efficiency_percent": (
                    100.0 * metrics.thermal_efficiency
                ),
                "compression_work_kj_per_kg": (
                    metrics.compression_work_kj_per_kg
                ),
                "expansion_work_kj_per_kg": (
                    metrics.expansion_work_kj_per_kg
                ),
                "net_specific_work_kj_per_kg": (
                    metrics.net_specific_work_kj_per_kg
                ),
                "net_specific_power_kw_per_kg": (
                    metrics.net_specific_power_kw_per_kg
                ),
                "work_consumption_ratio": metrics.work_consumption_ratio,
                "maximum_pressure_kpa": metrics.maximum_pressure_kpa,
                "maximum_temperature_k": metrics.maximum_temperature_k,
            }
        ]
    )


def calculate_sensitivity_results(
    parameters: ModelParameters = CASE_STUDY_PARAMETERS,
) -> pd.DataFrame:
    """Evaluate all 120 operating points from the sensitivity study."""
    records: list[dict[str, float]] = []
    for ignition_timing_degrees in IGNITION_TIMINGS_DEGREES:
        for engine_speed_rpm in ENGINE_SPEEDS_RPM:
            result = simulate_cycle(
                engine_speed_rpm=engine_speed_rpm,
                ignition_timing_degrees=ignition_timing_degrees,
                parameters=parameters,
                crank_angle_grid_rad=case_study_crank_angle_grid_rad(
                    engine_speed_rpm,
                    ignition_timing_degrees,
                    parameters,
                ),
            )
            metrics = result.metrics
            records.append(
                {
                    "engine_speed_rpm": engine_speed_rpm,
                    "ignition_timing_degrees": ignition_timing_degrees,
                    "heat_addition_angle_degrees": (
                        6.0
                        * engine_speed_rpm
                        * parameters.heat_addition_duration_s
                    ),
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


def summarize_sensitivity_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return the global minimum and maximum of every response indicator."""
    records: list[dict[str, float | str]] = []
    for value_column, unit in RESULT_METRICS:
        minimum_row = results.loc[results[value_column].idxmin()]
        maximum_row = results.loc[results[value_column].idxmax()]
        records.append(
            {
                "metric": value_column,
                "unit": unit,
                "minimum_value": minimum_row[value_column],
                "minimum_engine_speed_rpm": minimum_row["engine_speed_rpm"],
                "minimum_ignition_timing_degrees": minimum_row[
                    "ignition_timing_degrees"
                ],
                "maximum_value": maximum_row[value_column],
                "maximum_engine_speed_rpm": maximum_row["engine_speed_rpm"],
                "maximum_ignition_timing_degrees": maximum_row[
                    "ignition_timing_degrees"
                ],
            }
        )
    return pd.DataFrame.from_records(records)


def _style_axis(axis: Axes) -> None:
    """Apply the case-study visual identity without relying on color."""
    axis.grid(
        True,
        which="major",
        color="0.72",
        linestyle=":",
        linewidth=0.7,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", width=0.8)


def _add_base_case_details(axis: Axes, result: CycleResult) -> None:
    """Add a compact, monochrome operating-point label to a diagram."""
    heat_addition_angle_degrees = (
        6.0
        * BASE_CASE_ENGINE_SPEED_RPM
        * CASE_STUDY_PARAMETERS.heat_addition_duration_s
    )
    details = (
        f"N = {BASE_CASE_ENGINE_SPEED_RPM:.0f} rpm\n"
        f"θ = {BASE_CASE_IGNITION_TIMING_DEGREES:.0f}°\n"
        f"δ = {heat_addition_angle_degrees:.0f}°\n"
        f"ηₜ = {100.0 * result.metrics.thermal_efficiency:.2f}%"
    )
    axis.text(
        0.98,
        0.96,
        details,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "square,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.8,
        },
    )


def _add_heat_addition_region(axis: Axes) -> None:
    """Mark heat addition with a hatch that remains visible in grayscale."""
    ignition_start_rad = float(
        np.deg2rad(BASE_CASE_IGNITION_TIMING_DEGREES)
    )
    heat_addition_angle_degrees = (
        6.0
        * BASE_CASE_ENGINE_SPEED_RPM
        * CASE_STUDY_PARAMETERS.heat_addition_duration_s
    )
    ignition_end_rad = ignition_start_rad + float(
        np.deg2rad(heat_addition_angle_degrees)
    )
    axis.axvspan(
        ignition_start_rad,
        ignition_end_rad,
        facecolor="0.92",
        edgecolor="0.45",
        hatch="///",
        linewidth=0.0,
        label="Adição de calor",
    )


def _configure_crank_angle_axis(axis: Axes) -> None:
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks(
        [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )


def _save_base_case_figure(
    figure: plt.Figure,
    destination: Path,
) -> None:
    figure.tight_layout()
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_case_study_base_case_artifacts(result: CycleResult) -> None:
    """Save the reference-point tables and five diagnostic diagrams."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    base_case_state_history(result).to_csv(
        BASE_CASE_STATES_PATH,
        index=False,
        float_format="%.10g",
    )
    base_case_summary(result).to_csv(
        BASE_CASE_SUMMARY_PATH,
        index=False,
        float_format="%.10g",
    )

    closed_specific_volume = np.append(
        result.specific_volume_m3_per_kg,
        result.specific_volume_m3_per_kg[0],
    )
    closed_pressure = np.append(result.pressure_kpa, result.pressure_kpa[0])
    closed_temperature = np.append(
        result.temperature_k,
        result.temperature_k[0],
    )

    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.loglog(
            closed_specific_volume,
            closed_pressure,
            color="black",
            linewidth=1.4,
        )
        axis.set_title(
            "Ponto de referência: pressão × volume específico em escala logarítmica",
            loc="left",
            fontweight="bold",
            pad=10.0,
        )
        axis.set_xlabel("Volume específico, v [m³/kg]")
        axis.set_ylabel("Pressão, P [kPa]")
        _style_axis(axis)
        _add_base_case_details(axis, result)
        _save_base_case_figure(
            figure,
            BASE_CASE_LOG_PRESSURE_VOLUME_FIGURE,
        )

        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.plot(
            closed_specific_volume,
            closed_pressure,
            color="black",
            linewidth=1.4,
        )
        axis.set_title(
            "Ponto de referência: pressão em função do volume específico",
            loc="left",
            fontweight="bold",
            pad=10.0,
        )
        axis.set_xlabel("Volume específico, v [m³/kg]")
        axis.set_ylabel("Pressão, P [kPa]")
        _style_axis(axis)
        _add_base_case_details(axis, result)
        _save_base_case_figure(figure, BASE_CASE_PRESSURE_VOLUME_FIGURE)

        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.plot(
            result.crank_angle_rad,
            result.pressure_kpa,
            color="black",
            linewidth=1.4,
            label="Pressão",
        )
        _add_heat_addition_region(axis)
        axis.set_title(
            "Ponto de referência: pressão em função do ângulo do virabrequim",
            loc="left",
            fontweight="bold",
            pad=10.0,
        )
        axis.set_xlabel("Ângulo do virabrequim, α [rad]")
        axis.set_ylabel("Pressão, P [kPa]")
        _configure_crank_angle_axis(axis)
        _style_axis(axis)
        axis.legend(frameon=False)
        _save_base_case_figure(figure, BASE_CASE_PRESSURE_ANGLE_FIGURE)

        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.plot(
            closed_specific_volume,
            closed_temperature,
            color="black",
            linewidth=1.4,
        )
        axis.set_title(
            "Ponto de referência: temperatura em função do volume específico",
            loc="left",
            fontweight="bold",
            pad=10.0,
        )
        axis.set_xlabel("Volume específico, v [m³/kg]")
        axis.set_ylabel("Temperatura, T [K]")
        _style_axis(axis)
        _add_base_case_details(axis, result)
        _save_base_case_figure(
            figure,
            BASE_CASE_TEMPERATURE_VOLUME_FIGURE,
        )

        interval_crank_angle_rad = 0.5 * (
            result.crank_angle_rad[:-1] + result.crank_angle_rad[1:]
        )
        finite_polytropic_exponent = np.where(
            np.isfinite(result.polytropic_exponent),
            result.polytropic_exponent,
            np.nan,
        )
        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.plot(
            interval_crank_angle_rad,
            finite_polytropic_exponent,
            color="black",
            linewidth=1.2,
            label="Expoente politrópico",
        )
        _add_heat_addition_region(axis)
        axis.set_title(
            "Ponto de referência: expoente politrópico em função do ângulo",
            loc="left",
            fontweight="bold",
            pad=10.0,
        )
        axis.set_xlabel("Ângulo do virabrequim, α [rad]")
        axis.set_ylabel("Expoente politrópico, n [−]")
        axis.set_yscale("symlog", linthresh=5.0)
        _configure_crank_angle_axis(axis)
        _style_axis(axis)
        axis.legend(frameon=False)
        _save_base_case_figure(
            figure,
            BASE_CASE_POLYTROPIC_EXPONENT_FIGURE,
        )


def _save_chart(
    results: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    vertical_axis_label: str,
    filename: str,
) -> None:
    grouped_results = results.groupby("ignition_timing_degrees", sort=True)
    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(9.0, 5.4))
        for (
            (ignition_timing_degrees, angle_results),
            series_style,
        ) in zip(
            grouped_results,
            BLACK_AND_WHITE_SERIES_STYLES,
            strict=True,
        ):
            axis.plot(
                angle_results["engine_speed_rpm"],
                angle_results[value_column],
                color="black",
                linewidth=1.25,
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markevery=2,
                label=f"{ignition_timing_degrees:.0f}°",
                **series_style,
            )

        axis.set_title(title, loc="left", fontweight="bold", pad=10.0)
        axis.set_xlabel("Rotação do motor, N [rpm]")
        axis.set_ylabel(vertical_axis_label)
        axis.set_xlim(ENGINE_SPEEDS_RPM[0], ENGINE_SPEEDS_RPM[-1])
        axis.set_xticks(
            np.linspace(ENGINE_SPEEDS_RPM[0], ENGINE_SPEEDS_RPM[-1], 6)
        )
        _style_axis(axis)
        axis.legend(
            title="Início da ignição, θ",
            fontsize="small",
            frameon=False,
            handlelength=3.5,
            ncols=2,
        )
        figure.tight_layout()
        figure.savefig(
            IMAGES_DIRECTORY / filename,
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_sensitivity_artifacts(results: pd.DataFrame) -> None:
    """Write the tabular results and all charts to the project directories."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.10g")
    summarize_sensitivity_results(results).to_csv(
        SUMMARY_PATH,
        index=False,
        float_format="%.10g",
    )

    chart_definitions = (
        (
            "thermal_efficiency_percent",
            "Eficiência térmica em função da rotação",
            "Eficiência térmica, ηₜ [%]",
            "thermal_efficiency_vs_engine_speed.png",
        ),
        (
            "net_specific_power_kw_per_kg",
            "Potência líquida específica em função da rotação",
            "Potência líquida específica [kW/kg]",
            "net_specific_power_vs_engine_speed.png",
        ),
        (
            "work_consumption_ratio",
            "Razão de consumo de trabalho em função da rotação",
            r"Razão de consumo de trabalho, $r_{ct}$ [−]",
            "work_consumption_ratio_vs_engine_speed.png",
        ),
        (
            "maximum_pressure_kpa",
            "Pressão máxima em função da rotação",
            "Pressão máxima, Pₘₐₓ [kPa]",
            "maximum_pressure_vs_engine_speed.png",
        ),
        (
            "maximum_temperature_k",
            "Temperatura máxima em função da rotação",
            "Temperatura máxima, Tₘₐₓ [K]",
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
    """Calculate the parametric sweep, persist its artifacts, and return it."""
    results = calculate_sensitivity_results()
    save_sensitivity_artifacts(results)
    return results


def run_case_study() -> tuple[CycleResult, pd.DataFrame]:
    """Run and persist the reference point and sensitivity analysis."""
    base_case_result = calculate_case_study_base_case()
    save_case_study_base_case_artifacts(base_case_result)
    sensitivity_results = run_sensitivity_analysis()
    return base_case_result, sensitivity_results


if __name__ == "__main__":
    generated_base_case, generated_results = run_case_study()
    print(f"Saved reference-point states to {BASE_CASE_STATES_PATH}")
    print(f"Saved reference-point summary to {BASE_CASE_SUMMARY_PATH}")
    print(f"Saved {len(generated_results)} operating points to {RESULTS_PATH}")
    print(f"Saved extrema summary to {SUMMARY_PATH}")
    print(f"Saved charts to {IMAGES_DIRECTORY}")
