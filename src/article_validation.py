"""Reproduce the variable-heat-addition-angle tests published in the article."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .base_case_analysis import (
    ARTICLE_BASE_PARAMETERS,
    ARTICLE_HEAT_ADDITION_STEP_DEGREES,
    ARTICLE_STROKE_INTERVALS,
    BASE_IGNITION_TIMING_DEGREES,
    IMAGES_DIRECTORY,
)
from .FTHA import CycleResult, simulate_cycle


ARTICLE_HEAT_ADDITION_ANGLES_DEGREES = (10.0, 30.0, 50.0, 70.0, 90.0, 110.0)
ARTICLE_PUBLISHED_EFFICIENCIES_PERCENT = (38.4, 37.0, 34.1, 30.6, 27.0, 23.7)

LOG_PRESSURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY / "article_variable_delta_log_pressure_vs_log_volume.png"
)
PRESSURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY / "article_variable_delta_pressure_vs_volume.png"
)
PRESSURE_ANGLE_FIGURE = (
    IMAGES_DIRECTORY / "article_variable_delta_pressure_vs_crank_angle.png"
)
TEMPERATURE_VOLUME_FIGURE = (
    IMAGES_DIRECTORY / "article_variable_delta_temperature_vs_volume.png"
)

_LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1)))
_GRAY_LEVELS = ("0.10", "0.25", "0.40", "0.55", "0.67", "0.78")


def article_crank_angle_grid_rad(
    heat_addition_angle_degrees: float,
) -> np.ndarray:
    """Return the article mesh for one angular heat-addition duration."""
    heat_addition_angle_degrees = float(heat_addition_angle_degrees)
    if heat_addition_angle_degrees <= 0.0:
        raise ValueError("heat_addition_angle_degrees must be positive.")

    ignition_start_rad = float(np.deg2rad(BASE_IGNITION_TIMING_DEGREES))
    ignition_end_rad = float(
        np.deg2rad(
            BASE_IGNITION_TIMING_DEGREES + heat_addition_angle_degrees
        )
    )
    if ignition_end_rad >= np.pi:
        raise ValueError("Heat addition must finish before 180 degrees.")

    heat_addition_intervals = round(
        heat_addition_angle_degrees / ARTICLE_HEAT_ADDITION_STEP_DEGREES
    )
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
    return np.concatenate((compression_grid, heat_addition_grid, expansion_grid))


def calculate_article_validation_results() -> dict[float, CycleResult]:
    """Simulate the six angular heat-addition durations reported in the article."""
    return {
        heat_addition_angle_degrees: simulate_cycle(
            engine_speed_rpm=None,
            ignition_timing_degrees=BASE_IGNITION_TIMING_DEGREES,
            parameters=ARTICLE_BASE_PARAMETERS,
            heat_addition_angle_degrees=heat_addition_angle_degrees,
            crank_angle_grid_rad=article_crank_angle_grid_rad(
                heat_addition_angle_degrees
            ),
        )
        for heat_addition_angle_degrees in ARTICLE_HEAT_ADDITION_ANGLES_DEGREES
    }


def _style_axis(axis: Axes) -> None:
    axis.grid(True, which="both", linestyle=":", color="0.65", alpha=0.9)


def _series_label(
    heat_addition_angle_degrees: float,
    result: CycleResult,
) -> str:
    efficiency_percent = 100.0 * result.metrics.thermal_efficiency
    return (
        rf"$\delta={heat_addition_angle_degrees:.0f}^\circ$, "
        rf"$\eta_t={efficiency_percent:.1f}\%$"
    )


def _plot_result(
    axis: Axes,
    heat_addition_angle_degrees: float,
    result: CycleResult,
    horizontal_values: np.ndarray,
    vertical_values: np.ndarray,
    *,
    logarithmic: bool = False,
) -> None:
    plot = axis.loglog if logarithmic else axis.plot
    series_index = ARTICLE_HEAT_ADDITION_ANGLES_DEGREES.index(
        heat_addition_angle_degrees
    )
    plot(
        horizontal_values,
        vertical_values,
        color=_GRAY_LEVELS[series_index],
        linestyle=_LINE_STYLES[series_index],
        linewidth=1.6,
        label=_series_label(heat_addition_angle_degrees, result),
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_article_validation_diagrams(
    results: Mapping[float, CycleResult],
) -> None:
    """Save the four variable-delta diagrams used in the article."""
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    title = (
        rf"Testes para $r=12$, $\theta={BASE_IGNITION_TIMING_DEGREES:.0f}^\circ$ "
        r"e $\delta$ variável"
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    for heat_addition_angle_degrees, result in results.items():
        closed_volume = np.append(
            result.specific_volume_m3_per_kg,
            result.specific_volume_m3_per_kg[0],
        )
        closed_pressure = np.append(result.pressure_kpa, result.pressure_kpa[0])
        _plot_result(
            axis,
            heat_addition_angle_degrees,
            result,
            closed_volume,
            closed_pressure,
            logarithmic=True,
        )
    axis.set_title(title)
    axis.set_xlabel(r"Volume específico, $v$ [m³/kg]")
    axis.set_ylabel(r"Pressão, $P$ [kPa]")
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, LOG_PRESSURE_VOLUME_FIGURE)

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    for heat_addition_angle_degrees, result in results.items():
        closed_volume = np.append(
            result.specific_volume_m3_per_kg,
            result.specific_volume_m3_per_kg[0],
        )
        closed_pressure = np.append(result.pressure_kpa, result.pressure_kpa[0])
        _plot_result(
            axis,
            heat_addition_angle_degrees,
            result,
            closed_volume,
            closed_pressure,
        )
    axis.set_title(title)
    axis.set_xlabel(r"Volume específico, $v$ [m³/kg]")
    axis.set_ylabel(r"Pressão, $P$ [kPa]")
    axis.set_ylim(bottom=0.0)
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, PRESSURE_VOLUME_FIGURE)

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    for heat_addition_angle_degrees, result in results.items():
        _plot_result(
            axis,
            heat_addition_angle_degrees,
            result,
            np.rad2deg(result.crank_angle_rad),
            result.pressure_kpa,
        )
    axis.set_title(title)
    axis.set_xlabel(r"Ângulo do virabrequim, $\alpha$ [graus]")
    axis.set_ylabel(r"Pressão, $P$ [kPa]")
    axis.set_xlim(-180.0, 180.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xticks(np.arange(-180.0, 181.0, 30.0))
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, PRESSURE_ANGLE_FIGURE)

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    for heat_addition_angle_degrees, result in results.items():
        closed_volume = np.append(
            result.specific_volume_m3_per_kg,
            result.specific_volume_m3_per_kg[0],
        )
        closed_temperature = np.append(
            result.temperature_k,
            result.temperature_k[0],
        )
        _plot_result(
            axis,
            heat_addition_angle_degrees,
            result,
            closed_volume,
            closed_temperature,
        )
    axis.set_title(title)
    axis.set_xlabel(r"Volume específico, $v$ [m³/kg]")
    axis.set_ylabel(r"Temperatura, $T$ [K]")
    _style_axis(axis)
    axis.legend(fontsize="small", ncols=2)
    _save_figure(figure, TEMPERATURE_VOLUME_FIGURE)


def generate_article_validation_diagrams() -> dict[float, CycleResult]:
    """Calculate the published tests, save their diagrams, and return results."""
    results = calculate_article_validation_results()
    save_article_validation_diagrams(results)
    return results


if __name__ == "__main__":
    generated_results = generate_article_validation_diagrams()
    print(f"Saved article validation diagrams to {IMAGES_DIRECTORY}")
    for heat_addition_angle_degrees, result in generated_results.items():
        efficiency_percent = 100.0 * result.metrics.thermal_efficiency
        print(
            f"delta={heat_addition_angle_degrees:.0f} degrees: "
            f"eta_t={efficiency_percent:.3f}%"
        )
