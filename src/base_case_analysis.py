"""Generate the thermodynamic diagrams for the report base case."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .FTHA import CycleResult, ModelParameters, simulate_cycle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIRECTORY = PROJECT_ROOT / "img"
ARTICLE_BASE_PARAMETERS = ModelParameters(
    displacement_volume_m3=250e-6,
    cylinder_count=1,
    connecting_rod_to_crank_ratio=5.0,
    compression_ratio=12.0,
    initial_temperature_k=300.0,
    initial_pressure_kpa=100.0,
    specific_heat_input_kj_per_kg=1_000.0,
    working_fluid="CO2",
)
BASE_IGNITION_TIMING_DEGREES = -5.0
BASE_HEAT_ADDITION_ANGLE_DEGREES = 10.0
ARTICLE_STROKE_INTERVALS = 90
ARTICLE_HEAT_ADDITION_STEP_DEGREES = 0.5


def _base_case_label(result: CycleResult) -> str:
    return (
        f"Eficiência = {100.0 * result.metrics.thermal_efficiency:.2f}%\n"
        f"Ignição = {BASE_IGNITION_TIMING_DEGREES:.0f}°\n"
        f"Adição de calor = {BASE_HEAT_ADDITION_ANGLE_DEGREES:.0f}°"
    )


def _style_axis(axis: Axes) -> None:
    axis.grid(True, which="both", linestyle="--", alpha=0.45)


def _save_figure(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(IMAGES_DIRECTORY / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _heat_addition_limits_rad() -> tuple[float, float]:
    ignition_start_rad = float(np.deg2rad(BASE_IGNITION_TIMING_DEGREES))
    ignition_end_rad = ignition_start_rad + float(
        np.deg2rad(BASE_HEAT_ADDITION_ANGLE_DEGREES)
    )
    return ignition_start_rad, ignition_end_rad


def _article_crank_angle_grid_rad() -> np.ndarray:
    ignition_start_rad, ignition_end_rad = _heat_addition_limits_rad()
    heat_addition_intervals = round(
        BASE_HEAT_ADDITION_ANGLE_DEGREES
        / ARTICLE_HEAT_ADDITION_STEP_DEGREES
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
    return np.concatenate(
        (compression_grid, heat_addition_grid, expansion_grid)
    )


def _add_heat_addition_region(axis: Axes) -> None:
    ignition_start_rad, ignition_end_rad = _heat_addition_limits_rad()
    axis.axvspan(
        ignition_start_rad,
        ignition_end_rad,
        color="tab:orange",
        alpha=0.15,
        label="Adição de calor",
    )


def generate_base_case_diagrams() -> CycleResult:
    """Simulate the report base case and save its five diagrams in ``img``."""
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    result = simulate_cycle(
        engine_speed_rpm=None,
        ignition_timing_degrees=BASE_IGNITION_TIMING_DEGREES,
        parameters=ARTICLE_BASE_PARAMETERS,
        heat_addition_angle_degrees=BASE_HEAT_ADDITION_ANGLE_DEGREES,
        crank_angle_grid_rad=_article_crank_angle_grid_rad(),
    )

    # Close P-v and T-v paths with the constant-volume heat-rejection process.
    closed_specific_volume = np.append(
        result.specific_volume_m3_per_kg,
        result.specific_volume_m3_per_kg[0],
    )
    closed_pressure = np.append(result.pressure_kpa, result.pressure_kpa[0])
    closed_temperature = np.append(result.temperature_k, result.temperature_k[0])

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.loglog(
        closed_specific_volume,
        closed_pressure,
        linewidth=1.5,
        label=_base_case_label(result),
    )
    axis.set_title("Caso-base: log(P) × log(v)")
    axis.set_xlabel("Volume específico, v [m³/kg]")
    axis.set_ylabel("Pressão, P [kPa]")
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, "base_case_log_pressure_vs_log_specific_volume.png")

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(
        closed_specific_volume,
        closed_pressure,
        linewidth=1.5,
        label=_base_case_label(result),
    )
    axis.set_title("Caso-base: P × v")
    axis.set_xlabel("Volume específico, v [m³/kg]")
    axis.set_ylabel("Pressão, P [kPa]")
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, "base_case_pressure_vs_specific_volume.png")

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(
        result.crank_angle_rad,
        result.pressure_kpa,
        linewidth=1.5,
        label="Pressão",
    )
    _add_heat_addition_region(axis)
    axis.set_title("Caso-base: P × α")
    axis.set_xlabel("Ângulo do virabrequim, α [rad]")
    axis.set_ylabel("Pressão, P [kPa]")
    axis.set_xticks(
        [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
        ["−π", "−π/2", "0", "π/2", "π"],
    )
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, "base_case_pressure_vs_crank_angle.png")

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(
        closed_specific_volume,
        closed_temperature,
        linewidth=1.5,
        label=_base_case_label(result),
    )
    axis.set_title("Caso-base: T × v")
    axis.set_xlabel("Volume específico, v [m³/kg]")
    axis.set_ylabel("Temperatura, T [K]")
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, "base_case_temperature_vs_specific_volume.png")

    interval_crank_angle_rad = 0.5 * (
        result.crank_angle_rad[:-1] + result.crank_angle_rad[1:]
    )
    finite_exponent = np.where(
        np.isfinite(result.polytropic_exponent),
        result.polytropic_exponent,
        np.nan,
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(
        interval_crank_angle_rad,
        finite_exponent,
        linewidth=1.5,
        label="Expoente politrópico",
    )
    _add_heat_addition_region(axis)
    axis.set_title("Caso-base: n × α")
    axis.set_xlabel("Ângulo do virabrequim, α [rad]")
    axis.set_ylabel("n [−]")
    axis.set_yscale("symlog", linthresh=5.0)
    axis.set_xticks(
        [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
        ["−π", "−π/2", "0", "π/2", "π"],
    )
    _style_axis(axis)
    axis.legend(fontsize="small")
    _save_figure(figure, "base_case_polytropic_exponent_vs_crank_angle.png")

    return result


if __name__ == "__main__":
    base_case_result = generate_base_case_diagrams()
    print(f"Saved base-case diagrams to {IMAGES_DIRECTORY}")
    print(base_case_result.metrics)
