"""Finite-time heat-addition (FTHA) Otto-cycle objective model.

This module is the reusable Python version of
``notebooks/e2.2_case_study.ipynb``. It contains no plotting or notebook UI
code, so importing it has no side effects and it can be called repeatedly by
multi-objective optimization algorithms.

The objective vector follows the minimization convention commonly adopted by
optimization libraries. Thermal efficiency and net specific power are
therefore returned with a negative sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

from .gas_prop import Gas


FloatArray = NDArray[np.float64]

DECISION_VARIABLE_NAMES: Final[tuple[str, str]] = (
    "engine_speed_rpm",
    "ignition_timing_degrees",
)
DECISION_LOWER_BOUNDS: Final[FloatArray] = np.array([500.0, -120.0])
DECISION_UPPER_BOUNDS: Final[FloatArray] = np.array([10_000.0, 0.0])
OBJECTIVE_NAMES: Final[tuple[str, str, str, str, str]] = (
    "negative_thermal_efficiency",
    "negative_net_specific_power_kw_per_kg",
    "work_consumption_ratio",
    "maximum_pressure_kpa",
    "maximum_temperature_k",
)


@dataclass(frozen=True, slots=True)
class ModelParameters:
    """Fixed engine, fluid, mesh, and convergence parameters."""

    displacement_volume_m3: float = 1.0e-3
    cylinder_count: int = 4
    connecting_rod_to_crank_ratio: float = 4.6
    compression_ratio: float = 12.0
    heat_addition_duration_s: float = 1_800e-6
    initial_temperature_k: float = 293.15
    initial_pressure_kpa: float = 85.0
    specific_heat_input_kj_per_kg: float = 430.0
    mesh_resolution: int = 360
    working_fluid: str = "CO"
    convergence_tolerance: float = 1e-8
    maximum_iterations: int = 100

    def __post_init__(self) -> None:
        positive_values = {
            "displacement_volume_m3": self.displacement_volume_m3,
            "cylinder_count": self.cylinder_count,
            "connecting_rod_to_crank_ratio": self.connecting_rod_to_crank_ratio,
            "compression_ratio": self.compression_ratio,
            "heat_addition_duration_s": self.heat_addition_duration_s,
            "initial_temperature_k": self.initial_temperature_k,
            "initial_pressure_kpa": self.initial_pressure_kpa,
            "specific_heat_input_kj_per_kg": self.specific_heat_input_kj_per_kg,
            "mesh_resolution": self.mesh_resolution,
            "convergence_tolerance": self.convergence_tolerance,
            "maximum_iterations": self.maximum_iterations,
        }
        invalid_names = [name for name, value in positive_values.items() if value <= 0]
        if invalid_names:
            raise ValueError(
                "Model parameters must be positive: " + ", ".join(invalid_names)
            )
        if self.cylinder_count != int(self.cylinder_count):
            raise ValueError("cylinder_count must be an integer.")
        if self.mesh_resolution < 2:
            raise ValueError("mesh_resolution must be at least 2.")
        if self.compression_ratio <= 1.0:
            raise ValueError("compression_ratio must be greater than 1.")
        if self.connecting_rod_to_crank_ratio <= 1.0:
            raise ValueError("connecting_rod_to_crank_ratio must be greater than 1.")


@dataclass(frozen=True, slots=True)
class CycleMetrics:
    """Performance indicators calculated for one operating point."""

    thermal_efficiency: float
    compression_work_kj_per_kg: float
    expansion_work_kj_per_kg: float
    net_specific_work_kj_per_kg: float
    net_specific_power_kw_per_kg: float
    work_consumption_ratio: float
    maximum_pressure_kpa: float
    maximum_temperature_k: float

    def as_minimization_objectives(self) -> FloatArray:
        """Return all report indicators using a minimization convention."""
        return np.array(
            [
                -self.thermal_efficiency,
                -self.net_specific_power_kw_per_kg,
                self.work_consumption_ratio,
                self.maximum_pressure_kpa,
                self.maximum_temperature_k,
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Thermodynamic state history and derived metrics for one cycle."""

    crank_angle_rad: FloatArray
    total_volume_m3: FloatArray
    specific_volume_m3_per_kg: FloatArray
    pressure_kpa: FloatArray
    temperature_k: FloatArray
    specific_internal_energy_kj_per_kg: FloatArray
    heat_added_kj_per_kg: FloatArray
    work_on_gas_kj_per_kg: FloatArray
    polytropic_exponent: FloatArray
    metrics: CycleMetrics


DEFAULT_PARAMETERS: Final = ModelParameters()


@lru_cache(maxsize=None)
def _get_gas(working_fluid: str) -> Gas:
    return Gas(working_fluid)


def _heat_addition_fraction(
    crank_angle_rad: FloatArray,
    ignition_timing_rad: float,
    heat_addition_angle_rad: float,
) -> FloatArray:
    normalized_angle = (crank_angle_rad - ignition_timing_rad) / heat_addition_angle_rad
    fraction = 0.5 - 0.5 * np.cos(np.pi * normalized_angle)
    return np.where(
        crank_angle_rad < ignition_timing_rad,
        0.0,
        np.where(
            crank_angle_rad <= ignition_timing_rad + heat_addition_angle_rad,
            fraction,
            1.0,
        ),
    )


def _cylinder_volume(
    crank_angle_rad: FloatArray,
    parameters: ModelParameters,
) -> FloatArray:
    displacement_per_cylinder_m3 = (
        parameters.displacement_volume_m3 / parameters.cylinder_count
    )
    clearance_volume_m3 = displacement_per_cylinder_m3 / (
        parameters.compression_ratio - 1.0
    )
    bore_m = (4.0 * displacement_per_cylinder_m3 / np.pi) ** (1.0 / 3.0)
    stroke_m = bore_m
    crank_radius_m = stroke_m / 2.0
    connecting_rod_length_m = (
        parameters.connecting_rod_to_crank_ratio * crank_radius_m
    )

    piston_displacement_m = connecting_rod_length_m * (
        1.0
        - np.sqrt(
            1.0
            - (crank_radius_m / connecting_rod_length_m) ** 2
            * np.sin(crank_angle_rad) ** 2
        )
    ) + crank_radius_m * (1.0 - np.cos(crank_angle_rad))
    return np.pi * piston_displacement_m * bore_m**2 / 4.0 + clearance_volume_m3


def _polytropic_work_on_gas(
    initial_pressure_kpa: float,
    polytropic_exponent: float,
    initial_specific_volume_m3_per_kg: float,
    final_specific_volume_m3_per_kg: float,
) -> float:
    volume_ratio = (
        initial_specific_volume_m3_per_kg / final_specific_volume_m3_per_kg
    )
    if np.isclose(polytropic_exponent, 1.0):
        return (
            initial_pressure_kpa
            * initial_specific_volume_m3_per_kg
            * np.log(volume_ratio)
        )
    return (
        initial_pressure_kpa
        * initial_specific_volume_m3_per_kg
        / (1.0 - polytropic_exponent)
        * (1.0 - volume_ratio ** (polytropic_exponent - 1.0))
    )


def _polytropic_exponent(
    final_pressure_kpa: float,
    initial_pressure_kpa: float,
    initial_specific_volume_m3_per_kg: float,
    final_specific_volume_m3_per_kg: float,
) -> float:
    return np.log(final_pressure_kpa / initial_pressure_kpa) / np.log(
        initial_specific_volume_m3_per_kg / final_specific_volume_m3_per_kg
    )


def _calculate_metrics(
    work_on_gas_kj_per_kg: FloatArray,
    heat_added_kj_per_kg: FloatArray,
    pressure_kpa: FloatArray,
    temperature_k: FloatArray,
    engine_speed_rpm: float | None,
) -> CycleMetrics:
    compression_work_kj_per_kg = float(
        work_on_gas_kj_per_kg[work_on_gas_kj_per_kg > 0.0].sum()
    )
    expansion_work_kj_per_kg = float(
        -work_on_gas_kj_per_kg[work_on_gas_kj_per_kg < 0.0].sum()
    )
    net_work_output_kj_per_kg = expansion_work_kj_per_kg - compression_work_kj_per_kg
    supplied_heat_kj_per_kg = float(
        heat_added_kj_per_kg[heat_added_kj_per_kg > 0.0].sum()
    )
    if supplied_heat_kj_per_kg <= 0.0:
        raise ValueError("The simulated cycle received no heat input.")
    if expansion_work_kj_per_kg <= 0.0:
        raise ValueError("The simulated cycle produced no expansion work.")

    net_specific_power_kw_per_kg = (
        np.nan
        if engine_speed_rpm is None
        else net_work_output_kj_per_kg * engine_speed_rpm / 120.0
    )
    return CycleMetrics(
        thermal_efficiency=net_work_output_kj_per_kg / supplied_heat_kj_per_kg,
        compression_work_kj_per_kg=compression_work_kj_per_kg,
        expansion_work_kj_per_kg=expansion_work_kj_per_kg,
        net_specific_work_kj_per_kg=net_work_output_kj_per_kg,
        net_specific_power_kw_per_kg=net_specific_power_kw_per_kg,
        work_consumption_ratio=compression_work_kj_per_kg / expansion_work_kj_per_kg,
        maximum_pressure_kpa=float(pressure_kpa.max()),
        maximum_temperature_k=float(temperature_k.max()),
    )


def simulate_cycle(
    engine_speed_rpm: float | None,
    ignition_timing_degrees: float,
    parameters: ModelParameters = DEFAULT_PARAMETERS,
    *,
    heat_addition_angle_degrees: float | None = None,
    crank_angle_grid_rad: FloatArray | None = None,
) -> CycleResult:
    """Simulate one four-stroke FTHA Otto cycle.

    Ignition timing is measured relative to top dead center; negative values
    indicate ignition before TDC. By default, the heat-addition angle is derived
    from engine speed and combustion duration. ``heat_addition_angle_degrees``
    can instead define it directly, as in the article's sensitivity study. When
    engine speed is ``None``, specific power is returned as ``nan``.
    """
    ignition_timing_degrees = float(ignition_timing_degrees)
    if not np.isfinite(ignition_timing_degrees):
        raise ValueError("ignition_timing_degrees must be finite.")
    if engine_speed_rpm is not None:
        engine_speed_rpm = float(engine_speed_rpm)
        if not np.isfinite(engine_speed_rpm) or engine_speed_rpm <= 0.0:
            raise ValueError("engine_speed_rpm must be a finite positive number.")

    if heat_addition_angle_degrees is None:
        if engine_speed_rpm is None:
            raise ValueError(
                "engine_speed_rpm is required when heat_addition_angle_degrees "
                "is not provided."
            )
        angular_speed_rad_per_s = 2.0 * np.pi * engine_speed_rpm / 60.0
        heat_addition_angle_rad = (
            angular_speed_rad_per_s * parameters.heat_addition_duration_s
        )
    else:
        heat_addition_angle_degrees = float(heat_addition_angle_degrees)
        if (
            not np.isfinite(heat_addition_angle_degrees)
            or heat_addition_angle_degrees <= 0.0
        ):
            raise ValueError(
                "heat_addition_angle_degrees must be a finite positive number."
            )
        heat_addition_angle_rad = np.deg2rad(heat_addition_angle_degrees)

    working_gas = _get_gas(parameters.working_fluid)
    if crank_angle_grid_rad is None:
        crank_angle_rad = np.linspace(
            -np.pi,
            np.pi,
            2 * parameters.mesh_resolution,
            dtype=float,
        )
    else:
        crank_angle_rad = np.asarray(crank_angle_grid_rad, dtype=float)
        if crank_angle_rad.ndim != 1 or crank_angle_rad.size < 3:
            raise ValueError("crank_angle_grid_rad must be a one-dimensional grid.")
        if not np.all(np.isfinite(crank_angle_rad)):
            raise ValueError("crank_angle_grid_rad must contain only finite values.")
        if not np.all(np.diff(crank_angle_rad) > 0.0):
            raise ValueError("crank_angle_grid_rad must be strictly increasing.")
        if not np.isclose(crank_angle_rad[0], -np.pi) or not np.isclose(
            crank_angle_rad[-1], np.pi
        ):
            raise ValueError("crank_angle_grid_rad must span from -pi to pi.")

    ignition_timing_rad = np.deg2rad(ignition_timing_degrees)

    total_volume_m3 = _cylinder_volume(crank_angle_rad, parameters)
    initial_specific_volume_m3_per_kg = working_gas.specific_volume(
        parameters.initial_temperature_k,
        parameters.initial_pressure_kpa,
    )
    cylinder_gas_mass_kg = total_volume_m3[0] / initial_specific_volume_m3_per_kg
    specific_volume_m3_per_kg = total_volume_m3 / cylinder_gas_mass_kg

    heat_fraction = _heat_addition_fraction(
        crank_angle_rad,
        ignition_timing_rad,
        heat_addition_angle_rad,
    )
    heat_added_kj_per_kg = (
        np.diff(heat_fraction) * parameters.specific_heat_input_kj_per_kg
    )

    number_of_states = crank_angle_rad.size
    number_of_intervals = number_of_states - 1
    pressure_kpa = np.empty(number_of_states, dtype=float)
    temperature_k = np.empty(number_of_states, dtype=float)
    internal_energy_kj_per_kg = np.empty(number_of_states, dtype=float)
    work_on_gas_kj_per_kg = np.empty(number_of_intervals, dtype=float)
    polytropic_exponent = np.empty(number_of_intervals, dtype=float)

    pressure_kpa[0] = parameters.initial_pressure_kpa
    temperature_k[0] = parameters.initial_temperature_k
    internal_energy_kj_per_kg[0] = working_gas.specific_internal_energy(
        parameters.initial_temperature_k
    )

    for interval_index in range(number_of_intervals):
        initial_volume = specific_volume_m3_per_kg[interval_index]
        final_volume = specific_volume_m3_per_kg[interval_index + 1]
        interval_heat = heat_added_kj_per_kg[interval_index]

        if abs(initial_volume - final_volume) <= parameters.convergence_tolerance:
            interval_work = 0.0
            exponent = np.inf
            next_internal_energy = (
                internal_energy_kj_per_kg[interval_index] + interval_heat
            )
            next_temperature = working_gas.temperature_from_internal_energy(
                next_internal_energy
            )
            next_pressure = working_gas.pressure(next_temperature, final_volume)
        else:
            exponent = 1.0 + working_gas.specific_gas_constant / (
                working_gas.specific_heat_constant_volume(
                    temperature_k[interval_index]
                )
            )
            previous_work = _polytropic_work_on_gas(
                pressure_kpa[interval_index],
                exponent,
                initial_volume,
                final_volume,
            )

            for _ in range(parameters.maximum_iterations):
                next_internal_energy = (
                    internal_energy_kj_per_kg[interval_index]
                    + interval_heat
                    + previous_work
                )
                next_temperature = working_gas.temperature_from_internal_energy(
                    next_internal_energy
                )
                next_pressure = working_gas.pressure(next_temperature, final_volume)
                exponent = _polytropic_exponent(
                    next_pressure,
                    pressure_kpa[interval_index],
                    initial_volume,
                    final_volume,
                )
                interval_work = _polytropic_work_on_gas(
                    pressure_kpa[interval_index],
                    exponent,
                    initial_volume,
                    final_volume,
                )
                if abs(interval_work - previous_work) < parameters.convergence_tolerance:
                    break
                previous_work = interval_work
            else:
                raise RuntimeError(
                    "Polytropic iteration did not converge at interval "
                    f"{interval_index}."
                )

            # Recalculate the final state with the converged work value.
            next_internal_energy = (
                internal_energy_kj_per_kg[interval_index]
                + interval_heat
                + interval_work
            )
            next_temperature = working_gas.temperature_from_internal_energy(
                next_internal_energy
            )
            next_pressure = working_gas.pressure(next_temperature, final_volume)

        work_on_gas_kj_per_kg[interval_index] = interval_work
        polytropic_exponent[interval_index] = exponent
        internal_energy_kj_per_kg[interval_index + 1] = next_internal_energy
        temperature_k[interval_index + 1] = next_temperature
        pressure_kpa[interval_index + 1] = next_pressure

    metrics = _calculate_metrics(
        work_on_gas_kj_per_kg,
        heat_added_kj_per_kg,
        pressure_kpa,
        temperature_k,
        engine_speed_rpm,
    )
    return CycleResult(
        crank_angle_rad=crank_angle_rad,
        total_volume_m3=total_volume_m3,
        specific_volume_m3_per_kg=specific_volume_m3_per_kg,
        pressure_kpa=pressure_kpa,
        temperature_k=temperature_k,
        specific_internal_energy_kj_per_kg=internal_energy_kj_per_kg,
        heat_added_kj_per_kg=heat_added_kj_per_kg,
        work_on_gas_kj_per_kg=work_on_gas_kj_per_kg,
        polytropic_exponent=polytropic_exponent,
        metrics=metrics,
    )


def evaluate_operating_point(
    engine_speed_rpm: float,
    ignition_timing_degrees: float,
    parameters: ModelParameters = DEFAULT_PARAMETERS,
) -> CycleMetrics:
    """Return only the performance indicators for an operating point."""
    return simulate_cycle(
        engine_speed_rpm,
        ignition_timing_degrees,
        parameters,
    ).metrics


def objective_function(
    decision_variables: Sequence[float] | FloatArray,
    parameters: ModelParameters = DEFAULT_PARAMETERS,
) -> FloatArray:
    """Evaluate the five notebook indicators as minimization objectives.

    Decision variables must be ordered as ``[engine_speed_rpm,
    ignition_timing_degrees]`` and lie inside the study domain defined by
    ``DECISION_LOWER_BOUNDS`` and ``DECISION_UPPER_BOUNDS``.
    """
    decisions = np.asarray(decision_variables, dtype=float)
    if decisions.shape != (2,):
        raise ValueError(
            "decision_variables must contain exactly [engine_speed_rpm, "
            "ignition_timing_degrees]."
        )
    if not np.all(np.isfinite(decisions)):
        raise ValueError("decision_variables must contain only finite values.")
    if np.any(decisions < DECISION_LOWER_BOUNDS) or np.any(
        decisions > DECISION_UPPER_BOUNDS
    ):
        raise ValueError(
            "decision_variables are outside the notebook study bounds: "
            f"{DECISION_LOWER_BOUNDS.tolist()} to {DECISION_UPPER_BOUNDS.tolist()}."
        )

    metrics = evaluate_operating_point(
        engine_speed_rpm=decisions[0],
        ignition_timing_degrees=decisions[1],
        parameters=parameters,
    )
    return metrics.as_minimization_objectives()


__all__ = [
    "CycleMetrics",
    "CycleResult",
    "DECISION_LOWER_BOUNDS",
    "DECISION_UPPER_BOUNDS",
    "DECISION_VARIABLE_NAMES",
    "DEFAULT_PARAMETERS",
    "ModelParameters",
    "OBJECTIVE_NAMES",
    "evaluate_operating_point",
    "objective_function",
    "simulate_cycle",
]
