"""Ideal-gas properties based on polynomial heat-capacity data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_UNIVERSAL_GAS_CONSTANT = 8.31447  # kJ/(kmol K)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
_GAS_DATA = pd.read_csv(_DATA_PATH, sep=";")


class Gas:
    """Represent an ideal gas with temperature-dependent heat capacities.

    The polynomial coefficients come from ``data.csv``. Temperatures are in K,
    pressures in kPa, specific volumes in m³/kg, and specific energies in
    kJ/kg.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        matching_rows = _GAS_DATA[
            (_GAS_DATA["name"] == name) | (_GAS_DATA["sub"] == name)
        ]
        if matching_rows.empty:
            available_names = ", ".join(_GAS_DATA["sub"].astype(str))
            raise ValueError(
                f"Unknown gas {name!r}. Available abbreviations: {available_names}."
            )

        row = matching_rows.iloc[0]
        self.constant_pressure_coefficients = tuple(
            float(row[column]) for column in ("a", "b", "c", "d")
        )
        self.molecular_mass = float(row["M"])
        self.specific_gas_constant = (
            _UNIVERSAL_GAS_CONSTANT / self.molecular_mass
        )
        first_coefficient, *remaining_coefficients = (
            self.constant_pressure_coefficients
        )
        self.constant_volume_coefficients = (
            first_coefficient - self.molecular_mass * self.specific_gas_constant,
            *remaining_coefficients,
        )

    def specific_heat_constant_pressure(self, temperature_k: float) -> float:
        """Return specific heat at constant pressure in kJ/(kg K)."""
        coefficient_a, coefficient_b, coefficient_c, coefficient_d = (
            self.constant_pressure_coefficients
        )
        molar_heat_capacity = (
            coefficient_a
            + coefficient_b * temperature_k
            + coefficient_c * temperature_k**2
            + coefficient_d * temperature_k**3
        )
        return molar_heat_capacity / self.molecular_mass

    def specific_heat_constant_volume(self, temperature_k: float) -> float:
        """Return specific heat at constant volume in kJ/(kg K)."""
        coefficient_a, coefficient_b, coefficient_c, coefficient_d = (
            self.constant_volume_coefficients
        )
        molar_heat_capacity = (
            coefficient_a
            + coefficient_b * temperature_k
            + coefficient_c * temperature_k**2
            + coefficient_d * temperature_k**3
        )
        return molar_heat_capacity / self.molecular_mass

    def specific_internal_energy(self, temperature_k: float) -> float:
        """Return specific internal energy in kJ/kg."""
        coefficient_a, coefficient_b, coefficient_c, coefficient_d = (
            self.constant_volume_coefficients
        )
        molar_internal_energy = (
            coefficient_a * temperature_k
            + coefficient_b * temperature_k**2 / 2.0
            + coefficient_c * temperature_k**3 / 3.0
            + coefficient_d * temperature_k**4 / 4.0
        )
        return molar_internal_energy / self.molecular_mass

    def temperature_from_internal_energy(
        self,
        target_internal_energy_kj_per_kg: float,
        *,
        tolerance: float = 1.48e-8,
        maximum_iterations: int = 100,
    ) -> float:
        """Invert internal energy with Newton's method and return temperature."""
        initial_specific_heat = (
            self.constant_volume_coefficients[0] / self.molecular_mass
        )
        temperature_k = target_internal_energy_kj_per_kg / initial_specific_heat

        for _ in range(maximum_iterations):
            calculated_internal_energy = self.specific_internal_energy(temperature_k)
            residual = target_internal_energy_kj_per_kg - calculated_internal_energy
            if abs(residual) < tolerance:
                return temperature_k
            temperature_k += residual / self.specific_heat_constant_volume(temperature_k)

        raise RuntimeError(
            "Temperature calculation did not converge within "
            f"{maximum_iterations} iterations."
        )

    def temperature(self, pressure_kpa: float, specific_volume_m3_per_kg: float) -> float:
        """Return temperature from the ideal-gas equation of state."""
        return pressure_kpa * specific_volume_m3_per_kg / self.specific_gas_constant

    def pressure(self, temperature_k: float, specific_volume_m3_per_kg: float) -> float:
        """Return pressure in kPa from the ideal-gas equation of state."""
        return temperature_k * self.specific_gas_constant / specific_volume_m3_per_kg

    def specific_volume(self, temperature_k: float, pressure_kpa: float) -> float:
        """Return specific volume in m³/kg from the ideal-gas equation of state."""
        return temperature_k * self.specific_gas_constant / pressure_kpa

    # Backward-compatible aliases used by the original notebooks.
    @property
    def R(self) -> float:  # noqa: N802
        return self.specific_gas_constant

    @property
    def M(self) -> float:  # noqa: N802
        return self.molecular_mass

    @property
    def cp(self) -> tuple[float, float, float, float]:
        return self.constant_pressure_coefficients

    @property
    def cv(self) -> tuple[float, float, float, float]:
        return self.constant_volume_coefficients

    def cp_T(self, T: float) -> float:  # noqa: N802
        return self.specific_heat_constant_pressure(T)

    def cv_T(self, T: float) -> float:  # noqa: N802
        return self.specific_heat_constant_volume(T)

    def u(self, T: float) -> float:  # noqa: N802
        return self.specific_internal_energy(T)

    def Tu(self, uk: float, ul: float | None = None) -> float:  # noqa: ARG002, N802
        return self.temperature_from_internal_energy(uk)

    def T(self, P: float, v: float) -> float:  # noqa: N802
        return self.temperature(P, v)

    def P(self, T: float, v: float) -> float:  # noqa: N802
        return self.pressure(T, v)

    def v(self, T: float, P: float) -> float:  # noqa: N802
        return self.specific_volume(T, P)
