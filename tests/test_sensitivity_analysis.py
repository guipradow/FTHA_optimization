"""Tests for the persisted article-parameter sensitivity study."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.base_case_analysis import ARTICLE_BASE_PARAMETERS
from src.FTHA import CycleMetrics
from src.sensitivity_analysis import (
    BASE_CASE_ENGINE_SPEED_RPM,
    BASE_CASE_IGNITION_TIMING_DEGREES,
    BASE_CASE_LOG_PRESSURE_VOLUME_FIGURE,
    BASE_CASE_POLYTROPIC_EXPONENT_FIGURE,
    BASE_CASE_PRESSURE_ANGLE_FIGURE,
    BASE_CASE_PRESSURE_VOLUME_FIGURE,
    BASE_CASE_TEMPERATURE_VOLUME_FIGURE,
    CASE_STUDY_PARAMETERS,
    COMPRESSION_RATIOS,
    CONNECTING_ROD_TO_CRANK_RATIOS,
    ENGINE_SPEEDS_RPM,
    IGNITION_TIMINGS_DEGREES,
    RESULT_METRICS,
    base_case_state_history,
    base_case_summary,
    case_study_crank_angle_grid_rad,
    calculate_case_study_base_case,
    calculate_design_space_screening_results,
    calculate_sensitivity_results,
    summarize_sensitivity_results,
)


class SensitivityAnalysisTests(unittest.TestCase):
    def test_fixed_case_study_parameters_match_the_article(self) -> None:
        parameters = CASE_STUDY_PARAMETERS

        for field_name in ARTICLE_BASE_PARAMETERS.__dataclass_fields__:
            if field_name != "heat_addition_duration_s":
                self.assertEqual(
                    getattr(parameters, field_name),
                    getattr(ARTICLE_BASE_PARAMETERS, field_name),
                )
        self.assertEqual(parameters.displacement_volume_m3, 250e-6)
        self.assertEqual(parameters.cylinder_count, 1)
        self.assertEqual(parameters.connecting_rod_to_crank_ratio, 5.0)
        self.assertEqual(parameters.compression_ratio, 12.0)
        self.assertEqual(parameters.heat_addition_duration_s, 2_500e-6)
        self.assertEqual(parameters.initial_temperature_k, 300.0)
        self.assertEqual(parameters.initial_pressure_kpa, 100.0)
        self.assertEqual(parameters.specific_heat_input_kj_per_kg, 1_000.0)
        self.assertEqual(parameters.mesh_resolution, 360)
        self.assertEqual(parameters.working_fluid, "CO2")

    def test_reference_operating_point_is_reproduced(self) -> None:
        result = calculate_case_study_base_case()
        metrics = result.metrics

        self.assertEqual(BASE_CASE_ENGINE_SPEED_RPM, 4_800.0)
        self.assertEqual(BASE_CASE_IGNITION_TIMING_DEGREES, -15.0)
        self.assertEqual(result.crank_angle_rad.size, 325)
        self.assertAlmostEqual(metrics.thermal_efficiency, 0.3325672737, 9)
        self.assertAlmostEqual(
            metrics.net_specific_power_kw_per_kg,
            13_302.69095,
            5,
        )
        self.assertAlmostEqual(metrics.maximum_pressure_kpa, 3_026.236015, 6)
        self.assertAlmostEqual(metrics.maximum_temperature_k, 1_281.300887, 6)

        state_history = base_case_state_history(result)
        summary = base_case_summary(result).iloc[0]
        self.assertEqual(len(state_history), 325)
        self.assertTrue(
            np.isnan(state_history.iloc[-1]["polytropic_exponent_to_next_state"])
        )
        self.assertEqual(summary["heat_addition_angle_degrees"], 72.0)

        expected_figures = (
            BASE_CASE_LOG_PRESSURE_VOLUME_FIGURE,
            BASE_CASE_PRESSURE_VOLUME_FIGURE,
            BASE_CASE_PRESSURE_ANGLE_FIGURE,
            BASE_CASE_TEMPERATURE_VOLUME_FIGURE,
            BASE_CASE_POLYTROPIC_EXPONENT_FIGURE,
        )
        self.assertTrue(all(path.parent.name == "img" for path in expected_figures))

    def test_case_study_grid_uses_article_resolution(self) -> None:
        minimum_speed_grid = case_study_crank_angle_grid_rad(500.0, -120.0)
        maximum_speed_grid = case_study_crank_angle_grid_rad(10_000.0, 0.0)

        self.assertEqual(minimum_speed_grid.size, 196)
        self.assertEqual(maximum_speed_grid.size, 481)
        self.assertAlmostEqual(minimum_speed_grid[0], -np.pi)
        self.assertAlmostEqual(maximum_speed_grid[-1], np.pi)
        self.assertTrue(np.all(np.diff(minimum_speed_grid) > 0.0))
        self.assertTrue(np.all(np.diff(maximum_speed_grid) > 0.0))

    @patch("src.sensitivity_analysis.simulate_cycle")
    def test_sweep_shape_and_derived_heat_addition_angle(
        self,
        simulate_cycle_mock,
    ) -> None:
        simulate_cycle_mock.return_value = SimpleNamespace(
            metrics=CycleMetrics(
                thermal_efficiency=0.3,
                compression_work_kj_per_kg=200.0,
                expansion_work_kj_per_kg=500.0,
                net_specific_work_kj_per_kg=300.0,
                net_specific_power_kw_per_kg=10_000.0,
                work_consumption_ratio=0.4,
                maximum_pressure_kpa=4_000.0,
                maximum_temperature_k=1_400.0,
            )
        )

        results = calculate_sensitivity_results()

        expected_points = len(ENGINE_SPEEDS_RPM) * len(
            IGNITION_TIMINGS_DEGREES
        )
        self.assertEqual(len(results), expected_points)
        self.assertEqual(simulate_cycle_mock.call_count, expected_points)
        self.assertTrue(
            all(
                call.kwargs["parameters"] is CASE_STUDY_PARAMETERS
                for call in simulate_cycle_mock.call_args_list
            )
        )
        np.testing.assert_allclose(
            results["heat_addition_angle_degrees"].unique(),
            6.0
            * ENGINE_SPEEDS_RPM
            * CASE_STUDY_PARAMETERS.heat_addition_duration_s,
        )

        summary = summarize_sensitivity_results(results)
        self.assertEqual(tuple(summary["metric"]), tuple(zip(*RESULT_METRICS))[0])

    @patch("src.sensitivity_analysis.simulate_cycle")
    def test_four_decision_screening_covers_geometric_domain(
        self,
        simulate_cycle_mock,
    ) -> None:
        simulate_cycle_mock.return_value = SimpleNamespace(
            metrics=CycleMetrics(
                thermal_efficiency=0.3,
                compression_work_kj_per_kg=200.0,
                expansion_work_kj_per_kg=500.0,
                net_specific_work_kj_per_kg=300.0,
                net_specific_power_kw_per_kg=10_000.0,
                work_consumption_ratio=0.4,
                maximum_pressure_kpa=4_000.0,
                maximum_temperature_k=1_400.0,
            )
        )

        results = calculate_design_space_screening_results()

        expected_points = (
            len(ENGINE_SPEEDS_RPM)
            * len(IGNITION_TIMINGS_DEGREES)
            * len(COMPRESSION_RATIOS)
            * len(CONNECTING_ROD_TO_CRANK_RATIOS)
        )
        self.assertEqual(len(results), expected_points)
        self.assertEqual(simulate_cycle_mock.call_count, expected_points)
        np.testing.assert_allclose(
            np.sort(results["compression_ratio"].unique()),
            COMPRESSION_RATIOS,
        )
        np.testing.assert_allclose(
            np.sort(results["connecting_rod_to_crank_ratio"].unique()),
            CONNECTING_ROD_TO_CRANK_RATIOS,
        )
        self.assertTrue(
            all(
                call.kwargs["parameters"].compression_ratio
                in COMPRESSION_RATIOS
                for call in simulate_cycle_mock.call_args_list
            )
        )

        summary = summarize_sensitivity_results(results)
        self.assertIn("minimum_compression_ratio", summary.columns)
        self.assertIn(
            "maximum_connecting_rod_to_crank_ratio",
            summary.columns,
        )


if __name__ == "__main__":
    unittest.main()
