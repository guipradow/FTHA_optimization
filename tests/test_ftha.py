"""Regression and interface tests for the FTHA objective model."""

import unittest
from pathlib import Path

import numpy as np

from src.base_case_analysis import (
    ARTICLE_BASE_PARAMETERS,
    BASE_HEAT_ADDITION_ANGLE_DEGREES,
    BASE_IGNITION_TIMING_DEGREES,
    IMAGES_DIRECTORY,
    _article_crank_angle_grid_rad,
)
from src.FTHA import (
    DECISION_LOWER_BOUNDS,
    DECISION_UPPER_BOUNDS,
    DECISION_VARIABLE_NAMES,
    denormalize_decisions,
    normalize_decisions,
    objective_function,
    simulate_cycle,
)
from src.gas_prop import _DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ObjectiveFunctionTests(unittest.TestCase):
    def test_project_paths_follow_src_layout(self) -> None:
        self.assertEqual(_DATA_PATH, PROJECT_ROOT / "data" / "data.csv")
        self.assertTrue(_DATA_PATH.is_file())
        self.assertEqual(IMAGES_DIRECTORY, PROJECT_ROOT / "img")

    def test_report_reference_case(self) -> None:
        objectives = objective_function([4_500.0, -48.0, 12.0, 4.4])

        np.testing.assert_allclose(
            objectives,
            [
                -0.520617321067,
                -8_394.954302207,
                0.664414532972,
                4_632.436525561,
                1_331.457126156,
            ],
            rtol=1e-8,
        )

    def test_four_decision_variables_and_bounds_are_explicit(self) -> None:
        self.assertEqual(
            DECISION_VARIABLE_NAMES,
            (
                "engine_speed_rpm",
                "ignition_timing_degrees",
                "compression_ratio",
                "connecting_rod_to_crank_ratio",
            ),
        )
        np.testing.assert_array_equal(
            DECISION_LOWER_BOUNDS,
            [500.0, -120.0, 8.0, 3.2],
        )
        np.testing.assert_array_equal(
            DECISION_UPPER_BOUNDS,
            [10_000.0, 0.0, 12.0, 4.4],
        )

    def test_decision_normalization_maps_midpoints_and_round_trips(self) -> None:
        physical_decisions = np.array([5_250.0, -60.0, 10.0, 3.8])

        normalized = normalize_decisions(physical_decisions)

        np.testing.assert_allclose(normalized, np.full(4, 0.5), atol=1e-15)
        np.testing.assert_allclose(
            denormalize_decisions(normalized),
            physical_decisions,
            atol=1e-15,
        )

    def test_compression_ratio_changes_cycle_performance(self) -> None:
        low_compression = objective_function([4_800.0, -15.0, 8.0, 3.8])
        high_compression = objective_function([4_800.0, -15.0, 12.0, 3.8])

        # With all other inputs fixed, the higher compression ratio improves
        # both maximized indicators (represented here by more-negative values).
        self.assertLess(high_compression[0], low_compression[0])
        self.assertLess(high_compression[1], low_compression[1])
        self.assertNotAlmostEqual(high_compression[3], low_compression[3])

    def test_connecting_rod_ratio_changes_cycle_performance(self) -> None:
        short_rod = objective_function([4_800.0, -15.0, 10.0, 3.2])
        long_rod = objective_function([4_800.0, -15.0, 10.0, 4.4])

        self.assertGreater(abs(long_rod[0] - short_rod[0]), 1e-4)
        self.assertGreater(abs(long_rod[1] - short_rod[1]), 1.0)

    def test_state_and_interval_lengths_are_consistent(self) -> None:
        result = simulate_cycle(4_500.0, -48.0)

        self.assertEqual(result.crank_angle_rad.size, 720)
        self.assertEqual(result.pressure_kpa.size, 720)
        self.assertEqual(result.temperature_k.size, 720)
        self.assertEqual(result.work_on_gas_kj_per_kg.size, 719)
        self.assertEqual(result.heat_added_kj_per_kg.size, 719)
        self.assertEqual(result.polytropic_exponent.size, 719)

    def test_article_base_case(self) -> None:
        result = simulate_cycle(
            engine_speed_rpm=None,
            ignition_timing_degrees=BASE_IGNITION_TIMING_DEGREES,
            parameters=ARTICLE_BASE_PARAMETERS,
            heat_addition_angle_degrees=BASE_HEAT_ADDITION_ANGLE_DEGREES,
            crank_angle_grid_rad=_article_crank_angle_grid_rad(),
        )

        self.assertEqual(result.crank_angle_rad.size, 201)
        self.assertAlmostEqual(result.metrics.thermal_efficiency, 0.38409351, 7)
        self.assertTrue(np.isnan(result.metrics.net_specific_power_kw_per_kg))

    def test_objective_rejects_values_outside_study_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the design-space bounds"):
            objective_function([10_001.0, -48.0, 10.0, 3.8])

        with self.assertRaisesRegex(ValueError, "outside the design-space bounds"):
            objective_function([4_500.0, -48.0, 12.1, 3.8])

        with self.assertRaisesRegex(ValueError, "outside the unit-hypercube bounds"):
            denormalize_decisions([0.5, 0.5, 0.5, 1.01])

    def test_objective_rejects_wrong_decision_vector_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            objective_function([4_500.0, -48.0])


if __name__ == "__main__":
    unittest.main()
