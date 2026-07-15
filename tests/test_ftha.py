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
from src.FTHA import objective_function, simulate_cycle
from src.gas_prop import _DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ObjectiveFunctionTests(unittest.TestCase):
    def test_project_paths_follow_src_layout(self) -> None:
        self.assertEqual(_DATA_PATH, PROJECT_ROOT / "data" / "data.csv")
        self.assertTrue(_DATA_PATH.is_file())
        self.assertEqual(IMAGES_DIRECTORY, PROJECT_ROOT / "img")

    def test_report_reference_case(self) -> None:
        objectives = objective_function([4_500.0, -48.0])

        np.testing.assert_allclose(
            objectives,
            [
                -0.521110813637,
                -8_402.911869889,
                0.663950657834,
                4_630.584052030,
                1_330.923996900,
            ],
            rtol=1e-8,
        )

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
        with self.assertRaisesRegex(ValueError, "outside the notebook study bounds"):
            objective_function([10_001.0, -48.0])

    def test_objective_rejects_wrong_decision_vector_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            objective_function([4_500.0])


if __name__ == "__main__":
    unittest.main()
