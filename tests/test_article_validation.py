"""Regression tests for the variable-delta cases published in the article."""

import unittest
from pathlib import Path

import numpy as np

from src.article_validation import (
    ARTICLE_HEAT_ADDITION_ANGLES_DEGREES,
    ARTICLE_PUBLISHED_EFFICIENCY_DECIMAL_PLACES,
    ARTICLE_PUBLISHED_EFFICIENCIES_PERCENT,
    LOG_PRESSURE_VOLUME_FIGURE,
    PRESSURE_ANGLE_FIGURE,
    PRESSURE_VOLUME_FIGURE,
    TEMPERATURE_VOLUME_FIGURE,
    article_crank_angle_grid_rad,
    calculate_article_validation_results,
)
from src.base_case_analysis import (
    ARTICLE_HEAT_ADDITION_STEP_DEGREES,
    ARTICLE_STROKE_INTERVALS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArticleValidationTests(unittest.TestCase):
    def test_published_efficiencies_are_reproduced_at_reported_precision(
        self,
    ) -> None:
        results = calculate_article_validation_results()
        efficiencies_percent = [
            100.0 * result.metrics.thermal_efficiency
            for result in results.values()
        ]
        maximum_pressures = [
            result.metrics.maximum_pressure_kpa for result in results.values()
        ]

        self.assertEqual(
            tuple(results),
            ARTICLE_HEAT_ADDITION_ANGLES_DEGREES,
        )
        np.testing.assert_array_equal(
            np.round(
                efficiencies_percent,
                decimals=ARTICLE_PUBLISHED_EFFICIENCY_DECIMAL_PLACES,
            ),
            ARTICLE_PUBLISHED_EFFICIENCIES_PERCENT,
        )
        self.assertTrue(np.all(np.diff(efficiencies_percent) < 0.0))
        self.assertTrue(np.all(np.diff(maximum_pressures) < 0.0))

    def test_article_mesh_sizes(self) -> None:
        for heat_addition_angle_degrees in ARTICLE_HEAT_ADDITION_ANGLES_DEGREES:
            expected_states = (
                2 * ARTICLE_STROKE_INTERVALS
                + round(
                    heat_addition_angle_degrees
                    / ARTICLE_HEAT_ADDITION_STEP_DEGREES
                )
                + 1
            )
            self.assertEqual(
                article_crank_angle_grid_rad(
                    heat_addition_angle_degrees
                ).size,
                expected_states,
            )

    def test_validation_figure_paths(self) -> None:
        expected_paths = (
            LOG_PRESSURE_VOLUME_FIGURE,
            PRESSURE_VOLUME_FIGURE,
            PRESSURE_ANGLE_FIGURE,
            TEMPERATURE_VOLUME_FIGURE,
        )
        self.assertTrue(
            all(path.parent == PROJECT_ROOT / "img" for path in expected_paths)
        )


if __name__ == "__main__":
    unittest.main()
