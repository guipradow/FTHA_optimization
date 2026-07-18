"""Tests for the fixed-budget NSGA-III sensitivity design."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.nsga3_sensitivity_analysis import (
    BASELINE_PARAMETER_TUPLE,
    CROSSOVER_ETA_LEVELS,
    CROSSOVER_PROBABILITY_LEVELS,
    EVALUATIONS_PER_RUN,
    MUTATION_ETA_LEVELS,
    MUTATION_PROBABILITY_LEVELS,
    POPULATION_GENERATION_LEVELS,
    SensitivityRunResult,
    _holm_adjust,
    _load_checkpoint,
    _safe_evaluate_normalized,
    _save_checkpoint,
    baseline_configuration_id,
    build_numerical_tables,
    generate_screening_design,
    select_confirmation_configuration_ids,
)


class TestNSGA3SensitivityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = generate_screening_design()

    def test_design_is_unique_and_contains_the_baseline(self) -> None:
        self.assertEqual(len(self.design), 36)
        self.assertEqual(len({item.parameter_tuple() for item in self.design}), 36)
        baseline_id = baseline_configuration_id(self.design)
        baseline = next(
            item for item in self.design if item.configuration_id == baseline_id
        )
        self.assertEqual(baseline.parameter_tuple(), BASELINE_PARAMETER_TUPLE)

    def test_every_factor_level_is_balanced(self) -> None:
        population_counts = Counter(item.population_size for item in self.design)
        self.assertEqual(
            population_counts,
            Counter({population: 9 for population, _ in POPULATION_GENERATION_LEVELS}),
        )
        for values, expected_levels in (
            (
                [item.crossover_probability for item in self.design],
                CROSSOVER_PROBABILITY_LEVELS,
            ),
            ([item.crossover_eta for item in self.design], CROSSOVER_ETA_LEVELS),
            ([item.mutation_eta for item in self.design], MUTATION_ETA_LEVELS),
            (
                [item.mutation_probability_per_variable for item in self.design],
                MUTATION_PROBABILITY_LEVELS,
            ),
        ):
            self.assertEqual(
                Counter(values),
                Counter({level: 12 for level in expected_levels}),
            )

    def test_all_population_generation_pairs_use_the_same_budget(self) -> None:
        for configuration in self.design:
            self.assertEqual(configuration.evaluations, EVALUATIONS_PER_RUN)
            self.assertEqual(
                configuration.reference_partitions,
                configuration.population_size - 1,
            )

    def test_confirmation_selection_includes_baseline_top_three_and_worst(self) -> None:
        results = []
        for index, configuration in enumerate(self.design, start=1):
            quality = 0.4 + index / 100.0
            results.append(
                SensitivityRunResult(
                    configuration_id=configuration.configuration_id,
                    run=1,
                    seed=1,
                    runtime_seconds=1.0,
                    evaluations=EVALUATIONS_PER_RUN,
                    normalized_decisions=np.array([[0.5, 0.5]]),
                    scaled_objectives=np.array([[-quality, -quality]]),
                    convergence=np.array([[0.0, 24.0, quality**2]]),
                )
            )
        selected = select_confirmation_configuration_ids(self.design, results)
        expected_top = {
            item.configuration_id for item in self.design[-3:]
        }
        self.assertEqual(len(selected), 5)
        self.assertIn(baseline_configuration_id(self.design), selected)
        self.assertTrue(expected_top.issubset(selected))
        self.assertIn(self.design[0].configuration_id, selected)

    def test_holm_adjustment_is_monotonic_in_sorted_p_values(self) -> None:
        adjusted = _holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    @patch(
        "src.nsga3_sensitivity_analysis._evaluate_normalized",
        side_effect=RuntimeError("no convergence"),
    )
    def test_nonconvergent_model_point_receives_dominated_penalty(self, _) -> None:
        self.assertEqual(_safe_evaluate_normalized([0.5, 0.5]), (0.0, 0.0))

    @patch(
        "src.nsga3_sensitivity_analysis._evaluate_normalized",
        side_effect=RuntimeError("unexpected model failure"),
    )
    def test_unrelated_runtime_error_is_not_hidden(self, _) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected model failure"):
            _safe_evaluate_normalized([0.5, 0.5])

    def test_checkpoint_uses_portable_records(self) -> None:
        result = self._synthetic_result(self.design[0], 1, 0.75)
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pkl"
            with (
                patch(
                    "src.nsga3_sensitivity_analysis.REPORTS_DIRECTORY",
                    Path(temporary_directory),
                ),
                patch(
                    "src.nsga3_sensitivity_analysis.CHECKPOINT_PATH",
                    checkpoint_path,
                ),
            ):
                _save_checkpoint(self.design, [result])
                restored = _load_checkpoint(self.design)

        self.assertEqual(len(restored), 1)
        self.assertIsInstance(restored[0], SensitivityRunResult)
        np.testing.assert_array_equal(
            restored[0].scaled_objectives,
            result.scaled_objectives,
        )

    def test_numerical_tables_cover_screening_and_held_out_confirmation(
        self,
    ) -> None:
        screening_results = []
        for configuration_index, configuration in enumerate(self.design):
            for run in range(1, 8):
                quality = 0.60 + configuration_index / 1_000 + run / 100_000
                screening_results.append(
                    self._synthetic_result(configuration, run, quality)
                )
        selected = select_confirmation_configuration_ids(
            self.design, screening_results
        )
        configuration_map = {
            item.configuration_id: item for item in self.design
        }
        results = list(screening_results)
        for configuration_id in selected:
            configuration = configuration_map[configuration_id]
            for run in range(8, 22):
                quality = (
                    0.60
                    + self.design.index(configuration) / 1_000
                    + run / 100_000
                )
                results.append(self._synthetic_result(configuration, run, quality))

        tables = build_numerical_tables(self.design, selected, results)

        self.assertEqual(len(tables["runs"]), 322)
        self.assertEqual(len(tables["summary"]), 36)
        self.assertEqual(len(tables["pairwise"]), 4)
        self.assertEqual(len(tables["best"]), 1)
        self.assertEqual(int(tables["runs"]["evaluations"].min()), 504)

    @staticmethod
    def _synthetic_result(configuration, run: int, quality: float):
        objectives = np.array(
            [[-quality, -0.25], [-0.35, -quality]], dtype=float
        )
        return SensitivityRunResult(
            configuration_id=configuration.configuration_id,
            run=run,
            seed=run,
            runtime_seconds=1.0,
            evaluations=EVALUATIONS_PER_RUN,
            normalized_decisions=np.array([[0.25, 0.75], [0.75, 0.25]]),
            scaled_objectives=objectives,
            convergence=np.array(
                [
                    [0.0, configuration.population_size, 0.1],
                    [configuration.generations, EVALUATIONS_PER_RUN, 0.2],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
