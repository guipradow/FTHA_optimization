"""Unit tests for multiobjective orchestration and result aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from src.multiobjective_optimization import (
    AlgorithmRunResult,
    build_result_tables,
    crowding_distances,
    denormalize_decisions,
    nondominated_indices,
    run_experiment,
)


class TestMultiobjectiveOptimization(unittest.TestCase):
    def test_denormalize_decisions_maps_unit_square_to_study_bounds(self) -> None:
        np.testing.assert_allclose(
            denormalize_decisions([0.0, 0.0]), [500.0, -120.0]
        )
        np.testing.assert_allclose(
            denormalize_decisions([1.0, 1.0]), [10_000.0, 0.0]
        )

    def test_nondominated_indices_remove_dominated_and_duplicate_rows(self) -> None:
        objectives = np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]
        )
        np.testing.assert_array_equal(
            nondominated_indices(objectives), [0, 1, 2]
        )

    def test_crowding_distance_preserves_both_extremes(self) -> None:
        objectives = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        distances = crowding_distances(objectives)
        self.assertTrue(np.isinf(distances[0]))
        self.assertTrue(np.isinf(distances[2]))
        self.assertAlmostEqual(distances[1], 2.0)

    def test_result_tables_report_run_mean_standard_deviation_and_best(self) -> None:
        results = [
            AlgorithmRunResult(
                algorithm="NSGA-II",
                framework="DEAP",
                run=run,
                seed=100 + run,
                runtime_seconds=float(run),
                evaluations=12,
                normalized_decisions=np.array([[0.0, 1.0], [1.0, 0.0]]),
                scaled_objectives=np.array(
                    [[-0.9 - 0.01 * run, -0.2], [-0.5, -0.9 - 0.01 * run]]
                ),
            )
            for run in (1, 2)
        ]
        pareto, per_run, summary, best = build_result_tables(results)

        self.assertEqual(len(pareto), 4)
        self.assertEqual(len(per_run), 2)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(best), 1)
        self.assertEqual(summary.loc[0, "runs"], 2)
        self.assertAlmostEqual(summary.loc[0, "runtime_seconds_mean"], 1.5)
        self.assertAlmostEqual(
            summary.loc[0, "runtime_seconds_std"], np.sqrt(0.5)
        )
        self.assertEqual(summary.loc[0, "evaluations_per_run_mean"], 12.0)

    def test_invalid_population_size_is_rejected_before_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            run_experiment(
                runs=1,
                population_size=6,
                generations=1,
                workers=1,
                algorithm_names=("NSGA-II",),
            )


if __name__ == "__main__":
    unittest.main()
