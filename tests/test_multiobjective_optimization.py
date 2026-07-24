"""Unit tests for multiobjective orchestration and result aggregation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multiobjective_optimization import (
    AlgorithmRunResult,
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    MOPSO_COGNITIVE,
    MOPSO_INERTIA,
    MOPSO_SOCIAL,
    MUTATION_PROBABILITY_PER_VARIABLE,
    N_DECISION_VARIABLES,
    _evaluate_normalized,
    build_fixed_cardinality_analysis,
    build_result_tables,
    configuration_table,
    crowding_distances,
    denormalize_decisions,
    fixed_cardinality_indices,
    nondominated_indices,
    run_experiment,
)


class TestMultiobjectiveOptimization(unittest.TestCase):
    @patch(
        "src.multiobjective_optimization.simulate_cycle",
        side_effect=RuntimeError("Polytropic iteration did not converge."),
    )
    def test_nonconvergent_model_point_receives_dominated_penalty(self, _) -> None:
        self.assertEqual(_evaluate_normalized([0.5, 0.5, 0.5, 0.5]), (0.0, 0.0))

    @patch(
        "src.multiobjective_optimization.simulate_cycle",
        side_effect=RuntimeError("unexpected model failure"),
    )
    def test_unrelated_runtime_error_is_not_hidden(self, _) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected model failure"):
            _evaluate_normalized([0.5, 0.5, 0.5, 0.5])

    def test_denormalize_decisions_maps_unit_hypercube_to_study_bounds(self) -> None:
        np.testing.assert_allclose(
            denormalize_decisions([0.0, 0.0, 0.0, 0.0]),
            [500.0, -120.0, 8.0, 3.2],
        )
        np.testing.assert_allclose(
            denormalize_decisions([1.0, 1.0, 1.0, 1.0]),
            [10_000.0, 0.0, 12.0, 4.4],
        )

    def test_baseline_budget_and_dimension_dependent_parameters(self) -> None:
        self.assertEqual(N_DECISION_VARIABLES, 4)
        self.assertEqual(DEFAULT_POPULATION_SIZE * (DEFAULT_GENERATIONS + 1), 4_848)
        self.assertEqual(MUTATION_PROBABILITY_PER_VARIABLE, 0.25)
        self.assertEqual(
            (MOPSO_INERTIA, MOPSO_COGNITIVE, MOPSO_SOCIAL),
            (0.4, 1.0, 1.0),
        )
        configuration = configuration_table(
            runs=21,
            population_size=DEFAULT_POPULATION_SIZE,
            generations=DEFAULT_GENERATIONS,
            base_seed=1,
            workers=1,
        )
        self.assertEqual(configuration.loc[0, "nominal_evaluations_per_run"], 4_848)
        self.assertIn("compression_ratio_lower_bound", configuration.columns)
        self.assertIn(
            "connecting_rod_to_crank_ratio_upper_bound",
            configuration.columns,
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

    def test_fixed_cardinality_pruning_is_deterministic_and_keeps_extremes(
        self,
    ) -> None:
        objectives = np.array(
            [
                [0.0, 1.0],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.6, 0.4],
                [0.8, 0.2],
                [1.0, 0.0],
            ]
        )
        first = fixed_cardinality_indices(objectives, maximum_size=4)
        second = fixed_cardinality_indices(objectives, maximum_size=4)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertIn(0, first)
        self.assertIn(5, first)

    def test_fixed_cardinality_analysis_builds_statistics_and_median_runs(
        self,
    ) -> None:
        pareto_records = []
        run_records = []
        for algorithm_index, algorithm_name in enumerate(("A", "B")):
            for run in (1, 2, 3):
                point_count = 4 if algorithm_name == "A" else 3
                efficiency_shift = 0.5 * run - algorithm_index
                for solution in range(point_count):
                    pareto_records.append(
                        {
                            "algorithm": algorithm_name,
                            "framework": "test",
                            "run": run,
                            "seed": 100 * algorithm_index + run,
                            "solution": solution + 1,
                            "engine_speed_rpm": 1_000.0 + solution,
                            "ignition_timing_degrees": -10.0 - solution,
                            "compression_ratio": 10.0,
                            "connecting_rod_to_crank_ratio": 4.0,
                            "thermal_efficiency_percent": (
                                20.0 + 5.0 * solution + efficiency_shift
                            ),
                            "net_specific_power_kw_per_kg": (
                                400.0 - 100.0 * solution
                            ),
                            "compromise_score": 0.0,
                            "is_run_compromise": False,
                        }
                    )
                run_records.append(
                    {
                        "algorithm": algorithm_name,
                        "framework": "test",
                        "run": run,
                        "seed": 100 * algorithm_index + run,
                        "evaluations": 10,
                        "front_size": point_count,
                        "hypervolume": 0.5,
                        "runtime_seconds": 1.0,
                    }
                )

        analysis = build_fixed_cardinality_analysis(
            pd.DataFrame.from_records(pareto_records),
            pd.DataFrame.from_records(run_records),
            maximum_size=3,
        )

        self.assertEqual(len(analysis.pareto), 18)
        self.assertTrue(
            (
                analysis.runs.loc[
                    analysis.runs["algorithm"] == "A", "front_size_hv48"
                ]
                == 3
            ).all()
        )
        self.assertEqual(len(analysis.summary), 2)
        self.assertEqual(len(analysis.global_test), 1)
        self.assertEqual(
            analysis.global_test.loc[0, "degrees_of_freedom"], 1
        )
        self.assertEqual(len(analysis.pairwise_tests), 1)
        self.assertEqual(
            int(analysis.runs["is_hv48_representative_run"].sum()), 2
        )

    def test_result_tables_report_run_mean_standard_deviation_and_best(self) -> None:
        results = [
            AlgorithmRunResult(
                algorithm="NSGA-II",
                framework="DEAP",
                run=run,
                seed=100 + run,
                runtime_seconds=float(run),
                evaluations=12,
                normalized_decisions=np.array(
                    [[0.0, 1.0, 0.25, 0.75], [1.0, 0.0, 0.75, 0.25]]
                ),
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
        self.assertIn("compression_ratio", pareto.columns)
        self.assertIn("connecting_rod_to_crank_ratio", pareto.columns)
        self.assertIn("compression_ratio_mean", summary.columns)
        self.assertIn("best_connecting_rod_to_crank_ratio", summary.columns)

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
