"""Compare four optimizers for efficiency-power FTHA optimization.

The two benefits are converted to scaled minimization objectives.  Every
thermodynamic evaluation uses the physical parameters and the variable-size
crank-angle grid from the article-based case study.

NSGA-II and NSGA-III are provided by DEAP, MOEA/D by pymoo, and MOPSO extends
the PySwarm velocity update with the external nondominated repository proposed
by Coello Coello and Lechuga.  PySwarm itself exposes a scalar objective API,
so the repository and Pareto leader-selection layer are implemented here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Each thermodynamic evaluation is single-threaded at the Python level.  Limit
# BLAS inside spawned workers to avoid multiplying eight processes by the BLAS
# default thread count on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions
from pyswarm import pso

from .FTHA import DECISION_LOWER_BOUNDS, DECISION_UPPER_BOUNDS, simulate_cycle
from .sensitivity_analysis import (
    BLACK_AND_WHITE_SERIES_STYLES,
    CASE_STUDY_PARAMETERS,
    CHART_STYLE,
    case_study_crank_angle_grid_rad,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
IMAGES_DIRECTORY = PROJECT_ROOT / "img"

PARETO_SOLUTIONS_PATH = REPORTS_DIRECTORY / "multiobjective_pareto_solutions.csv"
RUN_STATISTICS_PATH = REPORTS_DIRECTORY / "multiobjective_run_statistics.csv"
SUMMARY_PATH = REPORTS_DIRECTORY / "multiobjective_summary.csv"
BEST_SOLUTIONS_PATH = REPORTS_DIRECTORY / "multiobjective_best_solutions.csv"
CONFIGURATION_PATH = REPORTS_DIRECTORY / "multiobjective_configuration.csv"
PARETO_FIGURE_PATH = IMAGES_DIRECTORY / "multiobjective_pareto_front.png"
RUNTIME_FIGURE_PATH = IMAGES_DIRECTORY / "multiobjective_runtime_boxplot.png"

ALGORITHM_NAMES = ("NSGA-II", "NSGA-III", "MOPSO", "MOEA/D")
FRAMEWORK_NAMES = {
    "NSGA-II": "DEAP",
    "NSGA-III": "DEAP",
    "MOPSO": "PySwarm + Pareto repository",
    "MOEA/D": "pymoo",
}

DEFAULT_RUNS = 21
DEFAULT_POPULATION_SIZE = 24
DEFAULT_GENERATIONS = 20
DEFAULT_BASE_SEED = 20_260_718
DEFAULT_WORKERS = min(8, max(1, mp.cpu_count()))

# Fixed scales keep both objective magnitudes of order one.  They are rounded
# upper limits based on the preceding sensitivity study (38.280% and
# 25,672.8 kW/kg), not fitted separately for each stochastic run.
EFFICIENCY_SCALE_PERCENT = 40.0
POWER_SCALE_KW_PER_KG = 27_000.0

SBX_PROBABILITY = 0.9
SBX_DISTRIBUTION_INDEX = 20.0
MUTATION_DISTRIBUTION_INDEX = 20.0
MUTATION_PROBABILITY_PER_VARIABLE = 0.5
MOEAD_NEIGHBORS = 10
MOEAD_NEIGHBOR_MATING_PROBABILITY = 0.9

# Reuse the defaults exposed by the installed PySwarm implementation.  The
# MOPSO extension changes leader selection, not the velocity equation.
_PYSWARM_SIGNATURE = inspect.signature(pso)
MOPSO_INERTIA = float(_PYSWARM_SIGNATURE.parameters["omega"].default)
MOPSO_COGNITIVE = float(_PYSWARM_SIGNATURE.parameters["phip"].default)
MOPSO_SOCIAL = float(_PYSWARM_SIGNATURE.parameters["phig"].default)
MOPSO_MUTATION_PROBABILITY = 0.10


if not hasattr(creator, "FTHABiObjectiveFitness"):
    creator.create("FTHABiObjectiveFitness", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "FTHABiObjectiveIndividual"):
    creator.create(
        "FTHABiObjectiveIndividual",
        list,
        fitness=creator.FTHABiObjectiveFitness,
    )


@dataclass(slots=True)
class AlgorithmRunResult:
    """Final nondominated set and execution metadata for one run."""

    algorithm: str
    framework: str
    run: int
    seed: int
    runtime_seconds: float
    evaluations: int
    normalized_decisions: np.ndarray
    scaled_objectives: np.ndarray


def denormalize_decisions(normalized_decisions: Sequence[float]) -> np.ndarray:
    """Map a point from the unit square to ``[N, theta]``."""
    normalized = np.asarray(normalized_decisions, dtype=float)
    if normalized.shape != (2,):
        raise ValueError("normalized_decisions must contain exactly two values.")
    return DECISION_LOWER_BOUNDS + normalized * (
        DECISION_UPPER_BOUNDS - DECISION_LOWER_BOUNDS
    )


def _evaluate_normalized(normalized_decisions: Sequence[float]) -> tuple[float, float]:
    """Return scaled minimization objectives for one normalized point."""
    engine_speed_rpm, ignition_timing_degrees = denormalize_decisions(
        normalized_decisions
    )
    result = simulate_cycle(
        engine_speed_rpm=float(engine_speed_rpm),
        ignition_timing_degrees=float(ignition_timing_degrees),
        parameters=CASE_STUDY_PARAMETERS,
        crank_angle_grid_rad=case_study_crank_angle_grid_rad(
            float(engine_speed_rpm),
            float(ignition_timing_degrees),
            CASE_STUDY_PARAMETERS,
        ),
    )
    return (
        -100.0
        * result.metrics.thermal_efficiency
        / EFFICIENCY_SCALE_PERCENT,
        -result.metrics.net_specific_power_kw_per_kg / POWER_SCALE_KW_PER_KG,
    )


def _evaluate_many(
    points: np.ndarray,
    pool: mp.pool.Pool | None,
) -> np.ndarray:
    values = (
        list(map(_evaluate_normalized, points))
        if pool is None
        else pool.map(_evaluate_normalized, list(points))
    )
    return np.asarray(values, dtype=float)


def nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Return indices of unique nondominated rows for minimization."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("objectives must have shape (n, 2).")
    keep: list[int] = []
    seen: set[tuple[float, float]] = set()
    for index, candidate in enumerate(values):
        key = tuple(np.round(candidate, decimals=14))
        if key in seen:
            continue
        seen.add(key)
        dominated = np.any(
            np.all(values <= candidate, axis=1)
            & np.any(values < candidate, axis=1)
        )
        if not dominated:
            keep.append(index)
    return np.asarray(keep, dtype=int)


def _clean_front(
    normalized_decisions: np.ndarray,
    scaled_objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = nondominated_indices(scaled_objectives)
    decisions = np.asarray(normalized_decisions, dtype=float)[indices]
    objectives = np.asarray(scaled_objectives, dtype=float)[indices]
    order = np.argsort(objectives[:, 0], kind="stable")
    return decisions[order], objectives[order]


def _assign_fitness(population: Iterable, pool: mp.pool.Pool | None) -> int:
    invalid = [individual for individual in population if not individual.fitness.valid]
    if not invalid:
        return 0
    values = _evaluate_many(np.asarray(invalid, dtype=float), pool)
    for individual, fitness in zip(invalid, values, strict=True):
        individual.fitness.values = tuple(fitness)
    return len(invalid)


def _make_deap_toolbox(pool: mp.pool.Pool | None) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.FTHABiObjectiveIndividual,
        toolbox.attr_float,
        2,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate_normalized)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[0.0, 0.0],
        up=[1.0, 1.0],
        eta=SBX_DISTRIBUTION_INDEX,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[0.0, 0.0],
        up=[1.0, 1.0],
        eta=MUTATION_DISTRIBUTION_INDEX,
        indpb=MUTATION_PROBABILITY_PER_VARIABLE,
    )
    if pool is not None:
        toolbox.register("map", pool.map)
    return toolbox


def run_nsga2(
    seed: int,
    population_size: int,
    generations: int,
    pool: mp.pool.Pool | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run the canonical elitist DEAP NSGA-II loop."""
    random.seed(seed)
    np.random.seed(seed)
    toolbox = _make_deap_toolbox(pool)
    start = time.perf_counter()
    population = toolbox.population(n=population_size)
    evaluations = _assign_fitness(population, pool)
    population = tools.selNSGA2(population, population_size)

    for _ in range(generations):
        offspring = tools.selTournamentDCD(population, population_size)
        offspring = [toolbox.clone(individual) for individual in offspring]
        for first, second in zip(offspring[::2], offspring[1::2], strict=True):
            if random.random() <= SBX_PROBABILITY:
                toolbox.mate(first, second)
            toolbox.mutate(first)
            toolbox.mutate(second)
            del first.fitness.values, second.fitness.values
        evaluations += _assign_fitness(offspring, pool)
        population = tools.selNSGA2(population + offspring, population_size)

    runtime = time.perf_counter() - start
    decisions = np.asarray(population, dtype=float)
    objectives = np.asarray(
        [individual.fitness.values for individual in population], dtype=float
    )
    decisions, objectives = _clean_front(decisions, objectives)
    return decisions, objectives, evaluations, runtime


def run_nsga3(
    seed: int,
    population_size: int,
    generations: int,
    pool: mp.pool.Pool | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run reference-point NSGA-III using DEAP selection."""
    random.seed(seed)
    np.random.seed(seed)
    toolbox = _make_deap_toolbox(pool)
    reference_points = tools.uniform_reference_points(
        nobj=2, p=population_size - 1
    )
    start = time.perf_counter()
    population = toolbox.population(n=population_size)
    evaluations = _assign_fitness(population, pool)
    population = tools.selNSGA3(
        population, population_size, ref_points=reference_points
    )

    for _ in range(generations):
        offspring = algorithms.varAnd(
            population,
            toolbox,
            cxpb=SBX_PROBABILITY,
            mutpb=1.0,
        )
        evaluations += _assign_fitness(offspring, pool)
        population = tools.selNSGA3(
            population + offspring,
            population_size,
            ref_points=reference_points,
        )

    runtime = time.perf_counter() - start
    decisions = np.asarray(population, dtype=float)
    objectives = np.asarray(
        [individual.fitness.values for individual in population], dtype=float
    )
    decisions, objectives = _clean_front(decisions, objectives)
    return decisions, objectives, evaluations, runtime


def crowding_distances(objectives: np.ndarray) -> np.ndarray:
    """Calculate NSGA-II crowding distances for a nondominated archive."""
    values = np.asarray(objectives, dtype=float)
    number_of_points = len(values)
    distances = np.zeros(number_of_points, dtype=float)
    if number_of_points <= 2:
        distances.fill(np.inf)
        return distances
    for objective_index in range(values.shape[1]):
        order = np.argsort(values[:, objective_index], kind="stable")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = values[order[-1], objective_index] - values[order[0], objective_index]
        if np.isclose(span, 0.0):
            continue
        interior = order[1:-1]
        distances[interior] += (
            values[order[2:], objective_index]
            - values[order[:-2], objective_index]
        ) / span
    return distances


def _update_archive(
    archive_positions: np.ndarray,
    archive_objectives: np.ndarray,
    positions: np.ndarray,
    objectives: np.ndarray,
    maximum_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    all_positions = np.vstack((archive_positions, positions))
    all_objectives = np.vstack((archive_objectives, objectives))
    indices = nondominated_indices(all_objectives)
    archive_positions = all_positions[indices]
    archive_objectives = all_objectives[indices]

    while len(archive_positions) > maximum_size:
        crowding = crowding_distances(archive_objectives)
        finite = np.flatnonzero(np.isfinite(crowding))
        if finite.size:
            minimum = crowding[finite].min()
            candidates = finite[np.isclose(crowding[finite], minimum)]
        else:
            candidates = np.arange(len(archive_positions))
        remove_index = int(rng.choice(candidates))
        archive_positions = np.delete(archive_positions, remove_index, axis=0)
        archive_objectives = np.delete(archive_objectives, remove_index, axis=0)
    return archive_positions, archive_objectives


def _select_mopso_leaders(
    archive_positions: np.ndarray,
    archive_objectives: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    crowding = crowding_distances(archive_objectives)
    finite = crowding[np.isfinite(crowding)]
    boundary_weight = 2.0 * finite.max() if finite.size else 1.0
    weights = np.where(np.isfinite(crowding), crowding, boundary_weight)
    weights = np.maximum(weights, np.finfo(float).eps)
    probabilities = weights / weights.sum()
    selected = rng.choice(
        len(archive_positions), size=count, replace=True, p=probabilities
    )
    return archive_positions[selected]


def run_mopso(
    seed: int,
    population_size: int,
    generations: int,
    pool: mp.pool.Pool | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run a Pareto-repository extension of the PySwarm PSO update."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    positions = rng.random((population_size, 2))
    velocities = rng.uniform(-1.0, 1.0, size=(population_size, 2))
    objectives = _evaluate_many(positions, pool)
    evaluations = population_size
    personal_best_positions = positions.copy()
    personal_best_objectives = objectives.copy()
    archive_positions = np.empty((0, 2), dtype=float)
    archive_objectives = np.empty((0, 2), dtype=float)
    archive_positions, archive_objectives = _update_archive(
        archive_positions,
        archive_objectives,
        positions,
        objectives,
        maximum_size=4 * population_size,
        rng=rng,
    )

    for generation in range(generations):
        leaders = _select_mopso_leaders(
            archive_positions, archive_objectives, population_size, rng
        )
        velocities = (
            MOPSO_INERTIA * velocities
            + MOPSO_COGNITIVE
            * rng.random((population_size, 2))
            * (personal_best_positions - positions)
            + MOPSO_SOCIAL
            * rng.random((population_size, 2))
            * (leaders - positions)
        )
        velocities = np.clip(velocities, -1.0, 1.0)
        positions = positions + velocities
        outside = (positions < 0.0) | (positions > 1.0)
        positions = np.clip(positions, 0.0, 1.0)
        velocities[outside] *= -0.5

        mutation_probability = MOPSO_MUTATION_PROBABILITY * (
            1.0 - generation / max(generations, 1)
        )
        mutated = rng.random(population_size) < mutation_probability
        if np.any(mutated):
            dimensions = rng.integers(0, 2, size=int(mutated.sum()))
            rows = np.flatnonzero(mutated)
            positions[rows, dimensions] = np.clip(
                positions[rows, dimensions] + rng.normal(0.0, 0.10, len(rows)),
                0.0,
                1.0,
            )

        objectives = _evaluate_many(positions, pool)
        evaluations += population_size
        for index in range(population_size):
            current_dominates = np.all(
                objectives[index] <= personal_best_objectives[index]
            ) and np.any(objectives[index] < personal_best_objectives[index])
            personal_dominates = np.all(
                personal_best_objectives[index] <= objectives[index]
            ) and np.any(personal_best_objectives[index] < objectives[index])
            if current_dominates or (
                not personal_dominates and rng.random() < 0.5
            ):
                personal_best_positions[index] = positions[index]
                personal_best_objectives[index] = objectives[index]

        archive_positions, archive_objectives = _update_archive(
            archive_positions,
            archive_objectives,
            positions,
            objectives,
            maximum_size=4 * population_size,
            rng=rng,
        )

    runtime = time.perf_counter() - start
    decisions, objectives = _clean_front(
        archive_positions, archive_objectives
    )
    return decisions, objectives, evaluations, runtime


class FTHABiObjectiveProblem(ElementwiseProblem):
    """pymoo adapter for the normalized two-variable FTHA problem."""

    def __init__(self, runner=None) -> None:
        super().__init__(
            n_var=2,
            n_obj=2,
            xl=np.zeros(2),
            xu=np.ones(2),
            elementwise_runner=runner,
        )

    def _evaluate(self, decisions, out, *args, **kwargs) -> None:
        out["F"] = np.asarray(_evaluate_normalized(decisions), dtype=float)


def run_moead(
    seed: int,
    population_size: int,
    generations: int,
    pool: mp.pool.Pool | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run pymoo MOEA/D with uniform two-objective reference directions."""
    reference_directions = get_reference_directions(
        "uniform", 2, n_partitions=population_size - 1
    )
    runner = None if pool is None else StarmapParallelization(pool.starmap)
    problem = FTHABiObjectiveProblem(runner=runner)
    algorithm = MOEAD(
        reference_directions,
        n_neighbors=min(MOEAD_NEIGHBORS, population_size),
        prob_neighbor_mating=MOEAD_NEIGHBOR_MATING_PROBABILITY,
        crossover=SBX(prob=SBX_PROBABILITY, eta=SBX_DISTRIBUTION_INDEX),
        mutation=PM(
            prob=MUTATION_PROBABILITY_PER_VARIABLE,
            eta=MUTATION_DISTRIBUTION_INDEX,
        ),
    )
    start = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations + 1),
        seed=seed,
        verbose=False,
        save_history=False,
    )
    runtime = time.perf_counter() - start
    decisions, objectives = _clean_front(
        np.atleast_2d(result.X), np.atleast_2d(result.F)
    )
    return decisions, objectives, int(result.algorithm.evaluator.n_eval), runtime


RUNNERS: dict[
    str,
    Callable[
        [int, int, int, mp.pool.Pool | None],
        tuple[np.ndarray, np.ndarray, int, float],
    ],
] = {
    "NSGA-II": run_nsga2,
    "NSGA-III": run_nsga3,
    "MOPSO": run_mopso,
    "MOEA/D": run_moead,
}


def run_experiment(
    runs: int = DEFAULT_RUNS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    workers: int = DEFAULT_WORKERS,
    algorithm_names: Sequence[str] = ALGORITHM_NAMES,
) -> list[AlgorithmRunResult]:
    """Execute all stochastic repetitions, interleaving algorithm order."""
    if runs < 1 or generations < 1:
        raise ValueError("runs and generations must be positive.")
    if population_size < 4 or population_size % 4:
        raise ValueError("population_size must be a multiple of four and >= 4.")
    unknown = set(algorithm_names) - set(ALGORITHM_NAMES)
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")

    pool = None
    results: list[AlgorithmRunResult] = []
    try:
        if workers > 1:
            pool = mp.get_context("spawn").Pool(processes=workers)
        names = list(algorithm_names)
        for run_index in range(1, runs + 1):
            offset = (run_index - 1) % len(names)
            ordered_names = names[offset:] + names[:offset]
            for algorithm_name in ordered_names:
                canonical_index = ALGORITHM_NAMES.index(algorithm_name)
                seed = base_seed + 10_000 * canonical_index + run_index
                decisions, objectives, evaluations, runtime = RUNNERS[
                    algorithm_name
                ](seed, population_size, generations, pool)
                result = AlgorithmRunResult(
                    algorithm=algorithm_name,
                    framework=FRAMEWORK_NAMES[algorithm_name],
                    run=run_index,
                    seed=seed,
                    runtime_seconds=runtime,
                    evaluations=evaluations,
                    normalized_decisions=decisions,
                    scaled_objectives=objectives,
                )
                results.append(result)
                print(
                    f"{algorithm_name} run {run_index:02d}/{runs}: "
                    f"{len(objectives)} nondominated solutions, "
                    f"{evaluations} evaluations, {runtime:.2f} s",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return results


def _benefits_from_objectives(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    return np.column_stack(
        (
            -objectives[:, 0] * EFFICIENCY_SCALE_PERCENT,
            -objectives[:, 1] * POWER_SCALE_KW_PER_KG,
        )
    )


def _compromise_scores(
    benefits: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> np.ndarray:
    spans = np.maximum(ideal - nadir, np.finfo(float).eps)
    losses = (ideal - benefits) / spans
    return np.sqrt(np.mean(losses**2, axis=1))


def build_result_tables(
    results: Sequence[AlgorithmRunResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create complete fronts, per-run statistics, summary, and best rows."""
    if not results:
        raise ValueError("At least one result is required.")

    all_objectives = np.vstack([result.scaled_objectives for result in results])
    pooled_indices = nondominated_indices(all_objectives)
    pooled_benefits = _benefits_from_objectives(all_objectives[pooled_indices])
    ideal = pooled_benefits.max(axis=0)
    nadir = pooled_benefits.min(axis=0)
    hypervolume = HV(ref_point=np.zeros(2))

    pareto_records: list[dict[str, float | int | str]] = []
    run_records: list[dict[str, float | int | str]] = []
    for result in results:
        decisions = np.asarray(
            [denormalize_decisions(point) for point in result.normalized_decisions]
        )
        benefits = _benefits_from_objectives(result.scaled_objectives)
        scores = _compromise_scores(benefits, ideal, nadir)
        representative_index = int(np.argmin(scores))
        hv_value = float(hypervolume(result.scaled_objectives))
        for solution_index, (decision, benefit, score) in enumerate(
            zip(decisions, benefits, scores, strict=True), start=1
        ):
            pareto_records.append(
                {
                    "algorithm": result.algorithm,
                    "framework": result.framework,
                    "run": result.run,
                    "seed": result.seed,
                    "solution": solution_index,
                    "engine_speed_rpm": decision[0],
                    "ignition_timing_degrees": decision[1],
                    "thermal_efficiency_percent": benefit[0],
                    "net_specific_power_kw_per_kg": benefit[1],
                    "compromise_score": score,
                    "is_run_compromise": (
                        solution_index - 1 == representative_index
                    ),
                }
            )
        representative_decision = decisions[representative_index]
        representative_benefit = benefits[representative_index]
        run_records.append(
            {
                "algorithm": result.algorithm,
                "framework": result.framework,
                "run": result.run,
                "seed": result.seed,
                "evaluations": result.evaluations,
                "front_size": len(result.scaled_objectives),
                "hypervolume": hv_value,
                "runtime_seconds": result.runtime_seconds,
                "compromise_engine_speed_rpm": representative_decision[0],
                "compromise_ignition_timing_degrees": representative_decision[1],
                "compromise_thermal_efficiency_percent": (
                    representative_benefit[0]
                ),
                "compromise_net_specific_power_kw_per_kg": (
                    representative_benefit[1]
                ),
                "compromise_score": scores[representative_index],
            }
        )

    pareto_table = pd.DataFrame.from_records(pareto_records)
    run_table = pd.DataFrame.from_records(run_records)
    summary_records: list[dict[str, float | int | str]] = []
    best_records: list[dict[str, float | int | str]] = []
    for algorithm_name in ALGORITHM_NAMES:
        algorithm_runs = run_table[run_table["algorithm"] == algorithm_name]
        algorithm_solutions = pareto_table[
            pareto_table["algorithm"] == algorithm_name
        ]
        if algorithm_runs.empty:
            continue
        best = algorithm_solutions.loc[
            algorithm_solutions["compromise_score"].idxmin()
        ]
        best_records.append(best.to_dict())
        summary_records.append(
            {
                "algorithm": algorithm_name,
                "framework": FRAMEWORK_NAMES[algorithm_name],
                "runs": len(algorithm_runs),
                "evaluations_per_run_mean": algorithm_runs["evaluations"].mean(),
                "front_size_mean": algorithm_runs["front_size"].mean(),
                "front_size_std": algorithm_runs["front_size"].std(ddof=1),
                "hypervolume_mean": algorithm_runs["hypervolume"].mean(),
                "hypervolume_std": algorithm_runs["hypervolume"].std(ddof=1),
                "hypervolume_best": algorithm_runs["hypervolume"].max(),
                "runtime_seconds_mean": algorithm_runs["runtime_seconds"].mean(),
                "runtime_seconds_std": algorithm_runs["runtime_seconds"].std(ddof=1),
                "runtime_seconds_min": algorithm_runs["runtime_seconds"].min(),
                "runtime_seconds_max": algorithm_runs["runtime_seconds"].max(),
                "engine_speed_rpm_mean": algorithm_runs[
                    "compromise_engine_speed_rpm"
                ].mean(),
                "engine_speed_rpm_std": algorithm_runs[
                    "compromise_engine_speed_rpm"
                ].std(ddof=1),
                "ignition_timing_degrees_mean": algorithm_runs[
                    "compromise_ignition_timing_degrees"
                ].mean(),
                "ignition_timing_degrees_std": algorithm_runs[
                    "compromise_ignition_timing_degrees"
                ].std(ddof=1),
                "thermal_efficiency_percent_mean": algorithm_runs[
                    "compromise_thermal_efficiency_percent"
                ].mean(),
                "thermal_efficiency_percent_std": algorithm_runs[
                    "compromise_thermal_efficiency_percent"
                ].std(ddof=1),
                "net_specific_power_kw_per_kg_mean": algorithm_runs[
                    "compromise_net_specific_power_kw_per_kg"
                ].mean(),
                "net_specific_power_kw_per_kg_std": algorithm_runs[
                    "compromise_net_specific_power_kw_per_kg"
                ].std(ddof=1),
                "best_engine_speed_rpm": best["engine_speed_rpm"],
                "best_ignition_timing_degrees": best[
                    "ignition_timing_degrees"
                ],
                "best_thermal_efficiency_percent": best[
                    "thermal_efficiency_percent"
                ],
                "best_net_specific_power_kw_per_kg": best[
                    "net_specific_power_kw_per_kg"
                ],
                "best_compromise_score": best["compromise_score"],
                "best_run": best["run"],
                "best_seed": best["seed"],
            }
        )

    summary_table = pd.DataFrame.from_records(summary_records)
    best_table = pd.DataFrame.from_records(best_records)
    return pareto_table, run_table, summary_table, best_table


def configuration_table(
    runs: int,
    population_size: int,
    generations: int,
    base_seed: int,
    workers: int,
) -> pd.DataFrame:
    """Return a machine-readable record of every fixed optimizer choice."""
    return pd.DataFrame.from_records(
        [
            {
                "runs_per_algorithm": runs,
                "population_or_swarm_size": population_size,
                "generations": generations,
                "nominal_evaluations_per_run": population_size
                * (generations + 1),
                "base_seed": base_seed,
                "parallel_workers": workers,
                "engine_speed_lower_rpm": DECISION_LOWER_BOUNDS[0],
                "engine_speed_upper_rpm": DECISION_UPPER_BOUNDS[0],
                "ignition_timing_lower_degrees": DECISION_LOWER_BOUNDS[1],
                "ignition_timing_upper_degrees": DECISION_UPPER_BOUNDS[1],
                "efficiency_scale_percent": EFFICIENCY_SCALE_PERCENT,
                "power_scale_kw_per_kg": POWER_SCALE_KW_PER_KG,
                "sbx_probability": SBX_PROBABILITY,
                "sbx_distribution_index": SBX_DISTRIBUTION_INDEX,
                "mutation_distribution_index": MUTATION_DISTRIBUTION_INDEX,
                "mutation_probability_per_variable": (
                    MUTATION_PROBABILITY_PER_VARIABLE
                ),
                "mopso_inertia": MOPSO_INERTIA,
                "mopso_cognitive": MOPSO_COGNITIVE,
                "mopso_social": MOPSO_SOCIAL,
                "mopso_repository_size": 4 * population_size,
                "moead_neighbors": min(MOEAD_NEIGHBORS, population_size),
                "moead_neighbor_mating_probability": (
                    MOEAD_NEIGHBOR_MATING_PROBABILITY
                ),
                "deap_version": importlib.metadata.version("deap"),
                "pyswarm_version": importlib.metadata.version("pyswarm"),
                "pymoo_version": importlib.metadata.version("pymoo"),
            }
        ]
    )


def _save_pareto_figure(
    pareto_table: pd.DataFrame,
    run_table: pd.DataFrame,
) -> None:
    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for algorithm_index, algorithm_name in enumerate(ALGORITHM_NAMES):
            algorithm_runs = run_table[run_table["algorithm"] == algorithm_name]
            if algorithm_runs.empty:
                continue
            best_run = int(
                algorithm_runs.loc[algorithm_runs["hypervolume"].idxmax(), "run"]
            )
            front = pareto_table[
                (pareto_table["algorithm"] == algorithm_name)
                & (pareto_table["run"] == best_run)
            ].sort_values("thermal_efficiency_percent")
            style = BLACK_AND_WHITE_SERIES_STYLES[algorithm_index]
            axis.plot(
                front["thermal_efficiency_percent"],
                front["net_specific_power_kw_per_kg"],
                color="black",
                linewidth=1.1,
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor="black",
                label=f"{algorithm_name} (execução {best_run})",
                **style,
            )
        axis.set_xlabel(r"Eficiência térmica, $\eta_t$ [%]")
        axis.set_ylabel(r"Potência líquida específica [kW/kg]")
        axis.grid(True, color="0.82", linewidth=0.6, linestyle=":")
        axis.legend(frameon=True, edgecolor="black", fontsize=8)
        figure.savefig(PARETO_FIGURE_PATH, dpi=300)
        plt.close(figure)


def _save_runtime_figure(run_table: pd.DataFrame) -> None:
    names = [name for name in ALGORITHM_NAMES if name in set(run_table["algorithm"])]
    data = [
        run_table.loc[run_table["algorithm"] == name, "runtime_seconds"].to_numpy()
        for name in names
    ]
    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        boxplot = axis.boxplot(
            data,
            tick_labels=names,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
            },
        )
        hatches = ("", "///", "xxx", "...")
        for patch, hatch in zip(boxplot["boxes"], hatches):
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_hatch(hatch)
        axis.set_ylabel("Tempo por execução [s]")
        axis.grid(True, axis="y", color="0.82", linewidth=0.6, linestyle=":")
        figure.savefig(RUNTIME_FIGURE_PATH, dpi=300)
        plt.close(figure)


def save_results(
    results: Sequence[AlgorithmRunResult],
    runs: int,
    population_size: int,
    generations: int,
    base_seed: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist numerical tables and black-and-white-safe figures."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pareto, per_run, summary, best = build_result_tables(results)
    pareto.to_csv(PARETO_SOLUTIONS_PATH, index=False)
    per_run.to_csv(RUN_STATISTICS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    best.to_csv(BEST_SOLUTIONS_PATH, index=False)
    configuration_table(
        runs, population_size, generations, base_seed, workers
    ).to_csv(CONFIGURATION_PATH, index=False)
    _save_pareto_figure(pareto, per_run)
    _save_runtime_figure(per_run)
    return pareto, per_run, summary, best


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--population-size", type=int, default=DEFAULT_POPULATION_SIZE
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=list(ALGORITHM_NAMES),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    results = run_experiment(
        runs=arguments.runs,
        population_size=arguments.population_size,
        generations=arguments.generations,
        base_seed=arguments.base_seed,
        workers=arguments.workers,
        algorithm_names=arguments.algorithms,
    )
    _, _, summary, _ = save_results(
        results,
        runs=arguments.runs,
        population_size=arguments.population_size,
        generations=arguments.generations,
        base_seed=arguments.base_seed,
        workers=arguments.workers,
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
