"""Compare four optimizers for efficiency-power FTHA optimization.

The two benefits are converted to scaled minimization objectives.  Every
thermodynamic evaluation uses the physical parameters and the variable-size
crank-angle grid from the article-based case study.

The four decisions are engine speed, ignition timing, compression ratio, and
connecting-rod-to-crank ratio.  They are optimized in a unit hypercube so that
the variation operators see comparable numerical ranges.

NSGA-II and NSGA-III are provided by DEAP, MOEA/D by pymoo, and MOPSO extends
the PySwarm velocity update with the external nondominated repository proposed
by Coello Coello and Lechuga.  PySwarm itself exposes a scalar objective API,
so the repository and Pareto leader-selection layer are implemented here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import multiprocessing as mp
import os
import pickle
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Each thermodynamic evaluation is single-threaded at the Python level.  Limit
# BLAS inside spawned workers to avoid multiplying eight processes by the BLAS
# default thread count on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _configure_worker_numeric_threads() -> None:
    """Ensure each spawned process uses one native numerical thread."""
    for variable_name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        # This runs immediately before ``spawn``.  Children therefore import
        # NumPy with a single native thread even when the parent environment
        # defines a larger default, avoiding process-by-thread oversubscription.
        os.environ[variable_name] = "1"

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
from .FTHA import (
    DECISION_LOWER_BOUNDS,
    DECISION_UPPER_BOUNDS,
    DECISION_VARIABLE_NAMES,
    simulate_cycle,
)
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
CHECKPOINT_PATH = REPORTS_DIRECTORY / "multiobjective_checkpoint.pkl"
PARETO_FIGURE_PATH = IMAGES_DIRECTORY / "multiobjective_pareto_front.png"
RUNTIME_FIGURE_PATH = IMAGES_DIRECTORY / "multiobjective_runtime_boxplot.png"
DECISION_FIGURE_PATH = (
    IMAGES_DIRECTORY / "multiobjective_constructive_decisions.png"
)

ALGORITHM_NAMES = ("NSGA-II", "NSGA-III", "MOPSO", "MOEA/D")
FRAMEWORK_NAMES = {
    "NSGA-II": "DEAP",
    "NSGA-III": "DEAP",
    "MOPSO": "PySwarm + Pareto repository",
    "MOEA/D": "pymoo",
}

DEFAULT_RUNS = 21
# Forty-eight individuals provide 48 candidate locations along the
# biobjective front and satisfy the NSGA-II DCD multiple-of-four requirement.
# One hundred evolutionary cycles lies inside the 80--120 iteration range
# studied in the original MOPSO experiments and gives four decisions enough
# search depth.  Including the initial population, this is 48 * 101 = 4,848
# thermodynamic evaluations per algorithm and run.
DEFAULT_POPULATION_SIZE = 48
DEFAULT_GENERATIONS = 100
DEFAULT_BASE_SEED = 20_260_718
DEFAULT_WORKERS = min(8, max(1, mp.cpu_count()))

N_DECISION_VARIABLES = len(DECISION_VARIABLE_NAMES)
if DECISION_LOWER_BOUNDS.shape != (N_DECISION_VARIABLES,) or (
    DECISION_UPPER_BOUNDS.shape != (N_DECISION_VARIABLES,)
):
    raise RuntimeError("Decision names and bounds must have matching dimensions.")

# Fixed scales keep both objective magnitudes of order one.  They are rounded
# upper limits based on the preceding sensitivity study (38.280% and
# 25,672.8 kW/kg), not fitted separately for each stochastic run.
EFFICIENCY_SCALE_PERCENT = 40.0
POWER_SCALE_KW_PER_KG = 27_000.0

SBX_PROBABILITY = 0.9
SBX_DISTRIBUTION_INDEX = 20.0
MUTATION_DISTRIBUTION_INDEX = 20.0
# The conventional 1/n_var rule changes one coordinate on average whenever
# polynomial mutation is applied; n_var=4 therefore gives p_m=0.25.
MUTATION_PROBABILITY_PER_VARIABLE = 1.0 / N_DECISION_VARIABLES
MOEAD_NEIGHBORS = 10
MOEAD_NEIGHBOR_MATING_PROBABILITY = 0.9

# Coello Coello and Lechuga's MOPSO experiment used omega=0.4 and unit
# cognitive/social coefficients.  The decaying Gaussian perturbation is a
# local implementation adaptation (not a parameter from the original paper)
# used to preserve exploration in the bounded four-dimensional domain.
MOPSO_INERTIA = 0.4
MOPSO_COGNITIVE = 1.0
MOPSO_SOCIAL = 1.0
MOPSO_PERTURBATION_INITIAL_PROBABILITY = 0.10
MOPSO_PERTURBATION_STANDARD_DEVIATION = 0.10


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
    """Map a unit-hypercube point to the physical decision bounds."""
    normalized = np.asarray(normalized_decisions, dtype=float)
    if normalized.shape != (N_DECISION_VARIABLES,):
        raise ValueError(
            "normalized_decisions must contain exactly "
            f"{N_DECISION_VARIABLES} values ordered as "
            f"{list(DECISION_VARIABLE_NAMES)}."
        )
    return DECISION_LOWER_BOUNDS + normalized * (
        DECISION_UPPER_BOUNDS - DECISION_LOWER_BOUNDS
    )


def _evaluate_normalized(normalized_decisions: Sequence[float]) -> tuple[float, float]:
    """Return scaled minimization objectives for one normalized point."""
    (
        engine_speed_rpm,
        ignition_timing_degrees,
        compression_ratio,
        connecting_rod_to_crank_ratio,
    ) = denormalize_decisions(normalized_decisions)
    parameters = replace(
        CASE_STUDY_PARAMETERS,
        compression_ratio=float(compression_ratio),
        connecting_rod_to_crank_ratio=float(connecting_rod_to_crank_ratio),
    )
    try:
        result = simulate_cycle(
            engine_speed_rpm=float(engine_speed_rpm),
            ignition_timing_degrees=float(ignition_timing_degrees),
            parameters=parameters,
            crank_angle_grid_rad=case_study_crank_angle_grid_rad(
                float(engine_speed_rpm),
                float(ignition_timing_degrees),
                parameters,
            ),
        )
    except RuntimeError as error:
        if "converg" not in str(error).lower():
            raise
        # Both feasible benefits are positive and become negative minimization
        # objectives after scaling.  Zero performance is therefore dominated
        # by every useful point while still consuming its budgeted evaluation.
        return (0.0, 0.0)
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
        N_DECISION_VARIABLES,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate_normalized)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[0.0] * N_DECISION_VARIABLES,
        up=[1.0] * N_DECISION_VARIABLES,
        eta=SBX_DISTRIBUTION_INDEX,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[0.0] * N_DECISION_VARIABLES,
        up=[1.0] * N_DECISION_VARIABLES,
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
    shape = (population_size, N_DECISION_VARIABLES)
    positions = rng.random(shape)
    velocities = rng.uniform(-1.0, 1.0, size=shape)
    objectives = _evaluate_many(positions, pool)
    evaluations = population_size
    personal_best_positions = positions.copy()
    personal_best_objectives = objectives.copy()
    archive_positions = np.empty((0, N_DECISION_VARIABLES), dtype=float)
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
            * rng.random(shape)
            * (personal_best_positions - positions)
            + MOPSO_SOCIAL
            * rng.random(shape)
            * (leaders - positions)
        )
        velocities = np.clip(velocities, -1.0, 1.0)
        positions = positions + velocities
        outside = (positions < 0.0) | (positions > 1.0)
        positions = np.clip(positions, 0.0, 1.0)
        velocities[outside] *= -0.5

        perturbation_probability = MOPSO_PERTURBATION_INITIAL_PROBABILITY * (
            1.0 - generation / max(generations, 1)
        )
        perturbed = rng.random(population_size) < perturbation_probability
        if np.any(perturbed):
            dimensions = rng.integers(
                0, N_DECISION_VARIABLES, size=int(perturbed.sum())
            )
            rows = np.flatnonzero(perturbed)
            positions[rows, dimensions] = np.clip(
                positions[rows, dimensions]
                + rng.normal(
                    0.0,
                    MOPSO_PERTURBATION_STANDARD_DEVIATION,
                    len(rows),
                ),
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
    """pymoo adapter for the normalized FTHA decision hypercube."""

    def __init__(self, runner=None) -> None:
        runner_arguments = (
            {} if runner is None else {"elementwise_runner": runner}
        )
        super().__init__(
            n_var=N_DECISION_VARIABLES,
            n_obj=2,
            xl=np.zeros(N_DECISION_VARIABLES),
            xu=np.ones(N_DECISION_VARIABLES),
            **runner_arguments,
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
            prob=1.0,
            prob_var=MUTATION_PROBABILITY_PER_VARIABLE,
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


def _checkpoint_metadata(
    runs: int,
    population_size: int,
    generations: int,
    base_seed: int,
    algorithm_names: Sequence[str],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "runs": runs,
        "population_size": population_size,
        "generations": generations,
        "base_seed": base_seed,
        "algorithm_names": tuple(algorithm_names),
        "decision_variable_names": DECISION_VARIABLE_NAMES,
    }


def _load_checkpoint(
    metadata: dict[str, object],
) -> list[AlgorithmRunResult]:
    if not CHECKPOINT_PATH.exists():
        return []
    try:
        with CHECKPOINT_PATH.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError):
        return []
    if payload.get("metadata") != metadata:
        return []
    return list(payload.get("results", []))


def _save_checkpoint(
    metadata: dict[str, object],
    results: Sequence[AlgorithmRunResult],
) -> None:
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(
            {"metadata": metadata, "results": list(results)},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary_path.replace(CHECKPOINT_PATH)


def run_experiment(
    runs: int = DEFAULT_RUNS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    workers: int = DEFAULT_WORKERS,
    algorithm_names: Sequence[str] = ALGORITHM_NAMES,
    resume: bool = True,
) -> list[AlgorithmRunResult]:
    """Execute interleaved repetitions and checkpoint every completed run."""
    if runs < 1 or generations < 1:
        raise ValueError("runs and generations must be positive.")
    if population_size < 4 or population_size % 4:
        raise ValueError("population_size must be a multiple of four and >= 4.")
    unknown = set(algorithm_names) - set(ALGORITHM_NAMES)
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")

    names = list(algorithm_names)
    metadata = _checkpoint_metadata(
        runs,
        population_size,
        generations,
        base_seed,
        names,
    )
    results = _load_checkpoint(metadata) if resume else []
    completed = {(result.algorithm, result.run) for result in results}
    if results:
        print(f"Resuming {len(results)} completed algorithm runs.", flush=True)

    pool = None
    try:
        if workers > 1:
            _configure_worker_numeric_threads()
            pool = mp.get_context("spawn").Pool(processes=workers)
        for run_index in range(1, runs + 1):
            offset = (run_index - 1) % len(names)
            ordered_names = names[offset:] + names[:offset]
            for algorithm_name in ordered_names:
                if (algorithm_name, run_index) in completed:
                    continue
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
                _save_checkpoint(metadata, results)
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
                    **dict(
                        zip(DECISION_VARIABLE_NAMES, decision, strict=True)
                    ),
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
                **{
                    f"compromise_{name}": value
                    for name, value in zip(
                        DECISION_VARIABLE_NAMES,
                        representative_decision,
                        strict=True,
                    )
                },
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
        summary_record: dict[str, float | int | str] = {
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
        for decision_name in DECISION_VARIABLE_NAMES:
            compromise_column = f"compromise_{decision_name}"
            summary_record[f"{decision_name}_mean"] = algorithm_runs[
                compromise_column
            ].mean()
            summary_record[f"{decision_name}_std"] = algorithm_runs[
                compromise_column
            ].std(ddof=1)
            summary_record[f"best_{decision_name}"] = best[decision_name]
        summary_records.append(summary_record)

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
    is_baseline_budget = (
        population_size == DEFAULT_POPULATION_SIZE
        and generations == DEFAULT_GENERATIONS
    )
    budget_rationale = (
        "48 front candidates and 100 evolutionary cycles; the population is "
        "divisible by four for NSGA-II and 100 lies in the 80-120 iteration "
        "range studied for the original MOPSO"
        if is_baseline_budget
        else "command-line budget override; see population and generation fields"
    )
    runs_rationale = (
        "21 exceeds the 20 repetitions in the original NSGA-III study, gives "
        "an observed median run, and preserves a feasible budget"
        if runs == DEFAULT_RUNS
        else "command-line repetition override; see runs_per_algorithm"
    )
    record: dict[str, object] = {
        "runs_per_algorithm": runs,
        "number_of_decision_variables": N_DECISION_VARIABLES,
        "population_or_swarm_size": population_size,
        "generations": generations,
        "nominal_evaluations_per_run": population_size * (generations + 1),
        "is_baseline_budget": is_baseline_budget,
        "budget_rationale": budget_rationale,
        "runs_rationale": runs_rationale,
        "base_seed": base_seed,
        "parallel_workers": workers,
        "efficiency_scale_percent": EFFICIENCY_SCALE_PERCENT,
        "power_scale_kw_per_kg": POWER_SCALE_KW_PER_KG,
        "nonconvergent_point_policy": (
            "count evaluation and assign dominated scaled objectives (0, 0)"
        ),
        "sbx_probability": SBX_PROBABILITY,
        "sbx_distribution_index": SBX_DISTRIBUTION_INDEX,
        "mutation_distribution_index": MUTATION_DISTRIBUTION_INDEX,
        "genetic_operator_rationale": (
            "pc=0.9 and eta_c=eta_m=20 are established real-coded NSGA-II "
            "baselines; common operators isolate selection differences"
        ),
        "mutation_probability_per_variable": MUTATION_PROBABILITY_PER_VARIABLE,
        "mutation_probability_rationale": (
            "1/n_var, changing one coordinate on average per mutation call"
        ),
        "mopso_inertia": MOPSO_INERTIA,
        "mopso_cognitive": MOPSO_COGNITIVE,
        "mopso_social": MOPSO_SOCIAL,
        "mopso_parameter_rationale": (
            "omega=0.4 and c1=c2=1.0 follow Coello Coello and Lechuga (2004)"
        ),
        "mopso_repository_size": 4 * population_size,
        "mopso_repository_rationale": (
            "four archive slots per particle balance Pareto diversity and memory"
        ),
        "mopso_perturbation_initial_probability": (
            MOPSO_PERTURBATION_INITIAL_PROBABILITY
        ),
        "mopso_perturbation_standard_deviation": (
            MOPSO_PERTURBATION_STANDARD_DEVIATION
        ),
        "mopso_perturbation_rationale": (
            "decaying local Gaussian adaptation for bounded-domain exploration"
        ),
        "moead_neighbors": min(MOEAD_NEIGHBORS, population_size),
        "moead_neighbor_mating_probability": MOEAD_NEIGHBOR_MATING_PROBABILITY,
        "moead_rationale": (
            "10 neighbors preserve approximately the original 20 percent "
            "neighborhood ratio; delta=0.9 favors local cooperation while "
            "retaining 10 percent global mating"
        ),
        "compression_ratio_bounds_rationale": (
            "8-12 spans conventional spark-ignition values and the published "
            "FTHA reference case; it is a study domain, not a universal limit"
        ),
        "connecting_rod_ratio_bounds_rationale": (
            "3.2 approximates the Toyota 1ZZ-FE production geometry and 4.4 "
            "is the upper end of a published parametric study"
        ),
        "deap_version": importlib.metadata.version("deap"),
        "pyswarm_version": importlib.metadata.version("pyswarm"),
        "pymoo_version": importlib.metadata.version("pymoo"),
    }
    for name, lower, upper in zip(
        DECISION_VARIABLE_NAMES,
        DECISION_LOWER_BOUNDS,
        DECISION_UPPER_BOUNDS,
        strict=True,
    ):
        record[f"{name}_lower_bound"] = lower
        record[f"{name}_upper_bound"] = upper
    return pd.DataFrame.from_records([record])


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


def _save_constructive_decisions_figure(
    pareto_table: pd.DataFrame,
    run_table: pd.DataFrame,
) -> None:
    """Show how the two constructive decisions vary along the best fronts."""
    panel_definitions = (
        (
            "compression_ratio",
            "thermal_efficiency_percent",
            r"Taxa de compressão, $r$",
            r"Eficiência térmica, $\eta_t$ [%]",
        ),
        (
            "compression_ratio",
            "net_specific_power_kw_per_kg",
            r"Taxa de compressão, $r$",
            r"Potência líquida específica [kW/kg]",
        ),
        (
            "connecting_rod_to_crank_ratio",
            "thermal_efficiency_percent",
            r"Razão biela--manivela, $L/R$",
            r"Eficiência térmica, $\eta_t$ [%]",
        ),
        (
            "connecting_rod_to_crank_ratio",
            "net_specific_power_kw_per_kg",
            r"Razão biela--manivela, $L/R$",
            r"Potência líquida específica [kW/kg]",
        ),
    )
    with plt.rc_context(CHART_STYLE):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(8.0, 6.6),
            constrained_layout=True,
        )
        for algorithm_index, algorithm_name in enumerate(ALGORITHM_NAMES):
            algorithm_runs = run_table[run_table["algorithm"] == algorithm_name]
            if algorithm_runs.empty:
                continue
            best_run = int(
                algorithm_runs.loc[
                    algorithm_runs["hypervolume"].idxmax(), "run"
                ]
            )
            front = pareto_table[
                (pareto_table["algorithm"] == algorithm_name)
                & (pareto_table["run"] == best_run)
            ]
            style = BLACK_AND_WHITE_SERIES_STYLES[algorithm_index]
            for axis, (x_column, y_column, x_label, y_label) in zip(
                axes.flat,
                panel_definitions,
                strict=True,
            ):
                axis.plot(
                    front[x_column],
                    front[y_column],
                    linestyle="none",
                    marker=style["marker"],
                    color="black",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=4.0,
                    label=f"{algorithm_name} (execução {best_run})",
                )
                axis.set_xlabel(x_label)
                axis.set_ylabel(y_label)
                axis.grid(True, color="0.82", linewidth=0.6, linestyle=":")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=2,
            frameon=True,
            edgecolor="black",
            fontsize=8,
        )
        figure.savefig(DECISION_FIGURE_PATH, dpi=300)
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
    _save_constructive_decisions_figure(pareto, per_run)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing compatible benchmark checkpoint.",
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
        resume=not arguments.no_resume,
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
