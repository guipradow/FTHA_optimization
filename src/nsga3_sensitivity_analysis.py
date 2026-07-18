"""Sensitivity analysis of NSGA-III hyperparameters for the FTHA problem.

The experiment uses a balanced discrete Latin hypercube for screening and a
held-out confirmation stage.  Every run has exactly 504 direct thermodynamic
model evaluations, irrespective of population size.  Common random seeds make
comparisons paired, while runs 8--21 are never used to select configurations.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from scipy import stats
from scipy.stats import qmc

from .multiobjective_optimization import (
    BLACK_AND_WHITE_SERIES_STYLES,
    CHART_STYLE,
    EFFICIENCY_SCALE_PERCENT,
    POWER_SCALE_KW_PER_KG,
    _clean_front,
    _compromise_scores,
    _evaluate_normalized,
    denormalize_decisions,
    nondominated_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
IMAGES_DIRECTORY = PROJECT_ROOT / "img"

DESIGN_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_design.csv"
PARETO_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_pareto_solutions.csv"
REFERENCE_FRONT_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_reference_front.csv"
RUNS_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_runs.csv"
SUMMARY_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_summary.csv"
EFFECTS_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_main_effects.csv"
IMPORTANCE_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_factor_importance.csv"
INTERACTIONS_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_interactions.csv"
PAIRWISE_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_pairwise_tests.csv"
CONVERGENCE_PATH = REPORTS_DIRECTORY / "nsga3_sensitivity_convergence.csv"
BEST_CONFIGURATION_PATH = (
    REPORTS_DIRECTORY / "nsga3_sensitivity_best_configuration.csv"
)
CHECKPOINT_PATH = REPORTS_DIRECTORY / ".nsga3_sensitivity_checkpoint.pkl"

MAIN_EFFECTS_FIGURE_PATH = IMAGES_DIRECTORY / "nsga3_sensitivity_main_effects.png"
CONFIRMATION_FIGURE_PATH = (
    IMAGES_DIRECTORY / "nsga3_sensitivity_confirmation_boxplot.png"
)
CONVERGENCE_FIGURE_PATH = IMAGES_DIRECTORY / "nsga3_sensitivity_convergence.png"

SCREENING_CONFIGURATION_COUNT = 36
SCREENING_RUNS = 7
TOTAL_CONFIRMATION_RUNS = 21
CONFIRMATION_CONFIGURATION_COUNT = 5
EVALUATIONS_PER_RUN = 504
DESIGN_SEED = 20_260_719
RUN_SEED_BASE = 31_000_000
DEFAULT_WORKERS = min(8, max(1, mp.cpu_count()))
BOOTSTRAP_RESAMPLES = 10_000

POPULATION_GENERATION_LEVELS = (
    (12, 41),
    (24, 20),
    (36, 13),
    (56, 8),
)
CROSSOVER_PROBABILITY_LEVELS = (0.7, 0.9, 1.0)
CROSSOVER_ETA_LEVELS = (10.0, 20.0, 30.0)
MUTATION_ETA_LEVELS = (10.0, 20.0, 40.0)
MUTATION_PROBABILITY_LEVELS = (0.25, 0.5, 1.0)

FACTOR_COLUMNS = (
    "population_size",
    "crossover_probability",
    "crossover_eta",
    "mutation_eta",
    "mutation_probability_per_variable",
)
FACTOR_LABELS = {
    "population_size": "População / gerações",
    "crossover_probability": r"$p_c$",
    "crossover_eta": r"$\eta_c$",
    "mutation_eta": r"$\eta_m$",
    "mutation_probability_per_variable": r"$p_m$ por variável",
}


@dataclass(frozen=True, slots=True)
class NSGA3Configuration:
    """One fixed-budget NSGA-III hyperparameter configuration."""

    configuration_id: str
    population_size: int
    generations: int
    crossover_probability: float
    crossover_eta: float
    mutation_eta: float
    mutation_probability_per_variable: float

    @property
    def reference_partitions(self) -> int:
        return self.population_size - 1

    @property
    def evaluations(self) -> int:
        return self.population_size * (self.generations + 1)

    def parameter_tuple(self) -> tuple[int, float, float, float, float]:
        return (
            self.population_size,
            self.crossover_probability,
            self.crossover_eta,
            self.mutation_eta,
            self.mutation_probability_per_variable,
        )


@dataclass(slots=True)
class SensitivityRunResult:
    """Final front and convergence trace for one configuration and seed."""

    configuration_id: str
    run: int
    seed: int
    runtime_seconds: float
    evaluations: int
    normalized_decisions: np.ndarray
    scaled_objectives: np.ndarray
    convergence: np.ndarray


BASELINE_PARAMETER_TUPLE = (24, 0.9, 20.0, 20.0, 0.5)


def generate_screening_design(seed: int = DESIGN_SEED) -> list[NSGA3Configuration]:
    """Generate a unique, level-balanced discrete Latin hypercube."""
    level_counts = np.array([4, 3, 3, 3, 3], dtype=int)
    for attempt in range(10_000):
        sampler = qmc.LatinHypercube(d=5, scramble=True, seed=seed + attempt)
        unit_design = sampler.random(n=SCREENING_CONFIGURATION_COUNT)
        level_indices = np.minimum(
            (unit_design * level_counts).astype(int), level_counts - 1
        )
        tuples: list[tuple[int, float, float, float, float]] = []
        for row in level_indices:
            population_size, _ = POPULATION_GENERATION_LEVELS[row[0]]
            tuples.append(
                (
                    population_size,
                    CROSSOVER_PROBABILITY_LEVELS[row[1]],
                    CROSSOVER_ETA_LEVELS[row[2]],
                    MUTATION_ETA_LEVELS[row[3]],
                    MUTATION_PROBABILITY_LEVELS[row[4]],
                )
            )
        if len(set(tuples)) != SCREENING_CONFIGURATION_COUNT:
            continue
        if BASELINE_PARAMETER_TUPLE not in tuples:
            continue
        configurations: list[NSGA3Configuration] = []
        generation_by_population = dict(POPULATION_GENERATION_LEVELS)
        for index, parameters in enumerate(tuples, start=1):
            (
                population_size,
                crossover_probability,
                crossover_eta,
                mutation_eta,
                mutation_probability,
            ) = parameters
            configurations.append(
                NSGA3Configuration(
                    configuration_id=f"C{index:02d}",
                    population_size=population_size,
                    generations=generation_by_population[population_size],
                    crossover_probability=crossover_probability,
                    crossover_eta=crossover_eta,
                    mutation_eta=mutation_eta,
                    mutation_probability_per_variable=mutation_probability,
                )
            )
        return configurations
    raise RuntimeError("Could not generate a unique balanced screening design.")


def baseline_configuration_id(
    configurations: Sequence[NSGA3Configuration],
) -> str:
    for configuration in configurations:
        if configuration.parameter_tuple() == BASELINE_PARAMETER_TUPLE:
            return configuration.configuration_id
    raise ValueError("The screening design does not contain the baseline.")


def _configuration_record(configuration: NSGA3Configuration) -> dict[str, object]:
    record = asdict(configuration)
    record["reference_partitions"] = configuration.reference_partitions
    record["nominal_evaluations"] = configuration.evaluations
    return record


def _make_toolbox(configuration: NSGA3Configuration) -> base.Toolbox:
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
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=[0.0, 0.0],
        up=[1.0, 1.0],
        eta=configuration.crossover_eta,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=[0.0, 0.0],
        up=[1.0, 1.0],
        eta=configuration.mutation_eta,
        indpb=configuration.mutation_probability_per_variable,
    )
    return toolbox


def _assign_fitness(population: Iterable, pool: mp.pool.Pool | None) -> int:
    invalid = [individual for individual in population if not individual.fitness.valid]
    if not invalid:
        return 0
    points = np.asarray(invalid, dtype=float)
    values = (
        list(map(_safe_evaluate_normalized, points))
        if pool is None
        else pool.map(_safe_evaluate_normalized, list(points))
    )
    for individual, fitness in zip(invalid, values, strict=True):
        individual.fitness.values = tuple(fitness)
    return len(invalid)


def _safe_evaluate_normalized(
    normalized_decisions: Sequence[float],
) -> tuple[float, float]:
    """Penalize a rare nonconvergent thermodynamic point as dominated."""
    try:
        return _evaluate_normalized(normalized_decisions)
    except RuntimeError as error:
        if "converg" not in str(error).lower():
            raise
        # Feasible FTHA benefits have negative minimization objectives.  The
        # physical zero-performance point is therefore dominated by every
        # useful solution and cannot enter a final Pareto front.
        return (0.0, 0.0)


def _population_objectives(population: Sequence) -> np.ndarray:
    return np.asarray(
        [individual.fitness.values for individual in population], dtype=float
    )


def run_nsga3_configuration(
    configuration: NSGA3Configuration,
    run: int,
    seed: int,
    pool: mp.pool.Pool | None,
) -> SensitivityRunResult:
    """Run one parameterized NSGA-III replication and retain convergence."""
    if configuration.evaluations != EVALUATIONS_PER_RUN:
        raise ValueError("Every configuration must have exactly 504 evaluations.")
    random.seed(seed)
    np.random.seed(seed)
    toolbox = _make_toolbox(configuration)
    reference_points = tools.uniform_reference_points(
        nobj=2, p=configuration.reference_partitions
    )
    hypervolume = HV(ref_point=np.zeros(2))

    start = time.perf_counter()
    population = toolbox.population(n=configuration.population_size)
    evaluations = _assign_fitness(population, pool)
    population = tools.selNSGA3(
        population,
        configuration.population_size,
        ref_points=reference_points,
    )
    history = [
        (0, evaluations, float(hypervolume(_population_objectives(population))))
    ]

    for generation in range(1, configuration.generations + 1):
        offspring = algorithms.varAnd(
            population,
            toolbox,
            cxpb=configuration.crossover_probability,
            mutpb=1.0,
        )
        evaluations += _assign_fitness(offspring, pool)
        population = tools.selNSGA3(
            population + offspring,
            configuration.population_size,
            ref_points=reference_points,
        )
        history.append(
            (
                generation,
                evaluations,
                float(hypervolume(_population_objectives(population))),
            )
        )

    runtime = time.perf_counter() - start
    decisions, objectives = _clean_front(
        np.asarray(population, dtype=float),
        _population_objectives(population),
    )
    return SensitivityRunResult(
        configuration_id=configuration.configuration_id,
        run=run,
        seed=seed,
        runtime_seconds=runtime,
        evaluations=evaluations,
        normalized_decisions=decisions,
        scaled_objectives=objectives,
        convergence=np.asarray(history, dtype=float),
    )


def _checkpoint_metadata(
    configurations: Sequence[NSGA3Configuration],
) -> list[dict[str, object]]:
    return [_configuration_record(configuration) for configuration in configurations]


def _save_checkpoint(
    configurations: Sequence[NSGA3Configuration],
    results: Sequence[SensitivityRunResult],
) -> None:
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(
            {
                "metadata": _checkpoint_metadata(configurations),
                # Store plain records so a checkpoint written with
                # ``python -m`` is not tied to the transient ``__main__``
                # module name when it is resumed later.
                "results": [asdict(result) for result in results],
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary_path.replace(CHECKPOINT_PATH)


def _load_checkpoint(
    configurations: Sequence[NSGA3Configuration],
) -> list[SensitivityRunResult]:
    if not CHECKPOINT_PATH.exists():
        return []
    with CHECKPOINT_PATH.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("metadata") != _checkpoint_metadata(configurations):
        raise RuntimeError("Checkpoint metadata does not match this experiment.")
    return [
        item
        if isinstance(item, SensitivityRunResult)
        else SensitivityRunResult(**item)
        for item in payload.get("results", [])
    ]


def _screening_hypervolume_table(
    configurations: Sequence[NSGA3Configuration],
    results: Sequence[SensitivityRunResult],
) -> pd.DataFrame:
    configuration_map = {
        configuration.configuration_id: configuration
        for configuration in configurations
    }
    hypervolume = HV(ref_point=np.zeros(2))
    records = []
    for result in results:
        if result.run > SCREENING_RUNS:
            continue
        configuration = configuration_map[result.configuration_id]
        record = _configuration_record(configuration)
        record.update(
            {
                "run": result.run,
                "hypervolume": float(hypervolume(result.scaled_objectives)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def select_confirmation_configuration_ids(
    configurations: Sequence[NSGA3Configuration],
    screening_results: Sequence[SensitivityRunResult],
) -> list[str]:
    """Select baseline, top three, and worst screening configuration."""
    table = _screening_hypervolume_table(configurations, screening_results)
    means = (
        table.groupby("configuration_id", as_index=False)["hypervolume"]
        .mean()
        .sort_values("hypervolume", ascending=False, kind="stable")
    )
    baseline_id = baseline_configuration_id(configurations)
    ordered: list[str] = [baseline_id]
    for configuration_id in means.head(3)["configuration_id"]:
        if configuration_id not in ordered:
            ordered.append(configuration_id)
    worst_id = str(means.iloc[-1]["configuration_id"])
    if worst_id not in ordered:
        ordered.append(worst_id)
    for configuration_id in means["configuration_id"]:
        if len(ordered) >= CONFIRMATION_CONFIGURATION_COUNT:
            break
        if configuration_id not in ordered:
            ordered.append(str(configuration_id))
    return ordered[:CONFIRMATION_CONFIGURATION_COUNT]


def _run_missing_stage(
    configurations: Sequence[NSGA3Configuration],
    checkpoint_configurations: Sequence[NSGA3Configuration],
    run_numbers: Iterable[int],
    pool: mp.pool.Pool | None,
    results: list[SensitivityRunResult],
) -> None:
    configuration_map = {
        configuration.configuration_id: configuration
        for configuration in configurations
    }
    completed = {(result.configuration_id, result.run) for result in results}
    configuration_ids = list(configuration_map)
    run_numbers = list(run_numbers)
    for run in run_numbers:
        offset = (run - 1) % len(configuration_ids)
        ordered_ids = configuration_ids[offset:] + configuration_ids[:offset]
        for configuration_id in ordered_ids:
            if (configuration_id, run) in completed:
                continue
            result = run_nsga3_configuration(
                configuration_map[configuration_id],
                run=run,
                seed=RUN_SEED_BASE + run,
                pool=pool,
            )
            results.append(result)
            completed.add((configuration_id, run))
            _save_checkpoint(checkpoint_configurations, results)
            print(
                f"{configuration_id} run {run:02d}: "
                f"{len(result.scaled_objectives)} solutions, "
                f"{result.runtime_seconds:.2f} s",
                flush=True,
            )


def execute_sensitivity_analysis(
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> tuple[
    list[NSGA3Configuration],
    list[str],
    list[SensitivityRunResult],
]:
    """Run screening and held-out confirmation with checkpoint support."""
    configurations = generate_screening_design()
    results = _load_checkpoint(configurations) if resume else []
    pool = None
    try:
        if workers > 1:
            pool = mp.get_context("spawn").Pool(processes=workers)
        _run_missing_stage(
            configurations,
            configurations,
            range(1, SCREENING_RUNS + 1),
            pool,
            results,
        )
        confirmation_ids = select_confirmation_configuration_ids(
            configurations, results
        )
        confirmation_configurations = [
            configuration
            for configuration in configurations
            if configuration.configuration_id in confirmation_ids
        ]
        _run_missing_stage(
            confirmation_configurations,
            configurations,
            range(SCREENING_RUNS + 1, TOTAL_CONFIRMATION_RUNS + 1),
            pool,
            results,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    results.sort(key=lambda item: (item.configuration_id, item.run))
    return configurations, confirmation_ids, results


def _benefits_from_objectives(objectives: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            -objectives[:, 0] * EFFICIENCY_SCALE_PERCENT,
            -objectives[:, 1] * POWER_SCALE_KW_PER_KG,
        )
    )


def _spacing(objectives: np.ndarray) -> float:
    if len(objectives) < 3:
        return 0.0
    distances = np.linalg.norm(
        objectives[:, None, :] - objectives[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(nearest.std(ddof=1))


def _bootstrap_mean_interval(
    values: Sequence[float],
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def build_numerical_tables(
    configurations: Sequence[NSGA3Configuration],
    confirmation_ids: Sequence[str],
    results: Sequence[SensitivityRunResult],
) -> dict[str, pd.DataFrame]:
    """Build fronts, metrics, summaries, effects, and paired tests."""
    configuration_map = {
        configuration.configuration_id: configuration
        for configuration in configurations
    }
    all_objectives = np.vstack([result.scaled_objectives for result in results])
    reference_indices = nondominated_indices(all_objectives)
    reference_objectives = all_objectives[reference_indices]
    reference_benefits = _benefits_from_objectives(reference_objectives)
    ideal = reference_benefits.max(axis=0)
    nadir = reference_benefits.min(axis=0)
    hypervolume = HV(ref_point=np.zeros(2))
    igd_plus = IGDPlus(reference_objectives)

    pareto_records: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    convergence_records: list[dict[str, object]] = []
    for result in results:
        configuration = configuration_map[result.configuration_id]
        decisions = np.asarray(
            [denormalize_decisions(point) for point in result.normalized_decisions]
        )
        benefits = _benefits_from_objectives(result.scaled_objectives)
        scores = _compromise_scores(benefits, ideal, nadir)
        compromise_index = int(np.argmin(scores))
        for solution, (decision, benefit, score) in enumerate(
            zip(decisions, benefits, scores, strict=True), start=1
        ):
            pareto_records.append(
                {
                    "configuration_id": result.configuration_id,
                    "run": result.run,
                    "seed": result.seed,
                    "solution": solution,
                    "engine_speed_rpm": decision[0],
                    "ignition_timing_degrees": decision[1],
                    "thermal_efficiency_percent": benefit[0],
                    "net_specific_power_kw_per_kg": benefit[1],
                    "compromise_score": score,
                    "is_run_compromise": solution - 1 == compromise_index,
                }
            )
        record = _configuration_record(configuration)
        record.update(
            {
                "stage": (
                    "screening" if result.run <= SCREENING_RUNS else "confirmation"
                ),
                "is_baseline": (
                    configuration.parameter_tuple() == BASELINE_PARAMETER_TUPLE
                ),
                "is_confirmed": result.configuration_id in confirmation_ids,
                "run": result.run,
                "seed": result.seed,
                "evaluations": result.evaluations,
                "front_size": len(result.scaled_objectives),
                "hypervolume": float(hypervolume(result.scaled_objectives)),
                "igd_plus": float(igd_plus(result.scaled_objectives)),
                "spacing": _spacing(result.scaled_objectives),
                "runtime_seconds": result.runtime_seconds,
                "maximum_efficiency_percent": benefits[:, 0].max(),
                "maximum_power_kw_per_kg": benefits[:, 1].max(),
                "compromise_engine_speed_rpm": decisions[compromise_index, 0],
                "compromise_ignition_timing_degrees": decisions[
                    compromise_index, 1
                ],
                "compromise_efficiency_percent": benefits[compromise_index, 0],
                "compromise_power_kw_per_kg": benefits[compromise_index, 1],
                "compromise_score": scores[compromise_index],
            }
        )
        run_records.append(record)
        for generation, evaluations, hv_value in result.convergence:
            convergence_records.append(
                {
                    "configuration_id": result.configuration_id,
                    "run": result.run,
                    "seed": result.seed,
                    "generation": int(generation),
                    "evaluations": int(evaluations),
                    "hypervolume": hv_value,
                }
            )

    pareto_table = pd.DataFrame.from_records(pareto_records)
    run_table = pd.DataFrame.from_records(run_records)
    convergence_table = pd.DataFrame.from_records(convergence_records)
    reference_table = pd.DataFrame(
        {
            "thermal_efficiency_percent": reference_benefits[:, 0],
            "net_specific_power_kw_per_kg": reference_benefits[:, 1],
        }
    ).sort_values("thermal_efficiency_percent")

    screening_means = (
        run_table[run_table["stage"] == "screening"]
        .groupby("configuration_id")["hypervolume"]
        .mean()
        .sort_values(ascending=False)
    )
    screening_rank = {
        configuration_id: rank
        for rank, configuration_id in enumerate(screening_means.index, start=1)
    }
    summary_records: list[dict[str, object]] = []
    for configuration_index, configuration in enumerate(configurations):
        rows = run_table[
            run_table["configuration_id"] == configuration.configuration_id
        ]
        screening = rows[rows["stage"] == "screening"]
        confirmation = rows[rows["stage"] == "confirmation"]
        interval_low, interval_high = _bootstrap_mean_interval(
            rows["hypervolume"], DESIGN_SEED + configuration_index
        )
        summary_records.append(
            {
                **_configuration_record(configuration),
                "is_baseline": (
                    configuration.parameter_tuple() == BASELINE_PARAMETER_TUPLE
                ),
                "is_confirmed": configuration.configuration_id in confirmation_ids,
                "screening_rank": screening_rank[configuration.configuration_id],
                "runs": len(rows),
                "screening_hypervolume_mean": screening["hypervolume"].mean(),
                "screening_hypervolume_std": screening["hypervolume"].std(ddof=1),
                "confirmation_hypervolume_mean": confirmation[
                    "hypervolume"
                ].mean(),
                "confirmation_hypervolume_std": confirmation[
                    "hypervolume"
                ].std(ddof=1),
                "hypervolume_mean": rows["hypervolume"].mean(),
                "hypervolume_std": rows["hypervolume"].std(ddof=1),
                "hypervolume_ci95_low": interval_low,
                "hypervolume_ci95_high": interval_high,
                "igd_plus_mean": rows["igd_plus"].mean(),
                "igd_plus_std": rows["igd_plus"].std(ddof=1),
                "spacing_mean": rows["spacing"].mean(),
                "runtime_seconds_mean": rows["runtime_seconds"].mean(),
                "runtime_seconds_std": rows["runtime_seconds"].std(ddof=1),
                "front_size_mean": rows["front_size"].mean(),
                "compromise_engine_speed_rpm_mean": rows[
                    "compromise_engine_speed_rpm"
                ].mean(),
                "compromise_engine_speed_rpm_std": rows[
                    "compromise_engine_speed_rpm"
                ].std(ddof=1),
                "compromise_ignition_timing_degrees_mean": rows[
                    "compromise_ignition_timing_degrees"
                ].mean(),
                "compromise_ignition_timing_degrees_std": rows[
                    "compromise_ignition_timing_degrees"
                ].std(ddof=1),
                "compromise_efficiency_percent_mean": rows[
                    "compromise_efficiency_percent"
                ].mean(),
                "compromise_efficiency_percent_std": rows[
                    "compromise_efficiency_percent"
                ].std(ddof=1),
                "compromise_power_kw_per_kg_mean": rows[
                    "compromise_power_kw_per_kg"
                ].mean(),
                "compromise_power_kw_per_kg_std": rows[
                    "compromise_power_kw_per_kg"
                ].std(ddof=1),
            }
        )
    summary_table = pd.DataFrame.from_records(summary_records)

    effects_table = _main_effects_table(run_table)
    importance_table = _factor_importance_table(run_table)
    interactions_table = _interaction_table(run_table)
    pairwise_table = _pairwise_confirmation_tests(run_table, configurations)

    confirmed_summary = summary_table[summary_table["is_confirmed"]].copy()
    best_id = (
        confirmed_summary.sort_values(
            ["confirmation_hypervolume_mean", "confirmation_hypervolume_std"],
            ascending=[False, True],
            kind="stable",
        )
        .iloc[0]["configuration_id"]
    )
    best_configuration_table = confirmed_summary[
        confirmed_summary["configuration_id"] == best_id
    ].copy()
    best_configuration_table.insert(
        0, "selection_basis", "highest held-out mean hypervolume (runs 8-21)"
    )

    design_table = pd.DataFrame.from_records(
        [
            {
                **_configuration_record(configuration),
                "is_baseline": (
                    configuration.parameter_tuple() == BASELINE_PARAMETER_TUPLE
                ),
                "selected_for_confirmation": (
                    configuration.configuration_id in confirmation_ids
                ),
            }
            for configuration in configurations
        ]
    )
    return {
        "design": design_table,
        "pareto": pareto_table,
        "reference": reference_table,
        "runs": run_table,
        "summary": summary_table,
        "effects": effects_table,
        "importance": importance_table,
        "interactions": interactions_table,
        "pairwise": pairwise_table,
        "convergence": convergence_table,
        "best": best_configuration_table,
    }


def _main_effects_table(run_table: pd.DataFrame) -> pd.DataFrame:
    screening = run_table[run_table["stage"] == "screening"]
    grand_mean = screening["hypervolume"].mean()
    records: list[dict[str, object]] = []
    for factor in FACTOR_COLUMNS:
        for level, rows in screening.groupby(factor, sort=True):
            records.append(
                {
                    "factor": factor,
                    "level": level,
                    "configurations": rows["configuration_id"].nunique(),
                    "runs": len(rows),
                    "hypervolume_mean": rows["hypervolume"].mean(),
                    "hypervolume_std": rows["hypervolume"].std(ddof=1),
                    "hypervolume_standard_error": rows["hypervolume"].std(ddof=1)
                    / np.sqrt(len(rows)),
                    "delta_from_grand_mean": rows["hypervolume"].mean()
                    - grand_mean,
                }
            )
    return pd.DataFrame.from_records(records)


def _dummy_columns(table: pd.DataFrame, column: str) -> np.ndarray:
    categorical = pd.Categorical(table[column])
    codes = categorical.codes
    number_of_levels = len(categorical.categories)
    if number_of_levels <= 1:
        return np.empty((len(table), 0), dtype=float)
    return np.column_stack(
        [(codes == level).astype(float) for level in range(1, number_of_levels)]
    )


def _design_matrix(
    table: pd.DataFrame,
    factors: Sequence[str],
    include_seed: bool = True,
) -> np.ndarray:
    columns = [np.ones((len(table), 1), dtype=float)]
    for factor in factors:
        columns.append(_dummy_columns(table, factor))
    if include_seed:
        columns.append(_dummy_columns(table, "run"))
    return np.column_stack(columns)


def _residual_sum_squares(
    matrix: np.ndarray,
    response: np.ndarray,
) -> tuple[float, int]:
    fitted = matrix @ np.linalg.lstsq(matrix, response, rcond=None)[0]
    residual = response - fitted
    return float(residual @ residual), int(np.linalg.matrix_rank(matrix))


def _factor_importance_table(run_table: pd.DataFrame) -> pd.DataFrame:
    screening = run_table[run_table["stage"] == "screening"].copy()
    response = screening["hypervolume"].to_numpy(dtype=float)
    full_matrix = _design_matrix(screening, FACTOR_COLUMNS)
    full_sse, full_rank = _residual_sum_squares(full_matrix, response)
    residual_df = len(screening) - full_rank
    records = []
    for factor in FACTOR_COLUMNS:
        reduced_factors = [item for item in FACTOR_COLUMNS if item != factor]
        reduced_sse, reduced_rank = _residual_sum_squares(
            _design_matrix(screening, reduced_factors), response
        )
        effect_ss = max(0.0, reduced_sse - full_sse)
        effect_df = full_rank - reduced_rank
        f_statistic = (
            (effect_ss / effect_df) / (full_sse / residual_df)
            if effect_df > 0 and full_sse > 0.0
            else np.nan
        )
        records.append(
            {
                "factor": factor,
                "degrees_of_freedom": effect_df,
                "sum_of_squares": effect_ss,
                "f_statistic": f_statistic,
                "p_value": stats.f.sf(f_statistic, effect_df, residual_df),
                "partial_eta_squared": effect_ss / (effect_ss + full_sse),
            }
        )
    table = pd.DataFrame.from_records(records)
    table["main_effect_share_percent"] = (
        100.0 * table["sum_of_squares"] / table["sum_of_squares"].sum()
    )
    return table.sort_values("partial_eta_squared", ascending=False)


def _interaction_table(run_table: pd.DataFrame) -> pd.DataFrame:
    screening = run_table[run_table["stage"] == "screening"].copy()
    response = screening["hypervolume"].to_numpy(dtype=float)
    base_matrix = _design_matrix(screening, FACTOR_COLUMNS)
    base_sse, base_rank = _residual_sum_squares(base_matrix, response)
    records = []
    for first_index, first in enumerate(FACTOR_COLUMNS):
        for second in FACTOR_COLUMNS[first_index + 1 :]:
            first_dummies = _dummy_columns(screening, first)
            second_dummies = _dummy_columns(screening, second)
            interaction_columns = np.column_stack(
                [
                    first_dummies[:, i] * second_dummies[:, j]
                    for i in range(first_dummies.shape[1])
                    for j in range(second_dummies.shape[1])
                ]
            )
            full_matrix = np.column_stack((base_matrix, interaction_columns))
            full_sse, full_rank = _residual_sum_squares(full_matrix, response)
            effect_df = full_rank - base_rank
            residual_df = len(screening) - full_rank
            effect_ss = max(0.0, base_sse - full_sse)
            f_statistic = (
                (effect_ss / effect_df) / (full_sse / residual_df)
                if effect_df > 0 and full_sse > 0.0
                else np.nan
            )
            records.append(
                {
                    "first_factor": first,
                    "second_factor": second,
                    "degrees_of_freedom": effect_df,
                    "sum_of_squares": effect_ss,
                    "f_statistic": f_statistic,
                    "p_value": stats.f.sf(f_statistic, effect_df, residual_df),
                    "partial_eta_squared": effect_ss / (effect_ss + full_sse),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        "partial_eta_squared", ascending=False
    )


def _rank_biserial_difference(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0.0].sum()
    negative = ranks[nonzero < 0.0].sum()
    return float((positive - negative) / (positive + negative))


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_maximum = 0.0
    number = len(p_values)
    for rank, index in enumerate(order):
        corrected = min(1.0, (number - rank) * p_values[index])
        running_maximum = max(running_maximum, corrected)
        adjusted[index] = running_maximum
    return adjusted


def _pairwise_confirmation_tests(
    run_table: pd.DataFrame,
    configurations: Sequence[NSGA3Configuration],
) -> pd.DataFrame:
    baseline_id = baseline_configuration_id(configurations)
    held_out = run_table[run_table["stage"] == "confirmation"]
    baseline = held_out[held_out["configuration_id"] == baseline_id][
        ["run", "hypervolume"]
    ].rename(columns={"hypervolume": "baseline_hypervolume"})
    records = []
    for configuration_id in sorted(held_out["configuration_id"].unique()):
        if configuration_id == baseline_id:
            continue
        candidate = held_out[held_out["configuration_id"] == configuration_id][
            ["run", "hypervolume"]
        ].rename(columns={"hypervolume": "candidate_hypervolume"})
        paired = baseline.merge(candidate, on="run", validate="one_to_one")
        differences = (
            paired["candidate_hypervolume"]
            - paired["baseline_hypervolume"]
        ).to_numpy()
        test = stats.wilcoxon(differences, alternative="two-sided")
        interval_low, interval_high = _bootstrap_mean_interval(
            differences, DESIGN_SEED + len(records) + 1
        )
        records.append(
            {
                "baseline_configuration_id": baseline_id,
                "candidate_configuration_id": configuration_id,
                "held_out_pairs": len(paired),
                "mean_hypervolume_difference": differences.mean(),
                "median_hypervolume_difference": np.median(differences),
                "mean_difference_ci95_low": interval_low,
                "mean_difference_ci95_high": interval_high,
                "wilcoxon_statistic": test.statistic,
                "p_value": test.pvalue,
                "rank_biserial_correlation": _rank_biserial_difference(
                    differences
                ),
            }
        )
    table = pd.DataFrame.from_records(records)
    table["holm_adjusted_p_value"] = _holm_adjust(table["p_value"])
    return table.sort_values("mean_hypervolume_difference", ascending=False)


def _save_main_effects_figure(effects: pd.DataFrame) -> None:
    with plt.rc_context(CHART_STYLE):
        figure, axes = plt.subplots(
            2, 3, figsize=(10.0, 6.2), constrained_layout=True
        )
        for axis, factor in zip(axes.flat, FACTOR_COLUMNS, strict=False):
            rows = effects[effects["factor"] == factor].sort_values("level")
            labels = [f"{value:g}" for value in rows["level"]]
            if factor == "population_size":
                generation_map = dict(POPULATION_GENERATION_LEVELS)
                labels = [
                    f"{int(value)}/{generation_map[int(value)]}"
                    for value in rows["level"]
                ]
            positions = np.arange(len(rows))
            axis.errorbar(
                positions,
                rows["hypervolume_mean"],
                yerr=1.96 * rows["hypervolume_standard_error"],
                color="black",
                linestyle="-",
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                capsize=3,
                linewidth=1.0,
            )
            axis.set_xticks(positions, labels)
            axis.set_title(FACTOR_LABELS[factor])
            axis.grid(True, color="0.82", linewidth=0.6, linestyle=":")
            axis.set_ylabel("Hipervolume médio")
        axes.flat[-1].axis("off")
        figure.savefig(MAIN_EFFECTS_FIGURE_PATH, dpi=300)
        plt.close(figure)


def _save_confirmation_figure(
    run_table: pd.DataFrame,
    confirmation_ids: Sequence[str],
    baseline_id: str,
) -> None:
    data = [
        run_table.loc[
            run_table["configuration_id"] == configuration_id,
            "hypervolume",
        ].to_numpy()
        for configuration_id in confirmation_ids
    ]
    labels = [
        f"{configuration_id}\n(referência)"
        if configuration_id == baseline_id
        else configuration_id
        for configuration_id in confirmation_ids
    ]
    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        boxplot = axis.boxplot(
            data,
            tick_labels=labels,
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
        hatches = ("", "///", "xxx", "...", "\\\\")
        for patch, hatch in zip(boxplot["boxes"], hatches, strict=True):
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_hatch(hatch)
        axis.set_ylabel("Hipervolume")
        axis.grid(True, axis="y", color="0.82", linewidth=0.6, linestyle=":")
        figure.savefig(CONFIRMATION_FIGURE_PATH, dpi=300)
        plt.close(figure)


def _save_convergence_figure(
    convergence: pd.DataFrame,
    confirmation_ids: Sequence[str],
) -> None:
    with plt.rc_context(CHART_STYLE):
        figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        for index, configuration_id in enumerate(confirmation_ids):
            rows = convergence[
                convergence["configuration_id"] == configuration_id
            ]
            mean_curve = (
                rows.groupby("evaluations", as_index=False)["hypervolume"]
                .mean()
                .sort_values("evaluations")
            )
            style = BLACK_AND_WHITE_SERIES_STYLES[index]
            axis.plot(
                mean_curve["evaluations"],
                mean_curve["hypervolume"],
                color="black",
                linewidth=1.1,
                markersize=3.5,
                markerfacecolor="white",
                markeredgecolor="black",
                label=configuration_id,
                **style,
            )
        axis.set_xlabel("Avaliações do modelo FTHA")
        axis.set_ylabel("Hipervolume médio acumulado")
        axis.grid(True, color="0.82", linewidth=0.6, linestyle=":")
        axis.legend(frameon=True, edgecolor="black", fontsize=8)
        figure.savefig(CONVERGENCE_FIGURE_PATH, dpi=300)
        plt.close(figure)


def save_analysis(
    configurations: Sequence[NSGA3Configuration],
    confirmation_ids: Sequence[str],
    results: Sequence[SensitivityRunResult],
) -> dict[str, pd.DataFrame]:
    """Persist all numerical and graphical sensitivity artifacts."""
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tables = build_numerical_tables(configurations, confirmation_ids, results)
    path_by_table = {
        "design": DESIGN_PATH,
        "pareto": PARETO_PATH,
        "reference": REFERENCE_FRONT_PATH,
        "runs": RUNS_PATH,
        "summary": SUMMARY_PATH,
        "effects": EFFECTS_PATH,
        "importance": IMPORTANCE_PATH,
        "interactions": INTERACTIONS_PATH,
        "pairwise": PAIRWISE_PATH,
        "convergence": CONVERGENCE_PATH,
        "best": BEST_CONFIGURATION_PATH,
    }
    for name, path in path_by_table.items():
        tables[name].to_csv(path, index=False)
    baseline_id = baseline_configuration_id(configurations)
    _save_main_effects_figure(tables["effects"])
    _save_confirmation_figure(tables["runs"], confirmation_ids, baseline_id)
    _save_convergence_figure(tables["convergence"], confirmation_ids)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return tables


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing compatible checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    configurations, confirmation_ids, results = execute_sensitivity_analysis(
        workers=arguments.workers,
        resume=not arguments.no_resume,
    )
    tables = save_analysis(configurations, confirmation_ids, results)
    print("Confirmed configurations:", ", ".join(confirmation_ids), flush=True)
    print(tables["best"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
