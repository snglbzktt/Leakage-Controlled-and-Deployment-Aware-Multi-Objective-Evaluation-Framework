from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import (
    friedmanchisquare,
    rankdata,
)


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase7_statistical_analysis_protocol_v3_2.json"
)

PREFLIGHT_LOCK_PATH = (
    AUDIT
    / "phase7_statistical_analysis_preflight_locked_v3_2.json"
)

JOINED_MATRIX_PATH = (
    AUDIT
    / "phase7_statistical_analysis_joined_matrix_v3_2.csv"
)

COMPARISON_PLAN_PATH = (
    AUDIT
    / "phase7_statistical_comparison_plan_v3_2.csv"
)

OUTPUT_DESCRIPTIVE = (
    AUDIT
    / "phase7_statistical_descriptive_summary_v3_2.csv"
)

OUTPUT_OMNIBUS = (
    AUDIT
    / "phase7_statistical_omnibus_results_v3_2.csv"
)

OUTPUT_POSTHOC = (
    AUDIT
    / "phase7_statistical_posthoc_vs_B0_results_v3_2.csv"
)

OUTPUT_SUMMARY = (
    AUDIT
    / "phase7_statistical_analysis_summary_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase7_statistical_analysis_completed_v3_2.json"
)

PROTOCOL_VERSION = "phase7_statistical_analysis_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
    "DQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
)

NON_BASELINE_VARIANTS = tuple(
    variant
    for variant in VARIANTS
    if variant != "B0"
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

METRICS = {
    "test_fingerprint_macro_f1": {
        "endpoint_class": "confirmatory",
        "domain": "predictive",
        "direction": "higher_is_better",
        "unit": "proportion",
    },
    "batch_1_median_ms": {
        "endpoint_class": "confirmatory",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "batch_32_samples_per_second": {
        "endpoint_class": "confirmatory",
        "domain": "deployment",
        "direction": "higher_is_better",
        "unit": "samples_per_second",
    },
    "serialized_state_dict_bytes": {
        "endpoint_class": "confirmatory",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "bytes",
    },
    "test_raw_weighted_macro_f1": {
        "endpoint_class": "exploratory",
        "domain": "predictive",
        "direction": "higher_is_better",
        "unit": "proportion",
    },
    "batch_1_p95_ms": {
        "endpoint_class": "exploratory",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "batch_32_p95_ms": {
        "endpoint_class": "exploratory",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "milliseconds",
    },
    "peak_inference_RSS_delta_bytes": {
        "endpoint_class": "exploratory",
        "domain": "deployment",
        "direction": "lower_is_better",
        "unit": "bytes",
    },
}

EXPECTED_JOINED_ROWS = 110
EXPECTED_DESCRIPTIVE_ROWS = 176
EXPECTED_OMNIBUS_ROWS = 16
EXPECTED_POSTHOC_ROWS = 160

GLOBAL_RANDOM_SEED = 2026
MONTE_CARLO_PERMUTATIONS = 100_000
PERMUTATION_CHUNK_SIZE = 5_000
EXACT_SIGN_FLIP_ASSIGNMENTS = 32
EXACT_BOOTSTRAP_RESAMPLES = 3_125
ALPHA = 0.05
TOLERANCE = 1e-15

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    replace_with_retry(
        temporary,
        path,
    )


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    replace_with_retry(
        temporary,
        path,
    )


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def finite_float(
    value: Any,
    field_name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise RuntimeError(
            f"Non-finite value in {field_name}: {value}"
        )

    return number


def population_summary(
    values: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(
            np.std(
                array,
                ddof=0,
            )
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def friedman_statistic(
    rank_matrix: np.ndarray,
) -> float:
    n_blocks, k_variants = (
        rank_matrix.shape
    )

    rank_sums = np.sum(
        rank_matrix,
        axis=0,
        dtype=np.float64,
    )

    statistic = (
        12.0
        / (
            n_blocks
            * k_variants
            * (
                k_variants + 1
            )
        )
        * float(
            np.sum(
                rank_sums ** 2
            )
        )
        - 3.0
        * n_blocks
        * (
            k_variants + 1
        )
    )

    return float(statistic)


def monte_carlo_friedman_p(
    rank_matrix: np.ndarray,
    observed_statistic: float,
    random_seed: int,
) -> tuple[float, int]:
    rng = np.random.default_rng(
        random_seed
    )

    n_blocks, k_variants = (
        rank_matrix.shape
    )

    exceedances = 0
    completed = 0

    while completed < MONTE_CARLO_PERMUTATIONS:
        chunk_size = min(
            PERMUTATION_CHUNK_SIZE,
            (
                MONTE_CARLO_PERMUTATIONS
                - completed
            ),
        )

        rank_sums = np.zeros(
            (
                chunk_size,
                k_variants,
            ),
            dtype=np.float64,
        )

        for block_index in range(
            n_blocks
        ):
            random_values = rng.random(
                (
                    chunk_size,
                    k_variants,
                )
            )

            permutations = np.argsort(
                random_values,
                axis=1,
            )

            rank_sums += (
                rank_matrix[
                    block_index
                ][permutations]
            )

        null_statistics = (
            12.0
            / (
                n_blocks
                * k_variants
                * (
                    k_variants + 1
                )
            )
            * np.sum(
                rank_sums ** 2,
                axis=1,
            )
            - 3.0
            * n_blocks
            * (
                k_variants + 1
            )
        )

        exceedances += int(
            np.count_nonzero(
                null_statistics
                >= (
                    observed_statistic
                    - TOLERANCE
                )
            )
        )

        completed += chunk_size

    p_value = (
        exceedances + 1
    ) / (
        MONTE_CARLO_PERMUTATIONS
        + 1
    )

    return (
        float(p_value),
        int(exceedances),
    )


SIGN_MATRIX = np.asarray(
    list(
        itertools.product(
            (-1.0, 1.0),
            repeat=len(SEEDS),
        )
    ),
    dtype=np.float64,
)

BOOTSTRAP_INDEX_MATRIX = np.asarray(
    list(
        itertools.product(
            range(len(SEEDS)),
            repeat=len(SEEDS),
        )
    ),
    dtype=np.int64,
)


def exact_sign_flip_test(
    raw_deltas: np.ndarray,
) -> dict[str, float | int]:
    deltas = np.asarray(
        raw_deltas,
        dtype=np.float64,
    )

    observed = float(
        np.mean(deltas)
    )

    null_statistics = np.abs(
        np.mean(
            SIGN_MATRIX
            * deltas[
                np.newaxis,
                :
            ],
            axis=1,
        )
    )

    exceedances = int(
        np.count_nonzero(
            null_statistics
            >= (
                abs(observed)
                - TOLERANCE
            )
        )
    )

    p_value = (
        exceedances
        / len(
            SIGN_MATRIX
        )
    )

    return {
        "observed_mean_raw_delta": observed,
        "exact_two_sided_p": float(
            p_value
        ),
        "exceedance_assignments": (
            exceedances
        ),
        "exact_assignments": int(
            len(
                SIGN_MATRIX
            )
        ),
    }


def exhaustive_bootstrap_ci(
    raw_deltas: np.ndarray,
) -> tuple[float, float]:
    deltas = np.asarray(
        raw_deltas,
        dtype=np.float64,
    )

    bootstrap_means = np.mean(
        deltas[
            BOOTSTRAP_INDEX_MATRIX
        ],
        axis=1,
    )

    lower, upper = np.percentile(
        bootstrap_means,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(lower),
        float(upper),
    )


def rank_biserial_improvement(
    improvement_deltas: np.ndarray,
) -> float:
    deltas = np.asarray(
        improvement_deltas,
        dtype=np.float64,
    )

    nonzero = deltas[
        np.abs(deltas)
        > TOLERANCE
    ]

    if len(nonzero) == 0:
        return 0.0

    absolute_ranks = rankdata(
        np.abs(nonzero),
        method="average",
    )

    positive_rank_sum = float(
        np.sum(
            absolute_ranks[
                nonzero > 0
            ]
        )
    )

    negative_rank_sum = float(
        np.sum(
            absolute_ranks[
                nonzero < 0
            ]
        )
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0.0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


def holm_adjust(
    p_values: list[float],
) -> tuple[
    list[float],
    list[bool],
]:
    count = len(p_values)

    order = sorted(
        range(count),
        key=lambda index: (
            p_values[index],
            index,
        ),
    )

    adjusted_sorted: list[
        float
    ] = []

    running_maximum = 0.0

    for rank, index in enumerate(
        order
    ):
        multiplier = (
            count - rank
        )

        candidate = min(
            1.0,
            multiplier
            * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_sorted.append(
            running_maximum
        )

    adjusted = [
        1.0
    ] * count

    for index, value in zip(
        order,
        adjusted_sorted,
    ):
        adjusted[index] = float(
            value
        )

    rejected = [
        value < ALPHA
        or math.isclose(
            value,
            ALPHA,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        for value in adjusted
    ]

    return (
        adjusted,
        rejected,
    )


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    JOINED_MATRIX_PATH,
    COMPARISON_PLAN_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_DESCRIPTIVE,
    OUTPUT_OMNIBUS,
    OUTPUT_POSTHOC,
    OUTPUT_SUMMARY,
    OUTPUT_COMPLETION,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 7 statistical output already exists; "
            f"refusing to overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

joined_rows = read_csv(
    JOINED_MATRIX_PATH
)

plan_rows = read_csv(
    COMPARISON_PLAN_PATH
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and protocol.get(
            "all_checks_passed"
        )
        is True
    ),
    "preflight_locked": (
        preflight_lock.get("status")
        == "locked"
        and preflight_lock.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and preflight_lock.get(
            "ready_for_locked_statistical_execution"
        )
        is True
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        preflight_lock.get(
            "protocol_sha256"
        )
        == sha256_file(
            PROTOCOL_PATH
        )
    ),
    "joined_matrix_hash_matches": (
        preflight_lock.get(
            "joined_matrix_sha256"
        )
        == sha256_file(
            JOINED_MATRIX_PATH
        )
    ),
    "comparison_plan_hash_matches": (
        preflight_lock.get(
            "comparison_plan_sha256"
        )
        == sha256_file(
            COMPARISON_PLAN_PATH
        )
    ),
    "joined_row_count_110": (
        len(joined_rows)
        == EXPECTED_JOINED_ROWS
    ),
    "comparison_plan_row_count_176": (
        len(plan_rows)
        == (
            EXPECTED_OMNIBUS_ROWS
            + EXPECTED_POSTHOC_ROWS
        )
    ),
    "inferential_computation_was_not_previously_run": (
        preflight_lock.get(
            "inferential_computation_performed"
        )
        is False
    ),
    "final_model_not_selected": (
        preflight_lock.get(
            "final_model_selected"
        )
        is False
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 7 statistical execution entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

joined_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in joined_rows
}

expected_keys = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

if set(
    joined_lookup.keys()
) != expected_keys:
    raise RuntimeError(
        "Phase 7 joined matrix is incomplete."
    )

print("=" * 92)
print("PHASE 7 LOCKED STATISTICAL ANALYSIS")
print("=" * 92)
print(
    "Joined seed-level rows          : 110"
)
print(
    "Architectures                   : 2"
)
print(
    "Metrics                         : 8"
)
print(
    "Omnibus analyses                : 16"
)
print(
    "Post-hoc comparisons            : 160"
)
print(
    "Omnibus permutations each       : 100000"
)
print(
    "Exact sign-flip assignments     : 32"
)
print(
    "Exact bootstrap resamples       : 3125"
)
print(
    "Holm correction                 : within architecture-metric family"
)
print(
    "Final model selected            : False"
)
print()

descriptive_rows: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for metric_name, specification in METRICS.items():
        for variant in VARIANTS:
            values = np.asarray(
                [
                    finite_float(
                        joined_lookup[
                            (
                                architecture,
                                variant,
                                seed,
                            )
                        ][metric_name],
                        metric_name,
                    )
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )

            summary = population_summary(
                values
            )

            descriptive_rows.append(
                {
                    "endpoint_class": (
                        specification[
                            "endpoint_class"
                        ]
                    ),
                    "architecture": architecture,
                    "metric": metric_name,
                    "metric_domain": (
                        specification[
                            "domain"
                        ]
                    ),
                    "direction": (
                        specification[
                            "direction"
                        ]
                    ),
                    "unit": (
                        specification[
                            "unit"
                        ]
                    ),
                    "variant": variant,
                    "seed_count": len(
                        SEEDS
                    ),
                    "mean": summary["mean"],
                    "median": summary[
                        "median"
                    ],
                    "std_population": (
                        summary[
                            "std_population"
                        ]
                    ),
                    "minimum": summary[
                        "minimum"
                    ],
                    "maximum": summary[
                        "maximum"
                    ],
                }
            )

if len(
    descriptive_rows
) != EXPECTED_DESCRIPTIVE_ROWS:
    raise RuntimeError(
        "Descriptive result row count mismatch."
    )

atomic_csv(
    OUTPUT_DESCRIPTIVE,
    descriptive_rows,
    [
        "endpoint_class",
        "architecture",
        "metric",
        "metric_domain",
        "direction",
        "unit",
        "variant",
        "seed_count",
        "mean",
        "median",
        "std_population",
        "minimum",
        "maximum",
    ],
)

omnibus_rows: list[
    dict[str, Any]
] = []

analysis_index = 0

for architecture_index, architecture in enumerate(
    ARCHITECTURES
):
    for metric_index, (
        metric_name,
        specification,
    ) in enumerate(
        METRICS.items()
    ):
        analysis_index += 1

        value_matrix = np.asarray(
            [
                [
                    finite_float(
                        joined_lookup[
                            (
                                architecture,
                                variant,
                                seed,
                            )
                        ][metric_name],
                        metric_name,
                    )
                    for variant in VARIANTS
                ]
                for seed in SEEDS
            ],
            dtype=np.float64,
        )

        rank_matrix = np.vstack(
            [
                rankdata(
                    row,
                    method="average",
                )
                for row in value_matrix
            ]
        )

        observed_statistic = (
            friedman_statistic(
                rank_matrix
            )
        )

        random_seed = (
            GLOBAL_RANDOM_SEED
            + architecture_index
            * 10_000
            + metric_index
            * 100
        )

        (
            permutation_p,
            exceedances,
        ) = monte_carlo_friedman_p(
            rank_matrix,
            observed_statistic,
            random_seed,
        )

        scipy_result = (
            friedmanchisquare(
                *[
                    value_matrix[
                        :,
                        variant_index,
                    ]
                    for variant_index in range(
                        len(
                            VARIANTS
                        )
                    )
                ]
            )
        )

        kendalls_w = (
            observed_statistic
            / (
                len(SEEDS)
                * (
                    len(VARIANTS)
                    - 1
                )
            )
        )

        omnibus_rows.append(
            {
                "endpoint_class": (
                    specification[
                        "endpoint_class"
                    ]
                ),
                "family_id": (
                    f"{specification['endpoint_class']}"
                    f"__{architecture}"
                    f"__{metric_name}"
                ),
                "architecture": architecture,
                "metric": metric_name,
                "metric_domain": (
                    specification[
                        "domain"
                    ]
                ),
                "direction": (
                    specification[
                        "direction"
                    ]
                ),
                "unit": (
                    specification[
                        "unit"
                    ]
                ),
                "block_count": len(SEEDS),
                "variant_count": len(
                    VARIANTS
                ),
                "friedman_statistic": (
                    observed_statistic
                ),
                "kendalls_W": (
                    kendalls_w
                ),
                "monte_carlo_permutations": (
                    MONTE_CARLO_PERMUTATIONS
                ),
                "monte_carlo_random_seed": (
                    random_seed
                ),
                "monte_carlo_exceedances": (
                    exceedances
                ),
                "monte_carlo_p_plus_one": (
                    permutation_p
                ),
                "scipy_asymptotic_statistic": float(
                    scipy_result.statistic
                ),
                "scipy_asymptotic_p": float(
                    scipy_result.pvalue
                ),
                "nominal_alpha": ALPHA,
                "monte_carlo_reject_nominal": (
                    permutation_p
                    <= ALPHA
                ),
                "sensitivity_only_scipy": True,
            }
        )

        print(
            f"[omnibus {analysis_index}/16] "
            f"{architecture} | {metric_name} | "
            f"Q={observed_statistic:.6f} | "
            f"MC p={permutation_p:.6f}",
            flush=True,
        )

if len(
    omnibus_rows
) != EXPECTED_OMNIBUS_ROWS:
    raise RuntimeError(
        "Omnibus result row count mismatch."
    )

atomic_csv(
    OUTPUT_OMNIBUS,
    omnibus_rows,
    [
        "endpoint_class",
        "family_id",
        "architecture",
        "metric",
        "metric_domain",
        "direction",
        "unit",
        "block_count",
        "variant_count",
        "friedman_statistic",
        "kendalls_W",
        "monte_carlo_permutations",
        "monte_carlo_random_seed",
        "monte_carlo_exceedances",
        "monte_carlo_p_plus_one",
        "scipy_asymptotic_statistic",
        "scipy_asymptotic_p",
        "nominal_alpha",
        "monte_carlo_reject_nominal",
        "sensitivity_only_scipy",
    ],
)

posthoc_rows: list[
    dict[str, Any]
] = []

comparison_index = 0

for architecture in ARCHITECTURES:
    for metric_name, specification in METRICS.items():
        family_start = len(
            posthoc_rows
        )

        baseline_values = np.asarray(
            [
                finite_float(
                    joined_lookup[
                        (
                            architecture,
                            "B0",
                            seed,
                        )
                    ][metric_name],
                    metric_name,
                )
                for seed in SEEDS
            ],
            dtype=np.float64,
        )

        family_rows: list[
            dict[str, Any]
        ] = []

        for variant in NON_BASELINE_VARIANTS:
            comparison_index += 1

            candidate_values = np.asarray(
                [
                    finite_float(
                        joined_lookup[
                            (
                                architecture,
                                variant,
                                seed,
                            )
                        ][metric_name],
                        metric_name,
                    )
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )

            raw_deltas = (
                candidate_values
                - baseline_values
            )

            if (
                specification[
                    "direction"
                ]
                == "higher_is_better"
            ):
                improvement_deltas = (
                    raw_deltas.copy()
                )
            else:
                improvement_deltas = (
                    -raw_deltas
                )

            sign_flip = (
                exact_sign_flip_test(
                    raw_deltas
                )
            )

            (
                bootstrap_lower,
                bootstrap_upper,
            ) = exhaustive_bootstrap_ci(
                raw_deltas
            )

            wins = int(
                np.count_nonzero(
                    improvement_deltas
                    > TOLERANCE
                )
            )

            ties = int(
                np.count_nonzero(
                    np.abs(
                        improvement_deltas
                    )
                    <= TOLERANCE
                )
            )

            losses = int(
                np.count_nonzero(
                    improvement_deltas
                    < -TOLERANCE
                )
            )

            row = {
                "endpoint_class": (
                    specification[
                        "endpoint_class"
                    ]
                ),
                "family_id": (
                    f"{specification['endpoint_class']}"
                    f"__{architecture}"
                    f"__{metric_name}"
                ),
                "architecture": architecture,
                "metric": metric_name,
                "metric_domain": (
                    specification[
                        "domain"
                    ]
                ),
                "direction": (
                    specification[
                        "direction"
                    ]
                ),
                "unit": (
                    specification[
                        "unit"
                    ]
                ),
                "reference_variant": "B0",
                "candidate_variant": variant,
                "paired_seed_count": len(
                    SEEDS
                ),
                "reference_mean": float(
                    np.mean(
                        baseline_values
                    )
                ),
                "candidate_mean": float(
                    np.mean(
                        candidate_values
                    )
                ),
                "mean_raw_delta_candidate_minus_B0": float(
                    np.mean(
                        raw_deltas
                    )
                ),
                "median_raw_delta_candidate_minus_B0": float(
                    np.median(
                        raw_deltas
                    )
                ),
                "mean_improvement_delta": float(
                    np.mean(
                        improvement_deltas
                    )
                ),
                "median_improvement_delta": float(
                    np.median(
                        improvement_deltas
                    )
                ),
                "bootstrap_95_CI_raw_delta_lower": (
                    bootstrap_lower
                ),
                "bootstrap_95_CI_raw_delta_upper": (
                    bootstrap_upper
                ),
                "paired_rank_biserial_improvement": (
                    rank_biserial_improvement(
                        improvement_deltas
                    )
                ),
                "improvement_wins": wins,
                "improvement_ties": ties,
                "improvement_losses": losses,
                "exact_sign_flip_assignments": (
                    sign_flip[
                        "exact_assignments"
                    ]
                ),
                "exact_sign_flip_exceedances": (
                    sign_flip[
                        "exceedance_assignments"
                    ]
                ),
                "exact_two_sided_p_raw": (
                    sign_flip[
                        "exact_two_sided_p"
                    ]
                ),
                "Holm_adjusted_p": None,
                "Holm_reject_alpha_0_05": (
                    None
                ),
                "omnibus_monte_carlo_p": (
                    next(
                        row[
                            "monte_carlo_p_plus_one"
                        ]
                        for row in omnibus_rows
                        if row[
                            "architecture"
                        ]
                        == architecture
                        and row[
                            "metric"
                        ]
                        == metric_name
                    )
                ),
                "final_model_selection_used": (
                    False
                ),
            }

            family_rows.append(
                row
            )

        raw_p_values = [
            float(
                row[
                    "exact_two_sided_p_raw"
                ]
            )
            for row in family_rows
        ]

        adjusted_values, rejected = (
            holm_adjust(
                raw_p_values
            )
        )

        for row, adjusted, reject in zip(
            family_rows,
            adjusted_values,
            rejected,
        ):
            row[
                "Holm_adjusted_p"
            ] = adjusted

            row[
                "Holm_reject_alpha_0_05"
            ] = reject

        posthoc_rows.extend(
            family_rows
        )

        if (
            len(posthoc_rows)
            - family_start
        ) != 10:
            raise RuntimeError(
                "Post-hoc family row count mismatch."
            )

        family_best = max(
            family_rows,
            key=lambda row: float(
                row[
                    "mean_improvement_delta"
                ]
            ),
        )

        print(
            f"[family {len(posthoc_rows) // 10}/16] "
            f"{architecture} | {metric_name} | "
            "10 comparisons | best mean improvement="
            f"{family_best['candidate_variant']} "
            f"({float(family_best['mean_improvement_delta']):.9g})",
            flush=True,
        )

if len(
    posthoc_rows
) != EXPECTED_POSTHOC_ROWS:
    raise RuntimeError(
        "Post-hoc result row count mismatch."
    )

atomic_csv(
    OUTPUT_POSTHOC,
    posthoc_rows,
    [
        "endpoint_class",
        "family_id",
        "architecture",
        "metric",
        "metric_domain",
        "direction",
        "unit",
        "reference_variant",
        "candidate_variant",
        "paired_seed_count",
        "reference_mean",
        "candidate_mean",
        "mean_raw_delta_candidate_minus_B0",
        "median_raw_delta_candidate_minus_B0",
        "mean_improvement_delta",
        "median_improvement_delta",
        "bootstrap_95_CI_raw_delta_lower",
        "bootstrap_95_CI_raw_delta_upper",
        "paired_rank_biserial_improvement",
        "improvement_wins",
        "improvement_ties",
        "improvement_losses",
        "exact_sign_flip_assignments",
        "exact_sign_flip_exceedances",
        "exact_two_sided_p_raw",
        "Holm_adjusted_p",
        "Holm_reject_alpha_0_05",
        "omnibus_monte_carlo_p",
        "final_model_selection_used",
    ],
)

confirmatory_omnibus_rejections = sum(
    row["endpoint_class"]
    == "confirmatory"
    and bool(
        row[
            "monte_carlo_reject_nominal"
        ]
    )
    for row in omnibus_rows
)

exploratory_omnibus_rejections = sum(
    row["endpoint_class"]
    == "exploratory"
    and bool(
        row[
            "monte_carlo_reject_nominal"
        ]
    )
    for row in omnibus_rows
)

confirmatory_Holm_rejections = sum(
    row["endpoint_class"]
    == "confirmatory"
    and bool(
        row[
            "Holm_reject_alpha_0_05"
        ]
    )
    for row in posthoc_rows
)

exploratory_Holm_rejections = sum(
    row["endpoint_class"]
    == "exploratory"
    and bool(
        row[
            "Holm_reject_alpha_0_05"
        ]
    )
    for row in posthoc_rows
)

minimum_raw_posthoc_p = min(
    float(
        row[
            "exact_two_sided_p_raw"
        ]
    )
    for row in posthoc_rows
)

minimum_Holm_p = min(
    float(
        row[
            "Holm_adjusted_p"
        ]
    )
    for row in posthoc_rows
)

global_checks = {
    "descriptive_rows_176": (
        len(descriptive_rows)
        == EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_rows_16": (
        len(omnibus_rows)
        == EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_rows_160": (
        len(posthoc_rows)
        == EXPECTED_POSTHOC_ROWS
    ),
    "all_omnibus_values_finite": all(
        math.isfinite(
            float(
                row[
                    "friedman_statistic"
                ]
            )
        )
        and math.isfinite(
            float(
                row[
                    "monte_carlo_p_plus_one"
                ]
            )
        )
        for row in omnibus_rows
    ),
    "all_posthoc_values_finite": all(
        math.isfinite(
            float(
                row[
                    "mean_raw_delta_candidate_minus_B0"
                ]
            )
        )
        and math.isfinite(
            float(
                row[
                    "exact_two_sided_p_raw"
                ]
            )
        )
        and math.isfinite(
            float(
                row[
                    "Holm_adjusted_p"
                ]
            )
        )
        for row in posthoc_rows
    ),
    "all_exact_p_values_at_least_0_0625": all(
        float(
            row[
                "exact_two_sided_p_raw"
            ]
        )
        >= (
            0.0625
            - TOLERANCE
        )
        for row in posthoc_rows
    ),
    "Holm_p_not_below_raw_p": all(
        float(
            row[
                "Holm_adjusted_p"
            ]
        )
        + TOLERANCE
        >= float(
            row[
                "exact_two_sided_p_raw"
            ]
        )
        for row in posthoc_rows
    ),
    "wins_ties_losses_sum_to_5": all(
        int(
            row[
                "improvement_wins"
            ]
        )
        + int(
            row[
                "improvement_ties"
            ]
        )
        + int(
            row[
                "improvement_losses"
            ]
        )
        == 5
        for row in posthoc_rows
    ),
    "final_model_not_selected": all(
        row[
            "final_model_selection_used"
        ]
        is False
        for row in posthoc_rows
    ),
}

failed_global_checks = [
    name
    for name, passed
    in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "Phase 7 statistical execution global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

summary = {
    "status": "completed",
    "phase": 7,
    "artifact_name": "locked_statistical_analysis",
    "protocol_version": PROTOCOL_VERSION,
    "completed_at_utc": utc_now(),
    "descriptive_row_count": (
        EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_analysis_count": (
        EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_comparison_count": (
        EXPECTED_POSTHOC_ROWS
    ),
    "confirmatory_omnibus_rejections_nominal": (
        confirmatory_omnibus_rejections
    ),
    "exploratory_omnibus_rejections_nominal": (
        exploratory_omnibus_rejections
    ),
    "confirmatory_posthoc_Holm_rejections": (
        confirmatory_Holm_rejections
    ),
    "exploratory_posthoc_Holm_rejections": (
        exploratory_Holm_rejections
    ),
    "minimum_raw_posthoc_p": (
        minimum_raw_posthoc_p
    ),
    "minimum_Holm_adjusted_p": (
        minimum_Holm_p
    ),
    "small_sample_interpretation": {
        "paired_seed_count": 5,
        "minimum_possible_two_sided_exact_p": (
            0.0625
        ),
        "absence_of_p_below_0_05_is_not_equivalence": (
            True
        ),
        "effect_sizes_and_consistency_required": (
            True
        ),
    },
    "entry_checks": entry_checks,
    "global_checks": global_checks,
    "artifacts": {
        "descriptive_summary": file_record(
            OUTPUT_DESCRIPTIVE
        ),
        "omnibus_results": file_record(
            OUTPUT_OMNIBUS
        ),
        "posthoc_results": file_record(
            OUTPUT_POSTHOC
        ),
    },
    "final_model_selected": False,
    "ready_for_independent_verification": True,
    "ready_for_multi_objective_decision_preflight": (
        False
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SUMMARY,
    summary,
)

completion = {
    "status": "completed",
    "phase": 7,
    "artifact_name": "locked_statistical_analysis",
    "protocol_version": PROTOCOL_VERSION,
    "completed_at_utc": utc_now(),
    "descriptive_row_count": (
        EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_analysis_count": (
        EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_comparison_count": (
        EXPECTED_POSTHOC_ROWS
    ),
    "descriptive_summary": str(
        OUTPUT_DESCRIPTIVE
    ),
    "descriptive_summary_sha256": (
        sha256_file(
            OUTPUT_DESCRIPTIVE
        )
    ),
    "omnibus_results": str(
        OUTPUT_OMNIBUS
    ),
    "omnibus_results_sha256": (
        sha256_file(
            OUTPUT_OMNIBUS
        )
    ),
    "posthoc_results": str(
        OUTPUT_POSTHOC
    ),
    "posthoc_results_sha256": (
        sha256_file(
            OUTPUT_POSTHOC
        )
    ),
    "summary_json": str(
        OUTPUT_SUMMARY
    ),
    "summary_json_sha256": (
        sha256_file(
            OUTPUT_SUMMARY
        )
    ),
    "final_model_selected": False,
    "ready_for_independent_verification": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

print()
print("=" * 92)
print("PHASE 7 STATISTICAL ANALYSIS SUMMARY")
print("=" * 92)
print(
    "Descriptive rows                : 176"
)
print(
    "Omnibus analyses                : 16"
)
print(
    "Post-hoc comparisons            : 160"
)
print(
    "Confirmatory omnibus rejections : "
    f"{confirmatory_omnibus_rejections}"
)
print(
    "Exploratory omnibus rejections  : "
    f"{exploratory_omnibus_rejections}"
)
print(
    "Confirmatory Holm rejections    : "
    f"{confirmatory_Holm_rejections}"
)
print(
    "Exploratory Holm rejections     : "
    f"{exploratory_Holm_rejections}"
)
print(
    "Minimum raw exact p             : "
    f"{minimum_raw_posthoc_p:.6f}"
)
print(
    "Minimum Holm-adjusted p         : "
    f"{minimum_Holm_p:.6f}"
)
print(
    "Five-seed exact p floor         : 0.062500"
)
print(
    "Final model selected            : False"
)
print(
    "Ready for independent verify    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 7 LOCKED STATISTICAL ANALYSIS COMPLETED"
)
