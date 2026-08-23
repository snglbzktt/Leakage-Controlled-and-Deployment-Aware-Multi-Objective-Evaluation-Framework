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
from scipy.stats import friedmanchisquare, rankdata


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

DESCRIPTIVE_PATH = (
    AUDIT
    / "phase7_statistical_descriptive_summary_v3_2.csv"
)

OMNIBUS_PATH = (
    AUDIT
    / "phase7_statistical_omnibus_results_v3_2.csv"
)

POSTHOC_PATH = (
    AUDIT
    / "phase7_statistical_posthoc_vs_B0_results_v3_2.csv"
)

SUMMARY_PATH = (
    AUDIT
    / "phase7_statistical_analysis_summary_v3_2.json"
)

COMPLETION_PATH = (
    AUDIT
    / "phase7_statistical_analysis_completed_v3_2.json"
)

OUTPUT_VERIFIED_OMNIBUS = (
    AUDIT
    / "phase7_statistical_omnibus_verified_v3_2.csv"
)

OUTPUT_VERIFIED_POSTHOC = (
    AUDIT
    / "phase7_statistical_posthoc_verified_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase7_statistical_analysis_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase7_statistical_analysis_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase7_statistical_analysis_lock_manifest_v3_2.json"
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
ALPHA = 0.05

TOLERANCE = 1e-10
RANK_TOLERANCE = 1e-15

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
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
            os.replace(source, destination)
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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


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


def close_enough(
    observed: Any,
    expected: Any,
    tolerance: float = TOLERANCE,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


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
            np.std(array, ddof=0)
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def friedman_statistic(
    rank_matrix: np.ndarray,
) -> float:
    n_blocks, k_variants = rank_matrix.shape

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
            * (k_variants + 1)
        )
        * float(np.sum(rank_sums ** 2))
        - 3.0
        * n_blocks
        * (k_variants + 1)
    )

    return float(statistic)


def monte_carlo_friedman_p(
    rank_matrix: np.ndarray,
    observed_statistic: float,
    random_seed: int,
) -> tuple[float, int]:
    rng = np.random.default_rng(random_seed)

    n_blocks, k_variants = rank_matrix.shape

    exceedances = 0
    completed = 0

    while completed < MONTE_CARLO_PERMUTATIONS:
        chunk_size = min(
            PERMUTATION_CHUNK_SIZE,
            MONTE_CARLO_PERMUTATIONS - completed,
        )

        rank_sums = np.zeros(
            (chunk_size, k_variants),
            dtype=np.float64,
        )

        for block_index in range(n_blocks):
            random_values = rng.random(
                (chunk_size, k_variants)
            )

            permutations = np.argsort(
                random_values,
                axis=1,
            )

            rank_sums += rank_matrix[
                block_index
            ][permutations]

        null_statistics = (
            12.0
            / (
                n_blocks
                * k_variants
                * (k_variants + 1)
            )
            * np.sum(
                rank_sums ** 2,
                axis=1,
            )
            - 3.0
            * n_blocks
            * (k_variants + 1)
        )

        exceedances += int(
            np.count_nonzero(
                null_statistics
                >= (
                    observed_statistic
                    - RANK_TOLERANCE
                )
            )
        )

        completed += chunk_size

    return (
        float(
            (exceedances + 1)
            / (
                MONTE_CARLO_PERMUTATIONS + 1
            )
        ),
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
) -> tuple[float, int]:
    deltas = np.asarray(
        raw_deltas,
        dtype=np.float64,
    )

    observed = float(np.mean(deltas))

    null_statistics = np.abs(
        np.mean(
            SIGN_MATRIX
            * deltas[np.newaxis, :],
            axis=1,
        )
    )

    exceedances = int(
        np.count_nonzero(
            null_statistics
            >= (
                abs(observed)
                - RANK_TOLERANCE
            )
        )
    )

    return (
        float(
            exceedances
            / len(SIGN_MATRIX)
        ),
        exceedances,
    )


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
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def rank_biserial_improvement(
    improvement_deltas: np.ndarray,
) -> float:
    deltas = np.asarray(
        improvement_deltas,
        dtype=np.float64,
    )

    nonzero = deltas[
        np.abs(deltas)
        > RANK_TOLERANCE
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

    adjusted_sorted: list[float] = []
    running_maximum = 0.0

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (count - rank)
            * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_sorted.append(
            running_maximum
        )

    adjusted = [1.0] * count

    for index, value in zip(
        order,
        adjusted_sorted,
    ):
        adjusted[index] = float(value)

    rejected = [
        value <= ALPHA
        or math.isclose(
            value,
            ALPHA,
            rel_tol=0.0,
            abs_tol=RANK_TOLERANCE,
        )
        for value in adjusted
    ]

    return adjusted, rejected


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    JOINED_MATRIX_PATH,
    COMPARISON_PLAN_PATH,
    DESCRIPTIVE_PATH,
    OMNIBUS_PATH,
    POSTHOC_PATH,
    SUMMARY_PATH,
    COMPLETION_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_OMNIBUS,
    OUTPUT_VERIFIED_POSTHOC,
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 7 verification artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(PROTOCOL_PATH)
preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)
summary = read_json(SUMMARY_PATH)
completion = read_json(COMPLETION_PATH)

joined_rows = read_csv(
    JOINED_MATRIX_PATH
)
plan_rows = read_csv(
    COMPARISON_PLAN_PATH
)
descriptive_rows = read_csv(
    DESCRIPTIVE_PATH
)
saved_omnibus_rows = read_csv(
    OMNIBUS_PATH
)
saved_posthoc_rows = read_csv(
    POSTHOC_PATH
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status") == "locked"
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
            "ready_for_locked_statistical_execution"
        )
        is True
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "summary_completed": (
        summary.get("status")
        == "completed"
        and summary.get(
            "all_checks_passed"
        )
        is True
        and summary.get(
            "ready_for_independent_verification"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "descriptive_hash_matches": (
        completion.get(
            "descriptive_summary_sha256"
        )
        == sha256_file(
            DESCRIPTIVE_PATH
        )
    ),
    "omnibus_hash_matches": (
        completion.get(
            "omnibus_results_sha256"
        )
        == sha256_file(
            OMNIBUS_PATH
        )
    ),
    "posthoc_hash_matches": (
        completion.get(
            "posthoc_results_sha256"
        )
        == sha256_file(
            POSTHOC_PATH
        )
    ),
    "summary_hash_matches": (
        completion.get(
            "summary_json_sha256"
        )
        == sha256_file(
            SUMMARY_PATH
        )
    ),
    "joined_row_count_110": (
        len(joined_rows)
        == EXPECTED_JOINED_ROWS
    ),
    "plan_row_count_176": (
        len(plan_rows)
        == (
            EXPECTED_OMNIBUS_ROWS
            + EXPECTED_POSTHOC_ROWS
        )
    ),
    "descriptive_row_count_176": (
        len(descriptive_rows)
        == EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_row_count_16": (
        len(saved_omnibus_rows)
        == EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_row_count_160": (
        len(saved_posthoc_rows)
        == EXPECTED_POSTHOC_ROWS
    ),
    "final_model_not_selected": (
        summary.get(
            "final_model_selected"
        )
        is False
        and completion.get(
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
        "Phase 7 verification entry gate failed: "
        + ", ".join(failed_entry_checks)
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

if set(joined_lookup.keys()) != expected_keys:
    raise RuntimeError(
        "Phase 7 joined matrix is incomplete."
    )

descriptive_lookup = {
    (
        row["architecture"],
        row["metric"],
        row["variant"],
    ): row
    for row in descriptive_rows
}

saved_omnibus_lookup = {
    (
        row["architecture"],
        row["metric"],
    ): row
    for row in saved_omnibus_rows
}

saved_posthoc_lookup = {
    (
        row["architecture"],
        row["metric"],
        row["candidate_variant"],
    ): row
    for row in saved_posthoc_rows
}

print("=" * 92)
print("PHASE 7 STATISTICAL ANALYSIS INDEPENDENT VERIFICATION")
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
    "Monte Carlo permutations each   : 100000"
)
print(
    "Exact sign-flip recomputed      : Yes"
)
print(
    "Exact bootstrap recomputed      : Yes"
)
print(
    "Holm correction recomputed      : Yes"
)
print(
    "Model inference repeated        : False"
)
print(
    "Final model selected            : False"
)
print()

verified_omnibus_rows: list[
    dict[str, Any]
] = []

verified_posthoc_rows: list[
    dict[str, Any]
] = []

descriptive_checks_passed = 0
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

            observed_summary = population_summary(
                values
            )

            saved_descriptive = (
                descriptive_lookup[
                    (
                        architecture,
                        metric_name,
                        variant,
                    )
                ]
            )

            descriptive_checks = {
                "identity": (
                    saved_descriptive[
                        "endpoint_class"
                    ]
                    == specification[
                        "endpoint_class"
                    ]
                    and saved_descriptive[
                        "metric_domain"
                    ]
                    == specification[
                        "domain"
                    ]
                    and saved_descriptive[
                        "direction"
                    ]
                    == specification[
                        "direction"
                    ]
                    and saved_descriptive[
                        "unit"
                    ]
                    == specification[
                        "unit"
                    ]
                ),
                "seed_count": (
                    int(
                        saved_descriptive[
                            "seed_count"
                        ]
                    )
                    == len(SEEDS)
                ),
                "mean": close_enough(
                    saved_descriptive["mean"],
                    observed_summary["mean"],
                ),
                "median": close_enough(
                    saved_descriptive["median"],
                    observed_summary["median"],
                ),
                "std": close_enough(
                    saved_descriptive[
                        "std_population"
                    ],
                    observed_summary[
                        "std_population"
                    ],
                ),
                "minimum": close_enough(
                    saved_descriptive[
                        "minimum"
                    ],
                    observed_summary[
                        "minimum"
                    ],
                ),
                "maximum": close_enough(
                    saved_descriptive[
                        "maximum"
                    ],
                    observed_summary[
                        "maximum"
                    ],
                ),
            }

            failed_descriptive = [
                name
                for name, passed
                in descriptive_checks.items()
                if not passed
            ]

            if failed_descriptive:
                raise RuntimeError(
                    "Descriptive verification failed for "
                    f"{architecture} {metric_name} "
                    f"{variant}: "
                    + ", ".join(
                        failed_descriptive
                    )
                )

            descriptive_checks_passed += 1

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

        scipy_result = friedmanchisquare(
            *[
                value_matrix[
                    :,
                    variant_index,
                ]
                for variant_index in range(
                    len(VARIANTS)
                )
            ]
        )

        kendalls_w = (
            observed_statistic
            / (
                len(SEEDS)
                * (
                    len(VARIANTS) - 1
                )
            )
        )

        saved_omnibus = (
            saved_omnibus_lookup[
                (
                    architecture,
                    metric_name,
                )
            ]
        )

        omnibus_checks = {
            "identity": (
                saved_omnibus[
                    "endpoint_class"
                ]
                == specification[
                    "endpoint_class"
                ]
                and saved_omnibus[
                    "metric_domain"
                ]
                == specification[
                    "domain"
                ]
                and saved_omnibus[
                    "direction"
                ]
                == specification[
                    "direction"
                ]
                and saved_omnibus[
                    "unit"
                ]
                == specification[
                    "unit"
                ]
            ),
            "counts": (
                int(
                    saved_omnibus[
                        "block_count"
                    ]
                )
                == len(SEEDS)
                and int(
                    saved_omnibus[
                        "variant_count"
                    ]
                )
                == len(VARIANTS)
                and int(
                    saved_omnibus[
                        "monte_carlo_permutations"
                    ]
                )
                == MONTE_CARLO_PERMUTATIONS
                and int(
                    saved_omnibus[
                        "monte_carlo_random_seed"
                    ]
                )
                == random_seed
            ),
            "friedman_statistic": close_enough(
                saved_omnibus[
                    "friedman_statistic"
                ],
                observed_statistic,
            ),
            "kendalls_W": close_enough(
                saved_omnibus[
                    "kendalls_W"
                ],
                kendalls_w,
            ),
            "exceedances": (
                int(
                    saved_omnibus[
                        "monte_carlo_exceedances"
                    ]
                )
                == exceedances
            ),
            "monte_carlo_p": close_enough(
                saved_omnibus[
                    "monte_carlo_p_plus_one"
                ],
                permutation_p,
            ),
            "scipy_statistic": close_enough(
                saved_omnibus[
                    "scipy_asymptotic_statistic"
                ],
                float(
                    scipy_result.statistic
                ),
            ),
            "scipy_p": close_enough(
                saved_omnibus[
                    "scipy_asymptotic_p"
                ],
                float(
                    scipy_result.pvalue
                ),
            ),
            "reject_flag": (
                parse_bool(
                    saved_omnibus[
                        "monte_carlo_reject_nominal"
                    ]
                )
                == (
                    permutation_p <= ALPHA
                )
            ),
            "sensitivity_flag": (
                parse_bool(
                    saved_omnibus[
                        "sensitivity_only_scipy"
                    ]
                )
                is True
            ),
        }

        failed_omnibus = [
            name
            for name, passed
            in omnibus_checks.items()
            if not passed
        ]

        if failed_omnibus:
            raise RuntimeError(
                "Omnibus verification failed for "
                f"{architecture} {metric_name}: "
                + ", ".join(
                    failed_omnibus
                )
            )

        verified_omnibus_rows.append(
            {
                "endpoint_class": (
                    specification[
                        "endpoint_class"
                    ]
                ),
                "architecture": architecture,
                "metric": metric_name,
                "friedman_statistic": (
                    observed_statistic
                ),
                "kendalls_W": kendalls_w,
                "monte_carlo_p_plus_one": (
                    permutation_p
                ),
                "monte_carlo_exceedances": (
                    exceedances
                ),
                "scipy_asymptotic_p": float(
                    scipy_result.pvalue
                ),
                "all_checks_passed": True,
            }
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

        raw_p_values: list[float] = []

        for variant in NON_BASELINE_VARIANTS:
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
                specification["direction"]
                == "higher_is_better"
            ):
                improvement_deltas = (
                    raw_deltas.copy()
                )
            else:
                improvement_deltas = (
                    -raw_deltas
                )

            exact_p, exact_exceedances = (
                exact_sign_flip_test(
                    raw_deltas
                )
            )

            ci_lower, ci_upper = (
                exhaustive_bootstrap_ci(
                    raw_deltas
                )
            )

            wins = int(
                np.count_nonzero(
                    improvement_deltas
                    > RANK_TOLERANCE
                )
            )

            ties = int(
                np.count_nonzero(
                    np.abs(
                        improvement_deltas
                    )
                    <= RANK_TOLERANCE
                )
            )

            losses = int(
                np.count_nonzero(
                    improvement_deltas
                    < -RANK_TOLERANCE
                )
            )

            family_rows.append(
                {
                    "variant": variant,
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
                    "mean_raw_delta": float(
                        np.mean(
                            raw_deltas
                        )
                    ),
                    "median_raw_delta": float(
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
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "rank_biserial": (
                        rank_biserial_improvement(
                            improvement_deltas
                        )
                    ),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "exact_exceedances": (
                        exact_exceedances
                    ),
                    "exact_p": exact_p,
                }
            )

            raw_p_values.append(
                exact_p
            )

        adjusted_values, rejected = (
            holm_adjust(
                raw_p_values
            )
        )

        for (
            recomputed,
            adjusted,
            reject,
        ) in zip(
            family_rows,
            adjusted_values,
            rejected,
        ):
            variant = recomputed["variant"]

            saved = saved_posthoc_lookup[
                (
                    architecture,
                    metric_name,
                    variant,
                )
            ]

            posthoc_checks = {
                "identity": (
                    saved[
                        "endpoint_class"
                    ]
                    == specification[
                        "endpoint_class"
                    ]
                    and saved[
                        "metric_domain"
                    ]
                    == specification[
                        "domain"
                    ]
                    and saved[
                        "direction"
                    ]
                    == specification[
                        "direction"
                    ]
                    and saved["unit"]
                    == specification["unit"]
                    and saved[
                        "reference_variant"
                    ]
                    == "B0"
                ),
                "paired_seed_count": (
                    int(
                        saved[
                            "paired_seed_count"
                        ]
                    )
                    == len(SEEDS)
                ),
                "reference_mean": close_enough(
                    saved["reference_mean"],
                    recomputed[
                        "reference_mean"
                    ],
                ),
                "candidate_mean": close_enough(
                    saved["candidate_mean"],
                    recomputed[
                        "candidate_mean"
                    ],
                ),
                "mean_raw_delta": close_enough(
                    saved[
                        "mean_raw_delta_candidate_minus_B0"
                    ],
                    recomputed[
                        "mean_raw_delta"
                    ],
                ),
                "median_raw_delta": close_enough(
                    saved[
                        "median_raw_delta_candidate_minus_B0"
                    ],
                    recomputed[
                        "median_raw_delta"
                    ],
                ),
                "mean_improvement": close_enough(
                    saved[
                        "mean_improvement_delta"
                    ],
                    recomputed[
                        "mean_improvement_delta"
                    ],
                ),
                "median_improvement": close_enough(
                    saved[
                        "median_improvement_delta"
                    ],
                    recomputed[
                        "median_improvement_delta"
                    ],
                ),
                "bootstrap_lower": close_enough(
                    saved[
                        "bootstrap_95_CI_raw_delta_lower"
                    ],
                    recomputed["ci_lower"],
                ),
                "bootstrap_upper": close_enough(
                    saved[
                        "bootstrap_95_CI_raw_delta_upper"
                    ],
                    recomputed["ci_upper"],
                ),
                "rank_biserial": close_enough(
                    saved[
                        "paired_rank_biserial_improvement"
                    ],
                    recomputed[
                        "rank_biserial"
                    ],
                ),
                "win_tie_loss": (
                    int(
                        saved[
                            "improvement_wins"
                        ]
                    )
                    == recomputed["wins"]
                    and int(
                        saved[
                            "improvement_ties"
                        ]
                    )
                    == recomputed["ties"]
                    and int(
                        saved[
                            "improvement_losses"
                        ]
                    )
                    == recomputed["losses"]
                ),
                "assignments_and_exceedances": (
                    int(
                        saved[
                            "exact_sign_flip_assignments"
                        ]
                    )
                    == len(SIGN_MATRIX)
                    and int(
                        saved[
                            "exact_sign_flip_exceedances"
                        ]
                    )
                    == recomputed[
                        "exact_exceedances"
                    ]
                ),
                "raw_p": close_enough(
                    saved[
                        "exact_two_sided_p_raw"
                    ],
                    recomputed["exact_p"],
                ),
                "Holm_adjusted_p": close_enough(
                    saved[
                        "Holm_adjusted_p"
                    ],
                    adjusted,
                ),
                "Holm_reject": (
                    parse_bool(
                        saved[
                            "Holm_reject_alpha_0_05"
                        ]
                    )
                    == reject
                ),
                "omnibus_p": close_enough(
                    saved[
                        "omnibus_monte_carlo_p"
                    ],
                    permutation_p,
                ),
                "selection_not_used": (
                    parse_bool(
                        saved[
                            "final_model_selection_used"
                        ]
                    )
                    is False
                ),
            }

            failed_posthoc = [
                name
                for name, passed
                in posthoc_checks.items()
                if not passed
            ]

            if failed_posthoc:
                raise RuntimeError(
                    "Post-hoc verification failed for "
                    f"{architecture} {metric_name} "
                    f"{variant}: "
                    + ", ".join(
                        failed_posthoc
                    )
                )

            verified_posthoc_rows.append(
                {
                    "endpoint_class": (
                        specification[
                            "endpoint_class"
                        ]
                    ),
                    "architecture": architecture,
                    "metric": metric_name,
                    "reference_variant": "B0",
                    "candidate_variant": variant,
                    "mean_raw_delta_candidate_minus_B0": (
                        recomputed[
                            "mean_raw_delta"
                        ]
                    ),
                    "mean_improvement_delta": (
                        recomputed[
                            "mean_improvement_delta"
                        ]
                    ),
                    "paired_rank_biserial_improvement": (
                        recomputed[
                            "rank_biserial"
                        ]
                    ),
                    "exact_two_sided_p_raw": (
                        recomputed["exact_p"]
                    ),
                    "Holm_adjusted_p": adjusted,
                    "Holm_reject_alpha_0_05": (
                        reject
                    ),
                    "all_checks_passed": True,
                }
            )

        print(
            f"[{analysis_index}/16] "
            f"{architecture} | {metric_name} | "
            "descriptive=verified | "
            "omnibus=verified | "
            "posthoc=verified | "
            "all checks=True",
            flush=True,
        )

if (
    descriptive_checks_passed
    != EXPECTED_DESCRIPTIVE_ROWS
):
    raise RuntimeError(
        "Verified descriptive row count mismatch."
    )

if len(
    verified_omnibus_rows
) != EXPECTED_OMNIBUS_ROWS:
    raise RuntimeError(
        "Verified omnibus row count mismatch."
    )

if len(
    verified_posthoc_rows
) != EXPECTED_POSTHOC_ROWS:
    raise RuntimeError(
        "Verified post-hoc row count mismatch."
    )

atomic_csv(
    OUTPUT_VERIFIED_OMNIBUS,
    verified_omnibus_rows,
    [
        "endpoint_class",
        "architecture",
        "metric",
        "friedman_statistic",
        "kendalls_W",
        "monte_carlo_p_plus_one",
        "monte_carlo_exceedances",
        "scipy_asymptotic_p",
        "all_checks_passed",
    ],
)

atomic_csv(
    OUTPUT_VERIFIED_POSTHOC,
    verified_posthoc_rows,
    [
        "endpoint_class",
        "architecture",
        "metric",
        "reference_variant",
        "candidate_variant",
        "mean_raw_delta_candidate_minus_B0",
        "mean_improvement_delta",
        "paired_rank_biserial_improvement",
        "exact_two_sided_p_raw",
        "Holm_adjusted_p",
        "Holm_reject_alpha_0_05",
        "all_checks_passed",
    ],
)

recomputed_confirmatory_omnibus = sum(
    row["endpoint_class"]
    == "confirmatory"
    and float(
        row[
            "monte_carlo_p_plus_one"
        ]
    )
    <= ALPHA
    for row in verified_omnibus_rows
)

recomputed_exploratory_omnibus = sum(
    row["endpoint_class"]
    == "exploratory"
    and float(
        row[
            "monte_carlo_p_plus_one"
        ]
    )
    <= ALPHA
    for row in verified_omnibus_rows
)

recomputed_confirmatory_Holm = sum(
    row["endpoint_class"]
    == "confirmatory"
    and bool(
        row[
            "Holm_reject_alpha_0_05"
        ]
    )
    for row in verified_posthoc_rows
)

recomputed_exploratory_Holm = sum(
    row["endpoint_class"]
    == "exploratory"
    and bool(
        row[
            "Holm_reject_alpha_0_05"
        ]
    )
    for row in verified_posthoc_rows
)

minimum_raw_p = min(
    float(
        row[
            "exact_two_sided_p_raw"
        ]
    )
    for row in verified_posthoc_rows
)

minimum_Holm_p = min(
    float(
        row[
            "Holm_adjusted_p"
        ]
    )
    for row in verified_posthoc_rows
)

summary_checks = {
    "descriptive_count_matches": (
        int(
            summary[
                "descriptive_row_count"
            ]
        )
        == EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_count_matches": (
        int(
            summary[
                "omnibus_analysis_count"
            ]
        )
        == EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_count_matches": (
        int(
            summary[
                "posthoc_comparison_count"
            ]
        )
        == EXPECTED_POSTHOC_ROWS
    ),
    "confirmatory_omnibus_matches": (
        int(
            summary[
                "confirmatory_omnibus_rejections_nominal"
            ]
        )
        == recomputed_confirmatory_omnibus
    ),
    "exploratory_omnibus_matches": (
        int(
            summary[
                "exploratory_omnibus_rejections_nominal"
            ]
        )
        == recomputed_exploratory_omnibus
    ),
    "confirmatory_Holm_matches": (
        int(
            summary[
                "confirmatory_posthoc_Holm_rejections"
            ]
        )
        == recomputed_confirmatory_Holm
    ),
    "exploratory_Holm_matches": (
        int(
            summary[
                "exploratory_posthoc_Holm_rejections"
            ]
        )
        == recomputed_exploratory_Holm
    ),
    "minimum_raw_p_matches": close_enough(
        summary[
            "minimum_raw_posthoc_p"
        ],
        minimum_raw_p,
    ),
    "minimum_Holm_p_matches": close_enough(
        summary[
            "minimum_Holm_adjusted_p"
        ],
        minimum_Holm_p,
    ),
    "final_model_not_selected": (
        summary[
            "final_model_selected"
        ]
        is False
    ),
}

failed_summary_checks = [
    name
    for name, passed
    in summary_checks.items()
    if not passed
]

if failed_summary_checks:
    raise RuntimeError(
        "Phase 7 summary verification failed: "
        + ", ".join(
            failed_summary_checks
        )
    )

global_checks = {
    "descriptive_rows_verified_176": (
        descriptive_checks_passed
        == EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_rows_verified_16": (
        len(verified_omnibus_rows)
        == EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_rows_verified_160": (
        len(verified_posthoc_rows)
        == EXPECTED_POSTHOC_ROWS
    ),
    "all_omnibus_checks_passed": all(
        row[
            "all_checks_passed"
        ]
        for row in verified_omnibus_rows
    ),
    "all_posthoc_checks_passed": all(
        row[
            "all_checks_passed"
        ]
        for row in verified_posthoc_rows
    ),
    "minimum_raw_exact_p_0_0625": close_enough(
        minimum_raw_p,
        0.0625,
    ),
    "minimum_Holm_p_0_625": close_enough(
        minimum_Holm_p,
        0.625,
    ),
    "no_posthoc_Holm_rejections": (
        recomputed_confirmatory_Holm
        + recomputed_exploratory_Holm
        == 0
    ),
    "model_inference_not_repeated": True,
    "final_model_not_selected": True,
}

failed_global_checks = [
    name
    for name, passed
    in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "Phase 7 verification global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 7,
    "artifact_name": (
        "statistical_analysis_independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "verification_policy": {
        "descriptive_statistics_recomputed": True,
        "monte_carlo_omnibus_recomputed": True,
        "exact_sign_flip_recomputed": True,
        "exhaustive_bootstrap_recomputed": True,
        "Holm_adjustment_recomputed": True,
        "model_inference_repeated": False,
    },
    "counts": {
        "descriptive_rows": (
            EXPECTED_DESCRIPTIVE_ROWS
        ),
        "omnibus_analyses": (
            EXPECTED_OMNIBUS_ROWS
        ),
        "posthoc_comparisons": (
            EXPECTED_POSTHOC_ROWS
        ),
    },
    "verified_findings": {
        "confirmatory_omnibus_rejections_nominal": (
            recomputed_confirmatory_omnibus
        ),
        "exploratory_omnibus_rejections_nominal": (
            recomputed_exploratory_omnibus
        ),
        "confirmatory_posthoc_Holm_rejections": (
            recomputed_confirmatory_Holm
        ),
        "exploratory_posthoc_Holm_rejections": (
            recomputed_exploratory_Holm
        ),
        "minimum_raw_exact_p": (
            minimum_raw_p
        ),
        "minimum_Holm_adjusted_p": (
            minimum_Holm_p
        ),
    },
    "entry_checks": entry_checks,
    "summary_checks": summary_checks,
    "global_checks": global_checks,
    "verified_omnibus_csv": file_record(
        OUTPUT_VERIFIED_OMNIBUS
    ),
    "verified_posthoc_csv": file_record(
        OUTPUT_VERIFIED_POSTHOC
    ),
    "source_artifacts": {
        "protocol": file_record(
            PROTOCOL_PATH
        ),
        "preflight_lock": file_record(
            PREFLIGHT_LOCK_PATH
        ),
        "joined_matrix": file_record(
            JOINED_MATRIX_PATH
        ),
        "comparison_plan": file_record(
            COMPARISON_PLAN_PATH
        ),
        "descriptive_summary": file_record(
            DESCRIPTIVE_PATH
        ),
        "omnibus_results": file_record(
            OMNIBUS_PATH
        ),
        "posthoc_results": file_record(
            POSTHOC_PATH
        ),
        "summary": file_record(
            SUMMARY_PATH
        ),
        "completion": file_record(
            COMPLETION_PATH
        ),
    },
    "ready_for_multi_objective_decision_preflight": (
        True
    ),
    "final_model_selected": False,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 7,
    "artifact_name": (
        "statistical_analysis"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "descriptive_row_count": (
        EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_analysis_count": (
        EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_comparison_count": (
        EXPECTED_POSTHOC_ROWS
    ),
    "verified_omnibus_csv": str(
        OUTPUT_VERIFIED_OMNIBUS
    ),
    "verified_omnibus_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_OMNIBUS
        )
    ),
    "verified_posthoc_csv": str(
        OUTPUT_VERIFIED_POSTHOC
    ),
    "verified_posthoc_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_POSTHOC
        )
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "descriptive_summary_sha256": (
        sha256_file(
            DESCRIPTIVE_PATH
        )
    ),
    "omnibus_results_sha256": (
        sha256_file(
            OMNIBUS_PATH
        )
    ),
    "posthoc_results_sha256": (
        sha256_file(
            POSTHOC_PATH
        )
    ),
    "model_inference_repeated_by_verifier": (
        False
    ),
    "final_model_selected": False,
    "ready_for_multi_objective_decision_preflight": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 7,
    "artifact_name": (
        "statistical_analysis"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PROTOCOL_PATH
        ),
        file_record(
            PREFLIGHT_LOCK_PATH
        ),
        file_record(
            JOINED_MATRIX_PATH
        ),
        file_record(
            COMPARISON_PLAN_PATH
        ),
        file_record(
            DESCRIPTIVE_PATH
        ),
        file_record(
            OMNIBUS_PATH
        ),
        file_record(
            POSTHOC_PATH
        ),
        file_record(
            SUMMARY_PATH
        ),
        file_record(
            COMPLETION_PATH
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFIED_OMNIBUS
        ),
        file_record(
            OUTPUT_VERIFIED_POSTHOC
        ),
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "descriptive_row_count": (
        EXPECTED_DESCRIPTIVE_ROWS
    ),
    "omnibus_analysis_count": (
        EXPECTED_OMNIBUS_ROWS
    ),
    "posthoc_comparison_count": (
        EXPECTED_POSTHOC_ROWS
    ),
    "model_inference_repeated_by_verifier": (
        False
    ),
    "final_model_selected": False,
    "ready_for_multi_objective_decision_preflight": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 7 STATISTICAL ANALYSIS VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Descriptive rows verified       : 176"
)
print(
    "Omnibus analyses verified       : 16"
)
print(
    "Post-hoc comparisons verified   : 160"
)
print(
    "Monte Carlo analyses recomputed : PASSED"
)
print(
    "Exact sign-flip recomputed      : PASSED"
)
print(
    "Exact bootstrap recomputed      : PASSED"
)
print(
    "Holm correction recomputed      : PASSED"
)
print(
    "Confirmatory omnibus rejections : "
    f"{recomputed_confirmatory_omnibus}"
)
print(
    "Exploratory omnibus rejections  : "
    f"{recomputed_exploratory_omnibus}"
)
print(
    "Confirmatory Holm rejections    : "
    f"{recomputed_confirmatory_Holm}"
)
print(
    "Exploratory Holm rejections     : "
    f"{recomputed_exploratory_Holm}"
)
print(
    "Minimum raw exact p             : "
    f"{minimum_raw_p:.6f}"
)
print(
    "Minimum Holm-adjusted p         : "
    f"{minimum_Holm_p:.6f}"
)
print(
    "Model inference repeated        : False"
)
print(
    "Final model selected            : False"
)
print(
    "Phase 7 status                  : LOCKED"
)
print(
    "Ready for decision preflight    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 7 STATISTICAL ANALYSIS VERIFIED AND LOCKED"
)
