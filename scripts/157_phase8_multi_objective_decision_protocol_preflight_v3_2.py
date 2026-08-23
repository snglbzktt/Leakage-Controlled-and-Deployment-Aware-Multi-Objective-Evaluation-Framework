from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_LOCK = (
    AUDIT
    / "phase5_compression_locked_v3_2.json"
)

PHASE5_GROUP_SUMMARY = (
    AUDIT
    / "phase5_compression_group_summary_v3_2.csv"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE6_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_locked_v3_2.json"
)

PHASE6_GROUP_SUMMARY = (
    AUDIT
    / "phase6_deployment_benchmark_group_summary_v3_2.csv"
)

PHASE6_ALL_RUNS = (
    AUDIT
    / "phase6_deployment_benchmark_all_runs_v3_2.csv"
)

PHASE7_LOCK = (
    AUDIT
    / "phase7_statistical_analysis_locked_v3_2.json"
)

PHASE7_POSTHOC = (
    AUDIT
    / "phase7_statistical_posthoc_vs_B0_results_v3_2.csv"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase8_multi_objective_decision_protocol_v3_2.json"
)

OUTPUT_DECISION_MATRIX = (
    AUDIT
    / "phase8_multi_objective_decision_input_matrix_v3_2.csv"
)

OUTPUT_CRITERIA = (
    AUDIT
    / "phase8_multi_objective_decision_criteria_v3_2.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase8_multi_objective_decision_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase8_multi_objective_decision_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase8_multi_objective_decision_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase8_multi_objective_decision_v3_2"
PHASE5_PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
PHASE6_PROTOCOL_VERSION = "phase6_deployment_benchmark_v3_2"
PHASE7_PROTOCOL_VERSION = "phase7_statistical_analysis_v3_2"

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

COMPRESSED_CANDIDATE_VARIANTS = (
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

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_GROUP_KEYS = {
    (
        architecture,
        variant,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
}

EXPECTED_RUN_KEYS = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

PRIMARY_PROFILE = "balanced"

GATE_PROFILES = {
    "strict": {
        "maximum_mean_macro_f1_loss": 0.0010,
        "maximum_worst_seed_macro_f1_loss": 0.0025,
        "maximum_mean_raw_weighted_macro_f1_loss": 0.0020,
    },
    "balanced": {
        "maximum_mean_macro_f1_loss": 0.0020,
        "maximum_worst_seed_macro_f1_loss": 0.0050,
        "maximum_mean_raw_weighted_macro_f1_loss": 0.0030,
    },
    "relaxed": {
        "maximum_mean_macro_f1_loss": 0.0050,
        "maximum_worst_seed_macro_f1_loss": 0.0100,
        "maximum_mean_raw_weighted_macro_f1_loss": 0.0070,
    },
}

PARETO_OBJECTIVES = (
    {
        "metric": "mean_test_fingerprint_macro_f1",
        "direction": "maximize",
        "role": "predictive_quality",
    },
    {
        "metric": "mean_serialized_state_dict_bytes",
        "direction": "minimize",
        "role": "storage",
    },
    {
        "metric": "mean_batch_1_median_ms",
        "direction": "minimize",
        "role": "single_sample_latency",
    },
    {
        "metric": "mean_batch_32_samples_per_second",
        "direction": "maximize",
        "role": "throughput",
    },
)

PRIMARY_SELECTION_ORDER = (
    {
        "criterion": "eligible_under_balanced_profile",
        "direction": "required_true",
    },
    {
        "criterion": "Pareto_nondominated",
        "direction": "required_true",
    },
    {
        "criterion": "mean_serialized_state_dict_bytes",
        "direction": "ascending",
    },
    {
        "criterion": "mean_batch_1_median_ms",
        "direction": "ascending",
    },
    {
        "criterion": "mean_batch_32_samples_per_second",
        "direction": "descending",
    },
    {
        "criterion": "mean_test_fingerprint_macro_f1",
        "direction": "descending",
    },
    {
        "criterion": "architecture",
        "direction": "lexicographic_ascending",
    },
    {
        "criterion": "variant",
        "direction": "lexicographic_ascending",
    },
)

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


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(
        value
    ).strip().lower()

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


required_paths = (
    PHASE5_LOCK,
    PHASE5_GROUP_SUMMARY,
    PHASE5_MASTER_MATRIX,
    PHASE6_LOCK,
    PHASE6_GROUP_SUMMARY,
    PHASE6_ALL_RUNS,
    PHASE7_LOCK,
    PHASE7_POSTHOC,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_DECISION_MATRIX,
    OUTPUT_CRITERIA,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 8 decision-preflight artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

phase5_lock = read_json(
    PHASE5_LOCK
)

phase6_lock = read_json(
    PHASE6_LOCK
)

phase7_lock = read_json(
    PHASE7_LOCK
)

phase5_groups = read_csv(
    PHASE5_GROUP_SUMMARY
)

phase5_runs = read_csv(
    PHASE5_MASTER_MATRIX
)

phase6_groups = read_csv(
    PHASE6_GROUP_SUMMARY
)

phase6_runs = read_csv(
    PHASE6_ALL_RUNS
)

posthoc_rows = read_csv(
    PHASE7_POSTHOC
)

entry_checks = {
    "phase5_locked": (
        phase5_lock.get("status")
        == "locked"
        and phase5_lock.get(
            "protocol_version"
        )
        == PHASE5_PROTOCOL_VERSION
        and phase5_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase6_locked": (
        phase6_lock.get("status")
        == "locked"
        and phase6_lock.get(
            "protocol_version"
        )
        == PHASE6_PROTOCOL_VERSION
        and phase6_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase7_locked": (
        phase7_lock.get("status")
        == "locked"
        and phase7_lock.get(
            "protocol_version"
        )
        == PHASE7_PROTOCOL_VERSION
        and phase7_lock.get(
            "all_checks_passed"
        )
        is True
        and phase7_lock.get(
            "ready_for_multi_objective_decision_preflight"
        )
        is True
    ),
    "phase5_group_hash_matches": (
        phase5_lock.get(
            "group_summary_sha256"
        )
        == sha256_file(
            PHASE5_GROUP_SUMMARY
        )
    ),
    "phase5_master_hash_matches": (
        phase5_lock.get(
            "master_run_matrix_sha256"
        )
        == sha256_file(
            PHASE5_MASTER_MATRIX
        )
    ),
    "phase6_group_hash_matches": (
        phase6_lock.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            PHASE6_GROUP_SUMMARY
        )
    ),
    "phase6_runs_hash_matches": (
        phase6_lock.get(
            "all_runs_csv_sha256"
        )
        == sha256_file(
            PHASE6_ALL_RUNS
        )
    ),
    "phase7_posthoc_hash_matches": (
        phase7_lock.get(
            "posthoc_results_sha256"
        )
        == sha256_file(
            PHASE7_POSTHOC
        )
    ),
    "phase5_final_model_not_selected": (
        phase5_lock.get(
            "final_model_selected"
        )
        is False
    ),
    "phase6_final_model_not_selected": (
        phase6_lock.get(
            "final_model_selected"
        )
        is False
    ),
    "phase7_final_model_not_selected": (
        phase7_lock.get(
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
        "Phase 8 decision-preflight entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

phase5_group_lookup = {
    (
        row["architecture"],
        row["variant"],
    ): row
    for row in phase5_groups
}

phase6_group_lookup = {
    (
        row["architecture"],
        row["variant"],
    ): row
    for row in phase6_groups
}

phase5_run_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in phase5_runs
}

phase6_run_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in phase6_runs
}

if set(
    phase5_group_lookup.keys()
) != EXPECTED_GROUP_KEYS:
    raise RuntimeError(
        "Phase 5 group matrix is incomplete."
    )

if set(
    phase6_group_lookup.keys()
) != EXPECTED_GROUP_KEYS:
    raise RuntimeError(
        "Phase 6 group matrix is incomplete."
    )

if set(
    phase5_run_lookup.keys()
) != EXPECTED_RUN_KEYS:
    raise RuntimeError(
        "Phase 5 run matrix is incomplete."
    )

if set(
    phase6_run_lookup.keys()
) != EXPECTED_RUN_KEYS:
    raise RuntimeError(
        "Phase 6 run matrix is incomplete."
    )

posthoc_lookup = {
    (
        row["architecture"],
        row["metric"],
        row["candidate_variant"],
    ): row
    for row in posthoc_rows
}

decision_rows: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    baseline_group = (
        phase5_group_lookup[
            (
                architecture,
                "B0",
            )
        ]
    )

    baseline_deployment = (
        phase6_group_lookup[
            (
                architecture,
                "B0",
            )
        ]
    )

    baseline_macro_f1 = finite_float(
        baseline_group[
            "mean_test_fingerprint_macro_f1"
        ],
        "baseline_macro_f1",
    )

    baseline_raw_weighted = finite_float(
        baseline_group[
            "mean_test_raw_weighted_macro_f1"
        ],
        "baseline_raw_weighted",
    )

    baseline_state_bytes = finite_float(
        baseline_deployment[
            "mean_serialized_state_dict_bytes"
        ],
        "baseline_state_bytes",
    )

    baseline_latency = finite_float(
        baseline_deployment[
            "mean_batch_1_median_ms"
        ],
        "baseline_latency",
    )

    baseline_throughput = finite_float(
        baseline_deployment[
            "mean_batch_32_samples_per_second"
        ],
        "baseline_throughput",
    )

    for variant in VARIANTS:
        predictive = phase5_group_lookup[
            (
                architecture,
                variant,
            )
        ]

        deployment = phase6_group_lookup[
            (
                architecture,
                variant,
            )
        ]

        mean_macro_f1 = finite_float(
            predictive[
                "mean_test_fingerprint_macro_f1"
            ],
            "mean_macro_f1",
        )

        mean_raw_weighted = finite_float(
            predictive[
                "mean_test_raw_weighted_macro_f1"
            ],
            "mean_raw_weighted",
        )

        state_bytes = finite_float(
            deployment[
                "mean_serialized_state_dict_bytes"
            ],
            "state_bytes",
        )

        latency = finite_float(
            deployment[
                "mean_batch_1_median_ms"
            ],
            "latency",
        )

        throughput = finite_float(
            deployment[
                "mean_batch_32_samples_per_second"
            ],
            "throughput",
        )

        seed_losses: list[float] = []

        for seed in SEEDS:
            baseline_seed = finite_float(
                phase5_run_lookup[
                    (
                        architecture,
                        "B0",
                        seed,
                    )
                ][
                    "test_fingerprint_macro_f1"
                ],
                "baseline_seed_macro_f1",
            )

            candidate_seed = finite_float(
                phase5_run_lookup[
                    (
                        architecture,
                        variant,
                        seed,
                    )
                ][
                    "test_fingerprint_macro_f1"
                ],
                "candidate_seed_macro_f1",
            )

            seed_losses.append(
                baseline_seed
                - candidate_seed
            )

        if variant == "B0":
            macro_f1_effect = 0.0
            macro_f1_raw_p = 1.0
            macro_f1_Holm_p = 1.0
            macro_f1_rank_biserial = 0.0
        else:
            statistical = posthoc_lookup[
                (
                    architecture,
                    "test_fingerprint_macro_f1",
                    variant,
                )
            ]

            macro_f1_effect = finite_float(
                statistical[
                    "mean_improvement_delta"
                ],
                "macro_f1_effect",
            )

            macro_f1_raw_p = finite_float(
                statistical[
                    "exact_two_sided_p_raw"
                ],
                "macro_f1_raw_p",
            )

            macro_f1_Holm_p = finite_float(
                statistical[
                    "Holm_adjusted_p"
                ],
                "macro_f1_Holm_p",
            )

            macro_f1_rank_biserial = finite_float(
                statistical[
                    "paired_rank_biserial_improvement"
                ],
                "macro_f1_rank_biserial",
            )

        decision_rows.append(
            {
                "architecture": architecture,
                "variant": variant,
                "candidate_role": (
                    "baseline_reference"
                    if variant == "B0"
                    else (
                        "uncompressed_control"
                        if variant == "FP32-FT"
                        else "compressed_candidate"
                    )
                ),
                "representation": (
                    predictive[
                        "representation"
                    ]
                ),
                "mean_test_fingerprint_macro_f1": (
                    mean_macro_f1
                ),
                "std_test_fingerprint_macro_f1": finite_float(
                    predictive[
                        "std_test_fingerprint_macro_f1"
                    ],
                    "std_macro_f1",
                ),
                "mean_test_raw_weighted_macro_f1": (
                    mean_raw_weighted
                ),
                "mean_macro_f1_loss_vs_architecture_B0": (
                    baseline_macro_f1
                    - mean_macro_f1
                ),
                "worst_seed_macro_f1_loss_vs_architecture_B0": (
                    max(seed_losses)
                ),
                "mean_raw_weighted_macro_f1_loss_vs_architecture_B0": (
                    baseline_raw_weighted
                    - mean_raw_weighted
                ),
                "fragility_triggered_in_any_run": parse_bool(
                    predictive[
                        "fragility_triggered_in_any_run"
                    ]
                ),
                "mean_serialized_state_dict_bytes": (
                    state_bytes
                ),
                "state_size_ratio_vs_architecture_B0": (
                    state_bytes
                    / baseline_state_bytes
                ),
                "state_size_reduction_fraction_vs_architecture_B0": (
                    1.0
                    - (
                        state_bytes
                        / baseline_state_bytes
                    )
                ),
                "mean_batch_1_median_ms": (
                    latency
                ),
                "batch_1_latency_ratio_vs_architecture_B0": (
                    latency
                    / baseline_latency
                ),
                "batch_1_latency_reduction_fraction_vs_architecture_B0": (
                    1.0
                    - (
                        latency
                        / baseline_latency
                    )
                ),
                "mean_batch_32_samples_per_second": (
                    throughput
                ),
                "batch_32_throughput_ratio_vs_architecture_B0": (
                    throughput
                    / baseline_throughput
                ),
                "batch_32_throughput_gain_fraction_vs_architecture_B0": (
                    (
                        throughput
                        / baseline_throughput
                    )
                    - 1.0
                ),
                "mean_peak_inference_RSS_delta_bytes": finite_float(
                    deployment[
                        "mean_peak_inference_RSS_delta_bytes"
                    ],
                    "mean_peak_inference_RSS_delta_bytes",
                ),
                "macro_f1_mean_improvement_effect_vs_B0": (
                    macro_f1_effect
                ),
                "macro_f1_exact_two_sided_p_raw": (
                    macro_f1_raw_p
                ),
                "macro_f1_Holm_adjusted_p": (
                    macro_f1_Holm_p
                ),
                "macro_f1_rank_biserial_improvement": (
                    macro_f1_rank_biserial
                ),
                "decision_computation_performed": (
                    False
                ),
            }
        )

if len(
    decision_rows
) != 22:
    raise RuntimeError(
        "Phase 8 decision input row count mismatch."
    )

atomic_csv(
    OUTPUT_DECISION_MATRIX,
    decision_rows,
    [
        "architecture",
        "variant",
        "candidate_role",
        "representation",
        "mean_test_fingerprint_macro_f1",
        "std_test_fingerprint_macro_f1",
        "mean_test_raw_weighted_macro_f1",
        "mean_macro_f1_loss_vs_architecture_B0",
        "worst_seed_macro_f1_loss_vs_architecture_B0",
        "mean_raw_weighted_macro_f1_loss_vs_architecture_B0",
        "fragility_triggered_in_any_run",
        "mean_serialized_state_dict_bytes",
        "state_size_ratio_vs_architecture_B0",
        "state_size_reduction_fraction_vs_architecture_B0",
        "mean_batch_1_median_ms",
        "batch_1_latency_ratio_vs_architecture_B0",
        "batch_1_latency_reduction_fraction_vs_architecture_B0",
        "mean_batch_32_samples_per_second",
        "batch_32_throughput_ratio_vs_architecture_B0",
        "batch_32_throughput_gain_fraction_vs_architecture_B0",
        "mean_peak_inference_RSS_delta_bytes",
        "macro_f1_mean_improvement_effect_vs_B0",
        "macro_f1_exact_two_sided_p_raw",
        "macro_f1_Holm_adjusted_p",
        "macro_f1_rank_biserial_improvement",
        "decision_computation_performed",
    ],
)

criteria_rows: list[
    dict[str, Any]
] = []

for profile_name, thresholds in (
    GATE_PROFILES.items()
):
    criteria_rows.extend(
        [
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "candidate_role"
                ),
                "operator": "equals",
                "threshold": (
                    "compressed_candidate"
                ),
                "direction": (
                    "required"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "fragility_triggered_in_any_run"
                ),
                "operator": "equals",
                "threshold": False,
                "direction": (
                    "required"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "mean_macro_f1_loss_vs_architecture_B0"
                ),
                "operator": (
                    "less_than_or_equal"
                ),
                "threshold": thresholds[
                    "maximum_mean_macro_f1_loss"
                ],
                "direction": (
                    "lower_is_better"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "worst_seed_macro_f1_loss_vs_architecture_B0"
                ),
                "operator": (
                    "less_than_or_equal"
                ),
                "threshold": thresholds[
                    "maximum_worst_seed_macro_f1_loss"
                ],
                "direction": (
                    "lower_is_better"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "mean_raw_weighted_macro_f1_loss_vs_architecture_B0"
                ),
                "operator": (
                    "less_than_or_equal"
                ),
                "threshold": thresholds[
                    "maximum_mean_raw_weighted_macro_f1_loss"
                ],
                "direction": (
                    "lower_is_better"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
            {
                "stage": "eligibility",
                "profile": profile_name,
                "criterion": (
                    "deployment_improvement_count"
                ),
                "operator": (
                    "greater_than_or_equal"
                ),
                "threshold": 1,
                "direction": (
                    "required"
                ),
                "primary_profile": (
                    profile_name
                    == PRIMARY_PROFILE
                ),
            },
        ]
    )

for objective in PARETO_OBJECTIVES:
    criteria_rows.append(
        {
            "stage": "Pareto",
            "profile": PRIMARY_PROFILE,
            "criterion": objective[
                "metric"
            ],
            "operator": objective[
                "direction"
            ],
            "threshold": "",
            "direction": objective[
                "role"
            ],
            "primary_profile": True,
        }
    )

for order_index, rule in enumerate(
    PRIMARY_SELECTION_ORDER,
    start=1,
):
    criteria_rows.append(
        {
            "stage": (
                "deterministic_selection"
            ),
            "profile": PRIMARY_PROFILE,
            "criterion": rule[
                "criterion"
            ],
            "operator": rule[
                "direction"
            ],
            "threshold": (
                order_index
            ),
            "direction": (
                "selection_order"
            ),
            "primary_profile": True,
        }
    )

atomic_csv(
    OUTPUT_CRITERIA,
    criteria_rows,
    [
        "stage",
        "profile",
        "criterion",
        "operator",
        "threshold",
        "direction",
        "primary_profile",
    ],
)

protocol = {
    "status": "locked",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "provenance_statement": {
        "externally_preregistered": False,
        "protocol_locked_after_phases_5_6_7": (
            True
        ),
        "decision_computation_performed_before_lock": (
            False
        ),
        "interpretation": (
            "This is a transparent post-experiment "
            "engineering decision rule, not an external "
            "preregistration."
        ),
    },
    "decision_scope": {
        "architecture_count": 2,
        "variant_count": 11,
        "group_count": 22,
        "compressed_candidate_variants": list(
            COMPRESSED_CANDIDATE_VARIANTS
        ),
        "baseline_reference": "B0",
        "uncompressed_control": "FP32-FT",
        "seed_count": 5,
    },
    "eligibility_profiles": (
        GATE_PROFILES
    ),
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "common_hard_gates": {
        "candidate_role": (
            "compressed_candidate"
        ),
        "fragility_triggered_in_any_run": (
            False
        ),
        "minimum_deployment_improvement_count": (
            1
        ),
        "deployment_improvement_metrics": [
            (
                "state_size_reduction_fraction_"
                "vs_architecture_B0"
            ),
            (
                "batch_1_latency_reduction_fraction_"
                "vs_architecture_B0"
            ),
            (
                "batch_32_throughput_gain_fraction_"
                "vs_architecture_B0"
            ),
        ],
        "strict_positive_improvement_required": (
            True
        ),
    },
    "Pareto_analysis": {
        "performed_only_after_primary_eligibility": (
            True
        ),
        "objectives": list(
            PARETO_OBJECTIVES
        ),
        "dominance_rule": (
            "Candidate A dominates B when A is no worse "
            "on all four objectives and strictly better "
            "on at least one objective."
        ),
        "cross_architecture_comparison": (
            True
        ),
    },
    "primary_selection_rule": {
        "weighted_score_used": False,
        "selection_order": list(
            PRIMARY_SELECTION_ORDER
        ),
        "interpretation": (
            "Predictive quality and fragility are enforced "
            "as hard gates. Among eligible Pareto candidates, "
            "the smallest serialized model is preferred; "
            "latency, throughput, predictive quality, and "
            "stable lexical tie-breaks follow."
        ),
    },
    "sensitivity_analysis": {
        "profiles_reported": [
            "strict",
            "balanced",
            "relaxed",
        ],
        "primary_profile": (
            PRIMARY_PROFILE
        ),
        "selection_stability_reported": (
            True
        ),
        "primary_selection_not_changed_by_"
        "exploratory_profile": (
            True
        ),
    },
    "statistical_interpretation": {
        "Holm_adjusted_pairwise_rejections": (
            0
        ),
        "minimum_raw_exact_p": 0.0625,
        "minimum_Holm_adjusted_p": 0.625,
        "absence_of_significance_is_not_"
        "equivalence": (
            True
        ),
        "p_values_used_as_exclusion_gate": (
            False
        ),
        "effect_sizes_reported_descriptively": (
            True
        ),
    },
    "reporting_rules": {
        "report_all_22_groups": True,
        "report_all_gate_failures": True,
        "report_all_Pareto_candidates": True,
        "report_profile_sensitivity": True,
        "report_final_selection_trace": True,
        "final_selection_requires_independent_"
        "verification": (
            True
        ),
        "no_model_inference_in_phase8": True,
    },
    "source_artifacts": {
        "phase5_lock": file_record(
            PHASE5_LOCK
        ),
        "phase5_group_summary": file_record(
            PHASE5_GROUP_SUMMARY
        ),
        "phase5_master_matrix": file_record(
            PHASE5_MASTER_MATRIX
        ),
        "phase6_lock": file_record(
            PHASE6_LOCK
        ),
        "phase6_group_summary": file_record(
            PHASE6_GROUP_SUMMARY
        ),
        "phase6_all_runs": file_record(
            PHASE6_ALL_RUNS
        ),
        "phase7_lock": file_record(
            PHASE7_LOCK
        ),
        "phase7_posthoc": file_record(
            PHASE7_POSTHOC
        ),
    },
    "environment": {
        "python_version": (
            sys.version
        ),
        "platform": (
            platform.platform()
        ),
    },
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

preflight_checks = {
    "decision_input_rows_22": (
        len(decision_rows) == 22
    ),
    "decision_group_keys_exact": (
        {
            (
                row["architecture"],
                row["variant"],
            )
            for row in decision_rows
        }
        == EXPECTED_GROUP_KEYS
    ),
    "compressed_candidate_groups_18": (
        sum(
            row["candidate_role"]
            == "compressed_candidate"
            for row in decision_rows
        )
        == 18
    ),
    "baseline_groups_2": (
        sum(
            row["candidate_role"]
            == "baseline_reference"
            for row in decision_rows
        )
        == 2
    ),
    "uncompressed_control_groups_2": (
        sum(
            row["candidate_role"]
            == "uncompressed_control"
            for row in decision_rows
        )
        == 2
    ),
    "all_decision_values_finite": all(
        math.isfinite(
            float(
                row[field_name]
            )
        )
        for row in decision_rows
        for field_name in (
            "mean_test_fingerprint_macro_f1",
            "mean_test_raw_weighted_macro_f1",
            "mean_macro_f1_loss_vs_architecture_B0",
            "worst_seed_macro_f1_loss_vs_architecture_B0",
            "mean_raw_weighted_macro_f1_loss_vs_architecture_B0",
            "mean_serialized_state_dict_bytes",
            "mean_batch_1_median_ms",
            "mean_batch_32_samples_per_second",
        )
    ),
    "primary_profile_balanced": (
        PRIMARY_PROFILE == "balanced"
    ),
    "gate_profile_count_3": (
        len(GATE_PROFILES) == 3
    ),
    "Pareto_objective_count_4": (
        len(PARETO_OBJECTIVES) == 4
    ),
    "weighted_score_disabled": (
        protocol[
            "primary_selection_rule"
        ][
            "weighted_score_used"
        ]
        is False
    ),
    "decision_computation_not_performed": all(
        row[
            "decision_computation_performed"
        ]
        is False
        for row in decision_rows
    ),
    "model_inference_not_performed": (
        True
    ),
    "final_model_not_selected": (
        True
    ),
}

failed_preflight_checks = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight_checks:
    raise RuntimeError(
        "Phase 8 decision preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "protocol": file_record(
        OUTPUT_PROTOCOL
    ),
    "decision_input_matrix": file_record(
        OUTPUT_DECISION_MATRIX
    ),
    "criteria": file_record(
        OUTPUT_CRITERIA
    ),
    "decision_computation_performed": (
        False
    ),
    "model_inference_performed": (
        False
    ),
    "final_model_selected": False,
    "ready_for_locked_decision_execution": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "group_count": 22,
    "compressed_candidate_group_count": 18,
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "decision_input_matrix": str(
        OUTPUT_DECISION_MATRIX
    ),
    "decision_input_matrix_sha256": (
        sha256_file(
            OUTPUT_DECISION_MATRIX
        )
    ),
    "criteria": str(
        OUTPUT_CRITERIA
    ),
    "criteria_sha256": (
        sha256_file(
            OUTPUT_CRITERIA
        )
    ),
    "preflight_report": str(
        OUTPUT_PREFLIGHT
    ),
    "preflight_report_sha256": (
        sha256_file(
            OUTPUT_PREFLIGHT
        )
    ),
    "decision_computation_performed": (
        False
    ),
    "model_inference_performed": False,
    "final_model_selected": False,
    "ready_for_locked_decision_execution": (
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
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(PHASE5_LOCK),
        file_record(PHASE5_GROUP_SUMMARY),
        file_record(PHASE5_MASTER_MATRIX),
        file_record(PHASE6_LOCK),
        file_record(PHASE6_GROUP_SUMMARY),
        file_record(PHASE6_ALL_RUNS),
        file_record(PHASE7_LOCK),
        file_record(PHASE7_POSTHOC),
    ],
    "generated_artifacts": [
        file_record(OUTPUT_PROTOCOL),
        file_record(OUTPUT_DECISION_MATRIX),
        file_record(OUTPUT_CRITERIA),
        file_record(OUTPUT_PREFLIGHT),
        file_record(OUTPUT_LOCK),
    ],
    "group_count": 22,
    "compressed_candidate_group_count": 18,
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "decision_computation_performed": (
        False
    ),
    "model_inference_performed": False,
    "final_model_selected": False,
    "ready_for_locked_decision_execution": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 8 MULTI-OBJECTIVE DECISION PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Upstream Phase 5 status         : LOCKED"
)
print(
    "Upstream Phase 6 status         : LOCKED"
)
print(
    "Upstream Phase 7 status         : LOCKED"
)
print(
    "Decision input groups           : 22"
)
print(
    "Compressed candidate groups     : 18"
)
print(
    "Baseline references             : 2"
)
print(
    "Uncompressed controls           : 2"
)
print(
    "Primary eligibility profile     : balanced"
)
print(
    "Eligibility sensitivity profiles: strict / balanced / relaxed"
)
print(
    "Pareto objectives               : 4"
)
print(
    "Weighted score used             : False"
)
print(
    "Pairwise p-value selection gate : False"
)
print(
    "Decision computation run        : False"
)
print(
    "Model inference run             : False"
)
print(
    "Final model selected            : False"
)
print(
    "Protocol                        : "
    f"{OUTPUT_PROTOCOL}"
)
print(
    "Decision matrix                 : "
    f"{OUTPUT_DECISION_MATRIX}"
)
print(
    "Criteria                        : "
    f"{OUTPUT_CRITERIA}"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for decision execution    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 8 MULTI-OBJECTIVE DECISION PROTOCOL LOCKED"
)
