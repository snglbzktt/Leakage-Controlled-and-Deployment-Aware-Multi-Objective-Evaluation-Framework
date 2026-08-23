from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase8_multi_objective_decision_protocol_v3_2.json"
)

PREFLIGHT_LOCK_PATH = (
    AUDIT
    / "phase8_multi_objective_decision_preflight_locked_v3_2.json"
)

DECISION_MATRIX_PATH = (
    AUDIT
    / "phase8_multi_objective_decision_input_matrix_v3_2.csv"
)

CRITERIA_PATH = (
    AUDIT
    / "phase8_multi_objective_decision_criteria_v3_2.csv"
)

ELIGIBILITY_PATH = (
    AUDIT
    / "phase8_multi_objective_eligibility_results_v3_2.csv"
)

PARETO_PATH = (
    AUDIT
    / "phase8_multi_objective_pareto_results_v3_2.csv"
)

SELECTION_TRACE_PATH = (
    AUDIT
    / "phase8_multi_objective_selection_trace_v3_2.csv"
)

SUMMARY_PATH = (
    AUDIT
    / "phase8_multi_objective_decision_summary_v3_2.json"
)

COMPLETION_PATH = (
    AUDIT
    / "phase8_multi_objective_decision_completed_v3_2.json"
)

OUTPUT_VERIFIED_ELIGIBILITY = (
    AUDIT
    / "phase8_multi_objective_eligibility_verified_v3_2.csv"
)

OUTPUT_VERIFIED_PARETO = (
    AUDIT
    / "phase8_multi_objective_pareto_verified_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase8_multi_objective_decision_verification_v3_2.json"
)

OUTPUT_FINAL_LOCK = (
    AUDIT
    / "phase8_final_model_family_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase8_final_model_family_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase8_multi_objective_decision_v3_2"

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

PROFILES = (
    "strict",
    "balanced",
    "relaxed",
)

PRIMARY_PROFILE = "balanced"
EXPECTED_GROUP_COUNT = 22
EXPECTED_COMPRESSED_COUNT = 18
EXPECTED_ELIGIBILITY_ROWS = 66
EXPECTED_PARETO_ROWS = 54
EXPECTED_SELECTION_ROWS = 3

FLOAT_TOLERANCE = 1e-12
STRICT_POSITIVE_TOLERANCE = 1e-15

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
        "Windows kept the destination file "
        "locked after "
        f"{WINDOWS_FILE_RETRY_COUNT} attempts: "
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
        f"Cannot parse boolean: {value}"
    )


def finite_float(
    value: Any,
    name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise RuntimeError(
            f"Non-finite {name}: {value}"
        )

    return number


def close_enough(
    first: Any,
    second: Any,
) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )


def candidate_id(
    row: dict[str, Any],
) -> str:
    return (
        f"{row['architecture']}"
        f"::{row['variant']}"
    )


def no_worse(
    first: float,
    second: float,
    direction: str,
) -> bool:
    if direction == "maximize":
        return (
            first
            >= (
                second
                - STRICT_POSITIVE_TOLERANCE
            )
        )

    if direction == "minimize":
        return (
            first
            <= (
                second
                + STRICT_POSITIVE_TOLERANCE
            )
        )

    raise RuntimeError(
        f"Unknown objective direction: {direction}"
    )


def strictly_better(
    first: float,
    second: float,
    direction: str,
) -> bool:
    if direction == "maximize":
        return (
            first
            > (
                second
                + STRICT_POSITIVE_TOLERANCE
            )
        )

    if direction == "minimize":
        return (
            first
            < (
                second
                - STRICT_POSITIVE_TOLERANCE
            )
        )

    raise RuntimeError(
        f"Unknown objective direction: {direction}"
    )


def dominates(
    first: dict[str, Any],
    second: dict[str, Any],
    objectives: list[dict[str, str]],
) -> bool:
    all_no_worse = True
    at_least_one_better = False

    for objective in objectives:
        metric = objective["metric"]
        direction = objective["direction"]

        first_value = finite_float(
            first[metric],
            metric,
        )

        second_value = finite_float(
            second[metric],
            metric,
        )

        if not no_worse(
            first_value,
            second_value,
            direction,
        ):
            all_no_worse = False
            break

        if strictly_better(
            first_value,
            second_value,
            direction,
        ):
            at_least_one_better = True

    return (
        all_no_worse
        and at_least_one_better
    )


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    DECISION_MATRIX_PATH,
    CRITERIA_PATH,
    ELIGIBILITY_PATH,
    PARETO_PATH,
    SELECTION_TRACE_PATH,
    SUMMARY_PATH,
    COMPLETION_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_ELIGIBILITY,
    OUTPUT_VERIFIED_PARETO,
    OUTPUT_VERIFICATION,
    OUTPUT_FINAL_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 8 verification artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

summary = read_json(
    SUMMARY_PATH
)

completion = read_json(
    COMPLETION_PATH
)

decision_rows_raw = read_csv(
    DECISION_MATRIX_PATH
)

criteria_rows = read_csv(
    CRITERIA_PATH
)

saved_eligibility_rows = read_csv(
    ELIGIBILITY_PATH
)

saved_pareto_rows = read_csv(
    PARETO_PATH
)

saved_selection_rows = read_csv(
    SELECTION_TRACE_PATH
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
            "ready_for_locked_decision_execution"
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
            "ready_for_independent_verification"
        )
        is True
        and summary.get(
            "all_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "ready_for_independent_verification"
        )
        is True
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "eligibility_hash_matches": (
        completion.get(
            "eligibility_results_sha256"
        )
        == sha256_file(
            ELIGIBILITY_PATH
        )
    ),
    "Pareto_hash_matches": (
        completion.get(
            "Pareto_results_sha256"
        )
        == sha256_file(
            PARETO_PATH
        )
    ),
    "selection_trace_hash_matches": (
        completion.get(
            "selection_trace_sha256"
        )
        == sha256_file(
            SELECTION_TRACE_PATH
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
    "decision_rows_22": (
        len(decision_rows_raw)
        == EXPECTED_GROUP_COUNT
    ),
    "eligibility_rows_66": (
        len(saved_eligibility_rows)
        == EXPECTED_ELIGIBILITY_ROWS
    ),
    "Pareto_rows_54": (
        len(saved_pareto_rows)
        == EXPECTED_PARETO_ROWS
    ),
    "selection_rows_3": (
        len(saved_selection_rows)
        == EXPECTED_SELECTION_ROWS
    ),
    "model_inference_not_performed": (
        summary.get(
            "model_inference_performed"
        )
        is False
        and completion.get(
            "model_inference_performed"
        )
        is False
    ),
    "final_model_not_previously_locked": (
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
        "Phase 8 independent verification "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

if (
    protocol.get(
        "primary_profile"
    )
    != PRIMARY_PROFILE
):
    raise RuntimeError(
        "Primary profile mismatch."
    )

profile_thresholds = protocol[
    "eligibility_profiles"
]

objectives = list(
    protocol[
        "Pareto_analysis"
    ]["objectives"]
)

decision_rows: list[
    dict[str, Any]
] = []

numeric_fields = (
    "mean_test_fingerprint_macro_f1",
    "std_test_fingerprint_macro_f1",
    "mean_test_raw_weighted_macro_f1",
    "mean_macro_f1_loss_vs_architecture_B0",
    "worst_seed_macro_f1_loss_vs_architecture_B0",
    "mean_raw_weighted_macro_f1_loss_vs_architecture_B0",
    "mean_serialized_state_dict_bytes",
    "state_size_reduction_fraction_vs_architecture_B0",
    "mean_batch_1_median_ms",
    "batch_1_latency_reduction_fraction_vs_architecture_B0",
    "mean_batch_32_samples_per_second",
    "batch_32_throughput_gain_fraction_vs_architecture_B0",
    "mean_peak_inference_RSS_delta_bytes",
    "macro_f1_mean_improvement_effect_vs_B0",
    "macro_f1_exact_two_sided_p_raw",
    "macro_f1_Holm_adjusted_p",
    "macro_f1_rank_biserial_improvement",
)

for raw in decision_rows_raw:
    row: dict[str, Any] = {
        "architecture": raw[
            "architecture"
        ],
        "variant": raw["variant"],
        "candidate_role": raw[
            "candidate_role"
        ],
        "representation": raw[
            "representation"
        ],
        "fragility_triggered_in_any_run": (
            parse_bool(
                raw[
                    "fragility_triggered_in_any_run"
                ]
            )
        ),
    }

    for field_name in numeric_fields:
        row[field_name] = finite_float(
            raw[field_name],
            field_name,
        )

    decision_rows.append(row)

expected_group_keys = {
    (
        architecture,
        variant,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
}

observed_group_keys = {
    (
        row["architecture"],
        row["variant"],
    )
    for row in decision_rows
}

if observed_group_keys != expected_group_keys:
    raise RuntimeError(
        "Decision group matrix is incomplete."
    )

saved_eligibility_lookup = {
    (
        row["profile"],
        row["architecture"],
        row["variant"],
    ): row
    for row in saved_eligibility_rows
}

saved_pareto_lookup = {
    (
        row["profile"],
        row["architecture"],
        row["variant"],
    ): row
    for row in saved_pareto_rows
}

saved_selection_lookup = {
    row["profile"]: row
    for row in saved_selection_rows
}

print("=" * 92)
print("PHASE 8 MULTI-OBJECTIVE DECISION INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Input groups                    : 22"
)
print(
    "Eligibility rows                : 66"
)
print(
    "Pareto rows                     : 54"
)
print(
    "Selection profiles              : 3"
)
print(
    "Weighted score recomputed       : False"
)
print(
    "Eligibility recomputed          : True"
)
print(
    "Pareto dominance recomputed     : True"
)
print(
    "Selection rule recomputed       : True"
)
print(
    "Model inference repeated        : False"
)
print(
    "Single checkpoint selected      : False"
)
print()

verified_eligibility_rows: list[
    dict[str, Any]
] = []

verified_pareto_rows: list[
    dict[str, Any]
] = []

recomputed_selection: dict[
    str,
    dict[str, Any] | None,
] = {}

profile_counts: dict[
    str,
    dict[str, int],
] = {}

for profile in PROFILES:
    thresholds = profile_thresholds[
        profile
    ]

    maximum_mean_loss = finite_float(
        thresholds[
            "maximum_mean_macro_f1_loss"
        ],
        "maximum_mean_macro_f1_loss",
    )

    maximum_worst_loss = finite_float(
        thresholds[
            "maximum_worst_seed_macro_f1_loss"
        ],
        "maximum_worst_seed_macro_f1_loss",
    )

    maximum_weighted_loss = finite_float(
        thresholds[
            "maximum_mean_raw_weighted_macro_f1_loss"
        ],
        "maximum_mean_raw_weighted_macro_f1_loss",
    )

    candidates: list[
        dict[str, Any]
    ] = []

    for row in decision_rows:
        size_improved = (
            row[
                "state_size_reduction_fraction_vs_architecture_B0"
            ]
            > STRICT_POSITIVE_TOLERANCE
        )

        latency_improved = (
            row[
                "batch_1_latency_reduction_fraction_vs_architecture_B0"
            ]
            > STRICT_POSITIVE_TOLERANCE
        )

        throughput_improved = (
            row[
                "batch_32_throughput_gain_fraction_vs_architecture_B0"
            ]
            > STRICT_POSITIVE_TOLERANCE
        )

        deployment_improvement_count = sum(
            (
                size_improved,
                latency_improved,
                throughput_improved,
            )
        )

        gates = {
            "role_gate_passed": (
                row["candidate_role"]
                == "compressed_candidate"
            ),
            "fragility_gate_passed": (
                row[
                    "fragility_triggered_in_any_run"
                ]
                is False
            ),
            "mean_macro_f1_gate_passed": (
                row[
                    "mean_macro_f1_loss_vs_architecture_B0"
                ]
                <= (
                    maximum_mean_loss
                    + STRICT_POSITIVE_TOLERANCE
                )
            ),
            "worst_seed_macro_f1_gate_passed": (
                row[
                    "worst_seed_macro_f1_loss_vs_architecture_B0"
                ]
                <= (
                    maximum_worst_loss
                    + STRICT_POSITIVE_TOLERANCE
                )
            ),
            "raw_weighted_macro_f1_gate_passed": (
                row[
                    "mean_raw_weighted_macro_f1_loss_vs_architecture_B0"
                ]
                <= (
                    maximum_weighted_loss
                    + STRICT_POSITIVE_TOLERANCE
                )
            ),
            "deployment_improvement_gate_passed": (
                deployment_improvement_count
                >= 1
            ),
        }

        eligible = all(
            gates.values()
        )

        failed_gates: list[str] = []

        gate_failure_names = (
            (
                "role_gate_passed",
                "candidate_role",
            ),
            (
                "fragility_gate_passed",
                "fragility",
            ),
            (
                "mean_macro_f1_gate_passed",
                "mean_macro_f1_loss",
            ),
            (
                "worst_seed_macro_f1_gate_passed",
                "worst_seed_macro_f1_loss",
            ),
            (
                "raw_weighted_macro_f1_gate_passed",
                "mean_raw_weighted_macro_f1_loss",
            ),
            (
                "deployment_improvement_gate_passed",
                "deployment_improvement",
            ),
        )

        for gate_name, failure_name in (
            gate_failure_names
        ):
            if not gates[gate_name]:
                failed_gates.append(
                    failure_name
                )

        recomputed = {
            "profile": profile,
            "primary_profile": (
                profile
                == PRIMARY_PROFILE
            ),
            "architecture": (
                row["architecture"]
            ),
            "variant": row["variant"],
            "candidate_role": (
                row["candidate_role"]
            ),
            "representation": (
                row["representation"]
            ),
            **gates,
            "deployment_improvement_count": (
                deployment_improvement_count
            ),
            "state_size_improved": (
                size_improved
            ),
            "batch_1_latency_improved": (
                latency_improved
            ),
            "batch_32_throughput_improved": (
                throughput_improved
            ),
            "maximum_mean_macro_f1_loss": (
                maximum_mean_loss
            ),
            "maximum_worst_seed_macro_f1_loss": (
                maximum_worst_loss
            ),
            "maximum_mean_raw_weighted_macro_f1_loss": (
                maximum_weighted_loss
            ),
            "observed_mean_macro_f1_loss": (
                row[
                    "mean_macro_f1_loss_vs_architecture_B0"
                ]
            ),
            "observed_worst_seed_macro_f1_loss": (
                row[
                    "worst_seed_macro_f1_loss_vs_architecture_B0"
                ]
            ),
            "observed_mean_raw_weighted_macro_f1_loss": (
                row[
                    "mean_raw_weighted_macro_f1_loss_vs_architecture_B0"
                ]
            ),
            "eligible": eligible,
            "failed_gate_count": len(
                failed_gates
            ),
            "failed_gates": ";".join(
                failed_gates
            ),
        }

        saved = saved_eligibility_lookup[
            (
                profile,
                row["architecture"],
                row["variant"],
            )
        ]

        boolean_fields = (
            "primary_profile",
            "role_gate_passed",
            "fragility_gate_passed",
            "mean_macro_f1_gate_passed",
            "worst_seed_macro_f1_gate_passed",
            "raw_weighted_macro_f1_gate_passed",
            "deployment_improvement_gate_passed",
            "state_size_improved",
            "batch_1_latency_improved",
            "batch_32_throughput_improved",
            "eligible",
        )

        numeric_check_fields = (
            "deployment_improvement_count",
            "maximum_mean_macro_f1_loss",
            "maximum_worst_seed_macro_f1_loss",
            "maximum_mean_raw_weighted_macro_f1_loss",
            "observed_mean_macro_f1_loss",
            "observed_worst_seed_macro_f1_loss",
            "observed_mean_raw_weighted_macro_f1_loss",
            "failed_gate_count",
        )

        failed_checks: list[str] = []

        for field_name in boolean_fields:
            if (
                parse_bool(
                    saved[field_name]
                )
                != bool(
                    recomputed[
                        field_name
                    ]
                )
            ):
                failed_checks.append(
                    field_name
                )

        for field_name in numeric_check_fields:
            if not close_enough(
                saved[field_name],
                recomputed[field_name],
            ):
                failed_checks.append(
                    field_name
                )

        for field_name in (
            "candidate_role",
            "representation",
            "failed_gates",
        ):
            if (
                saved[field_name]
                != str(
                    recomputed[
                        field_name
                    ]
                )
            ):
                failed_checks.append(
                    field_name
                )

        if failed_checks:
            raise RuntimeError(
                "Eligibility verification "
                f"failed for {profile} "
                f"{row['architecture']} "
                f"{row['variant']}: "
                + ", ".join(
                    failed_checks
                )
            )

        verified_eligibility_rows.append(
            {
                "profile": profile,
                "architecture": (
                    row["architecture"]
                ),
                "variant": row["variant"],
                "eligible": eligible,
                "deployment_improvement_count": (
                    deployment_improvement_count
                ),
                "failed_gates": (
                    recomputed[
                        "failed_gates"
                    ]
                ),
                "all_checks_passed": (
                    True
                ),
            }
        )

        if (
            row["candidate_role"]
            == "compressed_candidate"
        ):
            candidates.append(
                {
                    **row,
                    "eligible": eligible,
                    "deployment_improvement_count": (
                        deployment_improvement_count
                    ),
                    "failed_gates": (
                        recomputed[
                            "failed_gates"
                        ]
                    ),
                }
            )

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["eligible"]
    ]

    nondominated_candidates: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:
        dominated_by_ids: list[str] = []

        if candidate["eligible"]:
            for other in (
                eligible_candidates
            ):
                if (
                    candidate_id(other)
                    == candidate_id(
                        candidate
                    )
                ):
                    continue

                if dominates(
                    other,
                    candidate,
                    objectives,
                ):
                    dominated_by_ids.append(
                        candidate_id(other)
                    )

        is_nondominated = (
            candidate["eligible"]
            and len(
                dominated_by_ids
            )
            == 0
        )

        if is_nondominated:
            nondominated_candidates.append(
                candidate
            )

        saved = saved_pareto_lookup[
            (
                profile,
                candidate[
                    "architecture"
                ],
                candidate["variant"],
            )
        ]

        pareto_checks = {
            "primary_profile": (
                parse_bool(
                    saved[
                        "primary_profile"
                    ]
                )
                == (
                    profile
                    == PRIMARY_PROFILE
                )
            ),
            "representation": (
                saved[
                    "representation"
                ]
                == candidate[
                    "representation"
                ]
            ),
            "eligible": (
                parse_bool(
                    saved["eligible"]
                )
                == candidate[
                    "eligible"
                ]
            ),
            "Pareto_nondominated": (
                parse_bool(
                    saved[
                        "Pareto_nondominated"
                    ]
                )
                == is_nondominated
            ),
            "dominated_by_count": (
                int(
                    saved[
                        "dominated_by_count"
                    ]
                )
                == len(
                    dominated_by_ids
                )
            ),
            "dominated_by": (
                saved["dominated_by"]
                == ";".join(
                    sorted(
                        dominated_by_ids
                    )
                )
            ),
            "mean_macro_f1": close_enough(
                saved[
                    "mean_test_fingerprint_macro_f1"
                ],
                candidate[
                    "mean_test_fingerprint_macro_f1"
                ],
            ),
            "state_bytes": close_enough(
                saved[
                    "mean_serialized_state_dict_bytes"
                ],
                candidate[
                    "mean_serialized_state_dict_bytes"
                ],
            ),
            "latency": close_enough(
                saved[
                    "mean_batch_1_median_ms"
                ],
                candidate[
                    "mean_batch_1_median_ms"
                ],
            ),
            "throughput": close_enough(
                saved[
                    "mean_batch_32_samples_per_second"
                ],
                candidate[
                    "mean_batch_32_samples_per_second"
                ],
            ),
            "deployment_count": (
                int(
                    saved[
                        "deployment_improvement_count"
                    ]
                )
                == candidate[
                    "deployment_improvement_count"
                ]
            ),
            "failed_gates": (
                saved["failed_gates"]
                == candidate[
                    "failed_gates"
                ]
            ),
        }

        failed_pareto_checks = [
            name
            for name, passed
            in pareto_checks.items()
            if not passed
        ]

        if failed_pareto_checks:
            raise RuntimeError(
                "Pareto verification failed "
                f"for {profile} "
                f"{candidate['architecture']} "
                f"{candidate['variant']}: "
                + ", ".join(
                    failed_pareto_checks
                )
            )

        verified_pareto_rows.append(
            {
                "profile": profile,
                "architecture": (
                    candidate[
                        "architecture"
                    ]
                ),
                "variant": (
                    candidate["variant"]
                ),
                "eligible": (
                    candidate["eligible"]
                ),
                "Pareto_nondominated": (
                    is_nondominated
                ),
                "dominated_by_count": len(
                    dominated_by_ids
                ),
                "all_checks_passed": (
                    True
                ),
            }
        )

    selected: dict[
        str,
        Any,
    ] | None = None

    if nondominated_candidates:
        selected = sorted(
            nondominated_candidates,
            key=lambda row: (
                row[
                    "mean_serialized_state_dict_bytes"
                ],
                row[
                    "mean_batch_1_median_ms"
                ],
                -row[
                    "mean_batch_32_samples_per_second"
                ],
                -row[
                    "mean_test_fingerprint_macro_f1"
                ],
                row["architecture"],
                row["variant"],
            ),
        )[0]

    saved_selection = (
        saved_selection_lookup[
            profile
        ]
    )

    selected_id = (
        candidate_id(selected)
        if selected is not None
        else ""
    )

    selection_checks = {
        "primary_profile": (
            parse_bool(
                saved_selection[
                    "primary_profile"
                ]
            )
            == (
                profile
                == PRIMARY_PROFILE
            )
        ),
        "candidate_count": (
            int(
                saved_selection[
                    "compressed_candidate_count"
                ]
            )
            == EXPECTED_COMPRESSED_COUNT
        ),
        "eligible_count": (
            int(
                saved_selection[
                    "eligible_candidate_count"
                ]
            )
            == len(
                eligible_candidates
            )
        ),
        "Pareto_count": (
            int(
                saved_selection[
                    "Pareto_candidate_count"
                ]
            )
            == len(
                nondominated_candidates
            )
        ),
        "selection_available": (
            parse_bool(
                saved_selection[
                    "selection_available"
                ]
            )
            == (
                selected is not None
            )
        ),
        "candidate_id": (
            saved_selection[
                "selected_candidate_id"
            ]
            == selected_id
        ),
        "architecture": (
            saved_selection[
                "selected_architecture"
            ]
            == (
                selected[
                    "architecture"
                ]
                if selected is not None
                else ""
            )
        ),
        "variant": (
            saved_selection[
                "selected_variant"
            ]
            == (
                selected["variant"]
                if selected is not None
                else ""
            )
        ),
        "representation": (
            saved_selection[
                "selected_representation"
            ]
            == (
                selected[
                    "representation"
                ]
                if selected is not None
                else ""
            )
        ),
        "inference_not_repeated": (
            parse_bool(
                saved_selection[
                    "model_inference_repeated"
                ]
            )
            is False
        ),
        "final_selection_not_locked": (
            parse_bool(
                saved_selection[
                    "final_selection_locked"
                ]
            )
            is False
        ),
    }

    if selected is not None:
        numeric_selection_checks = (
            (
                "selected_mean_macro_f1",
                "mean_test_fingerprint_macro_f1",
            ),
            (
                "selected_mean_state_bytes",
                "mean_serialized_state_dict_bytes",
            ),
            (
                "selected_mean_batch_1_median_ms",
                "mean_batch_1_median_ms",
            ),
            (
                "selected_mean_batch_32_samples_per_second",
                "mean_batch_32_samples_per_second",
            ),
            (
                "selected_mean_macro_f1_loss_vs_B0",
                "mean_macro_f1_loss_vs_architecture_B0",
            ),
            (
                "selected_worst_seed_macro_f1_loss_vs_B0",
                "worst_seed_macro_f1_loss_vs_architecture_B0",
            ),
            (
                "selected_deployment_improvement_count",
                "deployment_improvement_count",
            ),
        )

        for saved_name, selected_name in (
            numeric_selection_checks
        ):
            selection_checks[
                saved_name
            ] = close_enough(
                saved_selection[
                    saved_name
                ],
                selected[
                    selected_name
                ],
            )

        selection_checks[
            "selected_fragility"
        ] = (
            parse_bool(
                saved_selection[
                    "selected_fragility_triggered"
                ]
            )
            == selected[
                "fragility_triggered_in_any_run"
            ]
        )

    failed_selection_checks = [
        name
        for name, passed
        in selection_checks.items()
        if not passed
    ]

    if failed_selection_checks:
        raise RuntimeError(
            "Selection trace verification "
            f"failed for {profile}: "
            + ", ".join(
                failed_selection_checks
            )
        )

    recomputed_selection[
        profile
    ] = selected

    profile_counts[
        profile
    ] = {
        "eligible": len(
            eligible_candidates
        ),
        "Pareto": len(
            nondominated_candidates
        ),
    }

    print(
        f"{profile:<8} | "
        f"eligible={len(eligible_candidates):2d} | "
        f"Pareto={len(nondominated_candidates):2d} | "
        f"selection={selected_id or 'NONE'} | "
        "verified",
        flush=True,
    )

if len(
    verified_eligibility_rows
) != EXPECTED_ELIGIBILITY_ROWS:
    raise RuntimeError(
        "Verified eligibility row count "
        "mismatch."
    )

if len(
    verified_pareto_rows
) != EXPECTED_PARETO_ROWS:
    raise RuntimeError(
        "Verified Pareto row count mismatch."
    )

primary_selected = recomputed_selection[
    PRIMARY_PROFILE
]

if primary_selected is None:
    raise RuntimeError(
        "Balanced profile has no selection."
    )

primary_candidate_id = candidate_id(
    primary_selected
)

strict_selected = recomputed_selection[
    "strict"
]

relaxed_selected = recomputed_selection[
    "relaxed"
]

strict_id = (
    candidate_id(
        strict_selected
    )
    if strict_selected is not None
    else None
)

relaxed_id = (
    candidate_id(
        relaxed_selected
    )
    if relaxed_selected is not None
    else None
)

available_ids = [
    value
    for value in (
        strict_id,
        primary_candidate_id,
        relaxed_id,
    )
    if value is not None
]

selection_stable = (
    len(
        set(
            available_ids
        )
    )
    <= 1
)

summary_checks = {
    "primary_profile": (
        summary[
            "primary_profile"
        ]
        == PRIMARY_PROFILE
    ),
    "primary_candidate": (
        summary[
            "primary_provisional_selection"
        ][
            "candidate_id"
        ]
        == primary_candidate_id
    ),
    "primary_architecture": (
        summary[
            "primary_provisional_selection"
        ][
            "architecture"
        ]
        == primary_selected[
            "architecture"
        ]
    ),
    "primary_variant": (
        summary[
            "primary_provisional_selection"
        ][
            "variant"
        ]
        == primary_selected[
            "variant"
        ]
    ),
    "primary_representation": (
        summary[
            "primary_provisional_selection"
        ][
            "representation"
        ]
        == primary_selected[
            "representation"
        ]
    ),
    "primary_eligible": (
        summary[
            "primary_provisional_selection"
        ][
            "eligible"
        ]
        is True
    ),
    "primary_Pareto": (
        summary[
            "primary_provisional_selection"
        ][
            "Pareto_nondominated"
        ]
        is True
    ),
    "strict_selection": (
        summary[
            "sensitivity"
        ][
            "strict_selection"
        ]
        == strict_id
    ),
    "balanced_selection": (
        summary[
            "sensitivity"
        ][
            "balanced_selection"
        ]
        == primary_candidate_id
    ),
    "relaxed_selection": (
        summary[
            "sensitivity"
        ][
            "relaxed_selection"
        ]
        == relaxed_id
    ),
    "stability": (
        summary[
            "sensitivity"
        ][
            "selection_stable_across_available_profiles"
        ]
        == selection_stable
    ),
    "weighted_score_false": (
        summary[
            "decision_policy"
        ][
            "weighted_score_used"
        ]
        is False
    ),
    "p_value_gate_false": (
        summary[
            "decision_policy"
        ][
            "p_value_used_as_exclusion_gate"
        ]
        is False
    ),
    "model_inference_false": (
        summary[
            "decision_policy"
        ][
            "model_inference_repeated"
        ]
        is False
    ),
    "final_model_false_before_verification": (
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
        "Phase 8 summary verification "
        "failed: "
        + ", ".join(
            failed_summary_checks
        )
    )

atomic_csv(
    OUTPUT_VERIFIED_ELIGIBILITY,
    verified_eligibility_rows,
    [
        "profile",
        "architecture",
        "variant",
        "eligible",
        "deployment_improvement_count",
        "failed_gates",
        "all_checks_passed",
    ],
)

atomic_csv(
    OUTPUT_VERIFIED_PARETO,
    verified_pareto_rows,
    [
        "profile",
        "architecture",
        "variant",
        "eligible",
        "Pareto_nondominated",
        "dominated_by_count",
        "all_checks_passed",
    ],
)

global_checks = {
    "eligibility_rows_verified_66": (
        len(
            verified_eligibility_rows
        )
        == EXPECTED_ELIGIBILITY_ROWS
    ),
    "Pareto_rows_verified_54": (
        len(
            verified_pareto_rows
        )
        == EXPECTED_PARETO_ROWS
    ),
    "profile_counts_match": (
        profile_counts
        == {
            "strict": {
                "eligible": 5,
                "Pareto": 2,
            },
            "balanced": {
                "eligible": 11,
                "Pareto": 5,
            },
            "relaxed": {
                "eligible": 11,
                "Pareto": 5,
            },
        }
    ),
    "strict_selection_verified": (
        strict_id
        == (
            "tinyml_mlp"
            "::P50-FP32-FT"
        )
    ),
    "balanced_selection_verified": (
        primary_candidate_id
        == (
            "tinyml_mlp"
            "::P50-QAT"
        )
    ),
    "relaxed_selection_verified": (
        relaxed_id
        == (
            "tinyml_mlp"
            "::P50-QAT"
        )
    ),
    "selection_stability_false": (
        selection_stable
        is False
    ),
    "primary_candidate_is_eligible": (
        True
    ),
    "primary_candidate_is_Pareto": (
        True
    ),
    "weighted_score_not_used": (
        protocol[
            "primary_selection_rule"
        ][
            "weighted_score_used"
        ]
        is False
    ),
    "pairwise_p_not_used_as_gate": (
        protocol[
            "statistical_interpretation"
        ][
            "p_values_used_as_exclusion_gate"
        ]
        is False
    ),
    "model_inference_not_repeated": (
        True
    ),
    "single_seed_checkpoint_not_selected": (
        True
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
        "Phase 8 independent verification "
        "global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "verification_policy": {
        "eligibility_recomputed": True,
        "Pareto_dominance_recomputed": True,
        "deterministic_selection_recomputed": True,
        "weighted_score_used": False,
        "pairwise_p_value_used_as_gate": False,
        "model_inference_repeated": False,
    },
    "profile_results": {
        "strict": {
            **profile_counts[
                "strict"
            ],
            "selection": strict_id,
        },
        "balanced": {
            **profile_counts[
                "balanced"
            ],
            "selection": (
                primary_candidate_id
            ),
        },
        "relaxed": {
            **profile_counts[
                "relaxed"
            ],
            "selection": relaxed_id,
        },
    },
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "verified_primary_model_family": {
        "candidate_id": (
            primary_candidate_id
        ),
        "architecture": (
            primary_selected[
                "architecture"
            ]
        ),
        "variant": (
            primary_selected[
                "variant"
            ]
        ),
        "representation": (
            primary_selected[
                "representation"
            ]
        ),
        "mean_test_fingerprint_macro_f1": (
            primary_selected[
                "mean_test_fingerprint_macro_f1"
            ]
        ),
        "mean_serialized_state_dict_bytes": (
            primary_selected[
                "mean_serialized_state_dict_bytes"
            ]
        ),
        "mean_batch_1_median_ms": (
            primary_selected[
                "mean_batch_1_median_ms"
            ]
        ),
        "mean_batch_32_samples_per_second": (
            primary_selected[
                "mean_batch_32_samples_per_second"
            ]
        ),
        "mean_macro_f1_loss_vs_B0": (
            primary_selected[
                "mean_macro_f1_loss_vs_architecture_B0"
            ]
        ),
        "worst_seed_macro_f1_loss_vs_B0": (
            primary_selected[
                "worst_seed_macro_f1_loss_vs_architecture_B0"
            ]
        ),
        "fragility_triggered": (
            primary_selected[
                "fragility_triggered_in_any_run"
            ]
        ),
        "eligible_under_balanced": True,
        "Pareto_nondominated_under_balanced": (
            True
        ),
    },
    "sensitivity": {
        "selection_stable_across_profiles": (
            selection_stable
        ),
        "strict_profile_differs_from_primary": (
            strict_id
            != primary_candidate_id
        ),
        "relaxed_profile_matches_primary": (
            relaxed_id
            == primary_candidate_id
        ),
    },
    "entry_checks": entry_checks,
    "summary_checks": summary_checks,
    "global_checks": global_checks,
    "verified_eligibility_csv": (
        file_record(
            OUTPUT_VERIFIED_ELIGIBILITY
        )
    ),
    "verified_Pareto_csv": (
        file_record(
            OUTPUT_VERIFIED_PARETO
        )
    ),
    "final_model_family_selected": True,
    "single_seed_checkpoint_selected": False,
    "ready_for_final_artifact_protocol": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

final_lock = {
    "status": "locked",
    "phase": 8,
    "artifact_name": (
        "final_model_family_selection"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "selected_model_family": {
        "candidate_id": (
            primary_candidate_id
        ),
        "architecture": (
            primary_selected[
                "architecture"
            ]
        ),
        "variant": (
            primary_selected[
                "variant"
            ]
        ),
        "representation": (
            primary_selected[
                "representation"
            ]
        ),
    },
    "sensitivity_profile_selections": {
        "strict": strict_id,
        "balanced": (
            primary_candidate_id
        ),
        "relaxed": relaxed_id,
    },
    "selection_stable_across_profiles": (
        selection_stable
    ),
    "weighted_score_used": False,
    "pairwise_p_value_used_as_gate": (
        False
    ),
    "model_inference_repeated_by_verifier": (
        False
    ),
    "final_model_family_selected": (
        True
    ),
    "single_seed_checkpoint_selected": (
        False
    ),
    "single_seed_checkpoint_selection_note": (
        "The architecture-variant family is "
        "locked. A concrete deployment "
        "checkpoint must be chosen under a "
        "separate test-independent artifact "
        "protocol; no best-test-seed selection "
        "is permitted."
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "eligibility_results_sha256": (
        sha256_file(
            ELIGIBILITY_PATH
        )
    ),
    "Pareto_results_sha256": (
        sha256_file(
            PARETO_PATH
        )
    ),
    "selection_trace_sha256": (
        sha256_file(
            SELECTION_TRACE_PATH
        )
    ),
    "ready_for_final_artifact_protocol": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_FINAL_LOCK,
    final_lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 8,
    "artifact_name": (
        "final_model_family_selection"
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
            DECISION_MATRIX_PATH
        ),
        file_record(
            CRITERIA_PATH
        ),
        file_record(
            ELIGIBILITY_PATH
        ),
        file_record(
            PARETO_PATH
        ),
        file_record(
            SELECTION_TRACE_PATH
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
            OUTPUT_VERIFIED_ELIGIBILITY
        ),
        file_record(
            OUTPUT_VERIFIED_PARETO
        ),
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_FINAL_LOCK
        ),
    ],
    "selected_model_family": (
        primary_candidate_id
    ),
    "final_model_family_selected": (
        True
    ),
    "single_seed_checkpoint_selected": (
        False
    ),
    "ready_for_final_artifact_protocol": (
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
print("PHASE 8 FINAL MODEL FAMILY VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Eligibility rows verified       : 66"
)
print(
    "Pareto rows verified            : 54"
)
print(
    "Strict selection                : "
    f"{strict_id}"
)
print(
    "Balanced selection              : "
    f"{primary_candidate_id}"
)
print(
    "Relaxed selection               : "
    f"{relaxed_id}"
)
print(
    "Selection stable across profiles: "
    f"{selection_stable}"
)
print(
    "Primary profile                 : balanced"
)
print(
    "Final model family              : "
    f"{primary_candidate_id}"
)
print(
    "Representation                  : "
    f"{primary_selected['representation']}"
)
print(
    "Weighted score used             : False"
)
print(
    "Pairwise p-value exclusion gate : False"
)
print(
    "Model inference repeated        : False"
)
print(
    "Final model family selected     : True"
)
print(
    "Single checkpoint selected      : False"
)
print(
    "Ready for artifact protocol     : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 8 FINAL MODEL FAMILY VERIFIED AND LOCKED"
)
