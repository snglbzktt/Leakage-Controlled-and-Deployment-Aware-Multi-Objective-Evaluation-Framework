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

OUTPUT_ELIGIBILITY = (
    AUDIT
    / "phase8_multi_objective_eligibility_results_v3_2.csv"
)

OUTPUT_PARETO = (
    AUDIT
    / "phase8_multi_objective_pareto_results_v3_2.csv"
)

OUTPUT_SELECTION_TRACE = (
    AUDIT
    / "phase8_multi_objective_selection_trace_v3_2.csv"
)

OUTPUT_SUMMARY = (
    AUDIT
    / "phase8_multi_objective_decision_summary_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase8_multi_objective_decision_completed_v3_2.json"
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
EXPECTED_COMPRESSED_CANDIDATE_COUNT = 18
EXPECTED_ELIGIBILITY_ROWS = (
    EXPECTED_GROUP_COUNT
    * len(PROFILES)
)
EXPECTED_PARETO_ROWS = (
    EXPECTED_COMPRESSED_CANDIDATE_COUNT
    * len(PROFILES)
)
EXPECTED_SELECTION_ROWS = len(PROFILES)

NUMERIC_TOLERANCE = 1e-15

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

            if (
                attempt
                == WINDOWS_FILE_RETRY_COUNT
            ):
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
    if isinstance(
        value,
        bool,
    ):
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
            f"Non-finite value in "
            f"{field_name}: {value}"
        )

    return number


def candidate_id(
    row: dict[str, Any],
) -> str:
    return (
        f"{row['architecture']}"
        f"::{row['variant']}"
    )


def is_strictly_positive(
    value: float,
) -> bool:
    return (
        float(value)
        > NUMERIC_TOLERANCE
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
                - NUMERIC_TOLERANCE
            )
        )

    if direction == "minimize":
        return (
            first
            <= (
                second
                + NUMERIC_TOLERANCE
            )
        )

    raise RuntimeError(
        f"Unknown objective direction: "
        f"{direction}"
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
                + NUMERIC_TOLERANCE
            )
        )

    if direction == "minimize":
        return (
            first
            < (
                second
                - NUMERIC_TOLERANCE
            )
        )

    raise RuntimeError(
        f"Unknown objective direction: "
        f"{direction}"
    )


def dominates(
    first: dict[str, Any],
    second: dict[str, Any],
    objectives: list[dict[str, str]],
) -> bool:
    all_no_worse = True
    any_strictly_better = False

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
            any_strictly_better = True

    return (
        all_no_worse
        and any_strictly_better
    )


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    DECISION_MATRIX_PATH,
    CRITERIA_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_ELIGIBILITY,
    OUTPUT_PARETO,
    OUTPUT_SELECTION_TRACE,
    OUTPUT_SUMMARY,
    OUTPUT_COMPLETION,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 8 decision output "
            "already exists; refusing "
            f"to overwrite: {output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

decision_rows_raw = read_csv(
    DECISION_MATRIX_PATH
)

criteria_rows = read_csv(
    CRITERIA_PATH
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
            "ready_for_locked_decision_execution"
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
    "decision_matrix_hash_matches": (
        preflight_lock.get(
            "decision_input_matrix_sha256"
        )
        == sha256_file(
            DECISION_MATRIX_PATH
        )
    ),
    "criteria_hash_matches": (
        preflight_lock.get(
            "criteria_sha256"
        )
        == sha256_file(
            CRITERIA_PATH
        )
    ),
    "decision_group_count_22": (
        len(decision_rows_raw)
        == EXPECTED_GROUP_COUNT
    ),
    "decision_computation_not_previously_run": (
        preflight_lock.get(
            "decision_computation_performed"
        )
        is False
    ),
    "model_inference_not_previously_run": (
        preflight_lock.get(
            "model_inference_performed"
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
        "Phase 8 decision execution "
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

for profile in PROFILES:
    if profile not in profile_thresholds:
        raise RuntimeError(
            f"Missing profile: {profile}"
        )

objectives = list(
    protocol[
        "Pareto_analysis"
    ]["objectives"]
)

if len(objectives) != 4:
    raise RuntimeError(
        "Expected four Pareto objectives."
    )

decision_rows: list[
    dict[str, Any]
] = []

for raw in decision_rows_raw:
    decision_rows.append(
        {
            "architecture": (
                raw["architecture"]
            ),
            "variant": raw["variant"],
            "candidate_role": (
                raw["candidate_role"]
            ),
            "representation": (
                raw["representation"]
            ),
            "mean_test_fingerprint_macro_f1": (
                finite_float(
                    raw[
                        "mean_test_fingerprint_macro_f1"
                    ],
                    "mean_test_fingerprint_macro_f1",
                )
            ),
            "std_test_fingerprint_macro_f1": (
                finite_float(
                    raw[
                        "std_test_fingerprint_macro_f1"
                    ],
                    "std_test_fingerprint_macro_f1",
                )
            ),
            "mean_test_raw_weighted_macro_f1": (
                finite_float(
                    raw[
                        "mean_test_raw_weighted_macro_f1"
                    ],
                    "mean_test_raw_weighted_macro_f1",
                )
            ),
            "mean_macro_f1_loss_vs_architecture_B0": (
                finite_float(
                    raw[
                        "mean_macro_f1_loss_vs_architecture_B0"
                    ],
                    "mean_macro_f1_loss_vs_architecture_B0",
                )
            ),
            "worst_seed_macro_f1_loss_vs_architecture_B0": (
                finite_float(
                    raw[
                        "worst_seed_macro_f1_loss_vs_architecture_B0"
                    ],
                    "worst_seed_macro_f1_loss_vs_architecture_B0",
                )
            ),
            "mean_raw_weighted_macro_f1_loss_vs_architecture_B0": (
                finite_float(
                    raw[
                        "mean_raw_weighted_macro_f1_loss_vs_architecture_B0"
                    ],
                    "mean_raw_weighted_macro_f1_loss_vs_architecture_B0",
                )
            ),
            "fragility_triggered_in_any_run": (
                parse_bool(
                    raw[
                        "fragility_triggered_in_any_run"
                    ]
                )
            ),
            "mean_serialized_state_dict_bytes": (
                finite_float(
                    raw[
                        "mean_serialized_state_dict_bytes"
                    ],
                    "mean_serialized_state_dict_bytes",
                )
            ),
            "state_size_reduction_fraction_vs_architecture_B0": (
                finite_float(
                    raw[
                        "state_size_reduction_fraction_vs_architecture_B0"
                    ],
                    "state_size_reduction_fraction_vs_architecture_B0",
                )
            ),
            "mean_batch_1_median_ms": (
                finite_float(
                    raw[
                        "mean_batch_1_median_ms"
                    ],
                    "mean_batch_1_median_ms",
                )
            ),
            "batch_1_latency_reduction_fraction_vs_architecture_B0": (
                finite_float(
                    raw[
                        "batch_1_latency_reduction_fraction_vs_architecture_B0"
                    ],
                    "batch_1_latency_reduction_fraction_vs_architecture_B0",
                )
            ),
            "mean_batch_32_samples_per_second": (
                finite_float(
                    raw[
                        "mean_batch_32_samples_per_second"
                    ],
                    "mean_batch_32_samples_per_second",
                )
            ),
            "batch_32_throughput_gain_fraction_vs_architecture_B0": (
                finite_float(
                    raw[
                        "batch_32_throughput_gain_fraction_vs_architecture_B0"
                    ],
                    "batch_32_throughput_gain_fraction_vs_architecture_B0",
                )
            ),
            "mean_peak_inference_RSS_delta_bytes": (
                finite_float(
                    raw[
                        "mean_peak_inference_RSS_delta_bytes"
                    ],
                    "mean_peak_inference_RSS_delta_bytes",
                )
            ),
            "macro_f1_mean_improvement_effect_vs_B0": (
                finite_float(
                    raw[
                        "macro_f1_mean_improvement_effect_vs_B0"
                    ],
                    "macro_f1_mean_improvement_effect_vs_B0",
                )
            ),
            "macro_f1_exact_two_sided_p_raw": (
                finite_float(
                    raw[
                        "macro_f1_exact_two_sided_p_raw"
                    ],
                    "macro_f1_exact_two_sided_p_raw",
                )
            ),
            "macro_f1_Holm_adjusted_p": (
                finite_float(
                    raw[
                        "macro_f1_Holm_adjusted_p"
                    ],
                    "macro_f1_Holm_adjusted_p",
                )
            ),
            "macro_f1_rank_biserial_improvement": (
                finite_float(
                    raw[
                        "macro_f1_rank_biserial_improvement"
                    ],
                    "macro_f1_rank_biserial_improvement",
                )
            ),
        }
    )

observed_group_keys = {
    (
        row["architecture"],
        row["variant"],
    )
    for row in decision_rows
}

expected_group_keys = {
    (
        architecture,
        variant,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
}

if observed_group_keys != expected_group_keys:
    raise RuntimeError(
        "Decision input group keys "
        "are incomplete."
    )

print("=" * 92)
print("PHASE 8 LOCKED MULTI-OBJECTIVE DECISION EXECUTION")
print("=" * 92)
print(
    "Input groups                    : 22"
)
print(
    "Compressed candidate groups     : 18"
)
print(
    "Profiles                        : strict / balanced / relaxed"
)
print(
    "Primary profile                 : balanced"
)
print(
    "Pareto objectives               : 4"
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
    "Final selection locked          : False"
)
print()

eligibility_rows: list[
    dict[str, Any]
] = []

pareto_rows: list[
    dict[str, Any]
] = []

selection_rows: list[
    dict[str, Any]
] = []

profile_selected_ids: dict[
    str,
    str | None,
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

    maximum_worst_seed_loss = finite_float(
        thresholds[
            "maximum_worst_seed_macro_f1_loss"
        ],
        "maximum_worst_seed_macro_f1_loss",
    )

    maximum_raw_weighted_loss = finite_float(
        thresholds[
            "maximum_mean_raw_weighted_macro_f1_loss"
        ],
        "maximum_mean_raw_weighted_macro_f1_loss",
    )

    profile_candidate_records: list[
        dict[str, Any]
    ] = []

    for row in decision_rows:
        size_improved = (
            is_strictly_positive(
                row[
                    "state_size_reduction_fraction_vs_architecture_B0"
                ]
            )
        )

        latency_improved = (
            is_strictly_positive(
                row[
                    "batch_1_latency_reduction_fraction_vs_architecture_B0"
                ]
            )
        )

        throughput_improved = (
            is_strictly_positive(
                row[
                    "batch_32_throughput_gain_fraction_vs_architecture_B0"
                ]
            )
        )

        deployment_improvement_count = sum(
            (
                size_improved,
                latency_improved,
                throughput_improved,
            )
        )

        role_gate = (
            row["candidate_role"]
            == "compressed_candidate"
        )

        fragility_gate = (
            row[
                "fragility_triggered_in_any_run"
            ]
            is False
        )

        mean_macro_f1_gate = (
            row[
                "mean_macro_f1_loss_vs_architecture_B0"
            ]
            <= (
                maximum_mean_loss
                + NUMERIC_TOLERANCE
            )
        )

        worst_seed_gate = (
            row[
                "worst_seed_macro_f1_loss_vs_architecture_B0"
            ]
            <= (
                maximum_worst_seed_loss
                + NUMERIC_TOLERANCE
            )
        )

        raw_weighted_gate = (
            row[
                "mean_raw_weighted_macro_f1_loss_vs_architecture_B0"
            ]
            <= (
                maximum_raw_weighted_loss
                + NUMERIC_TOLERANCE
            )
        )

        deployment_gate = (
            deployment_improvement_count
            >= 1
        )

        eligible = all(
            (
                role_gate,
                fragility_gate,
                mean_macro_f1_gate,
                worst_seed_gate,
                raw_weighted_gate,
                deployment_gate,
            )
        )

        failed_gates: list[str] = []

        if not role_gate:
            failed_gates.append(
                "candidate_role"
            )

        if not fragility_gate:
            failed_gates.append(
                "fragility"
            )

        if not mean_macro_f1_gate:
            failed_gates.append(
                "mean_macro_f1_loss"
            )

        if not worst_seed_gate:
            failed_gates.append(
                "worst_seed_macro_f1_loss"
            )

        if not raw_weighted_gate:
            failed_gates.append(
                "mean_raw_weighted_macro_f1_loss"
            )

        if not deployment_gate:
            failed_gates.append(
                "deployment_improvement"
            )

        eligibility_rows.append(
            {
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
                "role_gate_passed": (
                    role_gate
                ),
                "fragility_gate_passed": (
                    fragility_gate
                ),
                "mean_macro_f1_gate_passed": (
                    mean_macro_f1_gate
                ),
                "worst_seed_macro_f1_gate_passed": (
                    worst_seed_gate
                ),
                "raw_weighted_macro_f1_gate_passed": (
                    raw_weighted_gate
                ),
                "deployment_improvement_gate_passed": (
                    deployment_gate
                ),
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
                    maximum_worst_seed_loss
                ),
                "maximum_mean_raw_weighted_macro_f1_loss": (
                    maximum_raw_weighted_loss
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
        )

        if (
            row["candidate_role"]
            == "compressed_candidate"
        ):
            profile_candidate_records.append(
                {
                    **row,
                    "eligible": eligible,
                    "deployment_improvement_count": (
                        deployment_improvement_count
                    ),
                    "failed_gates": (
                        ";".join(
                            failed_gates
                        )
                    ),
                }
            )

    eligible_candidates = [
        row
        for row in profile_candidate_records
        if row["eligible"]
    ]

    nondominated_candidates: list[
        dict[str, Any]
    ] = []

    for candidate in profile_candidate_records:
        dominated_by_ids: list[str] = []

        if candidate["eligible"]:
            for other in eligible_candidates:
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

        pareto_rows.append(
            {
                "profile": profile,
                "primary_profile": (
                    profile
                    == PRIMARY_PROFILE
                ),
                "architecture": (
                    candidate[
                        "architecture"
                    ]
                ),
                "variant": (
                    candidate["variant"]
                ),
                "representation": (
                    candidate[
                        "representation"
                    ]
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
                "dominated_by": ";".join(
                    sorted(
                        dominated_by_ids
                    )
                ),
                "mean_test_fingerprint_macro_f1": (
                    candidate[
                        "mean_test_fingerprint_macro_f1"
                    ]
                ),
                "mean_serialized_state_dict_bytes": (
                    candidate[
                        "mean_serialized_state_dict_bytes"
                    ]
                ),
                "mean_batch_1_median_ms": (
                    candidate[
                        "mean_batch_1_median_ms"
                    ]
                ),
                "mean_batch_32_samples_per_second": (
                    candidate[
                        "mean_batch_32_samples_per_second"
                    ]
                ),
                "deployment_improvement_count": (
                    candidate[
                        "deployment_improvement_count"
                    ]
                ),
                "failed_gates": (
                    candidate["failed_gates"]
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
                finite_float(
                    row[
                        "mean_serialized_state_dict_bytes"
                    ],
                    "mean_serialized_state_dict_bytes",
                ),
                finite_float(
                    row[
                        "mean_batch_1_median_ms"
                    ],
                    "mean_batch_1_median_ms",
                ),
                -finite_float(
                    row[
                        "mean_batch_32_samples_per_second"
                    ],
                    "mean_batch_32_samples_per_second",
                ),
                -finite_float(
                    row[
                        "mean_test_fingerprint_macro_f1"
                    ],
                    "mean_test_fingerprint_macro_f1",
                ),
                row["architecture"],
                row["variant"],
            ),
        )[0]

    selected_id = (
        candidate_id(selected)
        if selected is not None
        else None
    )

    profile_selected_ids[
        profile
    ] = selected_id

    selection_rows.append(
        {
            "profile": profile,
            "primary_profile": (
                profile
                == PRIMARY_PROFILE
            ),
            "compressed_candidate_count": (
                len(
                    profile_candidate_records
                )
            ),
            "eligible_candidate_count": (
                len(
                    eligible_candidates
                )
            ),
            "Pareto_candidate_count": (
                len(
                    nondominated_candidates
                )
            ),
            "selection_available": (
                selected is not None
            ),
            "selected_architecture": (
                selected[
                    "architecture"
                ]
                if selected is not None
                else ""
            ),
            "selected_variant": (
                selected["variant"]
                if selected is not None
                else ""
            ),
            "selected_representation": (
                selected[
                    "representation"
                ]
                if selected is not None
                else ""
            ),
            "selected_candidate_id": (
                selected_id or ""
            ),
            "selected_mean_macro_f1": (
                selected[
                    "mean_test_fingerprint_macro_f1"
                ]
                if selected is not None
                else ""
            ),
            "selected_mean_state_bytes": (
                selected[
                    "mean_serialized_state_dict_bytes"
                ]
                if selected is not None
                else ""
            ),
            "selected_mean_batch_1_median_ms": (
                selected[
                    "mean_batch_1_median_ms"
                ]
                if selected is not None
                else ""
            ),
            "selected_mean_batch_32_samples_per_second": (
                selected[
                    "mean_batch_32_samples_per_second"
                ]
                if selected is not None
                else ""
            ),
            "selected_mean_macro_f1_loss_vs_B0": (
                selected[
                    "mean_macro_f1_loss_vs_architecture_B0"
                ]
                if selected is not None
                else ""
            ),
            "selected_worst_seed_macro_f1_loss_vs_B0": (
                selected[
                    "worst_seed_macro_f1_loss_vs_architecture_B0"
                ]
                if selected is not None
                else ""
            ),
            "selected_fragility_triggered": (
                selected[
                    "fragility_triggered_in_any_run"
                ]
                if selected is not None
                else ""
            ),
            "selected_deployment_improvement_count": (
                selected[
                    "deployment_improvement_count"
                ]
                if selected is not None
                else ""
            ),
            "selection_rule": (
                "eligible_then_Pareto_then_"
                "state_latency_throughput_"
                "macroF1_lexical"
            ),
            "model_inference_repeated": (
                False
            ),
            "final_selection_locked": (
                False
            ),
        }
    )

    print(
        f"{profile:<8} | "
        f"eligible={len(eligible_candidates):2d} | "
        f"Pareto={len(nondominated_candidates):2d} | "
        f"provisional selection="
        f"{selected_id or 'NONE'}",
        flush=True,
    )

if len(
    eligibility_rows
) != EXPECTED_ELIGIBILITY_ROWS:
    raise RuntimeError(
        "Eligibility output row count "
        "mismatch."
    )

if len(
    pareto_rows
) != EXPECTED_PARETO_ROWS:
    raise RuntimeError(
        "Pareto output row count mismatch."
    )

if len(
    selection_rows
) != EXPECTED_SELECTION_ROWS:
    raise RuntimeError(
        "Selection trace row count "
        "mismatch."
    )

primary_selection = next(
    row
    for row in selection_rows
    if row["profile"]
    == PRIMARY_PROFILE
)

if not primary_selection[
    "selection_available"
]:
    raise RuntimeError(
        "The balanced primary profile "
        "did not produce a selectable "
        "candidate."
    )

available_selected_ids = [
    selected_id
    for selected_id
    in profile_selected_ids.values()
    if selected_id is not None
]

selection_stable_across_available_profiles = (
    len(
        set(
            available_selected_ids
        )
    )
    <= 1
)

primary_candidate_id = (
    primary_selection[
        "selected_candidate_id"
    ]
)

primary_pareto_record = next(
    row
    for row in pareto_rows
    if row["profile"]
    == PRIMARY_PROFILE
    and (
        f"{row['architecture']}"
        f"::{row['variant']}"
    )
    == primary_candidate_id
)

if not (
    primary_pareto_record[
        "eligible"
    ]
    and primary_pareto_record[
        "Pareto_nondominated"
    ]
):
    raise RuntimeError(
        "Primary provisional selection "
        "is not eligible and Pareto "
        "nondominated."
    )

atomic_csv(
    OUTPUT_ELIGIBILITY,
    eligibility_rows,
    [
        "profile",
        "primary_profile",
        "architecture",
        "variant",
        "candidate_role",
        "representation",
        "role_gate_passed",
        "fragility_gate_passed",
        "mean_macro_f1_gate_passed",
        "worst_seed_macro_f1_gate_passed",
        "raw_weighted_macro_f1_gate_passed",
        "deployment_improvement_gate_passed",
        "deployment_improvement_count",
        "state_size_improved",
        "batch_1_latency_improved",
        "batch_32_throughput_improved",
        "maximum_mean_macro_f1_loss",
        "maximum_worst_seed_macro_f1_loss",
        "maximum_mean_raw_weighted_macro_f1_loss",
        "observed_mean_macro_f1_loss",
        "observed_worst_seed_macro_f1_loss",
        "observed_mean_raw_weighted_macro_f1_loss",
        "eligible",
        "failed_gate_count",
        "failed_gates",
    ],
)

atomic_csv(
    OUTPUT_PARETO,
    pareto_rows,
    [
        "profile",
        "primary_profile",
        "architecture",
        "variant",
        "representation",
        "eligible",
        "Pareto_nondominated",
        "dominated_by_count",
        "dominated_by",
        "mean_test_fingerprint_macro_f1",
        "mean_serialized_state_dict_bytes",
        "mean_batch_1_median_ms",
        "mean_batch_32_samples_per_second",
        "deployment_improvement_count",
        "failed_gates",
    ],
)

atomic_csv(
    OUTPUT_SELECTION_TRACE,
    selection_rows,
    [
        "profile",
        "primary_profile",
        "compressed_candidate_count",
        "eligible_candidate_count",
        "Pareto_candidate_count",
        "selection_available",
        "selected_architecture",
        "selected_variant",
        "selected_representation",
        "selected_candidate_id",
        "selected_mean_macro_f1",
        "selected_mean_state_bytes",
        "selected_mean_batch_1_median_ms",
        "selected_mean_batch_32_samples_per_second",
        "selected_mean_macro_f1_loss_vs_B0",
        "selected_worst_seed_macro_f1_loss_vs_B0",
        "selected_fragility_triggered",
        "selected_deployment_improvement_count",
        "selection_rule",
        "model_inference_repeated",
        "final_selection_locked",
    ],
)

strict_selection = (
    profile_selected_ids[
        "strict"
    ]
)

balanced_selection = (
    profile_selected_ids[
        "balanced"
    ]
)

relaxed_selection = (
    profile_selected_ids[
        "relaxed"
    ]
)

global_checks = {
    "eligibility_rows_66": (
        len(eligibility_rows)
        == EXPECTED_ELIGIBILITY_ROWS
    ),
    "Pareto_rows_54": (
        len(pareto_rows)
        == EXPECTED_PARETO_ROWS
    ),
    "selection_trace_rows_3": (
        len(selection_rows)
        == EXPECTED_SELECTION_ROWS
    ),
    "balanced_selection_available": (
        balanced_selection is not None
    ),
    "primary_selection_eligible": (
        primary_pareto_record[
            "eligible"
        ]
        is True
    ),
    "primary_selection_Pareto": (
        primary_pareto_record[
            "Pareto_nondominated"
        ]
        is True
    ),
    "weighted_score_not_used": (
        protocol[
            "primary_selection_rule"
        ][
            "weighted_score_used"
        ]
        is False
    ),
    "p_value_not_used_as_gate": (
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
    "final_selection_not_locked": (
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
        "Phase 8 decision execution "
        "global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

summary = {
    "status": "completed",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_execution"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "profile_results": {
        row["profile"]: {
            "eligible_candidate_count": (
                row[
                    "eligible_candidate_count"
                ]
            ),
            "Pareto_candidate_count": (
                row[
                    "Pareto_candidate_count"
                ]
            ),
            "selection_available": (
                row[
                    "selection_available"
                ]
            ),
            "provisional_selected_candidate_id": (
                row[
                    "selected_candidate_id"
                ]
            ),
            "selected_architecture": (
                row[
                    "selected_architecture"
                ]
            ),
            "selected_variant": (
                row[
                    "selected_variant"
                ]
            ),
        }
        for row in selection_rows
    },
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "primary_provisional_selection": {
        "candidate_id": (
            primary_candidate_id
        ),
        "architecture": (
            primary_selection[
                "selected_architecture"
            ]
        ),
        "variant": (
            primary_selection[
                "selected_variant"
            ]
        ),
        "representation": (
            primary_selection[
                "selected_representation"
            ]
        ),
        "mean_test_fingerprint_macro_f1": (
            primary_selection[
                "selected_mean_macro_f1"
            ]
        ),
        "mean_serialized_state_dict_bytes": (
            primary_selection[
                "selected_mean_state_bytes"
            ]
        ),
        "mean_batch_1_median_ms": (
            primary_selection[
                "selected_mean_batch_1_median_ms"
            ]
        ),
        "mean_batch_32_samples_per_second": (
            primary_selection[
                "selected_mean_batch_32_samples_per_second"
            ]
        ),
        "mean_macro_f1_loss_vs_B0": (
            primary_selection[
                "selected_mean_macro_f1_loss_vs_B0"
            ]
        ),
        "worst_seed_macro_f1_loss_vs_B0": (
            primary_selection[
                "selected_worst_seed_macro_f1_loss_vs_B0"
            ]
        ),
        "fragility_triggered": (
            primary_selection[
                "selected_fragility_triggered"
            ]
        ),
        "deployment_improvement_count": (
            primary_selection[
                "selected_deployment_improvement_count"
            ]
        ),
        "eligible": True,
        "Pareto_nondominated": True,
    },
    "sensitivity": {
        "strict_selection": (
            strict_selection
        ),
        "balanced_selection": (
            balanced_selection
        ),
        "relaxed_selection": (
            relaxed_selection
        ),
        "selection_stable_across_available_profiles": (
            selection_stable_across_available_profiles
        ),
    },
    "decision_policy": {
        "weighted_score_used": False,
        "p_value_used_as_exclusion_gate": (
            False
        ),
        "model_inference_repeated": (
            False
        ),
        "final_selection_requires_independent_verification": (
            True
        ),
    },
    "entry_checks": entry_checks,
    "global_checks": global_checks,
    "artifacts": {
        "eligibility_results": (
            file_record(
                OUTPUT_ELIGIBILITY
            )
        ),
        "Pareto_results": (
            file_record(
                OUTPUT_PARETO
            )
        ),
        "selection_trace": (
            file_record(
                OUTPUT_SELECTION_TRACE
            )
        ),
    },
    "decision_computation_performed": (
        True
    ),
    "model_inference_performed": False,
    "provisional_selection_available": (
        True
    ),
    "final_model_selected": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SUMMARY,
    summary,
)

completion = {
    "status": "completed",
    "phase": 8,
    "artifact_name": (
        "multi_objective_decision_execution"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "primary_profile": (
        PRIMARY_PROFILE
    ),
    "primary_provisional_candidate_id": (
        primary_candidate_id
    ),
    "eligibility_results": str(
        OUTPUT_ELIGIBILITY
    ),
    "eligibility_results_sha256": (
        sha256_file(
            OUTPUT_ELIGIBILITY
        )
    ),
    "Pareto_results": str(
        OUTPUT_PARETO
    ),
    "Pareto_results_sha256": (
        sha256_file(
            OUTPUT_PARETO
        )
    ),
    "selection_trace": str(
        OUTPUT_SELECTION_TRACE
    ),
    "selection_trace_sha256": (
        sha256_file(
            OUTPUT_SELECTION_TRACE
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
    "decision_computation_performed": (
        True
    ),
    "model_inference_performed": False,
    "provisional_selection_available": (
        True
    ),
    "final_model_selected": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

print()
print("=" * 92)
print("PHASE 8 MULTI-OBJECTIVE DECISION SUMMARY")
print("=" * 92)
print(
    "Strict provisional selection    : "
    f"{strict_selection or 'NONE'}"
)
print(
    "Balanced provisional selection  : "
    f"{balanced_selection}"
)
print(
    "Relaxed provisional selection   : "
    f"{relaxed_selection or 'NONE'}"
)
print(
    "Selection stable across profiles: "
    f"{selection_stable_across_available_profiles}"
)
print(
    "Primary profile                 : balanced"
)
print(
    "Primary provisional selection   : "
    f"{primary_candidate_id}"
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
    "Decision computation performed  : True"
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
    "PHASE 8 LOCKED MULTI-OBJECTIVE DECISION COMPLETED"
)
