from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_protocol_locked_v2.json"
)

RUN_MATRIX_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_run_matrix_v2.csv"
)

RUN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "runs"
)

LOCK_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "locks"
)

EXECUTION_LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_execution_ledger_v2.csv"
)

DEVICE_LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_device_ledger_v2.csv"
)

DEVICE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "devices"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

DOC_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "early_lodo"
    / "devices"
)

EXPECTED_MODELS = [
    "tinyml_mlp_b0",
    "compact_dnn_b0",
    "hist_gradient_boosting_b0",
]

EXPECTED_SEED = 2026

TOLERANCE = 1e-10


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_name(
    value: str,
) -> str:
    result = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return result.strip("_")


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def save_json_atomic(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
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
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            restval="",
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def finite_unit_interval(
    value: Any,
) -> bool:
    if value is None:
        return False

    number = float(value)

    return (
        math.isfinite(number)
        and 0.0 <= number <= 1.0
    )


def optional_finite_unit_interval(
    value: Any,
) -> bool:
    if value is None:
        return True

    return finite_unit_interval(
        value
    )


def close_enough(
    left: Any,
    right: Any,
) -> bool:
    if left is None or right is None:
        return left is right

    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    )


def get_validation_macro_f1(
    lock_summary: dict[str, Any],
) -> float:
    metrics = lock_summary[
        "metrics"
    ]

    if (
        "best_validation_macro_f1"
        in metrics
    ):
        return float(
            metrics[
                "best_validation_macro_f1"
            ]
        )

    return float(
        metrics[
            "validation_macro_f1"
        ]
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="Danmini_Doorbell",
)

args = parser.parse_args()

held_out_device = str(
    args.device
)

device_safe_name = safe_name(
    held_out_device
)


required_files = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    EXECUTION_LEDGER_FILE,
]

missing_required_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_required_files:
    print("Missing required files:")

    for path in missing_required_files:
        print(f"- {path}")

    sys.exit(1)


protocol = load_json(
    PROTOCOL_FILE
)

run_matrix = load_csv(
    RUN_MATRIX_FILE
)

execution_ledger = load_csv(
    EXECUTION_LEDGER_FILE
)


device_run_rows = [
    row
    for row in run_matrix
    if row["held_out_device"]
    == held_out_device
]

device_ledger_rows = [
    row
    for row in execution_ledger
    if row["held_out_device"]
    == held_out_device
]


device_directory = (
    DEVICE_ROOT
    / device_safe_name
)

summary_file = (
    device_directory
    / "device_summary.json"
)

comparison_file = (
    device_directory
    / "model_comparison.csv"
)

report_file = (
    device_directory
    / "device_report.txt"
)

cache_manifest_file = (
    device_directory
    / "cache_release_manifest.csv"
)

release_manifest_file = (
    device_directory
    / "device_release_manifest.csv"
)

completion_note = (
    DOC_ROOT
    / (
        device_safe_name
        + "_COMPLETE.md"
    )
)


cache_directory = (
    CACHE_ROOT
    / device_safe_name
)

cache_summary_file = (
    cache_directory
    / "cache_summary.json"
)


if not cache_summary_file.exists():
    raise RuntimeError(
        "Validated device cache summary "
        "was not found."
    )

cache_summary = load_json(
    cache_summary_file
)


model_rows: list[
    dict[str, Any]
] = []

lock_summaries: dict[
    str,
    dict[str, Any]
] = {}

run_results: dict[
    str,
    dict[str, Any]
] = {}

artifact_paths: list[Path] = []


for model_id in EXPECTED_MODELS:
    matching_runs = [
        row
        for row in device_run_rows
        if row["model_id"]
        == model_id
    ]

    if len(matching_runs) != 1:
        continue

    run_spec = matching_runs[0]
    run_id = run_spec["run_id"]

    lock_summary_file = (
        LOCK_ROOT
        / run_id
        / "lock_summary.json"
    )

    run_result_file = (
        RUN_ROOT
        / run_id
        / "run_result.json"
    )

    if not lock_summary_file.exists():
        continue

    if not run_result_file.exists():
        continue

    lock_summary = load_json(
        lock_summary_file
    )

    run_result = load_json(
        run_result_file
    )

    lock_summaries[
        model_id
    ] = lock_summary

    run_results[
        model_id
    ] = run_result

    lock_metrics = lock_summary[
        "metrics"
    ]

    validation_macro_f1 = (
        get_validation_macro_f1(
            lock_summary
        )
    )

    test_macro_f1 = float(
        lock_metrics[
            "test_fingerprint_macro_f1"
        ]
    )

    test_accuracy = float(
        lock_metrics[
            "test_fingerprint_accuracy"
        ]
    )

    record_macro_f1 = float(
        lock_metrics[
            "test_record_weighted_macro_f1"
        ]
    )

    benign_fnr = lock_metrics.get(
        "benign_fnr"
    )

    gafgyt_fnr = lock_metrics.get(
        "gafgyt_fnr"
    )

    mirai_fnr = lock_metrics.get(
        "mirai_fnr"
    )

    local_assessment = (
        lock_summary[
            "local_gate_assessment"
        ]
    )

    local_severe_signal = bool(
        local_assessment[
            "local_severe_signal"
        ]
    )

    runtime = run_result.get(
        "runtime",
        {}
    )

    total_seconds = float(
        runtime.get(
            "total_seconds",
            0.0,
        )
    )

    model_information = (
        run_result.get(
            "model",
            {}
        )
    )

    parameter_count = (
        model_information.get(
            "parameter_count"
        )
    )

    training_information = (
        run_result.get(
            "training",
            {}
        )
    )

    best_epoch = (
        training_information.get(
            "best_epoch"
        )
    )

    completed_epoch = (
        training_information.get(
            "completed_epoch"
        )
    )

    stopped_early = (
        training_information.get(
            "stopped_early"
        )
    )

    completed_iterations = (
        model_information.get(
            "completed_iterations"
        )
    )

    result_data = run_result[
        "data"
    ]

    model_rows.append(
        {
            "run_id":
                run_id,

            "held_out_device":
                held_out_device,

            "model_id":
                model_id,

            "seed":
                int(
                    run_spec["seed"]
                ),

            "validation_macro_f1":
                validation_macro_f1,

            "test_fingerprint_macro_f1":
                test_macro_f1,

            "test_fingerprint_accuracy":
                test_accuracy,

            "test_record_weighted_macro_f1":
                record_macro_f1,

            "validation_test_macro_f1_drop":
                (
                    validation_macro_f1
                    - test_macro_f1
                ),

            "benign_fnr":
                benign_fnr,

            "gafgyt_fnr":
                gafgyt_fnr,

            "mirai_fnr":
                mirai_fnr,

            "local_severe_signal":
                local_severe_signal,

            "test_fingerprint_count":
                int(
                    result_data[
                        "test_fingerprint_count"
                    ]
                ),

            "test_class_counts":
                "|".join(
                    str(value)
                    for value
                    in result_data[
                        "test_class_counts"
                    ]
                ),

            "parameter_count":
                parameter_count,

            "completed_epoch":
                completed_epoch,

            "best_epoch":
                best_epoch,

            "stopped_early":
                stopped_early,

            "completed_iterations":
                completed_iterations,

            "total_seconds":
                total_seconds,

            "lock_summary_file":
                str(
                    lock_summary_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),

            "run_result_file":
                str(
                    run_result_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),
        }
    )

    artifact_paths.extend(
        [
            lock_summary_file,
            run_result_file,
        ]
    )


model_rows.sort(
    key=lambda row:
        EXPECTED_MODELS.index(
            row["model_id"]
        )
)


test_macro_values = [
    float(
        row[
            "test_fingerprint_macro_f1"
        ]
    )
    for row in model_rows
]

accuracy_values = [
    float(
        row[
            "test_fingerprint_accuracy"
        ]
    )
    for row in model_rows
]

record_macro_values = [
    float(
        row[
            "test_record_weighted_macro_f1"
        ]
    )
    for row in model_rows
]

runtime_values = [
    float(
        row["total_seconds"]
    )
    for row in model_rows
]


attack_fnr_values = []

for row in model_rows:
    for name in (
        "gafgyt_fnr",
        "mirai_fnr",
    ):
        value = row[name]

        if value is not None:
            attack_fnr_values.append(
                float(value)
            )


test_count_values = {
    int(
        row[
            "test_fingerprint_count"
        ]
    )
    for row in model_rows
}

test_class_count_values = {
    row["test_class_counts"]
    for row in model_rows
}


ranked_by_macro_f1 = sorted(
    model_rows,
    key=lambda row: (
        -float(
            row[
                "test_fingerprint_macro_f1"
            ]
        ),
        float(
            row["total_seconds"]
        ),
        row["model_id"],
    ),
)

ranked_by_runtime = sorted(
    model_rows,
    key=lambda row: (
        float(
            row["total_seconds"]
        ),
        -float(
            row[
                "test_fingerprint_macro_f1"
            ]
        ),
        row["model_id"],
    ),
)


for rank, row in enumerate(
    ranked_by_macro_f1,
    start=1,
):
    row[
        "test_macro_f1_rank"
    ] = rank


for rank, row in enumerate(
    ranked_by_runtime,
    start=1,
):
    row["runtime_rank"] = rank


comparison_rows = sorted(
    model_rows,
    key=lambda row:
        int(
            row[
                "test_macro_f1_rank"
            ]
        ),
)


ledger_by_run_id = {
    row["run_id"]: row
    for row in device_ledger_rows
}


ledger_rows_match = True

for row in model_rows:
    ledger_row = ledger_by_run_id.get(
        row["run_id"]
    )

    if ledger_row is None:
        ledger_rows_match = False
        continue

    if ledger_row.get("status") != "locked":
        ledger_rows_match = False

    if (
        ledger_row.get("model_id")
        != row["model_id"]
    ):
        ledger_rows_match = False

    if (
        ledger_row.get(
            "held_out_device"
        )
        != held_out_device
    ):
        ledger_rows_match = False

    if int(
        ledger_row.get(
            "seed",
            -1,
        )
    ) != EXPECTED_SEED:
        ledger_rows_match = False

    if not close_enough(
        ledger_row.get(
            "test_fingerprint_macro_f1"
        ),
        row[
            "test_fingerprint_macro_f1"
        ],
    ):
        ledger_rows_match = False


cache_files = sorted(
    path
    for path in cache_directory.iterdir()
    if path.is_file()
)


cache_manifest_rows = [
    {
        "relative_path":
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    }
    for path in cache_files
]


cache_total_bytes = sum(
    int(row["size_bytes"])
    for row in cache_manifest_rows
)


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked"
        and protocol.get(
            "all_checks_passed"
        )
        is True,

    "exactly_three_device_runs":
        len(device_run_rows)
        == 3,

    "expected_models_registered":
        {
            row["model_id"]
            for row in device_run_rows
        }
        == set(
            EXPECTED_MODELS
        ),

    "all_runs_seed_2026":
        all(
            int(row["seed"])
            == EXPECTED_SEED
            for row in device_run_rows
        ),

    "three_model_results_loaded":
        len(model_rows)
        == 3,

    "three_lock_summaries_loaded":
        len(lock_summaries)
        == 3,

    "three_run_results_loaded":
        len(run_results)
        == 3,

    "all_locks_locked":
        all(
            summary.get("status")
            == "locked"
            and summary.get(
                "all_checks_passed"
            )
            is True
            for summary
            in lock_summaries.values()
        ),

    "all_results_completed":
        all(
            result.get("status")
            == "completed"
            and result.get(
                "scientific_result"
            )
            is True
            and result.get(
                "all_checks_passed"
            )
            is True
            for result
            in run_results.values()
        ),

    "lock_identities_match":
        all(
            summary[
                "run"
            ]["held_out_device"]
            == held_out_device
            and summary[
                "run"
            ]["model_id"]
            == model_id
            and int(
                summary[
                    "run"
                ]["seed"]
            )
            == EXPECTED_SEED
            for model_id, summary
            in lock_summaries.items()
        ),

    "execution_ledger_has_three_rows":
        len(device_ledger_rows)
        == 3,

    "execution_ledger_matches":
        ledger_rows_match,

    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True
        and cache_summary.get(
            "held_out_device"
        )
        == held_out_device,

    "cache_manifest_nonempty":
        len(cache_manifest_rows)
        > 0,

    "all_test_counts_identical":
        len(test_count_values)
        == 1,

    "all_test_class_counts_identical":
        len(
            test_class_count_values
        )
        == 1,

    "all_primary_metrics_valid":
        all(
            finite_unit_interval(
                row[
                    "validation_macro_f1"
                ]
            )
            and finite_unit_interval(
                row[
                    "test_fingerprint_macro_f1"
                ]
            )
            and finite_unit_interval(
                row[
                    "test_fingerprint_accuracy"
                ]
            )
            and finite_unit_interval(
                row[
                    "test_record_weighted_macro_f1"
                ]
            )
            for row in model_rows
        ),

    "all_fnr_values_valid":
        all(
            optional_finite_unit_interval(
                row["benign_fnr"]
            )
            and optional_finite_unit_interval(
                row["gafgyt_fnr"]
            )
            and optional_finite_unit_interval(
                row["mirai_fnr"]
            )
            for row in model_rows
        ),

    "ranking_contains_three_models":
        len(ranked_by_macro_f1)
        == 3
        and {
            row["model_id"]
            for row
            in ranked_by_macro_f1
        }
        == set(
            EXPECTED_MODELS
        ),
}


all_checks_passed = all(
    validation_checks.values()
)

completed_at = utc_now()


device_summary = {
    "protocol_version":
        "early_lodo_device_completion_v2_1",

    "status":
        (
            "complete"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        completed_at,

    "held_out_device":
        held_out_device,

    "seed":
        EXPECTED_SEED,

    "completed_model_count":
        len(model_rows),

    "expected_models":
        EXPECTED_MODELS,

    "device_data": {
        "train_fingerprint_count":
            int(
                cache_summary[
                    "observed_counts"
                ]["train"]
            ),

        "validation_fingerprint_count":
            int(
                cache_summary[
                    "observed_counts"
                ]["validation"]
            ),

        "test_fingerprint_count":
            int(
                cache_summary[
                    "observed_counts"
                ]["test"]
            ),

        "test_record_occurrence_count":
            int(
                cache_summary[
                    "observed_test_occurrence_count"
                ]
            ),

        "test_class_counts":
            cache_summary[
                "observed_class_counts"
            ]["test"],
    },

    "model_results":
        comparison_rows,

    "device_statistics": {
        "mean_test_fingerprint_macro_f1":
            (
                statistics.mean(
                    test_macro_values
                )
                if test_macro_values
                else None
            ),

        "median_test_fingerprint_macro_f1":
            (
                statistics.median(
                    test_macro_values
                )
                if test_macro_values
                else None
            ),

        "minimum_test_fingerprint_macro_f1":
            (
                min(test_macro_values)
                if test_macro_values
                else None
            ),

        "maximum_test_fingerprint_macro_f1":
            (
                max(test_macro_values)
                if test_macro_values
                else None
            ),

        "mean_test_accuracy":
            (
                statistics.mean(
                    accuracy_values
                )
                if accuracy_values
                else None
            ),

        "mean_record_weighted_macro_f1":
            (
                statistics.mean(
                    record_macro_values
                )
                if record_macro_values
                else None
            ),

        "maximum_eligible_attack_fnr":
            (
                max(
                    attack_fnr_values
                )
                if attack_fnr_values
                else None
            ),

        "local_severe_cell_count":
            sum(
                bool(
                    row[
                        "local_severe_signal"
                    ]
                )
                for row in model_rows
            ),

        "any_local_severe_signal":
            any(
                bool(
                    row[
                        "local_severe_signal"
                    ]
                )
                for row in model_rows
            ),
    },

    "ranking": {
        "best_test_macro_f1_model":
            (
                ranked_by_macro_f1[
                    0
                ]["model_id"]
                if ranked_by_macro_f1
                else None
            ),

        "best_test_macro_f1":
            (
                ranked_by_macro_f1[
                    0
                ][
                    "test_fingerprint_macro_f1"
                ]
                if ranked_by_macro_f1
                else None
            ),

        "fastest_model":
            (
                ranked_by_runtime[
                    0
                ]["model_id"]
                if ranked_by_runtime
                else None
            ),

        "fastest_total_seconds":
            (
                ranked_by_runtime[
                    0
                ]["total_seconds"]
                if ranked_by_runtime
                else None
            ),
    },

    "interpretation": {
        "scope":
            (
                "This summary completes one "
                "held-out device across three "
                "locked models."
            ),

        "overall_early_lodo_gate":
            (
                "pending_remaining_devices"
            ),

        "general_model_ranking":
            (
                "not_permitted_until_all_"
                "devices_are_complete"
            ),
    },

    "cache": {
        "directory":
            str(cache_directory),

        "file_count":
            len(cache_manifest_rows),

        "total_bytes":
            cache_total_bytes,

        "manifest":
            str(cache_manifest_file),

        "rebuildable":
            True,

        "cleanup_eligible_after_lock":
            all_checks_passed,
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


device_directory.mkdir(
    parents=True,
    exist_ok=True,
)

write_csv_atomic(
    comparison_file,
    comparison_rows,
    list(
        comparison_rows[0].keys()
    ),
)

write_csv_atomic(
    cache_manifest_file,
    cache_manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)

save_json_atomic(
    summary_file,
    device_summary,
)


report_lines = [
    "=" * 86,
    "EARLY LODO DEVICE COMPLETION SUMMARY",
    "=" * 86,
    "",
    (
        f"Held-out device       : "
        f"{held_out_device}"
    ),
    (
        f"Completed models      : "
        f"{len(model_rows)}/3"
    ),
    (
        f"Train fingerprints    : "
        f"{cache_summary['observed_counts']['train']:,}"
    ),
    (
        f"Validation fingerprints: "
        f"{cache_summary['observed_counts']['validation']:,}"
    ),
    (
        f"Test fingerprints     : "
        f"{cache_summary['observed_counts']['test']:,}"
    ),
    (
        f"Test record count     : "
        f"{cache_summary['observed_test_occurrence_count']:,}"
    ),
    "",
    "MODEL COMPARISON",
]

for row in comparison_rows:
    report_lines.append(
        (
            f"rank={row['test_macro_f1_rank']} | "
            f"{row['model_id']} | "
            f"Macro-F1="
            f"{float(row['test_fingerprint_macro_f1']):.9f} | "
            f"Accuracy="
            f"{float(row['test_fingerprint_accuracy']):.9f} | "
            f"Gafgyt FNR="
            f"{row['gafgyt_fnr']} | "
            f"Mirai FNR="
            f"{row['mirai_fnr']} | "
            f"seconds="
            f"{float(row['total_seconds']):.3f} | "
            f"local_severe="
            f"{row['local_severe_signal']}"
        )
    )

report_lines.extend(
    [
        "",
        "DEVICE STATISTICS",
        (
            f"Mean Macro-F1        : "
            f"{statistics.mean(test_macro_values):.9f}"
        ),
        (
            f"Median Macro-F1      : "
            f"{statistics.median(test_macro_values):.9f}"
        ),
        (
            f"Minimum Macro-F1     : "
            f"{min(test_macro_values):.9f}"
        ),
        (
            f"Maximum Macro-F1     : "
            f"{max(test_macro_values):.9f}"
        ),
        (
            f"Maximum attack FNR   : "
            f"{max(attack_fnr_values):.9f}"
        ),
        (
            f"Local severe cells   : "
            f"{sum(bool(row['local_severe_signal']) for row in model_rows)}"
        ),
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report_lines.append(
        f"{name}: {passed}"
    )


report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


if all_checks_passed:
    completion_note.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_row = ranked_by_macro_f1[0]

    completion_note.write_text(
        f"""# Erken LODO Cihaz Tamamlama Kaydı

- Cihaz: `{held_out_device}`
- Tamamlama zamanı: `{completed_at}`
- Seed: `{EXPECTED_SEED}`
- Tamamlanan model sayısı: `3/3`

## Cihaz verisi

- Eğitim parmak izi: `{cache_summary["observed_counts"]["train"]}`
- Doğrulama parmak izi: `{cache_summary["observed_counts"]["validation"]}`
- Test parmak izi: `{cache_summary["observed_counts"]["test"]}`
- Test kayıt ağırlığı toplamı: `{cache_summary["observed_test_occurrence_count"]}`

## Cihaz içi sonuçlar

| Sıra | Model | Test Macro-F1 | Accuracy | Gafgyt FNR | Mirai FNR |
|---:|---|---:|---:|---:|---:|
"""
        + "\n".join(
            (
                f"| {row['test_macro_f1_rank']} "
                f"| {row['model_id']} "
                f"| {float(row['test_fingerprint_macro_f1']):.9f} "
                f"| {float(row['test_fingerprint_accuracy']):.9f} "
                f"| {row['gafgyt_fnr']} "
                f"| {row['mirai_fnr']} |"
            )
            for row in comparison_rows
        )
        + f"""

Cihaz içindeki en yüksek Macro-F1 değeri
`{best_row["model_id"]}` tarafından
`{float(best_row["test_fingerprint_macro_f1"]):.9f}` olarak üretildi.

Bu sıralama yalnız `{held_out_device}` cihazına aittir. Genel model
sıralaması dokuz cihaz tamamlanmadan yapılamaz.

## Yerel kırılganlık

- Yerel ciddi hücre sayısı:
  `{sum(bool(row["local_severe_signal"]) for row in model_rows)}`
- Genel erken LODO karar kapısı:
  `pending_remaining_devices`

## Cache

Cache dosyaları silinmeden önce SHA-256 manifesti oluşturuldu:

- `{cache_manifest_file.relative_to(PROJECT_ROOT)}`

Cache, kilitli birincil verilerden tekrar üretilebilir.
""",
        encoding="utf-8",
    )


release_paths = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    EXECUTION_LEDGER_FILE,
    cache_summary_file,
    cache_manifest_file,
    summary_file,
    comparison_file,
    report_file,
]

release_paths.extend(
    artifact_paths
)

if completion_note.exists():
    release_paths.append(
        completion_note
    )


release_manifest_rows = [
    {
        "relative_path":
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    }
    for path in release_paths
]


write_csv_atomic(
    release_manifest_file,
    release_manifest_rows,
    [
        "relative_path",
        "size_bytes",
        "sha256",
    ],
)


if all_checks_passed:
    device_ledger_fieldnames = [
        "held_out_device",
        "status",
        "completed_at",
        "seed",
        "completed_model_count",
        "best_test_macro_f1_model",
        "best_test_macro_f1",
        "mean_test_macro_f1",
        "minimum_test_macro_f1",
        "maximum_attack_fnr",
        "local_severe_cell_count",
        "any_local_severe_signal",
        "cache_total_bytes",
        "cache_cleanup_eligible",
        "device_summary_file",
        "release_manifest_file",
    ]

    existing_device_rows = (
        load_csv(
            DEVICE_LEDGER_FILE
        )
        if DEVICE_LEDGER_FILE.exists()
        else []
    )

    existing_device_rows = [
        row
        for row in existing_device_rows
        if row["held_out_device"]
        != held_out_device
    ]

    existing_device_rows.append(
        {
            "held_out_device":
                held_out_device,

            "status":
                "complete",

            "completed_at":
                completed_at,

            "seed":
                EXPECTED_SEED,

            "completed_model_count":
                len(model_rows),

            "best_test_macro_f1_model":
                ranked_by_macro_f1[
                    0
                ]["model_id"],

            "best_test_macro_f1":
                ranked_by_macro_f1[
                    0
                ][
                    "test_fingerprint_macro_f1"
                ],

            "mean_test_macro_f1":
                statistics.mean(
                    test_macro_values
                ),

            "minimum_test_macro_f1":
                min(test_macro_values),

            "maximum_attack_fnr":
                max(
                    attack_fnr_values
                ),

            "local_severe_cell_count":
                sum(
                    bool(
                        row[
                            "local_severe_signal"
                        ]
                    )
                    for row in model_rows
                ),

            "any_local_severe_signal":
                any(
                    bool(
                        row[
                            "local_severe_signal"
                        ]
                    )
                    for row in model_rows
                ),

            "cache_total_bytes":
                cache_total_bytes,

            "cache_cleanup_eligible":
                True,

            "device_summary_file":
                str(
                    summary_file.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "release_manifest_file":
                str(
                    release_manifest_file
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),
        }
    )

    existing_device_rows.sort(
        key=lambda row:
            row["held_out_device"]
    )

    write_csv_atomic(
        DEVICE_LEDGER_FILE,
        existing_device_rows,
        device_ledger_fieldnames,
    )


print("=" * 86)
print("EARLY LODO DEVICE COMPLETION")
print("=" * 86)
print(
    f"Held-out device : "
    f"{held_out_device}"
)
print(
    f"Completed models: "
    f"{len(model_rows)}/3"
)

print()
print("MODEL COMPARISON")

for row in comparison_rows:
    print(
        f"rank={row['test_macro_f1_rank']} | "
        f"{row['model_id']} | "
        f"Macro-F1="
        f"{float(row['test_fingerprint_macro_f1']):.9f} | "
        f"Accuracy="
        f"{float(row['test_fingerprint_accuracy']):.9f} | "
        f"Gafgyt FNR="
        f"{row['gafgyt_fnr']} | "
        f"Mirai FNR="
        f"{row['mirai_fnr']} | "
        f"seconds="
        f"{float(row['total_seconds']):.3f}"
    )

print()
print("DEVICE STATISTICS")
print(
    f"Mean Macro-F1      : "
    f"{statistics.mean(test_macro_values):.9f}"
)
print(
    f"Median Macro-F1    : "
    f"{statistics.median(test_macro_values):.9f}"
)
print(
    f"Minimum Macro-F1   : "
    f"{min(test_macro_values):.9f}"
)
print(
    f"Maximum attack FNR : "
    f"{max(attack_fnr_values):.9f}"
)
print(
    f"Local severe cells : "
    f"{sum(bool(row['local_severe_signal']) for row in model_rows)}"
)
print(
    f"Cache size         : "
    f"{cache_total_bytes / (1024 ** 3):.3f} GB"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Summary        : {summary_file}")
print(f"Comparison     : {comparison_file}")
print(f"Cache manifest : {cache_manifest_file}")
print(
    f"Release manifest: "
    f"{release_manifest_file}"
)
print(f"Device ledger  : {DEVICE_LEDGER_FILE}")
print(f"Note           : {completion_note}")

if not all_checks_passed:
    print()
    print(
        "EARLY LODO DEVICE "
        "COMPLETION FAILED"
    )
    sys.exit(1)

print()
print(
    "EARLY LODO DEVICE "
    "COMPLETION PASSED"
)
