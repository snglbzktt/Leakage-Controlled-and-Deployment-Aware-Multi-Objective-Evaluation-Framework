from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

AUDIT = ROOT / "results" / "v2" / "audit"
RUNS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
)

PROTOCOL = (
    AUDIT
    / "tabular_baseline_protocol_locked_v3.json"
)

RUN_MATRIX = (
    AUDIT
    / "tabular_baseline_run_matrix_v3.csv"
)

PROTOCOL_LOCK_MANIFEST = (
    AUDIT
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

CACHE_VERIFICATION = (
    AUDIT
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

LR_VERIFICATION = (
    AUDIT
    / "logistic_regression_b0_verification_v3_2_fixed.json"
)

DT_VERIFICATION = (
    AUDIT
    / "decision_tree_b0_verification_v3_2.json"
)

RF_VERIFICATION = (
    AUDIT
    / "random_forest_b0_verification_v3_2.json"
)

HGB_VERIFICATION = (
    AUDIT
    / "hist_gradient_boosting_b0_verification_v3_2.json"
)

LR_RUN_DIRECTORY = (
    RUNS_ROOT
    / "logistic_regression_b0__seed_2026"
)

LR_METRICS = (
    LR_RUN_DIRECTORY
    / "metrics.json"
)

LR_MANIFEST = (
    LR_RUN_DIRECTORY
    / "run_manifest.json"
)

DT_AGGREGATE = (
    AUDIT
    / "decision_tree_b0_aggregate_v3_2.json"
)

DT_RUNS_CSV = (
    AUDIT
    / "decision_tree_b0_runs_v3_2.csv"
)

RF_AGGREGATE = (
    AUDIT
    / "random_forest_b0_aggregate_v3_2.json"
)

RF_RUNS_CSV = (
    AUDIT
    / "random_forest_b0_runs_v3_2.csv"
)

HGB_SUMMARY = (
    AUDIT
    / "hist_gradient_boosting_b0_summary_v3_2.json"
)

OUTPUT_MODEL_SUMMARY_CSV = (
    AUDIT
    / "phase4_classical_baselines_model_summary_v3_2.csv"
)

OUTPUT_RUNS_CSV = (
    AUDIT
    / "phase4_classical_baselines_all_runs_v3_2.csv"
)

OUTPUT_SUMMARY_JSON = (
    AUDIT
    / "phase4_classical_baselines_summary_v3_2.json"
)

OUTPUT_COMPLETION_MARKER = (
    AUDIT
    / "phase4_classical_baselines_completed_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase4_classical_baselines_lock_manifest_v3_2.json"
)

EXPECTED_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)

EXPECTED_MODEL_IDS = {
    "logistic_regression_b0",
    "decision_tree_b0",
    "random_forest_b0",
    "hist_gradient_boosting_b0",
}

EXPECTED_TOTAL_RUNS = 12

EXPECTED_PRIMARY_MACRO_F1 = {
    "logistic_regression_b0": 0.996991730494754,
    "decision_tree_b0": 0.999582638,
    "random_forest_b0": 0.999459819,
    "hist_gradient_boosting_b0": 0.999738068,
}

EXPECTED_RUN_COUNTS = {
    "logistic_regression_b0": 1,
    "decision_tree_b0": 5,
    "random_forest_b0": 5,
    "hist_gradient_boosting_b0": 1,
}

TOLERANCE = 5e-9


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
    chunk_size: int = (
        8 * 1024 * 1024
    ),
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


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

    os.replace(
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

        for row in rows:
            writer.writerow(row)

    os.replace(
        temporary,
        path,
    )


def nested_get(
    value: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    current: Any = value

    for key in keys:
        current = current[key]

    return current


def as_float(
    value: Any,
) -> float:
    return float(value)


def close_enough(
    observed: float,
    expected: float,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    )


def verification_passed(
    path: Path,
) -> bool:
    value = read_json(path)

    return (
        value.get("status")
        == "passed"
        and value.get(
            "all_checks_passed"
        )
        is True
    )


def source_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


required_paths = [
    PROTOCOL,
    RUN_MATRIX,
    PROTOCOL_LOCK_MANIFEST,
    CACHE_VERIFICATION,
    LR_VERIFICATION,
    DT_VERIFICATION,
    RF_VERIFICATION,
    HGB_VERIFICATION,
    LR_METRICS,
    LR_MANIFEST,
    DT_AGGREGATE,
    DT_RUNS_CSV,
    RF_AGGREGATE,
    RF_RUNS_CSV,
    HGB_SUMMARY,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

output_paths = [
    OUTPUT_MODEL_SUMMARY_CSV,
    OUTPUT_RUNS_CSV,
    OUTPUT_SUMMARY_JSON,
    OUTPUT_COMPLETION_MARKER,
    OUTPUT_LOCK_MANIFEST,
]

existing_outputs = [
    path
    for path in output_paths
    if path.exists()
]

if existing_outputs:
    raise FileExistsError(
        "Phase 4 closure artifacts "
        "already exist; refusing to "
        "overwrite: "
        + ", ".join(
            str(path)
            for path in existing_outputs
        )
    )

protocol = read_json(PROTOCOL)
protocol_lock = read_json(
    PROTOCOL_LOCK_MANIFEST
)
cache_verification = read_json(
    CACHE_VERIFICATION
)

lr_verification = read_json(
    LR_VERIFICATION
)

dt_verification = read_json(
    DT_VERIFICATION
)

rf_verification = read_json(
    RF_VERIFICATION
)

hgb_verification = read_json(
    HGB_VERIFICATION
)

lr_metrics = read_json(
    LR_METRICS
)

lr_manifest = read_json(
    LR_MANIFEST
)

dt_aggregate = read_json(
    DT_AGGREGATE
)

rf_aggregate = read_json(
    RF_AGGREGATE
)

hgb_summary = read_json(
    HGB_SUMMARY
)

run_matrix_rows = read_csv(
    RUN_MATRIX
)

dt_run_rows = read_csv(
    DT_RUNS_CSV
)

rf_run_rows = read_csv(
    RF_RUNS_CSV
)

preflight_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
    ),
    "protocol_hash_matches": (
        protocol_lock.get(
            "protocol_file_sha256"
        )
        == sha256_file(PROTOCOL)
    ),
    "run_matrix_hash_matches": (
        protocol_lock.get(
            "run_matrix_file_sha256"
        )
        == sha256_file(RUN_MATRIX)
    ),
    "cache_verification_passed": (
        cache_verification.get(
            "status"
        )
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "logistic_verification_passed": (
        lr_verification.get(
            "status"
        )
        == "passed"
        and lr_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "decision_tree_verification_passed": (
        dt_verification.get(
            "status"
        )
        == "passed"
        and dt_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "random_forest_verification_passed": (
        rf_verification.get(
            "status"
        )
        == "passed"
        and rf_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "hgb_verification_passed": (
        hgb_verification.get(
            "status"
        )
        == "passed"
        and hgb_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "run_matrix_has_12_rows": (
        len(run_matrix_rows)
        == EXPECTED_TOTAL_RUNS
    ),
    "run_matrix_model_set_matches": (
        {
            str(
                row["model_id"]
            ).strip()
            for row in run_matrix_rows
        }
        == EXPECTED_MODEL_IDS
    ),
    "decision_tree_aggregate_completed": (
        dt_aggregate.get(
            "status"
        )
        == "completed"
        and dt_aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "random_forest_aggregate_completed": (
        rf_aggregate.get(
            "status"
        )
        == "completed"
        and rf_aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "hgb_summary_completed": (
        hgb_summary.get(
            "status"
        )
        == "completed"
        and hgb_summary.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
}

failed_preflight = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight:
    raise RuntimeError(
        "Phase 4 closure preflight "
        "failed: "
        + ", ".join(
            failed_preflight
        )
    )

lr_primary = nested_get(
    lr_metrics,
    (
        "test",
        "primary_fingerprint_level",
    ),
)

lr_weighted = nested_get(
    lr_metrics,
    (
        "test",
        "secondary_raw_record_weighted",
    ),
)

lr_model_row = {
    "model_id": (
        "logistic_regression_b0"
    ),
    "display_name": (
        "Logistic Regression"
    ),
    "run_count": 1,
    "seed_policy": "single_seed_2026",
    "input_space": (
        lr_manifest.get(
            "input_space"
        )
    ),
    "mean_test_fingerprint_macro_f1": (
        as_float(
            lr_primary["macro_f1"]
        )
    ),
    "std_test_fingerprint_macro_f1": (
        0.0
    ),
    "min_test_fingerprint_macro_f1": (
        as_float(
            lr_primary["macro_f1"]
        )
    ),
    "max_test_fingerprint_macro_f1": (
        as_float(
            lr_primary["macro_f1"]
        )
    ),
    "mean_test_fingerprint_accuracy": (
        as_float(
            lr_primary["accuracy"]
        )
    ),
    "mean_test_raw_weighted_macro_f1": (
        as_float(
            lr_weighted["macro_f1"]
        )
    ),
    "mean_test_gafgyt_fnr": (
        as_float(
            lr_primary[
                "per_class"
            ]["gafgyt"][
                "false_negative_rate"
            ]
        )
    ),
    "mean_test_mirai_fnr": (
        as_float(
            lr_primary[
                "per_class"
            ]["mirai"][
                "false_negative_rate"
            ]
        )
    ),
    "fragility_triggered_in_any_run": (
        bool(
            lr_metrics[
                "fragility_gate"
            ]["triggered"]
        )
    ),
    "scientific_status": (
        "completed_with_convergence_warning"
    ),
    "important_caveat": (
        "Locked max_iter=200 was "
        "reached; results retained "
        "without protocol modification."
    ),
}

dt_metrics = dt_aggregate[
    "aggregate_metrics"
]

dt_model_row = {
    "model_id": (
        "decision_tree_b0"
    ),
    "display_name": (
        "Decision Tree"
    ),
    "run_count": int(
        dt_aggregate["run_count"]
    ),
    "seed_policy": "five_locked_seeds",
    "input_space": (
        dt_aggregate["input_space"]
    ),
    "mean_test_fingerprint_macro_f1": (
        as_float(
            dt_metrics[
                "test_fingerprint_macro_f1"
            ]["mean"]
        )
    ),
    "std_test_fingerprint_macro_f1": (
        as_float(
            dt_metrics[
                "test_fingerprint_macro_f1"
            ]["std_population"]
        )
    ),
    "min_test_fingerprint_macro_f1": (
        as_float(
            dt_metrics[
                "test_fingerprint_macro_f1"
            ]["minimum"]
        )
    ),
    "max_test_fingerprint_macro_f1": (
        as_float(
            dt_metrics[
                "test_fingerprint_macro_f1"
            ]["maximum"]
        )
    ),
    "mean_test_fingerprint_accuracy": (
        as_float(
            dt_metrics[
                "test_fingerprint_accuracy"
            ]["mean"]
        )
    ),
    "mean_test_raw_weighted_macro_f1": (
        as_float(
            dt_metrics[
                "test_raw_weighted_macro_f1"
            ]["mean"]
        )
    ),
    "mean_test_gafgyt_fnr": (
        as_float(
            dt_metrics[
                "test_gafgyt_fnr"
            ]["mean"]
        )
    ),
    "mean_test_mirai_fnr": (
        as_float(
            dt_metrics[
                "test_mirai_fnr"
            ]["mean"]
        )
    ),
    "fragility_triggered_in_any_run": (
        bool(
            dt_aggregate[
                "fragility_gate_triggered_in_any_run"
            ]
        )
    ),
    "scientific_status": "completed",
    "important_caveat": "",
}

rf_metrics = rf_aggregate[
    "aggregate_metrics"
]

rf_model_row = {
    "model_id": (
        "random_forest_b0"
    ),
    "display_name": (
        "Random Forest"
    ),
    "run_count": int(
        rf_aggregate["run_count"]
    ),
    "seed_policy": "five_locked_seeds",
    "input_space": (
        rf_aggregate["input_space"]
    ),
    "mean_test_fingerprint_macro_f1": (
        as_float(
            rf_metrics[
                "test_fingerprint_macro_f1"
            ]["mean"]
        )
    ),
    "std_test_fingerprint_macro_f1": (
        as_float(
            rf_metrics[
                "test_fingerprint_macro_f1"
            ]["std_population"]
        )
    ),
    "min_test_fingerprint_macro_f1": (
        as_float(
            rf_metrics[
                "test_fingerprint_macro_f1"
            ]["minimum"]
        )
    ),
    "max_test_fingerprint_macro_f1": (
        as_float(
            rf_metrics[
                "test_fingerprint_macro_f1"
            ]["maximum"]
        )
    ),
    "mean_test_fingerprint_accuracy": (
        as_float(
            rf_metrics[
                "test_fingerprint_accuracy"
            ]["mean"]
        )
    ),
    "mean_test_raw_weighted_macro_f1": (
        as_float(
            rf_metrics[
                "test_raw_weighted_macro_f1"
            ]["mean"]
        )
    ),
    "mean_test_gafgyt_fnr": (
        as_float(
            rf_metrics[
                "test_gafgyt_fnr"
            ]["mean"]
        )
    ),
    "mean_test_mirai_fnr": (
        as_float(
            rf_metrics[
                "test_mirai_fnr"
            ]["mean"]
        )
    ),
    "fragility_triggered_in_any_run": (
        bool(
            rf_aggregate[
                "fragility_gate_triggered_in_any_run"
            ]
        )
    ),
    "scientific_status": "completed",
    "important_caveat": "",
}

hgb_model_row = {
    "model_id": (
        "hist_gradient_boosting_b0"
    ),
    "display_name": (
        "HistGradientBoosting"
    ),
    "run_count": 1,
    "seed_policy": "single_seed_2026",
    "input_space": (
        hgb_summary["input_space"]
    ),
    "mean_test_fingerprint_macro_f1": (
        as_float(
            hgb_summary[
                "test_fingerprint_macro_f1"
            ]
        )
    ),
    "std_test_fingerprint_macro_f1": (
        0.0
    ),
    "min_test_fingerprint_macro_f1": (
        as_float(
            hgb_summary[
                "test_fingerprint_macro_f1"
            ]
        )
    ),
    "max_test_fingerprint_macro_f1": (
        as_float(
            hgb_summary[
                "test_fingerprint_macro_f1"
            ]
        )
    ),
    "mean_test_fingerprint_accuracy": (
        as_float(
            hgb_summary[
                "test_fingerprint_accuracy"
            ]
        )
    ),
    "mean_test_raw_weighted_macro_f1": (
        as_float(
            hgb_summary[
                "test_raw_weighted_macro_f1"
            ]
        )
    ),
    "mean_test_gafgyt_fnr": (
        as_float(
            hgb_summary[
                "test_gafgyt_fnr"
            ]
        )
    ),
    "mean_test_mirai_fnr": (
        as_float(
            hgb_summary[
                "test_mirai_fnr"
            ]
        )
    ),
    "fragility_triggered_in_any_run": (
        bool(
            hgb_summary[
                "fragility_gate_triggered"
            ]
        )
    ),
    "scientific_status": "completed",
    "important_caveat": (
        "Single locked seed; descriptive "
        "comparison only."
    ),
}

model_rows = [
    lr_model_row,
    dt_model_row,
    rf_model_row,
    hgb_model_row,
]

model_ids = {
    row["model_id"]
    for row in model_rows
}

if model_ids != EXPECTED_MODEL_IDS:
    raise RuntimeError(
        "Phase 4 model set mismatch."
    )

metric_sanity_checks = {}

for row in model_rows:
    model_id = row["model_id"]

    metric_sanity_checks[
        f"{model_id}_run_count_matches"
    ] = (
        int(row["run_count"])
        == EXPECTED_RUN_COUNTS[
            model_id
        ]
    )

    metric_sanity_checks[
        f"{model_id}_macro_f1_matches_expected"
    ] = close_enough(
        row[
            "mean_test_fingerprint_macro_f1"
        ],
        EXPECTED_PRIMARY_MACRO_F1[
            model_id
        ],
    )

    metric_sanity_checks[
        f"{model_id}_fragility_false"
    ] = (
        row[
            "fragility_triggered_in_any_run"
        ]
        is False
    )

failed_metric_sanity = [
    name
    for name, passed
    in metric_sanity_checks.items()
    if not passed
]

if failed_metric_sanity:
    raise RuntimeError(
        "Phase 4 metric sanity "
        "checks failed: "
        + ", ".join(
            failed_metric_sanity
        )
    )

ranked_model_rows = sorted(
    model_rows,
    key=lambda row: row[
        "mean_test_fingerprint_macro_f1"
    ],
    reverse=True,
)

for rank, row in enumerate(
    ranked_model_rows,
    start=1,
):
    row[
        "descriptive_macro_f1_rank"
    ] = rank

    row[
        "ranking_is_not_model_selection"
    ] = True

model_fieldnames = [
    "descriptive_macro_f1_rank",
    "model_id",
    "display_name",
    "run_count",
    "seed_policy",
    "input_space",
    "mean_test_fingerprint_macro_f1",
    "std_test_fingerprint_macro_f1",
    "min_test_fingerprint_macro_f1",
    "max_test_fingerprint_macro_f1",
    "mean_test_fingerprint_accuracy",
    "mean_test_raw_weighted_macro_f1",
    "mean_test_gafgyt_fnr",
    "mean_test_mirai_fnr",
    "fragility_triggered_in_any_run",
    "scientific_status",
    "important_caveat",
    "ranking_is_not_model_selection",
]

all_run_rows: list[dict[str, Any]] = []

all_run_rows.append(
    {
        "run_id": (
            lr_metrics["run_id"]
        ),
        "model_id": (
            "logistic_regression_b0"
        ),
        "display_name": (
            "Logistic Regression"
        ),
        "seed": int(
            lr_metrics["seed"]
        ),
        "input_space": (
            lr_manifest.get(
                "input_space"
            )
        ),
        "test_fingerprint_macro_f1": (
            lr_model_row[
                "mean_test_fingerprint_macro_f1"
            ]
        ),
        "test_fingerprint_accuracy": (
            lr_model_row[
                "mean_test_fingerprint_accuracy"
            ]
        ),
        "test_raw_weighted_macro_f1": (
            lr_model_row[
                "mean_test_raw_weighted_macro_f1"
            ]
        ),
        "test_gafgyt_fnr": (
            lr_model_row[
                "mean_test_gafgyt_fnr"
            ]
        ),
        "test_mirai_fnr": (
            lr_model_row[
                "mean_test_mirai_fnr"
            ]
        ),
        "fragility_gate_triggered": (
            lr_model_row[
                "fragility_triggered_in_any_run"
            ]
        ),
        "test_evaluation_count": int(
            lr_metrics[
                "test_evaluation_count"
            ]
        ),
        "scientific_status": (
            "completed_with_convergence_warning"
        ),
    }
)

for source_row in dt_run_rows:
    all_run_rows.append(
        {
            "run_id": source_row[
                "run_id"
            ],
            "model_id": (
                "decision_tree_b0"
            ),
            "display_name": (
                "Decision Tree"
            ),
            "seed": int(
                source_row["seed"]
            ),
            "input_space": (
                dt_aggregate[
                    "input_space"
                ]
            ),
            "test_fingerprint_macro_f1": float(
                source_row[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": float(
                source_row[
                    "test_fingerprint_accuracy"
                ]
            ),
            "test_raw_weighted_macro_f1": float(
                source_row[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "test_gafgyt_fnr": float(
                source_row[
                    "test_gafgyt_fnr"
                ]
            ),
            "test_mirai_fnr": float(
                source_row[
                    "test_mirai_fnr"
                ]
            ),
            "fragility_gate_triggered": (
                str(
                    source_row[
                        "fragility_gate_triggered"
                    ]
                ).strip().lower()
                == "true"
            ),
            "test_evaluation_count": 1,
            "scientific_status": (
                "completed"
            ),
        }
    )

for source_row in rf_run_rows:
    all_run_rows.append(
        {
            "run_id": source_row[
                "run_id"
            ],
            "model_id": (
                "random_forest_b0"
            ),
            "display_name": (
                "Random Forest"
            ),
            "seed": int(
                source_row["seed"]
            ),
            "input_space": (
                rf_aggregate[
                    "input_space"
                ]
            ),
            "test_fingerprint_macro_f1": float(
                source_row[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": float(
                source_row[
                    "test_fingerprint_accuracy"
                ]
            ),
            "test_raw_weighted_macro_f1": float(
                source_row[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "test_gafgyt_fnr": float(
                source_row[
                    "test_gafgyt_fnr"
                ]
            ),
            "test_mirai_fnr": float(
                source_row[
                    "test_mirai_fnr"
                ]
            ),
            "fragility_gate_triggered": (
                str(
                    source_row[
                        "fragility_gate_triggered"
                    ]
                ).strip().lower()
                == "true"
            ),
            "test_evaluation_count": 1,
            "scientific_status": (
                "completed"
            ),
        }
    )

all_run_rows.append(
    {
        "run_id": (
            hgb_summary["run_id"]
        ),
        "model_id": (
            "hist_gradient_boosting_b0"
        ),
        "display_name": (
            "HistGradientBoosting"
        ),
        "seed": int(
            hgb_summary["seed"]
        ),
        "input_space": (
            hgb_summary[
                "input_space"
            ]
        ),
        "test_fingerprint_macro_f1": (
            hgb_model_row[
                "mean_test_fingerprint_macro_f1"
            ]
        ),
        "test_fingerprint_accuracy": (
            hgb_model_row[
                "mean_test_fingerprint_accuracy"
            ]
        ),
        "test_raw_weighted_macro_f1": (
            hgb_model_row[
                "mean_test_raw_weighted_macro_f1"
            ]
        ),
        "test_gafgyt_fnr": (
            hgb_model_row[
                "mean_test_gafgyt_fnr"
            ]
        ),
        "test_mirai_fnr": (
            hgb_model_row[
                "mean_test_mirai_fnr"
            ]
        ),
        "fragility_gate_triggered": (
            hgb_model_row[
                "fragility_triggered_in_any_run"
            ]
        ),
        "test_evaluation_count": int(
            hgb_summary[
                "test_evaluation_count"
            ]
        ),
        "scientific_status": (
            "completed"
        ),
    }
)

all_run_rows.sort(
    key=lambda row: (
        row["model_id"],
        int(row["seed"]),
    )
)

run_level_checks = {
    "total_run_count_is_12": (
        len(all_run_rows)
        == EXPECTED_TOTAL_RUNS
    ),
    "all_test_counts_are_one": (
        all(
            int(
                row[
                    "test_evaluation_count"
                ]
            )
            == 1
            for row in all_run_rows
        )
    ),
    "no_fragility_triggered": (
        all(
            row[
                "fragility_gate_triggered"
            ]
            is False
            for row in all_run_rows
        )
    ),
    "run_ids_are_unique": (
        len(
            {
                row["run_id"]
                for row in all_run_rows
            }
        )
        == EXPECTED_TOTAL_RUNS
    ),
}

failed_run_checks = [
    name
    for name, passed
    in run_level_checks.items()
    if not passed
]

if failed_run_checks:
    raise RuntimeError(
        "Phase 4 run-level checks "
        "failed: "
        + ", ".join(
            failed_run_checks
        )
    )

run_fieldnames = [
    "run_id",
    "model_id",
    "display_name",
    "seed",
    "input_space",
    "test_fingerprint_macro_f1",
    "test_fingerprint_accuracy",
    "test_raw_weighted_macro_f1",
    "test_gafgyt_fnr",
    "test_mirai_fnr",
    "fragility_gate_triggered",
    "test_evaluation_count",
    "scientific_status",
]

atomic_csv(
    OUTPUT_MODEL_SUMMARY_CSV,
    ranked_model_rows,
    model_fieldnames,
)

atomic_csv(
    OUTPUT_RUNS_CSV,
    all_run_rows,
    run_fieldnames,
)

source_paths = [
    PROTOCOL,
    RUN_MATRIX,
    PROTOCOL_LOCK_MANIFEST,
    CACHE_VERIFICATION,
    LR_VERIFICATION,
    DT_VERIFICATION,
    RF_VERIFICATION,
    HGB_VERIFICATION,
    LR_METRICS,
    LR_MANIFEST,
    DT_AGGREGATE,
    DT_RUNS_CSV,
    RF_AGGREGATE,
    RF_RUNS_CSV,
    HGB_SUMMARY,
]

source_inventory = [
    source_record(path)
    for path in source_paths
]

best_descriptive = (
    ranked_model_rows[0]
)

summary_json = {
    "status": "completed",
    "phase": 4,
    "phase_name": (
        "classical_tabular_baselines"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "model_count": len(
        ranked_model_rows
    ),
    "total_run_count": len(
        all_run_rows
    ),
    "models": ranked_model_rows,
    "descriptive_highest_mean_macro_f1": {
        "model_id": (
            best_descriptive[
                "model_id"
            ]
        ),
        "display_name": (
            best_descriptive[
                "display_name"
            ]
        ),
        "mean_test_fingerprint_macro_f1": (
            best_descriptive[
                "mean_test_fingerprint_macro_f1"
            ]
        ),
    },
    "interpretation_policy": {
        "ranking_is_descriptive_only": True,
        "no_final_model_selection": True,
        "single_seed_and_five_seed_results_are_not_claimed_as_equivalent_uncertainty_estimates": True,
        "tinyml_selection_requires_accuracy_size_latency_memory_and_device_generalization_evidence": True,
    },
    "scientific_caveats": [
        (
            "Logistic Regression reached "
            "the locked max_iter=200 limit; "
            "the completed result is retained "
            "with a convergence warning."
        ),
        (
            "HGB and Logistic Regression "
            "use one locked seed, while "
            "Decision Tree and Random Forest "
            "use five locked seeds."
        ),
        (
            "The descriptive ranking does "
            "not constitute final TinyML "
            "model selection."
        ),
    ],
    "preflight_checks": (
        preflight_checks
    ),
    "metric_sanity_checks": (
        metric_sanity_checks
    ),
    "run_level_checks": (
        run_level_checks
    ),
    "source_inventory": (
        source_inventory
    ),
    "model_summary_csv": str(
        OUTPUT_MODEL_SUMMARY_CSV
    ),
    "all_runs_csv": str(
        OUTPUT_RUNS_CSV
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SUMMARY_JSON,
    summary_json,
)

completion_marker = {
    "status": "locked",
    "phase": 4,
    "phase_name": (
        "classical_tabular_baselines"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "completed_at_utc": utc_now(),
    "model_count": 4,
    "total_run_count": 12,
    "all_model_verifications_passed": True,
    "all_test_evaluation_counts_equal_one": True,
    "fragility_triggered_in_any_run": False,
    "final_model_selection_performed": False,
    "next_phase_ready": True,
    "next_phase": 5,
    "summary_json": str(
        OUTPUT_SUMMARY_JSON
    ),
    "summary_json_sha256": (
        sha256_file(
            OUTPUT_SUMMARY_JSON
        )
    ),
    "model_summary_csv": str(
        OUTPUT_MODEL_SUMMARY_CSV
    ),
    "model_summary_csv_sha256": (
        sha256_file(
            OUTPUT_MODEL_SUMMARY_CSV
        )
    ),
    "all_runs_csv": str(
        OUTPUT_RUNS_CSV
    ),
    "all_runs_csv_sha256": (
        sha256_file(
            OUTPUT_RUNS_CSV
        )
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION_MARKER,
    completion_marker,
)

generated_inventory = [
    source_record(
        OUTPUT_SUMMARY_JSON
    ),
    source_record(
        OUTPUT_MODEL_SUMMARY_CSV
    ),
    source_record(
        OUTPUT_RUNS_CSV
    ),
    source_record(
        OUTPUT_COMPLETION_MARKER
    ),
]

lock_manifest = {
    "status": "locked",
    "phase": 4,
    "phase_name": (
        "classical_tabular_baselines"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": (
        source_inventory
    ),
    "generated_artifacts": (
        generated_inventory
    ),
    "model_count": 4,
    "total_run_count": 12,
    "next_phase_ready": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "summary_json_exists": (
        OUTPUT_SUMMARY_JSON.exists()
    ),
    "model_summary_csv_exists": (
        OUTPUT_MODEL_SUMMARY_CSV.exists()
    ),
    "all_runs_csv_exists": (
        OUTPUT_RUNS_CSV.exists()
    ),
    "completion_marker_exists": (
        OUTPUT_COMPLETION_MARKER.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "completion_marker_hash_matches": (
        read_json(
            OUTPUT_COMPLETION_MARKER
        )[
            "summary_json_sha256"
        ]
        == sha256_file(
            OUTPUT_SUMMARY_JSON
        )
    ),
    "lock_manifest_locked": (
        read_json(
            OUTPUT_LOCK_MANIFEST
        ).get("status")
        == "locked"
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Phase 4 post-lock checks "
        "failed: "
        + ", ".join(
            failed_post_checks
        )
    )

free_disk_gib = (
    shutil.disk_usage(ROOT).free
    / (1024**3)
)

print("=" * 92)
print("PHASE 4 CLASSICAL BASELINES CLOSURE SUMMARY")
print("=" * 92)
print(
    "Models locked                   : "
    f"{len(ranked_model_rows)}"
)
print(
    "Runs locked                     : "
    f"{len(all_run_rows)}"
)
print(
    "All independent verifications   : True"
)
print(
    "All test counts equal one       : True"
)
print(
    "Fragility in any run            : False"
)
print()
print("DESCRIPTIVE MACRO-F1 RANKING")
print("-" * 92)

for row in ranked_model_rows:
    print(
        f"{row['descriptive_macro_f1_rank']}. "
        f"{row['display_name']:<24} "
        f"{row['mean_test_fingerprint_macro_f1']:.9f} "
        f"(runs={row['run_count']})"
    )

print()
print(
    "Ranking is final model selection: False"
)
print(
    "Phase 4 status                 : LOCKED"
)
print(
    "Phase 5 ready                  : True"
)
print(
    "Summary JSON                   : "
    f"{OUTPUT_SUMMARY_JSON}"
)
print(
    "Model summary CSV              : "
    f"{OUTPUT_MODEL_SUMMARY_CSV}"
)
print(
    "All runs CSV                   : "
    f"{OUTPUT_RUNS_CSV}"
)
print(
    "Lock manifest                  : "
    f"{OUTPUT_LOCK_MANIFEST}"
)
print(
    "Free disk                      : "
    f"{free_disk_gib:.3f} GiB"
)
print(
    "All checks passed              : True"
)
print(
    "PHASE 4 CLASSICAL BASELINES LOCKED"
)
