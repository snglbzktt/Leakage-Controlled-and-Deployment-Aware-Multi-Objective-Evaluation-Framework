from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


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

RUNNER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "75_run_early_lodo_hgb_experiment_v2.py"
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

LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_execution_ledger_v2.csv"
)

DOC_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "early_lodo"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

NUM_CLASSES = 3

LOCAL_MACRO_F1_THRESHOLD = 0.85
LOCAL_ATTACK_FNR_THRESHOLD = 0.40

TOLERANCE = 1e-9


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(
    value: Any,
) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type "
        f"{type(value).__name__} "
        "is not JSON serializable"
    )


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
            default=json_default,
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


def required_value(
    configuration: dict[str, Any],
    *candidate_names: str,
) -> Any:
    for name in candidate_names:
        if name in configuration:
            return configuration[name]

    raise KeyError(
        "Required configuration value "
        "was not found: "
        + " | ".join(candidate_names)
    )


def confusion_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=(
            np.int64
            if sample_weights is None
            else np.float64
        ),
    )

    if sample_weights is None:
        weights = np.ones(
            len(labels),
            dtype=np.int64,
        )

    else:
        weights = np.asarray(
            sample_weights,
            dtype=np.float64,
        )

    np.add.at(
        confusion,
        (
            labels,
            predictions,
        ),
        weights,
    )

    return confusion


def metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, Any]:
    confusion = np.asarray(
        confusion,
        dtype=np.float64,
    )

    supports = confusion.sum(
        axis=1
    )

    predicted_counts = confusion.sum(
        axis=0
    )

    total = float(
        confusion.sum()
    )

    per_class = {}
    f1_values = []
    recall_values = []
    weighted_f1_numerator = 0.0

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        support = float(
            supports[class_id]
        )

        predicted_count = float(
            predicted_counts[class_id]
        )

        true_positive = float(
            confusion[
                class_id,
                class_id,
            ]
        )

        false_negative = (
            support - true_positive
        )

        false_positive = (
            predicted_count
            - true_positive
        )

        if support <= 0:
            precision = None
            recall = None
            f1_value = None
            fnr = None

        else:
            recall = (
                true_positive
                / support
            )

            fnr = (
                false_negative
                / support
            )

            precision = (
                0.0
                if predicted_count <= 0
                else (
                    true_positive
                    / predicted_count
                )
            )

            denominator = (
                2.0 * true_positive
                + false_positive
                + false_negative
            )

            f1_value = (
                0.0
                if denominator <= 0
                else (
                    2.0
                    * true_positive
                    / denominator
                )
            )

            f1_values.append(
                f1_value
            )

            recall_values.append(
                recall
            )

            weighted_f1_numerator += (
                support * f1_value
            )

        per_class[class_name] = {
            "support":
                support,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1_value,

            "fnr":
                fnr,
        }

    return {
        "accuracy":
            (
                0.0
                if total <= 0
                else float(
                    np.trace(confusion)
                    / total
                )
            ),

        "balanced_accuracy":
            (
                0.0
                if not recall_values
                else float(
                    np.mean(recall_values)
                )
            ),

        "macro_f1_present_classes":
            (
                0.0
                if not f1_values
                else float(
                    np.mean(f1_values)
                )
            ),

        "weighted_f1":
            (
                0.0
                if total <= 0
                else float(
                    weighted_f1_numerator
                    / total
                )
            ),

        "confusion_matrix":
            confusion.tolist(),

        "per_class":
            per_class,
    }


def metrics_match(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> bool:
    scalar_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1_present_classes",
        "weighted_f1",
    ]

    for name in scalar_names:
        if not math.isclose(
            float(expected[name]),
            float(observed[name]),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            return False

    for class_name in CLASS_NAMES:
        expected_class = (
            expected[
                "per_class"
            ][class_name]
        )

        observed_class = (
            observed[
                "per_class"
            ][class_name]
        )

        for name in (
            "support",
            "precision",
            "recall",
            "f1",
            "fnr",
        ):
            expected_value = (
                expected_class[name]
            )

            observed_value = (
                observed_class[name]
            )

            if (
                expected_value is None
                or observed_value is None
            ):
                if expected_value is not (
                    observed_value
                ):
                    return False

            elif not math.isclose(
                float(expected_value),
                float(observed_value),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ):
                return False

    return True


parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-id",
    required=True,
)

args = parser.parse_args()

run_id = str(
    args.run_id
)


run_directory = (
    RUN_ROOT
    / run_id
)

result_file = (
    run_directory
    / "run_result.json"
)

report_file = (
    run_directory
    / "run_report.txt"
)

model_file = (
    run_directory
    / "model.joblib"
)

validation_predictions_file = (
    run_directory
    / "validation_predictions.npy"
)

validation_probabilities_file = (
    run_directory
    / "validation_probabilities.npy"
)

test_predictions_file = (
    run_directory
    / "test_predictions.npy"
)

test_probabilities_file = (
    run_directory
    / "test_probabilities.npy"
)

execution_state_file = (
    run_directory
    / "execution_state.json"
)


required_files = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    RUNNER_SCRIPT,
    result_file,
    report_file,
    model_file,
    validation_predictions_file,
    validation_probabilities_file,
    test_predictions_file,
    test_probabilities_file,
    execution_state_file,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


protocol = load_json(
    PROTOCOL_FILE
)

run_matrix = load_csv(
    RUN_MATRIX_FILE
)

result = load_json(
    result_file
)

execution_state = load_json(
    execution_state_file
)


matching_run_rows = [
    row
    for row in run_matrix
    if row["run_id"] == run_id
]

if len(matching_run_rows) != 1:
    raise RuntimeError(
        "Run matrix row was not found "
        "exactly once."
    )

run_spec = matching_run_rows[0]

held_out_device = (
    run_spec["held_out_device"]
)

model_id = run_spec["model_id"]

seed = int(
    run_spec["seed"]
)


if model_id != (
    "hist_gradient_boosting_b0"
):
    raise RuntimeError(
        "Requested run is not an HGB run."
    )


cache_directory = (
    CACHE_ROOT
    / safe_name(
        held_out_device
    )
)

cache_summary_file = (
    cache_directory
    / "cache_summary.json"
)

y_validation_file = (
    cache_directory
    / "y_validation.npy"
)

y_test_file = (
    cache_directory
    / "y_test.npy"
)

test_occurrence_file = (
    cache_directory
    / "test_occurrence_count.npy"
)


for path in (
    cache_summary_file,
    y_validation_file,
    y_test_file,
    test_occurrence_file,
):
    if not path.exists():
        raise RuntimeError(
            f"Missing cache artifact: {path}"
        )


cache_summary = load_json(
    cache_summary_file
)


model_configurations = {
    model["model_id"]: model
    for model in protocol["models"]
}

model_configuration = (
    model_configurations[model_id]
)


locked_parameters = {
    "loss":
        str(
            required_value(
                model_configuration,
                "loss",
            )
        ),

    "learning_rate":
        float(
            required_value(
                model_configuration,
                "learning_rate",
            )
        ),

    "max_iter":
        int(
            required_value(
                model_configuration,
                "max_iter",
                "maximum_iterations",
            )
        ),

    "max_leaf_nodes":
        int(
            required_value(
                model_configuration,
                "max_leaf_nodes",
            )
        ),

    "max_depth":
        required_value(
            model_configuration,
            "max_depth",
        ),

    "min_samples_leaf":
        int(
            required_value(
                model_configuration,
                "min_samples_leaf",
            )
        ),

    "l2_regularization":
        float(
            required_value(
                model_configuration,
                "l2_regularization",
                "l2",
            )
        ),

    "max_bins":
        int(
            required_value(
                model_configuration,
                "max_bins",
            )
        ),

    "early_stopping":
        bool(
            required_value(
                model_configuration,
                "early_stopping",
            )
        ),

    "class_weight":
        str(
            required_value(
                model_configuration,
                "class_weight",
            )
        ),
}


protocol_sha256 = sha256_file(
    PROTOCOL_FILE
)

run_matrix_sha256 = sha256_file(
    RUN_MATRIX_FILE
)

cache_summary_sha256 = sha256_file(
    cache_summary_file
)

expected_identity = {
    "protocol_sha256":
        protocol_sha256,

    "run_matrix_sha256":
        run_matrix_sha256,

    "cache_summary_sha256":
        cache_summary_sha256,
}


result_run = result["run"]
result_model = result["model"]
result_validation = result[
    "validation"
]
result_test = result["test"]
result_checks = result[
    "validation_checks"
]


model = joblib.load(
    model_file
)

model_parameters = (
    model.get_params(
        deep=False
    )
)


y_validation = np.load(
    y_validation_file,
    mmap_mode="r",
    allow_pickle=False,
).astype(
    np.int64,
    copy=False,
)

y_test = np.load(
    y_test_file,
    mmap_mode="r",
    allow_pickle=False,
).astype(
    np.int64,
    copy=False,
)

test_occurrence = np.load(
    test_occurrence_file,
    mmap_mode="r",
    allow_pickle=False,
)


validation_predictions = np.load(
    validation_predictions_file,
    mmap_mode="r",
    allow_pickle=False,
)

validation_probabilities = np.load(
    validation_probabilities_file,
    mmap_mode="r",
    allow_pickle=False,
)

test_predictions = np.load(
    test_predictions_file,
    mmap_mode="r",
    allow_pickle=False,
)

test_probabilities = np.load(
    test_probabilities_file,
    mmap_mode="r",
    allow_pickle=False,
)


recomputed_validation_predictions = (
    np.argmax(
        validation_probabilities,
        axis=1,
    ).astype(
        np.int8,
        copy=False,
    )
)

recomputed_test_predictions = (
    np.argmax(
        test_probabilities,
        axis=1,
    ).astype(
        np.int8,
        copy=False,
    )
)


validation_confusion = (
    confusion_from_predictions(
        y_validation,
        validation_predictions.astype(
            np.int64,
            copy=False,
        ),
    )
)

fingerprint_confusion = (
    confusion_from_predictions(
        y_test,
        test_predictions.astype(
            np.int64,
            copy=False,
        ),
    )
)

record_confusion = (
    confusion_from_predictions(
        y_test,
        test_predictions.astype(
            np.int64,
            copy=False,
        ),
        sample_weights=(
            test_occurrence
        ),
    )
)


recomputed_validation_metrics = (
    metrics_from_confusion(
        validation_confusion
    )
)

recomputed_fingerprint_metrics = (
    metrics_from_confusion(
        fingerprint_confusion
    )
)

recomputed_record_metrics = (
    metrics_from_confusion(
        record_confusion
    )
)


fingerprint_metrics = (
    result_test[
        "fingerprint_level"
    ]
)

record_metrics = (
    result_test[
        "heldout_record_weighted"
    ]
)

validation_metrics = (
    result_validation[
        "fingerprint_level"
    ]
)


fingerprint_macro_f1 = float(
    fingerprint_metrics[
        "macro_f1_present_classes"
    ]
)

fingerprint_accuracy = float(
    fingerprint_metrics["accuracy"]
)

record_macro_f1 = float(
    record_metrics[
        "macro_f1_present_classes"
    ]
)

validation_macro_f1 = float(
    validation_metrics[
        "macro_f1_present_classes"
    ]
)


per_class = fingerprint_metrics[
    "per_class"
]

benign_fnr = per_class[
    "benign"
]["fnr"]

gafgyt_fnr = per_class[
    "gafgyt"
]["fnr"]

mirai_fnr = per_class[
    "mirai"
]["fnr"]


local_macro_f1_signal = (
    fingerprint_macro_f1
    < LOCAL_MACRO_F1_THRESHOLD
)

local_gafgyt_fnr_signal = (
    gafgyt_fnr is not None
    and float(gafgyt_fnr)
    > LOCAL_ATTACK_FNR_THRESHOLD
)

local_mirai_fnr_signal = (
    mirai_fnr is not None
    and float(mirai_fnr)
    > LOCAL_ATTACK_FNR_THRESHOLD
)

local_severe_signal = bool(
    local_macro_f1_signal
    or local_gafgyt_fnr_signal
    or local_mirai_fnr_signal
)


parameter_checks = {
    "loss":
        model_parameters["loss"]
        == locked_parameters["loss"],

    "learning_rate":
        math.isclose(
            float(
                model_parameters[
                    "learning_rate"
                ]
            ),
            locked_parameters[
                "learning_rate"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),

    "max_iter":
        int(
            model_parameters["max_iter"]
        )
        == locked_parameters["max_iter"],

    "max_leaf_nodes":
        int(
            model_parameters[
                "max_leaf_nodes"
            ]
        )
        == locked_parameters[
            "max_leaf_nodes"
        ],

    "max_depth":
        model_parameters["max_depth"]
        == locked_parameters[
            "max_depth"
        ],

    "min_samples_leaf":
        int(
            model_parameters[
                "min_samples_leaf"
            ]
        )
        == locked_parameters[
            "min_samples_leaf"
        ],

    "l2_regularization":
        math.isclose(
            float(
                model_parameters[
                    "l2_regularization"
                ]
            ),
            locked_parameters[
                "l2_regularization"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        ),

    "max_bins":
        int(
            model_parameters["max_bins"]
        )
        == locked_parameters["max_bins"],

    "early_stopping":
        bool(
            model_parameters[
                "early_stopping"
            ]
        )
        == locked_parameters[
            "early_stopping"
        ],

    "class_weight":
        model_parameters["class_weight"]
        == locked_parameters[
            "class_weight"
        ],

    "random_state":
        int(
            model_parameters[
                "random_state"
            ]
        )
        == seed,
}


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked"
        and protocol.get(
            "all_checks_passed"
        )
        is True,

    "run_result_completed":
        result.get("status")
        == "completed",

    "scientific_result_true":
        result.get(
            "scientific_result"
        )
        is True,

    "result_all_checks_passed":
        result.get(
            "all_checks_passed"
        )
        is True,

    "all_result_internal_checks_true":
        bool(result_checks)
        and all(
            value is True
            for value
            in result_checks.values()
        ),

    "run_id_matches":
        result_run["run_id"]
        == run_id,

    "held_out_device_matches":
        result_run[
            "held_out_device"
        ]
        == held_out_device,

    "model_id_matches":
        result_run["model_id"]
        == model_id,

    "seed_matches":
        int(result_run["seed"])
        == seed,

    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True
        and cache_summary.get(
            "held_out_device"
        )
        == held_out_device,

    "identity_hashes_match":
        result.get("identity")
        == expected_identity,

    "execution_state_completed":
        execution_state.get("status")
        == "completed",

    "execution_identity_matches":
        execution_state.get(
            "identity"
        )
        == {
            "run_id":
                run_id,

            **expected_identity,
        },

    "test_evaluation_count_one":
        int(
            result_test[
                "evaluation_count"
            ]
        )
        == 1
        and int(
            execution_state[
                "test_evaluation_count"
            ]
        )
        == 1,

    "validation_not_used_for_selection":
        result_validation.get(
            "selection_use"
        )
        is False,

    "one_hundred_iterations":
        int(model.n_iter_) == 100
        and int(
            result_model[
                "completed_iterations"
            ]
        )
        == 100,

    "model_classes_valid":
        np.array_equal(
            model.classes_,
            np.asarray(
                [0, 1, 2]
            ),
        ),

    "locked_model_parameters_match":
        all(
            parameter_checks.values()
        ),

    "validation_prediction_shape":
        validation_predictions.shape
        == (
            len(y_validation),
        ),

    "validation_probability_shape":
        validation_probabilities.shape
        == (
            len(y_validation),
            NUM_CLASSES,
        ),

    "test_prediction_shape":
        test_predictions.shape
        == (
            len(y_test),
        ),

    "test_probability_shape":
        test_probabilities.shape
        == (
            len(y_test),
            NUM_CLASSES,
        ),

    "probabilities_finite":
        bool(
            np.isfinite(
                validation_probabilities
            ).all()
            and np.isfinite(
                test_probabilities
            ).all()
        ),

    "probabilities_sum_one":
        bool(
            np.allclose(
                validation_probabilities.sum(
                    axis=1,
                    dtype=np.float64,
                ),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
            and np.allclose(
                test_probabilities.sum(
                    axis=1,
                    dtype=np.float64,
                ),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
        ),

    "validation_predictions_match_probabilities":
        np.array_equal(
            validation_predictions,
            recomputed_validation_predictions,
        ),

    "test_predictions_match_probabilities":
        np.array_equal(
            test_predictions,
            recomputed_test_predictions,
        ),

    "validation_confusion_matches_result":
        np.allclose(
            validation_confusion,
            np.asarray(
                validation_metrics[
                    "confusion_matrix"
                ]
            ),
            rtol=0.0,
            atol=TOLERANCE,
        ),

    "fingerprint_confusion_matches_result":
        np.allclose(
            fingerprint_confusion,
            np.asarray(
                fingerprint_metrics[
                    "confusion_matrix"
                ]
            ),
            rtol=0.0,
            atol=TOLERANCE,
        ),

    "record_confusion_matches_result":
        np.allclose(
            record_confusion,
            np.asarray(
                record_metrics[
                    "confusion_matrix"
                ]
            ),
            rtol=0.0,
            atol=TOLERANCE,
        ),

    "validation_metrics_recomputed":
        metrics_match(
            validation_metrics,
            recomputed_validation_metrics,
        ),

    "fingerprint_metrics_recomputed":
        metrics_match(
            fingerprint_metrics,
            recomputed_fingerprint_metrics,
        ),

    "record_metrics_recomputed":
        metrics_match(
            record_metrics,
            recomputed_record_metrics,
        ),

    "fingerprint_total_matches":
        math.isclose(
            float(
                fingerprint_confusion.sum()
            ),
            float(
                result["data"][
                    "test_fingerprint_count"
                ]
            ),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "record_weight_total_matches":
        math.isclose(
            float(
                record_confusion.sum()
            ),
            float(
                cache_summary[
                    "observed_test_occurrence_count"
                ]
            ),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),

    "reported_metrics_finite":
        all(
            math.isfinite(
                float(value)
            )
            for value in (
                validation_macro_f1,
                fingerprint_macro_f1,
                fingerprint_accuracy,
                record_macro_f1,
                result_validation[
                    "log_loss"
                ],
                result_test[
                    "fingerprint_log_loss"
                ],
                result_test[
                    "record_weighted_log_loss"
                ],
            )
        ),

    "all_output_files_exist":
        all(
            path.exists()
            for path in (
                result_file,
                report_file,
                model_file,
                validation_predictions_file,
                validation_probabilities_file,
                test_predictions_file,
                test_probabilities_file,
                execution_state_file,
            )
        ),
}


all_checks_passed = all(
    validation_checks.values()
)

locked_at = utc_now()


lock_directory = (
    LOCK_ROOT
    / run_id
)

lock_summary_file = (
    lock_directory
    / "lock_summary.json"
)

manifest_file = (
    lock_directory
    / "release_manifest.csv"
)

completion_note = (
    DOC_ROOT
    / (
        run_id
        + "_LOCKED.md"
    )
)


lock_summary = {
    "protocol_version":
        "early_lodo_hgb_run_lock_v2_1",

    "status":
        (
            "locked"
            if all_checks_passed
            else "failed"
        ),

    "locked_at":
        locked_at,

    "run": {
        "run_id":
            run_id,

        "held_out_device":
            held_out_device,

        "model_id":
            model_id,

        "seed":
            seed,

        "completed_iterations":
            int(model.n_iter_),

        "validation_selection_use":
            False,
    },

    "metrics": {
        "validation_macro_f1":
            validation_macro_f1,

        "test_fingerprint_macro_f1":
            fingerprint_macro_f1,

        "test_fingerprint_accuracy":
            fingerprint_accuracy,

        "test_record_weighted_macro_f1":
            record_macro_f1,

        "benign_fnr":
            benign_fnr,

        "gafgyt_fnr":
            gafgyt_fnr,

        "mirai_fnr":
            mirai_fnr,
    },

    "local_gate_assessment": {
        "macro_f1_threshold":
            LOCAL_MACRO_F1_THRESHOLD,

        "attack_fnr_threshold":
            LOCAL_ATTACK_FNR_THRESHOLD,

        "macro_f1_below_threshold":
            local_macro_f1_signal,

        "gafgyt_fnr_above_threshold":
            local_gafgyt_fnr_signal,

        "mirai_fnr_above_threshold":
            local_mirai_fnr_signal,

        "local_severe_signal":
            local_severe_signal,

        "overall_gate_status":
            (
                "pending_remaining_"
                "device_model_runs"
            ),
    },

    "identity": {
        "protocol_sha256":
            protocol_sha256,

        "run_matrix_sha256":
            run_matrix_sha256,

        "cache_summary_sha256":
            cache_summary_sha256,

        "run_result_sha256":
            sha256_file(result_file),

        "model_sha256":
            sha256_file(model_file),

        "validation_predictions_sha256":
            sha256_file(
                validation_predictions_file
            ),

        "validation_probabilities_sha256":
            sha256_file(
                validation_probabilities_file
            ),

        "test_predictions_sha256":
            sha256_file(
                test_predictions_file
            ),

        "test_probabilities_sha256":
            sha256_file(
                test_probabilities_file
            ),

        "execution_state_sha256":
            sha256_file(
                execution_state_file
            ),
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json_atomic(
    lock_summary_file,
    lock_summary,
)


if all_checks_passed:
    completion_note.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_test_drop = (
        validation_macro_f1
        - fingerprint_macro_f1
    )

    completion_note.write_text(
        f"""# Erken LODO HGB Bilimsel Koşu Kilidi

- Run ID: `{run_id}`
- Kilit zamanı: `{locked_at}`
- Held-out cihaz: `{held_out_device}`
- Model: `{model_id}`
- Seed: `{seed}`
- Tamamlanan iterasyon: `{model.n_iter_}`
- Validation seçim amacıyla kullanıldı: `False`

## Sonuçlar

- Validation Macro-F1: `{validation_macro_f1:.9f}`
- Test fingerprint Macro-F1: `{fingerprint_macro_f1:.9f}`
- Validation-test farkı: `{validation_test_drop:.9f}`
- Test accuracy: `{fingerprint_accuracy:.9f}`
- Record-weighted Macro-F1: `{record_macro_f1:.9f}`
- Benign FNR: `{benign_fnr}`
- Gafgyt FNR: `{gafgyt_fnr}`
- Mirai FNR: `{mirai_fnr}`

## Yerel karar

- Macro-F1 < 0.85: `{local_macro_f1_signal}`
- Gafgyt FNR > 0.40: `{local_gafgyt_fnr_signal}`
- Mirai FNR > 0.40: `{local_mirai_fnr_signal}`
- Yerel ciddi kırılganlık: `{local_severe_signal}`

Bu karar yalnızca tek cihaz-model hücresine aittir. Genel erken
LODO değerlendirmesi kalan cihaz ve model koşuları tamamlandıktan
sonra yapılacaktır.

## Kilitli yapıtlar

- `{result_file.relative_to(PROJECT_ROOT)}`
- `{model_file.relative_to(PROJECT_ROOT)}`
- `{validation_predictions_file.relative_to(PROJECT_ROOT)}`
- `{validation_probabilities_file.relative_to(PROJECT_ROOT)}`
- `{test_predictions_file.relative_to(PROJECT_ROOT)}`
- `{test_probabilities_file.relative_to(PROJECT_ROOT)}`
- `{execution_state_file.relative_to(PROJECT_ROOT)}`
- `{lock_summary_file.relative_to(PROJECT_ROOT)}`
- `{manifest_file.relative_to(PROJECT_ROOT)}`
""",
        encoding="utf-8",
    )


    manifest_paths = [
        RUNNER_SCRIPT,
        PROTOCOL_FILE,
        RUN_MATRIX_FILE,
        cache_summary_file,
        result_file,
        report_file,
        model_file,
        validation_predictions_file,
        validation_probabilities_file,
        test_predictions_file,
        test_probabilities_file,
        execution_state_file,
        lock_summary_file,
        completion_note,
    ]

    manifest_rows = [
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
        for path in manifest_paths
    ]

    write_csv_atomic(
        manifest_file,
        manifest_rows,
        [
            "relative_path",
            "size_bytes",
            "sha256",
        ],
    )


    ledger_fieldnames = [
        "run_id",
        "status",
        "locked_at",
        "held_out_device",
        "model_id",
        "seed",
        "completed_epoch",
        "best_epoch",
        "stopped_early",
        "best_validation_macro_f1",
        "validation_macro_f1",
        "validation_selection_use",
        "completed_iterations",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_record_weighted_macro_f1",
        "benign_fnr",
        "gafgyt_fnr",
        "mirai_fnr",
        "local_severe_signal",
        "model_artifact_type",
        "result_file",
        "lock_summary_file",
    ]

    ledger_rows = (
        load_csv(LEDGER_FILE)
        if LEDGER_FILE.exists()
        else []
    )

    normalized_rows = []

    for row in ledger_rows:
        if not row.get(
            "validation_macro_f1"
        ):
            row[
                "validation_macro_f1"
            ] = row.get(
                "best_validation_macro_f1",
                "",
            )

        if not row.get(
            "validation_selection_use"
        ):
            row[
                "validation_selection_use"
            ] = (
                "True"
                if row.get("model_id")
                in (
                    "tinyml_mlp_b0",
                    "compact_dnn_b0",
                )
                else ""
            )

        if not row.get(
            "model_artifact_type"
        ):
            row[
                "model_artifact_type"
            ] = (
                "torch"
                if row.get("model_id")
                in (
                    "tinyml_mlp_b0",
                    "compact_dnn_b0",
                )
                else ""
            )

        normalized_rows.append(row)

    ledger_rows = [
        row
        for row in normalized_rows
        if row["run_id"] != run_id
    ]

    ledger_rows.append(
        {
            "run_id":
                run_id,

            "status":
                "locked",

            "locked_at":
                locked_at,

            "held_out_device":
                held_out_device,

            "model_id":
                model_id,

            "seed":
                seed,

            "completed_epoch":
                "",

            "best_epoch":
                "",

            "stopped_early":
                "",

            "best_validation_macro_f1":
                "",

            "validation_macro_f1":
                validation_macro_f1,

            "validation_selection_use":
                False,

            "completed_iterations":
                int(model.n_iter_),

            "test_fingerprint_macro_f1":
                fingerprint_macro_f1,

            "test_fingerprint_accuracy":
                fingerprint_accuracy,

            "test_record_weighted_macro_f1":
                record_macro_f1,

            "benign_fnr":
                benign_fnr,

            "gafgyt_fnr":
                gafgyt_fnr,

            "mirai_fnr":
                mirai_fnr,

            "local_severe_signal":
                local_severe_signal,

            "model_artifact_type":
                "joblib",

            "result_file":
                str(
                    result_file.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "lock_summary_file":
                str(
                    lock_summary_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
        }
    )

    ledger_rows.sort(
        key=lambda row: row["run_id"]
    )

    write_csv_atomic(
        LEDGER_FILE,
        ledger_rows,
        ledger_fieldnames,
    )


print("=" * 86)
print("EARLY LODO HGB SCIENTIFIC RUN LOCK")
print("=" * 86)
print(f"Run id          : {run_id}")
print(f"Held-out device : {held_out_device}")
print(f"Model           : {model_id}")
print(f"Iterations      : {model.n_iter_}")
print(
    f"Validation F1   : "
    f"{validation_macro_f1:.9f}"
)
print(
    f"Test Macro-F1   : "
    f"{fingerprint_macro_f1:.9f}"
)
print(
    f"Gafgyt FNR      : "
    f"{gafgyt_fnr}"
)
print(
    f"Mirai FNR       : "
    f"{mirai_fnr}"
)
print(
    f"Local severe    : "
    f"{local_severe_signal}"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Lock summary : {lock_summary_file}")

if all_checks_passed:
    print(f"Manifest     : {manifest_file}")
    print(f"Ledger       : {LEDGER_FILE}")
    print(f"Note         : {completion_note}")

if not all_checks_passed:
    print()
    print(
        "EARLY LODO HGB SCIENTIFIC "
        "RUN LOCK FAILED"
    )
    sys.exit(1)

print()
print(
    "EARLY LODO HGB SCIENTIFIC "
    "RUN LOCKED"
)
