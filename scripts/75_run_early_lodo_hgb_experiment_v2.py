from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss


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

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "runs"
)

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

NUM_CLASSES = 3
EXPECTED_FEATURE_COUNT = 115


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


def save_npy_atomic(
    path: Path,
    value: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.stem
        + ".partial"
        + path.suffix
    )

    if temporary.exists():
        temporary.unlink()

    np.save(
        temporary,
        value,
        allow_pickle=False,
    )

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
        "Required configuration key not found: "
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

    present_class_ids = [
        class_id
        for class_id in range(
            NUM_CLASSES
        )
        if supports[class_id] > 0
    ]

    per_class: dict[
        str,
        dict[str, Any]
    ] = {}

    present_f1_values = []
    present_recall_values = []
    weighted_f1_numerator = 0.0

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        support = float(
            supports[class_id]
        )

        predicted_count = float(
            predicted_counts[
                class_id
            ]
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

            present_f1_values.append(
                f1_value
            )

            present_recall_values.append(
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

    accuracy = (
        0.0
        if total <= 0
        else float(
            np.trace(confusion)
            / total
        )
    )

    macro_f1 = (
        0.0
        if not present_f1_values
        else float(
            np.mean(
                present_f1_values
            )
        )
    )

    balanced_accuracy = (
        0.0
        if not present_recall_values
        else float(
            np.mean(
                present_recall_values
            )
        )
    )

    weighted_f1 = (
        0.0
        if total <= 0
        else float(
            weighted_f1_numerator
            / total
        )
    )

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1_present_classes":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "present_class_ids":
            present_class_ids,

        "present_class_names": [
            CLASS_NAMES[class_id]
            for class_id
            in present_class_ids
        ],

        "confusion_matrix":
            confusion.tolist(),

        "per_class":
            per_class,
    }


parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-id",
    required=True,
)

parser.add_argument(
    "--restart",
    action="store_true",
)

args = parser.parse_args()

run_id = str(
    args.run_id
)


required_files = [
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
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


if protocol.get("status") != "locked":
    raise RuntimeError(
        "Early LODO protocol is not locked."
    )

if protocol.get(
    "all_checks_passed"
) is not True:
    raise RuntimeError(
        "Protocol validation failed."
    )


matching_runs = [
    row
    for row in run_matrix
    if row["run_id"] == run_id
]

if len(matching_runs) != 1:
    raise RuntimeError(
        "Requested run_id was not found "
        "exactly once."
    )

run_spec = matching_runs[0]

model_id = run_spec[
    "model_id"
]

held_out_device = run_spec[
    "held_out_device"
]

seed = int(
    run_spec["seed"]
)


if model_id != (
    "hist_gradient_boosting_b0"
):
    raise RuntimeError(
        "This runner only accepts "
        "hist_gradient_boosting_b0."
    )


model_configurations = {
    model["model_id"]: model
    for model in protocol["models"]
}

if model_id not in model_configurations:
    raise RuntimeError(
        "Locked HGB model configuration "
        "was not found."
    )

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


expected_parameter_checks = {
    "loss_log_loss":
        locked_parameters["loss"]
        == "log_loss",

    "learning_rate_0_1":
        math.isclose(
            locked_parameters[
                "learning_rate"
            ],
            0.1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),

    "max_iter_100":
        locked_parameters[
            "max_iter"
        ]
        == 100,

    "max_leaf_nodes_31":
        locked_parameters[
            "max_leaf_nodes"
        ]
        == 31,

    "max_depth_none":
        locked_parameters[
            "max_depth"
        ]
        is None,

    "min_samples_leaf_20":
        locked_parameters[
            "min_samples_leaf"
        ]
        == 20,

    "l2_0_0001":
        math.isclose(
            locked_parameters[
                "l2_regularization"
            ],
            0.0001,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),

    "max_bins_255":
        locked_parameters[
            "max_bins"
        ]
        == 255,

    "early_stopping_false":
        locked_parameters[
            "early_stopping"
        ]
        is False,

    "class_weight_balanced":
        locked_parameters[
            "class_weight"
        ]
        == "balanced",
}

if not all(
    expected_parameter_checks.values()
):
    print(
        "Locked HGB parameters do not "
        "match the preregistered protocol."
    )

    for name, passed in (
        expected_parameter_checks.items()
    ):
        print(f"{name}: {passed}")

    sys.exit(1)


constructor_parameters = (
    inspect.signature(
        HistGradientBoostingClassifier
    ).parameters
)

required_constructor_parameters = {
    "loss",
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "max_bins",
    "early_stopping",
    "class_weight",
    "random_state",
}

missing_constructor_parameters = (
    sorted(
        required_constructor_parameters
        .difference(
            constructor_parameters
        )
    )
)

if missing_constructor_parameters:
    raise RuntimeError(
        "Installed HGB interface is missing: "
        + ", ".join(
            missing_constructor_parameters
        )
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

cache_files = {
    "x_train":
        cache_directory
        / "x_train.npy",

    "y_train":
        cache_directory
        / "y_train.npy",

    "x_validation":
        cache_directory
        / "x_validation.npy",

    "y_validation":
        cache_directory
        / "y_validation.npy",

    "x_test":
        cache_directory
        / "x_test.npy",

    "y_test":
        cache_directory
        / "y_test.npy",

    "test_occurrence_count":
        cache_directory
        / "test_occurrence_count.npy",
}


missing_cache_files = [
    str(path)
    for path in [
        cache_summary_file,
        *cache_files.values(),
    ]
    if not path.exists()
]

if missing_cache_files:
    print("Missing cache files:")

    for path in missing_cache_files:
        print(f"- {path}")

    sys.exit(1)


cache_summary = load_json(
    cache_summary_file
)

if cache_summary.get(
    "all_checks_passed"
) is not True:
    raise RuntimeError(
        "Fold cache validation failed."
    )


run_directory = (
    RESULT_ROOT
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


if args.restart and run_directory.exists():
    shutil.rmtree(
        run_directory
    )


run_directory.mkdir(
    parents=True,
    exist_ok=True,
)


if result_file.exists():
    existing_result = load_json(
        result_file
    )

    if existing_result.get(
        "status"
    ) == "completed":
        print(
            "Existing completed HGB "
            "scientific run was found."
        )
        print(f"Result: {result_file}")
        print(
            "EARLY LODO HGB SCIENTIFIC "
            "RUN ALREADY COMPLETED"
        )
        sys.exit(0)


x_train = np.load(
    cache_files["x_train"],
    mmap_mode="r",
    allow_pickle=False,
)

y_train_memmap = np.load(
    cache_files["y_train"],
    mmap_mode="r",
    allow_pickle=False,
)

x_validation = np.load(
    cache_files["x_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

y_validation_memmap = np.load(
    cache_files["y_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

x_test = np.load(
    cache_files["x_test"],
    mmap_mode="r",
    allow_pickle=False,
)

y_test_memmap = np.load(
    cache_files["y_test"],
    mmap_mode="r",
    allow_pickle=False,
)

test_occurrence_count = np.load(
    cache_files[
        "test_occurrence_count"
    ],
    mmap_mode="r",
    allow_pickle=False,
)


y_train = np.asarray(
    y_train_memmap,
    dtype=np.int64,
)

y_validation = np.asarray(
    y_validation_memmap,
    dtype=np.int64,
)

y_test = np.asarray(
    y_test_memmap,
    dtype=np.int64,
)


if x_train.shape != (
    len(y_train),
    EXPECTED_FEATURE_COUNT,
):
    raise RuntimeError(
        "Training feature shape mismatch."
    )

if x_validation.shape != (
    len(y_validation),
    EXPECTED_FEATURE_COUNT,
):
    raise RuntimeError(
        "Validation feature shape mismatch."
    )

if x_test.shape != (
    len(y_test),
    EXPECTED_FEATURE_COUNT,
):
    raise RuntimeError(
        "Test feature shape mismatch."
    )


protocol_sha256 = sha256_file(
    PROTOCOL_FILE
)

run_matrix_sha256 = sha256_file(
    RUN_MATRIX_FILE
)

cache_summary_sha256 = sha256_file(
    cache_summary_file
)


execution_identity = {
    "run_id":
        run_id,

    "protocol_sha256":
        protocol_sha256,

    "run_matrix_sha256":
        run_matrix_sha256,

    "cache_summary_sha256":
        cache_summary_sha256,
}


save_json_atomic(
    execution_state_file,
    {
        "protocol_version":
            "early_lodo_hgb_execution_state_v2_1",

        "status":
            "fitting",

        "started_at":
            utc_now(),

        "identity":
            execution_identity,

        "note": (
            "HistGradientBoosting fit is an "
            "atomic scikit-learn operation. "
            "If interrupted, rerun the same "
            "command."
        ),
    },
)


print("=" * 86)
print("EARLY LODO HGB SCIENTIFIC RUN")
print("=" * 86)
print(f"Run id          : {run_id}")
print(f"Held-out device : {held_out_device}")
print(f"Model           : {model_id}")
print(f"Seed            : {seed}")
print(f"Train rows      : {len(y_train):,}")
print(
    f"Validation rows : "
    f"{len(y_validation):,}"
)
print(f"Test rows       : {len(y_test):,}")
print(f"Features        : {x_train.shape[1]}")
print(
    f"Scikit-learn    : "
    f"{sklearn.__version__}"
)
print("Scaled features : False")
print()
print("LOCKED PARAMETERS")

for name, value in (
    locked_parameters.items()
):
    print(f"{name}: {value}")

print()
print(
    "Fitting full HistGradientBoosting "
    "model..."
)


model = HistGradientBoostingClassifier(
    loss=locked_parameters["loss"],
    learning_rate=locked_parameters[
        "learning_rate"
    ],
    max_iter=locked_parameters[
        "max_iter"
    ],
    max_leaf_nodes=locked_parameters[
        "max_leaf_nodes"
    ],
    max_depth=locked_parameters[
        "max_depth"
    ],
    min_samples_leaf=locked_parameters[
        "min_samples_leaf"
    ],
    l2_regularization=locked_parameters[
        "l2_regularization"
    ],
    max_bins=locked_parameters[
        "max_bins"
    ],
    categorical_features=None,
    early_stopping=locked_parameters[
        "early_stopping"
    ],
    class_weight=locked_parameters[
        "class_weight"
    ],
    random_state=seed,
    verbose=1,
)


total_started = time.perf_counter()
fit_started = time.perf_counter()

model.fit(
    x_train,
    y_train,
)

fit_seconds = (
    time.perf_counter()
    - fit_started
)


temporary_model_file = (
    model_file.with_suffix(
        model_file.suffix + ".tmp"
    )
)

if temporary_model_file.exists():
    temporary_model_file.unlink()

joblib.dump(
    model,
    temporary_model_file,
)

os.replace(
    temporary_model_file,
    model_file,
)


save_json_atomic(
    execution_state_file,
    {
        "protocol_version":
            "early_lodo_hgb_execution_state_v2_1",

        "status":
            "validating",

        "updated_at":
            utc_now(),

        "identity":
            execution_identity,

        "fit_seconds":
            fit_seconds,

        "iterations":
            int(model.n_iter_),
    },
)


print()
print("Evaluating validation data...")

validation_started = time.perf_counter()

validation_probabilities = (
    model.predict_proba(
        x_validation
    ).astype(
        np.float32,
        copy=False,
    )
)

validation_predictions = (
    model.classes_[
        np.argmax(
            validation_probabilities,
            axis=1,
        )
    ].astype(
        np.int8,
        copy=False,
    )
)

validation_seconds = (
    time.perf_counter()
    - validation_started
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

validation_metrics = (
    metrics_from_confusion(
        validation_confusion
    )
)

validation_log_loss = float(
    log_loss(
        y_validation,
        validation_probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )
)


save_npy_atomic(
    validation_predictions_file,
    validation_predictions,
)

save_npy_atomic(
    validation_probabilities_file,
    validation_probabilities,
)


save_json_atomic(
    execution_state_file,
    {
        "protocol_version":
            "early_lodo_hgb_execution_state_v2_1",

        "status":
            "testing",

        "updated_at":
            utc_now(),

        "identity":
            execution_identity,

        "fit_seconds":
            fit_seconds,

        "validation_seconds":
            validation_seconds,

        "notice":
            "Test evaluation begins once.",
    },
)


print()
print("=" * 86)
print("FINAL TEST EVALUATION")
print("=" * 86)
print(
    "Test data are evaluated exactly once."
)


test_started = time.perf_counter()

test_probabilities = (
    model.predict_proba(
        x_test
    ).astype(
        np.float32,
        copy=False,
    )
)

test_predictions = (
    model.classes_[
        np.argmax(
            test_probabilities,
            axis=1,
        )
    ].astype(
        np.int8,
        copy=False,
    )
)

test_seconds = (
    time.perf_counter()
    - test_started
)


test_predictions_int64 = (
    test_predictions.astype(
        np.int64,
        copy=False,
    )
)

fingerprint_confusion = (
    confusion_from_predictions(
        y_test,
        test_predictions_int64,
    )
)

record_weighted_confusion = (
    confusion_from_predictions(
        y_test,
        test_predictions_int64,
        sample_weights=(
            test_occurrence_count
        ),
    )
)

fingerprint_metrics = (
    metrics_from_confusion(
        fingerprint_confusion
    )
)

record_weighted_metrics = (
    metrics_from_confusion(
        record_weighted_confusion
    )
)

test_log_loss = float(
    log_loss(
        y_test,
        test_probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )
)

record_weighted_log_loss = float(
    log_loss(
        y_test,
        test_probabilities,
        labels=[
            0,
            1,
            2,
        ],
        sample_weight=(
            test_occurrence_count
        ),
    )
)


save_npy_atomic(
    test_predictions_file,
    test_predictions,
)

save_npy_atomic(
    test_probabilities_file,
    test_probabilities,
)


total_seconds = (
    time.perf_counter()
    - total_started
)


probability_row_sums = (
    test_probabilities.sum(
        axis=1,
        dtype=np.float64,
    )
)

validation_probability_row_sums = (
    validation_probabilities.sum(
        axis=1,
        dtype=np.float64,
    )
)


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked"
        and protocol.get(
            "all_checks_passed"
        )
        is True,

    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True,

    "run_id_matches":
        run_spec["run_id"]
        == run_id,

    "model_id_matches":
        model_id
        == "hist_gradient_boosting_b0",

    "locked_parameters_match":
        all(
            expected_parameter_checks.values()
        ),

    "constructor_parameters_supported":
        len(
            missing_constructor_parameters
        )
        == 0,

    "train_row_count_matches":
        len(y_train)
        == int(
            run_spec[
                "train_fingerprint_count"
            ]
        ),

    "validation_row_count_matches":
        len(y_validation)
        == int(
            run_spec[
                "validation_fingerprint_count"
            ]
        ),

    "test_row_count_matches":
        len(y_test)
        == int(
            run_spec[
                "test_fingerprint_count"
            ]
        ),

    "features_are_float32":
        x_train.dtype
        == np.float32
        and x_validation.dtype
        == np.float32
        and x_test.dtype
        == np.float32,

    "features_unscaled":
        True,

    "all_train_classes_present":
        np.array_equal(
            np.unique(y_train),
            np.asarray(
                [0, 1, 2]
            ),
        ),

    "model_classes_valid":
        np.array_equal(
            model.classes_,
            np.asarray(
                [0, 1, 2]
            ),
        ),

    "one_hundred_iterations":
        int(model.n_iter_)
        == locked_parameters[
            "max_iter"
        ]
        == 100,

    "validation_probability_shape":
        validation_probabilities.shape
        == (
            len(y_validation),
            NUM_CLASSES,
        ),

    "test_probability_shape":
        test_probabilities.shape
        == (
            len(y_test),
            NUM_CLASSES,
        ),

    "validation_probabilities_finite":
        bool(
            np.isfinite(
                validation_probabilities
            ).all()
        ),

    "test_probabilities_finite":
        bool(
            np.isfinite(
                test_probabilities
            ).all()
        ),

    "validation_probability_rows_sum_one":
        bool(
            np.allclose(
                validation_probability_row_sums,
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
        ),

    "test_probability_rows_sum_one":
        bool(
            np.allclose(
                probability_row_sums,
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
        ),

    "validation_metric_finite":
        math.isfinite(
            validation_metrics[
                "macro_f1_present_classes"
            ]
        )
        and math.isfinite(
            validation_log_loss
        ),

    "fingerprint_test_metric_finite":
        math.isfinite(
            fingerprint_metrics[
                "macro_f1_present_classes"
            ]
        )
        and math.isfinite(
            test_log_loss
        ),

    "record_weighted_metric_finite":
        math.isfinite(
            record_weighted_metrics[
                "macro_f1_present_classes"
            ]
        )
        and math.isfinite(
            record_weighted_log_loss
        ),

    "test_evaluated_once":
        True,

    "model_file_created":
        model_file.exists(),

    "validation_predictions_created":
        validation_predictions_file.exists()
        and validation_probabilities_file.exists(),

    "test_predictions_created":
        test_predictions_file.exists()
        and test_probabilities_file.exists(),
}


all_checks_passed = all(
    validation_checks.values()
)


run_result = {
    "protocol_version":
        "early_lodo_hgb_result_v2_1",

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "scientific_result":
        True,

    "run": {
        "run_id":
            run_id,

        "held_out_device":
            held_out_device,

        "held_out_device_id":
            int(
                run_spec[
                    "held_out_device_id"
                ]
            ),

        "model_id":
            model_id,

        "seed":
            seed,

        "task":
            run_spec["task"],

        "pipeline":
            run_spec["pipeline"],

        "fold_policy":
            run_spec["fold_policy"],
    },

    "data": {
        "train_fingerprint_count":
            len(y_train),

        "validation_fingerprint_count":
            len(y_validation),

        "test_fingerprint_count":
            len(y_test),

        "train_class_counts":
            np.bincount(
                y_train,
                minlength=3,
            ).tolist(),

        "validation_class_counts":
            np.bincount(
                y_validation,
                minlength=3,
            ).tolist(),

        "test_class_counts":
            np.bincount(
                y_test,
                minlength=3,
            ).tolist(),

        "test_record_weight_sum":
            int(
                test_occurrence_count.sum(
                    dtype=np.uint64
                )
            ),

        "scaled_features":
            False,
    },

    "model": {
        "class":
            (
                "HistGradientBoostingClassifier"
            ),

        "software_version":
            sklearn.__version__,

        "joblib_version":
            joblib.__version__,

        "parameters":
            locked_parameters,

        "random_state":
            seed,

        "completed_iterations":
            int(model.n_iter_),

        "class_order":
            model.classes_.tolist(),
    },

    "validation": {
        "selection_use":
            False,

        "purpose":
            "Reporting only",

        "evaluation_count":
            1,

        "evaluation_seconds":
            validation_seconds,

        "log_loss":
            validation_log_loss,

        "fingerprint_level":
            validation_metrics,
    },

    "test": {
        "evaluation_count":
            1,

        "evaluation_seconds":
            test_seconds,

        "fingerprint_log_loss":
            test_log_loss,

        "record_weighted_log_loss":
            record_weighted_log_loss,

        "fingerprint_level":
            fingerprint_metrics,

        "heldout_record_weighted":
            record_weighted_metrics,
    },

    "runtime": {
        "fit_seconds":
            fit_seconds,

        "validation_seconds":
            validation_seconds,

        "test_seconds":
            test_seconds,

        "total_seconds":
            total_seconds,
    },

    "identity": {
        "protocol_sha256":
            protocol_sha256,

        "run_matrix_sha256":
            run_matrix_sha256,

        "cache_summary_sha256":
            cache_summary_sha256,
    },

    "outputs": {
        "model":
            str(model_file),

        "validation_predictions":
            str(
                validation_predictions_file
            ),

        "validation_probabilities":
            str(
                validation_probabilities_file
            ),

        "test_predictions":
            str(
                test_predictions_file
            ),

        "test_probabilities":
            str(
                test_probabilities_file
            ),

        "result":
            str(result_file),

        "report":
            str(report_file),

        "execution_state":
            str(execution_state_file),
    },

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json_atomic(
    result_file,
    run_result,
)


report_lines = [
    "=" * 86,
    "EARLY LODO HGB SCIENTIFIC RUN RESULT",
    "=" * 86,
    "",
    (
        f"Run id                    : "
        f"{run_id}"
    ),
    (
        f"Held-out device           : "
        f"{held_out_device}"
    ),
    (
        f"Model                     : "
        f"{model_id}"
    ),
    (
        f"Seed                      : "
        f"{seed}"
    ),
    (
        f"Iterations                : "
        f"{model.n_iter_}"
    ),
    (
        f"Fit seconds               : "
        f"{fit_seconds:.3f}"
    ),
    "",
    "VALIDATION — REPORTING ONLY",
    (
        f"Macro-F1                  : "
        f"{validation_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Accuracy                  : "
        f"{validation_metrics['accuracy']:.9f}"
    ),
    (
        f"Log loss                  : "
        f"{validation_log_loss:.9f}"
    ),
    "",
    "FINGERPRINT-LEVEL TEST",
    (
        f"Accuracy                  : "
        f"{fingerprint_metrics['accuracy']:.9f}"
    ),
    (
        f"Balanced accuracy         : "
        f"{fingerprint_metrics['balanced_accuracy']:.9f}"
    ),
    (
        f"Macro-F1                  : "
        f"{fingerprint_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Weighted-F1               : "
        f"{fingerprint_metrics['weighted_f1']:.9f}"
    ),
    (
        f"Log loss                  : "
        f"{test_log_loss:.9f}"
    ),
    "",
    "HELD-OUT RECORD-WEIGHTED TEST",
    (
        f"Accuracy                  : "
        f"{record_weighted_metrics['accuracy']:.9f}"
    ),
    (
        f"Balanced accuracy         : "
        f"{record_weighted_metrics['balanced_accuracy']:.9f}"
    ),
    (
        f"Macro-F1                  : "
        f"{record_weighted_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Weighted-F1               : "
        f"{record_weighted_metrics['weighted_f1']:.9f}"
    ),
    (
        f"Log loss                  : "
        f"{record_weighted_log_loss:.9f}"
    ),
    "",
    "PER-CLASS FINGERPRINT METRICS",
]

for class_name in CLASS_NAMES:
    class_metrics = (
        fingerprint_metrics[
            "per_class"
        ][class_name]
    )

    report_lines.append(
        (
            f"{class_name} | "
            f"support="
            f"{class_metrics['support']} | "
            f"precision="
            f"{class_metrics['precision']} | "
            f"recall="
            f"{class_metrics['recall']} | "
            f"f1="
            f"{class_metrics['f1']} | "
            f"fnr="
            f"{class_metrics['fnr']}"
        )
    )

report_lines.extend(
    [
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
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


save_json_atomic(
    execution_state_file,
    {
        "protocol_version":
            "early_lodo_hgb_execution_state_v2_1",

        "status":
            (
                "completed"
                if all_checks_passed
                else "failed"
            ),

        "completed_at":
            utc_now(),

        "identity":
            execution_identity,

        "fit_seconds":
            fit_seconds,

        "validation_seconds":
            validation_seconds,

        "test_seconds":
            test_seconds,

        "test_evaluation_count":
            1,
    },
)


print()
print("=" * 86)
print("HGB SCIENTIFIC RUN SUMMARY")
print("=" * 86)
print(f"Run id                    : {run_id}")
print(f"Iterations                : {model.n_iter_}")
print(
    f"Validation Macro-F1       : "
    f"{validation_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Test fingerprint Macro-F1 : "
    f"{fingerprint_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Test fingerprint accuracy : "
    f"{fingerprint_metrics['accuracy']:.9f}"
)
print(
    f"Record-weighted Macro-F1  : "
    f"{record_weighted_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Fit seconds               : "
    f"{fit_seconds:.3f}"
)
print(
    f"Total seconds             : "
    f"{total_seconds:.3f}"
)

print()
print("PER-CLASS FINGERPRINT METRICS")

for class_name in CLASS_NAMES:
    class_metrics = (
        fingerprint_metrics[
            "per_class"
        ][class_name]
    )

    print(
        f"{class_name} | "
        f"recall="
        f"{class_metrics['recall']} | "
        f"f1="
        f"{class_metrics['f1']} | "
        f"fnr="
        f"{class_metrics['fnr']}"
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Result       : {result_file}")
print(f"Model        : {model_file}")
print(
    f"Test outputs : "
    f"{test_predictions_file}"
)
print(
    f"Probabilities: "
    f"{test_probabilities_file}"
)
print(f"Report       : {report_file}")

if not all_checks_passed:
    print()
    print(
        "EARLY LODO HGB SCIENTIFIC "
        "RUN FAILED"
    )
    sys.exit(1)

print()
print(
    "EARLY LODO HGB SCIENTIFIC "
    "RUN COMPLETED"
)
