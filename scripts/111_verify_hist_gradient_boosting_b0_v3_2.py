from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path.cwd()

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

CACHE_MANIFEST = FINAL_CACHE / "manifest.json"

FINAL_CACHE_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

LOGISTIC_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "logistic_regression_b0_verification_v3_2_fixed.json"
)

DECISION_TREE_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_verification_v3_2.json"
)

RANDOM_FOREST_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "random_forest_b0_verification_v3_2.json"
)

BASE_PROTOCOL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

BASE_LOCK_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

ADDENDUM = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_v3_2.json"
)

ADDENDUM_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_manifest_v3_2.json"
)

RUNNER_SCRIPT = (
    ROOT
    / "scripts"
    / "110_run_hist_gradient_boosting_b0_v3_2.py"
)

SUMMARY_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "hist_gradient_boosting_b0_summary_v3_2.json"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "hist_gradient_boosting_b0_verification_v3_2.json"
)

MODEL_ID = "hist_gradient_boosting_b0"
EXPECTED_RUN_ID = (
    "hist_gradient_boosting_b0__seed_2026"
)
EXPECTED_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)
EXPECTED_BASE_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_1"
)
EXPECTED_INPUT_SPACE = (
    "canonical_float32_unscaled"
)
EXPECTED_SEED = 2026

EXPECTED_FEATURE_COUNT = 115
EXPECTED_TRAIN_FINGERPRINTS = 1_534_583
EXPECTED_VALIDATION_FINGERPRINTS = 371_797
EXPECTED_TEST_FINGERPRINTS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

CLASS_LABELS = np.asarray(
    [0, 1, 2],
    dtype=np.int64,
)

CLASS_NAMES = (
    "benign",
    "gafgyt",
    "mirai",
)

CHUNK_SIZE = 100_000
METRIC_TOLERANCE = 1e-10

LOCKED_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0001,
    "max_bins": 255,
    "class_weight": "balanced",
    "early_stopping": False,
}


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


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def close_enough(
    observed: float,
    expected: float,
    tolerance: float = (
        METRIC_TOLERANCE
    ),
) -> bool:
    return (
        abs(
            float(observed)
            - float(expected)
        )
        <= tolerance
    )


def metric_view(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: (
        np.ndarray | None
    ),
) -> dict[str, Any]:
    (
        precision,
        recall,
        f1,
        support,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average=None,
            sample_weight=sample_weight,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        sample_weight=sample_weight,
    )

    if sample_weight is not None:
        matrix = np.rint(
            matrix
        ).astype(np.int64)
    else:
        matrix = matrix.astype(
            np.int64
        )

    support_sum = float(
        np.sum(support)
    )

    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
                sample_weight=(
                    sample_weight
                ),
            )
        ),
        "macro_precision": float(
            np.mean(precision)
        ),
        "macro_recall": float(
            np.mean(recall)
        ),
        "balanced_accuracy": float(
            np.mean(recall)
        ),
        "macro_f1": float(
            np.mean(f1)
        ),
        "weighted_f1": float(
            np.average(
                f1,
                weights=support,
            )
            if support_sum > 0
            else 0.0
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                sample_weight=(
                    sample_weight
                ),
            )
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="macro",
                sample_weight=(
                    sample_weight
                ),
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "per_class": {},
    }

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        result["per_class"][
            class_name
        ] = {
            "label": int(
                CLASS_LABELS[index]
            ),
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1[index]
            ),
            "support": float(
                support[index]
            ),
            "false_negative_rate": float(
                1.0 - recall[index]
            ),
        }

    return result


def compare_metric_view(
    stored: dict[str, Any],
    recomputed: dict[str, Any],
) -> dict[str, bool]:
    scalar_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "roc_auc_ovr_macro",
    )

    checks: dict[str, bool] = {
        f"{name}_matches": (
            close_enough(
                recomputed[name],
                stored[name],
            )
        )
        for name in scalar_names
    }

    checks[
        "confusion_matrix_matches"
    ] = (
        stored["confusion_matrix"]
        == recomputed[
            "confusion_matrix"
        ]
    )

    for class_name in CLASS_NAMES:
        for metric_name in (
            "precision",
            "recall",
            "f1",
            "support",
            "false_negative_rate",
        ):
            checks[
                f"{class_name}_"
                f"{metric_name}_matches"
            ] = close_enough(
                recomputed[
                    "per_class"
                ][class_name][metric_name],
                stored[
                    "per_class"
                ][class_name][metric_name],
            )

    return checks


required_paths = [
    FINAL_CACHE,
    CACHE_MANIFEST,
    FINAL_CACHE_VERIFICATION,
    LOGISTIC_VERIFICATION,
    DECISION_TREE_VERIFICATION,
    RANDOM_FOREST_VERIFICATION,
    BASE_PROTOCOL,
    BASE_LOCK_MANIFEST,
    ADDENDUM,
    ADDENDUM_MANIFEST,
    RUNNER_SCRIPT,
    SUMMARY_JSON,
    FINAL_CACHE / "X_train.npy",
    FINAL_CACHE / "X_validation.npy",
    FINAL_CACHE / "y_validation.npy",
    (
        FINAL_CACHE
        / "raw_row_count_validation.npy"
    ),
    FINAL_CACHE / "X_test.npy",
    FINAL_CACHE / "y_test.npy",
    (
        FINAL_CACHE
        / "raw_row_count_test.npy"
    ),
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_REPORT.exists():
    raise FileExistsError(
        "HGB verification report "
        "already exists; refusing "
        f"to overwrite: {OUTPUT_REPORT}"
    )

cache_manifest = read_json(
    CACHE_MANIFEST
)

cache_verification = read_json(
    FINAL_CACHE_VERIFICATION
)

logistic_verification = read_json(
    LOGISTIC_VERIFICATION
)

decision_tree_verification = (
    read_json(
        DECISION_TREE_VERIFICATION
    )
)

random_forest_verification = (
    read_json(
        RANDOM_FOREST_VERIFICATION
    )
)

base_protocol = read_json(
    BASE_PROTOCOL
)

base_lock_manifest = read_json(
    BASE_LOCK_MANIFEST
)

addendum = read_json(
    ADDENDUM
)

addendum_manifest = read_json(
    ADDENDUM_MANIFEST
)

summary = read_json(
    SUMMARY_JSON
)

run_matrix_path = resolve_project_path(
    str(
        base_protocol[
            "run_policy"
        ]["run_matrix_file"]
    )
)

with run_matrix_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    run_matrix_rows = list(
        csv.DictReader(handle)
    )

selected_rows = [
    row
    for row in run_matrix_rows
    if str(
        row.get(
            "model_id",
            "",
        )
    ).strip()
    == MODEL_ID
]

if len(selected_rows) != 1:
    raise RuntimeError(
        "Expected one locked HGB "
        f"run; found {len(selected_rows)}."
    )

matrix_row = selected_rows[0]

run_id = str(
    matrix_row["run_id"]
).strip()

seed = int(
    matrix_row["seed"]
)

run_directory = resolve_project_path(
    str(
        matrix_row[
            "output_directory"
        ]
    )
)

required_run_paths = [
    run_directory,
    run_directory / "metrics.json",
    run_directory / "run_manifest.json",
    run_directory / "run_status.json",
    run_directory / "model.joblib",
    (
        run_directory
        / "model_parameters.npz"
    ),
    (
        run_directory
        / "validation_predictions.npz"
    ),
    (
        run_directory
        / "test_predictions.npz"
    ),
]

for path in required_run_paths:
    if not path.exists():
        raise FileNotFoundError(path)

metrics_path = (
    run_directory / "metrics.json"
)

manifest_path = (
    run_directory
    / "run_manifest.json"
)

status_path = (
    run_directory
    / "run_status.json"
)

metrics = read_json(
    metrics_path
)

manifest = read_json(
    manifest_path
)

status = read_json(
    status_path
)

x_train = np.load(
    FINAL_CACHE / "X_train.npy",
    mmap_mode="r",
)

x_validation = np.load(
    FINAL_CACHE
    / "X_validation.npy",
    mmap_mode="r",
)

y_validation_cache = np.asarray(
    np.load(
        FINAL_CACHE
        / "y_validation.npy",
        mmap_mode="r",
    ),
    dtype=np.int8,
)

raw_validation_cache = np.asarray(
    np.load(
        FINAL_CACHE
        / "raw_row_count_validation.npy",
        mmap_mode="r",
    ),
    dtype=np.int64,
)

x_test = np.load(
    FINAL_CACHE / "X_test.npy",
    mmap_mode="r",
)

y_test_cache = np.asarray(
    np.load(
        FINAL_CACHE / "y_test.npy",
        mmap_mode="r",
    ),
    dtype=np.int8,
)

raw_test_cache = np.asarray(
    np.load(
        FINAL_CACHE
        / "raw_row_count_test.npy",
        mmap_mode="r",
    ),
    dtype=np.int64,
)

global_checks = {
    "cache_manifest_completed": (
        cache_manifest.get("status")
        == "completed"
        and cache_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "cache_verification_passed": (
        cache_verification.get("status")
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "logistic_verification_passed": (
        logistic_verification.get(
            "status"
        )
        == "passed"
        and logistic_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "decision_tree_verification_passed": (
        decision_tree_verification.get(
            "status"
        )
        == "passed"
        and decision_tree_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "random_forest_verification_passed": (
        random_forest_verification.get(
            "status"
        )
        == "passed"
        and random_forest_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "base_protocol_locked": (
        base_protocol.get("status")
        == "locked"
        and base_protocol.get(
            "protocol_version"
        )
        == EXPECTED_BASE_PROTOCOL_VERSION
    ),
    "base_protocol_hash_matches": (
        base_lock_manifest.get(
            "protocol_file_sha256"
        )
        == sha256_file(BASE_PROTOCOL)
    ),
    "run_matrix_hash_matches": (
        base_lock_manifest.get(
            "run_matrix_file_sha256"
        )
        == sha256_file(
            run_matrix_path
        )
    ),
    "addendum_locked": (
        addendum.get("status")
        == "locked"
        and addendum.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "addendum_hash_matches": (
        addendum_manifest.get(
            "addendum_sha256"
        )
        == sha256_file(ADDENDUM)
    ),
    "run_matrix_model_matches": (
        matrix_row.get("model_id")
        == MODEL_ID
    ),
    "run_matrix_run_id_matches": (
        run_id == EXPECTED_RUN_ID
    ),
    "run_matrix_seed_matches": (
        seed == EXPECTED_SEED
    ),
    "run_matrix_input_space_matches": (
        str(
            matrix_row["input_space"]
        ).strip().lower()
        == EXPECTED_INPUT_SPACE
    ),
    "train_shape_matches": (
        x_train.shape
        == (
            EXPECTED_TRAIN_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "validation_shape_matches": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "test_shape_matches": (
        x_test.shape
        == (
            EXPECTED_TEST_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "validation_raw_total_matches": (
        int(
            raw_validation_cache.sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(
            raw_test_cache.sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
    "sklearn_version_is_1_9_0": (
        sklearn.__version__
        == "1.9.0"
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
        "HGB global verification "
        "failed: "
        + ", ".join(
            failed_global_checks
        )
    )

model = joblib.load(
    run_directory / "model.joblib"
)

if not isinstance(
    model,
    HistGradientBoostingClassifier,
):
    raise RuntimeError(
        "Unexpected HGB model type: "
        f"{type(model)!r}"
    )

with np.load(
    run_directory
    / "model_parameters.npz"
) as parameters:
    saved_classes = np.array(
        parameters["classes"],
        copy=True,
    )

    saved_n_iter = int(
        parameters["n_iter"][0]
    )

    saved_n_trees_per_iteration = (
        int(
            parameters[
                "n_trees_per_iteration"
            ][0]
        )
    )

    saved_do_early_stopping = bool(
        parameters[
            "do_early_stopping"
        ][0]
    )

with np.load(
    run_directory
    / "validation_predictions.npz"
) as values:
    validation_saved = {
        name: np.array(
            values[name],
            copy=True,
        )
        for name in (
            "y_true",
            "y_pred",
            "probabilities",
            "raw_row_count",
        )
    }

with np.load(
    run_directory
    / "test_predictions.npz"
) as values:
    test_saved = {
        name: np.array(
            values[name],
            copy=True,
        )
        for name in (
            "y_true",
            "y_pred",
            "probabilities",
            "raw_row_count",
        )
    }

prediction_artifact_checks = {
    "validation_y_matches_cache": (
        np.array_equal(
            validation_saved["y_true"],
            y_validation_cache,
        )
    ),
    "validation_raw_counts_match_cache": (
        np.array_equal(
            validation_saved[
                "raw_row_count"
            ],
            raw_validation_cache,
        )
    ),
    "test_y_matches_cache": (
        np.array_equal(
            test_saved["y_true"],
            y_test_cache,
        )
    ),
    "test_raw_counts_match_cache": (
        np.array_equal(
            test_saved[
                "raw_row_count"
            ],
            raw_test_cache,
        )
    ),
    "validation_probabilities_float32": (
        validation_saved[
            "probabilities"
        ].dtype
        == np.float32
    ),
    "test_probabilities_float32": (
        test_saved[
            "probabilities"
        ].dtype
        == np.float32
    ),
    "validation_probability_shape_matches": (
        validation_saved[
            "probabilities"
        ].shape
        == (
            EXPECTED_VALIDATION_FINGERPRINTS,
            3,
        )
    ),
    "test_probability_shape_matches": (
        test_saved[
            "probabilities"
        ].shape
        == (
            EXPECTED_TEST_FINGERPRINTS,
            3,
        )
    ),
    "validation_probabilities_finite": (
        bool(
            np.isfinite(
                validation_saved[
                    "probabilities"
                ]
            ).all()
        )
    ),
    "test_probabilities_finite": (
        bool(
            np.isfinite(
                test_saved[
                    "probabilities"
                ]
            ).all()
        )
    ),
    "validation_probability_rows_sum_to_one": (
        bool(
            np.allclose(
                validation_saved[
                    "probabilities"
                ].sum(axis=1),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        )
    ),
    "test_probability_rows_sum_to_one": (
        bool(
            np.allclose(
                test_saved[
                    "probabilities"
                ].sum(axis=1),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        )
    ),
}

independent_prediction_checks = {
    "validation_predictions_match_unscaled_model": True,
    "validation_probabilities_match_unscaled_model": True,
    "test_predictions_match_unscaled_model": True,
    "test_probabilities_match_unscaled_model": True,
}

print("=" * 92)
print("HIST GRADIENT BOOSTING B0 INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Verifying saved predictions "
    "against the unscaled cache..."
)

for (
    split_name,
    x_source,
    saved,
) in (
    (
        "validation",
        x_validation,
        validation_saved,
    ),
    (
        "test",
        x_test,
        test_saved,
    ),
):
    for start in range(
        0,
        len(x_source),
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            len(x_source),
        )

        model_input = np.asarray(
            x_source[start:end],
            dtype=np.float32,
        )

        observed_predictions = (
            model.predict(
                model_input
            ).astype(np.int8)
        )

        observed_probabilities = (
            model.predict_proba(
                model_input
            ).astype(np.float32)
        )

        if not np.array_equal(
            observed_predictions,
            saved["y_pred"][
                start:end
            ],
        ):
            independent_prediction_checks[
                f"{split_name}_"
                "predictions_match_"
                "unscaled_model"
            ] = False

        if not np.array_equal(
            observed_probabilities,
            saved["probabilities"][
                start:end
            ],
        ):
            independent_prediction_checks[
                f"{split_name}_"
                "probabilities_match_"
                "unscaled_model"
            ] = False

validation_primary = metric_view(
    y_true=validation_saved[
        "y_true"
    ].astype(np.int64),
    y_pred=validation_saved[
        "y_pred"
    ].astype(np.int64),
    probabilities=validation_saved[
        "probabilities"
    ].astype(
        np.float32,
        copy=False,
    ),
    sample_weight=None,
)

validation_weighted = metric_view(
    y_true=validation_saved[
        "y_true"
    ].astype(np.int64),
    y_pred=validation_saved[
        "y_pred"
    ].astype(np.int64),
    probabilities=validation_saved[
        "probabilities"
    ].astype(
        np.float32,
        copy=False,
    ),
    sample_weight=validation_saved[
        "raw_row_count"
    ].astype(np.int64),
)

test_primary = metric_view(
    y_true=test_saved[
        "y_true"
    ].astype(np.int64),
    y_pred=test_saved[
        "y_pred"
    ].astype(np.int64),
    probabilities=test_saved[
        "probabilities"
    ].astype(
        np.float32,
        copy=False,
    ),
    sample_weight=None,
)

test_weighted = metric_view(
    y_true=test_saved[
        "y_true"
    ].astype(np.int64),
    y_pred=test_saved[
        "y_pred"
    ].astype(np.int64),
    probabilities=test_saved[
        "probabilities"
    ].astype(
        np.float32,
        copy=False,
    ),
    sample_weight=test_saved[
        "raw_row_count"
    ].astype(np.int64),
)

metric_checks: dict[str, bool] = {}

for (
    prefix,
    stored,
    recomputed,
) in (
    (
        "validation_primary",
        metrics["validation"][
            "primary_fingerprint_level"
        ],
        validation_primary,
    ),
    (
        "validation_weighted",
        metrics["validation"][
            "secondary_raw_record_weighted"
        ],
        validation_weighted,
    ),
    (
        "test_primary",
        metrics["test"][
            "primary_fingerprint_level"
        ],
        test_primary,
    ),
    (
        "test_weighted",
        metrics["test"][
            "secondary_raw_record_weighted"
        ],
        test_weighted,
    ),
):
    metric_checks.update(
        {
            f"{prefix}_{name}": passed
            for name, passed
            in compare_metric_view(
                stored=stored,
                recomputed=recomputed,
            ).items()
        }
    )

parameter_checks = {
    "classes_match": (
        np.array_equal(
            model.classes_,
            CLASS_LABELS,
        )
        and np.array_equal(
            saved_classes,
            CLASS_LABELS,
        )
    ),
    "loss_matches": (
        model.loss
        == LOCKED_PARAMETERS["loss"]
    ),
    "learning_rate_matches": (
        close_enough(
            model.learning_rate,
            LOCKED_PARAMETERS[
                "learning_rate"
            ],
            tolerance=0.0,
        )
    ),
    "max_iter_matches": (
        model.max_iter
        == LOCKED_PARAMETERS[
            "max_iter"
        ]
    ),
    "max_leaf_nodes_matches": (
        model.max_leaf_nodes
        == LOCKED_PARAMETERS[
            "max_leaf_nodes"
        ]
    ),
    "max_depth_matches": (
        model.max_depth
        is LOCKED_PARAMETERS[
            "max_depth"
        ]
    ),
    "min_samples_leaf_matches": (
        model.min_samples_leaf
        == LOCKED_PARAMETERS[
            "min_samples_leaf"
        ]
    ),
    "l2_regularization_matches": (
        close_enough(
            model.l2_regularization,
            LOCKED_PARAMETERS[
                "l2_regularization"
            ],
            tolerance=0.0,
        )
    ),
    "max_bins_matches": (
        model.max_bins
        == LOCKED_PARAMETERS[
            "max_bins"
        ]
    ),
    "class_weight_matches": (
        model.class_weight
        == LOCKED_PARAMETERS[
            "class_weight"
        ]
    ),
    "early_stopping_parameter_matches": (
        model.early_stopping
        is LOCKED_PARAMETERS[
            "early_stopping"
        ]
    ),
    "random_state_matches": (
        model.random_state
        == EXPECTED_SEED
    ),
    "n_iter_is_100": (
        int(model.n_iter_)
        == 100
        == saved_n_iter
        == int(
            metrics["training"][
                "n_iter"
            ]
        )
    ),
    "n_trees_per_iteration_is_3": (
        int(
            model.n_trees_per_iteration_
        )
        == 3
        == saved_n_trees_per_iteration
        == int(
            metrics["training"][
                "n_trees_per_iteration"
            ]
        )
    ),
    "early_stopping_not_used": (
        bool(
            model.do_early_stopping_
        )
        is False
        and saved_do_early_stopping
        is False
        and metrics["training"][
            "do_early_stopping"
        ]
        is False
    ),
    "internal_iteration_count_matches": (
        len(model._predictors)
        == int(model.n_iter_)
    ),
    "internal_total_tree_count_is_300": (
        sum(
            len(iteration)
            for iteration
            in model._predictors
        )
        == 300
    ),
}

artifact_records = manifest.get(
    "artifacts"
)

if not isinstance(
    artifact_records,
    list,
):
    raise RuntimeError(
        "HGB artifact inventory "
        "is missing."
    )

artifact_checks = []

for item in artifact_records:
    artifact_path = (
        run_directory
        / str(
            item["relative_path"]
        )
    )

    exists = artifact_path.exists()

    observed_size = (
        int(
            artifact_path.stat().st_size
        )
        if exists
        else None
    )

    observed_hash = (
        sha256_file(
            artifact_path
        )
        if exists
        else None
    )

    artifact_checks.append(
        {
            "relative_path": str(
                item["relative_path"]
            ),
            "exists": exists,
            "size_matches": (
                observed_size
                == int(
                    item["size_bytes"]
                )
            ),
            "sha256_matches": (
                observed_hash
                == str(
                    item["sha256"]
                )
            ),
        }
    )

fragility_recomputed = {
    "macro_f1_below_0_85": (
        test_primary["macro_f1"]
        < 0.85
    ),
    "gafgyt_fnr_above_0_40": (
        test_primary["per_class"][
            "gafgyt"
        ]["false_negative_rate"]
        > 0.40
    ),
    "mirai_fnr_above_0_40": (
        test_primary["per_class"][
            "mirai"
        ]["false_negative_rate"]
        > 0.40
    ),
}

fragility_recomputed[
    "triggered"
] = any(
    fragility_recomputed.values()
)

summary_checks = {
    "summary_completed": (
        summary.get("status")
        == "completed"
    ),
    "summary_integrity_passed": (
        summary.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "summary_run_id_matches": (
        summary.get("run_id")
        == EXPECTED_RUN_ID
    ),
    "summary_model_id_matches": (
        summary.get("model_id")
        == MODEL_ID
    ),
    "summary_seed_matches": (
        int(summary["seed"])
        == EXPECTED_SEED
    ),
    "summary_protocol_matches": (
        summary.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "summary_input_space_matches": (
        summary.get("input_space")
        == EXPECTED_INPUT_SPACE
    ),
    "summary_test_count_is_one": (
        int(
            summary[
                "test_evaluation_count"
            ]
        )
        == 1
    ),
    "summary_test_not_used_for_selection": (
        summary.get(
            "test_used_for_selection"
        )
        is False
    ),
    "summary_metrics_hash_matches": (
        summary.get(
            "metrics_sha256"
        )
        == sha256_file(
            metrics_path
        )
    ),
    "summary_manifest_hash_matches": (
        summary.get(
            "run_manifest_sha256"
        )
        == sha256_file(
            manifest_path
        )
    ),
    "summary_runner_hash_matches": (
        summary.get(
            "runner_script_sha256"
        )
        == sha256_file(
            RUNNER_SCRIPT
        )
    ),
    "summary_macro_f1_matches": (
        close_enough(
            summary[
                "test_fingerprint_macro_f1"
            ],
            test_primary["macro_f1"],
        )
    ),
    "summary_accuracy_matches": (
        close_enough(
            summary[
                "test_fingerprint_accuracy"
            ],
            test_primary["accuracy"],
        )
    ),
    "summary_weighted_macro_f1_matches": (
        close_enough(
            summary[
                "test_raw_weighted_macro_f1"
            ],
            test_weighted[
                "macro_f1"
            ],
        )
    ),
    "summary_gafgyt_fnr_matches": (
        close_enough(
            summary[
                "test_gafgyt_fnr"
            ],
            test_primary[
                "per_class"
            ]["gafgyt"][
                "false_negative_rate"
            ],
        )
    ),
    "summary_mirai_fnr_matches": (
        close_enough(
            summary[
                "test_mirai_fnr"
            ],
            test_primary[
                "per_class"
            ]["mirai"][
                "false_negative_rate"
            ],
        )
    ),
}

run_checks = {
    "manifest_completed": (
        manifest.get("status")
        == "completed"
    ),
    "status_completed": (
        status.get("status")
        == "completed"
        and status.get("stage")
        == "completed"
    ),
    "run_id_matches": (
        metrics.get("run_id")
        == EXPECTED_RUN_ID
        and manifest.get("run_id")
        == EXPECTED_RUN_ID
        and status.get("run_id")
        == EXPECTED_RUN_ID
    ),
    "model_id_matches": (
        metrics.get("model_id")
        == MODEL_ID
        and manifest.get("model_id")
        == MODEL_ID
        and status.get("model_id")
        == MODEL_ID
    ),
    "seed_matches": (
        int(metrics["seed"])
        == EXPECTED_SEED
        and int(manifest["seed"])
        == EXPECTED_SEED
        and int(status["seed"])
        == EXPECTED_SEED
    ),
    "protocol_version_matches": (
        metrics.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
        and manifest.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "input_space_exact_unscaled": (
        manifest.get("input_space")
        == EXPECTED_INPUT_SPACE
    ),
    "input_parser_policy_exact": (
        manifest.get(
            "input_space_parser_policy"
        )
        == (
            "exact locked-token equality; "
            "no substring matching"
        )
    ),
    "training_count_matches": (
        int(
            metrics["training"][
                "fingerprint_count"
            ]
        )
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "training_weight_is_none": (
        metrics["training"].get(
            "sample_weight"
        )
        is None
        and manifest.get(
            "fit_sample_weight"
        )
        is None
    ),
    "test_evaluation_count_is_one": (
        int(
            metrics[
                "test_evaluation_count"
            ]
        )
        == 1
        and int(
            manifest[
                "expected_test_evaluation_count"
            ]
        )
        == 1
        and int(
            manifest[
                "observed_test_evaluation_count"
            ]
        )
        == 1
    ),
    "fragility_not_triggered": (
        metrics["fragility_gate"].get(
            "triggered"
        )
        is False
        and manifest[
            "fragility_gate"
        ].get("triggered")
        is False
        and status.get(
            "fragility_gate_triggered"
        )
        is False
        and fragility_recomputed[
            "triggered"
        ]
        is False
    ),
    "runner_hash_matches": (
        manifest.get(
            "runner_script_sha256"
        )
        == sha256_file(
            RUNNER_SCRIPT
        )
    ),
    "run_matrix_hash_matches": (
        manifest.get(
            "run_matrix_sha256"
        )
        == sha256_file(
            run_matrix_path
        )
    ),
    "cache_manifest_hash_matches": (
        manifest.get(
            "cache_manifest_sha256"
        )
        == sha256_file(
            CACHE_MANIFEST
        )
    ),
    "all_prediction_artifact_checks_pass": (
        all(
            prediction_artifact_checks.values()
        )
    ),
    "all_unscaled_prediction_checks_pass": (
        all(
            independent_prediction_checks.values()
        )
    ),
    "all_metric_recomputations_match": (
        all(
            metric_checks.values()
        )
    ),
    "all_parameter_checks_pass": (
        all(
            parameter_checks.values()
        )
    ),
    "all_summary_checks_pass": (
        all(
            summary_checks.values()
        )
    ),
    "all_artifacts_exist": (
        all(
            item["exists"]
            for item
            in artifact_checks
        )
    ),
    "all_artifact_sizes_match": (
        all(
            item["size_matches"]
            for item
            in artifact_checks
        )
    ),
    "all_artifact_hashes_match": (
        all(
            item["sha256_matches"]
            for item
            in artifact_checks
        )
    ),
    "manifest_integrity_passed": (
        manifest.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
}

failed_run_checks = [
    name
    for name, passed
    in run_checks.items()
    if not passed
]

final_checks = {
    "all_global_checks_pass": (
        all(
            global_checks.values()
        )
    ),
    "all_run_checks_pass": (
        all(
            run_checks.values()
        )
    ),
    "fragility_gate_not_triggered": (
        fragility_recomputed[
            "triggered"
        ]
        is False
    ),
}

failed_final_checks = [
    name
    for name, passed
    in final_checks.items()
    if not passed
]

all_failed_checks = [
    *failed_run_checks,
    *failed_final_checks,
]

report = {
    "status": (
        "passed"
        if not all_failed_checks
        else "failed"
    ),
    "generated_at_utc": utc_now(),
    "run_id": EXPECTED_RUN_ID,
    "model_id": MODEL_ID,
    "seed": EXPECTED_SEED,
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "input_space": (
        EXPECTED_INPUT_SPACE
    ),
    "independent_unscaled_inference": (
        all(
            independent_prediction_checks.values()
        )
    ),
    "test_evaluation_count": 1,
    "test_fingerprint_macro_f1": (
        test_primary["macro_f1"]
    ),
    "test_fingerprint_accuracy": (
        test_primary["accuracy"]
    ),
    "test_raw_weighted_macro_f1": (
        test_weighted["macro_f1"]
    ),
    "test_gafgyt_fnr": (
        test_primary["per_class"][
            "gafgyt"
        ]["false_negative_rate"]
    ),
    "test_mirai_fnr": (
        test_primary["per_class"][
            "mirai"
        ]["false_negative_rate"]
    ),
    "artifact_count_verified": (
        len(artifact_checks)
    ),
    "global_checks": global_checks,
    "prediction_artifact_checks": (
        prediction_artifact_checks
    ),
    "independent_prediction_checks": (
        independent_prediction_checks
    ),
    "metric_checks": metric_checks,
    "parameter_checks": (
        parameter_checks
    ),
    "summary_checks": (
        summary_checks
    ),
    "artifact_checks": (
        artifact_checks
    ),
    "run_checks": run_checks,
    "final_checks": final_checks,
    "failed_checks": (
        all_failed_checks
    ),
    "free_disk_gib": (
        shutil.disk_usage(ROOT).free
        / (1024**3)
    ),
    "all_checks_passed": (
        not all_failed_checks
    ),
}

atomic_json(
    OUTPUT_REPORT,
    report,
)

print()
print("=" * 92)
print("HIST GRADIENT BOOSTING B0 VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Run ID                         : "
    f"{EXPECTED_RUN_ID}"
)
print(
    "Input space                    : "
    f"{EXPECTED_INPUT_SPACE}"
)
print(
    "Independent unscaled inference : "
    f"{all(independent_prediction_checks.values())}"
)
print(
    "Iterations verified            : "
    f"{model.n_iter_}"
)
print(
    "Trees verified                 : "
    f"{sum(len(value) for value in model._predictors)}"
)
print(
    "Test evaluation count          : 1"
)
print(
    "Test fingerprint Macro-F1      : "
    f"{test_primary['macro_f1']:.9f}"
)
print(
    "Test fingerprint accuracy      : "
    f"{test_primary['accuracy']:.9f}"
)
print(
    "Test raw-weighted Macro-F1     : "
    f"{test_weighted['macro_f1']:.9f}"
)
print(
    "Test Gafgyt FNR                : "
    f"{test_primary['per_class']['gafgyt']['false_negative_rate']:.9f}"
)
print(
    "Test Mirai FNR                 : "
    f"{test_primary['per_class']['mirai']['false_negative_rate']:.9f}"
)
print(
    "Artifacts verified             : "
    f"{len(artifact_checks)}"
)
print(
    "Fragility gate triggered       : False"
)
print(
    "Verification report            : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed              : "
    f"{not all_failed_checks}"
)

if all_failed_checks:
    print(
        "Failed checks                  : "
        f"{all_failed_checks}"
    )

    raise RuntimeError(
        "HGB independent "
        "verification failed."
    )

print(
    "HIST GRADIENT BOOSTING B0 VERIFICATION COMPLETED"
)
