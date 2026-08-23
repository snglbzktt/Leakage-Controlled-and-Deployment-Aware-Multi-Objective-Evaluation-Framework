from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


ROOT = Path.cwd()

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

FINAL_CACHE_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_final_cache_verification_v3_2.json"
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

INVALIDATION_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_invalidation_v3_2.json"
)

INVALID_RUN_ARCHIVE = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "invalid_runs"
    / "decision_tree_b0_input_space_parser_bug_v3_2"
)

AGGREGATE_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_aggregate_v3_2.json"
)

AGGREGATE_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_runs_v3_2.csv"
)

RUNNER_SCRIPT = (
    ROOT
    / "scripts"
    / "106_run_decision_tree_b0_five_seeds_v3_2_fixed.py"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_verification_v3_2.json"
)

MODEL_ID = "decision_tree_b0"
EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
EXPECTED_BASE_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"
EXPECTED_INPUT_SPACE = "canonical_float32_unscaled"

EXPECTED_SEEDS = [42, 123, 2026, 3407, 8192]
EXPECTED_RUN_COUNT = 5
EXPECTED_FEATURE_COUNT = 115
EXPECTED_TRAIN_FINGERPRINTS = 1_534_583
EXPECTED_VALIDATION_FINGERPRINTS = 371_797
EXPECTED_TEST_FINGERPRINTS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
CHUNK_SIZE = 100_000
METRIC_TOLERANCE = 1e-10

LOCKED_PARAMETERS = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": 20,
    "min_samples_split": 40,
    "min_samples_leaf": 20,
    "max_features": None,
    "class_weight": "balanced",
    "ccp_alpha": 0.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def close_enough(
    observed: float,
    expected: float,
    tolerance: float = METRIC_TOLERANCE,
) -> bool:
    return abs(float(observed) - float(expected)) <= tolerance


def metric_view(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    precision, recall, f1, support = (
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
        matrix = np.rint(matrix).astype(np.int64)
    else:
        matrix = matrix.astype(np.int64)

    support_sum = float(np.sum(support))

    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            np.average(f1, weights=support)
            if support_sum > 0
            else 0.0
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                sample_weight=sample_weight,
            )
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="macro",
                sample_weight=sample_weight,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": {},
    }

    for index, class_name in enumerate(CLASS_NAMES):
        result["per_class"][class_name] = {
            "label": int(CLASS_LABELS[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
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
        f"{name}_matches": close_enough(
            recomputed[name],
            stored[name],
        )
        for name in scalar_names
    }

    checks["confusion_matrix_matches"] = (
        stored["confusion_matrix"]
        == recomputed["confusion_matrix"]
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
                f"{class_name}_{metric_name}_matches"
            ] = close_enough(
                recomputed["per_class"][class_name][
                    metric_name
                ],
                stored["per_class"][class_name][
                    metric_name
                ],
            )

    return checks


def aggregate_numeric(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


required_paths = [
    FINAL_CACHE,
    FINAL_CACHE_VERIFICATION,
    BASE_PROTOCOL,
    BASE_LOCK_MANIFEST,
    ADDENDUM,
    ADDENDUM_MANIFEST,
    INVALIDATION_REPORT,
    INVALID_RUN_ARCHIVE,
    INVALID_RUN_ARCHIVE / "invalidation_manifest.json",
    AGGREGATE_JSON,
    AGGREGATE_CSV,
    RUNNER_SCRIPT,
    FINAL_CACHE / "X_validation.npy",
    FINAL_CACHE / "y_validation.npy",
    FINAL_CACHE / "raw_row_count_validation.npy",
    FINAL_CACHE / "X_test.npy",
    FINAL_CACHE / "y_test.npy",
    FINAL_CACHE / "raw_row_count_test.npy",
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_REPORT.exists():
    raise FileExistsError(
        "Decision Tree verification report already exists; "
        f"refusing to overwrite: {OUTPUT_REPORT}"
    )

cache_verification = read_json(
    FINAL_CACHE_VERIFICATION
)
base_protocol = read_json(BASE_PROTOCOL)
base_lock_manifest = read_json(
    BASE_LOCK_MANIFEST
)
addendum = read_json(ADDENDUM)
addendum_manifest = read_json(
    ADDENDUM_MANIFEST
)
invalidation_report = read_json(
    INVALIDATION_REPORT
)
invalidation_manifest = read_json(
    INVALID_RUN_ARCHIVE
    / "invalidation_manifest.json"
)
aggregate = read_json(AGGREGATE_JSON)

run_matrix_path = resolve_project_path(
    str(
        base_protocol["run_policy"][
            "run_matrix_file"
        ]
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
    if str(row.get("model_id", "")).strip()
    == MODEL_ID
]

selected_rows.sort(
    key=lambda row: int(row["seed"])
)

selected_seeds = [
    int(row["seed"])
    for row in selected_rows
]

with AGGREGATE_CSV.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    aggregate_csv_rows = list(
        csv.DictReader(handle)
    )

aggregate_csv_rows.sort(
    key=lambda row: int(row["seed"])
)

x_validation = np.load(
    FINAL_CACHE / "X_validation.npy",
    mmap_mode="r",
)

y_validation_cache = np.asarray(
    np.load(
        FINAL_CACHE / "y_validation.npy",
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
        FINAL_CACHE / "raw_row_count_test.npy",
        mmap_mode="r",
    ),
    dtype=np.int64,
)

global_checks = {
    "final_cache_verification_passed": (
        cache_verification.get("status")
        == "passed"
        and cache_verification.get(
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
        == sha256_file(run_matrix_path)
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
    "invalidation_completed": (
        invalidation_report.get("status")
        == "completed"
        and invalidation_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "invalid_runs_quarantined": (
        invalidation_manifest.get("status")
        == "invalidated_and_quarantined"
        and invalidation_manifest.get(
            "rerun_required"
        )
        is True
    ),
    "aggregate_completed": (
        aggregate.get("status")
        == "completed"
        and aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "aggregate_protocol_version_matches": (
        aggregate.get("protocol_version")
        == EXPECTED_PROTOCOL_VERSION
    ),
    "aggregate_run_count_is_five": (
        int(aggregate["run_count"])
        == EXPECTED_RUN_COUNT
    ),
    "aggregate_seed_set_matches": (
        [int(value) for value in aggregate["seeds"]]
        == EXPECTED_SEEDS
    ),
    "aggregate_input_space_exact_unscaled": (
        aggregate.get("input_space")
        == EXPECTED_INPUT_SPACE
    ),
    "aggregate_parser_policy_exact": (
        aggregate.get(
            "input_space_parser_policy"
        )
        == (
            "exact locked-token equality; "
            "no substring matching"
        )
    ),
    "aggregate_csv_hash_matches": (
        aggregate.get(
            "aggregate_csv_sha256"
        )
        == sha256_file(AGGREGATE_CSV)
    ),
    "aggregate_runner_hash_matches": (
        aggregate.get(
            "runner_script_sha256"
        )
        == sha256_file(RUNNER_SCRIPT)
    ),
    "run_matrix_selected_count_is_five": (
        len(selected_rows)
        == EXPECTED_RUN_COUNT
    ),
    "run_matrix_seed_set_matches": (
        selected_seeds
        == EXPECTED_SEEDS
    ),
    "run_matrix_input_space_exact_unscaled": all(
        str(row["input_space"]).strip().lower()
        == EXPECTED_INPUT_SPACE
        for row in selected_rows
    ),
    "aggregate_csv_count_is_five": (
        len(aggregate_csv_rows)
        == EXPECTED_RUN_COUNT
    ),
    "temporary_scaled_matrix_absent": (
        not (
            ROOT
            / "results"
            / "v2"
            / "tabular_baselines"
            / "decision_tree_b0_scaled_train_float32.tmp.npy"
        ).exists()
    ),
    "validation_cache_shape_matches": (
        x_validation.shape
        == (
            EXPECTED_VALIDATION_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "test_cache_shape_matches": (
        x_test.shape
        == (
            EXPECTED_TEST_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "validation_raw_total_matches": (
        int(raw_validation_cache.sum())
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(raw_test_cache.sum())
        == EXPECTED_TEST_RAW_ROWS
    ),
    "sklearn_version_is_1_9_0": (
        sklearn.__version__ == "1.9.0"
    ),
}

failed_global_checks = [
    name
    for name, passed in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "Decision Tree global verification failed: "
        + ", ".join(failed_global_checks)
    )

run_reports: list[dict[str, Any]] = []

print("=" * 92)
print("DECISION TREE B0 INDEPENDENT VERIFICATION")
print("=" * 92)

for run_number, row in enumerate(
    selected_rows,
    start=1,
):
    run_id = str(row["run_id"]).strip()
    seed = int(row["seed"])
    run_directory = resolve_project_path(
        str(row["output_directory"])
    )

    required_run_paths = [
        run_directory,
        run_directory / "metrics.json",
        run_directory / "run_manifest.json",
        run_directory / "run_status.json",
        run_directory / "model.joblib",
        run_directory / "model_parameters.npz",
        run_directory / "validation_predictions.npz",
        run_directory / "test_predictions.npz",
    ]

    for path in required_run_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = read_json(
        run_directory / "metrics.json"
    )
    manifest = read_json(
        run_directory / "run_manifest.json"
    )
    status = read_json(
        run_directory / "run_status.json"
    )

    model = joblib.load(
        run_directory / "model.joblib"
    )

    if not isinstance(
        model,
        DecisionTreeClassifier,
    ):
        raise RuntimeError(
            f"Unexpected model type for {run_id}."
        )

    with np.load(
        run_directory / "model_parameters.npz"
    ) as parameters:
        saved_classes = np.array(
            parameters["classes"],
            copy=True,
        )
        saved_importances = np.array(
            parameters["feature_importances"],
            copy=True,
        )
        saved_node_count = int(
            parameters["tree_node_count"][0]
        )
        saved_max_depth = int(
            parameters["tree_max_depth"][0]
        )
        saved_n_leaves = int(
            parameters["n_leaves"][0]
        )

    with np.load(
        run_directory
        / "validation_predictions.npz"
    ) as values:
        validation_saved = {
            name: np.array(values[name], copy=True)
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
            name: np.array(values[name], copy=True)
            for name in (
                "y_true",
                "y_pred",
                "probabilities",
                "raw_row_count",
            )
        }

    prediction_checks = {
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
            test_saved["probabilities"].shape
            == (
                EXPECTED_TEST_FINGERPRINTS,
                3,
            )
        ),
        "validation_probabilities_float32": (
            validation_saved[
                "probabilities"
            ].dtype
            == np.float32
        ),
        "test_probabilities_float32": (
            test_saved["probabilities"].dtype
            == np.float32
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
                    test_saved["probabilities"]
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

    print(
        f"[{run_number}/{EXPECTED_RUN_COUNT}] "
        f"{run_id} | verifying unscaled predictions...",
        flush=True,
    )

    for split_name, x_source, saved in (
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
                saved["y_pred"][start:end],
            ):
                independent_prediction_checks[
                    f"{split_name}_predictions_match_unscaled_model"
                ] = False

            if not np.array_equal(
                observed_probabilities,
                saved["probabilities"][start:end],
            ):
                independent_prediction_checks[
                    f"{split_name}_probabilities_match_unscaled_model"
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
        ].astype(np.float32, copy=False),
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
        ].astype(np.float32, copy=False),
        sample_weight=validation_saved[
            "raw_row_count"
        ].astype(np.int64),
    )

    test_primary = metric_view(
        y_true=test_saved["y_true"].astype(
            np.int64
        ),
        y_pred=test_saved["y_pred"].astype(
            np.int64
        ),
        probabilities=test_saved[
            "probabilities"
        ].astype(np.float32, copy=False),
        sample_weight=None,
    )

    test_weighted = metric_view(
        y_true=test_saved["y_true"].astype(
            np.int64
        ),
        y_pred=test_saved["y_pred"].astype(
            np.int64
        ),
        probabilities=test_saved[
            "probabilities"
        ].astype(np.float32, copy=False),
        sample_weight=test_saved[
            "raw_row_count"
        ].astype(np.int64),
    )

    metric_checks: dict[str, bool] = {}

    for prefix, stored, recomputed in (
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

    artifact_records = manifest.get(
        "artifacts"
    )

    if not isinstance(
        artifact_records,
        list,
    ):
        raise RuntimeError(
            f"Artifact inventory missing for {run_id}."
        )

    artifact_checks = []

    for item in artifact_records:
        artifact_path = (
            run_directory
            / str(item["relative_path"])
        )

        exists = artifact_path.exists()
        observed_size = (
            int(artifact_path.stat().st_size)
            if exists
            else None
        )
        observed_hash = (
            sha256_file(artifact_path)
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
                    == int(item["size_bytes"])
                ),
                "sha256_matches": (
                    observed_hash
                    == str(item["sha256"])
                ),
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
        "feature_importance_shape_matches": (
            model.feature_importances_.shape
            == (EXPECTED_FEATURE_COUNT,)
            and saved_importances.shape
            == (EXPECTED_FEATURE_COUNT,)
        ),
        "feature_importances_match": (
            np.array_equal(
                model.feature_importances_,
                saved_importances,
            )
        ),
        "node_count_matches": (
            int(model.tree_.node_count)
            == saved_node_count
            == int(
                metrics["training"][
                    "tree_node_count"
                ]
            )
        ),
        "max_depth_matches": (
            int(model.tree_.max_depth)
            == saved_max_depth
            == int(
                metrics["training"][
                    "tree_max_depth"
                ]
            )
        ),
        "leaf_count_matches": (
            int(model.get_n_leaves())
            == saved_n_leaves
            == int(
                metrics["training"][
                    "n_leaves"
                ]
            )
        ),
        "criterion_matches": (
            model.criterion
            == LOCKED_PARAMETERS["criterion"]
        ),
        "splitter_matches": (
            model.splitter
            == LOCKED_PARAMETERS["splitter"]
        ),
        "max_depth_parameter_matches": (
            model.max_depth
            == LOCKED_PARAMETERS["max_depth"]
        ),
        "min_samples_split_matches": (
            model.min_samples_split
            == LOCKED_PARAMETERS[
                "min_samples_split"
            ]
        ),
        "min_samples_leaf_matches": (
            model.min_samples_leaf
            == LOCKED_PARAMETERS[
                "min_samples_leaf"
            ]
        ),
        "max_features_matches": (
            model.max_features
            is LOCKED_PARAMETERS["max_features"]
        ),
        "class_weight_matches": (
            model.class_weight
            == LOCKED_PARAMETERS[
                "class_weight"
            ]
        ),
        "ccp_alpha_matches": (
            close_enough(
                model.ccp_alpha,
                LOCKED_PARAMETERS["ccp_alpha"],
                tolerance=0.0,
            )
        ),
        "random_state_matches": (
            model.random_state == seed
        ),
    }

    fragility_recomputed = {
        "macro_f1_below_0_85": (
            test_primary["macro_f1"] < 0.85
        ),
        "gafgyt_fnr_above_0_40": (
            test_primary["per_class"]["gafgyt"][
                "false_negative_rate"
            ]
            > 0.40
        ),
        "mirai_fnr_above_0_40": (
            test_primary["per_class"]["mirai"][
                "false_negative_rate"
            ]
            > 0.40
        ),
    }

    fragility_recomputed["triggered"] = any(
        fragility_recomputed.values()
    )

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
            metrics.get("run_id") == run_id
            and manifest.get("run_id") == run_id
            and status.get("run_id") == run_id
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
            int(metrics.get("seed")) == seed
            and int(manifest.get("seed")) == seed
            and int(status.get("seed")) == seed
        ),
        "protocol_version_matches": (
            metrics.get("protocol_version")
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
        "fragility_gate_not_triggered": (
            metrics["fragility_gate"].get(
                "triggered"
            )
            is False
            and manifest["fragility_gate"].get(
                "triggered"
            )
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
            == sha256_file(RUNNER_SCRIPT)
        ),
        "prior_invalidation_hash_matches": (
            manifest.get(
                "prior_invalidation_report_sha256"
            )
            == sha256_file(
                INVALIDATION_REPORT
            )
        ),
        "all_prediction_artifact_checks_pass": all(
            prediction_checks.values()
        ),
        "all_unscaled_prediction_checks_pass": all(
            independent_prediction_checks.values()
        ),
        "all_metric_recomputations_match": all(
            metric_checks.values()
        ),
        "all_parameter_checks_pass": all(
            parameter_checks.values()
        ),
        "all_artifacts_exist": all(
            item["exists"]
            for item in artifact_checks
        ),
        "all_artifact_sizes_match": all(
            item["size_matches"]
            for item in artifact_checks
        ),
        "all_artifact_hashes_match": all(
            item["sha256_matches"]
            for item in artifact_checks
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
        for name, passed in run_checks.items()
        if not passed
    ]

    if failed_run_checks:
        raise RuntimeError(
            f"Decision Tree verification failed "
            f"for {run_id}: {failed_run_checks}"
        )

    run_reports.append(
        {
            "run_id": run_id,
            "seed": seed,
            "run_directory": str(
                run_directory
            ),
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
            "tree_node_count": int(
                model.tree_.node_count
            ),
            "tree_max_depth": int(
                model.tree_.max_depth
            ),
            "n_leaves": int(
                model.get_n_leaves()
            ),
            "artifact_count": len(
                artifact_checks
            ),
            "checks": run_checks,
            "all_checks_passed": True,
        }
    )

    print(
        f"  Macro-F1={test_primary['macro_f1']:.9f} | "
        f"raw-weighted={test_weighted['macro_f1']:.9f} | "
        f"all checks=True"
    )

aggregate_recomputed = {
    "test_fingerprint_macro_f1": aggregate_numeric(
        [
            row["test_fingerprint_macro_f1"]
            for row in run_reports
        ]
    ),
    "test_fingerprint_accuracy": aggregate_numeric(
        [
            row["test_fingerprint_accuracy"]
            for row in run_reports
        ]
    ),
    "test_raw_weighted_macro_f1": aggregate_numeric(
        [
            row["test_raw_weighted_macro_f1"]
            for row in run_reports
        ]
    ),
    "test_gafgyt_fnr": aggregate_numeric(
        [
            row["test_gafgyt_fnr"]
            for row in run_reports
        ]
    ),
    "test_mirai_fnr": aggregate_numeric(
        [
            row["test_mirai_fnr"]
            for row in run_reports
        ]
    ),
}

aggregate_metric_checks: dict[str, bool] = {}

for metric_name, recomputed_values in (
    aggregate_recomputed.items()
):
    stored_values = aggregate[
        "aggregate_metrics"
    ][metric_name]

    for statistic_name in (
        "mean",
        "std_population",
        "minimum",
        "maximum",
    ):
        aggregate_metric_checks[
            f"{metric_name}_{statistic_name}_matches"
        ] = close_enough(
            recomputed_values[
                statistic_name
            ],
            stored_values[
                statistic_name
            ],
        )

csv_checks = {
    "csv_run_ids_match": (
        [row["run_id"] for row in aggregate_csv_rows]
        == [row["run_id"] for row in run_reports]
    ),
    "csv_seeds_match": (
        [int(row["seed"]) for row in aggregate_csv_rows]
        == EXPECTED_SEEDS
    ),
    "csv_macro_f1_values_match": all(
        close_enough(
            float(csv_row[
                "test_fingerprint_macro_f1"
            ]),
            report_row[
                "test_fingerprint_macro_f1"
            ],
        )
        for csv_row, report_row
        in zip(
            aggregate_csv_rows,
            run_reports,
        )
    ),
    "csv_raw_weighted_values_match": all(
        close_enough(
            float(csv_row[
                "test_raw_weighted_macro_f1"
            ]),
            report_row[
                "test_raw_weighted_macro_f1"
            ],
        )
        for csv_row, report_row
        in zip(
            aggregate_csv_rows,
            run_reports,
        )
    ),
}

final_checks = {
    "all_global_checks_pass": all(
        global_checks.values()
    ),
    "all_run_checks_pass": all(
        row["all_checks_passed"]
        for row in run_reports
    ),
    "all_aggregate_metrics_match": all(
        aggregate_metric_checks.values()
    ),
    "all_aggregate_csv_checks_pass": all(
        csv_checks.values()
    ),
    "fragility_not_triggered_in_any_run": (
        aggregate.get(
            "fragility_gate_triggered_in_any_run"
        )
        is False
    ),
}

failed_final_checks = [
    name
    for name, passed in final_checks.items()
    if not passed
]

report = {
    "status": (
        "passed"
        if not failed_final_checks
        else "failed"
    ),
    "generated_at_utc": utc_now(),
    "model_id": MODEL_ID,
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "input_space": (
        EXPECTED_INPUT_SPACE
    ),
    "input_space_parser_policy": (
        "exact locked-token equality; "
        "no substring matching"
    ),
    "run_count": len(run_reports),
    "seeds": EXPECTED_SEEDS,
    "runs": run_reports,
    "recomputed_aggregate_metrics": (
        aggregate_recomputed
    ),
    "stored_aggregate_metrics": (
        aggregate["aggregate_metrics"]
    ),
    "global_checks": global_checks,
    "aggregate_metric_checks": (
        aggregate_metric_checks
    ),
    "aggregate_csv_checks": (
        csv_checks
    ),
    "final_checks": final_checks,
    "failed_checks": (
        failed_final_checks
    ),
    "free_disk_gib": (
        shutil.disk_usage(ROOT).free
        / (1024**3)
    ),
    "all_checks_passed": (
        not failed_final_checks
    ),
}

atomic_json(OUTPUT_REPORT, report)

macro = aggregate_recomputed[
    "test_fingerprint_macro_f1"
]

weighted = aggregate_recomputed[
    "test_raw_weighted_macro_f1"
]

print()
print("=" * 92)
print("DECISION TREE B0 VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs verified                  : "
    f"{len(run_reports)}"
)
print(
    "Seeds                          : "
    f"{EXPECTED_SEEDS}"
)
print(
    "Input space                    : "
    f"{EXPECTED_INPUT_SPACE}"
)
print(
    "Independent unscaled inference : True"
)
print(
    "Mean test fingerprint Macro-F1 : "
    f"{macro['mean']:.9f}"
)
print(
    "Std test fingerprint Macro-F1  : "
    f"{macro['std_population']:.9f}"
)
print(
    "Min / max fingerprint Macro-F1 : "
    f"{macro['minimum']:.9f} / "
    f"{macro['maximum']:.9f}"
)
print(
    "Mean raw-weighted Macro-F1     : "
    f"{weighted['mean']:.9f}"
)
print(
    "Fragility in any run           : False"
)
print(
    "Verification report            : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed              : "
    f"{not failed_final_checks}"
)

if failed_final_checks:
    print(
        "Failed checks                  : "
        f"{failed_final_checks}"
    )

    raise RuntimeError(
        "Decision Tree aggregate verification failed."
    )

print(
    "DECISION TREE B0 VERIFICATION COMPLETED"
)
