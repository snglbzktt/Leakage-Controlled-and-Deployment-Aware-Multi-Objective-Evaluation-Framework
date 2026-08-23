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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path.cwd()

RUN_DIRECTORY = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
    / "logistic_regression_b0__seed_2026"
)

METRICS_PATH = RUN_DIRECTORY / "metrics.json"
RUN_MANIFEST_PATH = RUN_DIRECTORY / "run_manifest.json"
RUN_STATUS_PATH = RUN_DIRECTORY / "run_status.json"
MODEL_PATH = RUN_DIRECTORY / "model.joblib"
MODEL_PARAMETERS_PATH = RUN_DIRECTORY / "model_parameters.npz"

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

CACHE_MANIFEST_PATH = FINAL_CACHE / "manifest.json"

FINAL_CACHE_VERIFICATION_PATH = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

BASE_PROTOCOL_PATH = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

BASE_LOCK_MANIFEST_PATH = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

ADDENDUM_PATH = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_v3_2.json"
)

ADDENDUM_MANIFEST_PATH = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_manifest_v3_2.json"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "logistic_regression_b0_verification_v3_2_fixed.json"
)

EXPECTED_RUN_ID = "logistic_regression_b0__seed_2026"
EXPECTED_MODEL_ID = "logistic_regression_b0"
EXPECTED_SEED = 2026
EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
EXPECTED_BASE_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"
EXPECTED_STATUS = "completed_with_convergence_warning"

EXPECTED_FEATURE_COUNT = 115
EXPECTED_TRAIN_FINGERPRINTS = 1_534_583
EXPECTED_VALIDATION_FINGERPRINTS = 371_797
EXPECTED_TEST_FINGERPRINTS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")

LOCKED_PARAMETERS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": "balanced",
}

METRIC_TOLERANCE = 1e-10


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


def directory_size_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


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


def close_enough(
    observed: float,
    expected: float,
    tolerance: float = METRIC_TOLERANCE,
) -> bool:
    return abs(float(observed) - float(expected)) <= tolerance


def recompute_metric_view(
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


def load_prediction_file(
    split_name: str,
) -> dict[str, np.ndarray]:
    path = (
        RUN_DIRECTORY
        / f"{split_name}_predictions.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as values:
        required_keys = {
            "y_true",
            "y_pred",
            "probabilities",
            "raw_row_count",
        }

        if set(values.files) != required_keys:
            raise RuntimeError(
                f"Unexpected keys in {path.name}: "
                f"{values.files}"
            )

        return {
            name: np.array(
                values[name],
                copy=True,
            )
            for name in required_keys
        }


required_paths = [
    RUN_DIRECTORY,
    METRICS_PATH,
    RUN_MANIFEST_PATH,
    RUN_STATUS_PATH,
    MODEL_PATH,
    MODEL_PARAMETERS_PATH,
    CACHE_MANIFEST_PATH,
    FINAL_CACHE_VERIFICATION_PATH,
    BASE_PROTOCOL_PATH,
    BASE_LOCK_MANIFEST_PATH,
    ADDENDUM_PATH,
    ADDENDUM_MANIFEST_PATH,
    RUN_DIRECTORY / "validation_predictions.npz",
    RUN_DIRECTORY / "test_predictions.npz",
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_REPORT.exists():
    raise FileExistsError(
        "Verification report already exists; "
        f"refusing to overwrite: {OUTPUT_REPORT}"
    )

metrics = read_json(METRICS_PATH)
run_manifest = read_json(RUN_MANIFEST_PATH)
run_status = read_json(RUN_STATUS_PATH)
cache_manifest = read_json(CACHE_MANIFEST_PATH)
cache_verification = read_json(
    FINAL_CACHE_VERIFICATION_PATH
)
base_protocol = read_json(BASE_PROTOCOL_PATH)
base_lock_manifest = read_json(
    BASE_LOCK_MANIFEST_PATH
)
addendum = read_json(ADDENDUM_PATH)
addendum_manifest = read_json(
    ADDENDUM_MANIFEST_PATH
)

model = joblib.load(MODEL_PATH)

if not isinstance(model, LogisticRegression):
    raise RuntimeError(
        "Loaded model is not LogisticRegression."
    )

with np.load(MODEL_PARAMETERS_PATH) as model_parameters:
    classes = np.array(
        model_parameters["classes"],
        copy=True,
    )
    coefficients = np.array(
        model_parameters["coef"],
        copy=True,
    )
    intercept = np.array(
        model_parameters["intercept"],
        copy=True,
    )
    n_iter = np.array(
        model_parameters["n_iter"],
        copy=True,
    )

validation_values = load_prediction_file(
    "validation"
)

test_values = load_prediction_file("test")

prediction_shape_checks = {
    "validation_y_true_shape": (
        validation_values["y_true"].shape
        == (EXPECTED_VALIDATION_FINGERPRINTS,)
    ),
    "validation_y_pred_shape": (
        validation_values["y_pred"].shape
        == (EXPECTED_VALIDATION_FINGERPRINTS,)
    ),
    "validation_probabilities_shape": (
        validation_values["probabilities"].shape
        == (
            EXPECTED_VALIDATION_FINGERPRINTS,
            3,
        )
    ),
    "validation_raw_count_shape": (
        validation_values["raw_row_count"].shape
        == (EXPECTED_VALIDATION_FINGERPRINTS,)
    ),
    "test_y_true_shape": (
        test_values["y_true"].shape
        == (EXPECTED_TEST_FINGERPRINTS,)
    ),
    "test_y_pred_shape": (
        test_values["y_pred"].shape
        == (EXPECTED_TEST_FINGERPRINTS,)
    ),
    "test_probabilities_shape": (
        test_values["probabilities"].shape
        == (EXPECTED_TEST_FINGERPRINTS, 3)
    ),
    "test_raw_count_shape": (
        test_values["raw_row_count"].shape
        == (EXPECTED_TEST_FINGERPRINTS,)
    ),
    "validation_raw_rows_match": (
        int(
            validation_values[
                "raw_row_count"
            ].astype(np.int64).sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_rows_match": (
        int(
            test_values[
                "raw_row_count"
            ].astype(np.int64).sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
    "all_validation_probabilities_finite": (
        bool(
            np.isfinite(
                validation_values[
                    "probabilities"
                ]
            ).all()
        )
    ),
    "all_test_probabilities_finite": (
        bool(
            np.isfinite(
                test_values["probabilities"]
            ).all()
        )
    ),
    "validation_probability_rows_sum_to_one": (
        bool(
            np.allclose(
                validation_values[
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
                test_values[
                    "probabilities"
                ].sum(axis=1),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        )
    ),
}

failed_prediction_checks = [
    name
    for name, passed
    in prediction_shape_checks.items()
    if not passed
]

if failed_prediction_checks:
    raise RuntimeError(
        "Prediction artifact validation failed: "
        + ", ".join(failed_prediction_checks)
    )

validation_primary_recomputed = (
    recompute_metric_view(
        y_true=validation_values[
            "y_true"
        ].astype(np.int64),
        y_pred=validation_values[
            "y_pred"
        ].astype(np.int64),
        probabilities=validation_values[
            "probabilities"
        ].astype(np.float32, copy=False),
        sample_weight=None,
    )
)

validation_weighted_recomputed = (
    recompute_metric_view(
        y_true=validation_values[
            "y_true"
        ].astype(np.int64),
        y_pred=validation_values[
            "y_pred"
        ].astype(np.int64),
        probabilities=validation_values[
            "probabilities"
        ].astype(np.float32, copy=False),
        sample_weight=validation_values[
            "raw_row_count"
        ].astype(np.int64),
    )
)

test_primary_recomputed = recompute_metric_view(
    y_true=test_values["y_true"].astype(
        np.int64
    ),
    y_pred=test_values["y_pred"].astype(
        np.int64
    ),
    probabilities=test_values[
        "probabilities"
    ].astype(np.float32, copy=False),
    sample_weight=None,
)

test_weighted_recomputed = recompute_metric_view(
    y_true=test_values["y_true"].astype(
        np.int64
    ),
    y_pred=test_values["y_pred"].astype(
        np.int64
    ),
    probabilities=test_values[
        "probabilities"
    ].astype(np.float32, copy=False),
    sample_weight=test_values[
        "raw_row_count"
    ].astype(np.int64),
)

metric_checks = {}

metric_checks.update(
    {
        f"validation_primary_{name}": passed
        for name, passed in compare_metric_view(
            stored=metrics["validation"][
                "primary_fingerprint_level"
            ],
            recomputed=(
                validation_primary_recomputed
            ),
        ).items()
    }
)

metric_checks.update(
    {
        f"validation_weighted_{name}": passed
        for name, passed in compare_metric_view(
            stored=metrics["validation"][
                "secondary_raw_record_weighted"
            ],
            recomputed=(
                validation_weighted_recomputed
            ),
        ).items()
    }
)

metric_checks.update(
    {
        f"test_primary_{name}": passed
        for name, passed in compare_metric_view(
            stored=metrics["test"][
                "primary_fingerprint_level"
            ],
            recomputed=test_primary_recomputed,
        ).items()
    }
)

metric_checks.update(
    {
        f"test_weighted_{name}": passed
        for name, passed in compare_metric_view(
            stored=metrics["test"][
                "secondary_raw_record_weighted"
            ],
            recomputed=test_weighted_recomputed,
        ).items()
    }
)

manifest_artifacts = run_manifest.get(
    "artifacts"
)

if not isinstance(
    manifest_artifacts,
    list,
):
    raise RuntimeError(
        "Run manifest artifact inventory is missing."
    )

artifact_checks: list[
    dict[str, Any]
] = []

for item in manifest_artifacts:
    path = (
        RUN_DIRECTORY
        / str(item["relative_path"])
    )

    exists = path.exists()
    observed_size = (
        int(path.stat().st_size)
        if exists
        else None
    )
    observed_hash = (
        sha256_file(path)
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
    "model_classes_match": (
        np.array_equal(
            model.classes_,
            CLASS_LABELS,
        )
        and np.array_equal(
            classes,
            CLASS_LABELS,
        )
    ),
    "model_coefficient_shape_matches": (
        model.coef_.shape
        == (3, EXPECTED_FEATURE_COUNT)
        and coefficients.shape
        == (3, EXPECTED_FEATURE_COUNT)
    ),
    "model_intercept_shape_matches": (
        model.intercept_.shape == (3,)
        and intercept.shape == (3,)
    ),
    "model_n_iter_is_200": (
        model.n_iter_.shape == (1,)
        and int(model.n_iter_[0]) == 200
        and n_iter.shape == (1,)
        and int(n_iter[0]) == 200
    ),
    "saved_coefficients_match_model": (
        np.array_equal(
            coefficients,
            model.coef_,
        )
    ),
    "saved_intercept_matches_model": (
        np.array_equal(
            intercept,
            model.intercept_,
        )
    ),
    "model_penalty_matches": (
        model.penalty
        == LOCKED_PARAMETERS["penalty"]
    ),
    "model_C_matches": (
        close_enough(
            model.C,
            LOCKED_PARAMETERS["C"],
            tolerance=0.0,
        )
    ),
    "model_solver_matches": (
        model.solver
        == LOCKED_PARAMETERS["solver"]
    ),
    "model_max_iter_matches": (
        model.max_iter
        == LOCKED_PARAMETERS["max_iter"]
    ),
    "model_tol_matches": (
        close_enough(
            model.tol,
            LOCKED_PARAMETERS["tol"],
            tolerance=0.0,
        )
    ),
    "model_fit_intercept_matches": (
        model.fit_intercept
        is LOCKED_PARAMETERS["fit_intercept"]
    ),
    "model_class_weight_matches": (
        model.class_weight
        == LOCKED_PARAMETERS["class_weight"]
    ),
    "model_random_state_matches": (
        model.random_state == EXPECTED_SEED
    ),
}

warnings_list = metrics["training"].get(
    "warnings",
    []
)

convergence_warning_present = any(
    str(item.get("category"))
    == "ConvergenceWarning"
    for item in warnings_list
)

stored_test_primary = metrics["test"][
    "primary_fingerprint_level"
]

stored_gafgyt_fnr = stored_test_primary[
    "attack_false_negative_rates"
]["gafgyt"]

stored_mirai_fnr = stored_test_primary[
    "attack_false_negative_rates"
]["mirai"]

recomputed_fragility = {
    "macro_f1_below_0_85": (
        test_primary_recomputed["macro_f1"]
        < 0.85
    ),
    "gafgyt_fnr_above_0_40": (
        test_primary_recomputed[
            "per_class"
        ]["gafgyt"]["false_negative_rate"]
        > 0.40
    ),
    "mirai_fnr_above_0_40": (
        test_primary_recomputed[
            "per_class"
        ]["mirai"]["false_negative_rate"]
        > 0.40
    ),
}

recomputed_fragility["triggered"] = any(
    recomputed_fragility.values()
)

checks = {
    "run_manifest_status_matches_warning": (
        run_manifest.get("status")
        == EXPECTED_STATUS
    ),
    "run_status_matches_warning": (
        run_status.get("status")
        == EXPECTED_STATUS
    ),
    "run_status_stage_completed": (
        run_status.get("stage")
        == "completed"
    ),
    "run_id_matches": (
        metrics.get("run_id")
        == EXPECTED_RUN_ID
        and run_manifest.get("run_id")
        == EXPECTED_RUN_ID
        and run_status.get("run_id")
        == EXPECTED_RUN_ID
    ),
    "model_id_matches": (
        metrics.get("model_id")
        == EXPECTED_MODEL_ID
        and run_manifest.get("model_id")
        == EXPECTED_MODEL_ID
        and run_status.get("model_id")
        == EXPECTED_MODEL_ID
    ),
    "seed_matches": (
        int(metrics.get("seed"))
        == EXPECTED_SEED
        and int(run_manifest.get("seed"))
        == EXPECTED_SEED
        and int(run_status.get("seed"))
        == EXPECTED_SEED
    ),
    "protocol_version_matches": (
        metrics.get("protocol_version")
        == EXPECTED_PROTOCOL_VERSION
        and run_manifest.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "final_cache_verification_passed": (
        cache_verification.get("status")
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "cache_manifest_completed": (
        cache_manifest.get("status")
        == "completed"
        and cache_manifest.get(
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
        == sha256_file(BASE_PROTOCOL_PATH)
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
        == sha256_file(ADDENDUM_PATH)
    ),
    "training_fingerprint_count_matches": (
        int(
            metrics["training"][
                "fingerprint_count"
            ]
        )
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "training_sample_weight_is_none": (
        metrics["training"].get(
            "sample_weight"
        )
        is None
        and run_manifest.get(
            "fit_sample_weight"
        )
        is None
    ),
    "training_converged_false": (
        metrics["training"].get(
            "converged"
        )
        is False
        and run_manifest.get(
            "converged"
        )
        is False
        and run_status.get(
            "converged"
        )
        is False
    ),
    "convergence_warning_present": (
        convergence_warning_present
    ),
    "test_evaluation_count_is_one": (
        int(
            metrics[
                "test_evaluation_count"
            ]
        )
        == 1
        and int(
            run_manifest[
                "observed_test_evaluation_count"
            ]
        )
        == 1
        and int(
            run_manifest[
                "expected_test_evaluation_count"
            ]
        )
        == 1
    ),
    "temporary_scaled_train_removed": (
        not (
            RUN_DIRECTORY
            / "X_train_scaled_float32.npy"
        ).exists()
    ),
    "fragility_gate_not_triggered": (
        metrics["fragility_gate"].get(
            "triggered"
        )
        is False
        and run_manifest[
            "fragility_gate"
        ].get("triggered")
        is False
        and run_status.get(
            "fragility_gate_triggered"
        )
        is False
        and recomputed_fragility[
            "triggered"
        ]
        is False
    ),
    "stored_gafgyt_fnr_matches_recomputed": (
        close_enough(
            stored_gafgyt_fnr,
            test_primary_recomputed[
                "per_class"
            ]["gafgyt"][
                "false_negative_rate"
            ],
        )
    ),
    "stored_mirai_fnr_matches_recomputed": (
        close_enough(
            stored_mirai_fnr,
            test_primary_recomputed[
                "per_class"
            ]["mirai"][
                "false_negative_rate"
            ],
        )
    ),
    "all_prediction_checks_pass": all(
        prediction_shape_checks.values()
    ),
    "all_metric_recomputations_match": all(
        metric_checks.values()
    ),
    "all_parameter_checks_pass": all(
        parameter_checks.values()
    ),
    "all_manifest_artifacts_exist": all(
        item["exists"]
        for item in artifact_checks
    ),
    "all_manifest_artifact_sizes_match": all(
        item["size_matches"]
        for item in artifact_checks
    ),
    "all_manifest_artifact_hashes_match": all(
        item["sha256_matches"]
        for item in artifact_checks
    ),
    "run_manifest_integrity_passed": (
        run_manifest.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "sklearn_version_is_1_9_0": (
        sklearn.__version__ == "1.9.0"
    ),
}

failed_checks = [
    name
    for name, passed in checks.items()
    if not passed
]

report = {
    "status": (
        "passed"
        if not failed_checks
        else "failed"
    ),
    "generated_at_utc": utc_now(),
    "run_id": EXPECTED_RUN_ID,
    "model_id": EXPECTED_MODEL_ID,
    "seed": EXPECTED_SEED,
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "scientific_status": (
        "completed_with_convergence_warning"
    ),
    "interpretation": (
        "The locked max_iter=200 run reached the "
        "iteration limit and is retained exactly as "
        "pre-specified. Metrics and artifacts are valid, "
        "but coefficient convergence was not established."
    ),
    "verification_precision_policy": (
        "Recompute probability-based metrics from the saved "
        "float32 probability artifacts without casting them to "
        "float64, matching the locked runner metric path exactly."
    ),
    "run_directory": str(RUN_DIRECTORY),
    "run_directory_size_bytes": (
        directory_size_bytes(RUN_DIRECTORY)
    ),
    "free_disk_gib": (
        shutil.disk_usage(ROOT).free
        / (1024**3)
    ),
    "stored_metrics": {
        "test_fingerprint_macro_f1": (
            stored_test_primary["macro_f1"]
        ),
        "test_fingerprint_accuracy": (
            stored_test_primary["accuracy"]
        ),
        "test_raw_weighted_macro_f1": (
            metrics["test"][
                "secondary_raw_record_weighted"
            ]["macro_f1"]
        ),
        "test_gafgyt_fnr": (
            stored_gafgyt_fnr
        ),
        "test_mirai_fnr": (
            stored_mirai_fnr
        ),
    },
    "recomputed_metrics": {
        "validation_primary": (
            validation_primary_recomputed
        ),
        "validation_raw_weighted": (
            validation_weighted_recomputed
        ),
        "test_primary": (
            test_primary_recomputed
        ),
        "test_raw_weighted": (
            test_weighted_recomputed
        ),
    },
    "recomputed_fragility_gate": (
        recomputed_fragility
    ),
    "prediction_checks": (
        prediction_shape_checks
    ),
    "metric_checks": metric_checks,
    "parameter_checks": (
        parameter_checks
    ),
    "artifact_checks": (
        artifact_checks
    ),
    "checks": checks,
    "failed_checks": failed_checks,
    "all_checks_passed": (
        not failed_checks
    ),
}

atomic_json(OUTPUT_REPORT, report)

print("=" * 92)
print("LOGISTIC REGRESSION B0 INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Run ID                     : "
    f"{EXPECTED_RUN_ID}"
)
print(
    "Scientific status          : "
    f"{EXPECTED_STATUS}"
)
print(
    "Converged                  : False"
)
print(
    "Observed n_iter            : "
    f"{model.n_iter_.tolist()}"
)
print(
    "Test evaluation count      : "
    f"{metrics['test_evaluation_count']}"
)
print(
    "Test fingerprint Macro-F1  : "
    f"{test_primary_recomputed['macro_f1']:.9f}"
)
print(
    "Test fingerprint accuracy  : "
    f"{test_primary_recomputed['accuracy']:.9f}"
)
print(
    "Test raw-weighted Macro-F1 : "
    f"{test_weighted_recomputed['macro_f1']:.9f}"
)
print(
    "Test Gafgyt FNR            : "
    f"{test_primary_recomputed['per_class']['gafgyt']['false_negative_rate']:.9f}"
)
print(
    "Test Mirai FNR             : "
    f"{test_primary_recomputed['per_class']['mirai']['false_negative_rate']:.9f}"
)
print(
    "Fragility gate triggered   : "
    f"{recomputed_fragility['triggered']}"
)
print(
    "Run artifacts verified     : "
    f"{len(artifact_checks):,}"
)
print(
    "Run directory size         : "
    f"{directory_size_bytes(RUN_DIRECTORY) / (1024**2):.3f} MiB"
)
print(
    "Free disk                  : "
    f"{shutil.disk_usage(ROOT).free / (1024**3):.3f} GiB"
)

print()
print("VALIDATION CHECKS")

for name, passed in checks.items():
    print(f"  {name}: {passed}")

print()
print(
    "Verification report        : "
    f"{OUTPUT_REPORT}"
)
print(
    "All checks passed          : "
    f"{not failed_checks}"
)

if failed_checks:
    print(
        "Failed checks              : "
        f"{failed_checks}"
    )

    raise RuntimeError(
        "Logistic-regression independent "
        "verification failed."
    )

print(
    "LOGISTIC REGRESSION B0 VERIFICATION COMPLETED"
)
