from __future__ import annotations

import os

# Keep BLAS/OpenMP resource use predictable on the 16 GB host.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import csv
import hashlib
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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

FINAL_VERIFICATION = (
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

EXPECTED_MODEL_ID = "logistic_regression_b0"
EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
EXPECTED_BASE_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"

FEATURE_COUNT = 115
CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
CHUNK_SIZE = 100_000
MINIMUM_FREE_DISK_GIB = 1.5

LOCKED_PARAMETERS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": "balanced",
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


def normalize_none_token(value: str) -> bool:
    return value.strip().lower() in {
        "",
        "none",
        "null",
        "false",
        "0",
        "no",
        "not_used",
        "unused",
    }


def load_selected_run(
    base_protocol: dict[str, Any],
) -> tuple[dict[str, str], Path]:
    run_matrix_path = resolve_project_path(
        str(
            base_protocol["run_policy"][
                "run_matrix_file"
            ]
        )
    )

    if not run_matrix_path.exists():
        raise FileNotFoundError(run_matrix_path)

    with run_matrix_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))

    selected = [
        row
        for row in rows
        if str(row.get("model_id", "")).strip()
        == EXPECTED_MODEL_ID
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one logistic-regression run; "
            f"found {len(selected)}."
        )

    return selected[0], run_matrix_path


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

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    support_sum = float(np.sum(support))

    weighted_f1 = float(
        np.average(f1, weights=support)
        if support_sum > 0
        else 0.0
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

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
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

    result["attack_false_negative_rates"] = {
        "gafgyt": result["per_class"]["gafgyt"][
            "false_negative_rate"
        ],
        "mirai": result["per_class"]["mirai"][
            "false_negative_rate"
        ],
    }

    return result


def save_confusion_csv(
    path: Path,
    matrix: list[list[int]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["true\\pred", *CLASS_NAMES]
        )

        for class_name, row in zip(
            CLASS_NAMES,
            matrix,
        ):
            writer.writerow([class_name, *row])


def save_per_class_csv(
    path: Path,
    primary: dict[str, Any],
    weighted: dict[str, Any],
) -> None:
    fieldnames = [
        "class_name",
        "label",
        "fingerprint_precision",
        "fingerprint_recall",
        "fingerprint_f1",
        "fingerprint_support",
        "fingerprint_fnr",
        "raw_weighted_precision",
        "raw_weighted_recall",
        "raw_weighted_f1",
        "raw_weighted_support",
        "raw_weighted_fnr",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for class_name in CLASS_NAMES:
            p = primary["per_class"][class_name]
            w = weighted["per_class"][class_name]

            writer.writerow(
                {
                    "class_name": class_name,
                    "label": p["label"],
                    "fingerprint_precision": p["precision"],
                    "fingerprint_recall": p["recall"],
                    "fingerprint_f1": p["f1"],
                    "fingerprint_support": p["support"],
                    "fingerprint_fnr": p[
                        "false_negative_rate"
                    ],
                    "raw_weighted_precision": w["precision"],
                    "raw_weighted_recall": w["recall"],
                    "raw_weighted_f1": w["f1"],
                    "raw_weighted_support": w["support"],
                    "raw_weighted_fnr": w[
                        "false_negative_rate"
                    ],
                }
            )


def evaluate_split(
    model: LogisticRegression,
    split_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    output_directory: Path,
) -> dict[str, Any]:
    x = np.load(
        FINAL_CACHE / f"X_{split_name}.npy",
        mmap_mode="r",
    )

    y = np.asarray(
        np.load(
            FINAL_CACHE / f"y_{split_name}.npy",
            mmap_mode="r",
        ),
        dtype=np.int64,
    )

    raw_counts = np.asarray(
        np.load(
            FINAL_CACHE
            / f"raw_row_count_{split_name}.npy",
            mmap_mode="r",
        ),
        dtype=np.int64,
    )

    predictions = np.empty(
        len(x),
        dtype=np.int8,
    )

    probabilities = np.empty(
        (len(x), len(CLASS_LABELS)),
        dtype=np.float32,
    )

    started = time.perf_counter()

    for start in range(0, len(x), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(x))

        raw = np.asarray(
            x[start:end],
            dtype=np.float64,
        )

        scaled = (
            (raw - scaler_mean) / scaler_scale
        ).astype(np.float32)

        if not np.isfinite(scaled).all():
            raise RuntimeError(
                f"Non-finite scaled values in {split_name}."
            )

        predictions[start:end] = model.predict(
            scaled
        ).astype(np.int8)

        probabilities[start:end] = (
            model.predict_proba(scaled).astype(
                np.float32
            )
        )

    primary = metric_view(
        y_true=y,
        y_pred=predictions,
        probabilities=probabilities,
        sample_weight=None,
    )

    raw_weighted = metric_view(
        y_true=y,
        y_pred=predictions,
        probabilities=probabilities,
        sample_weight=raw_counts,
    )

    np.savez_compressed(
        output_directory
        / f"{split_name}_predictions.npz",
        y_true=y.astype(np.int8),
        y_pred=predictions,
        probabilities=probabilities,
        raw_row_count=raw_counts,
    )

    save_confusion_csv(
        output_directory
        / f"{split_name}_confusion_fingerprint.csv",
        primary["confusion_matrix"],
    )

    save_confusion_csv(
        output_directory
        / f"{split_name}_confusion_raw_weighted.csv",
        raw_weighted["confusion_matrix"],
    )

    save_per_class_csv(
        output_directory
        / f"{split_name}_per_class_metrics.csv",
        primary=primary,
        weighted=raw_weighted,
    )

    return {
        "split": split_name,
        "fingerprint_count": int(len(x)),
        "represented_raw_row_count": int(
            raw_counts.sum()
        ),
        "primary_fingerprint_level": primary,
        "secondary_raw_record_weighted": raw_weighted,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }


def artifact_inventory(
    output_directory: Path,
) -> list[dict[str, Any]]:
    excluded = {
        "run_manifest.json",
        "run_status.json",
        "X_train_scaled_float32.npy",
    }

    records: list[dict[str, Any]] = []

    for path in sorted(output_directory.iterdir()):
        if (
            not path.is_file()
            or path.name in excluded
        ):
            continue

        records.append(
            {
                "relative_path": path.name,
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(path),
            }
        )

    return records


def main() -> None:
    started = time.perf_counter()

    required_paths = [
        FINAL_CACHE,
        CACHE_MANIFEST,
        FINAL_VERIFICATION,
        BASE_PROTOCOL,
        BASE_LOCK_MANIFEST,
        ADDENDUM,
        ADDENDUM_MANIFEST,
        FINAL_CACHE / "scaler_mean.npy",
        FINAL_CACHE / "scaler_scale.npy",
        FINAL_CACHE / "X_train.npy",
        FINAL_CACHE / "y_train.npy",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    verification = read_json(FINAL_VERIFICATION)
    cache_manifest = read_json(CACHE_MANIFEST)
    base_protocol = read_json(BASE_PROTOCOL)
    base_lock_manifest = read_json(
        BASE_LOCK_MANIFEST
    )
    addendum = read_json(ADDENDUM)
    addendum_manifest = read_json(
        ADDENDUM_MANIFEST
    )

    selected_run, run_matrix_path = (
        load_selected_run(base_protocol)
    )

    run_id = str(selected_run["run_id"]).strip()
    seed = int(selected_run["seed"])
    output_directory = resolve_project_path(
        str(selected_run["output_directory"])
    )

    fit_sample_weight = str(
        selected_run.get(
            "fit_sample_weight",
            "",
        )
    )

    input_space = str(
        selected_run.get("input_space", "")
    ).strip().lower()

    expected_test_evaluations = int(
        selected_run.get(
            "expected_test_evaluation_count",
            "1",
        )
    )

    checks = {
        "final_verification_passed": (
            verification.get("status")
            == "passed"
            and verification.get(
                "all_checks_passed"
            )
            is True
        ),
        "cache_manifest_completed": (
            cache_manifest.get("status")
            == "completed"
        ),
        "cache_protocol_version_matches": (
            cache_manifest.get(
                "protocol_version"
            )
            == EXPECTED_PROTOCOL_VERSION
        ),
        "base_protocol_locked": (
            base_protocol.get("status")
            == "locked"
        ),
        "base_protocol_version_matches": (
            base_protocol.get(
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
        ),
        "addendum_version_matches": (
            addendum.get(
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
        "selected_model_is_logistic_regression": (
            selected_run.get("model_id")
            == EXPECTED_MODEL_ID
        ),
        "fit_sample_weight_is_none": (
            normalize_none_token(
                fit_sample_weight
            )
        ),
        "input_space_is_scaled_float32": (
            "float32" in input_space
            and (
                "scaled" in input_space
                or "standard" in input_space
            )
        ),
        "expected_test_evaluation_count_is_one": (
            expected_test_evaluations == 1
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

    if failed_checks:
        raise RuntimeError(
            "Logistic-regression preflight failed: "
            + ", ".join(failed_checks)
        )

    if output_directory.exists():
        raise FileExistsError(
            "Run output directory already exists; "
            f"refusing to overwrite: {output_directory}"
        )

    free_disk_gib = (
        shutil.disk_usage(ROOT).free
        / (1024**3)
    )

    if free_disk_gib < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(
            "Insufficient free disk space: "
            f"{free_disk_gib:.3f} GiB < "
            f"{MINIMUM_FREE_DISK_GIB:.3f} GiB"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    run_status_path = (
        output_directory / "run_status.json"
    )

    atomic_json(
        run_status_path,
        {
            "status": "running",
            "stage": "preflight_complete",
            "started_at_utc": utc_now(),
            "run_id": run_id,
            "model_id": EXPECTED_MODEL_ID,
            "seed": seed,
        },
    )

    print("=" * 92)
    print("PHASE 4 CLASSICAL BASELINE: LOGISTIC REGRESSION B0")
    print("=" * 92)
    print(f"Run ID          : {run_id}")
    print(f"Seed            : {seed}")
    print(f"Output directory: {output_directory}")
    print(f"Free disk       : {free_disk_gib:.3f} GiB")
    print(f"Train unit      : fingerprint")
    print(f"Fit weighting   : none")
    print(f"Input space     : {input_space}")
    print(f"Test evaluations: 1")

    scaler_mean = np.load(
        FINAL_CACHE / "scaler_mean.npy"
    )

    scaler_scale = np.load(
        FINAL_CACHE / "scaler_scale.npy"
    )

    if (
        scaler_mean.shape != (FEATURE_COUNT,)
        or scaler_scale.shape
        != (FEATURE_COUNT,)
        or scaler_mean.dtype != np.float64
        or scaler_scale.dtype != np.float64
        or not np.isfinite(scaler_mean).all()
        or not np.isfinite(scaler_scale).all()
        or np.any(scaler_scale <= 0)
    ):
        raise RuntimeError(
            "Invalid locked scaler artifacts."
        )

    x_train = np.load(
        FINAL_CACHE / "X_train.npy",
        mmap_mode="r",
    )

    y_train = np.asarray(
        np.load(
            FINAL_CACHE / "y_train.npy",
            mmap_mode="r",
        ),
        dtype=np.int64,
    )

    temporary_scaled_train = (
        output_directory
        / "X_train_scaled_float32.npy"
    )

    print()
    print("Materializing scaled float32 training matrix...")

    scaled_train = np.lib.format.open_memmap(
        temporary_scaled_train,
        mode="w+",
        dtype=np.float32,
        shape=x_train.shape,
    )

    for start in range(
        0,
        len(x_train),
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            len(x_train),
        )

        raw = np.asarray(
            x_train[start:end],
            dtype=np.float64,
        )

        scaled_chunk = (
            (raw - scaler_mean)
            / scaler_scale
        ).astype(np.float32)

        if not np.isfinite(
            scaled_chunk
        ).all():
            raise RuntimeError(
                "Non-finite scaled training values."
            )

        scaled_train[start:end] = (
            scaled_chunk
        )

        print(
            "[scaled train] "
            f"{end:,}/{len(x_train):,}",
            flush=True,
        )

    scaled_train.flush()

    atomic_json(
        run_status_path,
        {
            "status": "running",
            "stage": "training",
            "updated_at_utc": utc_now(),
            "run_id": run_id,
            "model_id": EXPECTED_MODEL_ID,
            "seed": seed,
        },
    )

    print()
    print("Training locked logistic regression...")

    model = LogisticRegression(
        penalty=LOCKED_PARAMETERS["penalty"],
        C=LOCKED_PARAMETERS["C"],
        solver=LOCKED_PARAMETERS["solver"],
        max_iter=LOCKED_PARAMETERS["max_iter"],
        tol=LOCKED_PARAMETERS["tol"],
        fit_intercept=LOCKED_PARAMETERS[
            "fit_intercept"
        ],
        class_weight=LOCKED_PARAMETERS[
            "class_weight"
        ],
        random_state=seed,
    )

    training_started = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:
        warnings.simplefilter("always")

        model.fit(
            scaled_train,
            y_train,
        )

    training_elapsed = (
        time.perf_counter()
        - training_started
    )

    warning_records = [
        {
            "category": (
                warning.category.__name__
            ),
            "message": str(
                warning.message
            ),
        }
        for warning in captured_warnings
    ]

    convergence_warnings = [
        row
        for row in warning_records
        if row["category"]
        == ConvergenceWarning.__name__
    ]

    maximum_iterations_observed = int(
        np.max(model.n_iter_)
    )

    converged = bool(
        not convergence_warnings
        and maximum_iterations_observed
        < LOCKED_PARAMETERS["max_iter"]
    )

    print(
        "Training completed | "
        f"elapsed={training_elapsed / 60.0:.2f} min | "
        f"n_iter={model.n_iter_.tolist()} | "
        f"converged={converged}"
    )

    model_path = (
        output_directory
        / "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
        compress=3,
    )

    np.savez_compressed(
        output_directory
        / "model_parameters.npz",
        classes=model.classes_,
        coef=model.coef_,
        intercept=model.intercept_,
        n_iter=model.n_iter_,
    )

    atomic_json(
        run_status_path,
        {
            "status": "running",
            "stage": "validation_evaluation",
            "updated_at_utc": utc_now(),
            "run_id": run_id,
            "model_id": EXPECTED_MODEL_ID,
            "seed": seed,
        },
    )

    print()
    print("Evaluating validation split...")

    validation_metrics = evaluate_split(
        model=model,
        split_name="validation",
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_directory=output_directory,
    )

    test_evaluation_count = 0

    atomic_json(
        run_status_path,
        {
            "status": "running",
            "stage": "single_test_evaluation",
            "updated_at_utc": utc_now(),
            "run_id": run_id,
            "model_id": EXPECTED_MODEL_ID,
            "seed": seed,
        },
    )

    print("Evaluating test split exactly once...")

    test_metrics = evaluate_split(
        model=model,
        split_name="test",
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_directory=output_directory,
    )

    test_evaluation_count += 1

    if (
        test_evaluation_count
        != expected_test_evaluations
    ):
        raise RuntimeError(
            "Test evaluation count mismatch: "
            f"{test_evaluation_count} != "
            f"{expected_test_evaluations}"
        )

    fragility_gate = {
        "macro_f1_below_0_85": (
            test_metrics[
                "primary_fingerprint_level"
            ]["macro_f1"]
            < 0.85
        ),
        "gafgyt_fnr_above_0_40": (
            test_metrics[
                "primary_fingerprint_level"
            ][
                "attack_false_negative_rates"
            ]["gafgyt"]
            > 0.40
        ),
        "mirai_fnr_above_0_40": (
            test_metrics[
                "primary_fingerprint_level"
            ][
                "attack_false_negative_rates"
            ]["mirai"]
            > 0.40
        ),
    }

    fragility_gate["triggered"] = any(
        fragility_gate.values()
    )

    metrics = {
        "run_id": run_id,
        "model_id": EXPECTED_MODEL_ID,
        "seed": seed,
        "protocol_version": (
            EXPECTED_PROTOCOL_VERSION
        ),
        "parameters": LOCKED_PARAMETERS,
        "training": {
            "fingerprint_count": int(
                len(x_train)
            ),
            "sample_weight": None,
            "elapsed_seconds": (
                training_elapsed
            ),
            "n_iter": (
                model.n_iter_.tolist()
            ),
            "maximum_iterations_observed": (
                maximum_iterations_observed
            ),
            "converged": converged,
            "warnings": warning_records,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_evaluation_count": (
            test_evaluation_count
        ),
        "fragility_gate": fragility_gate,
    }

    atomic_json(
        output_directory / "metrics.json",
        metrics,
    )

    print()
    print("Removing temporary scaled training matrix...")

    del scaled_train
    del x_train

    temporary_scaled_train.unlink()

    artifacts = artifact_inventory(
        output_directory
    )

    completed_at = utc_now()

    final_status = (
        "completed"
        if converged
        else "completed_with_convergence_warning"
    )

    run_manifest = {
        "status": final_status,
        "completed_at_utc": completed_at,
        "run_id": run_id,
        "model_id": EXPECTED_MODEL_ID,
        "architecture": selected_run.get(
            "architecture"
        ),
        "seed": seed,
        "protocol_version": (
            EXPECTED_PROTOCOL_VERSION
        ),
        "input_space": input_space,
        "fit_sample_weight": None,
        "expected_test_evaluation_count": (
            expected_test_evaluations
        ),
        "observed_test_evaluation_count": (
            test_evaluation_count
        ),
        "cache": str(FINAL_CACHE),
        "cache_manifest": str(
            CACHE_MANIFEST
        ),
        "cache_manifest_sha256": (
            sha256_file(CACHE_MANIFEST)
        ),
        "final_verification": str(
            FINAL_VERIFICATION
        ),
        "final_verification_sha256": (
            sha256_file(
                FINAL_VERIFICATION
            )
        ),
        "base_protocol": str(
            BASE_PROTOCOL
        ),
        "base_protocol_sha256": (
            sha256_file(BASE_PROTOCOL)
        ),
        "addendum": str(ADDENDUM),
        "addendum_sha256": (
            sha256_file(ADDENDUM)
        ),
        "run_matrix": str(
            run_matrix_path
        ),
        "run_matrix_sha256": (
            sha256_file(run_matrix_path)
        ),
        "runner_script": str(
            Path(__file__).resolve()
        ),
        "runner_script_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
        "parameters": LOCKED_PARAMETERS,
        "training_elapsed_seconds": (
            training_elapsed
        ),
        "total_elapsed_seconds": (
            time.perf_counter() - started
        ),
        "converged": converged,
        "fragility_gate": fragility_gate,
        "artifacts": artifacts,
        "preflight_checks": checks,
        "all_integrity_checks_passed": True,
    }

    atomic_json(
        output_directory
        / "run_manifest.json",
        run_manifest,
    )

    atomic_json(
        run_status_path,
        {
            "status": final_status,
            "stage": "completed",
            "completed_at_utc": (
                completed_at
            ),
            "run_id": run_id,
            "model_id": (
                EXPECTED_MODEL_ID
            ),
            "seed": seed,
            "converged": converged,
            "fragility_gate_triggered": (
                fragility_gate[
                    "triggered"
                ]
            ),
        },
    )

    test_primary = test_metrics[
        "primary_fingerprint_level"
    ]

    test_weighted = test_metrics[
        "secondary_raw_record_weighted"
    ]

    print()
    print("=" * 92)
    print("LOGISTIC REGRESSION B0 RESULT")
    print("=" * 92)
    print(
        "Run ID                         : "
        f"{run_id}"
    )
    print(
        "Training elapsed               : "
        f"{training_elapsed / 60.0:.2f} minutes"
    )
    print(
        "Converged                      : "
        f"{converged}"
    )
    print(
        "Test evaluation count          : "
        f"{test_evaluation_count}"
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
        f"{test_primary['attack_false_negative_rates']['gafgyt']:.9f}"
    )
    print(
        "Test Mirai FNR                 : "
        f"{test_primary['attack_false_negative_rates']['mirai']:.9f}"
    )
    print(
        "Fragility gate triggered       : "
        f"{fragility_gate['triggered']}"
    )
    print(
        "Output directory               : "
        f"{output_directory}"
    )
    print(
        "All integrity checks passed    : True"
    )
    print(
        "LOGISTIC REGRESSION B0 COMPLETED"
    )


if __name__ == "__main__":
    main()
