from __future__ import annotations

import os

# Random Forest parallelism is locked to n_jobs=2.
# Keep inner numerical libraries single-threaded to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

AGGREGATE_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "random_forest_b0_aggregate_v3_2.json"
)

AGGREGATE_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "random_forest_b0_runs_v3_2.csv"
)

MODEL_ID = "random_forest_b0"
EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
EXPECTED_BASE_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"
EXPECTED_INPUT_SPACE = "canonical_float32_unscaled"

EXPECTED_SEEDS = [42, 123, 2026, 3407, 8192]
EXPECTED_RUN_COUNT = 5

CLASS_LABELS = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
CHUNK_SIZE = 100_000
MINIMUM_FREE_DISK_GIB = 2.0

LOCKED_PARAMETERS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 20,
    "min_samples_split": 40,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.5,
    "class_weight": "balanced_subsample",
    "n_jobs": 2,
    "oob_score": False,
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


def input_space_requires_scaling(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized == "canonical_float32_unscaled":
        return False

    if normalized == "train_only_standardized_float32":
        return True

    raise RuntimeError(
        "Unsupported locked input space: "
        f"{value!r}"
    )


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
    model: RandomForestClassifier,
    split_name: str,
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
        (len(x), 3),
        dtype=np.float32,
    )

    started = time.perf_counter()

    for start in range(0, len(x), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(x))

        model_input = np.asarray(
            x[start:end],
            dtype=np.float32,
        )

        if not np.isfinite(model_input).all():
            raise RuntimeError(
                f"Non-finite model input in {split_name}."
            )

        predictions[start:end] = model.predict(
            model_input
        ).astype(np.int8)

        probabilities[start:end] = (
            model.predict_proba(
                model_input
            ).astype(np.float32)
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


def main() -> None:
    started = time.perf_counter()

    required_paths = [
        FINAL_CACHE,
        CACHE_MANIFEST,
        FINAL_CACHE_VERIFICATION,
        LOGISTIC_VERIFICATION,
        DECISION_TREE_VERIFICATION,
        BASE_PROTOCOL,
        BASE_LOCK_MANIFEST,
        ADDENDUM,
        ADDENDUM_MANIFEST,
        FINAL_CACHE / "X_train.npy",
        FINAL_CACHE / "y_train.npy",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    for path in (
        AGGREGATE_JSON,
        AGGREGATE_CSV,
    ):
        if path.exists():
            raise FileExistsError(
                "Random Forest aggregate artifact already exists; "
                f"refusing to overwrite: {path}"
            )

    cache_verification = read_json(
        FINAL_CACHE_VERIFICATION
    )
    logistic_verification = read_json(
        LOGISTIC_VERIFICATION
    )
    decision_tree_verification = read_json(
        DECISION_TREE_VERIFICATION
    )
    cache_manifest = read_json(CACHE_MANIFEST)
    base_protocol = read_json(BASE_PROTOCOL)
    base_lock_manifest = read_json(
        BASE_LOCK_MANIFEST
    )
    addendum = read_json(ADDENDUM)
    addendum_manifest = read_json(
        ADDENDUM_MANIFEST
    )

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
        all_rows = list(csv.DictReader(handle))

    selected_rows = [
        row
        for row in all_rows
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

    input_spaces = {
        str(row["input_space"]).strip().lower()
        for row in selected_rows
    }

    checks = {
        "final_cache_verification_passed": (
            cache_verification.get("status")
            == "passed"
            and cache_verification.get(
                "all_checks_passed"
            )
            is True
        ),
        "logistic_verification_passed": (
            logistic_verification.get("status")
            == "passed"
            and logistic_verification.get(
                "all_checks_passed"
            )
            is True
        ),
        "decision_tree_verification_passed": (
            decision_tree_verification.get("status")
            == "passed"
            and decision_tree_verification.get(
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
        "cache_protocol_version_matches": (
            cache_manifest.get(
                "protocol_version"
            )
            == EXPECTED_PROTOCOL_VERSION
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
        "run_count_is_five": (
            len(selected_rows)
            == EXPECTED_RUN_COUNT
        ),
        "seed_set_matches": (
            selected_seeds
            == EXPECTED_SEEDS
        ),
        "single_input_space": (
            len(input_spaces) == 1
        ),
        "locked_input_space_is_exact_unscaled": (
            input_spaces
            == {EXPECTED_INPUT_SPACE}
        ),
        "all_fit_sample_weights_none": all(
            normalize_none_token(
                str(
                    row.get(
                        "fit_sample_weight",
                        "",
                    )
                )
            )
            for row in selected_rows
        ),
        "all_test_evaluation_counts_one": all(
            int(
                row.get(
                    "expected_test_evaluation_count",
                    "1",
                )
            )
            == 1
            for row in selected_rows
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
            "Random Forest preflight failed: "
            + ", ".join(failed_checks)
        )

    output_directories = [
        resolve_project_path(
            str(row["output_directory"])
        )
        for row in selected_rows
    ]

    existing_outputs = [
        path
        for path in output_directories
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Random Forest output directories already exist: "
            + ", ".join(
                str(path)
                for path in existing_outputs
            )
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

    input_space = next(iter(input_spaces))
    use_scaled = input_space_requires_scaling(
        input_space
    )

    if use_scaled:
        raise RuntimeError(
            "Random Forest protocol requires exact "
            "canonical_float32_unscaled input."
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

    print("=" * 92)
    print("PHASE 4 CLASSICAL BASELINE: RANDOM FOREST B0")
    print("=" * 92)
    print(f"Runs            : {len(selected_rows)}")
    print(f"Seeds           : {selected_seeds}")
    print(f"Input space     : {input_space}")
    print(f"Scaling applied : {use_scaled}")
    print("Parser policy   : exact locked-token equality")
    print("Fit weighting   : none")
    print("n_jobs          : 2")
    print("Test evaluations: one per run")
    print(f"Free disk       : {free_disk_gib:.3f} GiB")

    run_summaries: list[dict[str, Any]] = []

    for run_number, row in enumerate(
        selected_rows,
        start=1,
    ):
        run_started = time.perf_counter()
        run_id = str(row["run_id"]).strip()
        seed = int(row["seed"])
        output_directory = resolve_project_path(
            str(row["output_directory"])
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
                "stage": "training",
                "started_at_utc": utc_now(),
                "run_id": run_id,
                "model_id": MODEL_ID,
                "seed": seed,
            },
        )

        print()
        print("-" * 92)
        print(
            f"RUN {run_number}/{len(selected_rows)} | "
            f"{run_id}"
        )
        print("-" * 92)
        print(
            "Training 100 trees with n_jobs=2...",
            flush=True,
        )

        model = RandomForestClassifier(
            n_estimators=LOCKED_PARAMETERS[
                "n_estimators"
            ],
            criterion=LOCKED_PARAMETERS[
                "criterion"
            ],
            max_depth=LOCKED_PARAMETERS[
                "max_depth"
            ],
            min_samples_split=(
                LOCKED_PARAMETERS[
                    "min_samples_split"
                ]
            ),
            min_samples_leaf=(
                LOCKED_PARAMETERS[
                    "min_samples_leaf"
                ]
            ),
            max_features=LOCKED_PARAMETERS[
                "max_features"
            ],
            bootstrap=LOCKED_PARAMETERS[
                "bootstrap"
            ],
            max_samples=LOCKED_PARAMETERS[
                "max_samples"
            ],
            class_weight=LOCKED_PARAMETERS[
                "class_weight"
            ],
            n_jobs=LOCKED_PARAMETERS[
                "n_jobs"
            ],
            oob_score=LOCKED_PARAMETERS[
                "oob_score"
            ],
            random_state=seed,
            verbose=1,
        )

        training_started = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        training_elapsed = (
            time.perf_counter()
            - training_started
        )

        model_path = (
            output_directory / "model.joblib"
        )

        print(
            "Saving fitted forest...",
            flush=True,
        )

        joblib.dump(
            model,
            model_path,
            compress=3,
        )

        tree_node_counts = np.asarray(
            [
                estimator.tree_.node_count
                for estimator in model.estimators_
            ],
            dtype=np.int64,
        )

        tree_depths = np.asarray(
            [
                estimator.tree_.max_depth
                for estimator in model.estimators_
            ],
            dtype=np.int64,
        )

        tree_leaf_counts = np.asarray(
            [
                estimator.get_n_leaves()
                for estimator in model.estimators_
            ],
            dtype=np.int64,
        )

        np.savez_compressed(
            output_directory
            / "model_parameters.npz",
            classes=model.classes_,
            feature_importances=(
                model.feature_importances_
            ),
            tree_node_counts=tree_node_counts,
            tree_depths=tree_depths,
            tree_leaf_counts=tree_leaf_counts,
        )

        atomic_json(
            run_status_path,
            {
                "status": "running",
                "stage": "validation_evaluation",
                "updated_at_utc": utc_now(),
                "run_id": run_id,
                "model_id": MODEL_ID,
                "seed": seed,
            },
        )

        print(
            "Evaluating validation split...",
            flush=True,
        )

        validation_metrics = evaluate_split(
            model=model,
            split_name="validation",
            output_directory=output_directory,
        )

        atomic_json(
            run_status_path,
            {
                "status": "running",
                "stage": "single_test_evaluation",
                "updated_at_utc": utc_now(),
                "run_id": run_id,
                "model_id": MODEL_ID,
                "seed": seed,
            },
        )

        print(
            "Evaluating test split exactly once...",
            flush=True,
        )

        test_metrics = evaluate_split(
            model=model,
            split_name="test",
            output_directory=output_directory,
        )

        test_evaluation_count = 1

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
            "model_id": MODEL_ID,
            "seed": seed,
            "protocol_version": (
                EXPECTED_PROTOCOL_VERSION
            ),
            "parameters": {
                **LOCKED_PARAMETERS,
                "random_state": seed,
            },
            "training": {
                "fingerprint_count": int(
                    len(x_train)
                ),
                "sample_weight": None,
                "elapsed_seconds": (
                    training_elapsed
                ),
                "tree_count": int(
                    len(model.estimators_)
                ),
                "mean_tree_node_count": float(
                    np.mean(tree_node_counts)
                ),
                "mean_tree_depth": float(
                    np.mean(tree_depths)
                ),
                "mean_tree_leaf_count": float(
                    np.mean(tree_leaf_counts)
                ),
                "minimum_tree_depth": int(
                    np.min(tree_depths)
                ),
                "maximum_tree_depth": int(
                    np.max(tree_depths)
                ),
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

        artifacts = artifact_inventory(
            output_directory
        )

        completed_at = utc_now()

        run_manifest = {
            "status": "completed",
            "completed_at_utc": completed_at,
            "run_id": run_id,
            "model_id": MODEL_ID,
            "architecture": row.get(
                "architecture"
            ),
            "seed": seed,
            "protocol_version": (
                EXPECTED_PROTOCOL_VERSION
            ),
            "input_space": input_space,
            "fit_sample_weight": None,
            "expected_test_evaluation_count": 1,
            "observed_test_evaluation_count": 1,
            "cache": str(FINAL_CACHE),
            "cache_manifest": str(
                CACHE_MANIFEST
            ),
            "cache_manifest_sha256": (
                sha256_file(CACHE_MANIFEST)
            ),
            "final_verification": str(
                FINAL_CACHE_VERIFICATION
            ),
            "final_verification_sha256": (
                sha256_file(
                    FINAL_CACHE_VERIFICATION
                )
            ),
            "decision_tree_verification": str(
                DECISION_TREE_VERIFICATION
            ),
            "decision_tree_verification_sha256": (
                sha256_file(
                    DECISION_TREE_VERIFICATION
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
            "input_space_parser_policy": (
                "exact locked-token equality; "
                "no substring matching"
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
            "parameters": {
                **LOCKED_PARAMETERS,
                "random_state": seed,
            },
            "training_elapsed_seconds": (
                training_elapsed
            ),
            "total_elapsed_seconds": (
                time.perf_counter()
                - run_started
            ),
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
                "status": "completed",
                "stage": "completed",
                "completed_at_utc": (
                    completed_at
                ),
                "run_id": run_id,
                "model_id": MODEL_ID,
                "seed": seed,
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

        summary = {
            "run_id": run_id,
            "seed": seed,
            "training_elapsed_seconds": (
                training_elapsed
            ),
            "tree_count": int(
                len(model.estimators_)
            ),
            "mean_tree_node_count": float(
                np.mean(tree_node_counts)
            ),
            "mean_tree_depth": float(
                np.mean(tree_depths)
            ),
            "mean_tree_leaf_count": float(
                np.mean(tree_leaf_counts)
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
                test_primary[
                    "attack_false_negative_rates"
                ]["gafgyt"]
            ),
            "test_mirai_fnr": (
                test_primary[
                    "attack_false_negative_rates"
                ]["mirai"]
            ),
            "fragility_gate_triggered": (
                fragility_gate["triggered"]
            ),
            "output_directory": str(
                output_directory
            ),
        }

        run_summaries.append(summary)

        print(
            "Training elapsed           : "
            f"{training_elapsed / 60.0:.2f} min"
        )
        print(
            "Trees / mean nodes / depth : "
            f"{len(model.estimators_)} / "
            f"{summary['mean_tree_node_count']:.1f} / "
            f"{summary['mean_tree_depth']:.2f}"
        )
        print(
            "Test fingerprint Macro-F1  : "
            f"{test_primary['macro_f1']:.9f}"
        )
        print(
            "Test raw-weighted Macro-F1 : "
            f"{test_weighted['macro_f1']:.9f}"
        )
        print(
            "Gafgyt FNR / Mirai FNR     : "
            f"{summary['test_gafgyt_fnr']:.9f} / "
            f"{summary['test_mirai_fnr']:.9f}"
        )
        print(
            "Fragility gate triggered   : "
            f"{fragility_gate['triggered']}"
        )

        del model

    aggregate_fields = [
        "run_id",
        "seed",
        "training_elapsed_seconds",
        "tree_count",
        "mean_tree_node_count",
        "mean_tree_depth",
        "mean_tree_leaf_count",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "fragility_gate_triggered",
        "output_directory",
    ]

    with AGGREGATE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=aggregate_fields,
        )
        writer.writeheader()
        writer.writerows(run_summaries)

    aggregate = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "protocol_version": (
            EXPECTED_PROTOCOL_VERSION
        ),
        "run_count": len(run_summaries),
        "seeds": [
            row["seed"]
            for row in run_summaries
        ],
        "input_space": input_space,
        "input_space_parser_policy": (
            "exact locked-token equality; "
            "no substring matching"
        ),
        "parameters": LOCKED_PARAMETERS,
        "test_evaluation_count_per_run": 1,
        "test_used_for_selection": False,
        "runs": run_summaries,
        "aggregate_metrics": {
            "test_fingerprint_macro_f1": (
                aggregate_numeric(
                    [
                        row[
                            "test_fingerprint_macro_f1"
                        ]
                        for row in run_summaries
                    ]
                )
            ),
            "test_fingerprint_accuracy": (
                aggregate_numeric(
                    [
                        row[
                            "test_fingerprint_accuracy"
                        ]
                        for row in run_summaries
                    ]
                )
            ),
            "test_raw_weighted_macro_f1": (
                aggregate_numeric(
                    [
                        row[
                            "test_raw_weighted_macro_f1"
                        ]
                        for row in run_summaries
                    ]
                )
            ),
            "test_gafgyt_fnr": (
                aggregate_numeric(
                    [
                        row["test_gafgyt_fnr"]
                        for row in run_summaries
                    ]
                )
            ),
            "test_mirai_fnr": (
                aggregate_numeric(
                    [
                        row["test_mirai_fnr"]
                        for row in run_summaries
                    ]
                )
            ),
            "training_elapsed_seconds": (
                aggregate_numeric(
                    [
                        row[
                            "training_elapsed_seconds"
                        ]
                        for row in run_summaries
                    ]
                )
            ),
        },
        "fragility_gate_triggered_in_any_run": any(
            row["fragility_gate_triggered"]
            for row in run_summaries
        ),
        "run_matrix": str(run_matrix_path),
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
        "aggregate_csv": str(AGGREGATE_CSV),
        "aggregate_csv_sha256": (
            sha256_file(AGGREGATE_CSV)
        ),
        "preflight_checks": checks,
        "all_integrity_checks_passed": True,
        "total_elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    atomic_json(AGGREGATE_JSON, aggregate)

    macro = aggregate[
        "aggregate_metrics"
    ]["test_fingerprint_macro_f1"]

    weighted = aggregate[
        "aggregate_metrics"
    ]["test_raw_weighted_macro_f1"]

    print()
    print("=" * 92)
    print("RANDOM FOREST B0 FIVE-SEED SUMMARY")
    print("=" * 92)
    print(
        "Runs completed                 : "
        f"{len(run_summaries)}"
    )
    print(
        "Seeds                          : "
        f"{[row['seed'] for row in run_summaries]}"
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
        "Fragility in any run           : "
        f"{aggregate['fragility_gate_triggered_in_any_run']}"
    )
    print(
        "Aggregate JSON                 : "
        f"{AGGREGATE_JSON}"
    )
    print(
        "Aggregate CSV                  : "
        f"{AGGREGATE_CSV}"
    )
    print(
        "All integrity checks passed    : True"
    )
    print(
        "RANDOM FOREST B0 FIVE-SEED RUNS COMPLETED"
    )


if __name__ == "__main__":
    main()
