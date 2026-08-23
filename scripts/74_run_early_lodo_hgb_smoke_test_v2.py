from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)


PROJECT_ROOT = Path.cwd()

SEED = 2026
HELD_OUT_DEVICE = "Danmini_Doorbell"
MODEL_ID = "hist_gradient_boosting_b0"

TRAIN_QUOTA_PER_CLASS = 10_000
VALIDATION_QUOTA_PER_CLASS = 2_000
TEST_QUOTA_PER_CLASS = 2_000

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

NUM_CLASSES = 3
EXPECTED_FEATURE_COUNT = 115

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
    / HELD_OUT_DEVICE
)

CACHE_SUMMARY_FILE = (
    CACHE_DIRECTORY
    / "cache_summary.json"
)

CACHE_FILES = {
    "x_train":
        CACHE_DIRECTORY
        / "x_train.npy",

    "y_train":
        CACHE_DIRECTORY
        / "y_train.npy",

    "train_global_indices":
        CACHE_DIRECTORY
        / "train_global_indices.npy",

    "x_validation":
        CACHE_DIRECTORY
        / "x_validation.npy",

    "y_validation":
        CACHE_DIRECTORY
        / "y_validation.npy",

    "validation_global_indices":
        CACHE_DIRECTORY
        / "validation_global_indices.npy",

    "x_test":
        CACHE_DIRECTORY
        / "x_test.npy",

    "y_test":
        CACHE_DIRECTORY
        / "y_test.npy",

    "test_global_indices":
        CACHE_DIRECTORY
        / "test_global_indices.npy",
}

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "smoke"
    / "early_lodo_hgb"
)

OUTPUT_JSON = (
    OUTPUT_ROOT
    / "danmini_hgb_smoke_v2.json"
)

OUTPUT_REPORT = (
    OUTPUT_ROOT
    / "danmini_hgb_smoke_v2.txt"
)

OUTPUT_MODEL = (
    OUTPUT_ROOT
    / "danmini_hgb_smoke_v2.joblib"
)


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


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
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


def select_balanced_sample(
    labels: np.ndarray,
    quota_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    selected_parts = []

    for class_id in range(
        NUM_CLASSES
    ):
        positions = np.flatnonzero(
            labels == class_id
        )

        if len(positions) < quota_per_class:
            raise RuntimeError(
                f"Class {class_id} has only "
                f"{len(positions):,} rows; "
                f"quota is "
                f"{quota_per_class:,}."
            )

        selected = rng.choice(
            positions,
            size=quota_per_class,
            replace=False,
        )

        selected_parts.append(
            selected.astype(
                np.int64,
                copy=False,
            )
        )

    combined = np.concatenate(
        selected_parts
    )

    rng.shuffle(
        combined
    )

    return combined


def confusion_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    np.add.at(
        confusion,
        (
            labels,
            predictions,
        ),
        1,
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

        if support == 0:
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
                if predicted_count == 0
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
                if denominator == 0
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
        if total == 0
        else float(
            np.trace(confusion)
            / total
        )
    )

    macro_f1 = (
        0.0
        if not f1_values
        else float(
            np.mean(f1_values)
        )
    )

    balanced_accuracy = (
        0.0
        if not recall_values
        else float(
            np.mean(recall_values)
        )
    )

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1_present_classes":
            macro_f1,

        "confusion_matrix":
            confusion.tolist(),

        "per_class":
            per_class,
    }


required_files = [
    CACHE_SUMMARY_FILE,
    *CACHE_FILES.values(),
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


cache_summary = load_json(
    CACHE_SUMMARY_FILE
)

if cache_summary.get(
    "all_checks_passed"
) is not True:
    raise RuntimeError(
        "Danmini fold cache is not validated."
    )


constructor_parameters = (
    inspect.signature(
        HistGradientBoostingClassifier
    ).parameters
)

required_parameters = {
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

missing_parameters = sorted(
    required_parameters.difference(
        constructor_parameters
    )
)

if missing_parameters:
    raise RuntimeError(
        "Installed HistGradientBoosting "
        "interface is missing parameters: "
        + ", ".join(
            missing_parameters
        )
    )


x_train_memmap = np.load(
    CACHE_FILES["x_train"],
    mmap_mode="r",
    allow_pickle=False,
)

y_train_memmap = np.load(
    CACHE_FILES["y_train"],
    mmap_mode="r",
    allow_pickle=False,
)

train_global_memmap = np.load(
    CACHE_FILES[
        "train_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

x_validation_memmap = np.load(
    CACHE_FILES["x_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

y_validation_memmap = np.load(
    CACHE_FILES["y_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

validation_global_memmap = np.load(
    CACHE_FILES[
        "validation_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)

x_test_memmap = np.load(
    CACHE_FILES["x_test"],
    mmap_mode="r",
    allow_pickle=False,
)

y_test_memmap = np.load(
    CACHE_FILES["y_test"],
    mmap_mode="r",
    allow_pickle=False,
)

test_global_memmap = np.load(
    CACHE_FILES[
        "test_global_indices"
    ],
    mmap_mode="r",
    allow_pickle=False,
)


train_positions = select_balanced_sample(
    y_train_memmap,
    TRAIN_QUOTA_PER_CLASS,
    SEED + 1,
)

validation_positions = (
    select_balanced_sample(
        y_validation_memmap,
        VALIDATION_QUOTA_PER_CLASS,
        SEED + 2,
    )
)

test_positions = select_balanced_sample(
    y_test_memmap,
    TEST_QUOTA_PER_CLASS,
    SEED + 3,
)


x_train = np.asarray(
    x_train_memmap[
        train_positions
    ],
    dtype=np.float32,
)

y_train = np.asarray(
    y_train_memmap[
        train_positions
    ],
    dtype=np.int64,
)

train_global_indices = np.asarray(
    train_global_memmap[
        train_positions
    ],
    dtype=np.int64,
)


x_validation = np.asarray(
    x_validation_memmap[
        validation_positions
    ],
    dtype=np.float32,
)

y_validation = np.asarray(
    y_validation_memmap[
        validation_positions
    ],
    dtype=np.int64,
)

validation_global_indices = np.asarray(
    validation_global_memmap[
        validation_positions
    ],
    dtype=np.int64,
)


x_test = np.asarray(
    x_test_memmap[
        test_positions
    ],
    dtype=np.float32,
)

y_test = np.asarray(
    y_test_memmap[
        test_positions
    ],
    dtype=np.int64,
)

test_global_indices = np.asarray(
    test_global_memmap[
        test_positions
    ],
    dtype=np.int64,
)


train_validation_overlap = int(
    np.intersect1d(
        train_global_indices,
        validation_global_indices,
    ).size
)

train_test_overlap = int(
    np.intersect1d(
        train_global_indices,
        test_global_indices,
    ).size
)

validation_test_overlap = int(
    np.intersect1d(
        validation_global_indices,
        test_global_indices,
    ).size
)


print("=" * 82)
print("PHASE 2G-5 EARLY LODO HGB SMOKE TEST")
print("=" * 82)
print(f"Held-out device : {HELD_OUT_DEVICE}")
print(f"Model           : {MODEL_ID}")
print(f"Scikit-learn    : {sklearn.__version__}")
print(f"Train samples   : {len(y_train):,}")
print(
    f"Validation      : "
    f"{len(y_validation):,}"
)
print(f"Test samples    : {len(y_test):,}")
print()


model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=5,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0001,
    max_bins=255,
    categorical_features=None,
    early_stopping=False,
    class_weight="balanced",
    random_state=SEED,
    verbose=1,
)


started_at = time.perf_counter()

model.fit(
    x_train,
    y_train,
)

fit_seconds = (
    time.perf_counter()
    - started_at
)


validation_started = time.perf_counter()

validation_predictions = model.predict(
    x_validation
).astype(
    np.int64,
    copy=False,
)

validation_seconds = (
    time.perf_counter()
    - validation_started
)


test_started = time.perf_counter()

test_predictions = model.predict(
    x_test
).astype(
    np.int64,
    copy=False,
)

test_seconds = (
    time.perf_counter()
    - test_started
)


validation_confusion = (
    confusion_from_predictions(
        y_validation,
        validation_predictions,
    )
)

test_confusion = (
    confusion_from_predictions(
        y_test,
        test_predictions,
    )
)

validation_metrics = (
    metrics_from_confusion(
        validation_confusion
    )
)

test_metrics = metrics_from_confusion(
    test_confusion
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

temporary_model = OUTPUT_MODEL.with_suffix(
    OUTPUT_MODEL.suffix + ".tmp"
)

joblib.dump(
    model,
    temporary_model,
)

os.replace(
    temporary_model,
    OUTPUT_MODEL,
)


validation_checks = {
    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True,

    "hgb_parameters_supported":
        len(missing_parameters) == 0,

    "train_quota_met":
        len(y_train)
        == (
            TRAIN_QUOTA_PER_CLASS
            * NUM_CLASSES
        ),

    "validation_quota_met":
        len(y_validation)
        == (
            VALIDATION_QUOTA_PER_CLASS
            * NUM_CLASSES
        ),

    "test_quota_met":
        len(y_test)
        == (
            TEST_QUOTA_PER_CLASS
            * NUM_CLASSES
        ),

    "all_features_have_115_columns":
        x_train.shape[1]
        == EXPECTED_FEATURE_COUNT
        and x_validation.shape[1]
        == EXPECTED_FEATURE_COUNT
        and x_test.shape[1]
        == EXPECTED_FEATURE_COUNT,

    "all_features_finite":
        bool(
            np.isfinite(x_train).all()
            and np.isfinite(
                x_validation
            ).all()
            and np.isfinite(
                x_test
            ).all()
        ),

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

    "five_iterations_completed":
        int(model.n_iter_) == 5,

    "train_validation_overlap_zero":
        train_validation_overlap == 0,

    "train_test_overlap_zero":
        train_test_overlap == 0,

    "validation_test_overlap_zero":
        validation_test_overlap == 0,

    "validation_metric_finite":
        math.isfinite(
            validation_metrics[
                "macro_f1_present_classes"
            ]
        ),

    "test_metric_finite":
        math.isfinite(
            test_metrics[
                "macro_f1_present_classes"
            ]
        ),

    "model_file_created":
        OUTPUT_MODEL.exists(),
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "early_lodo_hgb_smoke_v2_1",

    "status":
        (
            "passed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "scientific_result":
        False,

    "purpose": (
        "HistGradientBoosting data, memory, "
        "training and prediction smoke test. "
        "Metrics must not be reported as "
        "scientific results."
    ),

    "held_out_device":
        HELD_OUT_DEVICE,

    "model_id":
        MODEL_ID,

    "seed":
        SEED,

    "software": {
        "scikit_learn":
            sklearn.__version__,

        "joblib":
            joblib.__version__,
    },

    "smoke_configuration": {
        "max_iter":
            5,

        "train_quota_per_class":
            TRAIN_QUOTA_PER_CLASS,

        "validation_quota_per_class":
            VALIDATION_QUOTA_PER_CLASS,

        "test_quota_per_class":
            TEST_QUOTA_PER_CLASS,

        "scaled_features":
            False,
    },

    "sample_class_counts": {
        "train":
            np.bincount(
                y_train,
                minlength=3,
            ).tolist(),

        "validation":
            np.bincount(
                y_validation,
                minlength=3,
            ).tolist(),

        "test":
            np.bincount(
                y_test,
                minlength=3,
            ).tolist(),
    },

    "overlap_counts": {
        "train_validation":
            train_validation_overlap,

        "train_test":
            train_test_overlap,

        "validation_test":
            validation_test_overlap,
    },

    "runtime_seconds": {
        "fit":
            fit_seconds,

        "validation_prediction":
            validation_seconds,

        "test_prediction":
            test_seconds,
    },

    "validation_metrics":
        validation_metrics,

    "test_metrics":
        test_metrics,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,

    "outputs": {
        "json":
            str(OUTPUT_JSON),

        "report":
            str(OUTPUT_REPORT),

        "model":
            str(OUTPUT_MODEL),
    },
}


save_json_atomic(
    OUTPUT_JSON,
    summary,
)


report_lines = [
    "=" * 82,
    "PHASE 2G-5 EARLY LODO HGB SMOKE TEST",
    "=" * 82,
    "",
    (
        f"Held-out device     : "
        f"{HELD_OUT_DEVICE}"
    ),
    (
        f"Model               : "
        f"{MODEL_ID}"
    ),
    (
        f"Train samples       : "
        f"{len(y_train):,}"
    ),
    (
        f"Validation samples  : "
        f"{len(y_validation):,}"
    ),
    (
        f"Test samples        : "
        f"{len(y_test):,}"
    ),
    (
        f"Fit seconds         : "
        f"{fit_seconds:.3f}"
    ),
    (
        f"Validation Macro-F1 : "
        f"{validation_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Test Macro-F1       : "
        f"{test_metrics['macro_f1_present_classes']:.9f}"
    ),
    "",
    "VALIDATION CHECKS",
]

for name, passed in (
    validation_checks.items()
):
    report_lines.append(
        f"{name}: {passed}"
    )


OUTPUT_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


print()
print("=" * 82)
print("PHASE 2G-5 HGB SMOKE SUMMARY")
print("=" * 82)
print(
    f"Iterations          : "
    f"{model.n_iter_}"
)
print(
    f"Fit seconds         : "
    f"{fit_seconds:.3f}"
)
print(
    f"Validation Macro-F1 : "
    f"{validation_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Test Macro-F1       : "
    f"{test_metrics['macro_f1_present_classes']:.9f}"
)

print()
print("OVERLAP COUNTS")
print(
    f"Train-Validation : "
    f"{train_validation_overlap}"
)
print(
    f"Train-Test       : "
    f"{train_test_overlap}"
)
print(
    f"Validation-Test  : "
    f"{validation_test_overlap}"
)

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(
    "NOTICE: Smoke metrics are not "
    "scientific experiment results."
)
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")
print(f"Model  : {OUTPUT_MODEL}")

if not all_checks_passed:
    print()
    print(
        "PHASE 2G-5 EARLY LODO "
        "HGB SMOKE TEST FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2G-5 EARLY LODO "
    "HGB SMOKE TEST PASSED"
)
