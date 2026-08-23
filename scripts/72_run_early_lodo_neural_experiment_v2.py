from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.models.nbaiot_models import create_model


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
INPUT_FEATURES = 115

EVALUATION_BATCH_SIZE = 8192
PROGRESS_BATCH_INTERVAL = 100


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
        "Object of type "
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


def set_reproducible_environment(
    seed: int,
) -> None:
    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True
    )

    torch.set_num_threads(
        max(
            1,
            min(
                4,
                os.cpu_count() or 1,
            ),
        )
    )


def atomic_torch_save(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        value,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def copy_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name:
            tensor.detach()
            .cpu()
            .clone()
        for name, tensor
        in model.state_dict().items()
    }


def transform_batch(
    raw_features: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    transformed = (
        raw_features.astype(
            np.float64,
            copy=False,
        )
        - scaler_mean
    ) / scaler_scale

    transformed = transformed.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        transformed
    ).all():
        raise RuntimeError(
            "Non-finite standardized features."
        )

    return transformed


def metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, Any]:
    confusion = np.asarray(
        confusion,
        dtype=np.float64,
    )

    total = float(
        confusion.sum()
    )

    supports = confusion.sum(
        axis=1
    )

    predicted_counts = confusion.sum(
        axis=0
    )

    per_class: dict[
        str,
        dict[str, Any]
    ] = {}

    present_class_ids = [
        class_id
        for class_id in range(
            NUM_CLASSES
        )
        if supports[class_id] > 0
    ]

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
            support
            - true_positive
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

            if predicted_count <= 0:
                precision = 0.0
            else:
                precision = (
                    true_positive
                    / predicted_count
                )

            denominator = (
                2.0 * true_positive
                + false_positive
                + false_negative
            )

            if denominator <= 0:
                f1_value = 0.0
            else:
                f1_value = (
                    2.0
                    * true_positive
                    / denominator
                )

            present_f1_values.append(
                f1_value
            )

            present_recall_values.append(
                recall
            )

            weighted_f1_numerator += (
                support
                * f1_value
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

    if total <= 0:
        accuracy = 0.0
        weighted_f1 = 0.0

    else:
        accuracy = float(
            np.trace(confusion)
            / total
        )

        weighted_f1 = (
            weighted_f1_numerator
            / total
        )

    macro_f1 = (
        float(
            np.mean(
                present_f1_values
            )
        )
        if present_f1_values
        else 0.0
    )

    balanced_accuracy = (
        float(
            np.mean(
                present_recall_values
            )
        )
        if present_recall_values
        else 0.0
    )

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1_present_classes":
            macro_f1,

        "weighted_f1":
            float(
                weighted_f1
            ),

        "present_class_ids":
            present_class_ids,

        "present_class_names": [
            CLASS_NAMES[
                class_id
            ]
            for class_id
            in present_class_ids
        ],

        "confusion_matrix":
            confusion.tolist(),

        "per_class":
            per_class,
    }


def evaluate(
    model: nn.Module,
    raw_features: np.ndarray,
    labels: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    model.eval()

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    weighted_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for start in range(
            0,
            len(labels),
            batch_size,
        ):
            end = min(
                len(labels),
                start + batch_size,
            )

            transformed = transform_batch(
                raw_features[start:end],
                scaler_mean,
                scaler_scale,
            )

            batch_labels_numpy = np.asarray(
                labels[start:end],
                dtype=np.int64,
            )

            batch_features = (
                torch.from_numpy(
                    transformed
                )
            )

            batch_labels = torch.from_numpy(
                batch_labels_numpy
            )

            logits = model(
                batch_features
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite evaluation loss."
                )

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            batch_count = (
                end - start
            )

            total_loss += (
                float(loss.item())
                * batch_count
            )

            total_examples += (
                batch_count
            )

            np.add.at(
                confusion,
                (
                    batch_labels_numpy,
                    predictions,
                ),
                1,
            )

            if sample_weights is None:
                weights = np.ones(
                    batch_count,
                    dtype=np.float64,
                )

            else:
                weights = np.asarray(
                    sample_weights[
                        start:end
                    ],
                    dtype=np.float64,
                )

            np.add.at(
                weighted_confusion,
                (
                    batch_labels_numpy,
                    predictions,
                ),
                weights,
            )

    fingerprint_metrics = (
        metrics_from_confusion(
            confusion
        )
    )

    record_weighted_metrics = (
        metrics_from_confusion(
            weighted_confusion
        )
    )

    return {
        "loss":
            (
                total_loss
                / total_examples
            ),

        "fingerprint":
            fingerprint_metrics,

        "record_weighted":
            record_weighted_metrics,
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    raw_features: np.ndarray,
    labels: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    batch_size: int,
    epoch_number: int,
    seed: int,
) -> dict[str, Any]:
    model.train()

    epoch_rng = np.random.default_rng(
        seed + epoch_number
    )

    permutation = epoch_rng.permutation(
        len(labels)
    )

    total_loss = 0.0
    total_examples = 0
    batch_count = 0

    epoch_started = time.perf_counter()

    for start in range(
        0,
        len(permutation),
        batch_size,
    ):
        end = min(
            len(permutation),
            start + batch_size,
        )

        batch_indices = permutation[
            start:end
        ]

        transformed = transform_batch(
            raw_features[
                batch_indices
            ],
            scaler_mean,
            scaler_scale,
        )

        batch_labels_numpy = np.asarray(
            labels[
                batch_indices
            ],
            dtype=np.int64,
        )

        batch_features = torch.from_numpy(
            transformed
        )

        batch_labels = torch.from_numpy(
            batch_labels_numpy
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            batch_features
        )

        loss = criterion(
            logits,
            batch_labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()
        optimizer.step()

        current_batch_size = (
            end - start
        )

        total_loss += (
            float(loss.item())
            * current_batch_size
        )

        total_examples += (
            current_batch_size
        )

        batch_count += 1

        if (
            batch_count == 1
            or batch_count
            % PROGRESS_BATCH_INTERVAL
            == 0
            or end
            == len(permutation)
        ):
            print(
                f"  epoch={epoch_number:02d} | "
                f"batch={batch_count:,} | "
                f"rows={end:,}/"
                f"{len(permutation):,} | "
                f"loss="
                f"{total_loss / total_examples:.7f}"
            )

    return {
        "loss":
            (
                total_loss
                / total_examples
            ),

        "batch_count":
            batch_count,

        "elapsed_seconds":
            (
                time.perf_counter()
                - epoch_started
            ),
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

if run_spec["model_family"] != "neural":
    raise RuntimeError(
        "This runner accepts only neural runs."
    )


model_id = run_spec[
    "model_id"
]

held_out_device = run_spec[
    "held_out_device"
]

seed = int(
    run_spec["seed"]
)


model_configurations = {
    model["model_id"]: model
    for model in protocol[
        "models"
    ]
}

if model_id not in model_configurations:
    raise RuntimeError(
        "Model configuration not found."
    )

model_configuration = (
    model_configurations[
        model_id
    ]
)


if model_id == "tinyml_mlp_b0":
    registry_model_name = (
        "tinyml_mlp"
    )

elif model_id == "compact_dnn_b0":
    registry_model_name = (
        "compact_dnn"
    )

else:
    raise RuntimeError(
        "Unsupported neural model_id: "
        f"{model_id}"
    )


batch_size = int(
    model_configuration[
        "batch_size"
    ]
)

maximum_epochs = int(
    model_configuration[
        "maximum_epochs"
    ]
)

patience = int(
    model_configuration[
        "early_stopping_patience"
    ]
)

minimum_delta = float(
    model_configuration[
        "early_stopping_min_delta"
    ]
)

learning_rate = float(
    model_configuration[
        "learning_rate"
    ]
)

weight_decay = float(
    model_configuration[
        "weight_decay"
    ]
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

    "scaler":
        cache_directory
        / "scaler.npz",
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

checkpoint_file = (
    run_directory
    / "checkpoint.pt"
)

best_model_file = (
    run_directory
    / "best_model.pt"
)

history_file = (
    run_directory
    / "history.csv"
)

result_file = (
    run_directory
    / "run_result.json"
)

report_file = (
    run_directory
    / "run_report.txt"
)


if args.restart and run_directory.exists():
    import shutil

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
            "Existing completed scientific "
            "run was found."
        )
        print(
            f"Result: {result_file}"
        )
        print(
            "EARLY LODO SCIENTIFIC "
            "RUN ALREADY COMPLETED"
        )
        sys.exit(0)


set_reproducible_environment(
    seed
)


x_train = np.load(
    cache_files["x_train"],
    mmap_mode="r",
    allow_pickle=False,
)

y_train = np.load(
    cache_files["y_train"],
    mmap_mode="r",
    allow_pickle=False,
)

x_validation = np.load(
    cache_files["x_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

y_validation = np.load(
    cache_files["y_validation"],
    mmap_mode="r",
    allow_pickle=False,
)

x_test = np.load(
    cache_files["x_test"],
    mmap_mode="r",
    allow_pickle=False,
)

y_test = np.load(
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


with np.load(
    cache_files["scaler"],
    allow_pickle=False,
) as scaler_archive:
    scaler_mean = np.asarray(
        scaler_archive["mean"],
        dtype=np.float64,
    )

    scaler_scale = np.asarray(
        scaler_archive["scale"],
        dtype=np.float64,
    )


if scaler_mean.shape != (
    INPUT_FEATURES,
):
    raise RuntimeError(
        "Scaler mean shape mismatch."
    )

if scaler_scale.shape != (
    INPUT_FEATURES,
):
    raise RuntimeError(
        "Scaler scale shape mismatch."
    )

if not np.all(
    scaler_scale > 0
):
    raise RuntimeError(
        "Scaler contains non-positive scales."
    )


train_class_counts = np.bincount(
    y_train.astype(
        np.int64
    ),
    minlength=NUM_CLASSES,
).astype(
    np.float64
)

class_weights = (
    1.0
    / np.sqrt(
        train_class_counts
    )
)

class_weights = (
    class_weights
    / class_weights.mean()
)


model_kwargs: dict[str, Any] = {}

if registry_model_name == "tinyml_mlp":
    model_kwargs[
        "hidden_units"
    ] = tuple(
        model_configuration[
            "hidden_layers"
        ]
    )

elif registry_model_name == "compact_dnn":
    model_kwargs[
        "hidden_units"
    ] = tuple(
        model_configuration[
            "hidden_layers"
        ]
    )

    model_kwargs[
        "dropout_probability"
    ] = float(
        model_configuration[
            "dropout"
        ]
    )


model = create_model(
    registry_model_name,
    input_features=INPUT_FEATURES,
    num_classes=NUM_CLASSES,
    **model_kwargs,
)

parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32,
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
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


start_epoch = 1
best_epoch = None
best_validation_macro_f1 = (
    float("-inf")
)

early_stop_reference = (
    float("-inf")
)

bad_epoch_count = 0
history: list[
    dict[str, Any]
] = []

best_state: dict[
    str,
    torch.Tensor
] | None = None

resumed = False


if checkpoint_file.exists():
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_identity = (
        checkpoint.get(
            "identity",
            {}
        )
    )

    expected_identity = {
        "run_id":
            run_id,

        "protocol_sha256":
            protocol_sha256,

        "run_matrix_sha256":
            run_matrix_sha256,

        "cache_summary_sha256":
            cache_summary_sha256,
    }

    if checkpoint_identity != (
        expected_identity
    ):
        raise RuntimeError(
            "Checkpoint identity does not "
            "match the locked experiment."
        )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state"
        ]
    )

    start_epoch = int(
        checkpoint[
            "next_epoch"
        ]
    )

    best_epoch = checkpoint[
        "best_epoch"
    ]

    best_validation_macro_f1 = float(
        checkpoint[
            "best_validation_macro_f1"
        ]
    )

    early_stop_reference = float(
        checkpoint[
            "early_stop_reference"
        ]
    )

    bad_epoch_count = int(
        checkpoint[
            "bad_epoch_count"
        ]
    )

    history = list(
        checkpoint["history"]
    )

    best_state = checkpoint[
        "best_model_state"
    ]

    resumed = True


print("=" * 86)
print("EARLY LODO SCIENTIFIC NEURAL RUN")
print("=" * 86)
print(f"Run id          : {run_id}")
print(f"Held-out device : {held_out_device}")
print(f"Model           : {model_id}")
print(f"Registry model  : {registry_model_name}")
print(f"Seed            : {seed}")
print(f"Parameters      : {parameter_count:,}")
print(f"Train rows      : {len(y_train):,}")
print(f"Validation rows : {len(y_validation):,}")
print(f"Test rows       : {len(y_test):,}")
print(f"Maximum epochs  : {maximum_epochs}")
print(f"Start epoch     : {start_epoch}")
print(f"Resumed         : {resumed}")
print()


scientific_started = time.perf_counter()
stopped_early = False
completed_epoch = start_epoch - 1


for epoch_number in range(
    start_epoch,
    maximum_epochs + 1,
):
    print(
        "=" * 86
    )

    print(
        f"EPOCH {epoch_number}/"
        f"{maximum_epochs}"
    )

    print(
        "=" * 86
    )

    training_result = (
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            raw_features=x_train,
            labels=y_train,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            batch_size=batch_size,
            epoch_number=epoch_number,
            seed=seed,
        )
    )

    validation_started = (
        time.perf_counter()
    )

    validation_result = evaluate(
        model=model,
        raw_features=x_validation,
        labels=y_validation,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        criterion=criterion,
        batch_size=EVALUATION_BATCH_SIZE,
        sample_weights=None,
    )

    validation_seconds = (
        time.perf_counter()
        - validation_started
    )

    validation_macro_f1 = float(
        validation_result[
            "fingerprint"
        ][
            "macro_f1_present_classes"
        ]
    )

    is_new_best = (
        validation_macro_f1
        > best_validation_macro_f1
    )

    if is_new_best:
        best_validation_macro_f1 = (
            validation_macro_f1
        )

        best_epoch = epoch_number

        best_state = copy_state_dict(
            model
        )

    patience_improvement = (
        validation_macro_f1
        > (
            early_stop_reference
            + minimum_delta
        )
    )

    if patience_improvement:
        early_stop_reference = (
            validation_macro_f1
        )

        bad_epoch_count = 0

    else:
        bad_epoch_count += 1

    epoch_row = {
        "epoch":
            epoch_number,

        "training_loss":
            training_result["loss"],

        "validation_loss":
            validation_result["loss"],

        "validation_accuracy":
            validation_result[
                "fingerprint"
            ]["accuracy"],

        "validation_balanced_accuracy":
            validation_result[
                "fingerprint"
            ][
                "balanced_accuracy"
            ],

        "validation_macro_f1":
            validation_macro_f1,

        "validation_weighted_f1":
            validation_result[
                "fingerprint"
            ]["weighted_f1"],

        "is_new_best":
            is_new_best,

        "patience_improvement":
            patience_improvement,

        "bad_epoch_count":
            bad_epoch_count,

        "training_seconds":
            training_result[
                "elapsed_seconds"
            ],

        "validation_seconds":
            validation_seconds,
    }

    history.append(
        epoch_row
    )

    completed_epoch = epoch_number

    should_stop = (
        bad_epoch_count >= patience
    )

    checkpoint = {
        "protocol_version":
            "early_lodo_neural_checkpoint_v2_1",

        "saved_at":
            utc_now(),

        "identity": {
            "run_id":
                run_id,

            "protocol_sha256":
                protocol_sha256,

            "run_matrix_sha256":
                run_matrix_sha256,

            "cache_summary_sha256":
                cache_summary_sha256,
        },

        "next_epoch":
            epoch_number + 1,

        "completed_epoch":
            epoch_number,

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "best_model_state":
            best_state,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_validation_macro_f1,

        "early_stop_reference":
            early_stop_reference,

        "bad_epoch_count":
            bad_epoch_count,

        "history":
            history,

        "should_stop":
            should_stop,
    }

    atomic_torch_save(
        checkpoint,
        checkpoint_file,
    )

    write_csv_atomic(
        history_file,
        history,
        list(
            history[0].keys()
        ),
    )

    print()
    print(
        f"Epoch training loss  : "
        f"{training_result['loss']:.9f}"
    )

    print(
        f"Validation loss      : "
        f"{validation_result['loss']:.9f}"
    )

    print(
        f"Validation Macro-F1  : "
        f"{validation_macro_f1:.9f}"
    )

    print(
        f"Best epoch           : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1   : "
        f"{best_validation_macro_f1:.9f}"
    )

    print(
        f"Bad epoch count      : "
        f"{bad_epoch_count}/{patience}"
    )

    print(
        f"Checkpoint           : "
        f"{checkpoint_file}"
    )

    print()

    if should_stop:
        stopped_early = True

        print(
            "Early stopping condition met."
        )

        break


if best_state is None:
    raise RuntimeError(
        "Best model state was not created."
    )

if best_epoch is None:
    raise RuntimeError(
        "Best epoch was not selected."
    )


model.load_state_dict(
    best_state
)

atomic_torch_save(
    {
        "protocol_version":
            "early_lodo_best_model_v2_1",

        "saved_at":
            utc_now(),

        "run_id":
            run_id,

        "held_out_device":
            held_out_device,

        "model_id":
            model_id,

        "registry_model_name":
            registry_model_name,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_validation_macro_f1,

        "parameter_count":
            parameter_count,

        "model_state":
            best_state,

        "model_kwargs":
            model_kwargs,

        "scaler_mean":
            scaler_mean,

        "scaler_scale":
            scaler_scale,

        "class_weights":
            class_weights,

        "identity": {
            "protocol_sha256":
                protocol_sha256,

            "run_matrix_sha256":
                run_matrix_sha256,

            "cache_summary_sha256":
                cache_summary_sha256,
        },
    },
    best_model_file,
)


print()
print("=" * 86)
print("FINAL TEST EVALUATION")
print("=" * 86)
print(
    "Test data are evaluated exactly once "
    "using the selected validation epoch."
)


test_started = time.perf_counter()

test_result = evaluate(
    model=model,
    raw_features=x_test,
    labels=y_test,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    criterion=criterion,
    batch_size=EVALUATION_BATCH_SIZE,
    sample_weights=test_occurrence_count,
)

test_seconds = (
    time.perf_counter()
    - test_started
)

total_seconds = (
    time.perf_counter()
    - scientific_started
)


validation_checks = {
    "protocol_locked":
        protocol.get("status")
        == "locked",

    "cache_validated":
        cache_summary.get(
            "all_checks_passed"
        )
        is True,

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

    "all_train_classes_present":
        bool(
            np.all(
                train_class_counts > 0
            )
        ),

    "scaler_dimensions_115":
        scaler_mean.shape
        == (
            INPUT_FEATURES,
        )
        and scaler_scale.shape
        == (
            INPUT_FEATURES,
        ),

    "class_weights_finite":
        bool(
            np.isfinite(
                class_weights
            ).all()
        ),

    "model_parameter_count_positive":
        parameter_count > 0,

    "history_nonempty":
        len(history) > 0,

    "best_epoch_valid":
        1
        <= int(best_epoch)
        <= completed_epoch,

    "best_validation_metric_finite":
        math.isfinite(
            best_validation_macro_f1
        ),

    "test_evaluated_once":
        True,

    "fingerprint_test_metric_finite":
        math.isfinite(
            test_result[
                "fingerprint"
            ][
                "macro_f1_present_classes"
            ]
        ),

    "record_weighted_test_metric_finite":
        math.isfinite(
            test_result[
                "record_weighted"
            ][
                "macro_f1_present_classes"
            ]
        ),

    "best_model_file_created":
        best_model_file.exists(),

    "checkpoint_file_created":
        checkpoint_file.exists(),

    "history_file_created":
        history_file.exists(),
}


all_checks_passed = all(
    validation_checks.values()
)


run_result = {
    "protocol_version":
        "early_lodo_neural_result_v2_1",

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

        "registry_model_name":
            registry_model_name,

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

        "test_record_weight":
            (
                "held-out-device "
                "occurrence count"
            ),

        "train_class_counts":
            train_class_counts.astype(
                np.int64
            ).tolist(),

        "validation_class_counts":
            np.bincount(
                y_validation.astype(
                    np.int64
                ),
                minlength=NUM_CLASSES,
            ).tolist(),

        "test_class_counts":
            np.bincount(
                y_test.astype(
                    np.int64
                ),
                minlength=NUM_CLASSES,
            ).tolist(),
    },

    "model": {
        "parameter_count":
            parameter_count,

        "model_kwargs":
            model_kwargs,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "batch_size":
            batch_size,

        "maximum_epochs":
            maximum_epochs,

        "early_stopping_patience":
            patience,

        "early_stopping_min_delta":
            minimum_delta,

        "class_weights":
            class_weights.tolist(),
    },

    "training": {
        "resumed":
            resumed,

        "completed_epoch":
            completed_epoch,

        "stopped_early":
            stopped_early,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_validation_macro_f1,

        "history":
            history,
    },

    "test": {
        "evaluation_count":
            1,

        "evaluation_seconds":
            test_seconds,

        "loss":
            test_result["loss"],

        "fingerprint_level":
            test_result[
                "fingerprint"
            ],

        "heldout_record_weighted":
            test_result[
                "record_weighted"
            ],
    },

    "runtime": {
        "total_seconds":
            total_seconds,

        "test_seconds":
            test_seconds,
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
        "checkpoint":
            str(checkpoint_file),

        "best_model":
            str(best_model_file),

        "history":
            str(history_file),

        "result":
            str(result_file),

        "report":
            str(report_file),
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


fingerprint_metrics = (
    test_result[
        "fingerprint"
    ]
)

record_metrics = (
    test_result[
        "record_weighted"
    ]
)


report_lines = [
    "=" * 86,
    "EARLY LODO SCIENTIFIC RUN RESULT",
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
        f"Parameters                : "
        f"{parameter_count:,}"
    ),
    (
        f"Completed epoch           : "
        f"{completed_epoch}"
    ),
    (
        f"Best epoch                : "
        f"{best_epoch}"
    ),
    (
        f"Best validation Macro-F1 : "
        f"{best_validation_macro_f1:.9f}"
    ),
    (
        f"Stopped early             : "
        f"{stopped_early}"
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
    "",
    "HELD-OUT RECORD-WEIGHTED TEST",
    (
        f"Accuracy                  : "
        f"{record_metrics['accuracy']:.9f}"
    ),
    (
        f"Balanced accuracy         : "
        f"{record_metrics['balanced_accuracy']:.9f}"
    ),
    (
        f"Macro-F1                  : "
        f"{record_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Weighted-F1               : "
        f"{record_metrics['weighted_f1']:.9f}"
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


print()
print("=" * 86)
print("SCIENTIFIC RUN SUMMARY")
print("=" * 86)
print(f"Run id                   : {run_id}")
print(f"Completed epoch          : {completed_epoch}")
print(f"Best epoch               : {best_epoch}")
print(
    f"Best validation Macro-F1: "
    f"{best_validation_macro_f1:.9f}"
)
print(
    f"Test fingerprint Macro-F1: "
    f"{fingerprint_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Test fingerprint accuracy: "
    f"{fingerprint_metrics['accuracy']:.9f}"
)
print(
    f"Record-weighted Macro-F1 : "
    f"{record_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Stopped early            : "
    f"{stopped_early}"
)
print(
    f"Total seconds            : "
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
print(f"Result     : {result_file}")
print(f"History    : {history_file}")
print(f"Best model : {best_model_file}")
print(f"Checkpoint : {checkpoint_file}")
print(f"Report     : {report_file}")

if not all_checks_passed:
    print()
    print(
        "EARLY LODO SCIENTIFIC "
        "RUN FAILED"
    )
    sys.exit(1)

print()
print(
    "EARLY LODO SCIENTIFIC "
    "RUN COMPLETED"
)
