from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

# Betik scripts klasöründen doğrudan çalıştırıldığında
# proje kökünü Python modül arama yoluna ekle.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT_BOOTSTRAP),
    )

from src.models.nbaiot_models import create_model


PROJECT_ROOT = Path.cwd()

PROTOCOL_FILE = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_numeric_order_training_pilot_v2.json"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "numeric_order_pilot_v2"
)

CACHE_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "numeric_order_pilot_cache_summary_v2.json"
)

CACHE_MANIFEST_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "numeric_order_pilot_cache_manifest_v2.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "numeric_order_training_pilot_v2"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

SMOKE_SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_smoke_v2.json"
)

RUN_SUMMARY_CSV = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_runs_v2.csv"
)

PAIRED_DELTA_CSV = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_paired_deltas_v2.csv"
)

FINAL_SUMMARY_JSON = (
    AUDIT_DIRECTORY
    / "numeric_order_training_pilot_summary_v2.json"
)


FEATURE_COUNT = 115
CLASS_COUNT = 3

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

EXPECTED_ROWS = {
    "train": 1_738_133,
    "validation": 371_884,
    "test": 372_659,
}

EXPECTED_PIPELINES = {
    "pipeline_a",
    "pipeline_b",
}

EXPECTED_MODELS = {
    "tinyml_mlp",
    "compact_dnn",
}

EXPECTED_SEEDS = {
    42,
    123,
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def save_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
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
        temporary_path,
        path,
    )


def set_reproducible_seed(
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


def state_dict_sha256(
    model: nn.Module,
) -> str:
    digest = hashlib.sha256()

    state = model.state_dict()

    for key in sorted(state):
        tensor = (
            state[key]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            key.encode("utf-8")
        )

        digest.update(
            str(tensor.dtype).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(tensor.shape)
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            tensor.numpy().tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def open_memmap(
    path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Önbellek dosyası yok: {path}"
        )

    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"{path.name}: boyut uyuşmuyor. "
            f"Beklenen={expected_shape}, "
            f"bulunan={array.shape}"
        )

    if array.dtype != np.dtype(
        expected_dtype
    ):
        raise RuntimeError(
            f"{path.name}: dtype uyuşmuyor. "
            f"Beklenen={np.dtype(expected_dtype)}, "
            f"bulunan={array.dtype}"
        )

    return array


def load_arrays(
    pipeline_name: str,
) -> dict[str, np.ndarray]:
    if pipeline_name not in EXPECTED_PIPELINES:
        raise ValueError(
            f"Geçersiz pipeline: {pipeline_name}"
        )

    pipeline_directory = (
        CACHE_ROOT
        / pipeline_name
    )

    shared_directory = (
        CACHE_ROOT
        / "shared"
    )

    arrays: dict[str, np.ndarray] = {}

    for split_name, row_count in (
        EXPECTED_ROWS.items()
    ):
        arrays[
            f"x_{split_name}"
        ] = open_memmap(
            pipeline_directory
            / f"x_{split_name}.npy",
            (
                row_count,
                FEATURE_COUNT,
            ),
            np.float32,
        )

        arrays[
            f"y_{split_name}"
        ] = open_memmap(
            shared_directory
            / f"y_{split_name}.npy",
            (
                row_count,
            ),
            np.int8,
        )

    arrays[
        "class_weights"
    ] = open_memmap(
        shared_directory
        / "class_weights.npy",
        (
            CLASS_COUNT,
        ),
        np.float32,
    )

    return arrays


def numpy_batch(
    features: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray | slice,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    feature_batch = np.array(
        features[indexes],
        dtype=np.float32,
        copy=True,
        order="C",
    )

    label_batch = np.array(
        labels[indexes],
        dtype=np.int64,
        copy=True,
        order="C",
    )

    return (
        torch.from_numpy(
            feature_batch
        ),
        torch.from_numpy(
            label_batch
        ),
    )


def confusion_from_predictions(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> np.ndarray:
    flattened = (
        targets.to(
            dtype=torch.int64
        )
        * CLASS_COUNT
        + predictions.to(
            dtype=torch.int64
        )
    )

    counts = torch.bincount(
        flattened,
        minlength=(
            CLASS_COUNT
            * CLASS_COUNT
        ),
    )

    return (
        counts.reshape(
            CLASS_COUNT,
            CLASS_COUNT,
        )
        .cpu()
        .numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )


def metrics_from_confusion(
    confusion: np.ndarray,
) -> dict[str, Any]:
    confusion_float = np.asarray(
        confusion,
        dtype=np.float64,
    )

    total = float(
        confusion_float.sum()
    )

    correct = float(
        np.trace(
            confusion_float
        )
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    per_class = {}
    f1_values = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        true_positive = float(
            confusion_float[
                class_index,
                class_index,
            ]
        )

        support = float(
            confusion_float[
                class_index,
                :
            ].sum()
        )

        predicted = float(
            confusion_float[
                :,
                class_index,
            ].sum()
        )

        recall = (
            true_positive / support
            if support > 0
            else None
        )

        precision = (
            true_positive / predicted
            if predicted > 0
            else 0.0
        )

        if recall is None:
            f1 = None
            false_negative_rate = None

        else:
            false_negative_rate = (
                1.0 - recall
            )

            denominator = (
                precision + recall
            )

            f1 = (
                2.0
                * precision
                * recall
                / denominator
                if denominator > 0
                else 0.0
            )

            f1_values.append(f1)

        per_class[
            class_name
        ] = {
            "support":
                int(support),

            "precision":
                precision,

            "recall":
                recall,

            "fnr":
                false_negative_rate,

            "f1":
                f1,
        }

    macro_f1 = (
        float(
            np.mean(
                np.asarray(
                    f1_values,
                    dtype=np.float64,
                )
            )
        )
        if f1_values
        else 0.0
    )

    return {
        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "per_class":
            per_class,

        "confusion_matrix":
            confusion.tolist(),
    }


def evaluate(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()

    confusion = np.zeros(
        (
            CLASS_COUNT,
            CLASS_COUNT,
        ),
        dtype=np.int64,
    )

    loss_sum = 0.0
    row_count = int(
        labels.shape[0]
    )

    with torch.inference_mode():
        for start in range(
            0,
            row_count,
            batch_size,
        ):
            end = min(
                start + batch_size,
                row_count,
            )

            feature_batch, label_batch = (
                numpy_batch(
                    features,
                    labels,
                    slice(start, end),
                )
            )

            logits = model(
                feature_batch
            )

            loss = criterion(
                logits,
                label_batch,
            )

            predictions = logits.argmax(
                dim=1
            )

            batch_rows = int(
                end - start
            )

            loss_sum += (
                float(loss.item())
                * batch_rows
            )

            confusion += (
                confusion_from_predictions(
                    label_batch,
                    predictions,
                )
            )

    metrics = metrics_from_confusion(
        confusion
    )

    metrics[
        "loss"
    ] = (
        loss_sum / row_count
    )

    metrics[
        "row_count"
    ] = row_count

    return metrics


def build_model_and_optimizer(
    model_name: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    str,
]:
    set_reproducible_seed(seed)

    model = create_model(
        model_name=model_name,
        input_features=FEATURE_COUNT,
        num_classes=CLASS_COUNT,
    )

    initial_hash = (
        state_dict_sha256(
            model
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return (
        model,
        optimizer,
        initial_hash,
    )


def run_smoke_test(
    protocol: dict[str, Any],
) -> None:
    print("=" * 78)
    print("FAZ 1E-B — EĞİTİM SMOKE TESTİ")
    print("=" * 78)

    smoke_rows = []

    batch_size = int(
        protocol[
            "training"
        ][
            "batch_size"
        ]
    )

    weight_decay = float(
        protocol[
            "training"
        ][
            "weight_decay"
        ]
    )

    for pipeline_name in (
        "pipeline_a",
        "pipeline_b",
    ):
        arrays = load_arrays(
            pipeline_name
        )

        class_weights = torch.from_numpy(
            np.array(
                arrays[
                    "class_weights"
                ],
                dtype=np.float32,
                copy=True,
            )
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        train_indexes = slice(
            0,
            batch_size,
        )

        validation_indexes = slice(
            0,
            batch_size,
        )

        for model_name in (
            "tinyml_mlp",
            "compact_dnn",
        ):
            model_protocol = (
                protocol[
                    "models"
                ][
                    model_name
                ]
            )

            learning_rate = float(
                model_protocol[
                    "learning_rate"
                ]
            )

            model, optimizer, initial_hash = (
                build_model_and_optimizer(
                    model_name=model_name,
                    seed=42,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
            )

            model.train()

            train_features, train_labels = (
                numpy_batch(
                    arrays[
                        "x_train"
                    ],
                    arrays[
                        "y_train"
                    ],
                    train_indexes,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                train_features
            )

            if logits.shape != (
                batch_size,
                CLASS_COUNT,
            ):
                raise RuntimeError(
                    f"{pipeline_name}/"
                    f"{model_name}: çıktı "
                    f"boyutu hatalı: "
                    f"{logits.shape}"
                )

            loss = criterion(
                logits,
                train_labels,
            )

            if not math.isfinite(
                float(loss.item())
            ):
                raise RuntimeError(
                    f"{pipeline_name}/"
                    f"{model_name}: "
                    "loss sonlu değil."
                )

            loss.backward()
            optimizer.step()

            model.eval()

            with torch.inference_mode():
                validation_features, (
                    validation_labels
                ) = numpy_batch(
                    arrays[
                        "x_validation"
                    ],
                    arrays[
                        "y_validation"
                    ],
                    validation_indexes,
                )

                validation_logits = model(
                    validation_features
                )

                validation_predictions = (
                    validation_logits.argmax(
                        dim=1
                    )
                )

                validation_confusion = (
                    confusion_from_predictions(
                        validation_labels,
                        validation_predictions,
                    )
                )

                validation_metrics = (
                    metrics_from_confusion(
                        validation_confusion
                    )
                )

            smoke_row = {
                "pipeline":
                    pipeline_name,

                "model":
                    model_name,

                "batch_size":
                    batch_size,

                "initial_state_sha256":
                    initial_hash,

                "training_loss":
                    float(loss.item()),

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_macro_f1":
                    validation_metrics[
                        "macro_f1"
                    ],

                "output_shape":
                    list(
                        logits.shape
                    ),

                "passed":
                    True,
            }

            smoke_rows.append(
                smoke_row
            )

            print(
                f"{pipeline_name} | "
                f"{model_name} | "
                f"loss={loss.item():.9f} | "
                f"val_macro_f1="
                f"{validation_metrics['macro_f1']:.9f} | "
                "PASSED"
            )

    matched_hash_checks = {}

    for model_name in (
        "tinyml_mlp",
        "compact_dnn",
    ):
        hashes = {
            row[
                "initial_state_sha256"
            ]
            for row in smoke_rows
            if row["model"] == model_name
        }

        matched_hash_checks[
            model_name
        ] = len(hashes) == 1

    summary = {
        "protocol_version":
            "numeric_order_training_smoke_v2_1",

        "status":
            "completed",

        "completed_at":
            utc_now(),

        "batch_size":
            batch_size,

        "combination_count":
            len(smoke_rows),

        "rows":
            smoke_rows,

        "matched_initialization_checks":
            matched_hash_checks,

        "all_checks_passed":
            (
                len(smoke_rows) == 4
                and all(
                    row["passed"]
                    for row in smoke_rows
                )
                and all(
                    matched_hash_checks.values()
                )
            ),
    }

    save_json(
        SMOKE_SUMMARY_FILE,
        summary,
    )

    print()
    print("MATCHED INITIALIZATION CHECKS")

    for name, passed in (
        matched_hash_checks.items()
    ):
        print(f"{name}: {passed}")

    print()
    print(f"Smoke özeti: {SMOKE_SUMMARY_FILE}")

    if not summary[
        "all_checks_passed"
    ]:
        print()
        print("FAZ 1E-B SMOKE TESTİ BAŞARISIZ")
        sys.exit(1)

    print()
    print("FAZ 1E-B SMOKE TESTİ BAŞARILI")


def save_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def run_training_condition(
    protocol: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(
        run_spec["run_id"]
    )

    pipeline_name = str(
        run_spec[
            "numeric_pipeline"
        ]
    )

    model_name = str(
        run_spec["model_name"]
    )

    seed = int(
        run_spec["seed"]
    )

    learning_rate = float(
        run_spec[
            "learning_rate"
        ]
    )

    run_directory = (
        OUTPUT_ROOT
        / pipeline_name
        / model_name
        / f"seed{seed}"
    )

    state_file = (
        run_directory
        / "run_state.json"
    )

    metrics_file = (
        run_directory
        / "metrics.json"
    )

    checkpoint_file = (
        run_directory
        / "best_checkpoint.pt"
    )

    history_file = (
        run_directory
        / "history.csv"
    )

    if state_file.exists():
        state = load_json(
            state_file
        )

        if (
            state.get("status")
            == "completed"
            and metrics_file.exists()
            and checkpoint_file.exists()
            and history_file.exists()
        ):
            print(
                f"{run_id}: tamamlanmış, "
                "atlanıyor."
            )

            return load_json(
                metrics_file
            )

    if run_directory.exists():
        shutil.rmtree(
            run_directory
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        state_file,
        {
            "run_id":
                run_id,

            "status":
                "running",

            "started_at":
                utc_now(),
        },
    )

    arrays = load_arrays(
        pipeline_name
    )

    class_weights_array = np.array(
        arrays[
            "class_weights"
        ],
        dtype=np.float32,
        copy=True,
    )

    class_weights = torch.from_numpy(
        class_weights_array
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    training_settings = (
        protocol[
            "training"
        ]
    )

    batch_size = int(
        training_settings[
            "batch_size"
        ]
    )

    max_epochs = int(
        training_settings[
            "max_epochs"
        ]
    )

    patience = int(
        training_settings[
            "early_stopping_patience"
        ]
    )

    minimum_delta = float(
        training_settings[
            "early_stopping_min_delta"
        ]
    )

    weight_decay = float(
        training_settings[
            "weight_decay"
        ]
    )

    model, optimizer, initial_hash = (
        build_model_and_optimizer(
            model_name=model_name,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    )

    process = psutil.Process(
        os.getpid()
    )

    peak_memory_bytes = int(
        process.memory_info().rss
    )

    training_started = (
        time.perf_counter()
    )

    history_rows = []

    best_validation_macro_f1 = (
        -math.inf
    )

    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0

    train_row_count = int(
        arrays[
            "y_train"
        ].shape[0]
    )

    print()
    print("=" * 78)
    print(f"RUN: {run_id}")
    print("=" * 78)
    print(
        f"pipeline={pipeline_name} | "
        f"model={model_name} | "
        f"seed={seed} | "
        f"lr={learning_rate}"
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        epoch_started = (
            time.perf_counter()
        )

        model.train()

        permutation_rng = (
            np.random.default_rng(
                seed
                + epoch
                * 1_000_003
            )
        )

        permutation = (
            permutation_rng.permutation(
                train_row_count
            )
        )

        train_confusion = np.zeros(
            (
                CLASS_COUNT,
                CLASS_COUNT,
            ),
            dtype=np.int64,
        )

        train_loss_sum = 0.0

        batch_number = 0

        for start in range(
            0,
            train_row_count,
            batch_size,
        ):
            end = min(
                start + batch_size,
                train_row_count,
            )

            batch_indexes = permutation[
                start:end
            ]

            feature_batch, label_batch = (
                numpy_batch(
                    arrays[
                        "x_train"
                    ],
                    arrays[
                        "y_train"
                    ],
                    batch_indexes,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                feature_batch
            )

            loss = criterion(
                logits,
                label_batch,
            )

            if not math.isfinite(
                float(loss.item())
            ):
                raise RuntimeError(
                    f"{run_id}/epoch{epoch}: "
                    "loss sonlu değil."
                )

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(
                dim=1
            )

            batch_rows = int(
                end - start
            )

            train_loss_sum += (
                float(loss.item())
                * batch_rows
            )

            train_confusion += (
                confusion_from_predictions(
                    label_batch,
                    predictions,
                )
            )

            batch_number += 1

            if batch_number % 25 == 0:
                peak_memory_bytes = max(
                    peak_memory_bytes,
                    int(
                        process
                        .memory_info()
                        .rss
                    ),
                )

            if (
                batch_number == 1
                or batch_number % 100 == 0
                or end == train_row_count
            ):
                print(
                    f"epoch={epoch:02d} | "
                    f"train="
                    f"{end:,}/"
                    f"{train_row_count:,}"
                )

        del permutation

        train_metrics = (
            metrics_from_confusion(
                train_confusion
            )
        )

        train_metrics[
            "loss"
        ] = (
            train_loss_sum
            / train_row_count
        )

        validation_metrics = evaluate(
            model=model,
            features=arrays[
                "x_validation"
            ],
            labels=arrays[
                "y_validation"
            ],
            criterion=criterion,
            batch_size=batch_size,
        )

        validation_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = (
            best_epoch == 0
            or
            validation_macro_f1
            >
            best_validation_macro_f1
            + minimum_delta
        )

        if improved:
            best_validation_macro_f1 = (
                validation_macro_f1
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_file,
                {
                    "run_id":
                        run_id,

                    "pipeline":
                        pipeline_name,

                    "model_name":
                        model_name,

                    "seed":
                        seed,

                    "epoch":
                        epoch,

                    "validation_macro_f1":
                        validation_macro_f1,

                    "initial_state_sha256":
                        initial_hash,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),
                },
            )

        else:
            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
        )

        history_rows.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_metrics[
                        "loss"
                    ],

                "train_accuracy":
                    train_metrics[
                        "accuracy"
                    ],

                "train_macro_f1":
                    train_metrics[
                        "macro_f1"
                    ],

                "validation_loss":
                    validation_metrics[
                        "loss"
                    ],

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_macro_f1":
                    validation_macro_f1,

                "improved":
                    improved,

                "epochs_without_improvement":
                    epochs_without_improvement,

                "epoch_seconds":
                    epoch_seconds,
            }
        )

        epochs_completed = epoch

        print(
            f"epoch={epoch:02d} | "
            f"train_f1="
            f"{train_metrics['macro_f1']:.9f} | "
            f"val_f1="
            f"{validation_macro_f1:.9f} | "
            f"best="
            f"{best_validation_macro_f1:.9f} | "
            f"wait="
            f"{epochs_without_improvement}/"
            f"{patience} | "
            f"seconds={epoch_seconds:.1f}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping: "
                f"epoch {epoch}"
            )
            break

    if not checkpoint_file.exists():
        raise RuntimeError(
            f"{run_id}: checkpoint "
            "oluşturulamadı."
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_validation_metrics = (
        evaluate(
            model=model,
            features=arrays[
                "x_validation"
            ],
            labels=arrays[
                "y_validation"
            ],
            criterion=criterion,
            batch_size=batch_size,
        )
    )

    # Test tam olarak bir kez değerlendirilir.
    test_metrics = evaluate(
        model=model,
        features=arrays[
            "x_test"
        ],
        labels=arrays[
            "y_test"
        ],
        criterion=criterion,
        batch_size=batch_size,
    )

    training_seconds = (
        time.perf_counter()
        - training_started
    )

    peak_memory_bytes = max(
        peak_memory_bytes,
        int(
            process.memory_info().rss
        ),
    )

    write_csv(
        history_file,
        history_rows,
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
            "improved",
            "epochs_without_improvement",
            "epoch_seconds",
        ],
    )

    result = {
        "protocol_version":
            "numeric_order_training_run_v2_1",

        "status":
            "completed",

        "completed_at":
            utc_now(),

        "run_id":
            run_id,

        "pipeline":
            pipeline_name,

        "model_name":
            model_name,

        "seed":
            seed,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "batch_size":
            batch_size,

        "max_epochs":
            max_epochs,

        "patience":
            patience,

        "minimum_delta":
            minimum_delta,

        "initial_state_sha256":
            initial_hash,

        "best_epoch":
            best_epoch,

        "epochs_completed":
            epochs_completed,

        "training_seconds":
            training_seconds,

        "peak_process_memory_mb":
            (
                peak_memory_bytes
                / (1024 ** 2)
            ),

        "validation":
            best_validation_metrics,

        "test":
            test_metrics,

        "test_evaluation_count":
            1,

        "checkpoint_file":
            str(checkpoint_file),

        "history_file":
            str(history_file),
    }

    save_json(
        metrics_file,
        result,
    )

    save_json(
        state_file,
        {
            "run_id":
                run_id,

            "status":
                "completed",

            "completed_at":
                utc_now(),

            "metrics_file":
                str(metrics_file),

            "checkpoint_file":
                str(checkpoint_file),

            "history_file":
                str(history_file),
        },
    )

    print(
        f"{run_id} tamamlandı | "
        f"best_epoch={best_epoch} | "
        f"val_f1="
        f"{best_validation_metrics['macro_f1']:.9f} | "
        f"test_f1="
        f"{test_metrics['macro_f1']:.9f}"
    )

    return result


def aggregate_results(
    protocol: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    if len(results) != 8:
        raise RuntimeError(
            "Sekiz tamamlanmış sonuç "
            f"bekleniyordu, bulunan "
            f"{len(results)}."
        )

    run_rows = []

    for result in results:
        run_rows.append(
            {
                "run_id":
                    result["run_id"],

                "pipeline":
                    result["pipeline"],

                "model_name":
                    result["model_name"],

                "seed":
                    result["seed"],

                "learning_rate":
                    result[
                        "learning_rate"
                    ],

                "initial_state_sha256":
                    result[
                        "initial_state_sha256"
                    ],

                "best_epoch":
                    result["best_epoch"],

                "epochs_completed":
                    result[
                        "epochs_completed"
                    ],

                "validation_loss":
                    result[
                        "validation"
                    ][
                        "loss"
                    ],

                "validation_accuracy":
                    result[
                        "validation"
                    ][
                        "accuracy"
                    ],

                "validation_macro_f1":
                    result[
                        "validation"
                    ][
                        "macro_f1"
                    ],

                "test_loss":
                    result[
                        "test"
                    ][
                        "loss"
                    ],

                "test_accuracy":
                    result[
                        "test"
                    ][
                        "accuracy"
                    ],

                "test_macro_f1":
                    result[
                        "test"
                    ][
                        "macro_f1"
                    ],

                "training_seconds":
                    result[
                        "training_seconds"
                    ],

                "peak_process_memory_mb":
                    result[
                        "peak_process_memory_mb"
                    ],

                "test_evaluation_count":
                    result[
                        "test_evaluation_count"
                    ],
            }
        )

    write_csv(
        RUN_SUMMARY_CSV,
        run_rows,
        [
            "run_id",
            "pipeline",
            "model_name",
            "seed",
            "learning_rate",
            "initial_state_sha256",
            "best_epoch",
            "epochs_completed",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
            "test_loss",
            "test_accuracy",
            "test_macro_f1",
            "training_seconds",
            "peak_process_memory_mb",
            "test_evaluation_count",
        ],
    )

    indexed = {
        (
            row["pipeline"],
            row["model_name"],
            int(row["seed"]),
        ): row
        for row in run_rows
    }

    paired_rows = []
    validation_deltas = []

    initialization_checks = {}

    for model_name in (
        "tinyml_mlp",
        "compact_dnn",
    ):
        for seed in (
            42,
            123,
        ):
            row_a = indexed[
                (
                    "pipeline_a",
                    model_name,
                    seed,
                )
            ]

            row_b = indexed[
                (
                    "pipeline_b",
                    model_name,
                    seed,
                )
            ]

            initialization_match = (
                row_a[
                    "initial_state_sha256"
                ]
                ==
                row_b[
                    "initial_state_sha256"
                ]
            )

            initialization_checks[
                f"{model_name}_seed{seed}"
            ] = initialization_match

            validation_delta = (
                float(
                    row_b[
                        "validation_macro_f1"
                    ]
                )
                -
                float(
                    row_a[
                        "validation_macro_f1"
                    ]
                )
            )

            test_delta = (
                float(
                    row_b[
                        "test_macro_f1"
                    ]
                )
                -
                float(
                    row_a[
                        "test_macro_f1"
                    ]
                )
            )

            validation_deltas.append(
                validation_delta
            )

            paired_rows.append(
                {
                    "model_name":
                        model_name,

                    "seed":
                        seed,

                    "pipeline_a_validation_macro_f1":
                        row_a[
                            "validation_macro_f1"
                        ],

                    "pipeline_b_validation_macro_f1":
                        row_b[
                            "validation_macro_f1"
                        ],

                    "validation_delta_b_minus_a":
                        validation_delta,

                    "pipeline_a_test_macro_f1":
                        row_a[
                            "test_macro_f1"
                        ],

                    "pipeline_b_test_macro_f1":
                        row_b[
                            "test_macro_f1"
                        ],

                    "test_delta_b_minus_a":
                        test_delta,

                    "initialization_match":
                        initialization_match,

                    "winner":
                        (
                            "pipeline_b"
                            if validation_delta > 0
                            else
                            "pipeline_a"
                            if validation_delta < 0
                            else
                            "tie"
                        ),
                }
            )

    write_csv(
        PAIRED_DELTA_CSV,
        paired_rows,
        [
            "model_name",
            "seed",
            "pipeline_a_validation_macro_f1",
            "pipeline_b_validation_macro_f1",
            "validation_delta_b_minus_a",
            "pipeline_a_test_macro_f1",
            "pipeline_b_test_macro_f1",
            "test_delta_b_minus_a",
            "initialization_match",
            "winner",
        ],
    )

    median_delta = float(
        median(
            validation_deltas
        )
    )

    pipeline_b_wins = sum(
        delta > 0
        for delta in validation_deltas
    )

    pipeline_a_wins = sum(
        delta < 0
        for delta in validation_deltas
    )

    equivalence_margin = float(
        protocol[
            "primary_analysis"
        ][
            "practical_equivalence_margin"
        ]
    )

    if (
        median_delta
        >= equivalence_margin
        and pipeline_b_wins >= 3
    ):
        decision = (
            "select_pipeline_b"
        )

        selected_pipeline = (
            "pipeline_b"
        )

    elif (
        median_delta
        <= -equivalence_margin
        and pipeline_a_wins >= 3
    ):
        decision = (
            "select_pipeline_a"
        )

        selected_pipeline = (
            "pipeline_a"
        )

    else:
        decision = (
            "practically_equivalent_"
            "retain_pipeline_a"
        )

        selected_pipeline = (
            "pipeline_a"
        )

    validation_checks = {
        "run_count":
            len(results) == 8,

        "paired_count":
            len(paired_rows) == 4,

        "all_initializations_matched":
            all(
                initialization_checks.values()
            ),

        "all_test_evaluated_once":
            all(
                int(
                    result[
                        "test_evaluation_count"
                    ]
                )
                == 1
                for result in results
            ),

        "all_validation_metrics_finite":
            all(
                math.isfinite(
                    float(
                        result[
                            "validation"
                        ][
                            "macro_f1"
                        ]
                    )
                )
                for result in results
            ),

        "all_test_metrics_finite":
            all(
                math.isfinite(
                    float(
                        result[
                            "test"
                        ][
                            "macro_f1"
                        ]
                    )
                )
                for result in results
            ),
    }

    summary = {
        "protocol_version":
            "numeric_order_training_pilot_v2_1",

        "status":
            "completed",

        "completed_at":
            utc_now(),

        "run_count":
            len(results),

        "paired_count":
            len(paired_rows),

        "validation_deltas_b_minus_a":
            validation_deltas,

        "median_validation_delta_b_minus_a":
            median_delta,

        "pipeline_b_win_count":
            pipeline_b_wins,

        "pipeline_a_win_count":
            pipeline_a_wins,

        "practical_equivalence_margin":
            equivalence_margin,

        "decision":
            decision,

        "selected_pipeline":
            selected_pipeline,

        "selection_used_test_metrics":
            False,

        "initialization_checks":
            initialization_checks,

        "validation_checks":
            validation_checks,

        "all_checks_passed":
            all(
                validation_checks.values()
            ),

        "run_summary_csv":
            str(RUN_SUMMARY_CSV),

        "paired_delta_csv":
            str(PAIRED_DELTA_CSV),
    }

    save_json(
        FINAL_SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 78)
    print("FAZ 1E EĞİTİM PİLOTU ÖZETİ")
    print("=" * 78)
    print(
        f"Median validation delta B-A: "
        f"{median_delta:+.9f}"
    )
    print(
        f"Pipeline B wins            : "
        f"{pipeline_b_wins}/4"
    )
    print(
        f"Pipeline A wins            : "
        f"{pipeline_a_wins}/4"
    )
    print(
        f"Karar                      : "
        f"{decision}"
    )
    print(
        f"Seçilen sayısal hat        : "
        f"{selected_pipeline}"
    )
    print()
    print("PAIRED DELTAS")

    for row in paired_rows:
        print(
            f"{row['model_name']} | "
            f"seed={row['seed']} | "
            f"val_delta="
            f"{row['validation_delta_b_minus_a']:+.9f} | "
            f"test_delta="
            f"{row['test_delta_b_minus_a']:+.9f} | "
            f"init_match="
            f"{row['initialization_match']}"
        )

    print()
    print("VALIDATION CHECKS")

    for name, passed in (
        validation_checks.items()
    ):
        print(f"{name}: {passed}")

    print()
    print(f"Run özeti : {RUN_SUMMARY_CSV}")
    print(f"Delta özeti: {PAIRED_DELTA_CSV}")
    print(f"Karar JSON : {FINAL_SUMMARY_JSON}")

    if not summary[
        "all_checks_passed"
    ]:
        print()
        print("FAZ 1E EĞİTİM PİLOTU BAŞARISIZ")
        sys.exit(1)

    print()
    print("FAZ 1E EĞİTİM PİLOTU BAŞARILI")


def run_full_training(
    protocol: dict[str, Any],
    requested_run_id: str | None,
) -> None:
    run_matrix = list(
        protocol[
            "run_matrix"
        ]
    )

    if len(run_matrix) != 8:
        raise RuntimeError(
            "Protokolde sekiz koşul yok."
        )

    if requested_run_id is not None:
        run_matrix = [
            run
            for run in run_matrix
            if run["run_id"]
            == requested_run_id
        ]

        if len(run_matrix) != 1:
            raise ValueError(
                "İstenen run_id protokolde "
                f"bulunamadı: {requested_run_id}"
            )

    results = []

    for run_spec in run_matrix:
        result = (
            run_training_condition(
                protocol,
                run_spec,
            )
        )

        results.append(result)

    if requested_run_id is None:
        aggregate_results(
            protocol,
            results,
        )


def validate_inputs() -> dict[str, Any]:
    required_files = [
        PROTOCOL_FILE,
        CACHE_SUMMARY_FILE,
        CACHE_MANIFEST_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print("Eksik giriş dosyaları:")

        for path in missing_files:
            print(f"- {path}")

        sys.exit(1)

    protocol = load_json(
        PROTOCOL_FILE
    )

    cache_summary = load_json(
        CACHE_SUMMARY_FILE
    )

    if (
        protocol.get("status")
        != "locked_before_training"
    ):
        raise RuntimeError(
            "Pilot protokolü kilitli değil."
        )

    if protocol.get(
        "run_count"
    ) != 8:
        raise RuntimeError(
            "Pilot run sayısı sekiz değil."
        )

    if cache_summary.get(
        "all_checks_passed"
    ) is not True:
        raise RuntimeError(
            "Pilot önbelleği doğrulanmamış."
        )

    observed_pipelines = {
        str(
            run[
                "numeric_pipeline"
            ]
        )
        for run in protocol[
            "run_matrix"
        ]
    }

    observed_models = {
        str(
            run[
                "model_name"
            ]
        )
        for run in protocol[
            "run_matrix"
        ]
    }

    observed_seeds = {
        int(
            run["seed"]
        )
        for run in protocol[
            "run_matrix"
        ]
    }

    if (
        observed_pipelines
        != EXPECTED_PIPELINES
        or observed_models
        != EXPECTED_MODELS
        or observed_seeds
        != EXPECTED_SEEDS
    ):
        raise RuntimeError(
            "Pilot koşul matrisi beklenen "
            "2×2×2 yapısında değil."
        )

    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT sayısal işlem sırası "
            "8 koşuluk eğitim pilotu."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Dört model-pipeline birleşimini "
            "tek batch ile sınar."
        ),
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Yalnız belirtilen run_id "
            "koşulunu çalıştırır."
        ),
    )

    arguments = parser.parse_args()

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    thread_count = min(
        8,
        max(
            1,
            os.cpu_count() or 1,
        ),
    )

    torch.set_num_threads(
        thread_count
    )

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    protocol = validate_inputs()

    if arguments.smoke:
        run_smoke_test(
            protocol
        )
        return

    run_full_training(
        protocol,
        arguments.run_id,
    )


if __name__ == "__main__":
    main()

