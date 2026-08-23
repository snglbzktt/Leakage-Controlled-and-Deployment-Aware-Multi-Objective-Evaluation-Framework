from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.models.nbaiot_models import create_model


SEED = 2026

HELD_OUT_DEVICE = (
    "Danmini_Doorbell"
)

MODEL_NAME = "tinyml_mlp"

INPUT_FEATURES = 115
NUM_CLASSES = 3

BATCH_SIZE = 4096
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001

TRAIN_QUOTA_PER_CLASS = 4096
VALIDATION_QUOTA_PER_CLASS = 2048
TEST_QUOTA_PER_CLASS = 2048

PARQUET_BATCH_SIZE = 50_000

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]


FEATURE_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_feature_alignment_v2"
    / "alignment.json"
)

MEMBERSHIP_ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
    / "alignment.json"
)

MEMBERSHIP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_membership_v2"
)

FEATURE_INDEX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_early_lodo_feature_alignment_v2"
)

DEVICE_MASK_FILE = (
    MEMBERSHIP_ROOT
    / "device_mask.npy"
)

FAMILY_FILE = (
    MEMBERSHIP_ROOT
    / "family3.npy"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "smoke"
    / "early_lodo"
)

OUTPUT_JSON = (
    OUTPUT_ROOT
    / "danmini_tinyml_mlp_smoke_v2.json"
)

OUTPUT_REPORT = (
    OUTPUT_ROOT
    / "danmini_tinyml_mlp_smoke_v2.txt"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(
    value: Any,
) -> Any:
    """Convert NumPy values to JSON-compatible Python values."""

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


def resolve_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path


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

    torch.set_num_threads(
        max(
            1,
            min(
                4,
                os.cpu_count() or 1,
            ),
        )
    )


def update_sample(
    storage: dict[
        int,
        dict[str, np.ndarray]
    ],
    class_id: int,
    candidate_features: np.ndarray,
    candidate_global_indices: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_source_codes: np.ndarray,
    quota: int,
) -> None:
    if len(candidate_scores) == 0:
        return

    if len(candidate_scores) > quota:
        selected = np.argpartition(
            candidate_scores,
            quota - 1,
        )[:quota]

        candidate_features = (
            candidate_features[
                selected
            ]
        )

        candidate_global_indices = (
            candidate_global_indices[
                selected
            ]
        )

        candidate_scores = (
            candidate_scores[
                selected
            ]
        )

        candidate_source_codes = (
            candidate_source_codes[
                selected
            ]
        )

    existing = storage.get(
        class_id
    )

    if existing is None:
        combined_features = (
            candidate_features
        )

        combined_indices = (
            candidate_global_indices
        )

        combined_scores = (
            candidate_scores
        )

        combined_sources = (
            candidate_source_codes
        )

    else:
        combined_features = (
            np.concatenate(
                [
                    existing["features"],
                    candidate_features,
                ],
                axis=0,
            )
        )

        combined_indices = np.concatenate(
            [
                existing["global_indices"],
                candidate_global_indices,
            ],
            axis=0,
        )

        combined_scores = np.concatenate(
            [
                existing["scores"],
                candidate_scores,
            ],
            axis=0,
        )

        combined_sources = np.concatenate(
            [
                existing["source_codes"],
                candidate_source_codes,
            ],
            axis=0,
        )

    if len(combined_scores) > quota:
        selected = np.argpartition(
            combined_scores,
            quota - 1,
        )[:quota]

        combined_features = (
            combined_features[
                selected
            ]
        )

        combined_indices = (
            combined_indices[
                selected
            ]
        )

        combined_scores = (
            combined_scores[
                selected
            ]
        )

        combined_sources = (
            combined_sources[
                selected
            ]
        )

    storage[class_id] = {
        "features":
            combined_features,

        "global_indices":
            combined_indices,

        "scores":
            combined_scores,

        "source_codes":
            combined_sources,
    }


def finalize_sample(
    storage: dict[
        int,
        dict[str, np.ndarray]
    ],
    quota: int,
    role: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    feature_parts = []
    label_parts = []
    index_parts = []
    source_parts = []

    for class_id in range(
        NUM_CLASSES
    ):
        item = storage.get(
            class_id
        )

        observed = (
            0
            if item is None
            else len(
                item["global_indices"]
            )
        )

        if observed < quota:
            raise RuntimeError(
                f"{role} class {class_id} "
                f"quota not met: "
                f"{observed}/{quota}"
            )

        order = np.argsort(
            item["scores"],
            kind="stable",
        )

        features = (
            item["features"][
                order
            ]
        )

        indices = (
            item["global_indices"][
                order
            ]
        )

        sources = (
            item["source_codes"][
                order
            ]
        )

        labels = np.full(
            quota,
            class_id,
            dtype=np.int64,
        )

        feature_parts.append(
            features
        )

        label_parts.append(
            labels
        )

        index_parts.append(
            indices
        )

        source_parts.append(
            sources
        )

    features = np.concatenate(
        feature_parts,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    labels = np.concatenate(
        label_parts,
        axis=0,
    )

    global_indices = np.concatenate(
        index_parts,
        axis=0,
    )

    source_codes = np.concatenate(
        source_parts,
        axis=0,
    )

    return (
        features,
        labels,
        global_indices,
        source_codes,
    )


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    present_classes = sorted(
        {
            int(value)
            for value in labels
        }
    )

    per_class = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        support = int(
            matrix[
                class_id,
                :,
            ].sum()
        )

        true_positive = int(
            matrix[
                class_id,
                class_id,
            ]
        )

        false_negative = (
            support
            - true_positive
        )

        predicted_count = int(
            matrix[
                :,
                class_id,
            ].sum()
        )

        false_positive = (
            predicted_count
            - true_positive
        )

        if support == 0:
            recall = None
            fnr = None
            f1_value = None

        else:
            recall = (
                true_positive
                / support
            )

            fnr = (
                false_negative
                / support
            )

            denominator = (
                2 * true_positive
                + false_positive
                + false_negative
            )

            f1_value = (
                0.0
                if denominator == 0
                else (
                    2
                    * true_positive
                    / denominator
                )
            )

        per_class[
            class_name
        ] = {
            "support":
                support,

            "recall":
                recall,

            "fnr":
                fnr,

            "f1":
                f1_value,
        }

    return {
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_f1_present_classes":
            float(
                f1_score(
                    labels,
                    predictions,
                    labels=present_classes,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "confusion_matrix":
            matrix.tolist(),

        "per_class":
            per_class,
    }


def evaluate_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
) -> tuple[
    float,
    np.ndarray,
]:
    dataset = TensorDataset(
        torch.from_numpy(
            features
        ),
        torch.from_numpy(
            labels
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    total_loss = 0.0
    total_examples = 0
    predictions = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            logits = model(
                batch_features
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            batch_count = int(
                batch_labels.shape[0]
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_count
            )

            total_examples += (
                batch_count
            )

            predictions.append(
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

    return (
        total_loss
        / total_examples,
        np.concatenate(
            predictions
        ),
    )


required_files = [
    FEATURE_ALIGNMENT_FILE,
    MEMBERSHIP_ALIGNMENT_FILE,
    DEVICE_MASK_FILE,
    FAMILY_FILE,
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


set_reproducible_seed(
    SEED
)

started_at = time.perf_counter()

feature_alignment = json.loads(
    FEATURE_ALIGNMENT_FILE.read_text(
        encoding="utf-8-sig",
    )
)

membership_alignment = json.loads(
    MEMBERSHIP_ALIGNMENT_FILE.read_text(
        encoding="utf-8-sig",
    )
)

feature_columns = (
    feature_alignment[
        "feature_columns"
    ]
)

if len(feature_columns) != INPUT_FEATURES:
    raise RuntimeError(
        "Feature count is not 115."
    )


device_entries = [
    item
    for item
    in membership_alignment[
        "devices"
    ]
    if item["device"]
    == HELD_OUT_DEVICE
]

if len(device_entries) != 1:
    raise RuntimeError(
        "Danmini device entry is invalid."
    )

device_entry = device_entries[0]

device_id = int(
    device_entry["device_id"]
)

held_out_bit = np.uint16(
    device_entry["bit_value"]
)


device_mask = np.load(
    DEVICE_MASK_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

family3 = np.load(
    FAMILY_FILE,
    mmap_mode="r",
    allow_pickle=False,
)


rng = np.random.default_rng(
    SEED
)

role_storage: dict[
    str,
    dict[
        int,
        dict[str, np.ndarray]
    ]
] = {
    "train": {},
    "validation": {},
    "test": {},
}

source_code_map = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


print("=" * 82)
print("PHASE 2F-1 EARLY LODO SMOKE TEST")
print("=" * 82)
print(f"Held-out device : {HELD_OUT_DEVICE}")
print(f"Model           : {MODEL_NAME}")
print(f"Seed            : {SEED}")
print()
print("Collecting smoke samples...")


for split_name in (
    "train",
    "validation",
    "test",
):
    split_entry = (
        feature_alignment[
            "splits"
        ][split_name]
    )

    feature_file = resolve_path(
        split_entry[
            "feature_file"
        ]
    )

    global_index_file = (
        FEATURE_INDEX_ROOT
        / (
            f"{split_name}_"
            f"global_indices.npy"
        )
    )

    global_indices = np.load(
        global_index_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    parquet_file = pq.ParquetFile(
        feature_file
    )

    local_offset = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=(
                PARQUET_BATCH_SIZE
            ),
            columns=feature_columns,
        ),
        start=1,
    ):
        batch_size = int(
            batch.num_rows
        )

        batch_global_indices = (
            global_indices[
                local_offset:
                local_offset + batch_size
            ]
        )

        batch_families = (
            family3[
                batch_global_indices
            ]
        )

        batch_device_mask = (
            device_mask[
                batch_global_indices
            ]
        )

        held_out_membership = (
            (
                batch_device_mask
                & held_out_bit
            )
            != 0
        )

        frame = batch.to_pandas()

        batch_features = frame.to_numpy(
            dtype=np.float32,
            copy=False,
        )

        if split_name == "train":
            role_name = "train"
            role_eligible = (
                ~held_out_membership
            )

            quota = (
                TRAIN_QUOTA_PER_CLASS
            )

            for class_id in range(
                NUM_CLASSES
            ):
                positions = np.flatnonzero(
                    role_eligible
                    & (
                        batch_families
                        == class_id
                    )
                )

                if len(positions) == 0:
                    continue

                update_sample(
                    role_storage[
                        role_name
                    ],
                    class_id,
                    batch_features[
                        positions
                    ],
                    np.asarray(
                        batch_global_indices[
                            positions
                        ],
                        dtype=np.int64,
                    ),
                    rng.random(
                        len(positions)
                    ),
                    np.full(
                        len(positions),
                        source_code_map[
                            split_name
                        ],
                        dtype=np.uint8,
                    ),
                    quota,
                )

        elif split_name == "validation":
            role_name = "validation"

            role_eligible = (
                ~held_out_membership
            )

            quota = (
                VALIDATION_QUOTA_PER_CLASS
            )

            for class_id in range(
                NUM_CLASSES
            ):
                positions = np.flatnonzero(
                    role_eligible
                    & (
                        batch_families
                        == class_id
                    )
                )

                if len(positions) == 0:
                    continue

                update_sample(
                    role_storage[
                        role_name
                    ],
                    class_id,
                    batch_features[
                        positions
                    ],
                    np.asarray(
                        batch_global_indices[
                            positions
                        ],
                        dtype=np.int64,
                    ),
                    rng.random(
                        len(positions)
                    ),
                    np.full(
                        len(positions),
                        source_code_map[
                            split_name
                        ],
                        dtype=np.uint8,
                    ),
                    quota,
                )

        test_eligible = (
            held_out_membership
        )

        for class_id in range(
            NUM_CLASSES
        ):
            positions = np.flatnonzero(
                test_eligible
                & (
                    batch_families
                    == class_id
                )
            )

            if len(positions) == 0:
                continue

            update_sample(
                role_storage["test"],
                class_id,
                batch_features[
                    positions
                ],
                np.asarray(
                    batch_global_indices[
                        positions
                    ],
                    dtype=np.int64,
                ),
                rng.random(
                    len(positions)
                ),
                np.full(
                    len(positions),
                    source_code_map[
                        split_name
                    ],
                    dtype=np.uint8,
                ),
                TEST_QUOTA_PER_CLASS,
            )

        local_offset += batch_size

    if local_offset != len(
        global_indices
    ):
        raise RuntimeError(
            f"{split_name} row count "
            "alignment failed."
        )

    print(
        f"{split_name} scanned: "
        f"{local_offset:,} rows"
    )


(
    x_train,
    y_train,
    train_global_indices,
    train_source_codes,
) = finalize_sample(
    role_storage["train"],
    TRAIN_QUOTA_PER_CLASS,
    "train",
)

(
    x_validation,
    y_validation,
    validation_global_indices,
    validation_source_codes,
) = finalize_sample(
    role_storage[
        "validation"
    ],
    VALIDATION_QUOTA_PER_CLASS,
    "validation",
)

(
    x_test,
    y_test,
    test_global_indices,
    test_source_codes,
) = finalize_sample(
    role_storage["test"],
    TEST_QUOTA_PER_CLASS,
    "test",
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

train_validation_overlap = int(
    np.intersect1d(
        train_global_indices,
        validation_global_indices,
    ).size
)


scaler = StandardScaler(
    copy=True
)

x_train_scaled = scaler.fit_transform(
    x_train.astype(
        np.float64,
        copy=False,
    )
).astype(
    np.float32
)

x_validation_scaled = scaler.transform(
    x_validation.astype(
        np.float64,
        copy=False,
    )
).astype(
    np.float32
)

x_test_scaled = scaler.transform(
    x_test.astype(
        np.float64,
        copy=False,
    )
).astype(
    np.float32
)


train_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES,
).astype(
    np.float64
)

class_weights = (
    1.0
    / np.sqrt(
        train_counts
    )
)

class_weights = (
    class_weights
    / class_weights.mean()
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32,
    )
)


model = create_model(
    MODEL_NAME,
    input_features=INPUT_FEATURES,
    num_classes=NUM_CLASSES,
)

parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


initial_parameters = [
    parameter.detach()
    .clone()
    for parameter
    in model.parameters()
]


train_dataset = TensorDataset(
    torch.from_numpy(
        x_train_scaled
    ),
    torch.from_numpy(
        y_train
    ),
)

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
    num_workers=0,
)


model.train()

total_training_loss = 0.0
total_training_examples = 0

for batch_features, batch_labels in (
    train_loader
):
    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        batch_features
    )

    loss = criterion(
        logits,
        batch_labels
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite training loss."
        )

    loss.backward()
    optimizer.step()

    batch_count = int(
        batch_labels.shape[0]
    )

    total_training_loss += (
        float(loss.item())
        * batch_count
    )

    total_training_examples += (
        batch_count
    )


training_loss = (
    total_training_loss
    / total_training_examples
)


parameter_change_sum = 0.0

for before, after in zip(
    initial_parameters,
    model.parameters(),
):
    parameter_change_sum += float(
        torch.sum(
            torch.abs(
                before
                - after.detach()
            )
        ).item()
    )


validation_loss, validation_predictions = (
    evaluate_model(
        model,
        x_validation_scaled,
        y_validation,
        criterion,
    )
)

test_loss, test_predictions = (
    evaluate_model(
        model,
        x_test_scaled,
        y_test,
        criterion,
    )
)


validation_metrics = calculate_metrics(
    y_validation,
    validation_predictions,
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions,
)


test_source_distribution = {
    split_name: int(
        np.count_nonzero(
            test_source_codes
            == source_code
        )
    )
    for split_name, source_code
    in source_code_map.items()
}


validation_checks = {
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

    "all_train_classes_present":
        np.all(
            np.bincount(
                y_train,
                minlength=NUM_CLASSES,
            )
            > 0
        ),

    "all_validation_classes_present":
        np.all(
            np.bincount(
                y_validation,
                minlength=NUM_CLASSES,
            )
            > 0
        ),

    "all_test_classes_present":
        np.all(
            np.bincount(
                y_test,
                minlength=NUM_CLASSES,
            )
            > 0
        ),

    "train_test_overlap_zero":
        train_test_overlap == 0,

    "validation_test_overlap_zero":
        validation_test_overlap
        == 0,

    "train_validation_overlap_zero":
        train_validation_overlap
        == 0,

    "train_rows_exclude_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        train_global_indices
                    ]
                    & held_out_bit
                )
                == 0
            )
        ),

    "validation_rows_exclude_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        validation_global_indices
                    ]
                    & held_out_bit
                )
                == 0
            )
        ),

    "test_rows_include_heldout":
        bool(
            np.all(
                (
                    device_mask[
                        test_global_indices
                    ]
                    & held_out_bit
                )
                != 0
            )
        ),

    "raw_features_finite":
        bool(
            np.isfinite(
                x_train
            ).all()
            and np.isfinite(
                x_validation
            ).all()
            and np.isfinite(
                x_test
            ).all()
        ),

    "scaled_features_finite":
        bool(
            np.isfinite(
                x_train_scaled
            ).all()
            and np.isfinite(
                x_validation_scaled
            ).all()
            and np.isfinite(
                x_test_scaled
            ).all()
        ),

    "scaler_has_115_features":
        int(
            scaler.n_features_in_
        )
        == INPUT_FEATURES,

    "model_parameter_count_positive":
        parameter_count > 0,

    "training_loss_finite":
        math.isfinite(
            training_loss
        ),

    "validation_loss_finite":
        math.isfinite(
            validation_loss
        ),

    "test_loss_finite":
        math.isfinite(
            test_loss
        ),

    "model_parameters_changed":
        parameter_change_sum > 0,

    "validation_metrics_finite":
        math.isfinite(
            validation_metrics[
                "macro_f1_present_classes"
            ]
        ),

    "test_metrics_finite":
        math.isfinite(
            test_metrics[
                "macro_f1_present_classes"
            ]
        ),
}


all_checks_passed = all(
    validation_checks.values()
)

elapsed_seconds = (
    time.perf_counter()
    - started_at
)


summary = {
    "protocol_version":
        "early_lodo_smoke_test_v2_1",

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
        "Pipeline smoke test only. "
        "Do not use these metrics in "
        "the manuscript."
    ),

    "held_out_device":
        HELD_OUT_DEVICE,

    "held_out_device_id":
        device_id,

    "model_name":
        MODEL_NAME,

    "seed":
        SEED,

    "epoch_count":
        1,

    "sample_sizes": {
        "train":
            len(y_train),

        "validation":
            len(y_validation),

        "test":
            len(y_test),
    },

    "sample_class_counts": {
        "train":
            np.bincount(
                y_train,
                minlength=NUM_CLASSES,
            ).tolist(),

        "validation":
            np.bincount(
                y_validation,
                minlength=NUM_CLASSES,
            ).tolist(),

        "test":
            np.bincount(
                y_test,
                minlength=NUM_CLASSES,
            ).tolist(),
    },

    "test_source_distribution":
        test_source_distribution,

    "overlap_counts": {
        "train_test":
            train_test_overlap,

        "validation_test":
            validation_test_overlap,

        "train_validation":
            train_validation_overlap,
    },

    "scaler": {
        "fit_source":
            "smoke training sample only",

        "feature_count":
            int(
                scaler.n_features_in_
            ),

        "maximum_absolute_mean":
            float(
                np.max(
                    np.abs(
                        scaler.mean_
                    )
                )
            ),

        "minimum_scale":
            float(
                np.min(
                    scaler.scale_
                )
            ),

        "maximum_scale":
            float(
                np.max(
                    scaler.scale_
                )
            ),
    },

    "model": {
        "parameter_count":
            parameter_count,

        "parameter_change_sum":
            parameter_change_sum,
    },

    "losses": {
        "training":
            training_loss,

        "validation":
            validation_loss,

        "test":
            test_loss,
    },

    "validation_metrics":
        validation_metrics,

    "test_metrics":
        test_metrics,

    "elapsed_seconds":
        elapsed_seconds,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        default=json_default,
    ),
    encoding="utf-8",
)


report_lines = [
    "=" * 82,
    "PHASE 2F-1 EARLY LODO SMOKE TEST",
    "=" * 82,
    "",
    (
        f"Held-out device      : "
        f"{HELD_OUT_DEVICE}"
    ),
    (
        f"Model                : "
        f"{MODEL_NAME}"
    ),
    (
        f"Parameters           : "
        f"{parameter_count:,}"
    ),
    (
        f"Elapsed seconds      : "
        f"{elapsed_seconds:.3f}"
    ),
    "",
    "SAMPLE COUNTS",
    (
        f"Train                : "
        f"{len(y_train):,} | "
        f"{np.bincount(y_train, minlength=3).tolist()}"
    ),
    (
        f"Validation           : "
        f"{len(y_validation):,} | "
        f"{np.bincount(y_validation, minlength=3).tolist()}"
    ),
    (
        f"Test                 : "
        f"{len(y_test):,} | "
        f"{np.bincount(y_test, minlength=3).tolist()}"
    ),
    "",
    "OVERLAP COUNTS",
    (
        f"Train-Test           : "
        f"{train_test_overlap}"
    ),
    (
        f"Validation-Test      : "
        f"{validation_test_overlap}"
    ),
    (
        f"Train-Validation     : "
        f"{train_validation_overlap}"
    ),
    "",
    "LOSSES",
    (
        f"Training             : "
        f"{training_loss:.9f}"
    ),
    (
        f"Validation           : "
        f"{validation_loss:.9f}"
    ),
    (
        f"Test                 : "
        f"{test_loss:.9f}"
    ),
    "",
    "SMOKE METRICS - NOT SCIENTIFIC RESULTS",
    (
        f"Validation Macro-F1  : "
        f"{validation_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Test Macro-F1        : "
        f"{test_metrics['macro_f1_present_classes']:.9f}"
    ),
    (
        f"Test accuracy        : "
        f"{test_metrics['accuracy']:.9f}"
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
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print()
print("=" * 82)
print("PHASE 2F-1 SMOKE TEST SUMMARY")
print("=" * 82)
print(
    f"Train samples        : "
    f"{len(y_train):,}"
)
print(
    f"Validation samples   : "
    f"{len(y_validation):,}"
)
print(
    f"Test samples         : "
    f"{len(y_test):,}"
)
print(
    f"Model parameters     : "
    f"{parameter_count:,}"
)
print(
    f"Training loss        : "
    f"{training_loss:.9f}"
)
print(
    f"Validation loss      : "
    f"{validation_loss:.9f}"
)
print(
    f"Test loss            : "
    f"{test_loss:.9f}"
)
print(
    f"Validation Macro-F1  : "
    f"{validation_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Test Macro-F1        : "
    f"{test_metrics['macro_f1_present_classes']:.9f}"
)
print(
    f"Parameter change sum : "
    f"{parameter_change_sum:.9f}"
)

print()
print("OVERLAP COUNTS")
print(
    f"Train-Test       : "
    f"{train_test_overlap}"
)
print(
    f"Validation-Test  : "
    f"{validation_test_overlap}"
)
print(
    f"Train-Validation : "
    f"{train_validation_overlap}"
)

print()
print("TEST SOURCE DISTRIBUTION")

for name, count in (
    test_source_distribution.items()
):
    print(
        f"{name}: {count}"
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

if not all_checks_passed:
    print()
    print(
        "PHASE 2F-1 EARLY LODO "
        "SMOKE TEST FAILED"
    )
    sys.exit(1)

print()
print(
    "PHASE 2F-1 EARLY LODO "
    "SMOKE TEST PASSED"
)
