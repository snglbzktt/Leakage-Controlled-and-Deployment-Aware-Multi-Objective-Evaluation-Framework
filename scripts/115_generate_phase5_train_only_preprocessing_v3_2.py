from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()

AUDIT = ROOT / "results" / "v2" / "audit"

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"
Y_TRAIN_PATH = FINAL_CACHE / "y_train.npy"

CACHE_MANIFEST = FINAL_CACHE / "manifest.json"

CACHE_VERIFICATION = (
    AUDIT
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

PHASE5_PROTOCOL_COMPLETION = (
    AUDIT
    / "phase5_protocol_locked_v3_2.json"
)

PHASE5_PROTOCOL_LOCK = (
    AUDIT
    / "phase5_protocol_lock_manifest_v3_2.json"
)

OUTPUT_DIRECTORY = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
)

SCALER_NPZ = (
    OUTPUT_DIRECTORY
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

CLASS_WEIGHTS_NPZ = (
    OUTPUT_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.npz"
)

CLASS_WEIGHTS_CSV = (
    OUTPUT_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.csv"
)

MANIFEST_JSON = (
    OUTPUT_DIRECTORY
    / "phase5_preprocessing_manifest_v3_2.json"
)

GENERATION_REPORT = (
    AUDIT
    / "phase5_preprocessing_generation_v3_2.json"
)

EXPECTED_PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

EXPECTED_DATA_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)

EXPECTED_TRAIN_FINGERPRINTS = 1_534_583
EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_LABELS = np.asarray(
    [0, 1, 2],
    dtype=np.int64,
)
EXPECTED_CLASS_NAMES = (
    "benign",
    "gafgyt",
    "mirai",
)

CHUNK_SIZE = 100_000
MINIMUM_FREE_DISK_GIB = 1.0


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
    chunk_size: int = 8 * 1024 * 1024,
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


def atomic_csv(
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
        newline="",
        encoding="utf-8",
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


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


required_paths = (
    FINAL_CACHE,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    CACHE_MANIFEST,
    CACHE_VERIFICATION,
    PHASE5_PROTOCOL,
    PHASE5_PROTOCOL_COMPLETION,
    PHASE5_PROTOCOL_LOCK,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    CLASS_WEIGHTS_CSV,
    MANIFEST_JSON,
    GENERATION_REPORT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 5 preprocessing artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

free_disk_gib = (
    shutil.disk_usage(ROOT).free
    / (1024**3)
)

if free_disk_gib < MINIMUM_FREE_DISK_GIB:
    raise RuntimeError(
        "Insufficient free disk space: "
        f"{free_disk_gib:.3f} GiB"
    )

protocol = read_json(
    PHASE5_PROTOCOL
)

protocol_completion = read_json(
    PHASE5_PROTOCOL_COMPLETION
)

protocol_lock = read_json(
    PHASE5_PROTOCOL_LOCK
)

cache_manifest = read_json(
    CACHE_MANIFEST
)

cache_verification = read_json(
    CACHE_VERIFICATION
)

protocol_checks = {
    "phase5_protocol_locked": (
        protocol.get("status")
        == "locked"
    ),
    "phase5_protocol_version_matches": (
        protocol.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "data_protocol_version_matches": (
        protocol.get(
            "data_protocol_version"
        )
        == EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "completion_locked": (
        protocol_completion.get("status")
        == "locked"
        and protocol_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "completion_protocol_hash_matches": (
        protocol_completion.get(
            "protocol_sha256"
        )
        == sha256_file(
            PHASE5_PROTOCOL
        )
    ),
    "lock_manifest_locked": (
        protocol_lock.get("status")
        == "locked"
        and protocol_lock.get(
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
    "cache_verification_passed": (
        cache_verification.get("status")
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "scaler_fit_split_is_train_only": (
        protocol["data"][
            "scaler_policy"
        ]["fit_split"]
        == "train_only"
    ),
    "validation_or_test_not_in_scaler_fit": (
        protocol["data"][
            "scaler_policy"
        ][
            "validation_or_test_in_fit"
        ]
        is False
    ),
    "class_weight_scheme_matches": (
        protocol["data"][
            "class_weight_policy"
        ]["scheme"]
        == (
            "inverse_square_root_"
            "frequency_mean1"
        )
    ),
    "raw_occurrence_weight_not_used_for_fit": (
        protocol["data"][
            "class_weight_policy"
        ][
            "raw_occurrence_weight_used_for_fit"
        ]
        is False
    ),
}

failed_protocol_checks = [
    name
    for name, passed
    in protocol_checks.items()
    if not passed
]

if failed_protocol_checks:
    raise RuntimeError(
        "Phase 5 preprocessing preflight "
        "failed: "
        + ", ".join(
            failed_protocol_checks
        )
    )

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

y_train = np.load(
    Y_TRAIN_PATH,
    mmap_mode="r",
)

shape_checks = {
    "x_train_shape_matches": (
        x_train.shape
        == (
            EXPECTED_TRAIN_FINGERPRINTS,
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "x_train_dtype_is_float32": (
        x_train.dtype == np.float32
    ),
    "y_train_shape_matches": (
        y_train.shape
        == (
            EXPECTED_TRAIN_FINGERPRINTS,
        )
    ),
    "labels_are_expected": (
        np.array_equal(
            np.unique(
                np.asarray(
                    y_train,
                    dtype=np.int64,
                )
            ),
            EXPECTED_CLASS_LABELS,
        )
    ),
}

failed_shape_checks = [
    name
    for name, passed
    in shape_checks.items()
    if not passed
]

if failed_shape_checks:
    raise RuntimeError(
        "Training-cache checks failed: "
        + ", ".join(
            failed_shape_checks
        )
    )

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 92)
print("PHASE 5 TRAIN-ONLY PREPROCESSING GENERATION")
print("=" * 92)
print(
    "Train fingerprints : "
    f"{len(x_train):,}"
)
print(
    "Features           : "
    f"{x_train.shape[1]}"
)
print(
    "Scaler fit split   : train only"
)
print(
    "Class weights      : inverse sqrt frequency, mean=1"
)
print(
    "Validation access  : 0"
)
print(
    "Test access        : 0"
)

scaler = StandardScaler(
    copy=True,
    with_mean=True,
    with_std=True,
)

class_counts = np.zeros(
    len(EXPECTED_CLASS_LABELS),
    dtype=np.int64,
)

print()
print("Fitting StandardScaler in chunks...")

for start in range(
    0,
    len(x_train),
    CHUNK_SIZE,
):
    end = min(
        start + CHUNK_SIZE,
        len(x_train),
    )

    x_chunk = np.asarray(
        x_train[start:end],
        dtype=np.float32,
    )

    y_chunk = np.asarray(
        y_train[start:end],
        dtype=np.int64,
    )

    if not np.isfinite(
        x_chunk
    ).all():
        raise RuntimeError(
            "Non-finite value found in "
            f"training chunk {start}:{end}."
        )

    scaler.partial_fit(
        x_chunk
    )

    class_counts += np.bincount(
        y_chunk,
        minlength=len(
            EXPECTED_CLASS_LABELS
        ),
    ).astype(np.int64)

    print(
        f"[train] {end:,}/"
        f"{len(x_train):,}",
        flush=True,
    )

if int(
    class_counts.sum()
) != EXPECTED_TRAIN_FINGERPRINTS:
    raise RuntimeError(
        "Class-count total does not "
        "match the training size."
    )

inverse_sqrt = (
    1.0
    / np.sqrt(
        class_counts.astype(
            np.float64
        )
    )
)

class_weights = (
    inverse_sqrt
    / np.mean(
        inverse_sqrt
    )
)

scaler_mean64 = np.asarray(
    scaler.mean_,
    dtype=np.float64,
)

scaler_var64 = np.asarray(
    scaler.var_,
    dtype=np.float64,
)

scaler_scale64 = np.asarray(
    scaler.scale_,
    dtype=np.float64,
)

scaler_mean32 = scaler_mean64.astype(
    np.float32
)

scaler_scale32 = scaler_scale64.astype(
    np.float32
)

scaler_checks = {
    "n_samples_seen_matches": (
        int(
            scaler.n_samples_seen_
        )
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "mean_shape_matches": (
        scaler_mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "variance_shape_matches": (
        scaler_var64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "scale_shape_matches": (
        scaler_scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "mean_is_finite": (
        bool(
            np.isfinite(
                scaler_mean64
            ).all()
        )
    ),
    "variance_is_finite": (
        bool(
            np.isfinite(
                scaler_var64
            ).all()
        )
    ),
    "scale_is_finite": (
        bool(
            np.isfinite(
                scaler_scale64
            ).all()
        )
    ),
    "variance_nonnegative": (
        bool(
            np.all(
                scaler_var64 >= 0.0
            )
        )
    ),
    "scale_positive": (
        bool(
            np.all(
                scaler_scale64 > 0.0
            )
        )
    ),
    "class_count_shape_matches": (
        class_counts.shape
        == (
            len(
                EXPECTED_CLASS_LABELS
            ),
        )
    ),
    "class_weights_are_finite": (
        bool(
            np.isfinite(
                class_weights
            ).all()
        )
    ),
    "class_weights_are_positive": (
        bool(
            np.all(
                class_weights > 0.0
            )
        )
    ),
    "class_weights_mean_is_one": (
        math.isclose(
            float(
                np.mean(
                    class_weights
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ),
}

failed_scaler_checks = [
    name
    for name, passed
    in scaler_checks.items()
    if not passed
]

if failed_scaler_checks:
    raise RuntimeError(
        "Generated preprocessing checks "
        "failed: "
        + ", ".join(
            failed_scaler_checks
        )
    )

np.savez_compressed(
    SCALER_NPZ,
    mean_float64=scaler_mean64,
    variance_float64=scaler_var64,
    scale_float64=scaler_scale64,
    mean_float32=scaler_mean32,
    scale_float32=scaler_scale32,
    n_samples_seen=np.asarray(
        [
            int(
                scaler.n_samples_seen_
            )
        ],
        dtype=np.int64,
    ),
    feature_count=np.asarray(
        [
            EXPECTED_FEATURE_COUNT
        ],
        dtype=np.int64,
    ),
    with_mean=np.asarray(
        [True],
        dtype=np.bool_,
    ),
    with_std=np.asarray(
        [True],
        dtype=np.bool_,
    ),
)

np.savez_compressed(
    CLASS_WEIGHTS_NPZ,
    class_labels=(
        EXPECTED_CLASS_LABELS
    ),
    class_counts=class_counts,
    class_weights_float64=(
        class_weights.astype(
            np.float64
        )
    ),
    class_weights_float32=(
        class_weights.astype(
            np.float32
        )
    ),
    normalization_mean=np.asarray(
        [
            float(
                np.mean(
                    class_weights
                )
            )
        ],
        dtype=np.float64,
    ),
)

class_weight_rows = []

for index, class_name in enumerate(
    EXPECTED_CLASS_NAMES
):
    class_weight_rows.append(
        {
            "class_label": int(
                EXPECTED_CLASS_LABELS[
                    index
                ]
            ),
            "class_name": class_name,
            "train_fingerprint_count": int(
                class_counts[index]
            ),
            "class_weight_float64": float(
                class_weights[index]
            ),
            "class_weight_float32": float(
                class_weights.astype(
                    np.float32
                )[index]
            ),
        }
    )

atomic_csv(
    CLASS_WEIGHTS_CSV,
    class_weight_rows,
    [
        "class_label",
        "class_name",
        "train_fingerprint_count",
        "class_weight_float64",
        "class_weight_float32",
    ],
)

manifest = {
    "status": "completed",
    "artifact_name": (
        "phase5_train_only_preprocessing"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "data_protocol_version": (
        EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "fit_scope": {
        "train_access_count": 1,
        "validation_access_count": 0,
        "test_access_count": 0,
        "scaler_fit_split": (
            "train_only"
        ),
        "class_weight_fit_split": (
            "train_only"
        ),
        "raw_occurrence_weight_used": False,
    },
    "training_data": {
        "x_train": file_record(
            X_TRAIN_PATH
        ),
        "y_train": file_record(
            Y_TRAIN_PATH
        ),
        "fingerprint_count": (
            EXPECTED_TRAIN_FINGERPRINTS
        ),
        "feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "x_dtype": str(
            x_train.dtype
        ),
    },
    "standard_scaler": {
        "implementation": (
            "sklearn.preprocessing.StandardScaler"
        ),
        "sklearn_version": (
            sklearn.__version__
        ),
        "streaming_fit": True,
        "chunk_size": CHUNK_SIZE,
        "with_mean": True,
        "with_std": True,
        "n_samples_seen": int(
            scaler.n_samples_seen_
        ),
        "zero_variance_feature_count": int(
            np.sum(
                scaler_var64 == 0.0
            )
        ),
        "application_rule": (
            "Use float64 mean and scale, "
            "then cast transformed output "
            "to float32."
        ),
        "artifact": file_record(
            SCALER_NPZ
        ),
    },
    "class_weights": {
        "scheme": (
            "inverse_square_root_"
            "frequency_mean1"
        ),
        "formula": (
            "w_c=(1/sqrt(n_c))/"
            "mean_j(1/sqrt(n_j))"
        ),
        "class_labels": (
            EXPECTED_CLASS_LABELS.tolist()
        ),
        "class_names": list(
            EXPECTED_CLASS_NAMES
        ),
        "class_counts": (
            class_counts.tolist()
        ),
        "class_weights_float64": (
            class_weights.tolist()
        ),
        "mean_weight": float(
            np.mean(
                class_weights
            )
        ),
        "artifact_npz": file_record(
            CLASS_WEIGHTS_NPZ
        ),
        "artifact_csv": file_record(
            CLASS_WEIGHTS_CSV
        ),
    },
    "source_locks": {
        "phase5_protocol": file_record(
            PHASE5_PROTOCOL
        ),
        "phase5_completion": file_record(
            PHASE5_PROTOCOL_COMPLETION
        ),
        "phase5_lock_manifest": file_record(
            PHASE5_PROTOCOL_LOCK
        ),
        "cache_manifest": file_record(
            CACHE_MANIFEST
        ),
        "cache_verification": file_record(
            CACHE_VERIFICATION
        ),
    },
    "checks": {
        **protocol_checks,
        **shape_checks,
        **scaler_checks,
    },
    "all_checks_passed": True,
}

atomic_json(
    MANIFEST_JSON,
    manifest,
)

generation_report = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "scaler_artifact": file_record(
        SCALER_NPZ
    ),
    "class_weights_artifact": (
        file_record(
            CLASS_WEIGHTS_NPZ
        )
    ),
    "class_weights_csv": file_record(
        CLASS_WEIGHTS_CSV
    ),
    "manifest": file_record(
        MANIFEST_JSON
    ),
    "train_fingerprint_count": (
        EXPECTED_TRAIN_FINGERPRINTS
    ),
    "feature_count": (
        EXPECTED_FEATURE_COUNT
    ),
    "class_counts": (
        class_counts.tolist()
    ),
    "class_weights_float64": (
        class_weights.tolist()
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "independent_verification_required": True,
    "all_checks_passed": True,
}

atomic_json(
    GENERATION_REPORT,
    generation_report,
)

print()
print("=" * 92)
print("PHASE 5 PREPROCESSING GENERATION SUMMARY")
print("=" * 92)
print(
    "Scaler samples                : "
    f"{int(scaler.n_samples_seen_):,}"
)
print(
    "Feature count                 : "
    f"{EXPECTED_FEATURE_COUNT}"
)
print(
    "Class counts                  : "
    f"{class_counts.tolist()}"
)
print(
    "Class weights                 : "
    + "["
    + ", ".join(
        f"{value:.9f}"
        for value in class_weights
    )
    + "]"
)
print(
    "Mean class weight             : "
    f"{np.mean(class_weights):.12f}"
)
print(
    "Validation access count       : 0"
)
print(
    "Test access count             : 0"
)
print(
    "Scaler artifact               : "
    f"{SCALER_NPZ}"
)
print(
    "Class-weight artifact         : "
    f"{CLASS_WEIGHTS_NPZ}"
)
print(
    "Manifest                      : "
    f"{MANIFEST_JSON}"
)
print(
    "Independent verification next : True"
)
print(
    "All checks passed             : True"
)
print(
    "PHASE 5 TRAIN-ONLY PREPROCESSING GENERATED"
)
