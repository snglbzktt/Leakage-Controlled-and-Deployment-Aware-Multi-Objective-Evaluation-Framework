from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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

PREPROCESSING_DIRECTORY = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
)

SCALER_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

CLASS_WEIGHTS_NPZ = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.npz"
)

CLASS_WEIGHTS_CSV = (
    PREPROCESSING_DIRECTORY
    / "phase5_train_only_class_weights_v3_2.csv"
)

PREPROCESSING_MANIFEST = (
    PREPROCESSING_DIRECTORY
    / "phase5_preprocessing_manifest_v3_2.json"
)

GENERATION_REPORT = (
    AUDIT
    / "phase5_preprocessing_generation_v3_2.json"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_preprocessing_verification_v3_2.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase5_preprocessing_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_preprocessing_lock_manifest_v3_2.json"
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

INDEPENDENT_CHUNK_SIZE = 73_777

MEAN_RTOL = 1e-9
MEAN_ATOL = 1e-11
VAR_RTOL = 2e-9
VAR_ATOL = 1e-11
WEIGHT_RTOL = 1e-12
WEIGHT_ATOL = 1e-12


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


def record_matches(
    record: dict[str, Any],
    path: Path,
) -> bool:
    return (
        str(record["path"]) == str(path)
        and int(record["size_bytes"])
        == int(path.stat().st_size)
        and str(record["sha256"])
        == sha256_file(path)
    )


def combine_moments(
    count_a: int,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    count_b: int,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    if count_a == 0:
        return (
            count_b,
            mean_b.copy(),
            m2_b.copy(),
        )

    if count_b == 0:
        return (
            count_a,
            mean_a.copy(),
            m2_a.copy(),
        )

    total = count_a + count_b

    delta = mean_b - mean_a

    combined_mean = (
        mean_a
        + delta
        * (count_b / total)
    )

    combined_m2 = (
        m2_a
        + m2_b
        + delta * delta
        * (
            count_a
            * count_b
            / total
        )
    )

    return (
        total,
        combined_mean,
        combined_m2,
    )


required_paths = (
    FINAL_CACHE,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    PHASE5_PROTOCOL,
    PHASE5_PROTOCOL_COMPLETION,
    PHASE5_PROTOCOL_LOCK,
    SCALER_NPZ,
    CLASS_WEIGHTS_NPZ,
    CLASS_WEIGHTS_CSV,
    PREPROCESSING_MANIFEST,
    GENERATION_REPORT,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_COMPLETION,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 5 preprocessing lock artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
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

manifest = read_json(
    PREPROCESSING_MANIFEST
)

generation_report = read_json(
    GENERATION_REPORT
)

preflight_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
    ),
    "protocol_version_matches": (
        protocol.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "protocol_completion_locked": (
        protocol_completion.get("status")
        == "locked"
        and protocol_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_completion_hash_matches": (
        protocol_completion.get(
            "protocol_sha256"
        )
        == sha256_file(
            PHASE5_PROTOCOL
        )
    ),
    "protocol_lock_manifest_locked": (
        protocol_lock.get("status")
        == "locked"
        and protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "generation_report_completed": (
        generation_report.get("status")
        == "completed"
        and generation_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "manifest_completed": (
        manifest.get("status")
        == "completed"
        and manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "manifest_protocol_matches": (
        manifest.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
    "manifest_data_protocol_matches": (
        manifest.get(
            "data_protocol_version"
        )
        == EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "train_only_scaler_fit_recorded": (
        manifest["fit_scope"][
            "scaler_fit_split"
        ]
        == "train_only"
    ),
    "train_only_class_weight_fit_recorded": (
        manifest["fit_scope"][
            "class_weight_fit_split"
        ]
        == "train_only"
    ),
    "validation_access_is_zero": (
        int(
            manifest["fit_scope"][
                "validation_access_count"
            ]
        )
        == 0
        and int(
            generation_report[
                "validation_access_count"
            ]
        )
        == 0
    ),
    "test_access_is_zero": (
        int(
            manifest["fit_scope"][
                "test_access_count"
            ]
        )
        == 0
        and int(
            generation_report[
                "test_access_count"
            ]
        )
        == 0
    ),
    "raw_occurrence_weight_not_used": (
        manifest["fit_scope"][
            "raw_occurrence_weight_used"
        ]
        is False
    ),
}

failed_preflight_checks = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight_checks:
    raise RuntimeError(
        "Phase 5 preprocessing verification "
        "preflight failed: "
        + ", ".join(
            failed_preflight_checks
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

input_checks = {
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
    "label_set_matches": (
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

failed_input_checks = [
    name
    for name, passed
    in input_checks.items()
    if not passed
]

if failed_input_checks:
    raise RuntimeError(
        "Phase 5 preprocessing input checks "
        "failed: "
        + ", ".join(
            failed_input_checks
        )
    )

with np.load(
    SCALER_NPZ
) as values:
    saved_mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    saved_var64 = np.array(
        values["variance_float64"],
        copy=True,
    )

    saved_scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

    saved_mean32 = np.array(
        values["mean_float32"],
        copy=True,
    )

    saved_scale32 = np.array(
        values["scale_float32"],
        copy=True,
    )

    saved_n_samples = int(
        values["n_samples_seen"][0]
    )

    saved_feature_count = int(
        values["feature_count"][0]
    )

    saved_with_mean = bool(
        values["with_mean"][0]
    )

    saved_with_std = bool(
        values["with_std"][0]
    )

with np.load(
    CLASS_WEIGHTS_NPZ
) as values:
    saved_class_labels = np.array(
        values["class_labels"],
        copy=True,
    )

    saved_class_counts = np.array(
        values["class_counts"],
        copy=True,
    )

    saved_weights64 = np.array(
        values["class_weights_float64"],
        copy=True,
    )

    saved_weights32 = np.array(
        values["class_weights_float32"],
        copy=True,
    )

    saved_normalization_mean = float(
        values["normalization_mean"][0]
    )

print("=" * 92)
print("PHASE 5 PREPROCESSING INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Independent chunk size : "
    f"{INDEPENDENT_CHUNK_SIZE:,}"
)
print(
    "Validation access      : 0"
)
print(
    "Test access            : 0"
)
print()
print(
    "Recomputing train moments "
    "with an independent batch-combination algorithm..."
)

total_count = 0

combined_mean = np.zeros(
    EXPECTED_FEATURE_COUNT,
    dtype=np.float64,
)

combined_m2 = np.zeros(
    EXPECTED_FEATURE_COUNT,
    dtype=np.float64,
)

independent_class_counts = np.zeros(
    len(EXPECTED_CLASS_LABELS),
    dtype=np.int64,
)

for start in range(
    0,
    len(x_train),
    INDEPENDENT_CHUNK_SIZE,
):
    end = min(
        start
        + INDEPENDENT_CHUNK_SIZE,
        len(x_train),
    )

    x_chunk = np.asarray(
        x_train[start:end],
        dtype=np.float64,
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

    batch_count = int(
        len(x_chunk)
    )

    batch_mean = np.mean(
        x_chunk,
        axis=0,
        dtype=np.float64,
    )

    centered = (
        x_chunk - batch_mean
    )

    batch_m2 = np.sum(
        centered * centered,
        axis=0,
        dtype=np.float64,
    )

    (
        total_count,
        combined_mean,
        combined_m2,
    ) = combine_moments(
        total_count,
        combined_mean,
        combined_m2,
        batch_count,
        batch_mean,
        batch_m2,
    )

    independent_class_counts += (
        np.bincount(
            y_chunk,
            minlength=len(
                EXPECTED_CLASS_LABELS
            ),
        ).astype(np.int64)
    )

    print(
        f"[verify train] {end:,}/"
        f"{len(x_train):,}",
        flush=True,
    )

independent_var = (
    combined_m2
    / total_count
)

independent_scale = np.sqrt(
    independent_var,
    dtype=np.float64,
)

independent_scale[
    independent_var == 0.0
] = 1.0

inverse_sqrt = (
    1.0
    / np.sqrt(
        independent_class_counts.astype(
            np.float64
        )
    )
)

independent_weights64 = (
    inverse_sqrt
    / np.mean(
        inverse_sqrt
    )
)

independent_weights32 = (
    independent_weights64.astype(
        np.float32
    )
)

numeric_checks = {
    "total_count_matches": (
        total_count
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "saved_n_samples_matches": (
        saved_n_samples
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "saved_feature_count_matches": (
        saved_feature_count
        == EXPECTED_FEATURE_COUNT
    ),
    "saved_with_mean_true": (
        saved_with_mean is True
    ),
    "saved_with_std_true": (
        saved_with_std is True
    ),
    "saved_array_shapes_match": (
        saved_mean64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and saved_var64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and saved_scale64.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and saved_mean32.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and saved_scale32.shape
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "independent_mean_matches": (
        np.allclose(
            combined_mean,
            saved_mean64,
            rtol=MEAN_RTOL,
            atol=MEAN_ATOL,
        )
    ),
    "independent_variance_matches": (
        np.allclose(
            independent_var,
            saved_var64,
            rtol=VAR_RTOL,
            atol=VAR_ATOL,
        )
    ),
    "independent_scale_matches": (
        np.allclose(
            independent_scale,
            saved_scale64,
            rtol=VAR_RTOL,
            atol=VAR_ATOL,
        )
    ),
    "float32_mean_is_exact_cast": (
        np.array_equal(
            saved_mean32,
            saved_mean64.astype(
                np.float32
            ),
        )
    ),
    "float32_scale_is_exact_cast": (
        np.array_equal(
            saved_scale32,
            saved_scale64.astype(
                np.float32
            ),
        )
    ),
    "saved_class_labels_match": (
        np.array_equal(
            saved_class_labels,
            EXPECTED_CLASS_LABELS,
        )
    ),
    "class_counts_match": (
        np.array_equal(
            independent_class_counts,
            saved_class_counts,
        )
    ),
    "class_count_total_matches": (
        int(
            independent_class_counts.sum()
        )
        == EXPECTED_TRAIN_FINGERPRINTS
    ),
    "class_weights_float64_match": (
        np.allclose(
            independent_weights64,
            saved_weights64,
            rtol=WEIGHT_RTOL,
            atol=WEIGHT_ATOL,
        )
    ),
    "class_weights_float32_match": (
        np.array_equal(
            independent_weights32,
            saved_weights32,
        )
    ),
    "normalization_mean_is_one": (
        math.isclose(
            saved_normalization_mean,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(
                np.mean(
                    independent_weights64
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ),
    "all_saved_values_are_finite": (
        bool(
            np.isfinite(
                saved_mean64
            ).all()
            and np.isfinite(
                saved_var64
            ).all()
            and np.isfinite(
                saved_scale64
            ).all()
            and np.isfinite(
                saved_weights64
            ).all()
        )
    ),
    "saved_scales_are_positive": (
        bool(
            np.all(
                saved_scale64 > 0.0
            )
        )
    ),
    "saved_class_weights_are_positive": (
        bool(
            np.all(
                saved_weights64 > 0.0
            )
        )
    ),
}

failed_numeric_checks = [
    name
    for name, passed
    in numeric_checks.items()
    if not passed
]

if failed_numeric_checks:
    diagnostics = {
        "maximum_absolute_mean_difference": float(
            np.max(
                np.abs(
                    combined_mean
                    - saved_mean64
                )
            )
        ),
        "maximum_absolute_variance_difference": float(
            np.max(
                np.abs(
                    independent_var
                    - saved_var64
                )
            )
        ),
        "maximum_absolute_scale_difference": float(
            np.max(
                np.abs(
                    independent_scale
                    - saved_scale64
                )
            )
        ),
        "maximum_absolute_weight_difference": float(
            np.max(
                np.abs(
                    independent_weights64
                    - saved_weights64
                )
            )
        ),
    }

    raise RuntimeError(
        "Phase 5 preprocessing numeric "
        "verification failed: "
        + ", ".join(
            failed_numeric_checks
        )
        + " | diagnostics="
        + json.dumps(
            diagnostics,
            ensure_ascii=True,
        )
    )

artifact_checks = {
    "manifest_scaler_record_matches": (
        record_matches(
            manifest[
                "standard_scaler"
            ]["artifact"],
            SCALER_NPZ,
        )
    ),
    "manifest_class_weights_npz_record_matches": (
        record_matches(
            manifest[
                "class_weights"
            ]["artifact_npz"],
            CLASS_WEIGHTS_NPZ,
        )
    ),
    "manifest_class_weights_csv_record_matches": (
        record_matches(
            manifest[
                "class_weights"
            ]["artifact_csv"],
            CLASS_WEIGHTS_CSV,
        )
    ),
    "generation_scaler_record_matches": (
        record_matches(
            generation_report[
                "scaler_artifact"
            ],
            SCALER_NPZ,
        )
    ),
    "generation_class_weights_record_matches": (
        record_matches(
            generation_report[
                "class_weights_artifact"
            ],
            CLASS_WEIGHTS_NPZ,
        )
    ),
    "generation_csv_record_matches": (
        record_matches(
            generation_report[
                "class_weights_csv"
            ],
            CLASS_WEIGHTS_CSV,
        )
    ),
    "generation_manifest_record_matches": (
        record_matches(
            generation_report[
                "manifest"
            ],
            PREPROCESSING_MANIFEST,
        )
    ),
    "manifest_protocol_hash_matches": (
        record_matches(
            manifest[
                "source_locks"
            ]["phase5_protocol"],
            PHASE5_PROTOCOL,
        )
    ),
    "manifest_protocol_completion_hash_matches": (
        record_matches(
            manifest[
                "source_locks"
            ]["phase5_completion"],
            PHASE5_PROTOCOL_COMPLETION,
        )
    ),
    "manifest_protocol_lock_hash_matches": (
        record_matches(
            manifest[
                "source_locks"
            ]["phase5_lock_manifest"],
            PHASE5_PROTOCOL_LOCK,
        )
    ),
    "manifest_x_train_hash_matches": (
        record_matches(
            manifest[
                "training_data"
            ]["x_train"],
            X_TRAIN_PATH,
        )
    ),
    "manifest_y_train_hash_matches": (
        record_matches(
            manifest[
                "training_data"
            ]["y_train"],
            Y_TRAIN_PATH,
        )
    ),
}

failed_artifact_checks = [
    name
    for name, passed
    in artifact_checks.items()
    if not passed
]

if failed_artifact_checks:
    raise RuntimeError(
        "Phase 5 preprocessing artifact "
        "verification failed: "
        + ", ".join(
            failed_artifact_checks
        )
    )

spot_indices = np.asarray(
    [
        0,
        1,
        17,
        2026,
        42_042,
        500_000,
        1_000_000,
        EXPECTED_TRAIN_FINGERPRINTS
        - 1,
    ],
    dtype=np.int64,
)

spot_source = np.asarray(
    x_train[spot_indices],
    dtype=np.float64,
)

spot_transformed = (
    (
        spot_source
        - saved_mean64
    )
    / saved_scale64
).astype(np.float32)

spot_checks = {
    "spot_transform_shape_matches": (
        spot_transformed.shape
        == (
            len(spot_indices),
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "spot_transform_dtype_float32": (
        spot_transformed.dtype
        == np.float32
    ),
    "spot_transform_finite": (
        bool(
            np.isfinite(
                spot_transformed
            ).all()
        )
    ),
}

failed_spot_checks = [
    name
    for name, passed
    in spot_checks.items()
    if not passed
]

if failed_spot_checks:
    raise RuntimeError(
        "Phase 5 scaler application "
        "spot checks failed: "
        + ", ".join(
            failed_spot_checks
        )
    )

all_checks = {
    **preflight_checks,
    **input_checks,
    **numeric_checks,
    **artifact_checks,
    **spot_checks,
}

verification_report = {
    "status": "passed",
    "artifact_name": (
        "phase5_train_only_preprocessing"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "data_protocol_version": (
        EXPECTED_DATA_PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "verification_method": {
        "scaler": (
            "Independent float64 batch moment "
            "combination using a different chunk "
            "size from generation."
        ),
        "class_weights": (
            "Independent exact class counting and "
            "inverse-square-root mean-one formula."
        ),
        "artifact_integrity": (
            "SHA-256 and file-size verification "
            "against generation records."
        ),
        "independent_chunk_size": (
            INDEPENDENT_CHUNK_SIZE
        ),
        "generation_chunk_size": (
            int(
                manifest[
                    "standard_scaler"
                ]["chunk_size"]
            )
        ),
    },
    "fit_scope_verified": {
        "train_access_count": 1,
        "validation_access_count": 0,
        "test_access_count": 0,
        "raw_occurrence_weight_used": False,
    },
    "train_fingerprint_count": (
        total_count
    ),
    "feature_count": (
        EXPECTED_FEATURE_COUNT
    ),
    "class_labels": (
        EXPECTED_CLASS_LABELS.tolist()
    ),
    "class_names": list(
        EXPECTED_CLASS_NAMES
    ),
    "class_counts": (
        independent_class_counts.tolist()
    ),
    "class_weights_float64": (
        independent_weights64.tolist()
    ),
    "maximum_absolute_differences": {
        "mean": float(
            np.max(
                np.abs(
                    combined_mean
                    - saved_mean64
                )
            )
        ),
        "variance": float(
            np.max(
                np.abs(
                    independent_var
                    - saved_var64
                )
            )
        ),
        "scale": float(
            np.max(
                np.abs(
                    independent_scale
                    - saved_scale64
                )
            )
        ),
        "class_weight": float(
            np.max(
                np.abs(
                    independent_weights64
                    - saved_weights64
                )
            )
        ),
    },
    "artifacts": {
        "scaler": file_record(
            SCALER_NPZ
        ),
        "class_weights_npz": file_record(
            CLASS_WEIGHTS_NPZ
        ),
        "class_weights_csv": file_record(
            CLASS_WEIGHTS_CSV
        ),
        "generation_manifest": file_record(
            PREPROCESSING_MANIFEST
        ),
        "generation_report": file_record(
            GENERATION_REPORT
        ),
    },
    "checks": all_checks,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification_report,
)

completion = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "train_only_preprocessing"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "scaler_fit_split": "train_only",
    "class_weight_fit_split": (
        "train_only"
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "train_fingerprint_count": (
        total_count
    ),
    "feature_count": (
        EXPECTED_FEATURE_COUNT
    ),
    "class_counts": (
        independent_class_counts.tolist()
    ),
    "scaler_artifact": str(
        SCALER_NPZ
    ),
    "scaler_sha256": sha256_file(
        SCALER_NPZ
    ),
    "class_weights_artifact": str(
        CLASS_WEIGHTS_NPZ
    ),
    "class_weights_sha256": (
        sha256_file(
            CLASS_WEIGHTS_NPZ
        )
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "ready_for_B0_training": True,
    "next_action": (
        "Run a fail-closed neural architecture "
        "and one-batch execution smoke test before "
        "the ten B0 training runs."
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

generated_inventory = [
    file_record(
        OUTPUT_VERIFICATION
    ),
    file_record(
        OUTPUT_COMPLETION
    ),
]

source_inventory = [
    file_record(
        PHASE5_PROTOCOL
    ),
    file_record(
        PHASE5_PROTOCOL_COMPLETION
    ),
    file_record(
        PHASE5_PROTOCOL_LOCK
    ),
    file_record(
        X_TRAIN_PATH
    ),
    file_record(
        Y_TRAIN_PATH
    ),
    file_record(
        SCALER_NPZ
    ),
    file_record(
        CLASS_WEIGHTS_NPZ
    ),
    file_record(
        CLASS_WEIGHTS_CSV
    ),
    file_record(
        PREPROCESSING_MANIFEST
    ),
    file_record(
        GENERATION_REPORT
    ),
]

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "train_only_preprocessing"
    ),
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": (
        source_inventory
    ),
    "generated_artifacts": (
        generated_inventory
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_B0_training": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "verification_exists": (
        OUTPUT_VERIFICATION.exists()
    ),
    "completion_exists": (
        OUTPUT_COMPLETION.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "completion_verification_hash_matches": (
        read_json(
            OUTPUT_COMPLETION
        )["verification_sha256"]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "lock_manifest_locked": (
        read_json(
            OUTPUT_LOCK_MANIFEST
        ).get("status")
        == "locked"
    ),
    "completion_ready_for_B0": (
        read_json(
            OUTPUT_COMPLETION
        ).get(
            "ready_for_B0_training"
        )
        is True
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Phase 5 preprocessing post-lock "
        "checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

free_disk_gib = (
    shutil.disk_usage(ROOT).free
    / (1024**3)
)

print()
print("=" * 92)
print("PHASE 5 PREPROCESSING LOCK SUMMARY")
print("=" * 92)
print(
    "Train fingerprints            : "
    f"{total_count:,}"
)
print(
    "Feature count                 : "
    f"{EXPECTED_FEATURE_COUNT}"
)
print(
    "Class counts                  : "
    f"{independent_class_counts.tolist()}"
)
print(
    "Class weights                 : "
    + "["
    + ", ".join(
        f"{value:.9f}"
        for value in independent_weights64
    )
    + "]"
)
print(
    "Independent chunk size        : "
    f"{INDEPENDENT_CHUNK_SIZE:,}"
)
print(
    "Generation chunk size         : "
    f"{manifest['standard_scaler']['chunk_size']:,}"
)
print(
    "Independent scaler match      : True"
)
print(
    "Independent class-weight match: True"
)
print(
    "Validation access count       : 0"
)
print(
    "Test access count             : 0"
)
print(
    "Preprocessing status          : LOCKED"
)
print(
    "Ready for B0 training         : True"
)
print(
    "Verification report           : "
    f"{OUTPUT_VERIFICATION}"
)
print(
    "Lock manifest                 : "
    f"{OUTPUT_LOCK_MANIFEST}"
)
print(
    "Free disk                     : "
    f"{free_disk_gib:.3f} GiB"
)
print(
    "All checks passed             : True"
)
print(
    "PHASE 5 TRAIN-ONLY PREPROCESSING LOCKED"
)
