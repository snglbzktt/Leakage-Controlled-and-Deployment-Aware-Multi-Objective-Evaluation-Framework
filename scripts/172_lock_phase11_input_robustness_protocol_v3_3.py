from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

CACHE_ROOT = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TEST_PATH = CACHE_ROOT / "X_test.npy"
Y_TEST_PATH = CACHE_ROOT / "y_test.npy"
RAW_WEIGHT_PATH = (
    CACHE_ROOT
    / "raw_row_count_test.npy"
)

SCALER_MEAN_PATH = (
    CACHE_ROOT
    / "scaler_mean.npy"
)

SCALER_SCALE_PATH = (
    CACHE_ROOT
    / "scaler_scale.npy"
)

CACHE_MANIFEST_PATH = (
    CACHE_ROOT
    / "manifest.json"
)

PREPROCESSING_VERIFICATION_PATH = (
    AUDIT
    / "phase5_preprocessing_verification_v3_2.json"
)

PREPROCESSING_LOCK_PATH = (
    AUDIT
    / "phase5_preprocessing_locked_v3_2.json"
)

B0_REGISTRY_PATH = (
    AUDIT
    / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PHASE9_SOURCE_REGISTRY_PATH = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
)

PHASE9_ARTIFACT_LOCK_PATH = (
    AUDIT
    / "phase9_final_deployment_artifact_locked_v3_2.json"
)

PHASE10_RELEASE_LOCK_PATH = (
    AUDIT
    / "phase10_release_archive_locked_v3_3.json"
)

HGB_SUMMARY_PATH = (
    AUDIT
    / "hist_gradient_boosting_b0_summary_v3_2.json"
)

HGB_VERIFICATION_PATH = (
    AUDIT
    / "hist_gradient_boosting_b0_verification_v3_2.json"
)

HGB_MODEL_PATH = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
    / "hist_gradient_boosting_b0__seed_2026"
    / "model.joblib"
)

BUNDLE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

MODEL_SOURCE_PATH = (
    BUNDLE_ROOT
    / "nbaiot_models.py"
)

QAT_ENGINE_PATH = (
    BUNDLE_ROOT
    / "phase5_qat_engine_v3_2.py"
)

BUNDLE_SCALER_PATH = (
    BUNDLE_ROOT
    / "train_only_standard_scaler_v3_2.npz"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase11_input_robustness_protocol_v3_3.json"
)

OUTPUT_RUN_MATRIX = (
    AUDIT
    / "phase11_input_robustness_run_matrix_v3_3.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase11_input_robustness_protocol_preflight_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase11_input_robustness_protocol_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase11_input_robustness_protocol_lock_manifest_v3_3.json"
)

PLANNED_RESULTS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase11_input_robustness_v3_3"
)

PROTOCOL_VERSION = (
    "phase11_input_robustness_v3_3"
)

EXPECTED_FEATURE_COUNT = 115
EXPECTED_TEST_ROWS = 371_796
EXPECTED_CLASS_LABELS = [0, 1, 2]
EXPECTED_CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]

CANONICAL_SEED = 2026
PERTURBATION_CHUNK_SIZE = 8192
NEURAL_BATCH_SIZE = 4096

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


MODEL_SPECS = [
    {
        "model_id": "tinyml_mlp_B0",
        "model_family": "tinyml_mlp",
        "variant": "B0",
        "representation": "float32",
        "seed": CANONICAL_SEED,
        "input_space": (
            "canonical_float32_perturbed_"
            "then_train_only_standardized"
        ),
    },
    {
        "model_id": "tinyml_mlp_P50_QAT",
        "model_family": "tinyml_mlp",
        "variant": "P50-QAT",
        "representation": "static_int8",
        "seed": CANONICAL_SEED,
        "input_space": (
            "canonical_float32_perturbed_"
            "then_train_only_standardized"
        ),
    },
    {
        "model_id": "hist_gradient_boosting_B0",
        "model_family": (
            "hist_gradient_boosting"
        ),
        "variant": "B0",
        "representation": (
            "joblib_float_model"
        ),
        "seed": CANONICAL_SEED,
        "input_space": (
            "canonical_float32_perturbed_unscaled"
        ),
    },
]


CONDITION_SPECS = [
    {
        "condition_id": "clean",
        "condition_order": 0,
        "family": "clean",
        "severity": 0.0,
        "random_seed": None,
        "definition": (
            "No perturbation. Canonical float32 "
            "test features are used unchanged."
        ),
    },
    {
        "condition_id": (
            "gaussian_noise_1pct"
        ),
        "condition_order": 1,
        "family": "gaussian_noise",
        "severity": 0.01,
        "random_seed": 2101,
        "definition": (
            "Add independent N(0,1) noise scaled "
            "by 0.01 times each train-only feature "
            "standard deviation."
        ),
    },
    {
        "condition_id": (
            "gaussian_noise_5pct"
        ),
        "condition_order": 2,
        "family": "gaussian_noise",
        "severity": 0.05,
        "random_seed": 2105,
        "definition": (
            "Add independent N(0,1) noise scaled "
            "by 0.05 times each train-only feature "
            "standard deviation."
        ),
    },
    {
        "condition_id": (
            "gaussian_noise_10pct"
        ),
        "condition_order": 3,
        "family": "gaussian_noise",
        "severity": 0.10,
        "random_seed": 2110,
        "definition": (
            "Add independent N(0,1) noise scaled "
            "by 0.10 times each train-only feature "
            "standard deviation."
        ),
    },
    {
        "condition_id": (
            "feature_mask_5pct"
        ),
        "condition_order": 4,
        "family": "feature_masking",
        "severity": 0.05,
        "random_seed": 2205,
        "definition": (
            "Mask each feature independently with "
            "probability 0.05 and replace masked "
            "values with the train-only feature mean."
        ),
    },
    {
        "condition_id": (
            "feature_mask_10pct"
        ),
        "condition_order": 5,
        "family": "feature_masking",
        "severity": 0.10,
        "random_seed": 2210,
        "definition": (
            "Mask each feature independently with "
            "probability 0.10 and replace masked "
            "values with the train-only feature mean."
        ),
    },
    {
        "condition_id": (
            "feature_mask_20pct"
        ),
        "condition_order": 6,
        "family": "feature_masking",
        "severity": 0.20,
        "random_seed": 2220,
        "definition": (
            "Mask each feature independently with "
            "probability 0.20 and replace masked "
            "values with the train-only feature mean."
        ),
    },
    {
        "condition_id": (
            "scale_drift_minus_10pct"
        ),
        "condition_order": 7,
        "family": "scale_drift",
        "severity": -0.10,
        "random_seed": None,
        "definition": (
            "Contract every feature around its "
            "train-only mean: mean + 0.90*(x-mean)."
        ),
    },
    {
        "condition_id": (
            "scale_drift_plus_10pct"
        ),
        "condition_order": 8,
        "family": "scale_drift",
        "severity": 0.10,
        "random_seed": None,
        "definition": (
            "Expand every feature around its "
            "train-only mean: mean + 1.10*(x-mean)."
        ),
    },
]


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


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
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

    replace_with_retry(
        temporary,
        path,
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


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


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
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(
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
    X_TEST_PATH,
    Y_TEST_PATH,
    RAW_WEIGHT_PATH,
    SCALER_MEAN_PATH,
    SCALER_SCALE_PATH,
    CACHE_MANIFEST_PATH,
    PREPROCESSING_VERIFICATION_PATH,
    PREPROCESSING_LOCK_PATH,
    B0_REGISTRY_PATH,
    PHASE9_SOURCE_REGISTRY_PATH,
    PHASE9_ARTIFACT_LOCK_PATH,
    PHASE10_RELEASE_LOCK_PATH,
    HGB_SUMMARY_PATH,
    HGB_VERIFICATION_PATH,
    HGB_MODEL_PATH,
    MODEL_SOURCE_PATH,
    QAT_ENGINE_PATH,
    BUNDLE_SCALER_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_RUN_MATRIX,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 11 protocol output already "
            f"exists; refusing to overwrite: {output_path}"
        )

if PLANNED_RESULTS_ROOT.exists():
    raise FileExistsError(
        "Phase 11 planned results directory "
        f"already exists: {PLANNED_RESULTS_ROOT}"
    )

cache_manifest = read_json(
    CACHE_MANIFEST_PATH
)

preprocessing_verification = read_json(
    PREPROCESSING_VERIFICATION_PATH
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK_PATH
)

phase9_artifact_lock = read_json(
    PHASE9_ARTIFACT_LOCK_PATH
)

phase10_release_lock = read_json(
    PHASE10_RELEASE_LOCK_PATH
)

hgb_summary = read_json(
    HGB_SUMMARY_PATH
)

hgb_verification = read_json(
    HGB_VERIFICATION_PATH
)

B0_rows = read_csv_rows(
    B0_REGISTRY_PATH
)

B0_matches = [
    row
    for row in B0_rows
    if (
        row.get("architecture")
        == "tinyml_mlp"
        and row.get("variant")
        == "B0"
        and row.get("seed")
        == str(CANONICAL_SEED)
    )
]

if len(B0_matches) != 1:
    raise RuntimeError(
        "Expected exactly one TinyML-MLP "
        "B0 seed-2026 registry row."
    )

B0_row = B0_matches[0]
B0_checkpoint_path = Path(
    B0_row["checkpoint_path"]
)

phase9_rows = read_csv_rows(
    PHASE9_SOURCE_REGISTRY_PATH
)

QAT_matches = [
    row
    for row in phase9_rows
    if (
        row.get("architecture")
        == "tinyml_mlp"
        and row.get("variant")
        == "P50-QAT"
        and row.get("seed")
        == str(CANONICAL_SEED)
    )
]

if len(QAT_matches) != 1:
    raise RuntimeError(
        "Expected exactly one selected "
        "P50-QAT seed-2026 source row."
    )

QAT_row = QAT_matches[0]
QAT_checkpoint_path = Path(
    QAT_row["checkpoint_path"]
)

for path in (
    B0_checkpoint_path,
    QAT_checkpoint_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

x_test = np.load(
    X_TEST_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

y_test = np.load(
    Y_TEST_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

raw_weights = np.load(
    RAW_WEIGHT_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

scaler_mean = np.load(
    SCALER_MEAN_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

scaler_scale = np.load(
    SCALER_SCALE_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

with np.load(
    BUNDLE_SCALER_PATH,
    allow_pickle=False,
) as bundled_scaler:
    bundled_mean = np.array(
        bundled_scaler["mean_float64"],
        copy=True,
    )
    bundled_scale = np.array(
        bundled_scaler["scale_float64"],
        copy=True,
    )

B0_checkpoint = torch.load(
    B0_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

QAT_checkpoint = torch.load(
    QAT_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

hgb_model = joblib.load(
    HGB_MODEL_PATH
)

entry_checks = {
    "cache_locked": (
        cache_manifest.get("status")
        in {
            "completed",
            "locked",
            "verified",
        }
        and cache_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "test_shape_exact": (
        tuple(x_test.shape)
        == (
            EXPECTED_TEST_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and tuple(y_test.shape)
        == (
            EXPECTED_TEST_ROWS,
        )
        and tuple(raw_weights.shape)
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "test_dtypes_exact": (
        str(x_test.dtype)
        == "float32"
        and str(y_test.dtype)
        == "int8"
        and str(raw_weights.dtype)
        == "int64"
    ),
    "scaler_shape_exact": (
        tuple(scaler_mean.shape)
        == (
            EXPECTED_FEATURE_COUNT,
        )
        and tuple(scaler_scale.shape)
        == (
            EXPECTED_FEATURE_COUNT,
        )
    ),
    "scaler_dtypes_exact": (
        str(scaler_mean.dtype)
        == "float64"
        and str(scaler_scale.dtype)
        == "float64"
    ),
    "preprocessing_verified": (
        preprocessing_verification.get(
            "status"
        )
        == "passed"
        and preprocessing_verification.get(
            "all_checks_passed"
        )
        is True
        and preprocessing_verification.get(
            "class_labels"
        )
        == EXPECTED_CLASS_LABELS
        and preprocessing_verification.get(
            "class_names"
        )
        == EXPECTED_CLASS_NAMES
    ),
    "preprocessing_lock_matches": (
        preprocessing_lock.get(
            "status"
        )
        == "locked"
        and preprocessing_lock.get(
            "verification_sha256"
        )
        == sha256_file(
            PREPROCESSING_VERIFICATION_PATH
        )
        and preprocessing_lock.get(
            "scaler_sha256"
        )
        == sha256_file(
            BUNDLE_SCALER_PATH
        )
        and preprocessing_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "bundled_scaler_matches_cache": (
        np.array_equal(
            np.asarray(scaler_mean),
            bundled_mean,
        )
        and np.array_equal(
            np.asarray(scaler_scale),
            bundled_scale,
        )
    ),
    "B0_checkpoint_hash_matches_registry": (
        sha256_file(
            B0_checkpoint_path
        )
        == B0_row[
            "checkpoint_sha256"
        ]
    ),
    "B0_checkpoint_identity": (
        B0_checkpoint.get(
            "architecture"
        )
        == "tinyml_mlp"
        and B0_checkpoint.get(
            "variant"
        )
        == "B0"
        and int(
            B0_checkpoint.get(
                "seed"
            )
        )
        == CANONICAL_SEED
        and B0_checkpoint.get(
            "model_symbol"
        )
        == "TinyMLMLP"
        and B0_checkpoint.get(
            "linear_widths"
        )
        == [
            [115, 64],
            [64, 32],
            [32, 3],
        ]
    ),
    "QAT_checkpoint_hash_matches_registry": (
        sha256_file(
            QAT_checkpoint_path
        )
        == QAT_row[
            "checkpoint_sha256"
        ]
    ),
    "QAT_checkpoint_identity": (
        QAT_checkpoint.get(
            "architecture"
        )
        == "tinyml_mlp"
        and QAT_checkpoint.get(
            "variant"
        )
        == "P50-QAT"
        and int(
            QAT_checkpoint.get(
                "seed"
            )
        )
        == CANONICAL_SEED
        and QAT_checkpoint.get(
            "model_symbol"
        )
        == "TinyMLMLP"
        and QAT_checkpoint.get(
            "QAT_backend"
        )
        == "onednn"
        and QAT_checkpoint.get(
            "float_linear_widths"
        )
        == [
            [115, 32],
            [32, 16],
            [16, 3],
        ]
    ),
    "phase9_final_artifact_locked": (
        phase9_artifact_lock.get(
            "status"
        )
        == "locked"
        and phase9_artifact_lock.get(
            "model_family"
        )
        == "tinyml_mlp::P50-QAT"
        and int(
            phase9_artifact_lock.get(
                "canonical_seed"
            )
        )
        == CANONICAL_SEED
        and phase9_artifact_lock.get(
            "final_deployment_artifact_locked"
        )
        is True
        and phase9_artifact_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase10_release_locked": (
        phase10_release_lock.get(
            "status"
        )
        == "locked"
        and phase10_release_lock.get(
            "release_locked"
        )
        is True
        and phase10_release_lock.get(
            "ready_for_distribution"
        )
        is True
        and phase10_release_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "hgb_summary_locked_identity": (
        hgb_summary.get("status")
        == "completed"
        and hgb_summary.get(
            "run_id"
        )
        == (
            "hist_gradient_boosting_b0__"
            "seed_2026"
        )
        and hgb_summary.get(
            "input_space"
        )
        == "canonical_float32_unscaled"
        and int(
            hgb_summary.get("seed")
        )
        == CANONICAL_SEED
        and hgb_summary.get(
            "test_used_for_selection"
        )
        is False
        and hgb_summary.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "hgb_verification_passed": (
        hgb_verification.get(
            "status"
        )
        == "passed"
        and hgb_verification.get(
            "all_checks_passed"
        )
        is True
        and bool(
            hgb_verification.get(
                "independent_unscaled_inference"
            )
        )
    ),
    "hgb_model_type_exact": (
        isinstance(
            hgb_model,
            HistGradientBoostingClassifier,
        )
    ),
    "hgb_model_parameters_exact": (
        hgb_model.loss
        == "log_loss"
        and np.isclose(
            hgb_model.learning_rate,
            0.1,
        )
        and hgb_model.max_iter
        == 100
        and hgb_model.max_leaf_nodes
        == 31
        and hgb_model.max_depth
        is None
        and hgb_model.min_samples_leaf
        == 20
        and np.isclose(
            hgb_model.l2_regularization,
            0.0001,
        )
        and hgb_model.max_bins
        == 255
        and hgb_model.class_weight
        == "balanced"
        and hgb_model.early_stopping
        is False
    ),
    "condition_count_9": (
        len(CONDITION_SPECS)
        == 9
    ),
    "model_count_3": (
        len(MODEL_SPECS)
        == 3
    ),
    "planned_evaluation_count_27": (
        len(CONDITION_SPECS)
        * len(MODEL_SPECS)
        == 27
    ),
    "planned_results_absent": (
        not PLANNED_RESULTS_ROOT.exists()
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 11 robustness protocol entry "
        "gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

run_rows: list[dict[str, Any]] = []

for model_order, model in enumerate(
    MODEL_SPECS
):
    for condition in CONDITION_SPECS:
        run_rows.append(
            {
                "run_id": (
                    f"{model['model_id']}__"
                    f"{condition['condition_id']}"
                ),
                "model_order": model_order,
                "condition_order": (
                    condition[
                        "condition_order"
                    ]
                ),
                "model_id": (
                    model["model_id"]
                ),
                "model_family": (
                    model["model_family"]
                ),
                "variant": (
                    model["variant"]
                ),
                "representation": (
                    model[
                        "representation"
                    ]
                ),
                "seed": model["seed"],
                "input_space": (
                    model["input_space"]
                ),
                "condition_id": (
                    condition[
                        "condition_id"
                    ]
                ),
                "condition_family": (
                    condition["family"]
                ),
                "severity": (
                    condition["severity"]
                ),
                "perturbation_seed": (
                    ""
                    if condition[
                        "random_seed"
                    ]
                    is None
                    else condition[
                        "random_seed"
                    ]
                ),
                "status": "planned",
                "test_used_for_model_selection": (
                    False
                ),
                "model_retraining": False,
            }
        )

write_csv(
    OUTPUT_RUN_MATRIX,
    run_rows,
    fieldnames=[
        "run_id",
        "model_order",
        "condition_order",
        "model_id",
        "model_family",
        "variant",
        "representation",
        "seed",
        "input_space",
        "condition_id",
        "condition_family",
        "severity",
        "perturbation_seed",
        "status",
        "test_used_for_model_selection",
        "model_retraining",
    ],
)

protocol = {
    "status": "locked",
    "phase": 11,
    "artifact_name": (
        "input_robustness_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "scientific_role": (
        "post-selection robustness analysis"
    ),
    "selection_policy": {
        "model_selection_performed": False,
        "checkpoint_selection_performed": (
            False
        ),
        "hyperparameter_tuning_performed": (
            False
        ),
        "test_used_for_selection": False,
        "results_may_change_final_model": False,
        "interpretation": (
            "The locked model family and canonical "
            "checkpoint remain unchanged. Phase 11 "
            "adds robustness evidence only."
        ),
    },
    "evaluation_design": {
        "model_count": 3,
        "condition_count": 9,
        "planned_evaluation_count": 27,
        "canonical_seed": CANONICAL_SEED,
        "shared_perturbations_across_models": (
            True
        ),
        "full_test_set_used": True,
        "primary_view": (
            "fingerprint-level"
        ),
        "secondary_view": (
            "raw-occurrence-weighted"
        ),
        "perturbation_chunk_size": (
            PERTURBATION_CHUNK_SIZE
        ),
        "neural_inference_batch_size": (
            NEURAL_BATCH_SIZE
        ),
        "temporary_perturbed_arrays_persisted": (
            False
        ),
    },
    "models": [
        {
            **MODEL_SPECS[0],
            "checkpoint": (
                file_record(
                    B0_checkpoint_path
                )
            ),
            "model_symbol": "TinyMLMLP",
            "hidden_dimensions": [64, 32],
        },
        {
            **MODEL_SPECS[1],
            "checkpoint": (
                file_record(
                    QAT_checkpoint_path
                )
            ),
            "model_symbol": "TinyMLMLP",
            "hidden_dimensions": [32, 16],
            "QAT_backend": "onednn",
        },
        {
            **MODEL_SPECS[2],
            "model_artifact": (
                file_record(
                    HGB_MODEL_PATH
                )
            ),
            "locked_parameters": (
                hgb_summary[
                    "parameters"
                ]
            ),
        },
    ],
    "conditions": CONDITION_SPECS,
    "perturbation_policy": {
        "base_space": (
            "canonical_float32_unscaled"
        ),
        "train_only_reference_mean": (
            file_record(
                SCALER_MEAN_PATH
            )
        ),
        "train_only_reference_scale": (
            file_record(
                SCALER_SCALE_PATH
            )
        ),
        "gaussian_noise_formula": (
            "x_perturbed = float32(x + severity "
            "* train_scale * Z), Z~N(0,1)"
        ),
        "feature_mask_formula": (
            "Each cell is masked independently; "
            "masked cells are replaced by the "
            "train-only feature mean."
        ),
        "scale_drift_formula": (
            "x_perturbed = float32(train_mean + "
            "factor*(x-train_mean)); factor is "
            "0.90 or 1.10."
        ),
        "clipping_applied": False,
        "nonfinite_values_allowed": False,
        "same_random_realization_for_all_models": (
            True
        ),
        "RNG": (
            "numpy.random.Generator(PCG64)"
        ),
    },
    "preprocessing_policy": {
        "HGB": (
            "Use perturbed canonical float32 "
            "features directly without scaling."
        ),
        "neural_models": (
            "After perturbation in canonical space, "
            "apply the locked train-only "
            "StandardScaler and cast to float32."
        ),
        "validation_data_access": False,
        "training_data_values_access": False,
        "train_only_mean_and_scale_metadata_used": (
            True
        ),
    },
    "metrics": {
        "fingerprint_level": [
            "macro_f1",
            "accuracy",
            "class_recall",
            "gafgyt_fnr",
            "mirai_fnr",
            "confusion_matrix",
        ],
        "raw_occurrence_weighted": [
            "weighted_macro_f1",
            "weighted_accuracy",
            "weighted_class_recall",
            "weighted_confusion_matrix",
        ],
        "robustness_derived": [
            "macro_f1_drop_from_model_clean",
            "macro_f1_retention_ratio",
            "weighted_macro_f1_drop_from_model_clean",
        ],
    },
    "fragility_rules": {
        "severe_fragility": (
            "macro_f1 < 0.85 OR gafgyt_fnr > "
            "0.40 OR mirai_fnr > 0.40"
        ),
        "material_sensitivity": (
            "macro_f1 drop from the same model's "
            "clean condition > 0.05"
        ),
        "rules_are_descriptive_only": True,
        "rules_do_not_trigger_model_reselection": (
            True
        ),
    },
    "source_artifacts": {
        "cache_manifest": (
            file_record(
                CACHE_MANIFEST_PATH
            )
        ),
        "preprocessing_verification": (
            file_record(
                PREPROCESSING_VERIFICATION_PATH
            )
        ),
        "preprocessing_lock": (
            file_record(
                PREPROCESSING_LOCK_PATH
            )
        ),
        "B0_registry": (
            file_record(
                B0_REGISTRY_PATH
            )
        ),
        "phase9_source_registry": (
            file_record(
                PHASE9_SOURCE_REGISTRY_PATH
            )
        ),
        "phase9_final_artifact_lock": (
            file_record(
                PHASE9_ARTIFACT_LOCK_PATH
            )
        ),
        "phase10_release_lock": (
            file_record(
                PHASE10_RELEASE_LOCK_PATH
            )
        ),
        "HGB_summary": (
            file_record(
                HGB_SUMMARY_PATH
            )
        ),
        "HGB_verification": (
            file_record(
                HGB_VERIFICATION_PATH
            )
        ),
        "model_source": (
            file_record(
                MODEL_SOURCE_PATH
            )
        ),
        "QAT_engine": (
            file_record(
                QAT_ENGINE_PATH
            )
        ),
        "bundled_scaler": (
            file_record(
                BUNDLE_SCALER_PATH
            )
        ),
        "X_test": (
            file_record(
                X_TEST_PATH
            )
        ),
        "y_test": (
            file_record(
                Y_TEST_PATH
            )
        ),
        "raw_test_weights": (
            file_record(
                RAW_WEIGHT_PATH
            )
        ),
    },
    "execution_performed": False,
    "model_inference_performed": False,
    "test_arrays_hashed_for_integrity": True,
    "test_array_values_used_for_evaluation": False,
    "test_labels_used_for_evaluation": False,
    "final_model_changed": False,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

preflight_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "run_matrix_rows_27": (
        len(run_rows) == 27
    ),
    "run_ids_unique": (
        len(
            {
                row["run_id"]
                for row in run_rows
            }
        )
        == 27
    ),
    "condition_order_complete": (
        {
            condition[
                "condition_order"
            ]
            for condition
            in CONDITION_SPECS
        }
        == set(range(9))
    ),
    "all_runs_post_selection": all(
        row[
            "test_used_for_model_selection"
        ]
        is False
        and row[
            "model_retraining"
        ]
        is False
        for row in run_rows
    ),
    "execution_not_performed": (
        protocol[
            "execution_performed"
        ]
        is False
    ),
    "inference_not_performed": (
        protocol[
            "model_inference_performed"
        ]
        is False
    ),
    "test_arrays_hashed_only": (
        protocol[
            "test_arrays_hashed_for_integrity"
        ]
        is True
        and protocol[
            "test_array_values_used_for_evaluation"
        ]
        is False
        and protocol[
            "test_labels_used_for_evaluation"
        ]
        is False
    ),
    "final_model_unchanged": (
        protocol[
            "final_model_changed"
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
        "Phase 11 robustness protocol "
        "preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 11,
    "artifact_name": (
        "input_robustness_protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "protocol": (
        file_record(
            OUTPUT_PROTOCOL
        )
    ),
    "run_matrix": (
        file_record(
            OUTPUT_RUN_MATRIX
        )
    ),
    "planned_model_count": 3,
    "planned_condition_count": 9,
    "planned_evaluation_count": 27,
    "execution_performed": False,
    "model_inference_performed": False,
    "test_arrays_hashed_for_integrity": True,
    "test_array_values_used_for_evaluation": False,
    "test_labels_used_for_evaluation": False,
    "final_model_changed": False,
    "ready_for_robustness_execution": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 11,
    "artifact_name": (
        "input_robustness_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "run_matrix": str(
        OUTPUT_RUN_MATRIX
    ),
    "run_matrix_sha256": (
        sha256_file(
            OUTPUT_RUN_MATRIX
        )
    ),
    "preflight_report": str(
        OUTPUT_PREFLIGHT
    ),
    "preflight_report_sha256": (
        sha256_file(
            OUTPUT_PREFLIGHT
        )
    ),
    "planned_evaluation_count": 27,
    "execution_performed": False,
    "model_inference_performed": False,
    "test_used_for_selection": False,
    "final_model_changed": False,
    "ready_for_robustness_execution": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 11,
    "artifact_name": (
        "input_robustness_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            CACHE_MANIFEST_PATH
        ),
        file_record(
            PREPROCESSING_VERIFICATION_PATH
        ),
        file_record(
            PREPROCESSING_LOCK_PATH
        ),
        file_record(
            B0_REGISTRY_PATH
        ),
        file_record(
            PHASE9_SOURCE_REGISTRY_PATH
        ),
        file_record(
            PHASE9_ARTIFACT_LOCK_PATH
        ),
        file_record(
            PHASE10_RELEASE_LOCK_PATH
        ),
        file_record(
            HGB_SUMMARY_PATH
        ),
        file_record(
            HGB_VERIFICATION_PATH
        ),
        file_record(
            HGB_MODEL_PATH
        ),
        file_record(
            B0_checkpoint_path
        ),
        file_record(
            QAT_checkpoint_path
        ),
        file_record(
            MODEL_SOURCE_PATH
        ),
        file_record(
            QAT_ENGINE_PATH
        ),
        file_record(
            BUNDLE_SCALER_PATH
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_PROTOCOL
        ),
        file_record(
            OUTPUT_RUN_MATRIX
        ),
        file_record(
            OUTPUT_PREFLIGHT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "planned_evaluation_count": 27,
    "execution_performed": False,
    "final_model_changed": False,
    "ready_for_robustness_execution": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 11 INPUT ROBUSTNESS PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Scientific role                 : post-selection robustness analysis"
)
print(
    "Models                          : 3"
)
print(
    "Conditions                      : 9"
)
print(
    "Planned evaluations             : 27"
)
print(
    "Model 1                         : tinyml_mlp B0 seed 2026"
)
print(
    "Model 2                         : tinyml_mlp P50-QAT seed 2026"
)
print(
    "Model 3                         : HistGradientBoosting B0 seed 2026"
)
print(
    "Clean condition                 : 1"
)
print(
    "Gaussian-noise conditions       : 3"
)
print(
    "Feature-masking conditions      : 3"
)
print(
    "Scale-drift conditions          : 2"
)
print(
    "Shared perturbations            : True"
)
print(
    "Full test set planned           : True"
)
print(
    "Test used for model selection   : False"
)
print(
    "Model retraining planned        : False"
)
print(
    "Validation data access          : False"
)
print(
    "Test arrays hashed for integrity: True"
)
print(
    "Test values evaluated           : False"
)
print(
    "Execution performed             : False"
)
print(
    "Model inference performed       : False"
)
print(
    "Final model changed             : False"
)
print(
    "Protocol status                 : LOCKED"
)
print(
    "Ready for robustness execution  : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 11 ROBUSTNESS PROTOCOL LOCKED"
)
