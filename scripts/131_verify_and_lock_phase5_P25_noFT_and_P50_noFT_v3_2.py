from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

Y_VALIDATION_PATH = FINAL_CACHE / "y_validation.npy"
RAW_VALIDATION_PATH = (
    FINAL_CACHE / "raw_row_count_validation.npy"
)

Y_TEST_PATH = FINAL_CACHE / "y_test.npy"
RAW_TEST_PATH = (
    FINAL_CACHE / "raw_row_count_test.npy"
)

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

LOCKED_MATRIX = (
    AUDIT / "phase5_locked_configuration_matrix_v3_2.csv"
)

PRUNING_SOURCES_VERIFIED_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_verified_locked_v3_2.json"
)

PRUNING_SOURCES_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

AGGREGATE_RUNS_CSV = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_group_summary_v3_2.csv"
)

AGGREGATE_JSON = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_completed_v3_2.json"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_pruning_noFT_evaluation_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "P25-noFT",
    "P50-noFT",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_CLASS_LABELS = np.asarray(
    [0, 1, 2],
    dtype=np.int64,
)

CLASS_NAMES = (
    "benign",
    "gafgyt",
    "mirai",
)

EXPECTED_VALIDATION_ROWS = 371_797
EXPECTED_TEST_ROWS = 371_796
EXPECTED_VALIDATION_RAW_ROWS = 1_059_390
EXPECTED_TEST_RAW_ROWS = 1_059_393

FLOAT_TOLERANCE = 1e-10


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


def read_csv(
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


def close_enough(
    observed: float,
    expected: float,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )


def metric_view(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=EXPECTED_CLASS_LABELS,
        average=None,
        sample_weight=sample_weight,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EXPECTED_CLASS_LABELS,
        sample_weight=sample_weight,
    )

    if sample_weight is None:
        matrix = matrix.astype(
            np.int64
        )
    else:
        matrix = np.rint(
            matrix
        ).astype(np.int64)

    support_total = float(
        np.sum(support)
    )

    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "macro_precision": float(
            np.mean(precision)
        ),
        "macro_recall": float(
            np.mean(recall)
        ),
        "balanced_accuracy": float(
            np.mean(recall)
        ),
        "macro_f1": float(
            np.mean(f1)
        ),
        "weighted_f1": float(
            np.average(
                f1,
                weights=support,
            )
            if support_total > 0
            else 0.0
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=EXPECTED_CLASS_LABELS,
                sample_weight=sample_weight,
            )
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=EXPECTED_CLASS_LABELS,
                multi_class="ovr",
                average="macro",
                sample_weight=sample_weight,
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "per_class": {},
    }

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        result["per_class"][
            class_name
        ] = {
            "label": int(
                EXPECTED_CLASS_LABELS[
                    index
                ]
            ),
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1[index]
            ),
            "support": float(
                support[index]
            ),
            "false_negative_rate": float(
                1.0 - recall[index]
            ),
        }

    return result


def metrics_match(
    recomputed: dict[str, Any],
    saved: dict[str, Any],
) -> bool:
    scalar_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "roc_auc_ovr_macro",
    )

    for key in scalar_keys:
        if not close_enough(
            recomputed[key],
            saved[key],
        ):
            return False

    if (
        recomputed[
            "confusion_matrix"
        ]
        != saved[
            "confusion_matrix"
        ]
    ):
        return False

    for class_name in CLASS_NAMES:
        for key in (
            "label",
            "precision",
            "recall",
            "f1",
            "support",
            "false_negative_rate",
        ):
            observed = recomputed[
                "per_class"
            ][class_name][key]

            expected = saved[
                "per_class"
            ][class_name][key]

            if key == "label":
                if int(observed) != int(
                    expected
                ):
                    return False
            elif not close_enough(
                observed,
                expected,
            ):
                return False

    return True


def verify_manifest_artifacts(
    run_directory: Path,
    manifest: dict[str, Any],
) -> bool:
    records = manifest[
        "artifacts"
    ]

    expected_names = {
        str(
            record[
                "relative_path"
            ]
        )
        for record in records
    }

    actual_names = {
        path.name
        for path in run_directory.iterdir()
        if path.is_file()
        and path.name
        not in {
            "run_manifest.json",
            "run_status.json",
        }
    }

    if expected_names != actual_names:
        return False

    for record in records:
        path = (
            run_directory
            / str(
                record[
                    "relative_path"
                ]
            )
        )

        if not path.exists():
            return False

        if int(
            record["size_bytes"]
        ) != int(
            path.stat().st_size
        ):
            return False

        if str(
            record["sha256"]
        ) != sha256_file(path):
            return False

    return True


def aggregate_statistics(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(array)
        ),
        "std_population": float(
            np.std(
                array,
                ddof=0,
            )
        ),
        "minimum": float(
            np.min(array)
        ),
        "maximum": float(
            np.max(array)
        ),
    }


required_paths = (
    Y_VALIDATION_PATH,
    RAW_VALIDATION_PATH,
    Y_TEST_PATH,
    RAW_TEST_PATH,
    PHASE5_PROTOCOL,
    LOCKED_MATRIX,
    PRUNING_SOURCES_VERIFIED_LOCK,
    PRUNING_SOURCES_REGISTRY,
    AGGREGATE_RUNS_CSV,
    GROUP_SUMMARY_CSV,
    AGGREGATE_JSON,
    COMPLETION_JSON,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "NoFT verification artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

sources_lock = read_json(
    PRUNING_SOURCES_VERIFIED_LOCK
)

aggregate = read_json(
    AGGREGATE_JSON
)

completion = read_json(
    COMPLETION_JSON
)

aggregate_rows = read_csv(
    AGGREGATE_RUNS_CSV
)

group_rows = read_csv(
    GROUP_SUMMARY_CSV
)

matrix_rows = read_csv(
    LOCKED_MATRIX
)

source_rows = read_csv(
    PRUNING_SOURCES_REGISTRY
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "sources_independently_verified": (
        sources_lock.get("status")
        == "locked"
        and sources_lock.get(
            "ready_for_P25_noFT_and_P50_noFT_evaluation"
        )
        is True
        and sources_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "source_registry_hash_matches": (
        sources_lock.get(
            "source_registry_sha256"
        )
        == sha256_file(
            PRUNING_SOURCES_REGISTRY
        )
    ),
    "aggregate_completed": (
        aggregate.get("status")
        == "completed"
        and aggregate.get(
            "run_count"
        )
        == 20
        and aggregate.get(
            "all_integrity_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "run_count"
        )
        == 20
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "aggregate_hash_matches_completion": (
        completion.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            AGGREGATE_JSON
        )
    ),
    "aggregate_runs_hash_matches_completion": (
        completion.get(
            "aggregate_runs_csv_sha256"
        )
        == sha256_file(
            AGGREGATE_RUNS_CSV
        )
    ),
    "group_summary_hash_matches_completion": (
        completion.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "twenty_aggregate_rows": (
        len(aggregate_rows)
        == 20
    ),
    "four_group_rows": (
        len(group_rows)
        == 4
    ),
    "twenty_source_rows": (
        len(source_rows)
        == 20
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
        "NoFT verification entry gate "
        "failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

expected_keys = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

aggregate_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in aggregate_rows
}

source_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in source_rows
}

matrix_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in matrix_rows
    if row["architecture"]
    in ARCHITECTURES
    and row["variant"]
    in VARIANTS
}

if set(
    aggregate_lookup.keys()
) != expected_keys:
    raise RuntimeError(
        "Aggregate run matrix mismatch."
    )

if set(
    source_lookup.keys()
) != expected_keys:
    raise RuntimeError(
        "Source registry matrix mismatch."
    )

if matrix_keys != expected_keys:
    raise RuntimeError(
        "Locked matrix mismatch."
    )

y_validation_cache = np.asarray(
    np.load(
        Y_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_validation_cache = np.asarray(
    np.load(
        RAW_VALIDATION_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

y_test_cache = np.asarray(
    np.load(
        Y_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

raw_test_cache = np.asarray(
    np.load(
        RAW_TEST_PATH,
        mmap_mode="r",
    ),
    dtype=np.int64,
)

cache_checks = {
    "validation_label_shape_matches": (
        y_validation_cache.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
    ),
    "validation_raw_shape_matches": (
        raw_validation_cache.shape
        == (
            EXPECTED_VALIDATION_ROWS,
        )
    ),
    "test_label_shape_matches": (
        y_test_cache.shape
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "test_raw_shape_matches": (
        raw_test_cache.shape
        == (
            EXPECTED_TEST_ROWS,
        )
    ),
    "validation_raw_total_matches": (
        int(
            raw_validation_cache.sum()
        )
        == EXPECTED_VALIDATION_RAW_ROWS
    ),
    "test_raw_total_matches": (
        int(
            raw_test_cache.sum()
        )
        == EXPECTED_TEST_RAW_ROWS
    ),
}

failed_cache_checks = [
    name
    for name, passed
    in cache_checks.items()
    if not passed
]

if failed_cache_checks:
    raise RuntimeError(
        "NoFT verification cache "
        "checks failed: "
        + ", ".join(
            failed_cache_checks
        )
    )

print("=" * 92)
print("PHASE 5 P25-noFT AND P50-noFT INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Runs                            : 20"
)
print(
    "Saved validation predictions    : read only"
)
print(
    "Saved test predictions          : read only"
)
print(
    "Validation model inference      : No"
)
print(
    "Test model inference            : No"
)
print()

verified_runs: list[
    dict[str, Any]
] = []

for index, key in enumerate(
    sorted(
        expected_keys,
        key=lambda value: (
            value[0],
            value[1],
            value[2],
        ),
    ),
    start=1,
):
    architecture, variant, seed = key

    aggregate_row = aggregate_lookup[
        key
    ]

    source_row = source_lookup[
        key
    ]

    run_directory = Path(
        aggregate_row[
            "output_directory"
        ]
    )

    required_run_paths = {
        "status": (
            run_directory
            / "run_status.json"
        ),
        "manifest": (
            run_directory
            / "run_manifest.json"
        ),
        "metrics": (
            run_directory
            / "metrics.json"
        ),
        "validation_predictions": (
            run_directory
            / "validation_predictions.npz"
        ),
        "test_predictions": (
            run_directory
            / "test_predictions.npz"
        ),
    }

    for path in (
        required_run_paths.values()
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    status = read_json(
        required_run_paths[
            "status"
        ]
    )

    manifest = read_json(
        required_run_paths[
            "manifest"
        ]
    )

    metrics = read_json(
        required_run_paths[
            "metrics"
        ]
    )

    with np.load(
        required_run_paths[
            "validation_predictions"
        ]
    ) as values:
        validation_true = np.array(
            values["y_true"],
            copy=True,
        ).astype(np.int64)

        validation_pred = np.array(
            values["y_pred"],
            copy=True,
        )

        validation_prob = np.array(
            values["probabilities"],
            copy=True,
        )

        validation_raw = np.array(
            values["raw_row_count"],
            copy=True,
        ).astype(np.int64)

    with np.load(
        required_run_paths[
            "test_predictions"
        ]
    ) as values:
        test_true = np.array(
            values["y_true"],
            copy=True,
        ).astype(np.int64)

        test_pred = np.array(
            values["y_pred"],
            copy=True,
        )

        test_prob = np.array(
            values["probabilities"],
            copy=True,
        )

        test_raw = np.array(
            values["raw_row_count"],
            copy=True,
        ).astype(np.int64)

    validation_primary = metric_view(
        validation_true,
        validation_pred,
        validation_prob,
        sample_weight=None,
    )

    validation_weighted = metric_view(
        validation_true,
        validation_pred,
        validation_prob,
        sample_weight=validation_raw,
    )

    test_primary = metric_view(
        test_true,
        test_pred,
        test_prob,
        sample_weight=None,
    )

    test_weighted = metric_view(
        test_true,
        test_pred,
        test_prob,
        sample_weight=test_raw,
    )

    source_checkpoint_path = Path(
        source_row[
            "source_checkpoint_path"
        ]
    )

    source_manifest_path = Path(
        source_row[
            "source_manifest_path"
        ]
    )

    checks = {
        "status_completed": (
            status.get("status")
            == "completed"
            and status.get("stage")
            == "completed"
        ),
        "identity_matches": (
            status.get(
                "architecture"
            )
            == architecture
            and status.get(
                "variant"
            )
            == variant
            and int(
                status.get("seed")
            )
            == seed
            and manifest.get(
                "architecture"
            )
            == architecture
            and manifest.get(
                "variant"
            )
            == variant
            and int(
                manifest.get("seed")
            )
            == seed
            and metrics.get(
                "architecture"
            )
            == architecture
            and metrics.get(
                "variant"
            )
            == variant
            and int(
                metrics.get("seed")
            )
            == seed
        ),
        "manifest_completed": (
            manifest.get("status")
            == "completed"
            and manifest.get(
                "all_integrity_checks_passed"
            )
            is True
        ),
        "manifest_artifacts_match": (
            verify_manifest_artifacts(
                run_directory,
                manifest,
            )
        ),
        "source_checkpoint_hash_matches": (
            source_checkpoint_path.exists()
            and sha256_file(
                source_checkpoint_path
            )
            == source_row[
                "source_checkpoint_sha256"
            ]
            == manifest[
                "source_checkpoint"
            ]["sha256"]
        ),
        "source_manifest_hash_matches": (
            source_manifest_path.exists()
            and sha256_file(
                source_manifest_path
            )
            == source_row[
                "source_manifest_sha256"
            ]
            == manifest[
                "source_manifest"
            ]["sha256"]
        ),
        "test_count_is_one": (
            int(
                status.get(
                    "test_evaluation_count"
                )
            )
            == 1
            and int(
                metrics.get(
                    "test_evaluation_count"
                )
            )
            == 1
            and int(
                manifest[
                    "fit_scope"
                ][
                    "test_evaluation_count"
                ]
            )
            == 1
            and int(
                aggregate_row[
                    "test_evaluation_count"
                ]
            )
            == 1
        ),
        "test_not_used_for_selection": (
            metrics[
                "selection"
            ][
                "test_used_for_selection"
            ]
            is False
            and metrics[
                "data_access"
            ][
                "test_used_for_selection"
            ]
            is False
            and manifest[
                "fit_scope"
            ][
                "test_used_for_selection"
            ]
            is False
            and str(
                aggregate_row[
                    "test_used_for_selection"
                ]
            ).lower()
            == "false"
        ),
        "fine_tuning_not_performed": (
            metrics.get(
                "evaluation_mode"
            )
            == (
                "fixed_no_fine_tuning"
            )
            and manifest.get(
                "evaluation_mode"
            )
            == (
                "fixed_no_fine_tuning"
            )
            and manifest[
                "fit_scope"
            ][
                "fine_tuning_performed"
            ]
            is False
            and str(
                aggregate_row[
                    "fine_tuning_performed"
                ]
            ).lower()
            == "false"
        ),
        "saved_validation_shapes_match": (
            validation_true.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
            and validation_pred.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
            and validation_prob.shape
            == (
                EXPECTED_VALIDATION_ROWS,
                3,
            )
            and validation_raw.shape
            == (
                EXPECTED_VALIDATION_ROWS,
            )
        ),
        "saved_test_shapes_match": (
            test_true.shape
            == (
                EXPECTED_TEST_ROWS,
            )
            and test_pred.shape
            == (
                EXPECTED_TEST_ROWS,
            )
            and test_prob.shape
            == (
                EXPECTED_TEST_ROWS,
                3,
            )
            and test_raw.shape
            == (
                EXPECTED_TEST_ROWS,
            )
        ),
        "saved_labels_match_cache": (
            np.array_equal(
                validation_true,
                y_validation_cache,
            )
            and np.array_equal(
                test_true,
                y_test_cache,
            )
        ),
        "saved_raw_counts_match_cache": (
            np.array_equal(
                validation_raw,
                raw_validation_cache,
            )
            and np.array_equal(
                test_raw,
                raw_test_cache,
            )
        ),
        "probabilities_finite": (
            np.isfinite(
                validation_prob
            ).all()
            and np.isfinite(
                test_prob
            ).all()
        ),
        "probabilities_normalized": (
            np.allclose(
                validation_prob.sum(
                    axis=1
                ),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
            and np.allclose(
                test_prob.sum(
                    axis=1
                ),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "validation_primary_metrics_recompute": (
            metrics_match(
                validation_primary,
                metrics[
                    "validation"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "validation_weighted_metrics_recompute": (
            metrics_match(
                validation_weighted,
                metrics[
                    "validation"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "test_primary_metrics_recompute": (
            metrics_match(
                test_primary,
                metrics[
                    "test"
                ][
                    "primary_fingerprint_level"
                ],
            )
        ),
        "test_weighted_metrics_recompute": (
            metrics_match(
                test_weighted,
                metrics[
                    "test"
                ][
                    "secondary_raw_record_weighted"
                ],
            )
        ),
        "aggregate_values_match": (
            close_enough(
                aggregate_row[
                    "validation_fingerprint_macro_f1"
                ],
                validation_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_fingerprint_accuracy"
                ],
                test_primary[
                    "accuracy"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_fingerprint_macro_f1"
                ],
                test_primary[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_raw_weighted_macro_f1"
                ],
                test_weighted[
                    "macro_f1"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_gafgyt_fnr"
                ],
                test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ],
            )
            and close_enough(
                aggregate_row[
                    "test_mirai_fnr"
                ],
                test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ],
            )
        ),
        "aggregate_hashes_match": (
            aggregate_row[
                "metrics_sha256"
            ]
            == sha256_file(
                required_run_paths[
                    "metrics"
                ]
            )
            and aggregate_row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                required_run_paths[
                    "manifest"
                ]
            )
        ),
        "fragility_flag_matches": (
            bool(
                metrics[
                    "fragility_gate"
                ]["triggered"]
            )
            == (
                str(
                    aggregate_row[
                        "fragility_gate_triggered"
                    ]
                ).lower()
                == "true"
            )
            == bool(
                status[
                    "fragility_gate_triggered"
                ]
            )
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            f"{architecture} seed {seed} "
            f"{variant} verification failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_runs.append(
        {
            "architecture": (
                architecture
            ),
            "variant": variant,
            "seed": seed,
            "validation_fingerprint_macro_f1": (
                validation_primary[
                    "macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": (
                test_primary[
                    "accuracy"
                ]
            ),
            "test_fingerprint_macro_f1": (
                test_primary[
                    "macro_f1"
                ]
            ),
            "test_raw_weighted_macro_f1": (
                test_weighted[
                    "macro_f1"
                ]
            ),
            "test_gafgyt_fnr": (
                test_primary[
                    "per_class"
                ]["gafgyt"][
                    "false_negative_rate"
                ]
            ),
            "test_mirai_fnr": (
                test_primary[
                    "per_class"
                ]["mirai"][
                    "false_negative_rate"
                ]
            ),
            "fragility_triggered": (
                bool(
                    metrics[
                        "fragility_gate"
                    ]["triggered"]
                )
            ),
            "test_evaluation_count": 1,
            "test_model_inference_repeated": False,
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/20] "
        f"{architecture} seed={seed} "
        f"{variant} | "
        "saved metrics=exact | "
        "test inference=False | "
        "fragility="
        f"{bool(metrics['fragility_gate']['triggered'])} | "
        "all checks=True",
        flush=True,
    )

recomputed_groups: list[
    dict[str, Any]
] = []

group_lookup = {
    (
        row["architecture"],
        row["variant"],
    ): row
    for row in group_rows
}

for architecture in ARCHITECTURES:
    for variant in VARIANTS:
        rows = [
            row
            for row in verified_runs
            if row["architecture"]
            == architecture
            and row["variant"]
            == variant
        ]

        saved_group = group_lookup[
            (
                architecture,
                variant,
            )
        ]

        statistics = {
            "validation_fingerprint_macro_f1": (
                aggregate_statistics(
                    [
                        row[
                            "validation_fingerprint_macro_f1"
                        ]
                        for row in rows
                    ]
                )
            ),
            "test_fingerprint_accuracy": (
                aggregate_statistics(
                    [
                        row[
                            "test_fingerprint_accuracy"
                        ]
                        for row in rows
                    ]
                )
            ),
            "test_fingerprint_macro_f1": (
                aggregate_statistics(
                    [
                        row[
                            "test_fingerprint_macro_f1"
                        ]
                        for row in rows
                    ]
                )
            ),
            "test_raw_weighted_macro_f1": (
                aggregate_statistics(
                    [
                        row[
                            "test_raw_weighted_macro_f1"
                        ]
                        for row in rows
                    ]
                )
            ),
            "test_gafgyt_fnr": (
                aggregate_statistics(
                    [
                        row[
                            "test_gafgyt_fnr"
                        ]
                        for row in rows
                    ]
                )
            ),
            "test_mirai_fnr": (
                aggregate_statistics(
                    [
                        row[
                            "test_mirai_fnr"
                        ]
                        for row in rows
                    ]
                )
            ),
        }

        group_checks = {
            "run_count_matches": (
                int(
                    saved_group[
                        "run_count"
                    ]
                )
                == 5
            ),
            "mean_validation_matches": (
                close_enough(
                    saved_group[
                        "mean_validation_fingerprint_macro_f1"
                    ],
                    statistics[
                        "validation_fingerprint_macro_f1"
                    ]["mean"],
                )
            ),
            "mean_accuracy_matches": (
                close_enough(
                    saved_group[
                        "mean_test_fingerprint_accuracy"
                    ],
                    statistics[
                        "test_fingerprint_accuracy"
                    ]["mean"],
                )
            ),
            "mean_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "mean_test_fingerprint_macro_f1"
                    ],
                    statistics[
                        "test_fingerprint_macro_f1"
                    ]["mean"],
                )
            ),
            "std_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "std_test_fingerprint_macro_f1"
                    ],
                    statistics[
                        "test_fingerprint_macro_f1"
                    ][
                        "std_population"
                    ],
                )
            ),
            "min_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "min_test_fingerprint_macro_f1"
                    ],
                    statistics[
                        "test_fingerprint_macro_f1"
                    ]["minimum"],
                )
            ),
            "max_macro_f1_matches": (
                close_enough(
                    saved_group[
                        "max_test_fingerprint_macro_f1"
                    ],
                    statistics[
                        "test_fingerprint_macro_f1"
                    ]["maximum"],
                )
            ),
            "mean_raw_weighted_matches": (
                close_enough(
                    saved_group[
                        "mean_test_raw_weighted_macro_f1"
                    ],
                    statistics[
                        "test_raw_weighted_macro_f1"
                    ]["mean"],
                )
            ),
            "mean_gafgyt_fnr_matches": (
                close_enough(
                    saved_group[
                        "mean_test_gafgyt_fnr"
                    ],
                    statistics[
                        "test_gafgyt_fnr"
                    ]["mean"],
                )
            ),
            "mean_mirai_fnr_matches": (
                close_enough(
                    saved_group[
                        "mean_test_mirai_fnr"
                    ],
                    statistics[
                        "test_mirai_fnr"
                    ]["mean"],
                )
            ),
            "test_count_one_matches": (
                int(
                    saved_group[
                        "test_evaluation_count_per_run"
                    ]
                )
                == 1
            ),
            "fragility_any_matches": (
                (
                    str(
                        saved_group[
                            "fragility_triggered_in_any_run"
                        ]
                    ).lower()
                    == "true"
                )
                == any(
                    row[
                        "fragility_triggered"
                    ]
                    for row in rows
                )
            ),
        }

        failed_group_checks = [
            name
            for name, passed
            in group_checks.items()
            if not passed
        ]

        if failed_group_checks:
            raise RuntimeError(
                f"{architecture} {variant} "
                "group verification failed: "
                + ", ".join(
                    failed_group_checks
                )
            )

        recomputed_groups.append(
            {
                "architecture": (
                    architecture
                ),
                "variant": variant,
                "run_count": 5,
                "aggregate_metrics": (
                    statistics
                ),
                "fragility_triggered_in_any_run": (
                    any(
                        row[
                            "fragility_triggered"
                        ]
                        for row in rows
                    )
                ),
                "all_checks_passed": True,
            }
        )

global_checks = {
    "twenty_runs_verified": (
        len(
            verified_runs
        )
        == 20
    ),
    "all_run_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_runs
        )
    ),
    "four_groups_verified": (
        len(
            recomputed_groups
        )
        == 4
    ),
    "all_group_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in recomputed_groups
        )
    ),
    "test_evaluation_count_one_all_runs": (
        all(
            row[
                "test_evaluation_count"
            ]
            == 1
            for row in verified_runs
        )
    ),
    "test_model_inference_not_repeated": (
        all(
            row[
                "test_model_inference_repeated"
            ]
            is False
            for row in verified_runs
        )
    ),
    "fragility_exists_as_expected": (
        any(
            row[
                "fragility_triggered"
            ]
            for row in verified_runs
        )
        is True
    ),
}

failed_global_checks = [
    name
    for name, passed
    in global_checks.items()
    if not passed
]

if failed_global_checks:
    raise RuntimeError(
        "NoFT global verification failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 5,
    "artifact_name": (
        "P25_noFT_and_P50_noFT_evaluation_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": 20,
    "group_count": 4,
    "verification_policy": {
        "saved_validation_prediction_reads": 20,
        "saved_test_prediction_reads": 20,
        "validation_model_inference_repeated": False,
        "test_model_inference_repeated": False,
        "metrics_recomputed_from_saved_predictions": True,
        "test_evaluation_count_verified_per_run": 1,
    },
    "verified_runs": (
        verified_runs
    ),
    "recomputed_groups": (
        recomputed_groups
    ),
    "entry_checks": (
        entry_checks
    ),
    "cache_checks": (
        cache_checks
    ),
    "global_checks": (
        global_checks
    ),
    "source_artifacts": {
        "aggregate_runs_csv": (
            file_record(
                AGGREGATE_RUNS_CSV
            )
        ),
        "group_summary_csv": (
            file_record(
                GROUP_SUMMARY_CSV
            )
        ),
        "aggregate_json": (
            file_record(
                AGGREGATE_JSON
            )
        ),
        "completion_json": (
            file_record(
                COMPLETION_JSON
            )
        ),
        "source_registry": (
            file_record(
                PRUNING_SOURCES_REGISTRY
            )
        ),
        "verified_sources_lock": (
            file_record(
                PRUNING_SOURCES_VERIFIED_LOCK
            )
        ),
    },
    "ready_for_P25_and_P50_fine_tuning": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "P25_noFT_and_P50_noFT_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": 20,
    "group_count": 4,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "fragility_triggered_in_any_run": True,
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "aggregate_runs_csv": str(
        AGGREGATE_RUNS_CSV
    ),
    "aggregate_runs_csv_sha256": (
        sha256_file(
            AGGREGATE_RUNS_CSV
        )
    ),
    "group_summary_csv": str(
        GROUP_SUMMARY_CSV
    ),
    "group_summary_csv_sha256": (
        sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "aggregate_json": str(
        AGGREGATE_JSON
    ),
    "aggregate_json_sha256": (
        sha256_file(
            AGGREGATE_JSON
        )
    ),
    "ready_for_P25_and_P50_fine_tuning": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "P25_noFT_and_P50_noFT_evaluation"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            LOCKED_MATRIX
        ),
        file_record(
            PRUNING_SOURCES_VERIFIED_LOCK
        ),
        file_record(
            PRUNING_SOURCES_REGISTRY
        ),
        file_record(
            AGGREGATE_RUNS_CSV
        ),
        file_record(
            GROUP_SUMMARY_CSV
        ),
        file_record(
            AGGREGATE_JSON
        ),
        file_record(
            COMPLETION_JSON
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "run_count": 20,
    "test_evaluation_count_per_run": 1,
    "test_model_inference_repeated_by_verifier": False,
    "ready_for_P25_and_P50_fine_tuning": True,
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
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "verification_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "verification_report_sha256"
        ]
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "ready_for_fine_tuning": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_P25_and_P50_fine_tuning"
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
        "NoFT verification post-lock "
        "checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 P25-noFT AND P50-noFT VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs independently verified     : 20"
)
print(
    "Groups independently verified   : 4"
)
print(
    "Saved validation metrics exact  : PASSED"
)
print(
    "Saved test metrics exact        : PASSED"
)
print(
    "Checkpoint and manifest hashes  : PASSED"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Test model inference repeated   : False"
)
print(
    "Fragility present               : True"
)
print(
    "Evaluation status               : LOCKED"
)
print(
    "Ready for P25/P50 fine-tuning   : True"
)
print(
    "Verification report             : "
    f"{OUTPUT_VERIFICATION}"
)
print(
    "Lock file                       : "
    f"{OUTPUT_LOCK}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 P25-noFT AND P50-noFT EVALUATION VERIFIED AND LOCKED"
)
