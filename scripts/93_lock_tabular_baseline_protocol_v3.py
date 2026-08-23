from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


ROOT = Path.cwd()

SPLIT_DATABASE = (
    ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)

CACHE_SIZING_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_cache_sizing_v3.json"
)

SOURCE_AUDIT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_protocol_only_source_audit_v3.json"
)

REHASH_AUDIT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_canonical_rehash_alignment_summary_v3.json"
)

PROTOCOL_FILE = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

RUN_MATRIX_FILE = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_run_matrix_v3.csv"
)

LOCK_MANIFEST_FILE = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
)

PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"

CLASSES = [
    "benign",
    "gafgyt",
    "mirai",
]

STOCHASTIC_SEEDS = [
    42,
    123,
    2026,
    3407,
    8192,
]

DETERMINISTIC_SEED = 2026

EXPECTED_GROUP_COUNT = 2_278_176
EXPECTED_RAW_ROW_COUNT = 7_062_606

EXPECTED_SPLIT_COUNTS = {
    "train": {
        "fingerprints": 1_534_583,
        "raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_796,
        "raw_rows": 1_059_389,
    },
    "test": {
        "fingerprints": 371_797,
        "raw_rows": 1_059_394,
    },
}

EXPECTED_FAMILY_SPLIT_COUNTS = {
    ("train", "benign"): (359_439, 389_152),
    ("train", "gafgyt"): (27_221, 1_986_790),
    ("train", "mirai"): (1_147_923, 2_567_881),
    ("validation", "benign"): (77_021, 83_389),
    ("validation", "gafgyt"): (48_793, 425_740),
    ("validation", "mirai"): (245_982, 550_260),
    ("test", "benign"): (77_022, 83_391),
    ("test", "gafgyt"): (48_793, 425_742),
    ("test", "mirai"): (245_982, 550_261),
}

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}

FAMILY_NAMES = {
    0: "benign",
    1: "gafgyt",
    2: "mirai",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
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
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "Run matrix is empty."
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


required_paths = [
    SPLIT_DATABASE,
    CACHE_SIZING_REPORT,
    SOURCE_AUDIT_REPORT,
    REHASH_AUDIT_REPORT,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    PROTOCOL_FILE,
    RUN_MATRIX_FILE,
    LOCK_MANIFEST_FILE,
):
    if output_path.exists():
        raise FileExistsError(
            "Locked artifact already exists; "
            f"refusing to overwrite: {output_path}"
        )

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

cache_sizing = json.loads(
    CACHE_SIZING_REPORT.read_text(
        encoding="utf-8"
    )
)

source_audit = json.loads(
    SOURCE_AUDIT_REPORT.read_text(
        encoding="utf-8"
    )
)

rehash_audit = json.loads(
    REHASH_AUDIT_REPORT.read_text(
        encoding="utf-8"
    )
)

database_uri = (
    SPLIT_DATABASE.resolve().as_uri()
    + "?mode=ro"
)

with sqlite3.connect(
    database_uri,
    uri=True,
    timeout=120.0,
) as connection:
    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    if quick_check.lower() != "ok":
        raise RuntimeError(
            "Split database quick_check failed: "
            f"{quick_check}"
        )

    group_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM fingerprint_stats
            """
        ).fetchone()[0]
    )

    raw_row_count = int(
        connection.execute(
            """
            SELECT SUM(raw_row_count)
            FROM fingerprint_stats
            """
        ).fetchone()[0]
    )

    cross_family_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM fingerprint_stats
            WHERE family_label_min
                  != family_label_max
            """
        ).fetchone()[0]
    )

    split_rows = connection.execute(
        """
        SELECT
            grouped_split,
            COUNT(*) AS fingerprint_count,
            SUM(raw_row_count) AS raw_row_count
        FROM fingerprint_stats
        GROUP BY grouped_split
        ORDER BY grouped_split
        """
    ).fetchall()

    family_split_rows = connection.execute(
        """
        SELECT
            grouped_split,
            family_label_min,
            COUNT(*) AS fingerprint_count,
            SUM(raw_row_count) AS raw_row_count
        FROM fingerprint_stats
        GROUP BY
            grouped_split,
            family_label_min
        ORDER BY
            grouped_split,
            family_label_min
        """
    ).fetchall()

observed_split_counts = {
    SPLIT_NAMES[int(split_code)]: {
        "fingerprints": int(fingerprint_count),
        "raw_rows": int(split_raw_rows),
    }
    for (
        split_code,
        fingerprint_count,
        split_raw_rows,
    ) in split_rows
}

observed_family_split_counts = {
    (
        SPLIT_NAMES[int(split_code)],
        FAMILY_NAMES[int(family_code)],
    ): (
        int(fingerprint_count),
        int(split_raw_rows),
    )
    for (
        split_code,
        family_code,
        fingerprint_count,
        split_raw_rows,
    ) in family_split_rows
}

model_definitions = [
    {
        "model_id": "logistic_regression_b0",
        "model_family": "classical_linear",
        "architecture": "logistic_regression",
        "input_space": "train_only_standardized_float32",
        "fit_sample_weight": "none",
        "deterministic": True,
        "seeds": [DETERMINISTIC_SEED],
        "parameters": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 200,
            "tol": 0.0001,
            "fit_intercept": True,
            "class_weight": "balanced",
            "random_state": DETERMINISTIC_SEED,
        },
    },
    {
        "model_id": "decision_tree_b0",
        "model_family": "classical_tree",
        "architecture": "decision_tree",
        "input_space": "canonical_float32_unscaled",
        "fit_sample_weight": "none",
        "deterministic": False,
        "seeds": STOCHASTIC_SEEDS,
        "parameters": {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": 20,
            "min_samples_split": 40,
            "min_samples_leaf": 20,
            "max_features": None,
            "class_weight": "balanced",
            "ccp_alpha": 0.0,
        },
    },
    {
        "model_id": "random_forest_b0",
        "model_family": "classical_ensemble",
        "architecture": "random_forest",
        "input_space": "canonical_float32_unscaled",
        "fit_sample_weight": "none",
        "deterministic": False,
        "seeds": STOCHASTIC_SEEDS,
        "parameters": {
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
        },
    },
    {
        "model_id": "hist_gradient_boosting_b0",
        "model_family": "classical_boosting",
        "architecture": "hist_gradient_boosting",
        "input_space": "canonical_float32_unscaled",
        "fit_sample_weight": "none",
        "deterministic": True,
        "seeds": [DETERMINISTIC_SEED],
        "parameters": {
            "loss": "log_loss",
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0001,
            "max_bins": 255,
            "categorical_features": None,
            "early_stopping": False,
            "class_weight": "balanced",
            "random_state": DETERMINISTIC_SEED,
        },
    },
]

run_matrix: list[dict[str, Any]] = []

for model in model_definitions:
    for seed in model["seeds"]:
        run_id = (
            f"{model['model_id']}"
            f"__seed_{int(seed)}"
        )

        run_matrix.append(
            {
                "run_id": run_id,
                "model_id": model["model_id"],
                "architecture": model["architecture"],
                "seed": int(seed),
                "input_space": model["input_space"],
                "fit_sample_weight": (
                    model["fit_sample_weight"]
                ),
                "expected_test_evaluation_count": 1,
                "output_directory": str(
                    RESULTS_ROOT / run_id
                ),
            }
        )

validation_checks = {
    "database_quick_check_ok": (
        quick_check.lower() == "ok"
    ),
    "group_count_matches": (
        group_count
        == EXPECTED_GROUP_COUNT
    ),
    "raw_row_count_matches": (
        raw_row_count
        == EXPECTED_RAW_ROW_COUNT
    ),
    "cross_family_group_count_zero": (
        cross_family_count == 0
    ),
    "split_counts_match": (
        observed_split_counts
        == EXPECTED_SPLIT_COUNTS
    ),
    "family_split_counts_match": (
        observed_family_split_counts
        == EXPECTED_FAMILY_SPLIT_COUNTS
    ),
    "cache_sizing_group_count_matches": (
        int(
            cache_sizing[
                "total_fingerprint_count"
            ]
        )
        == EXPECTED_GROUP_COUNT
    ),
    "cache_sizing_raw_rows_match": (
        int(
            cache_sizing[
                "total_raw_row_count"
            ]
        )
        == EXPECTED_RAW_ROW_COUNT
    ),
    "rehash_common_groups_match": (
        int(
            rehash_audit[
                "common_group_count"
            ]
        )
        == 2_278_174
    ),
    "rehash_protocol_only_groups_match": (
        int(
            rehash_audit[
                "protocol_only_group_count"
            ]
        )
        == 2
    ),
    "source_audit_protocol_only_rows_match": (
        int(
            source_audit[
                "protocol_only_raw_rows"
            ]
        )
        == 965_554
    ),
    "logistic_parameters_supported": all(
        name
        in inspect.signature(
            LogisticRegression
        ).parameters
        for name in (
            "penalty",
            "C",
            "solver",
            "max_iter",
            "tol",
            "fit_intercept",
            "class_weight",
            "random_state",
        )
    ),
    "decision_tree_parameters_supported": all(
        name
        in inspect.signature(
            DecisionTreeClassifier
        ).parameters
        for name in (
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "ccp_alpha",
            "random_state",
        )
    ),
    "random_forest_parameters_supported": all(
        name
        in inspect.signature(
            RandomForestClassifier
        ).parameters
        for name in (
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "class_weight",
            "n_jobs",
            "oob_score",
            "random_state",
        )
    ),
    "hgb_parameters_supported": all(
        name
        in inspect.signature(
            HistGradientBoostingClassifier
        ).parameters
        for name in (
            "loss",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
            "categorical_features",
            "early_stopping",
            "class_weight",
            "random_state",
        )
    ),
    "run_count_is_12": (
        len(run_matrix) == 12
    ),
    "run_ids_unique": (
        len(
            {
                row["run_id"]
                for row in run_matrix
            }
        )
        == len(run_matrix)
    ),
    "random_forest_n_jobs_limited_to_2": (
        model_definitions[2][
            "parameters"
        ]["n_jobs"]
        == 2
    ),
}

all_checks_passed = all(
    validation_checks.values()
)

if not all_checks_passed:
    failed_checks = [
        name
        for name, passed
        in validation_checks.items()
        if not passed
    ]

    raise RuntimeError(
        "Protocol lock validation failed: "
        + ", ".join(failed_checks)
    )

locked_at = utc_now()

protocol = {
    "protocol_version": PROTOCOL_VERSION,
    "status": "locked",
    "locked_at": locked_at,
    "task": {
        "name": "family_3",
        "classes": CLASSES,
        "feature_count": 115,
        "source_raw_row_count": (
            EXPECTED_RAW_ROW_COUNT
        ),
        "training_fingerprint_count": (
            EXPECTED_GROUP_COUNT
        ),
    },
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    },
    "data_policy": {
        "split_database": str(
            SPLIT_DATABASE
        ),
        "split_database_sha256": (
            sha256_file(SPLIT_DATABASE)
        ),
        "split_protocol_version": "2.3",
        "split_arm": "grouped_float32",
        "assignment_unit": (
            "one row per paired VS2 float32 fingerprint"
        ),
        "training_unit": (
            "one representative row per exact float32 fingerprint"
        ),
        "training_sample_weight": "none",
        "primary_evaluation_view": (
            "fingerprint_level"
        ),
        "secondary_evaluation_view": (
            "raw_record_weighted_by_raw_row_count"
        ),
        "cross_family_group_count": 0,
        "canonical_legacy_dataset_policy": (
            "not used directly because it excludes two "
            "family-consistent groups spanning raw gafgyt "
            "subtype labels"
        ),
        "protocol_only_recovered_groups": 2,
        "protocol_only_recovered_raw_rows": 965_554,
        "cache_design": (
            "float32 features plus labels, raw_row_count, "
            "and paired hashes for each grouped split"
        ),
    },
    "preprocessing_policy": {
        "canonical_numeric_type": "float32",
        "linear_model_pipeline": (
            "canonical float32 -> train-only StandardScaler "
            "computed in float64 -> standardized float32"
        ),
        "linear_scaled_overlap_policy": (
            "fail closed: require zero paired exact float32 "
            "fingerprint overlap across train, validation, and test"
        ),
        "tree_model_pipeline": (
            "canonical float32 without standardization"
        ),
        "feature_order": (
            "locked 115-column order from N-BaIoT CSV/Parquet schema"
        ),
        "non_finite_values_allowed": False,
    },
    "selection_policy": {
        "configuration": (
            "fixed preregistered B0 configurations"
        ),
        "validation_hyperparameter_tuning": False,
        "test_hyperparameter_tuning": False,
        "validation_usage": (
            "diagnostic reporting only; no model or parameter selection"
        ),
        "test_evaluation": (
            "exactly once per completed run"
        ),
    },
    "models": model_definitions,
    "run_policy": {
        "deterministic_models": {
            "model_ids": [
                "logistic_regression_b0",
                "hist_gradient_boosting_b0",
            ],
            "seed": DETERMINISTIC_SEED,
            "run_count_per_model": 1,
        },
        "seed_sensitive_models": {
            "model_ids": [
                "decision_tree_b0",
                "random_forest_b0",
            ],
            "seeds": STOCHASTIC_SEEDS,
            "run_count_per_model": 5,
        },
        "total_expected_runs": len(
            run_matrix
        ),
        "run_matrix_file": str(
            RUN_MATRIX_FILE
        ),
    },
    "resource_policy": {
        "observed_total_physical_ram_gb": 15.731,
        "observed_free_physical_ram_gb": 4.734,
        "observed_physical_cores": 14,
        "observed_logical_processors": 20,
        "random_forest_n_jobs": 2,
        "random_forest_parallelism_reason": (
            "bounded to reduce concurrent tree-building memory "
            "pressure on a 16 GB system"
        ),
        "minimum_free_disk_before_cache_build_gib": 2.0,
        "estimated_cache_with_margin_gib": float(
            cache_sizing[
                "estimated_cache_with_20_percent_margin_gib"
            ]
        ),
    },
    "reporting_policy": {
        "required_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_fnr",
            "confusion_matrix",
            "log_loss",
        ],
        "reporting_views": [
            "fingerprint_level",
            "raw_record_weighted",
        ],
        "prediction_artifacts": [
            "test_predictions",
            "test_probabilities",
        ],
        "timing_artifacts": [
            "fit_seconds",
            "validation_seconds",
            "test_seconds",
            "total_seconds",
        ],
        "model_size_artifact": (
            "serialized_model_size_bytes"
        ),
        "convergence_artifacts": [
            "iteration_count_or_tree_count",
            "convergence_warning_count",
        ],
    },
    "split_counts": (
        observed_split_counts
    ),
    "family_split_counts": [
        {
            "split": split_name,
            "family": family_name,
            "fingerprints": values[0],
            "raw_rows": values[1],
        }
        for (
            split_name,
            family_name,
        ), values
        in sorted(
            observed_family_split_counts.items()
        )
    ],
    "validation_checks": (
        validation_checks
    ),
    "all_checks_passed": (
        all_checks_passed
    ),
    "provenance": {
        "cache_sizing_report": str(
            CACHE_SIZING_REPORT
        ),
        "cache_sizing_report_sha256": (
            sha256_file(
                CACHE_SIZING_REPORT
            )
        ),
        "rehash_alignment_report": str(
            REHASH_AUDIT_REPORT
        ),
        "rehash_alignment_report_sha256": (
            sha256_file(
                REHASH_AUDIT_REPORT
            )
        ),
        "protocol_only_source_audit": str(
            SOURCE_AUDIT_REPORT
        ),
        "protocol_only_source_audit_sha256": (
            sha256_file(
                SOURCE_AUDIT_REPORT
            )
        ),
        "locker_script": str(
            Path(__file__).resolve()
        ),
        "locker_script_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
    },
}

atomic_csv(
    RUN_MATRIX_FILE,
    run_matrix,
)

atomic_json(
    PROTOCOL_FILE,
    protocol,
)

lock_manifest = {
    "protocol_version": PROTOCOL_VERSION,
    "status": "locked",
    "locked_at": locked_at,
    "protocol_file": str(
        PROTOCOL_FILE
    ),
    "protocol_file_sha256": (
        sha256_file(
            PROTOCOL_FILE
        )
    ),
    "run_matrix_file": str(
        RUN_MATRIX_FILE
    ),
    "run_matrix_file_sha256": (
        sha256_file(
            RUN_MATRIX_FILE
        )
    ),
    "split_database": str(
        SPLIT_DATABASE
    ),
    "split_database_sha256": (
        protocol[
            "data_policy"
        ][
            "split_database_sha256"
        ]
    ),
    "expected_run_count": len(
        run_matrix
    ),
    "all_checks_passed": True,
}

atomic_json(
    LOCK_MANIFEST_FILE,
    lock_manifest,
)

print("=" * 88)
print("TABULAR BASELINE PROTOCOL LOCK")
print("=" * 88)
print(f"Protocol version : {PROTOCOL_VERSION}")
print(f"Status           : locked")
print(f"Classes          : {CLASSES}")
print(f"Fingerprint rows : {group_count:,}")
print(f"Raw rows         : {raw_row_count:,}")
print(f"Model count      : {len(model_definitions)}")
print(f"Expected runs    : {len(run_matrix)}")
print()
print("RUN COUNTS BY MODEL")

for model in model_definitions:
    print(
        f"  {model['model_id']:<32} "
        f"{len(model['seeds'])}"
    )

print()
print("RESOURCE LOCK")
print("  Random Forest n_jobs : 2")
print("  Random Forest trees  : 100")
print("  Random Forest depth  : 20")
print("  Random Forest sample : 0.5")
print()
print("LOCKED FILES")
print(f"  Protocol : {PROTOCOL_FILE}")
print(f"  Matrix   : {RUN_MATRIX_FILE}")
print(f"  Manifest : {LOCK_MANIFEST_FILE}")
print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"  {name}: {passed}")

print()
print(f"All checks passed: {all_checks_passed}")
print("TABULAR BASELINE PROTOCOL LOCK COMPLETED")
