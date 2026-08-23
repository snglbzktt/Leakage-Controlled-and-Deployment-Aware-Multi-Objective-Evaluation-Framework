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


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

PHASE5_PROTOCOL_LOCK = (
    AUDIT / "phase5_protocol_lock_manifest_v3_2.json"
)

PREPROCESSING_LOCK = (
    AUDIT / "phase5_preprocessing_locked_v3_2.json"
)

SMOKE_LOCK = (
    AUDIT
    / "phase5_neural_architecture_smoke_test_locked_v3_2.json"
)

TINYML_LOCK = (
    AUDIT / "phase5_tinyml_mlp_b0_locked_v3_2.json"
)

TINYML_VERIFICATION = (
    AUDIT / "phase5_tinyml_mlp_b0_verification_v3_2.json"
)

TINYML_AGGREGATE = (
    AUDIT / "phase5_tinyml_mlp_b0_aggregate_v3_2.json"
)

TINYML_RUNS_CSV = (
    AUDIT / "phase5_tinyml_mlp_b0_runs_v3_2.csv"
)

COMPACT_LOCK = (
    AUDIT / "phase5_compact_dnn_b0_locked_v3_2.json"
)

COMPACT_VERIFICATION = (
    AUDIT / "phase5_compact_dnn_b0_verification_v3_2.json"
)

COMPACT_AGGREGATE = (
    AUDIT / "phase5_compact_dnn_b0_aggregate_v3_2.json"
)

COMPACT_RUNS_CSV = (
    AUDIT / "phase5_compact_dnn_b0_runs_v3_2.csv"
)

OUTPUT_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

OUTPUT_PAIRED_COMPARISON = (
    AUDIT / "phase5_B0_paired_seed_comparison_v3_2.csv"
)

OUTPUT_SUMMARY = (
    AUDIT / "phase5_B0_pair_summary_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT / "phase5_B0_pair_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_RUN_COUNT_PER_ARCHITECTURE = 5
EXPECTED_TOTAL_RUN_COUNT = 10

TOLERANCE = 1e-12


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


def close_enough(
    observed: float,
    expected: float,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    )


def artifact_record(
    manifest: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in manifest["artifacts"]
        if str(
            record["relative_path"]
        )
        == filename
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one artifact "
            f"record for {filename}; "
            f"found {len(matches)}."
        )

    return matches[0]


required_paths = (
    PHASE5_PROTOCOL,
    PHASE5_PROTOCOL_LOCK,
    PREPROCESSING_LOCK,
    SMOKE_LOCK,
    TINYML_LOCK,
    TINYML_VERIFICATION,
    TINYML_AGGREGATE,
    TINYML_RUNS_CSV,
    COMPACT_LOCK,
    COMPACT_VERIFICATION,
    COMPACT_AGGREGATE,
    COMPACT_RUNS_CSV,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_CHECKPOINT_REGISTRY,
    OUTPUT_PAIRED_COMPARISON,
    OUTPUT_SUMMARY,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 5 B0 pair artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

phase5_protocol = read_json(
    PHASE5_PROTOCOL
)

phase5_protocol_lock = read_json(
    PHASE5_PROTOCOL_LOCK
)

preprocessing_lock = read_json(
    PREPROCESSING_LOCK
)

smoke_lock = read_json(
    SMOKE_LOCK
)

tinyml_lock = read_json(
    TINYML_LOCK
)

tinyml_verification = read_json(
    TINYML_VERIFICATION
)

tinyml_aggregate = read_json(
    TINYML_AGGREGATE
)

tinyml_rows = read_csv(
    TINYML_RUNS_CSV
)

compact_lock = read_json(
    COMPACT_LOCK
)

compact_verification = read_json(
    COMPACT_VERIFICATION
)

compact_aggregate = read_json(
    COMPACT_AGGREGATE
)

compact_rows = read_csv(
    COMPACT_RUNS_CSV
)

entry_checks = {
    "phase5_protocol_locked": (
        phase5_protocol.get("status")
        == "locked"
        and phase5_protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "phase5_protocol_lock_passed": (
        phase5_protocol_lock.get(
            "status"
        )
        == "locked"
        and phase5_protocol_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "preprocessing_locked": (
        preprocessing_lock.get(
            "status"
        )
        == "locked"
        and preprocessing_lock.get(
            "all_checks_passed"
        )
        is True
        and preprocessing_lock.get(
            "ready_for_B0_training"
        )
        is True
    ),
    "smoke_test_locked": (
        smoke_lock.get("status")
        == "locked"
        and smoke_lock.get(
            "all_checks_passed"
        )
        is True
        and smoke_lock.get(
            "ready_for_B0_training_runs"
        )
        is True
    ),
    "tinyml_lock_passed": (
        tinyml_lock.get("status")
        == "locked"
        and tinyml_lock.get(
            "all_checks_passed"
        )
        is True
        and tinyml_lock.get(
            "run_count"
        )
        == 5
    ),
    "compact_lock_passed": (
        compact_lock.get("status")
        == "locked"
        and compact_lock.get(
            "all_checks_passed"
        )
        is True
        and compact_lock.get(
            "ready_for_phase5_B0_pair_lock"
        )
        is True
        and compact_lock.get(
            "run_count"
        )
        == 5
    ),
    "tinyml_verification_passed": (
        tinyml_verification.get(
            "status"
        )
        == "passed"
        and tinyml_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "compact_verification_passed": (
        compact_verification.get(
            "status"
        )
        == "passed"
        and compact_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "tinyml_verification_hash_matches": (
        tinyml_lock.get(
            "verification_sha256"
        )
        == sha256_file(
            TINYML_VERIFICATION
        )
    ),
    "compact_verification_hash_matches": (
        compact_lock.get(
            "verification_sha256"
        )
        == sha256_file(
            COMPACT_VERIFICATION
        )
    ),
    "tinyml_aggregate_hash_matches": (
        tinyml_lock.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            TINYML_AGGREGATE
        )
        and tinyml_lock.get(
            "aggregate_csv_sha256"
        )
        == sha256_file(
            TINYML_RUNS_CSV
        )
    ),
    "compact_aggregate_hash_matches": (
        compact_lock.get(
            "aggregate_json_sha256"
        )
        == sha256_file(
            COMPACT_AGGREGATE
        )
        and compact_lock.get(
            "aggregate_csv_sha256"
        )
        == sha256_file(
            COMPACT_RUNS_CSV
        )
    ),
    "no_fragility_in_tinyml": (
        tinyml_lock.get(
            "fragility_triggered_in_any_run"
        )
        is False
    ),
    "no_fragility_in_compact": (
        compact_lock.get(
            "fragility_triggered_in_any_run"
        )
        is False
    ),
    "test_count_one_tinyml": (
        int(
            tinyml_lock.get(
                "test_evaluation_count_per_run"
            )
        )
        == 1
    ),
    "test_count_one_compact": (
        int(
            compact_lock.get(
                "test_evaluation_count_per_run"
            )
        )
        == 1
    ),
    "test_inference_not_repeated_tinyml": (
        tinyml_lock.get(
            "test_model_inference_repeated_by_verifier"
        )
        is False
    ),
    "test_inference_not_repeated_compact": (
        compact_lock.get(
            "test_model_inference_repeated_by_verifier"
        )
        is False
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
        "Phase 5 B0 pair entry gate "
        "failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

architecture_sources = {
    "tinyml_mlp": {
        "lock": tinyml_lock,
        "verification": tinyml_verification,
        "aggregate": tinyml_aggregate,
        "rows": tinyml_rows,
        "lock_path": TINYML_LOCK,
        "verification_path": (
            TINYML_VERIFICATION
        ),
        "aggregate_path": (
            TINYML_AGGREGATE
        ),
        "aggregate_csv_path": (
            TINYML_RUNS_CSV
        ),
    },
    "compact_dnn": {
        "lock": compact_lock,
        "verification": compact_verification,
        "aggregate": compact_aggregate,
        "rows": compact_rows,
        "lock_path": COMPACT_LOCK,
        "verification_path": (
            COMPACT_VERIFICATION
        ),
        "aggregate_path": (
            COMPACT_AGGREGATE
        ),
        "aggregate_csv_path": (
            COMPACT_RUNS_CSV
        ),
    },
}

checkpoint_rows: list[
    dict[str, Any]
] = []

model_summaries: list[
    dict[str, Any]
] = []

run_lookup: dict[
    tuple[str, int],
    dict[str, Any],
] = {}

for architecture in ARCHITECTURES:
    source = architecture_sources[
        architecture
    ]

    aggregate = source[
        "aggregate"
    ]

    aggregate_rows = source[
        "rows"
    ]

    if (
        aggregate.get(
            "architecture"
        )
        != architecture
        or aggregate.get("variant")
        != "B0"
        or int(
            aggregate.get(
                "run_count"
            )
        )
        != EXPECTED_RUN_COUNT_PER_ARCHITECTURE
    ):
        raise RuntimeError(
            "Aggregate identity mismatch "
            f"for {architecture}."
        )

    if sorted(
        int(row["seed"])
        for row in aggregate_rows
    ) != list(SEEDS):
        raise RuntimeError(
            "Aggregate CSV seed set "
            f"mismatch for {architecture}."
        )

    for row in aggregate_rows:
        seed = int(row["seed"])

        run_directory = Path(
            row["output_directory"]
        )

        manifest_path = (
            run_directory
            / "run_manifest.json"
        )

        metrics_path = (
            run_directory
            / "metrics.json"
        )

        checkpoint_path = (
            run_directory
            / "best_checkpoint.pt"
        )

        status_path = (
            run_directory
            / "run_status.json"
        )

        for path in (
            run_directory,
            manifest_path,
            metrics_path,
            checkpoint_path,
            status_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        manifest = read_json(
            manifest_path
        )

        metrics = read_json(
            metrics_path
        )

        status = read_json(
            status_path
        )

        checkpoint_record = (
            artifact_record(
                manifest,
                "best_checkpoint.pt",
            )
        )

        run_checks = {
            "manifest_completed": (
                manifest.get("status")
                == "completed"
                and manifest.get(
                    "all_integrity_checks_passed"
                )
                is True
            ),
            "status_completed": (
                status.get("status")
                == "completed"
            ),
            "identity_matches": (
                manifest.get(
                    "architecture"
                )
                == architecture
                and manifest.get(
                    "variant"
                )
                == "B0"
                and int(
                    manifest.get("seed")
                )
                == seed
                and metrics.get(
                    "architecture"
                )
                == architecture
                and int(
                    metrics.get("seed")
                )
                == seed
            ),
            "test_count_is_one": (
                int(
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
            ),
            "test_not_used_for_selection": (
                metrics["selection"][
                    "test_used_for_selection"
                ]
                is False
                and manifest[
                    "fit_scope"
                ][
                    "test_used_for_selection"
                ]
                is False
            ),
            "fragility_false": (
                metrics[
                    "fragility_gate"
                ]["triggered"]
                is False
            ),
            "checkpoint_hash_matches_manifest": (
                str(
                    checkpoint_record[
                        "sha256"
                    ]
                )
                == sha256_file(
                    checkpoint_path
                )
            ),
            "checkpoint_size_matches_manifest": (
                int(
                    checkpoint_record[
                        "size_bytes"
                    ]
                )
                == int(
                    checkpoint_path.stat().st_size
                )
            ),
            "metrics_hash_matches_aggregate": (
                str(
                    row[
                        "metrics_sha256"
                    ]
                )
                == sha256_file(
                    metrics_path
                )
            ),
            "manifest_hash_matches_aggregate": (
                str(
                    row[
                        "run_manifest_sha256"
                    ]
                )
                == sha256_file(
                    manifest_path
                )
            ),
        }

        failed_run_checks = [
            name
            for name, passed
            in run_checks.items()
            if not passed
        ]

        if failed_run_checks:
            raise RuntimeError(
                f"{architecture} seed {seed} "
                "B0 registry check failed: "
                + ", ".join(
                    failed_run_checks
                )
            )

        registry_row = {
            "architecture": (
                architecture
            ),
            "variant": "B0",
            "seed": seed,
            "run_id": row["run_id"],
            "model_symbol": (
                aggregate[
                    "model_symbol"
                ]
            ),
            "best_epoch": int(
                row["best_epoch"]
            ),
            "epochs_completed": int(
                row["epochs_completed"]
            ),
            "checkpoint_path": str(
                checkpoint_path
            ),
            "checkpoint_size_bytes": int(
                checkpoint_path.stat().st_size
            ),
            "checkpoint_sha256": (
                sha256_file(
                    checkpoint_path
                )
            ),
            "run_manifest_path": str(
                manifest_path
            ),
            "run_manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
            "metrics_path": str(
                metrics_path
            ),
            "metrics_sha256": (
                sha256_file(
                    metrics_path
                )
            ),
            "test_fingerprint_macro_f1": float(
                row[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "test_fingerprint_accuracy": float(
                row[
                    "test_fingerprint_accuracy"
                ]
            ),
            "test_raw_weighted_macro_f1": float(
                row[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "test_gafgyt_fnr": float(
                row["test_gafgyt_fnr"]
            ),
            "test_mirai_fnr": float(
                row["test_mirai_fnr"]
            ),
            "parameter_count": int(
                row["parameter_count"]
            ),
            "linear_macs_per_sample": int(
                row[
                    "linear_macs_per_sample"
                ]
            ),
            "raw_tensor_bytes": int(
                row["raw_tensor_bytes"]
            ),
            "serialized_checkpoint_bytes": int(
                row[
                    "serialized_checkpoint_bytes"
                ]
            ),
            "test_evaluation_count": 1,
            "fragility_gate_triggered": False,
            "source_lock_path": str(
                source["lock_path"]
            ),
            "source_lock_sha256": (
                sha256_file(
                    source["lock_path"]
                )
            ),
            "status": (
                "locked_B0_source_checkpoint"
            ),
        }

        checkpoint_rows.append(
            registry_row
        )

        run_lookup[
            (
                architecture,
                seed,
            )
        ] = registry_row

    aggregate_metrics = aggregate[
        "aggregate_metrics"
    ]

    model_summaries.append(
        {
            "architecture": (
                architecture
            ),
            "run_count": 5,
            "seeds": list(SEEDS),
            "mean_test_fingerprint_macro_f1": float(
                aggregate_metrics[
                    "test_fingerprint_macro_f1"
                ]["mean"]
            ),
            "std_test_fingerprint_macro_f1": float(
                aggregate_metrics[
                    "test_fingerprint_macro_f1"
                ][
                    "std_population"
                ]
            ),
            "min_test_fingerprint_macro_f1": float(
                aggregate_metrics[
                    "test_fingerprint_macro_f1"
                ]["minimum"]
            ),
            "max_test_fingerprint_macro_f1": float(
                aggregate_metrics[
                    "test_fingerprint_macro_f1"
                ]["maximum"]
            ),
            "mean_test_raw_weighted_macro_f1": float(
                aggregate_metrics[
                    "test_raw_weighted_macro_f1"
                ]["mean"]
            ),
            "mean_test_gafgyt_fnr": float(
                aggregate_metrics[
                    "test_gafgyt_fnr"
                ]["mean"]
            ),
            "mean_test_mirai_fnr": float(
                aggregate_metrics[
                    "test_mirai_fnr"
                ]["mean"]
            ),
            "parameter_count": int(
                checkpoint_rows[-1][
                    "parameter_count"
                ]
            ),
            "linear_macs_per_sample": int(
                checkpoint_rows[-1][
                    "linear_macs_per_sample"
                ]
            ),
            "fragility_triggered_in_any_run": False,
        }
    )

if len(
    checkpoint_rows
) != EXPECTED_TOTAL_RUN_COUNT:
    raise RuntimeError(
        "B0 checkpoint registry does not "
        "contain exactly ten rows."
    )

checkpoint_rows.sort(
    key=lambda row: (
        row["architecture"],
        int(row["seed"]),
    )
)

registry_checks = {
    "registry_has_ten_rows": (
        len(checkpoint_rows)
        == EXPECTED_TOTAL_RUN_COUNT
    ),
    "registry_architecture_set_matches": (
        {
            row["architecture"]
            for row in checkpoint_rows
        }
        == set(ARCHITECTURES)
    ),
    "five_rows_per_architecture": all(
        sum(
            1
            for row in checkpoint_rows
            if row["architecture"]
            == architecture
        )
        == 5
        for architecture in ARCHITECTURES
    ),
    "seed_set_per_architecture_matches": all(
        sorted(
            int(row["seed"])
            for row in checkpoint_rows
            if row["architecture"]
            == architecture
        )
        == list(SEEDS)
        for architecture in ARCHITECTURES
    ),
    "checkpoint_hashes_unique": (
        len(
            {
                row["checkpoint_sha256"]
                for row in checkpoint_rows
            }
        )
        == EXPECTED_TOTAL_RUN_COUNT
    ),
    "all_test_counts_are_one": all(
        int(
            row[
                "test_evaluation_count"
            ]
        )
        == 1
        for row in checkpoint_rows
    ),
    "no_fragility": all(
        row[
            "fragility_gate_triggered"
        ]
        is False
        for row in checkpoint_rows
    ),
}

failed_registry_checks = [
    name
    for name, passed
    in registry_checks.items()
    if not passed
]

if failed_registry_checks:
    raise RuntimeError(
        "B0 checkpoint registry checks "
        "failed: "
        + ", ".join(
            failed_registry_checks
        )
    )

atomic_csv(
    OUTPUT_CHECKPOINT_REGISTRY,
    checkpoint_rows,
    [
        "architecture",
        "variant",
        "seed",
        "run_id",
        "model_symbol",
        "best_epoch",
        "epochs_completed",
        "checkpoint_path",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "run_manifest_path",
        "run_manifest_sha256",
        "metrics_path",
        "metrics_sha256",
        "test_fingerprint_macro_f1",
        "test_fingerprint_accuracy",
        "test_raw_weighted_macro_f1",
        "test_gafgyt_fnr",
        "test_mirai_fnr",
        "parameter_count",
        "linear_macs_per_sample",
        "raw_tensor_bytes",
        "serialized_checkpoint_bytes",
        "test_evaluation_count",
        "fragility_gate_triggered",
        "source_lock_path",
        "source_lock_sha256",
        "status",
    ],
)

paired_rows: list[
    dict[str, Any]
] = []

for seed in SEEDS:
    tinyml = run_lookup[
        ("tinyml_mlp", seed)
    ]

    compact = run_lookup[
        ("compact_dnn", seed)
    ]

    paired_rows.append(
        {
            "seed": seed,
            "tinyml_test_fingerprint_macro_f1": (
                tinyml[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "compact_test_fingerprint_macro_f1": (
                compact[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "compact_minus_tinyml_macro_f1": (
                compact[
                    "test_fingerprint_macro_f1"
                ]
                - tinyml[
                    "test_fingerprint_macro_f1"
                ]
            ),
            "tinyml_test_raw_weighted_macro_f1": (
                tinyml[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "compact_test_raw_weighted_macro_f1": (
                compact[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "compact_minus_tinyml_raw_weighted_macro_f1": (
                compact[
                    "test_raw_weighted_macro_f1"
                ]
                - tinyml[
                    "test_raw_weighted_macro_f1"
                ]
            ),
            "tinyml_parameter_count": (
                tinyml["parameter_count"]
            ),
            "compact_parameter_count": (
                compact["parameter_count"]
            ),
            "compact_to_tinyml_parameter_ratio": (
                compact["parameter_count"]
                / tinyml["parameter_count"]
            ),
            "tinyml_linear_macs_per_sample": (
                tinyml[
                    "linear_macs_per_sample"
                ]
            ),
            "compact_linear_macs_per_sample": (
                compact[
                    "linear_macs_per_sample"
                ]
            ),
            "compact_to_tinyml_mac_ratio": (
                compact[
                    "linear_macs_per_sample"
                ]
                / tinyml[
                    "linear_macs_per_sample"
                ]
            ),
            "interpretation": (
                "descriptive_only_not_model_selection"
            ),
        }
    )

atomic_csv(
    OUTPUT_PAIRED_COMPARISON,
    paired_rows,
    [
        "seed",
        "tinyml_test_fingerprint_macro_f1",
        "compact_test_fingerprint_macro_f1",
        "compact_minus_tinyml_macro_f1",
        "tinyml_test_raw_weighted_macro_f1",
        "compact_test_raw_weighted_macro_f1",
        "compact_minus_tinyml_raw_weighted_macro_f1",
        "tinyml_parameter_count",
        "compact_parameter_count",
        "compact_to_tinyml_parameter_ratio",
        "tinyml_linear_macs_per_sample",
        "compact_linear_macs_per_sample",
        "compact_to_tinyml_mac_ratio",
        "interpretation",
    ],
)

macro_deltas = np.asarray(
    [
        row[
            "compact_minus_tinyml_macro_f1"
        ]
        for row in paired_rows
    ],
    dtype=np.float64,
)

weighted_deltas = np.asarray(
    [
        row[
            "compact_minus_tinyml_raw_weighted_macro_f1"
        ]
        for row in paired_rows
    ],
    dtype=np.float64,
)

ranked_models = sorted(
    model_summaries,
    key=lambda row: row[
        "mean_test_fingerprint_macro_f1"
    ],
    reverse=True,
)

for index, row in enumerate(
    ranked_models,
    start=1,
):
    row[
        "descriptive_macro_f1_rank"
    ] = index

summary = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "B0_architecture_pair"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "generated_at_utc": utc_now(),
    "architecture_count": 2,
    "run_count": 10,
    "seeds": list(SEEDS),
    "architectures": (
        ranked_models
    ),
    "paired_seed_descriptive_comparison": {
        "compact_minus_tinyml_macro_f1_mean": float(
            np.mean(
                macro_deltas
            )
        ),
        "compact_minus_tinyml_macro_f1_std_population": float(
            np.std(
                macro_deltas,
                ddof=0,
            )
        ),
        "compact_minus_tinyml_raw_weighted_macro_f1_mean": float(
            np.mean(
                weighted_deltas
            )
        ),
        "comparison_is_final_model_selection": False,
        "comparison_is_statistical_superiority_claim": False,
    },
    "source_policy": {
        "ten_checkpoints_are_locked_B0_sources": True,
        "matching_architecture_and_seed_required_for_all_branches": True,
        "P25_and_P50_generated_independently_from_matching_B0": True,
        "FP32_FT_generated_from_matching_B0": True,
        "DQ_PTQ_QAT_generated_from_matching_B0": True,
        "legacy_checkpoints_allowed": False,
    },
    "test_policy": {
        "test_evaluation_count_per_run": 1,
        "test_used_for_selection": False,
        "test_model_inference_repeated_by_verifiers": False,
    },
    "entry_checks": entry_checks,
    "registry_checks": (
        registry_checks
    ),
    "checkpoint_registry_csv": str(
        OUTPUT_CHECKPOINT_REGISTRY
    ),
    "paired_comparison_csv": str(
        OUTPUT_PAIRED_COMPARISON
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SUMMARY,
    summary,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "B0_architecture_pair"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "architecture_count": 2,
    "run_count": 10,
    "seeds": list(SEEDS),
    "checkpoint_count": 10,
    "checkpoint_registry": str(
        OUTPUT_CHECKPOINT_REGISTRY
    ),
    "checkpoint_registry_sha256": (
        sha256_file(
            OUTPUT_CHECKPOINT_REGISTRY
        )
    ),
    "paired_comparison": str(
        OUTPUT_PAIRED_COMPARISON
    ),
    "paired_comparison_sha256": (
        sha256_file(
            OUTPUT_PAIRED_COMPARISON
        )
    ),
    "summary": str(
        OUTPUT_SUMMARY
    ),
    "summary_sha256": (
        sha256_file(
            OUTPUT_SUMMARY
        )
    ),
    "final_model_selected": False,
    "ready_for_branch_source_generation": True,
    "next_action": (
        "Build and verify branch-source "
        "artifacts for FP32-FT, P25-noFT, "
        "P50-noFT, DQ, PTQ, and QAT."
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

source_inventory = [
    file_record(
        PHASE5_PROTOCOL
    ),
    file_record(
        PHASE5_PROTOCOL_LOCK
    ),
    file_record(
        PREPROCESSING_LOCK
    ),
    file_record(
        SMOKE_LOCK
    ),
    file_record(
        TINYML_LOCK
    ),
    file_record(
        TINYML_VERIFICATION
    ),
    file_record(
        TINYML_AGGREGATE
    ),
    file_record(
        TINYML_RUNS_CSV
    ),
    file_record(
        COMPACT_LOCK
    ),
    file_record(
        COMPACT_VERIFICATION
    ),
    file_record(
        COMPACT_AGGREGATE
    ),
    file_record(
        COMPACT_RUNS_CSV
    ),
]

generated_inventory = [
    file_record(
        OUTPUT_CHECKPOINT_REGISTRY
    ),
    file_record(
        OUTPUT_PAIRED_COMPARISON
    ),
    file_record(
        OUTPUT_SUMMARY
    ),
    file_record(
        OUTPUT_LOCK
    ),
]

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "B0_architecture_pair"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": (
        source_inventory
    ),
    "generated_artifacts": (
        generated_inventory
    ),
    "checkpoint_count": 10,
    "ready_for_branch_source_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "checkpoint_registry_exists": (
        OUTPUT_CHECKPOINT_REGISTRY.exists()
    ),
    "paired_comparison_exists": (
        OUTPUT_PAIRED_COMPARISON.exists()
    ),
    "summary_exists": (
        OUTPUT_SUMMARY.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "lock_registry_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )[
            "checkpoint_registry_sha256"
        ]
        == sha256_file(
            OUTPUT_CHECKPOINT_REGISTRY
        )
    ),
    "lock_summary_hash_matches": (
        read_json(
            OUTPUT_LOCK
        )[
            "summary_sha256"
        ]
        == sha256_file(
            OUTPUT_SUMMARY
        )
    ),
    "ready_for_branch_sources": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_branch_source_generation"
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
        "Phase 5 B0 pair post-lock "
        "checks failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print("=" * 92)
print("PHASE 5 B0 ARCHITECTURE PAIR LOCK SUMMARY")
print("=" * 92)
print(
    "Architectures locked            : 2"
)
print(
    "B0 runs locked                  : 10"
)
print(
    "B0 source checkpoints registered: 10"
)
print(
    "Seeds                           : "
    f"{list(SEEDS)}"
)
print()

for row in ranked_models:
    print(
        f"{row['descriptive_macro_f1_rank']}. "
        f"{row['architecture']:<16} | "
        "mean Macro-F1="
        f"{row['mean_test_fingerprint_macro_f1']:.9f} | "
        "std="
        f"{row['std_test_fingerprint_macro_f1']:.9f} | "
        "parameters="
        f"{row['parameter_count']:,} | "
        "MACs/sample="
        f"{row['linear_macs_per_sample']:,}"
    )

print()
print(
    "Paired Compact-TinyML delta     : "
    f"{np.mean(macro_deltas):+.9f}"
)
print(
    "Final model selected            : False"
)
print(
    "Test evaluation count per run   : 1"
)
print(
    "Test inference repeated         : False"
)
print(
    "Fragility in any run            : False"
)
print(
    "B0 architecture pair status     : LOCKED"
)
print(
    "Ready for branch sources        : True"
)
print(
    "Checkpoint registry             : "
    f"{OUTPUT_CHECKPOINT_REGISTRY}"
)
print(
    "Pair summary                    : "
    f"{OUTPUT_SUMMARY}"
)
print(
    "Lock manifest                   : "
    f"{OUTPUT_LOCK_MANIFEST}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 B0 ARCHITECTURE PAIR LOCKED"
)
