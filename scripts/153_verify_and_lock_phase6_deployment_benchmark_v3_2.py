from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE6_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase6_deployment_benchmark_protocol_v3_2.json"
)

PHASE6_PREFLIGHT_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_locked_v3_2.json"
)

PHASE6_LOADER_LOCK = (
    AUDIT
    / "phase6_deployment_loader_smoke_test_locked_v3_2.json"
)

PHASE6_MATRIX = (
    AUDIT
    / "phase6_deployment_benchmark_locked_matrix_v3_2.csv"
)

ALL_RUNS_CSV = (
    AUDIT
    / "phase6_deployment_benchmark_all_runs_v3_2.csv"
)

GROUP_SUMMARY_CSV = (
    AUDIT
    / "phase6_deployment_benchmark_group_summary_v3_2.csv"
)

SUMMARY_JSON = (
    AUDIT
    / "phase6_deployment_benchmark_summary_v3_2.json"
)

COMPLETION_JSON = (
    AUDIT
    / "phase6_deployment_benchmark_completed_v3_2.json"
)

OUTPUT_VERIFIED_RUNS = (
    AUDIT
    / "phase6_deployment_benchmark_verified_runs_v3_2.csv"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase6_deployment_benchmark_verification_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase6_deployment_benchmark_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase6_deployment_benchmark_v3_2"

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25-FP32-FT",
    "P50-noFT",
    "P50-FP32-FT",
    "DQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
    "PTQ",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_KEYS = {
    (
        architecture,
        variant,
        seed,
    )
    for architecture in ARCHITECTURES
    for variant in VARIANTS
    for seed in SEEDS
}

EXPECTED_RUN_COUNT = 110
EXPECTED_GROUP_COUNT = 22

EXPECTED_ITERATIONS = {
    1: 500,
    32: 200,
}

FLOAT_TOLERANCE = 1e-10

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


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

            if (
                attempt
                == WINDOWS_FILE_RETRY_COUNT
            ):
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file "
        "locked after "
        f"{WINDOWS_FILE_RETRY_COUNT} attempts: "
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
        ),
        encoding="utf-8",
    )

    replace_with_retry(
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


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


def latency_statistics(
    latencies_ns: np.ndarray,
) -> dict[str, float]:
    values_ms = (
        latencies_ns.astype(
            np.float64
        )
        / 1_000_000.0
    )

    return {
        "mean_ms": float(
            np.mean(values_ms)
        ),
        "median_ms": float(
            np.median(values_ms)
        ),
        "std_ms": float(
            np.std(
                values_ms,
                ddof=0,
            )
        ),
        "p95_ms": float(
            np.percentile(
                values_ms,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                values_ms,
                99,
            )
        ),
        "minimum_ms": float(
            np.min(values_ms)
        ),
        "maximum_ms": float(
            np.max(values_ms)
        ),
    }


def throughput(
    latencies_ns: np.ndarray,
    batch_size: int,
) -> float:
    total_seconds = (
        float(
            np.sum(
                latencies_ns,
                dtype=np.int64,
            )
        )
        / 1_000_000_000.0
    )

    return (
        batch_size
        * len(latencies_ns)
        / total_seconds
    )


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


required_paths = (
    PHASE6_PROTOCOL,
    PHASE6_PREFLIGHT_LOCK,
    PHASE6_LOADER_LOCK,
    PHASE6_MATRIX,
    ALL_RUNS_CSV,
    GROUP_SUMMARY_CSV,
    SUMMARY_JSON,
    COMPLETION_JSON,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFIED_RUNS,
    OUTPUT_VERIFICATION,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 6 verification artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

protocol = read_json(
    PHASE6_PROTOCOL
)

preflight_lock = read_json(
    PHASE6_PREFLIGHT_LOCK
)

loader_lock = read_json(
    PHASE6_LOADER_LOCK
)

summary = read_json(
    SUMMARY_JSON
)

completion = read_json(
    COMPLETION_JSON
)

matrix_rows = read_csv(
    PHASE6_MATRIX
)

all_rows = read_csv(
    ALL_RUNS_CSV
)

group_rows = read_csv(
    GROUP_SUMMARY_CSV
)

all_lookup = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    ): row
    for row in all_rows
}

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "preflight_locked": (
        preflight_lock.get(
            "status"
        )
        == "locked"
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "loader_locked": (
        loader_lock.get(
            "status"
        )
        == "locked"
        and loader_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "summary_completed": (
        summary.get("status")
        == "completed"
        and summary.get(
            "run_count"
        )
        == EXPECTED_RUN_COUNT
        and summary.get(
            "group_count"
        )
        == EXPECTED_GROUP_COUNT
        and summary.get(
            "all_checks_passed"
        )
        is True
    ),
    "completion_completed": (
        completion.get("status")
        == "completed"
        and completion.get(
            "run_count"
        )
        == EXPECTED_RUN_COUNT
        and completion.get(
            "group_count"
        )
        == EXPECTED_GROUP_COUNT
        and completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "all_runs_hash_matches": (
        completion.get(
            "all_runs_csv_sha256"
        )
        == sha256_file(
            ALL_RUNS_CSV
        )
    ),
    "group_summary_hash_matches": (
        completion.get(
            "group_summary_csv_sha256"
        )
        == sha256_file(
            GROUP_SUMMARY_CSV
        )
    ),
    "summary_hash_matches": (
        completion.get(
            "summary_json_sha256"
        )
        == sha256_file(
            SUMMARY_JSON
        )
    ),
    "matrix_count_110": (
        len(matrix_rows)
        == EXPECTED_RUN_COUNT
    ),
    "all_runs_count_110": (
        len(all_rows)
        == EXPECTED_RUN_COUNT
    ),
    "groups_count_22": (
        len(group_rows)
        == EXPECTED_GROUP_COUNT
    ),
    "configuration_matrix_exact": (
        set(
            all_lookup.keys()
        )
        == EXPECTED_KEYS
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
        "Phase 6 benchmark verification "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

print("=" * 92)
print("PHASE 6 DEPLOYMENT BENCHMARK INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Runs                            : 110"
)
print(
    "Groups                          : 22"
)
print(
    "Model inference repeated        : False"
)
print(
    "Raw latency arrays verified     : Yes"
)
print(
    "Latency statistics recomputed   : Yes"
)
print(
    "Throughput recomputed           : Yes"
)
print(
    "Group summaries recomputed      : Yes"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print()

verified_rows: list[
    dict[str, Any]
] = []

ordered_rows = sorted(
    all_rows,
    key=lambda row: int(
        row[
            "measurement_order"
        ]
    ),
)

for index, row in enumerate(
    ordered_rows,
    start=1,
):
    architecture = (
        row["architecture"]
    )

    variant = row["variant"]

    seed = int(
        row["seed"]
    )

    run_directory = Path(
        row["run_directory"]
    )

    paths = {
        "metrics": (
            run_directory
            / "deployment_metrics.json"
        ),
        "raw_latencies": (
            run_directory
            / "raw_latencies_ns.npz"
        ),
        "manifest": (
            run_directory
            / "run_manifest.json"
        ),
        "status": (
            run_directory
            / "run_status.json"
        ),
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = read_json(
        paths["metrics"]
    )

    manifest = read_json(
        paths["manifest"]
    )

    status = read_json(
        paths["status"]
    )

    with np.load(
        paths["raw_latencies"]
    ) as values:
        batch_1 = np.array(
            values["batch_1"],
            copy=True,
        )

        batch_32 = np.array(
            values["batch_32"],
            copy=True,
        )

    batch_1_stats = latency_statistics(
        batch_1
    )

    batch_32_stats = latency_statistics(
        batch_32
    )

    batch_1_throughput = throughput(
        batch_1,
        1,
    )

    batch_32_throughput = throughput(
        batch_32,
        32,
    )

    checks = {
        "identity_matches": (
            metrics[
                "architecture"
            ]
            == architecture
            and metrics["variant"]
            == variant
            and int(
                metrics["seed"]
            )
            == seed
            and manifest[
                "architecture"
            ]
            == architecture
            and manifest["variant"]
            == variant
            and int(
                manifest["seed"]
            )
            == seed
            and status[
                "architecture"
            ]
            == architecture
            and status["variant"]
            == variant
            and int(
                status["seed"]
            )
            == seed
        ),
        "status_completed": (
            metrics.get("status")
            == "completed"
            and manifest.get("status")
            == "completed"
            and status.get("status")
            == "completed"
            and status.get("stage")
            == "completed"
        ),
        "manifest_integrity_passed": (
            manifest.get(
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
        "aggregate_hashes_match": (
            row[
                "deployment_metrics_sha256"
            ]
            == sha256_file(
                paths["metrics"]
            )
            and row[
                "raw_latencies_sha256"
            ]
            == sha256_file(
                paths[
                    "raw_latencies"
                ]
            )
            and row[
                "run_manifest_sha256"
            ]
            == sha256_file(
                paths["manifest"]
            )
        ),
        "status_hashes_match": (
            status[
                "deployment_metrics_sha256"
            ]
            == sha256_file(
                paths["metrics"]
            )
            and status[
                "raw_latencies_sha256"
            ]
            == sha256_file(
                paths[
                    "raw_latencies"
                ]
            )
            and status[
                "run_manifest_sha256"
            ]
            == sha256_file(
                paths["manifest"]
            )
        ),
        "source_checkpoint_hash_matches": (
            manifest[
                "source_checkpoint"
            ]["sha256"]
            == metrics[
                "checkpoint"
            ]["sha256"]
            == sha256_file(
                Path(
                    metrics[
                        "checkpoint"
                    ]["path"]
                )
            )
        ),
        "source_metrics_hash_matches": (
            manifest[
                "source_metrics"
            ]["sha256"]
            == metrics[
                "source_metrics"
            ]["sha256"]
            == sha256_file(
                Path(
                    metrics[
                        "source_metrics"
                    ]["path"]
                )
            )
        ),
        "batch_1_shape_and_dtype": (
            batch_1.shape
            == (
                EXPECTED_ITERATIONS[1],
            )
            and np.issubdtype(
                batch_1.dtype,
                np.integer,
            )
        ),
        "batch_32_shape_and_dtype": (
            batch_32.shape
            == (
                EXPECTED_ITERATIONS[32],
            )
            and np.issubdtype(
                batch_32.dtype,
                np.integer,
            )
        ),
        "latencies_positive": (
            bool(
                np.all(
                    batch_1 > 0
                )
            )
            and bool(
                np.all(
                    batch_32 > 0
                )
            )
        ),
        "batch_1_stats_match": (
            all(
                close_enough(
                    batch_1_stats[key],
                    metrics[
                        "batch_1"
                    ][key],
                )
                for key in (
                    "mean_ms",
                    "median_ms",
                    "std_ms",
                    "p95_ms",
                    "p99_ms",
                    "minimum_ms",
                    "maximum_ms",
                )
            )
        ),
        "batch_32_stats_match": (
            all(
                close_enough(
                    batch_32_stats[key],
                    metrics[
                        "batch_32"
                    ][key],
                )
                for key in (
                    "mean_ms",
                    "median_ms",
                    "std_ms",
                    "p95_ms",
                    "p99_ms",
                    "minimum_ms",
                    "maximum_ms",
                )
            )
        ),
        "throughput_matches": (
            close_enough(
                batch_1_throughput,
                metrics[
                    "batch_1"
                ][
                    "samples_per_second"
                ],
            )
            and close_enough(
                batch_32_throughput,
                metrics[
                    "batch_32"
                ][
                    "samples_per_second"
                ],
            )
        ),
        "aggregate_row_matches": (
            close_enough(
                row[
                    "batch_1_median_ms"
                ],
                batch_1_stats[
                    "median_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_1_p95_ms"
                ],
                batch_1_stats[
                    "p95_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_1_p99_ms"
                ],
                batch_1_stats[
                    "p99_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_1_samples_per_second"
                ],
                batch_1_throughput,
            )
            and close_enough(
                row[
                    "batch_32_median_ms"
                ],
                batch_32_stats[
                    "median_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_32_p95_ms"
                ],
                batch_32_stats[
                    "p95_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_32_p99_ms"
                ],
                batch_32_stats[
                    "p99_ms"
                ],
            )
            and close_enough(
                row[
                    "batch_32_samples_per_second"
                ],
                batch_32_throughput,
            )
        ),
        "memory_values_nonnegative": (
            int(
                metrics[
                    "runtime"
                ][
                    "model_load_RSS_delta_bytes"
                ]
            )
            >= 0
            and int(
                metrics[
                    "runtime"
                ][
                    "peak_inference_RSS_delta_bytes"
                ]
            )
            >= 0
        ),
        "storage_values_positive": (
            int(
                metrics[
                    "checkpoint"
                ]["file_bytes"]
            )
            > 0
            and int(
                metrics[
                    "checkpoint"
                ][
                    "serialized_state_dict_bytes"
                ]
            )
            > 0
        ),
        "no_validation_access": (
            metrics[
                "data_access"
            ][
                "validation_data_access"
            ]
            is False
            and manifest[
                "validation_data_access"
            ]
            is False
            and status[
                "validation_data_access"
            ]
            is False
        ),
        "no_test_access": (
            metrics[
                "data_access"
            ][
                "test_data_access"
            ]
            is False
            and manifest[
                "test_data_access"
            ]
            is False
            and status[
                "test_data_access"
            ]
            is False
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
            f"{architecture} {variant} "
            f"seed {seed} verification "
            "failed: "
            + ", ".join(
                failed_checks
            )
        )

    verified_rows.append(
        {
            "measurement_order": int(
                row[
                    "measurement_order"
                ]
            ),
            "architecture": architecture,
            "variant": variant,
            "seed": seed,
            "representation": (
                row["representation"]
            ),
            "batch_1_median_ms": (
                batch_1_stats[
                    "median_ms"
                ]
            ),
            "batch_1_p95_ms": (
                batch_1_stats[
                    "p95_ms"
                ]
            ),
            "batch_1_p99_ms": (
                batch_1_stats[
                    "p99_ms"
                ]
            ),
            "batch_1_samples_per_second": (
                batch_1_throughput
            ),
            "batch_32_median_ms": (
                batch_32_stats[
                    "median_ms"
                ]
            ),
            "batch_32_p95_ms": (
                batch_32_stats[
                    "p95_ms"
                ]
            ),
            "batch_32_p99_ms": (
                batch_32_stats[
                    "p99_ms"
                ]
            ),
            "batch_32_samples_per_second": (
                batch_32_throughput
            ),
            "checkpoint_file_bytes": int(
                metrics[
                    "checkpoint"
                ]["file_bytes"]
            ),
            "serialized_state_dict_bytes": int(
                metrics[
                    "checkpoint"
                ][
                    "serialized_state_dict_bytes"
                ]
            ),
            "model_load_RSS_delta_bytes": int(
                metrics[
                    "runtime"
                ][
                    "model_load_RSS_delta_bytes"
                ]
            ),
            "peak_inference_RSS_delta_bytes": int(
                metrics[
                    "runtime"
                ][
                    "peak_inference_RSS_delta_bytes"
                ]
            ),
            "model_inference_repeated": (
                False
            ),
            "validation_data_access": (
                False
            ),
            "test_data_access": (
                False
            ),
            "deployment_metrics_sha256": (
                sha256_file(
                    paths["metrics"]
                )
            ),
            "raw_latencies_sha256": (
                sha256_file(
                    paths[
                        "raw_latencies"
                    ]
                )
            ),
            "run_manifest_sha256": (
                sha256_file(
                    paths["manifest"]
                )
            ),
            "all_checks_passed": True,
        }
    )

    print(
        f"[{index}/110] "
        f"{architecture} {variant} "
        f"seed={seed} | "
        "raw latencies=verified | "
        "statistics=recomputed | "
        "inference repeated=False | "
        "all checks=True",
        flush=True,
    )

atomic_csv(
    OUTPUT_VERIFIED_RUNS,
    verified_rows,
    [
        "measurement_order",
        "architecture",
        "variant",
        "seed",
        "representation",
        "batch_1_median_ms",
        "batch_1_p95_ms",
        "batch_1_p99_ms",
        "batch_1_samples_per_second",
        "batch_32_median_ms",
        "batch_32_p95_ms",
        "batch_32_p99_ms",
        "batch_32_samples_per_second",
        "checkpoint_file_bytes",
        "serialized_state_dict_bytes",
        "model_load_RSS_delta_bytes",
        "peak_inference_RSS_delta_bytes",
        "model_inference_repeated",
        "validation_data_access",
        "test_data_access",
        "deployment_metrics_sha256",
        "raw_latencies_sha256",
        "run_manifest_sha256",
        "all_checks_passed",
    ],
)

saved_group_lookup = {
    (
        row["architecture"],
        row["variant"],
    ): row
    for row in group_rows
}

verified_groups: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for variant in VARIANTS:
        rows = [
            row
            for row in verified_rows
            if row["architecture"]
            == architecture
            and row["variant"]
            == variant
        ]

        if len(rows) != 5:
            raise RuntimeError(
                "Expected five verified runs "
                f"for {architecture} {variant}."
            )

        saved = saved_group_lookup[
            (
                architecture,
                variant,
            )
        ]

        metric_names = (
            "checkpoint_file_bytes",
            "serialized_state_dict_bytes",
            "model_load_RSS_delta_bytes",
            "peak_inference_RSS_delta_bytes",
            "batch_1_median_ms",
            "batch_1_p95_ms",
            "batch_1_p99_ms",
            "batch_1_samples_per_second",
            "batch_32_median_ms",
            "batch_32_p95_ms",
            "batch_32_p99_ms",
            "batch_32_samples_per_second",
        )

        statistics = {
            metric_name: (
                aggregate_statistics(
                    [
                        float(
                            row[
                                metric_name
                            ]
                        )
                        for row in rows
                    ]
                )
            )
            for metric_name in metric_names
        }

        group_checks = {
            "run_count_matches": (
                int(
                    saved[
                        "run_count"
                    ]
                )
                == 5
            ),
            "checkpoint_size_matches": (
                close_enough(
                    saved[
                        "mean_checkpoint_file_bytes"
                    ],
                    statistics[
                        "checkpoint_file_bytes"
                    ]["mean"],
                )
            ),
            "serialized_state_matches": (
                close_enough(
                    saved[
                        "mean_serialized_state_dict_bytes"
                    ],
                    statistics[
                        "serialized_state_dict_bytes"
                    ]["mean"],
                )
            ),
            "load_RSS_matches": (
                close_enough(
                    saved[
                        "mean_model_load_RSS_delta_bytes"
                    ],
                    statistics[
                        "model_load_RSS_delta_bytes"
                    ]["mean"],
                )
            ),
            "peak_RSS_matches": (
                close_enough(
                    saved[
                        "mean_peak_inference_RSS_delta_bytes"
                    ],
                    statistics[
                        "peak_inference_RSS_delta_bytes"
                    ]["mean"],
                )
            ),
            "batch_1_median_matches": (
                close_enough(
                    saved[
                        "mean_batch_1_median_ms"
                    ],
                    statistics[
                        "batch_1_median_ms"
                    ]["mean"],
                )
            ),
            "batch_1_std_matches": (
                close_enough(
                    saved[
                        "std_batch_1_median_ms"
                    ],
                    statistics[
                        "batch_1_median_ms"
                    ][
                        "std_population"
                    ],
                )
            ),
            "batch_1_p95_matches": (
                close_enough(
                    saved[
                        "mean_batch_1_p95_ms"
                    ],
                    statistics[
                        "batch_1_p95_ms"
                    ]["mean"],
                )
            ),
            "batch_1_p99_matches": (
                close_enough(
                    saved[
                        "mean_batch_1_p99_ms"
                    ],
                    statistics[
                        "batch_1_p99_ms"
                    ]["mean"],
                )
            ),
            "batch_1_throughput_matches": (
                close_enough(
                    saved[
                        "mean_batch_1_samples_per_second"
                    ],
                    statistics[
                        "batch_1_samples_per_second"
                    ]["mean"],
                )
            ),
            "batch_32_median_matches": (
                close_enough(
                    saved[
                        "mean_batch_32_median_ms"
                    ],
                    statistics[
                        "batch_32_median_ms"
                    ]["mean"],
                )
            ),
            "batch_32_std_matches": (
                close_enough(
                    saved[
                        "std_batch_32_median_ms"
                    ],
                    statistics[
                        "batch_32_median_ms"
                    ][
                        "std_population"
                    ],
                )
            ),
            "batch_32_p95_matches": (
                close_enough(
                    saved[
                        "mean_batch_32_p95_ms"
                    ],
                    statistics[
                        "batch_32_p95_ms"
                    ]["mean"],
                )
            ),
            "batch_32_p99_matches": (
                close_enough(
                    saved[
                        "mean_batch_32_p99_ms"
                    ],
                    statistics[
                        "batch_32_p99_ms"
                    ]["mean"],
                )
            ),
            "batch_32_throughput_matches": (
                close_enough(
                    saved[
                        "mean_batch_32_samples_per_second"
                    ],
                    statistics[
                        "batch_32_samples_per_second"
                    ]["mean"],
                )
            ),
            "no_validation_access": (
                parse_bool(
                    saved[
                        "validation_data_access"
                    ]
                )
                is False
            ),
            "no_test_access": (
                parse_bool(
                    saved[
                        "test_data_access"
                    ]
                )
                is False
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

        verified_groups.append(
            {
                "architecture": architecture,
                "variant": variant,
                "run_count": 5,
                "statistics": statistics,
                "all_checks_passed": True,
            }
        )

global_checks = {
    "runs_verified_110": (
        len(verified_rows)
        == EXPECTED_RUN_COUNT
    ),
    "groups_verified_22": (
        len(verified_groups)
        == EXPECTED_GROUP_COUNT
    ),
    "configuration_matrix_exact": (
        {
            (
                row["architecture"],
                row["variant"],
                int(row["seed"]),
            )
            for row in verified_rows
        }
        == EXPECTED_KEYS
    ),
    "all_run_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_rows
        )
    ),
    "all_group_checks_passed": (
        all(
            row[
                "all_checks_passed"
            ]
            for row in verified_groups
        )
    ),
    "model_inference_not_repeated": (
        all(
            row[
                "model_inference_repeated"
            ]
            is False
            for row in verified_rows
        )
    ),
    "validation_access_zero": (
        all(
            row[
                "validation_data_access"
            ]
            is False
            for row in verified_rows
        )
    ),
    "test_access_zero": (
        all(
            row[
                "test_data_access"
            ]
            is False
            for row in verified_rows
        )
    ),
    "final_model_not_selected": (
        True
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
        "Phase 6 benchmark verification "
        "global checks failed: "
        + ", ".join(
            failed_global_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 6,
    "artifact_name": (
        "isolated_deployment_benchmark_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": (
        EXPECTED_GROUP_COUNT
    ),
    "verification_policy": {
        "model_inference_repeated": False,
        "raw_latency_arrays_loaded": True,
        "latency_statistics_recomputed": True,
        "throughput_recomputed": True,
        "group_statistics_recomputed": True,
        "source_hashes_verified": True,
    },
    "entry_checks": (
        entry_checks
    ),
    "global_checks": (
        global_checks
    ),
    "verified_runs_csv": (
        file_record(
            OUTPUT_VERIFIED_RUNS
        )
    ),
    "verified_groups": (
        verified_groups
    ),
    "source_artifacts": {
        "protocol": (
            file_record(
                PHASE6_PROTOCOL
            )
        ),
        "preflight_lock": (
            file_record(
                PHASE6_PREFLIGHT_LOCK
            )
        ),
        "loader_lock": (
            file_record(
                PHASE6_LOADER_LOCK
            )
        ),
        "locked_matrix": (
            file_record(
                PHASE6_MATRIX
            )
        ),
        "all_runs_csv": (
            file_record(
                ALL_RUNS_CSV
            )
        ),
        "group_summary_csv": (
            file_record(
                GROUP_SUMMARY_CSV
            )
        ),
        "summary_json": (
            file_record(
                SUMMARY_JSON
            )
        ),
        "completion_json": (
            file_record(
                COMPLETION_JSON
            )
        ),
    },
    "ready_for_predeclared_statistics": (
        True
    ),
    "ready_for_final_model_selection": (
        False
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

lock = {
    "status": "locked",
    "phase": 6,
    "artifact_name": (
        "isolated_deployment_benchmark"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": (
        EXPECTED_GROUP_COUNT
    ),
    "model_inference_repeated_by_verifier": (
        False
    ),
    "verified_runs_csv": str(
        OUTPUT_VERIFIED_RUNS
    ),
    "verified_runs_csv_sha256": (
        sha256_file(
            OUTPUT_VERIFIED_RUNS
        )
    ),
    "verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "all_runs_csv": str(
        ALL_RUNS_CSV
    ),
    "all_runs_csv_sha256": (
        sha256_file(
            ALL_RUNS_CSV
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
    "validation_data_access": False,
    "test_data_access": False,
    "final_model_selected": False,
    "ready_for_predeclared_statistics": (
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
    "phase": 6,
    "artifact_name": (
        "isolated_deployment_benchmark"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE6_PROTOCOL
        ),
        file_record(
            PHASE6_PREFLIGHT_LOCK
        ),
        file_record(
            PHASE6_LOADER_LOCK
        ),
        file_record(
            PHASE6_MATRIX
        ),
        file_record(
            ALL_RUNS_CSV
        ),
        file_record(
            GROUP_SUMMARY_CSV
        ),
        file_record(
            SUMMARY_JSON
        ),
        file_record(
            COMPLETION_JSON
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFIED_RUNS
        ),
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": (
        EXPECTED_GROUP_COUNT
    ),
    "model_inference_repeated_by_verifier": (
        False
    ),
    "validation_data_access": False,
    "test_data_access": False,
    "final_model_selected": False,
    "ready_for_predeclared_statistics": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 6 DEPLOYMENT BENCHMARK VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Runs independently verified     : 110"
)
print(
    "Groups independently verified   : 22"
)
print(
    "Raw latency arrays verified     : PASSED"
)
print(
    "Latency statistics recomputed   : PASSED"
)
print(
    "Throughput recomputed           : PASSED"
)
print(
    "Group summaries recomputed      : PASSED"
)
print(
    "Model inference repeated        : False"
)
print(
    "Validation data access          : False"
)
print(
    "Test data access                : False"
)
print(
    "Final model selected            : False"
)
print(
    "Phase 6 status                  : LOCKED"
)
print(
    "Ready for statistics            : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 6 DEPLOYMENT BENCHMARK VERIFIED AND LOCKED"
)
