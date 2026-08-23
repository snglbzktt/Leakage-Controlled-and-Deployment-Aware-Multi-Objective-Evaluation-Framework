from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_LOCK = (
    AUDIT
    / "phase5_compression_locked_v3_2.json"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE5_GROUP_SUMMARY = (
    AUDIT
    / "phase5_compression_group_summary_v3_2.csv"
)

PHASE5_CLOSURE_REPORT = (
    AUDIT
    / "phase5_compression_closure_v3_2.json"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase6_deployment_benchmark_protocol_v3_2.json"
)

OUTPUT_MATRIX = (
    AUDIT
    / "phase6_deployment_benchmark_locked_matrix_v3_2.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase6_deployment_benchmark_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase6_deployment_benchmark_v3_2"
UPSTREAM_PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"

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

PRIMARY_BATCH_SIZE = 1
SECONDARY_BATCH_SIZE = 32

WARMUP_ITERATIONS = {
    1: 50,
    32: 30,
}

MEASURED_ITERATIONS = {
    1: 500,
    32: 200,
}

CPU_THREADS = 1
INTEROP_THREADS = 1

LATENCY_METRICS = (
    "mean_ms",
    "median_ms",
    "std_ms",
    "p95_ms",
    "p99_ms",
    "minimum_ms",
    "maximum_ms",
)

MEMORY_METRICS = (
    "model_load_RSS_delta_bytes",
    "peak_inference_RSS_delta_bytes",
)

STORAGE_METRICS = (
    "source_checkpoint_file_bytes",
    "serialized_state_dict_bytes",
)

ORDER_RANDOM_SEED = 2026
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


def package_version(
    package_name: str,
) -> str | None:
    try:
        from importlib.metadata import version

        return version(
            package_name
        )
    except Exception:
        return None


required_paths = (
    PHASE5_LOCK,
    PHASE5_MASTER_MATRIX,
    PHASE5_GROUP_SUMMARY,
    PHASE5_CLOSURE_REPORT,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_MATRIX,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 6 deployment preflight "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
        )

phase5_lock = read_json(
    PHASE5_LOCK
)

phase5_closure = read_json(
    PHASE5_CLOSURE_REPORT
)

master_rows = read_csv(
    PHASE5_MASTER_MATRIX
)

phase5_checks = {
    "phase5_status_locked": (
        phase5_lock.get("status")
        == "locked"
    ),
    "phase5_protocol_version_matches": (
        phase5_lock.get(
            "protocol_version"
        )
        == UPSTREAM_PROTOCOL_VERSION
    ),
    "phase5_run_count_110": (
        int(
            phase5_lock.get(
                "final_run_count"
            )
        )
        == EXPECTED_RUN_COUNT
    ),
    "phase5_group_count_22": (
        int(
            phase5_lock.get(
                "group_count"
            )
        )
        == 22
    ),
    "phase5_ready_for_deployment": (
        phase5_lock.get(
            "ready_for_deployment_measurements"
        )
        is True
    ),
    "phase5_all_checks_passed": (
        phase5_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase5_master_hash_matches": (
        phase5_lock.get(
            "master_run_matrix_sha256"
        )
        == sha256_file(
            PHASE5_MASTER_MATRIX
        )
    ),
    "phase5_group_hash_matches": (
        phase5_lock.get(
            "group_summary_sha256"
        )
        == sha256_file(
            PHASE5_GROUP_SUMMARY
        )
    ),
    "phase5_closure_hash_matches": (
        phase5_lock.get(
            "closure_report_sha256"
        )
        == sha256_file(
            PHASE5_CLOSURE_REPORT
        )
    ),
    "phase5_final_model_not_selected": (
        phase5_lock.get(
            "final_model_selected"
        )
        is False
        and phase5_closure.get(
            "final_model_selected"
        )
        is False
    ),
}

failed_phase5_checks = [
    name
    for name, passed
    in phase5_checks.items()
    if not passed
]

if failed_phase5_checks:
    raise RuntimeError(
        "Phase 6 deployment preflight "
        "entry gate failed: "
        + ", ".join(
            failed_phase5_checks
        )
    )

observed_keys = {
    (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
    for row in master_rows
}

matrix_checks = {
    "master_row_count_110": (
        len(master_rows)
        == EXPECTED_RUN_COUNT
    ),
    "master_configuration_matrix_exact": (
        observed_keys
        == EXPECTED_KEYS
    ),
    "test_count_one_all_runs": (
        all(
            int(
                row[
                    "test_evaluation_count"
                ]
            )
            == 1
            for row in master_rows
        )
    ),
    "test_not_used_for_selection": (
        all(
            row[
                "test_used_for_selection"
            ].strip().lower()
            == "false"
            for row in master_rows
        )
    ),
}

failed_matrix_checks = [
    name
    for name, passed
    in matrix_checks.items()
    if not passed
]

if failed_matrix_checks:
    raise RuntimeError(
        "Phase 6 deployment matrix "
        "checks failed: "
        + ", ".join(
            failed_matrix_checks
        )
    )

psutil_available = (
    importlib.util.find_spec(
        "psutil"
    )
    is not None
)

if not psutil_available:
    raise RuntimeError(
        "The psutil package is required "
        "for RSS memory measurements."
    )

timer_resolution = (
    time.get_clock_info(
        "perf_counter"
    ).resolution
)

free_disk_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    / (
        1024 ** 3
    )
)

environment = {
    "captured_at_utc": utc_now(),
    "hostname": platform.node(),
    "operating_system": (
        platform.platform()
    ),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python_version": (
        platform.python_version()
    ),
    "python_executable": (
        sys.executable
    ),
    "torch_version": (
        torch.__version__
    ),
    "numpy_version": (
        np.__version__
    ),
    "psutil_version": (
        package_version(
            "psutil"
        )
    ),
    "scikit_learn_version": (
        package_version(
            "scikit-learn"
        )
    ),
    "logical_cpu_count": (
        os.cpu_count()
    ),
    "torch_default_threads": (
        torch.get_num_threads()
    ),
    "torch_default_interop_threads": (
        torch.get_num_interop_threads()
    ),
    "quantized_engine_current": (
        torch.backends.quantized.engine
    ),
    "quantized_engines_supported": list(
        torch.backends.quantized.supported_engines
    ),
    "perf_counter_resolution_seconds": (
        timer_resolution
    ),
    "free_disk_gib": (
        free_disk_gib
    ),
}

protocol = {
    "status": "locked",
    "phase": 6,
    "artifact_name": (
        "deployment_benchmark_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "upstream_protocol_version": (
        UPSTREAM_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "scientific_objective": (
        "Measure deployment cost under a "
        "single fixed local CPU protocol "
        "without using validation or test "
        "data for model selection."
    ),
    "scope": {
        "architectures": list(
            ARCHITECTURES
        ),
        "variants": list(
            VARIANTS
        ),
        "seeds": list(SEEDS),
        "run_count": (
            EXPECTED_RUN_COUNT
        ),
        "group_count": 22,
        "measure_all_locked_phase5_runs": (
            True
        ),
    },
    "input_policy": {
        "source_split": "train_only",
        "feature_count": 115,
        "fixed_input_index_seed": 2026,
        "primary_batch_size": (
            PRIMARY_BATCH_SIZE
        ),
        "secondary_batch_size": (
            SECONDARY_BATCH_SIZE
        ),
        "test_data_access": False,
        "validation_data_access": False,
    },
    "runtime_policy": {
        "device": "cpu",
        "cpu_threads": (
            CPU_THREADS
        ),
        "interop_threads": (
            INTEROP_THREADS
        ),
        "deterministic_algorithms": (
            True
        ),
        "process_isolation_per_run": (
            True
        ),
        "inference_mode": (
            "torch.inference_mode"
        ),
        "warmup_iterations": {
            str(key): value
            for key, value
            in WARMUP_ITERATIONS.items()
        },
        "measured_iterations": {
            str(key): value
            for key, value
            in MEASURED_ITERATIONS.items()
        },
        "timer": (
            "time.perf_counter_ns"
        ),
        "fixed_run_order_seed": (
            ORDER_RANDOM_SEED
        ),
    },
    "latency_metrics": list(
        LATENCY_METRICS
    ),
    "throughput_metrics": [
        "samples_per_second",
    ],
    "memory_metrics": list(
        MEMORY_METRICS
    ),
    "storage_metrics": list(
        STORAGE_METRICS
    ),
    "aggregation_policy": {
        "seed_level_results_preserved": (
            True
        ),
        "group_mean": True,
        "group_population_std": True,
        "group_minimum": True,
        "group_maximum": True,
        "primary_latency": (
            "batch_1_median_ms"
        ),
        "secondary_latency": (
            "batch_32_median_ms"
        ),
    },
    "selection_policy": {
        "final_model_selected_during_phase6": (
            False
        ),
        "deployment_metrics_used_later_with": [
            "locked_phase5_predictive_metrics",
            "predeclared_statistical_comparisons",
        ],
    },
    "environment": (
        environment
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

ordered_rows = []

for row in master_rows:
    architecture = (
        row["architecture"]
    )

    variant = row["variant"]
    seed = int(row["seed"])

    run_id = (
        f"{architecture}__{variant}__"
        f"seed_{seed}"
    )

    ordered_rows.append(
        {
            "run_id": run_id,
            "architecture": (
                architecture
            ),
            "variant": variant,
            "seed": seed,
            "representation": (
                row["representation"]
            ),
            "metrics_path": (
                row["metrics_path"]
            ),
            "metrics_sha256": (
                row["metrics_sha256"]
            ),
            "run_directory": (
                row["run_directory"]
            ),
            "primary_batch_size": (
                PRIMARY_BATCH_SIZE
            ),
            "secondary_batch_size": (
                SECONDARY_BATCH_SIZE
            ),
            "batch_1_warmup_iterations": (
                WARMUP_ITERATIONS[1]
            ),
            "batch_1_measured_iterations": (
                MEASURED_ITERATIONS[1]
            ),
            "batch_32_warmup_iterations": (
                WARMUP_ITERATIONS[32]
            ),
            "batch_32_measured_iterations": (
                MEASURED_ITERATIONS[32]
            ),
            "cpu_threads": (
                CPU_THREADS
            ),
            "interop_threads": (
                INTEROP_THREADS
            ),
            "process_isolation": True,
            "test_data_access": False,
            "validation_data_access": False,
        }
    )

random_generator = random.Random(
    ORDER_RANDOM_SEED
)

random_generator.shuffle(
    ordered_rows
)

for order_index, row in enumerate(
    ordered_rows,
    start=1,
):
    row["measurement_order"] = (
        order_index
    )

ordered_rows.sort(
    key=lambda row: int(
        row[
            "measurement_order"
        ]
    )
)

atomic_csv(
    OUTPUT_MATRIX,
    ordered_rows,
    [
        "measurement_order",
        "run_id",
        "architecture",
        "variant",
        "seed",
        "representation",
        "metrics_path",
        "metrics_sha256",
        "run_directory",
        "primary_batch_size",
        "secondary_batch_size",
        "batch_1_warmup_iterations",
        "batch_1_measured_iterations",
        "batch_32_warmup_iterations",
        "batch_32_measured_iterations",
        "cpu_threads",
        "interop_threads",
        "process_isolation",
        "test_data_access",
        "validation_data_access",
    ],
)

preflight_checks = {
    "psutil_available": (
        psutil_available
    ),
    "perf_counter_resolution_positive": (
        timer_resolution > 0.0
    ),
    "logical_cpu_count_available": (
        environment[
            "logical_cpu_count"
        ]
        is not None
        and int(
            environment[
                "logical_cpu_count"
            ]
        )
        >= 1
    ),
    "matrix_row_count_110": (
        len(ordered_rows)
        == EXPECTED_RUN_COUNT
    ),
    "measurement_order_unique": (
        len(
            {
                int(
                    row[
                        "measurement_order"
                    ]
                )
                for row in ordered_rows
            }
        )
        == EXPECTED_RUN_COUNT
    ),
    "test_access_zero": (
        all(
            row[
                "test_data_access"
            ]
            is False
            for row in ordered_rows
        )
    ),
    "validation_access_zero": (
        all(
            row[
                "validation_data_access"
            ]
            is False
            for row in ordered_rows
        )
    ),
    "final_model_not_selected": (
        protocol[
            "selection_policy"
        ][
            "final_model_selected_during_phase6"
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
        "Phase 6 deployment preflight "
        "failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 6,
    "artifact_name": (
        "deployment_benchmark_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "phase5_checks": (
        phase5_checks
    ),
    "matrix_checks": (
        matrix_checks
    ),
    "preflight_checks": (
        preflight_checks
    ),
    "environment": environment,
    "protocol": (
        file_record(
            OUTPUT_PROTOCOL
        )
    ),
    "locked_matrix": (
        file_record(
            OUTPUT_MATRIX
        )
    ),
    "ready_for_loader_smoke_test": (
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
    "phase": 6,
    "artifact_name": (
        "deployment_benchmark_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": 22,
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "locked_matrix": str(
        OUTPUT_MATRIX
    ),
    "locked_matrix_sha256": (
        sha256_file(
            OUTPUT_MATRIX
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
    "test_data_access": False,
    "validation_data_access": False,
    "final_model_selected": False,
    "ready_for_loader_smoke_test": (
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
        "deployment_benchmark_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_LOCK
        ),
        file_record(
            PHASE5_MASTER_MATRIX
        ),
        file_record(
            PHASE5_GROUP_SUMMARY
        ),
        file_record(
            PHASE5_CLOSURE_REPORT
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_PROTOCOL
        ),
        file_record(
            OUTPUT_MATRIX
        ),
        file_record(
            OUTPUT_PREFLIGHT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "run_count": (
        EXPECTED_RUN_COUNT
    ),
    "group_count": 22,
    "test_data_access": False,
    "validation_data_access": False,
    "final_model_selected": False,
    "ready_for_loader_smoke_test": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 6 DEPLOYMENT BENCHMARK PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Upstream Phase 5 status         : LOCKED"
)
print(
    "Deployment configurations       : 110"
)
print(
    "Architecture-variant groups     : 22"
)
print(
    "Primary latency batch size      : 1"
)
print(
    "Secondary throughput batch size : 32"
)
print(
    "CPU threads                     : 1"
)
print(
    "Interop threads                 : 1"
)
print(
    "Process isolation per run       : True"
)
print(
    "Batch 1 warmup / measured       : "
    f"{WARMUP_ITERATIONS[1]} / "
    f"{MEASURED_ITERATIONS[1]}"
)
print(
    "Batch 32 warmup / measured      : "
    f"{WARMUP_ITERATIONS[32]} / "
    f"{MEASURED_ITERATIONS[32]}"
)
print(
    "Memory measurement backend      : psutil RSS"
)
print(
    "Test data access                : False"
)
print(
    "Validation data access          : False"
)
print(
    "Final model selected            : False"
)
print(
    "Free disk                       : "
    f"{free_disk_gib:.3f} GiB"
)
print(
    "Protocol                        : "
    f"{OUTPUT_PROTOCOL}"
)
print(
    "Locked matrix                   : "
    f"{OUTPUT_MATRIX}"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for loader smoke test     : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 6 DEPLOYMENT BENCHMARK PROTOCOL PREFLIGHT LOCKED"
)
