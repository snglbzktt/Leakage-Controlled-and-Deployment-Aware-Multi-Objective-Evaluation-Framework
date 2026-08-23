from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
CONFIG = ROOT / "configs" / "protocols"

ASSET_ROOT = ROOT / "results" / "v2" / "manuscript_assets_v3_3"
TABLE_DIR = ASSET_ROOT / "tables"
SOURCE_DIR = ASSET_ROOT / "source_data"
CAPTION_DIR = ASSET_ROOT / "captions"
METADATA_DIR = ASSET_ROOT / "metadata"

BINDING_PATH = CONFIG / "phase13_exact_source_binding_v3_3.json"
BINDING_LOCK_PATH = AUDIT / "phase13_exact_source_binding_locked_v3_3.json"
MAPPING_PATH = CONFIG / "phase13_manuscript_column_mapping_v3_3.json"
MAPPING_LOCK_PATH = AUDIT / "phase13_manuscript_column_mapping_locked_v3_3.json"

TABLE_03_LOCK = AUDIT / "phase13_table_03_compression_results_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_04_deployment_benchmarks_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_04_deployment_benchmarks_v3_3.md"
SOURCE_CSV = SOURCE_DIR / "table_04_deployment_benchmarks_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "table_04_deployment_benchmarks_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_04_deployment_benchmarks_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_04_deployment_benchmarks_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

ARCHITECTURE_LABELS = {
    "tinyml_mlp": "TinyML-MLP",
    "compact_dnn": "Compact-DNN",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None

    for attempt in range(1, WINDOWS_FILE_RETRY_COUNT + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(WINDOWS_FILE_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: {destination}"
    ) from last_error


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    replace_with_retry(temporary, path)


def get_bound_source(binding: dict[str, Any], source_id: str) -> Path:
    registry = binding["source_registry"]

    if source_id not in registry:
        raise KeyError(f"Source ID not found in binding: {source_id}")

    record = registry[source_id]
    path = Path(record["path"])

    if not path.exists():
        raise FileNotFoundError(path)

    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Bound source changed after lock: {path}")

    return path


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
    TABLE_03_LOCK,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for output_path in (
    TABLE_CSV,
    TABLE_MD,
    SOURCE_CSV,
    CAPTION_TXT,
    METADATA_JSON,
    LOCK_JSON,
):
    if output_path.exists():
        raise FileExistsError(
            "Table 04 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_03_lock = read_json(TABLE_03_LOCK)

entry_checks = {
    "binding_locked": (
        binding.get("status") == "locked"
        and binding.get("ready_for_asset_generation") is True
        and binding.get("all_checks_passed") is True
    ),
    "binding_hash_matches": (
        binding_lock["binding"]["sha256"] == sha256_file(BINDING_PATH)
    ),
    "mapping_validated": (
        mapping.get("status") == "validated"
        and mapping.get("ready_for_asset_generation") is True
        and mapping.get("all_checks_passed") is True
    ),
    "mapping_hash_matches": (
        mapping_lock["mapping"]["sha256"] == sha256_file(MAPPING_PATH)
    ),
    "table_03_locked": (
        table_03_lock.get("status") == "locked"
        and table_03_lock.get("ready_for_manuscript") is True
        and table_03_lock.get("all_checks_passed") is True
    ),
    "table_04_mapped": any(
        asset["asset_id"] == "table_04_deployment_benchmarks"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 04 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

group_path = get_bound_source(binding, "phase6_group_summary")
summary_path = get_bound_source(binding, "phase6_summary")

summary = read_json(summary_path)

with group_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    source_rows = list(csv.DictReader(handle))

required_columns = {
    "architecture",
    "variant",
    "representation",
    "run_count",
    "mean_serialized_state_dict_bytes",
    "mean_peak_inference_RSS_delta_bytes",
    "mean_batch_1_median_ms",
    "std_batch_1_median_ms",
    "mean_batch_1_p95_ms",
    "mean_batch_32_samples_per_second",
    "validation_data_access",
    "test_data_access",
}

headers = set(source_rows[0].keys()) if source_rows else set()

data_checks = {
    "source_rows_22": len(source_rows) == 22,
    "required_columns_present": required_columns.issubset(headers),
    "architecture_count_2": len({row["architecture"] for row in source_rows}) == 2,
    "eleven_groups_per_architecture": all(
        sum(1 for row in source_rows if row["architecture"] == architecture) == 11
        for architecture in ARCHITECTURE_LABELS
    ),
    "run_count_5_all": all(int(row["run_count"]) == 5 for row in source_rows),
    "validation_access_false_all": all(
        parse_bool(row["validation_data_access"]) is False
        for row in source_rows
    ),
    "test_access_false_all": all(
        parse_bool(row["test_data_access"]) is False
        for row in source_rows
    ),
    "summary_run_count_110": summary["run_count"] == 110,
    "summary_group_count_22": summary["group_count"] == 22,
    "summary_cpu_threads_1": summary["runtime_policy"]["cpu_threads"] == 1,
    "summary_interop_threads_1": summary["runtime_policy"]["interop_threads"] == 1,
    "summary_validation_false": summary["data_access"]["validation_data_access"] is False,
    "summary_test_false": summary["data_access"]["test_data_access"] is False,
    "summary_final_model_not_selected": summary["selection_policy"]["final_model_selected"] is False,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 04 source validation failed: "
        + ", ".join(failed_data_checks)
    )

architecture_order = {
    "tinyml_mlp": 0,
    "compact_dnn": 1,
}

variant_order = {
    "B0": 0,
    "FP32-FT": 1,
    "P25-noFT": 2,
    "P25-FP32-FT": 3,
    "P50-noFT": 4,
    "P50-FP32-FT": 5,
    "DQ": 6,
    "PTQ": 7,
    "QAT": 8,
    "P25-QAT": 9,
    "P50-QAT": 10,
}

source_rows_sorted = sorted(
    source_rows,
    key=lambda row: (
        architecture_order[row["architecture"]],
        variant_order[row["variant"]],
    ),
)

table_rows: list[dict[str, str]] = []

for row in source_rows_sorted:
    state_kib = float(row["mean_serialized_state_dict_bytes"]) / 1024.0
    peak_rss_mib = float(row["mean_peak_inference_RSS_delta_bytes"]) / (1024.0 ** 2)
    batch1_median = float(row["mean_batch_1_median_ms"])
    batch1_std = float(row["std_batch_1_median_ms"])
    batch1_p95 = float(row["mean_batch_1_p95_ms"])
    throughput = float(row["mean_batch_32_samples_per_second"])

    table_rows.append(
        {
            "Architecture": ARCHITECTURE_LABELS[row["architecture"]],
            "Variant": row["variant"],
            "Representation": row["representation"],
            "Runs": row["run_count"],
            "Serialized state (KiB)": f"{state_kib:.3f}",
            "Peak inference RSS Δ (MiB)": f"{peak_rss_mib:.3f}",
            "Batch-1 median ± std (ms)": (
                f"{batch1_median:.6f} ± {batch1_std:.6f}"
            ),
            "Batch-1 p95 (ms)": f"{batch1_p95:.6f}",
            "Batch-32 throughput (samples/s)": f"{throughput:.3f}",
            "Val/Test accessed": "No / No",
        }
    )

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

table_fieldnames = [
    "Architecture",
    "Variant",
    "Representation",
    "Runs",
    "Serialized state (KiB)",
    "Peak inference RSS Δ (MiB)",
    "Batch-1 median ± std (ms)",
    "Batch-1 p95 (ms)",
    "Batch-32 throughput (samples/s)",
    "Val/Test accessed",
]

with TABLE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(handle, fieldnames=table_fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

with SOURCE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(source_rows_sorted[0].keys()),
    )
    writer.writeheader()
    writer.writerows(source_rows_sorted)

md_lines = [
    "| Architecture | Variant | Representation | Runs | Serialized state (KiB) | Peak inference RSS Δ (MiB) | Batch-1 median ± std (ms) | Batch-1 p95 (ms) | Batch-32 throughput (samples/s) | Val/Test accessed |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
]

for row in table_rows:
    md_lines.append(
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|")
            for field in table_fieldnames
        )
        + " |"
    )

atomic_text(TABLE_MD, "\n".join(md_lines) + "\n")

caption = (
    "Table 4. Locked CPU deployment measurements for the 22 architecture–variant "
    "groups. Each value summarizes five isolated benchmark runs using one CPU thread "
    "and one inter-op thread. Batch-1 latency and batch-32 throughput were measured "
    "without validation- or test-data access. Phase 6 did not perform final model "
    "selection; the deployment measurements were subsequently used by the predeclared "
    "multi-objective decision stage."
)

atomic_text(CAPTION_TXT, caption + "\n")

fastest_batch1 = min(
    source_rows_sorted,
    key=lambda row: float(row["mean_batch_1_median_ms"]),
)

highest_throughput = max(
    source_rows_sorted,
    key=lambda row: float(row["mean_batch_32_samples_per_second"]),
)

smallest_state = min(
    source_rows_sorted,
    key=lambda row: float(row["mean_serialized_state_dict_bytes"]),
)

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_04_deployment_benchmarks",
    "generated_at_utc": utc_now(),
    "title": "Deployment Benchmark Results",
    "row_count": len(table_rows),
    "benchmark_policy": {
        "process_isolation_per_run": summary["runtime_policy"]["process_isolation_per_run"],
        "cpu_threads": summary["runtime_policy"]["cpu_threads"],
        "interop_threads": summary["runtime_policy"]["interop_threads"],
        "batch_1_warmup_iterations": summary["runtime_policy"]["batch_1_warmup_iterations"],
        "batch_1_measured_iterations": summary["runtime_policy"]["batch_1_measured_iterations"],
        "batch_32_warmup_iterations": summary["runtime_policy"]["batch_32_warmup_iterations"],
        "batch_32_measured_iterations": summary["runtime_policy"]["batch_32_measured_iterations"],
    },
    "descriptive_extremes": {
        "fastest_batch1": {
            "candidate_id": (
                f"{fastest_batch1['architecture']}::{fastest_batch1['variant']}"
            ),
            "mean_batch_1_median_ms": float(
                fastest_batch1["mean_batch_1_median_ms"]
            ),
        },
        "highest_batch32_throughput": {
            "candidate_id": (
                f"{highest_throughput['architecture']}::{highest_throughput['variant']}"
            ),
            "mean_batch_32_samples_per_second": float(
                highest_throughput["mean_batch_32_samples_per_second"]
            ),
        },
        "smallest_serialized_state": {
            "candidate_id": (
                f"{smallest_state['architecture']}::{smallest_state['variant']}"
            ),
            "mean_serialized_state_dict_bytes": float(
                smallest_state["mean_serialized_state_dict_bytes"]
            ),
        },
    },
    "final_model_selected_in_phase6": False,
    "validation_data_access": False,
    "test_data_access": False,
    "source_files": [
        {
            "source_id": "phase6_group_summary",
            "path": str(group_path),
            "sha256": sha256_file(group_path),
        },
        {
            "source_id": "phase6_summary",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
    ],
    "outputs": {
        "csv": {
            "path": str(TABLE_CSV),
            "sha256": sha256_file(TABLE_CSV),
            "size_bytes": TABLE_CSV.stat().st_size,
        },
        "markdown": {
            "path": str(TABLE_MD),
            "sha256": sha256_file(TABLE_MD),
            "size_bytes": TABLE_MD.stat().st_size,
        },
        "source_csv": {
            "path": str(SOURCE_CSV),
            "sha256": sha256_file(SOURCE_CSV),
            "size_bytes": SOURCE_CSV.stat().st_size,
        },
        "caption": {
            "path": str(CAPTION_TXT),
            "sha256": sha256_file(CAPTION_TXT),
            "size_bytes": CAPTION_TXT.stat().st_size,
        },
    },
    "entry_checks": entry_checks,
    "data_checks": data_checks,
    "model_inference_performed": False,
    "validation_data_accessed": False,
    "test_data_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "table_04_deployment_benchmarks",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "row_count": 22,
    "benchmark_run_count": 110,
    "cpu_threads": 1,
    "interop_threads": 1,
    "final_model_selected_in_phase6": False,
    "model_inference_performed": False,
    "validation_data_accessed": False,
    "test_data_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 04 - DEPLOYMENT BENCHMARKS")
print("=" * 96)
print(f"Table rows                       : {len(table_rows)}")
print("Benchmark runs                   : 110")
print(f"CPU threads                      : {summary['runtime_policy']['cpu_threads']}")
print(f"Interop threads                  : {summary['runtime_policy']['interop_threads']}")
print("Validation data accessed         : False")
print("Test data accessed               : False")
print("Final model selected in Phase 6 : False")
print()
print(
    "Fastest batch-1 candidate        : "
    f"{fastest_batch1['architecture']}::{fastest_batch1['variant']} "
    f"({float(fastest_batch1['mean_batch_1_median_ms']):.6f} ms)"
)
print(
    "Highest batch-32 throughput      : "
    f"{highest_throughput['architecture']}::{highest_throughput['variant']} "
    f"({float(highest_throughput['mean_batch_32_samples_per_second']):.3f} samples/s)"
)
print(
    "Smallest serialized state        : "
    f"{smallest_state['architecture']}::{smallest_state['variant']} "
    f"({float(smallest_state['mean_serialized_state_dict_bytes']) / 1024.0:.3f} KiB)"
)
print()
print(f"CSV                              : {TABLE_CSV}")
print(f"Markdown                         : {TABLE_MD}")
print(f"Source CSV                       : {SOURCE_CSV}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model inference performed        : False")
print("Validation data accessed         : False")
print("Test data accessed               : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 TABLE 04 GENERATED AND LOCKED")
