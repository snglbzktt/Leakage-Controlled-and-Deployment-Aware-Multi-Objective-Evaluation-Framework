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

TABLE_01_LOCK = AUDIT / "phase13_table_01_dataset_protocol_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_02_classical_baselines_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_02_classical_baselines_v3_3.md"
SOURCE_CSV = SOURCE_DIR / "table_02_classical_baselines_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "table_02_classical_baselines_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_02_classical_baselines_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_02_classical_baselines_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


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


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
    TABLE_01_LOCK,
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
            "Table 02 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_01_lock = read_json(TABLE_01_LOCK)

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
    "table_01_locked": (
        table_01_lock.get("status") == "locked"
        and table_01_lock.get("ready_for_manuscript") is True
        and table_01_lock.get("all_checks_passed") is True
    ),
    "table_02_mapped": any(
        asset["asset_id"] == "table_02_classical_baselines"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 02 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

source_path = get_bound_source(binding, "phase4_model_summary")

with source_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    source_rows = list(csv.DictReader(handle))

required_columns = {
    "descriptive_macro_f1_rank",
    "model_id",
    "display_name",
    "run_count",
    "seed_policy",
    "input_space",
    "mean_test_fingerprint_macro_f1",
    "std_test_fingerprint_macro_f1",
    "mean_test_raw_weighted_macro_f1",
    "mean_test_gafgyt_fnr",
    "mean_test_mirai_fnr",
    "scientific_status",
    "important_caveat",
    "ranking_is_not_model_selection",
}

headers = set(source_rows[0].keys()) if source_rows else set()

data_checks = {
    "source_rows_4": len(source_rows) == 4,
    "required_columns_present": required_columns.issubset(headers),
    "ranks_1_to_4": (
        sorted(int(row["descriptive_macro_f1_rank"]) for row in source_rows)
        == [1, 2, 3, 4]
    ),
    "ranking_not_model_selection_all": all(
        row["ranking_is_not_model_selection"].strip().lower() == "true"
        for row in source_rows
    ),
    "all_macro_f1_valid": all(
        0.0 <= float(row["mean_test_fingerprint_macro_f1"]) <= 1.0
        for row in source_rows
    ),
    "all_weighted_macro_f1_valid": all(
        0.0 <= float(row["mean_test_raw_weighted_macro_f1"]) <= 1.0
        for row in source_rows
    ),
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 02 source validation failed: "
        + ", ".join(failed_data_checks)
    )

source_rows_sorted = sorted(
    source_rows,
    key=lambda row: int(row["descriptive_macro_f1_rank"]),
)

table_rows: list[dict[str, str]] = []

for row in source_rows_sorted:
    macro_mean = float(row["mean_test_fingerprint_macro_f1"])
    macro_std = float(row["std_test_fingerprint_macro_f1"])
    weighted = float(row["mean_test_raw_weighted_macro_f1"])
    gafgyt_fnr = float(row["mean_test_gafgyt_fnr"])
    mirai_fnr = float(row["mean_test_mirai_fnr"])

    caveat = row["important_caveat"].strip()
    if not caveat:
        caveat = "None."

    table_rows.append(
        {
            "Rank": row["descriptive_macro_f1_rank"],
            "Model": row["display_name"],
            "Runs": row["run_count"],
            "Seed policy": row["seed_policy"],
            "Input space": row["input_space"],
            "Macro-F1 mean ± std": f"{macro_mean:.6f} ± {macro_std:.6f}",
            "Raw-weighted Macro-F1": f"{weighted:.6f}",
            "Gafgyt FNR": f"{gafgyt_fnr:.6f}",
            "Mirai FNR": f"{mirai_fnr:.6f}",
            "Scientific status / caveat": (
                f"{row['scientific_status']}. {caveat}"
            ),
        }
    )

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

table_fieldnames = [
    "Rank",
    "Model",
    "Runs",
    "Seed policy",
    "Input space",
    "Macro-F1 mean ± std",
    "Raw-weighted Macro-F1",
    "Gafgyt FNR",
    "Mirai FNR",
    "Scientific status / caveat",
]

with TABLE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=table_fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(source_rows_sorted[0].keys()),
    )
    writer.writeheader()
    writer.writerows(source_rows_sorted)

md_lines = [
    "| Rank | Model | Runs | Seed policy | Input space | Macro-F1 mean ± std | Raw-weighted Macro-F1 | Gafgyt FNR | Mirai FNR | Scientific status / caveat |",
    "|---:|---|---:|---|---|---:|---:|---:|---:|---|",
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
    "Table 2. Classical tabular baseline results on the locked grouped-float32 "
    "evaluation protocol. Ranking is descriptive only and does not constitute "
    "final TinyML model selection. HistGradientBoosting and Logistic Regression "
    "use a single locked seed, whereas Decision Tree and Random Forest use five "
    "locked seeds; therefore, their uncertainty estimates are not interpreted as equivalent. "
    "Logistic Regression is retained with its locked convergence-warning caveat."
)

atomic_text(CAPTION_TXT, caption + "\n")

highest = source_rows_sorted[0]

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_02_classical_baselines",
    "generated_at_utc": utc_now(),
    "title": "Classical Baseline Results",
    "row_count": len(table_rows),
    "descriptive_highest_model": {
        "rank": int(highest["descriptive_macro_f1_rank"]),
        "model_id": highest["model_id"],
        "display_name": highest["display_name"],
        "mean_test_fingerprint_macro_f1": float(
            highest["mean_test_fingerprint_macro_f1"]
        ),
    },
    "ranking_is_descriptive_only": True,
    "ranking_is_not_model_selection": True,
    "source_files": [
        {
            "source_id": "phase4_model_summary",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
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
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(METADATA_JSON, metadata)

lock = {
    "status": "locked",
    "phase": 13,
    "asset_id": "table_02_classical_baselines",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "row_count": 4,
    "ranking_is_descriptive_only": True,
    "ranking_is_not_model_selection": True,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 02 - CLASSICAL BASELINES")
print("=" * 96)
print(f"Table rows                       : {len(table_rows)}")
print("Ranking interpretation           : descriptive_only")
print("Final model selection performed  : False")
print()
for row in source_rows_sorted:
    print(
        f"Rank {int(row['descriptive_macro_f1_rank'])} | "
        f"{row['display_name']:<24} | "
        f"Macro-F1={float(row['mean_test_fingerprint_macro_f1']):.9f} | "
        f"runs={row['run_count']}"
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
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 TABLE 02 GENERATED AND LOCKED")
