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

TABLE_02_LOCK = AUDIT / "phase13_table_02_classical_baselines_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_03_compression_results_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_03_compression_results_v3_3.md"
SOURCE_CSV = SOURCE_DIR / "table_03_compression_results_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "table_03_compression_results_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_03_compression_results_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_03_compression_results_locked_v3_3.json"

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

    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

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
    TABLE_02_LOCK,
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
            "Table 03 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_02_lock = read_json(TABLE_02_LOCK)

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
    "table_02_locked": (
        table_02_lock.get("status") == "locked"
        and table_02_lock.get("ready_for_manuscript") is True
        and table_02_lock.get("all_checks_passed") is True
    ),
    "table_03_mapped": any(
        asset["asset_id"] == "table_03_compression_results"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 03 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

group_path = get_bound_source(binding, "phase5_group_summary")
closure_path = get_bound_source(binding, "phase5_closure")

closure = read_json(closure_path)

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
    "mean_test_fingerprint_macro_f1",
    "std_test_fingerprint_macro_f1",
    "mean_test_raw_weighted_macro_f1",
    "mean_test_gafgyt_fnr",
    "mean_test_mirai_fnr",
    "mean_state_size_ratio_vs_float",
    "fragility_triggered_in_any_run",
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
    "closure_run_count_110": closure["final_run_count"] == 110,
    "closure_group_count_22": closure["group_count"] == 22,
    "closure_final_model_not_selected": closure["final_model_selected"] is False,
    "test_not_used_for_selection": (
        closure["final_checks"]["test_not_used_for_selection"] is True
    ),
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 03 source validation failed: "
        + ", ".join(failed_data_checks)
    )

architecture_order = {
    "tinyml_mlp": 0,
    "compact_dnn": 1,
}

variant_order = {
    variant: index
    for index, variant in enumerate(closure["variants"])
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
    macro_mean = float(row["mean_test_fingerprint_macro_f1"])
    macro_std = float(row["std_test_fingerprint_macro_f1"])
    weighted = float(row["mean_test_raw_weighted_macro_f1"])
    state_ratio = float(row["mean_state_size_ratio_vs_float"])
    gafgyt_fnr = float(row["mean_test_gafgyt_fnr"])
    mirai_fnr = float(row["mean_test_mirai_fnr"])
    fragile = parse_bool(row["fragility_triggered_in_any_run"])

    table_rows.append(
        {
            "Architecture": ARCHITECTURE_LABELS[row["architecture"]],
            "Variant": row["variant"],
            "Representation": row["representation"],
            "Runs": row["run_count"],
            "Macro-F1 mean ± std": f"{macro_mean:.6f} ± {macro_std:.6f}",
            "Raw-weighted Macro-F1": f"{weighted:.6f}",
            "State-size ratio vs B0": f"{state_ratio:.4f}",
            "Gafgyt FNR": f"{gafgyt_fnr:.6f}",
            "Mirai FNR": f"{mirai_fnr:.6f}",
            "Fragility in any run": "Yes" if fragile else "No",
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
    "Macro-F1 mean ± std",
    "Raw-weighted Macro-F1",
    "State-size ratio vs B0",
    "Gafgyt FNR",
    "Mirai FNR",
    "Fragility in any run",
]

with TABLE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=table_fieldnames,
    )
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
    "| Architecture | Variant | Representation | Runs | Macro-F1 mean ± std | Raw-weighted Macro-F1 | State-size ratio vs B0 | Gafgyt FNR | Mirai FNR | Fragility in any run |",
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

fragility_groups = [
    {
        "architecture": row["architecture"],
        "variant": row["variant"],
    }
    for row in source_rows_sorted
    if parse_bool(row["fragility_triggered_in_any_run"])
]

caption = (
    "Table 3. Five-seed compression results for the 22 locked architecture–variant "
    "groups. Macro-F1 and raw-weighted Macro-F1 are reported together with the "
    "serialized state-size ratio relative to each architecture's floating-point "
    "baseline and attack-specific false-negative rates. These results are descriptive "
    "at this stage; no final model selection was performed in Phase 5, and test "
    "results were not used for selection."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_03_compression_results",
    "generated_at_utc": utc_now(),
    "title": "Compression Results",
    "row_count": len(table_rows),
    "architecture_count": 2,
    "variant_count_per_architecture": 11,
    "seed_count": closure["seed_count"],
    "fragility_group_count": len(fragility_groups),
    "fragility_groups": fragility_groups,
    "final_model_selected_in_phase5": False,
    "test_used_for_selection": False,
    "source_files": [
        {
            "source_id": "phase5_group_summary",
            "path": str(group_path),
            "sha256": sha256_file(group_path),
        },
        {
            "source_id": "phase5_closure",
            "path": str(closure_path),
            "sha256": sha256_file(closure_path),
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
    "asset_id": "table_03_compression_results",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "row_count": 22,
    "architecture_count": 2,
    "variant_count_per_architecture": 11,
    "fragility_group_count": len(fragility_groups),
    "final_model_selected_in_phase5": False,
    "test_used_for_selection": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 03 - COMPRESSION RESULTS")
print("=" * 96)
print(f"Table rows                       : {len(table_rows)}")
print("Architectures                    : 2")
print("Variants per architecture        : 11")
print(f"Locked seeds                     : {closure['seed_count']}")
print(f"Fragility groups                 : {len(fragility_groups)}")
print("Final model selected in Phase 5 : False")
print("Test used for selection          : False")
print()
for group in fragility_groups:
    print(
        "Fragility group                  : "
        f"{group['architecture']}::{group['variant']}"
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
print("PHASE 13 TABLE 03 GENERATED AND LOCKED")
