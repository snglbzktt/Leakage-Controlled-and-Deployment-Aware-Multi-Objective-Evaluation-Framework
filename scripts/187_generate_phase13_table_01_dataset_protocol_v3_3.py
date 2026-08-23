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

FIGURE_01_LOCK = AUDIT / "phase13_figure_01_methodology_overview_locked_v3_3.json"
FIGURE_02_LOCK = AUDIT / "phase13_figure_02_lodo_generalization_locked_v3_3.json"
FIGURE_03_LOCK = AUDIT / "phase13_figure_03_compression_tradeoff_locked_v3_3.json"
FIGURE_04_LOCK = AUDIT / "phase13_figure_04_pareto_selection_locked_v3_3.json"
FIGURE_05_LOCK = AUDIT / "phase13_figure_05_input_robustness_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_01_dataset_protocol_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_01_dataset_protocol_v3_3.md"
SOURCE_JSON = SOURCE_DIR / "table_01_dataset_protocol_source_v3_3.json"
CAPTION_TXT = CAPTION_DIR / "table_01_dataset_protocol_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_01_dataset_protocol_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_01_dataset_protocol_locked_v3_3.json"

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
    FIGURE_01_LOCK,
    FIGURE_02_LOCK,
    FIGURE_03_LOCK,
    FIGURE_04_LOCK,
    FIGURE_05_LOCK,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for output_path in (
    TABLE_CSV,
    TABLE_MD,
    SOURCE_JSON,
    CAPTION_TXT,
    METADATA_JSON,
    LOCK_JSON,
):
    if output_path.exists():
        raise FileExistsError(
            "Table 01 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)

figure_locks = [
    read_json(FIGURE_01_LOCK),
    read_json(FIGURE_02_LOCK),
    read_json(FIGURE_03_LOCK),
    read_json(FIGURE_04_LOCK),
    read_json(FIGURE_05_LOCK),
]

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
    "all_five_figures_locked": all(
        record.get("status") == "locked"
        and record.get("ready_for_manuscript") is True
        and record.get("all_checks_passed") is True
        for record in figure_locks
    ),
    "table_01_mapped": any(
        asset["asset_id"] == "table_01_dataset_protocol"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 01 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

lodo_path = get_bound_source(binding, "lodo_global_summary")
phase4_path = get_bound_source(binding, "phase4_summary")
phase5_path = get_bound_source(binding, "phase5_closure")

lodo = read_json(lodo_path)
phase4 = read_json(phase4_path)
phase5 = read_json(phase5_path)

data_checks = {
    "lodo_devices_9": lodo["scope"]["device_count"] == 9,
    "lodo_models_3": lodo["scope"]["model_count"] == 3,
    "lodo_runs_27": lodo["scope"]["scientific_run_count"] == 27,
    "lodo_two_class_devices_2": len(lodo["scope"]["two_class_test_devices"]) == 2,
    "lodo_severe_macro_f1_085": (
        abs(lodo["locked_thresholds"]["severe_macro_f1_below"] - 0.85) <= 1e-12
    ),
    "lodo_severe_attack_fnr_040": (
        abs(lodo["locked_thresholds"]["severe_attack_fnr_above"] - 0.40) <= 1e-12
    ),
    "lodo_validation_drop_010": (
        abs(
            lodo["locked_thresholds"][
                "meaningful_validation_test_macro_f1_drop_at_least"
            ]
            - 0.10
        )
        <= 1e-12
    ),
    "phase4_models_4": phase4["model_count"] == 4,
    "phase4_runs_12": phase4["total_run_count"] == 12,
    "phase4_descriptive_only": (
        phase4["interpretation_policy"]["ranking_is_descriptive_only"] is True
        and phase4["interpretation_policy"]["no_final_model_selection"] is True
    ),
    "phase5_architectures_2": phase5["architecture_count"] == 2,
    "phase5_variants_11": phase5["variant_count"] == 11,
    "phase5_seeds_5": phase5["seed_count"] == 5,
    "phase5_runs_110": phase5["final_run_count"] == 110,
    "phase5_groups_22": phase5["group_count"] == 22,
    "phase5_test_not_selection": (
        phase5["final_checks"]["test_not_used_for_selection"] is True
    ),
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 01 source validation failed: "
        + ", ".join(failed_data_checks)
    )

two_class_devices = ", ".join(
    lodo["scope"]["two_class_test_devices"]
)

rows = [
    {
        "Section": "Strict LODO",
        "Protocol item": "Held-out devices",
        "Locked value": str(lodo["scope"]["device_count"]),
        "Interpretation / note": "Nine device-level hold-outs were completed.",
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Model families per device",
        "Locked value": str(lodo["scope"]["model_count"]),
        "Interpretation / note": "Each held-out device was evaluated with three model families.",
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Scientific LODO runs",
        "Locked value": str(lodo["scope"]["scientific_run_count"]),
        "Interpretation / note": "Complete 9 × 3 device–model matrix.",
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Two-class held-out test devices",
        "Locked value": "2",
        "Interpretation / note": two_class_devices,
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Severe Macro-F1 threshold",
        "Locked value": "< 0.85",
        "Interpretation / note": "Triggers the locked severe-fragility rule.",
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Severe attack FNR threshold",
        "Locked value": "> 0.40",
        "Interpretation / note": "Applied to eligible attack classes.",
    },
    {
        "Section": "Strict LODO",
        "Protocol item": "Meaningful validation–test Macro-F1 drop",
        "Locked value": "≥ 0.10",
        "Interpretation / note": "Locked reporting threshold.",
    },
    {
        "Section": "Classical baselines",
        "Protocol item": "Baseline models",
        "Locked value": str(phase4["model_count"]),
        "Interpretation / note": "Classical tabular baseline comparison.",
    },
    {
        "Section": "Classical baselines",
        "Protocol item": "Baseline runs",
        "Locked value": str(phase4["total_run_count"]),
        "Interpretation / note": "Descriptive comparison only; not final model selection.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Architectures",
        "Locked value": str(phase5["architecture_count"]),
        "Interpretation / note": "TinyML-MLP and Compact-DNN experiment families.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Variants per architecture",
        "Locked value": str(phase5["variant_count"]),
        "Interpretation / note": "Eleven locked compression/quantization variants.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Locked seeds",
        "Locked value": str(phase5["seed_count"]),
        "Interpretation / note": "Seeds: 42, 123, 2026, 3407, 8192.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Final run count",
        "Locked value": str(phase5["final_run_count"]),
        "Interpretation / note": "2 architectures × 11 variants × 5 seeds.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Grouped configurations",
        "Locked value": str(phase5["group_count"]),
        "Interpretation / note": "Twenty-two architecture–variant groups.",
    },
    {
        "Section": "Compression study",
        "Protocol item": "Test used for selection",
        "Locked value": "No",
        "Interpretation / note": "Locked closure confirms test results were not used for selection.",
    },
]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "Section",
    "Protocol item",
    "Locked value",
    "Interpretation / note",
]

with TABLE_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_lines = [
    "| Section | Protocol item | Locked value | Interpretation / note |",
    "|---|---|---:|---|",
]

for row in rows:
    md_lines.append(
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|")
            for field in fieldnames
        )
        + " |"
    )

atomic_text(TABLE_MD, "\n".join(md_lines) + "\n")

source_export = {
    "lodo_global_summary": {
        "path": str(lodo_path),
        "sha256": sha256_file(lodo_path),
        "scope": lodo["scope"],
        "locked_thresholds": lodo["locked_thresholds"],
        "interpretation": {
            "class_coverage_note": lodo["interpretation"]["class_coverage_note"],
        },
    },
    "phase4_summary": {
        "path": str(phase4_path),
        "sha256": sha256_file(phase4_path),
        "model_count": phase4["model_count"],
        "total_run_count": phase4["total_run_count"],
        "interpretation_policy": phase4["interpretation_policy"],
        "scientific_caveats": phase4["scientific_caveats"],
    },
    "phase5_closure": {
        "path": str(phase5_path),
        "sha256": sha256_file(phase5_path),
        "architecture_count": phase5["architecture_count"],
        "variant_count": phase5["variant_count"],
        "seed_count": phase5["seed_count"],
        "final_run_count": phase5["final_run_count"],
        "group_count": phase5["group_count"],
        "test_evaluation_count_total": phase5["test_evaluation_count_total"],
        "final_checks": phase5["final_checks"],
    },
}

atomic_json(SOURCE_JSON, source_export)

caption = (
    "Table 1. Locked primary-study protocol summary covering strict leave-one-device-out "
    "generalization, classical baseline evaluation, and the fair-budget compression matrix. "
    "The table reports only values bound to verified Phase 3–5 evidence. The two-class "
    "held-out devices are retained with the locked class-coverage note, and test results "
    "were not used for compression-model selection."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_01_dataset_protocol",
    "generated_at_utc": utc_now(),
    "title": "Primary Study Dataset and Evaluation Protocol",
    "row_count": len(rows),
    "source_files": [
        {
            "source_id": "lodo_global_summary",
            "path": str(lodo_path),
            "sha256": sha256_file(lodo_path),
        },
        {
            "source_id": "phase4_summary",
            "path": str(phase4_path),
            "sha256": sha256_file(phase4_path),
        },
        {
            "source_id": "phase5_closure",
            "path": str(phase5_path),
            "sha256": sha256_file(phase5_path),
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
        "source_json": {
            "path": str(SOURCE_JSON),
            "sha256": sha256_file(SOURCE_JSON),
            "size_bytes": SOURCE_JSON.stat().st_size,
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
    "asset_id": "table_01_dataset_protocol",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "row_count": len(rows),
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 01 - DATASET / PROTOCOL")
print("=" * 96)
print(f"Table rows                       : {len(rows)}")
print(f"Strict LODO devices              : {lodo['scope']['device_count']}")
print(f"Strict LODO scientific runs      : {lodo['scope']['scientific_run_count']}")
print(f"Classical baseline models        : {phase4['model_count']}")
print(f"Classical baseline runs          : {phase4['total_run_count']}")
print(f"Compression architectures        : {phase5['architecture_count']}")
print(f"Compression variants             : {phase5['variant_count']}")
print(f"Compression seeds                : {phase5['seed_count']}")
print(f"Compression final runs           : {phase5['final_run_count']}")
print()
print(f"CSV                              : {TABLE_CSV}")
print(f"Markdown                         : {TABLE_MD}")
print(f"Source JSON                      : {SOURCE_JSON}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model inference performed        : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 TABLE 01 GENERATED AND LOCKED")
