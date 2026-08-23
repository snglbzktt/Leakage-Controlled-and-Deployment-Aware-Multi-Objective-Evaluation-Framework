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

TABLE_05_LOCK = AUDIT / "phase13_table_05_statistical_analysis_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_06_robustness_results_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_06_robustness_results_v3_3.md"
SOURCE_CSV = SOURCE_DIR / "table_06_robustness_results_source_v3_3.csv"
CAPTION_TXT = CAPTION_DIR / "table_06_robustness_results_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_06_robustness_results_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_06_robustness_results_locked_v3_3.json"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

MODEL_ORDER = [
    "tinyml_mlp_B0",
    "tinyml_mlp_P50_QAT",
    "hist_gradient_boosting_B0",
]

MODEL_LABELS = {
    "tinyml_mlp_B0": "TinyML-MLP B0",
    "tinyml_mlp_P50_QAT": "TinyML-MLP P50-QAT",
    "hist_gradient_boosting_B0": "HGB B0",
}

CONDITION_ORDER = [
    "clean",
    "gaussian_noise_1pct",
    "gaussian_noise_5pct",
    "gaussian_noise_10pct",
    "feature_mask_5pct",
    "feature_mask_10pct",
    "feature_mask_20pct",
    "scale_drift_minus_10pct",
    "scale_drift_plus_10pct",
]

CONDITION_LABELS = {
    "clean": "Clean",
    "gaussian_noise_1pct": "Gaussian noise 1%",
    "gaussian_noise_5pct": "Gaussian noise 5%",
    "gaussian_noise_10pct": "Gaussian noise 10%",
    "feature_mask_5pct": "Feature mask 5%",
    "feature_mask_10pct": "Feature mask 10%",
    "feature_mask_20pct": "Feature mask 20%",
    "scale_drift_minus_10pct": "Scale drift -10%",
    "scale_drift_plus_10pct": "Scale drift +10%",
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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


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
    TABLE_05_LOCK,
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
            "Table 06 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_05_lock = read_json(TABLE_05_LOCK)

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
    "table_05_locked": (
        table_05_lock.get("status") == "locked"
        and table_05_lock.get("ready_for_manuscript") is True
        and table_05_lock.get("all_checks_passed") is True
    ),
    "table_06_mapped": any(
        asset["asset_id"] == "table_06_robustness_results"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name for name, passed in entry_checks.items() if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 06 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

all_runs_path = get_bound_source(binding, "phase11_all_runs")
summary_path = get_bound_source(binding, "phase11_summary")

summary = read_json(summary_path)

source_rows: list[dict[str, Any]] = []

with all_runs_path.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    reader = csv.DictReader(handle)

    for raw in reader:
        source_rows.append(
            {
                "run_id": raw["run_id"],
                "model_id": raw["model_id"],
                "condition_id": raw["condition_id"],
                "condition_family": raw["condition_family"],
                "severity": raw["severity"],
                "test_rows": int(raw["test_rows"]),
                "represented_raw_rows": int(raw["represented_raw_rows"]),
                "fingerprint_macro_f1": float(raw["fingerprint_macro_f1"]),
                "raw_weighted_macro_f1": float(raw["raw_weighted_macro_f1"]),
                "macro_f1_drop_from_model_clean": float(
                    raw["macro_f1_drop_from_model_clean"]
                ),
                "macro_f1_retention_ratio": float(
                    raw["macro_f1_retention_ratio"]
                ),
                "severe_fragility": parse_bool(raw["severe_fragility"]),
                "material_sensitivity": parse_bool(
                    raw["material_sensitivity"]
                ),
                "model_retraining": parse_bool(raw["model_retraining"]),
                "test_used_for_model_selection": parse_bool(
                    raw["test_used_for_model_selection"]
                ),
                "final_model_changed": parse_bool(raw["final_model_changed"]),
            }
        )

observed_models = {row["model_id"] for row in source_rows}
observed_conditions = {row["condition_id"] for row in source_rows}

data_checks = {
    "source_rows_27": len(source_rows) == 27,
    "model_set_exact": observed_models == set(MODEL_ORDER),
    "condition_set_exact": observed_conditions == set(CONDITION_ORDER),
    "unique_model_condition_cells": (
        len(
            {
                (row["model_id"], row["condition_id"])
                for row in source_rows
            }
        )
        == 27
    ),
    "summary_evaluation_count_27": summary["evaluation_count"] == 27,
    "summary_severe_count_5": summary["severe_fragility_evaluation_count"] == 5,
    "summary_material_count_9": (
        summary["material_sensitivity_evaluation_count"] == 9
    ),
    "all_no_retraining": all(
        row["model_retraining"] is False
        for row in source_rows
    ),
    "all_test_not_selection": all(
        row["test_used_for_model_selection"] is False
        for row in source_rows
    ),
    "all_final_model_unchanged": all(
        row["final_model_changed"] is False
        for row in source_rows
    ),
    "summary_no_retraining": summary["model_retraining_performed"] is False,
    "summary_test_not_selection": (
        summary["test_used_for_model_selection"] is False
    ),
    "summary_final_model_unchanged": summary["final_model_changed"] is False,
}

failed_data_checks = [
    name for name, passed in data_checks.items() if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 06 source validation failed: "
        + ", ".join(failed_data_checks)
    )

lookup = {
    (row["model_id"], row["condition_id"]): row
    for row in source_rows
}

summary_consistency_checks = {
    "tinyml_b0_clean_matches": abs(
        lookup[("tinyml_mlp_B0", "clean")]["fingerprint_macro_f1"]
        - summary["model_summaries"]["tinyml_mlp_B0"]["clean_fingerprint_macro_f1"]
    ) <= 1e-12,
    "p50_qat_clean_matches": abs(
        lookup[("tinyml_mlp_P50_QAT", "clean")]["fingerprint_macro_f1"]
        - summary["model_summaries"]["tinyml_mlp_P50_QAT"]["clean_fingerprint_macro_f1"]
    ) <= 1e-12,
    "hgb_clean_matches": abs(
        lookup[("hist_gradient_boosting_B0", "clean")]["fingerprint_macro_f1"]
        - summary["model_summaries"]["hist_gradient_boosting_B0"]["clean_fingerprint_macro_f1"]
    ) <= 1e-12,
    "tinyml_b0_worst_condition": (
        summary["model_summaries"]["tinyml_mlp_B0"]["worst_condition"]
        == "feature_mask_20pct"
    ),
    "p50_qat_worst_condition": (
        summary["model_summaries"]["tinyml_mlp_P50_QAT"]["worst_condition"]
        == "feature_mask_20pct"
    ),
    "hgb_worst_condition": (
        summary["model_summaries"]["hist_gradient_boosting_B0"]["worst_condition"]
        == "scale_drift_plus_10pct"
    ),
}

failed_summary_checks = [
    name
    for name, passed in summary_consistency_checks.items()
    if not passed
]

if failed_summary_checks:
    raise RuntimeError(
        "Table 06 summary consistency failed: "
        + ", ".join(failed_summary_checks)
    )

table_rows: list[dict[str, str]] = []

for condition_id in CONDITION_ORDER:
    condition_rows = [
        lookup[(model_id, condition_id)]
        for model_id in MODEL_ORDER
    ]

    table_rows.append(
        {
            "Condition": CONDITION_LABELS[condition_id],
            "TinyML-MLP B0 Macro-F1": (
                f"{lookup[('tinyml_mlp_B0', condition_id)]['fingerprint_macro_f1']:.6f}"
            ),
            "TinyML-MLP B0 ΔF1": (
                f"{lookup[('tinyml_mlp_B0', condition_id)]['macro_f1_drop_from_model_clean']:.6f}"
            ),
            "P50-QAT Macro-F1": (
                f"{lookup[('tinyml_mlp_P50_QAT', condition_id)]['fingerprint_macro_f1']:.6f}"
            ),
            "P50-QAT ΔF1": (
                f"{lookup[('tinyml_mlp_P50_QAT', condition_id)]['macro_f1_drop_from_model_clean']:.6f}"
            ),
            "HGB B0 Macro-F1": (
                f"{lookup[('hist_gradient_boosting_B0', condition_id)]['fingerprint_macro_f1']:.6f}"
            ),
            "HGB B0 ΔF1": (
                f"{lookup[('hist_gradient_boosting_B0', condition_id)]['macro_f1_drop_from_model_clean']:.6f}"
            ),
            "Severe evaluations": str(
                sum(row["severe_fragility"] for row in condition_rows)
            ),
            "Material-sensitivity evaluations": str(
                sum(row["material_sensitivity"] for row in condition_rows)
            ),
        }
    )

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

table_fieldnames = [
    "Condition",
    "TinyML-MLP B0 Macro-F1",
    "TinyML-MLP B0 ΔF1",
    "P50-QAT Macro-F1",
    "P50-QAT ΔF1",
    "HGB B0 Macro-F1",
    "HGB B0 ΔF1",
    "Severe evaluations",
    "Material-sensitivity evaluations",
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

source_fieldnames = list(source_rows[0].keys())

with SOURCE_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=source_fieldnames,
    )
    writer.writeheader()

    for model_id in MODEL_ORDER:
        for condition_id in CONDITION_ORDER:
            writer.writerow(
                lookup[(model_id, condition_id)]
            )

md_lines = [
    "| Condition | TinyML-MLP B0 Macro-F1 | TinyML-MLP B0 ΔF1 | P50-QAT Macro-F1 | P50-QAT ΔF1 | HGB B0 Macro-F1 | HGB B0 ΔF1 | Severe evaluations | Material-sensitivity evaluations |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
    "Table 6. Locked input-robustness results for TinyML-MLP B0, the selected "
    "TinyML-MLP P50-QAT family, and HGB B0 across nine perturbation conditions. "
    "Macro-F1 and the absolute drop from each model's clean baseline are reported. "
    "Five severe-fragility evaluations were observed, all for HGB, and nine "
    "material-sensitivity evaluations were recorded overall. No model retraining "
    "was performed, test results were not used for model selection, and the final "
    "model was not changed."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_06_robustness_results",
    "generated_at_utc": utc_now(),
    "title": "Input Robustness Results",
    "table_row_count": len(table_rows),
    "evaluation_count": len(source_rows),
    "model_count": len(MODEL_ORDER),
    "condition_count": len(CONDITION_ORDER),
    "severe_fragility_evaluation_count": (
        summary["severe_fragility_evaluation_count"]
    ),
    "material_sensitivity_evaluation_count": (
        summary["material_sensitivity_evaluation_count"]
    ),
    "model_summaries": summary["model_summaries"],
    "model_retraining_performed": False,
    "test_used_for_model_selection": False,
    "final_model_changed": False,
    "source_files": [
        {
            "source_id": "phase11_all_runs",
            "path": str(all_runs_path),
            "sha256": sha256_file(all_runs_path),
        },
        {
            "source_id": "phase11_summary",
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
    "summary_consistency_checks": summary_consistency_checks,
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
    "asset_id": "table_06_robustness_results",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "table_row_count": 9,
    "evaluation_count": 27,
    "model_count": 3,
    "condition_count": 9,
    "severe_fragility_evaluation_count": 5,
    "material_sensitivity_evaluation_count": 9,
    "model_retraining_performed": False,
    "test_used_for_model_selection": False,
    "final_model_changed": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript": True,
    "all_checks_passed": True,
}

atomic_json(LOCK_JSON, lock)

print("=" * 96)
print("PHASE 13 TABLE 06 - ROBUSTNESS RESULTS")
print("=" * 96)
print(f"Table rows                       : {len(table_rows)}")
print(f"Locked evaluations               : {len(source_rows)}")
print("Models                           : 3")
print("Conditions                       : 9")
print(
    "Severe fragility evaluations    : "
    f"{summary['severe_fragility_evaluation_count']}"
)
print(
    "Material sensitivity evaluations: "
    f"{summary['material_sensitivity_evaluation_count']}"
)
print()
for model_id in MODEL_ORDER:
    model_summary = summary["model_summaries"][model_id]
    print(
        f"{MODEL_LABELS[model_id]:<22} | "
        f"clean={model_summary['clean_fingerprint_macro_f1']:.9f} | "
        f"worst={model_summary['worst_condition']} | "
        f"worst_F1={model_summary['worst_fingerprint_macro_f1']:.9f}"
    )
print()
print(f"CSV                              : {TABLE_CSV}")
print(f"Markdown                         : {TABLE_MD}")
print(f"Source CSV                       : {SOURCE_CSV}")
print(f"Caption                          : {CAPTION_TXT}")
print(f"Metadata                         : {METADATA_JSON}")
print(f"Lock                             : {LOCK_JSON}")
print()
print("Model retraining performed       : False")
print("Test used for model selection    : False")
print("Final model changed              : False")
print("Model inference performed        : False")
print("Test arrays accessed              : False")
print("Source files mutated              : False")
print("Ready for manuscript              : True")
print("All checks passed                 : True")
print("PHASE 13 TABLE 06 GENERATED AND LOCKED")
