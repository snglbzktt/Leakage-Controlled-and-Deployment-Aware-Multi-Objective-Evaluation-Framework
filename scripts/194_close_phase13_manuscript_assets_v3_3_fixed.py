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

BINDING_PATH = CONFIG / "phase13_exact_source_binding_v3_3.json"
BINDING_LOCK_PATH = AUDIT / "phase13_exact_source_binding_locked_v3_3.json"
MAPPING_PATH = CONFIG / "phase13_manuscript_column_mapping_v3_3.json"
MAPPING_LOCK_PATH = AUDIT / "phase13_manuscript_column_mapping_locked_v3_3.json"

MANIFEST_JSON = AUDIT / "phase13_manuscript_assets_manifest_v3_3.json"
MANIFEST_CSV = AUDIT / "phase13_manuscript_assets_manifest_v3_3.csv"
CLOSURE_JSON = AUDIT / "phase13_manuscript_assets_closure_v3_3.json"
CLOSURE_TXT = AUDIT / "phase13_manuscript_assets_closure_v3_3.txt"

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25

ASSETS = [
    {
        "order": 1,
        "asset_id": "figure_01_methodology_overview",
        "asset_type": "figure",
        "lock": AUDIT / "phase13_figure_01_methodology_overview_locked_v3_3.json",
        "required_output_keys": ["png", "svg", "source_csv", "caption"],
    },
    {
        "order": 2,
        "asset_id": "figure_02_lodo_generalization",
        "asset_type": "figure",
        "lock": AUDIT / "phase13_figure_02_lodo_generalization_locked_v3_3.json",
        "required_output_keys": ["png", "svg", "source_csv", "caption"],
    },
    {
        "order": 3,
        "asset_id": "figure_03_compression_tradeoff",
        "asset_type": "figure",
        "lock": AUDIT / "phase13_figure_03_compression_tradeoff_locked_v3_3.json",
        "required_output_keys": ["png", "svg", "source_csv", "caption"],
    },
    {
        "order": 4,
        "asset_id": "figure_04_pareto_selection",
        "asset_type": "figure",
        "lock": AUDIT / "phase13_figure_04_pareto_selection_locked_v3_3.json",
        "required_output_keys": ["png", "svg", "source_csv", "caption"],
    },
    {
        "order": 5,
        "asset_id": "figure_05_input_robustness",
        "asset_type": "figure",
        "lock": AUDIT / "phase13_figure_05_input_robustness_locked_v3_3.json",
        "required_output_keys": ["png", "svg", "source_csv", "caption"],
    },
    {
        "order": 6,
        "asset_id": "table_01_dataset_protocol",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_01_dataset_protocol_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_json", "caption"],
    },
    {
        "order": 7,
        "asset_id": "table_02_classical_baselines",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_02_classical_baselines_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_csv", "caption"],
    },
    {
        "order": 8,
        "asset_id": "table_03_compression_results",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_03_compression_results_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_csv", "caption"],
    },
    {
        "order": 9,
        "asset_id": "table_04_deployment_benchmarks",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_04_deployment_benchmarks_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_csv", "caption"],
    },
    {
        "order": 10,
        "asset_id": "table_05_statistical_analysis",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_05_statistical_analysis_locked_v3_3.json",
        "required_output_keys": [
            "csv",
            "markdown",
            "source_omnibus_csv",
            "source_posthoc_csv",
            "caption",
        ],
    },
    {
        "order": 11,
        "asset_id": "table_06_robustness_results",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_06_robustness_results_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_csv", "caption"],
    },
    {
        "order": 12,
        "asset_id": "table_07_reproducibility_release",
        "asset_type": "table",
        "lock": AUDIT / "phase13_table_07_reproducibility_release_locked_v3_3.json",
        "required_output_keys": ["csv", "markdown", "source_json", "caption"],
    },
]


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
        )
        + "\n",
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


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    MAPPING_PATH,
    MAPPING_LOCK_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

for asset in ASSETS:
    if not asset["lock"].exists():
        raise FileNotFoundError(asset["lock"])

for output_path in (
    MANIFEST_JSON,
    MANIFEST_CSV,
    CLOSURE_JSON,
    CLOSURE_TXT,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 manuscript closure output already exists; "
            f"refusing to overwrite: {output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)

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
    "mapping_asset_count_12": mapping["asset_count"] == 12,
    "declared_asset_count_12": len(ASSETS) == 12,
}

failed_entry_checks = [
    name
    for name, passed in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 13 closure entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

manifest_assets: list[dict[str, Any]] = []
manifest_rows: list[dict[str, Any]] = []
validation_errors: list[dict[str, Any]] = []

total_output_files = 0
total_output_bytes = 0

for asset in ASSETS:
    lock_path = asset["lock"]
    lock = read_json(lock_path)

    asset_checks = {
        "asset_id_matches": lock.get("asset_id") == asset["asset_id"],
        "status_locked": lock.get("status") == "locked",
        "ready_for_manuscript": lock.get("ready_for_manuscript") is True,
        "all_checks_passed": lock.get("all_checks_passed") is True,
        "model_inference_false": lock.get("model_inference_performed") is False,
        "source_mutation_false": lock.get("source_files_mutated") is False,
        "test_access_false": (
            (
                lock.get("test_arrays_accessed") is False
                if "test_arrays_accessed" in lock
                else lock.get("test_data_accessed") is False
                if "test_data_accessed" in lock
                else False
            )
        ),
        "outputs_present": isinstance(lock.get("outputs"), dict),
    }

    failed_asset_checks = [
        name
        for name, passed in asset_checks.items()
        if not passed
    ]

    if failed_asset_checks:
        validation_errors.append(
            {
                "asset_id": asset["asset_id"],
                "stage": "lock_checks",
                "failed_checks": failed_asset_checks,
            }
        )
        continue

    outputs = lock["outputs"]

    missing_output_keys = [
        key
        for key in asset["required_output_keys"]
        if key not in outputs
    ]

    if missing_output_keys:
        validation_errors.append(
            {
                "asset_id": asset["asset_id"],
                "stage": "output_keys",
                "missing_output_keys": missing_output_keys,
            }
        )
        continue

    asset_output_records = []

    for output_key in asset["required_output_keys"]:
        output_record = outputs[output_key]

        output_path = Path(output_record["path"])

        if not output_path.exists():
            validation_errors.append(
                {
                    "asset_id": asset["asset_id"],
                    "stage": "output_file",
                    "output_key": output_key,
                    "error": "missing",
                    "path": str(output_path),
                }
            )
            continue

        current_hash = sha256_file(output_path)
        locked_hash = output_record["sha256"]

        if current_hash != locked_hash:
            validation_errors.append(
                {
                    "asset_id": asset["asset_id"],
                    "stage": "output_hash",
                    "output_key": output_key,
                    "error": "hash_mismatch",
                    "path": str(output_path),
                    "locked_sha256": locked_hash,
                    "current_sha256": current_hash,
                }
            )
            continue

        current_size = output_path.stat().st_size
        locked_size = output_record.get("size_bytes")

        if locked_size is not None and current_size != locked_size:
            validation_errors.append(
                {
                    "asset_id": asset["asset_id"],
                    "stage": "output_size",
                    "output_key": output_key,
                    "error": "size_mismatch",
                    "path": str(output_path),
                    "locked_size_bytes": locked_size,
                    "current_size_bytes": current_size,
                }
            )
            continue

        relative_path = str(
            output_path.relative_to(ROOT)
        ).replace("\\", "/")

        asset_output_records.append(
            {
                "output_key": output_key,
                "relative_path": relative_path,
                "sha256": current_hash,
                "size_bytes": current_size,
            }
        )

        manifest_rows.append(
            {
                "asset_order": asset["order"],
                "asset_id": asset["asset_id"],
                "asset_type": asset["asset_type"],
                "output_key": output_key,
                "relative_path": relative_path,
                "sha256": current_hash,
                "size_bytes": current_size,
                "lock_relative_path": str(
                    lock_path.relative_to(ROOT)
                ).replace("\\", "/"),
                "lock_sha256": sha256_file(lock_path),
                "ready_for_manuscript": True,
                "all_checks_passed": True,
            }
        )

        total_output_files += 1
        total_output_bytes += current_size

    manifest_assets.append(
        {
            "order": asset["order"],
            "asset_id": asset["asset_id"],
            "asset_type": asset["asset_type"],
            "lock": {
                "path": str(lock_path),
                "sha256": sha256_file(lock_path),
            },
            "outputs": asset_output_records,
            "output_count": len(asset_output_records),
            "status": "verified",
            "ready_for_manuscript": True,
            "all_checks_passed": True,
        }
    )

if validation_errors:
    raise RuntimeError(
        "Phase 13 closure validation failed:\n"
        + json.dumps(
            validation_errors,
            indent=2,
            ensure_ascii=True,
        )
    )

figure_assets = [
    asset
    for asset in manifest_assets
    if asset["asset_type"] == "figure"
]

table_assets = [
    asset
    for asset in manifest_assets
    if asset["asset_type"] == "table"
]

closure_checks = {
    "all_12_assets_verified": len(manifest_assets) == 12,
    "five_figures_verified": len(figure_assets) == 5,
    "seven_tables_verified": len(table_assets) == 7,
    "all_declared_output_files_verified": (
        total_output_files
        == sum(
            len(asset["required_output_keys"])
            for asset in ASSETS
        )
    ),
    "all_assets_ready_for_manuscript": all(
        asset["ready_for_manuscript"] is True
        for asset in manifest_assets
    ),
    "all_asset_checks_passed": all(
        asset["all_checks_passed"] is True
        for asset in manifest_assets
    ),
    "model_inference_not_performed_by_closure": True,
    "test_arrays_not_accessed_by_closure": True,
    "source_files_not_mutated_by_closure": True,
    "phase12_external_dataset_deferred_optional": True,
}

failed_closure_checks = [
    name
    for name, passed in closure_checks.items()
    if not passed
]

if failed_closure_checks:
    raise RuntimeError(
        "Phase 13 final closure checks failed: "
        + ", ".join(failed_closure_checks)
    )

manifest = {
    "status": "verified",
    "phase": 13,
    "artifact_name": "manuscript_assets_manifest",
    "protocol_version": "phase13_manuscript_assets_v3_3",
    "verified_at_utc": utc_now(),
    "asset_root": str(ASSET_ROOT),
    "asset_count": len(manifest_assets),
    "figure_count": len(figure_assets),
    "table_count": len(table_assets),
    "verified_output_file_count": total_output_files,
    "verified_output_total_bytes": total_output_bytes,
    "assets": manifest_assets,
    "source_binding": {
        "path": str(BINDING_PATH),
        "sha256": sha256_file(BINDING_PATH),
    },
    "column_mapping": {
        "path": str(MAPPING_PATH),
        "sha256": sha256_file(MAPPING_PATH),
    },
    "phase12_external_dataset": "deferred_optional",
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_manuscript_drafting": True,
    "all_checks_passed": True,
}

atomic_json(MANIFEST_JSON, manifest)

with MANIFEST_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(manifest_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

closure = {
    "status": "locked",
    "phase": 13,
    "artifact_name": "manuscript_assets_closure",
    "protocol_version": "phase13_manuscript_assets_v3_3",
    "locked_at_utc": utc_now(),
    "manifest": {
        "path": str(MANIFEST_JSON),
        "sha256": sha256_file(MANIFEST_JSON),
    },
    "manifest_csv": {
        "path": str(MANIFEST_CSV),
        "sha256": sha256_file(MANIFEST_CSV),
    },
    "asset_count": 12,
    "figure_count": 5,
    "table_count": 7,
    "verified_output_file_count": total_output_files,
    "verified_output_total_bytes": total_output_bytes,
    "phase12_external_dataset": "deferred_optional",
    "closure_checks": closure_checks,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "phase13_manuscript_assets_locked": True,
    "ready_for_manuscript_drafting": True,
    "all_checks_passed": True,
}

atomic_json(CLOSURE_JSON, closure)

lines = [
    "=" * 100,
    "PHASE 13 MANUSCRIPT ASSETS CLOSURE",
    "=" * 100,
    f"Assets verified                  : {len(manifest_assets)}",
    f"Figures verified                 : {len(figure_assets)}",
    f"Tables verified                  : {len(table_assets)}",
    f"Output files hash-verified       : {total_output_files}",
    f"Output bytes verified            : {total_output_bytes}",
    "Phase 12 external dataset        : deferred_optional",
    "",
]

for asset in manifest_assets:
    lines.append(
        f"{asset['order']:02d}. "
        f"{asset['asset_id']:<40} | "
        f"type={asset['asset_type']:<6} | "
        f"outputs={asset['output_count']} | "
        "status=verified"
    )

lines.extend(
    [
        "",
        "=" * 100,
        "FINAL CLOSURE CHECKS",
        "=" * 100,
        "All 12 assets verified          : True",
        "5/5 figures verified            : True",
        "7/7 tables verified             : True",
        "All output hashes match         : True",
        "All assets ready for manuscript : True",
        "Model inference performed       : False",
        "Test arrays accessed            : False",
        "Source files mutated            : False",
        "Ready for manuscript drafting   : True",
        "All checks passed               : True",
        "PHASE 13 MANUSCRIPT ASSETS LOCKED",
        "",
    ]
)

atomic_text(CLOSURE_TXT, "\n".join(lines))

print("\n".join(lines))
print(f"Manifest JSON                    : {MANIFEST_JSON}")
print(f"Manifest CSV                     : {MANIFEST_CSV}")
print(f"Closure JSON                     : {CLOSURE_JSON}")
print(f"Closure TXT                      : {CLOSURE_TXT}")
