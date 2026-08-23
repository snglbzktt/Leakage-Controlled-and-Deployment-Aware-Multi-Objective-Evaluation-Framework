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

TABLE_06_LOCK = AUDIT / "phase13_table_06_robustness_results_locked_v3_3.json"

TABLE_CSV = TABLE_DIR / "table_07_reproducibility_release_v3_3.csv"
TABLE_MD = TABLE_DIR / "table_07_reproducibility_release_v3_3.md"
SOURCE_JSON = SOURCE_DIR / "table_07_reproducibility_release_source_v3_3.json"
CAPTION_TXT = CAPTION_DIR / "table_07_reproducibility_release_caption_v3_3.txt"
METADATA_JSON = METADATA_DIR / "table_07_reproducibility_release_metadata_v3_3.json"
LOCK_JSON = AUDIT / "phase13_table_07_reproducibility_release_locked_v3_3.json"

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
    TABLE_06_LOCK,
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
            "Table 07 output already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(BINDING_PATH)
binding_lock = read_json(BINDING_LOCK_PATH)
mapping = read_json(MAPPING_PATH)
mapping_lock = read_json(MAPPING_LOCK_PATH)
table_06_lock = read_json(TABLE_06_LOCK)

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
    "table_06_locked": (
        table_06_lock.get("status") == "locked"
        and table_06_lock.get("ready_for_manuscript") is True
        and table_06_lock.get("all_checks_passed") is True
    ),
    "table_07_mapped": any(
        asset["asset_id"] == "table_07_reproducibility_release"
        and asset["mapping_status"] == "validated"
        for asset in mapping["assets"]
    ),
}

failed_entry_checks = [
    name
    for name, passed in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Table 07 generation entry gate failed: "
        + ", ".join(failed_entry_checks)
    )

phase8_path = get_bound_source(binding, "phase8_final_model_lock")
phase9_path = get_bound_source(binding, "phase9_artifact_lock")
phase10_path = get_bound_source(binding, "phase10_release_lock")
phase11_path = get_bound_source(binding, "phase11_lock")

phase8 = read_json(phase8_path)
phase9 = read_json(phase9_path)
phase10 = read_json(phase10_path)
phase11 = read_json(phase11_path)

data_checks = {
    "phase8_locked": phase8["status"] == "locked",
    "phase8_primary_balanced": phase8["primary_profile"] == "balanced",
    "phase8_selected_p50_qat": (
        phase8["selected_model_family"]["candidate_id"]
        == "tinyml_mlp::P50-QAT"
    ),
    "phase8_weighted_score_false": phase8["weighted_score_used"] is False,
    "phase8_pairwise_p_gate_false": (
        phase8["pairwise_p_value_used_as_gate"] is False
    ),
    "phase8_checkpoint_not_selected": (
        phase8["single_seed_checkpoint_selected"] is False
    ),
    "phase9_locked": phase9["final_deployment_artifact_locked"] is True,
    "phase9_seed_2026": phase9["canonical_seed"] == 2026,
    "phase9_static_int8": phase9["representation"] == "static_int8",
    "phase9_class_order_exact": (
        phase9["class_labels_in_output_index_order"]
        == ["benign", "gafgyt", "mirai"]
    ),
    "phase9_checkpoint_hash_exact": (
        phase9["source_checkpoint_sha256"]
        == phase9["bundled_checkpoint_sha256"]
    ),
    "phase9_source_bundle_outputs_exact": (
        phase9["source_bundle_outputs_exact"] is True
    ),
    "phase9_roundtrip_exact": (
        phase9["serialization_roundtrip_exact"] is True
    ),
    "phase9_no_val_access": phase9["validation_data_access"] is False,
    "phase9_no_test_access": phase9["test_data_access"] is False,
    "phase10_locked": phase10["release_locked"] is True,
    "phase10_payload_15": phase10["payload_file_count"] == 15,
    "phase10_archive_16": phase10["archive_file_count"] == 16,
    "phase10_deterministic_exact": (
        phase10["deterministic_rebuild_exact"] is True
    ),
    "phase10_raw_absent": phase10["raw_dataset_included"] is False,
    "phase10_processed_absent": (
        phase10["processed_dataset_included"] is False
    ),
    "phase10_nonselected_absent": (
        phase10["nonselected_checkpoints_included"] is False
    ),
    "phase11_locked": phase11["phase11_locked"] is True,
    "phase11_no_retraining": (
        phase11["model_retraining_performed"] is False
    ),
    "phase11_test_not_selection": (
        phase11["test_used_for_model_selection"] is False
    ),
    "phase11_final_model_unchanged": (
        phase11["final_model_changed"] is False
    ),
}

failed_data_checks = [
    name
    for name, passed in data_checks.items()
    if not passed
]

if failed_data_checks:
    raise RuntimeError(
        "Table 07 source validation failed: "
        + ", ".join(failed_data_checks)
    )

strict_selection = phase8["sensitivity_profile_selections"]["strict"]
balanced_selection = phase8["sensitivity_profile_selections"]["balanced"]
relaxed_selection = phase8["sensitivity_profile_selections"]["relaxed"]

rows = [
    {
        "Section": "Model-family selection",
        "Reproducibility item": "Primary selection profile",
        "Locked value": phase8["primary_profile"],
        "Integrity / interpretation": (
            "Predeclared balanced profile."
        ),
    },
    {
        "Section": "Model-family selection",
        "Reproducibility item": "Selected model family",
        "Locked value": phase8["selected_model_family"]["candidate_id"],
        "Integrity / interpretation": (
            f"{phase8['selected_model_family']['representation']} representation."
        ),
    },
    {
        "Section": "Model-family selection",
        "Reproducibility item": "Sensitivity-profile selections",
        "Locked value": (
            f"strict={strict_selection}; "
            f"balanced={balanced_selection}; "
            f"relaxed={relaxed_selection}"
        ),
        "Integrity / interpretation": (
            f"Stable across profiles: {phase8['selection_stable_across_profiles']}."
        ),
    },
    {
        "Section": "Model-family selection",
        "Reproducibility item": "Weighted score used",
        "Locked value": "No",
        "Integrity / interpretation": (
            "Selection used eligibility, Pareto filtering, and deployment ordering."
        ),
    },
    {
        "Section": "Model-family selection",
        "Reproducibility item": "Pairwise p-value used as gate",
        "Locked value": "No",
        "Integrity / interpretation": (
            "Inferential p-values were not used as a model-selection gate."
        ),
    },
    {
        "Section": "Final artifact",
        "Reproducibility item": "Canonical checkpoint seed",
        "Locked value": str(phase9["canonical_seed"]),
        "Integrity / interpretation": (
            "Concrete deployment checkpoint chosen after family lock."
        ),
    },
    {
        "Section": "Final artifact",
        "Reproducibility item": "Output class order",
        "Locked value": "benign, gafgyt, mirai",
        "Integrity / interpretation": (
            "Locked inference-contract output index order."
        ),
    },
    {
        "Section": "Final artifact",
        "Reproducibility item": "Checkpoint SHA-256",
        "Locked value": phase9["bundled_checkpoint_sha256"],
        "Integrity / interpretation": (
            "Source and bundled checkpoint hashes are identical."
        ),
    },
    {
        "Section": "Final artifact",
        "Reproducibility item": "Source/bundle output equality",
        "Locked value": "Exact",
        "Integrity / interpretation": (
            "Locked verification reports exact source/bundle outputs."
        ),
    },
    {
        "Section": "Final artifact",
        "Reproducibility item": "Serialization round-trip",
        "Locked value": "Exact",
        "Integrity / interpretation": (
            "Serialization/deserialization round-trip reproduced exactly."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "Release name",
        "Locked value": phase10["release_name"],
        "Integrity / interpretation": (
            f"{phase10['payload_file_count']} payload files; "
            f"{phase10['archive_file_count']} archive files including manifest."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "Release tree SHA-256",
        "Locked value": phase10["release_directory_tree_sha256"],
        "Integrity / interpretation": (
            "Hash of the deterministic release-directory tree."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "ZIP SHA-256",
        "Locked value": phase10["zip_archive_sha256"],
        "Integrity / interpretation": (
            f"ZIP size: {phase10['zip_archive_size_bytes']} bytes."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "Deterministic rebuild",
        "Locked value": "Exact",
        "Integrity / interpretation": (
            "Independent rebuild reproduced the locked archive exactly."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "Datasets in release",
        "Locked value": "Raw: No; Processed: No",
        "Integrity / interpretation": (
            "Release excludes raw and processed datasets."
        ),
    },
    {
        "Section": "Release archive",
        "Reproducibility item": "Nonselected checkpoints in release",
        "Locked value": "No",
        "Integrity / interpretation": (
            "Release contains no nonselected checkpoints."
        ),
    },
    {
        "Section": "Post-selection robustness",
        "Reproducibility item": "Model retraining performed",
        "Locked value": "No",
        "Integrity / interpretation": (
            "Robustness analysis did not retrain any model."
        ),
    },
    {
        "Section": "Post-selection robustness",
        "Reproducibility item": "Test used for model selection",
        "Locked value": "No",
        "Integrity / interpretation": (
            "Robustness test results were not fed back into selection."
        ),
    },
    {
        "Section": "Post-selection robustness",
        "Reproducibility item": "Final model changed",
        "Locked value": "No",
        "Integrity / interpretation": (
            "The locked deployment model remained unchanged."
        ),
    },
]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "Section",
    "Reproducibility item",
    "Locked value",
    "Integrity / interpretation",
]

with TABLE_CSV.open(
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

md_lines = [
    "| Section | Reproducibility item | Locked value | Integrity / interpretation |",
    "|---|---|---|---|",
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
    "phase8_final_model_lock": {
        "path": str(phase8_path),
        "sha256": sha256_file(phase8_path),
        "primary_profile": phase8["primary_profile"],
        "selected_model_family": phase8["selected_model_family"],
        "sensitivity_profile_selections": (
            phase8["sensitivity_profile_selections"]
        ),
        "selection_stable_across_profiles": (
            phase8["selection_stable_across_profiles"]
        ),
        "weighted_score_used": phase8["weighted_score_used"],
        "pairwise_p_value_used_as_gate": (
            phase8["pairwise_p_value_used_as_gate"]
        ),
        "single_seed_checkpoint_selected": (
            phase8["single_seed_checkpoint_selected"]
        ),
    },
    "phase9_artifact_lock": {
        "path": str(phase9_path),
        "sha256": sha256_file(phase9_path),
        "model_family": phase9["model_family"],
        "representation": phase9["representation"],
        "canonical_seed": phase9["canonical_seed"],
        "class_labels_in_output_index_order": (
            phase9["class_labels_in_output_index_order"]
        ),
        "source_checkpoint_sha256": phase9["source_checkpoint_sha256"],
        "bundled_checkpoint_sha256": phase9["bundled_checkpoint_sha256"],
        "source_bundle_checkpoint_exact": (
            phase9["source_bundle_checkpoint_exact"]
        ),
        "source_bundle_outputs_exact": phase9["source_bundle_outputs_exact"],
        "serialization_roundtrip_exact": (
            phase9["serialization_roundtrip_exact"]
        ),
        "validation_data_access": phase9["validation_data_access"],
        "test_data_access": phase9["test_data_access"],
    },
    "phase10_release_lock": {
        "path": str(phase10_path),
        "sha256": sha256_file(phase10_path),
        "release_name": phase10["release_name"],
        "payload_file_count": phase10["payload_file_count"],
        "archive_file_count": phase10["archive_file_count"],
        "release_directory_tree_sha256": (
            phase10["release_directory_tree_sha256"]
        ),
        "zip_archive_sha256": phase10["zip_archive_sha256"],
        "zip_archive_size_bytes": phase10["zip_archive_size_bytes"],
        "deterministic_rebuild_exact": phase10["deterministic_rebuild_exact"],
        "raw_dataset_included": phase10["raw_dataset_included"],
        "processed_dataset_included": phase10["processed_dataset_included"],
        "nonselected_checkpoints_included": (
            phase10["nonselected_checkpoints_included"]
        ),
    },
    "phase11_lock": {
        "path": str(phase11_path),
        "sha256": sha256_file(phase11_path),
        "model_retraining_performed": (
            phase11["model_retraining_performed"]
        ),
        "test_used_for_model_selection": (
            phase11["test_used_for_model_selection"]
        ),
        "final_model_changed": phase11["final_model_changed"],
        "phase11_locked": phase11["phase11_locked"],
    },
}

atomic_json(SOURCE_JSON, source_export)

caption = (
    "Table 7. Reproducibility and release-integrity record for the selected "
    "TinyML deployment pipeline. The table documents the locked model-family "
    "selection, canonical deployment checkpoint, exact source/bundle and "
    "serialization checks, deterministic release hashes, release-content "
    "exclusions, and the post-selection robustness constraints. Robustness "
    "evaluation did not retrain the model, reuse test results for selection, "
    "or modify the final deployment model."
)

atomic_text(CAPTION_TXT, caption + "\n")

metadata = {
    "status": "generated",
    "phase": 13,
    "asset_id": "table_07_reproducibility_release",
    "generated_at_utc": utc_now(),
    "title": "Reproducibility and Release Integrity",
    "row_count": len(rows),
    "selected_model_family": (
        phase8["selected_model_family"]["candidate_id"]
    ),
    "selected_representation": (
        phase8["selected_model_family"]["representation"]
    ),
    "canonical_seed": phase9["canonical_seed"],
    "checkpoint_sha256": phase9["bundled_checkpoint_sha256"],
    "release_tree_sha256": (
        phase10["release_directory_tree_sha256"]
    ),
    "zip_archive_sha256": phase10["zip_archive_sha256"],
    "deterministic_rebuild_exact": (
        phase10["deterministic_rebuild_exact"]
    ),
    "model_retraining_performed": False,
    "test_used_for_model_selection": False,
    "final_model_changed": False,
    "source_files": [
        {
            "source_id": "phase8_final_model_lock",
            "path": str(phase8_path),
            "sha256": sha256_file(phase8_path),
        },
        {
            "source_id": "phase9_artifact_lock",
            "path": str(phase9_path),
            "sha256": sha256_file(phase9_path),
        },
        {
            "source_id": "phase10_release_lock",
            "path": str(phase10_path),
            "sha256": sha256_file(phase10_path),
        },
        {
            "source_id": "phase11_lock",
            "path": str(phase11_path),
            "sha256": sha256_file(phase11_path),
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
    "asset_id": "table_07_reproducibility_release",
    "locked_at_utc": utc_now(),
    "metadata": {
        "path": str(METADATA_JSON),
        "sha256": sha256_file(METADATA_JSON),
    },
    "outputs": metadata["outputs"],
    "row_count": len(rows),
    "selected_model_family": (
        phase8["selected_model_family"]["candidate_id"]
    ),
    "canonical_seed": phase9["canonical_seed"],
    "checkpoint_sha256": phase9["bundled_checkpoint_sha256"],
    "release_tree_sha256": (
        phase10["release_directory_tree_sha256"]
    ),
    "zip_archive_sha256": phase10["zip_archive_sha256"],
    "deterministic_rebuild_exact": True,
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
print("PHASE 13 TABLE 07 - REPRODUCIBILITY / RELEASE")
print("=" * 96)
print(f"Table rows                       : {len(rows)}")
print(
    "Selected model family            : "
    f"{phase8['selected_model_family']['candidate_id']}"
)
print(
    "Representation                   : "
    f"{phase8['selected_model_family']['representation']}"
)
print(f"Canonical deployment seed        : {phase9['canonical_seed']}")
print(
    "Checkpoint SHA-256               : "
    f"{phase9['bundled_checkpoint_sha256']}"
)
print(
    "Release tree SHA-256             : "
    f"{phase10['release_directory_tree_sha256']}"
)
print(
    "ZIP SHA-256                      : "
    f"{phase10['zip_archive_sha256']}"
)
print(
    "Deterministic rebuild exact      : "
    f"{phase10['deterministic_rebuild_exact']}"
)
print(
    "Raw dataset included             : "
    f"{phase10['raw_dataset_included']}"
)
print(
    "Processed dataset included       : "
    f"{phase10['processed_dataset_included']}"
)
print(
    "Nonselected checkpoints included: "
    f"{phase10['nonselected_checkpoints_included']}"
)
print()
print(f"CSV                              : {TABLE_CSV}")
print(f"Markdown                         : {TABLE_MD}")
print(f"Source JSON                      : {SOURCE_JSON}")
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
print("PHASE 13 TABLE 07 GENERATED AND LOCKED")
