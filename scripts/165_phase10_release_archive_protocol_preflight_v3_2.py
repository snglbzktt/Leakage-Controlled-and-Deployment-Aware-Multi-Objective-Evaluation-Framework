from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE8_FAMILY_LOCK = (
    AUDIT
    / "phase8_final_model_family_locked_v3_2.json"
)

PHASE9_ARTIFACT_LOCK = (
    AUDIT
    / "phase9_final_deployment_artifact_locked_v3_2.json"
)

PHASE9_LOCK_MANIFEST = (
    AUDIT
    / "phase9_final_deployment_artifact_lock_manifest_v3_2.json"
)

PHASE9_VERIFICATION = (
    AUDIT
    / "phase9_final_artifact_verification_v3_2.json"
)

PHASE9_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase9_final_artifact_protocol_v3_2.json"
)

ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

BUNDLE_ARTIFACT_LOCK = (
    ARTIFACT_ROOT
    / "artifact_lock.json"
)

RELEASE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "release"
)

PLANNED_STAGING_ROOT = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_2.building"
)

PLANNED_RELEASE_DIRECTORY = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_2"
)

PLANNED_ARCHIVE_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_2.zip"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase10_release_archive_protocol_v3_2.json"
)

OUTPUT_SOURCE_MANIFEST = (
    AUDIT
    / "phase10_release_archive_source_manifest_v3_2.json"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase10_release_archive_v3_2"
PHASE8_PROTOCOL_VERSION = "phase8_multi_objective_decision_v3_2"
PHASE9_PROTOCOL_VERSION = "phase9_final_artifact_v3_2"

EXPECTED_MODEL_FAMILY = "tinyml_mlp::P50-QAT"
EXPECTED_REPRESENTATION = "static_int8"
EXPECTED_CANONICAL_SEED = 2026

EXPECTED_BUNDLE_FILES = {
    "model_int8_checkpoint.pt",
    "nbaiot_models.py",
    "phase5_physical_pruning_engine_v3_2.py",
    "phase5_qat_engine_v3_2.py",
    "train_only_standard_scaler_v3_2.npz",
    "inference_contract.json",
    "artifact_manifest.json",
    "artifact_lock.json",
}

PLANNED_GENERATED_RELEASE_FILES = {
    "README_RELEASE.md",
    "DATA_NOT_INCLUDED.md",
    "release_manifest.json",
    "release_lock.json",
}

FIXED_ZIP_TIMESTAMP = [
    1980,
    1,
    1,
    0,
    0,
    0,
]

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

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination file locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
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


def file_record(
    path: Path,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }

    if relative_to is not None:
        record["relative_path"] = str(
            path.relative_to(relative_to)
        )

    return record


required_paths = (
    PHASE8_FAMILY_LOCK,
    PHASE9_ARTIFACT_LOCK,
    PHASE9_LOCK_MANIFEST,
    PHASE9_VERIFICATION,
    PHASE9_PROTOCOL,
    ARTIFACT_ROOT,
    BUNDLE_ARTIFACT_LOCK,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_SOURCE_MANIFEST,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 10 release-preflight output "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

for planned_path in (
    PLANNED_STAGING_ROOT,
    PLANNED_RELEASE_DIRECTORY,
    PLANNED_ARCHIVE_PATH,
):
    if planned_path.exists():
        raise FileExistsError(
            "Planned release output already exists; "
            "refusing to continue before protocol lock: "
            f"{planned_path}"
        )

phase8_lock = read_json(
    PHASE8_FAMILY_LOCK
)

phase9_lock = read_json(
    PHASE9_ARTIFACT_LOCK
)

phase9_manifest = read_json(
    PHASE9_LOCK_MANIFEST
)

phase9_verification = read_json(
    PHASE9_VERIFICATION
)

phase9_protocol = read_json(
    PHASE9_PROTOCOL
)

bundle_lock = read_json(
    BUNDLE_ARTIFACT_LOCK
)

observed_bundle_files = {
    path.name
    for path in ARTIFACT_ROOT.iterdir()
    if path.is_file()
}

entry_checks = {
    "phase8_family_locked": (
        phase8_lock.get("status")
        == "locked"
        and phase8_lock.get(
            "protocol_version"
        )
        == PHASE8_PROTOCOL_VERSION
        and phase8_lock.get(
            "final_model_family_selected"
        )
        is True
        and phase8_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase9_artifact_locked": (
        phase9_lock.get("status")
        == "locked"
        and phase9_lock.get(
            "protocol_version"
        )
        == PHASE9_PROTOCOL_VERSION
        and phase9_lock.get(
            "single_checkpoint_locked"
        )
        is True
        and phase9_lock.get(
            "final_deployment_artifact_locked"
        )
        is True
        and phase9_lock.get(
            "ready_for_release_archive"
        )
        is True
        and phase9_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase9_lock_manifest_locked": (
        phase9_manifest.get("status")
        == "locked"
        and phase9_manifest.get(
            "protocol_version"
        )
        == PHASE9_PROTOCOL_VERSION
        and phase9_manifest.get(
            "single_checkpoint_locked"
        )
        is True
        and phase9_manifest.get(
            "ready_for_release_archive"
        )
        is True
        and phase9_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase9_verification_passed": (
        phase9_verification.get(
            "status"
        )
        == "passed"
        and phase9_verification.get(
            "ready_to_lock_final_artifact"
        )
        is True
        and phase9_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase9_protocol_locked": (
        phase9_protocol.get("status")
        == "locked"
        and phase9_protocol.get(
            "protocol_version"
        )
        == PHASE9_PROTOCOL_VERSION
        and phase9_protocol.get(
            "all_checks_passed"
        )
        is True
    ),
    "bundle_lock_locked": (
        bundle_lock.get("status")
        == "locked"
        and bundle_lock.get(
            "single_checkpoint_locked"
        )
        is True
        and bundle_lock.get(
            "final_deployment_artifact_locked"
        )
        is True
        and bundle_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "model_family_matches": (
        phase9_lock.get(
            "model_family"
        )
        == EXPECTED_MODEL_FAMILY
        and bundle_lock.get(
            "model_family"
        )
        == EXPECTED_MODEL_FAMILY
    ),
    "representation_matches": (
        phase9_lock.get(
            "representation"
        )
        == EXPECTED_REPRESENTATION
        and bundle_lock.get(
            "representation"
        )
        == EXPECTED_REPRESENTATION
    ),
    "canonical_seed_matches": (
        int(
            phase9_lock.get(
                "canonical_seed"
            )
        )
        == EXPECTED_CANONICAL_SEED
        and int(
            bundle_lock.get(
                "canonical_seed"
            )
        )
        == EXPECTED_CANONICAL_SEED
    ),
    "bundle_file_set_exact": (
        observed_bundle_files
        == EXPECTED_BUNDLE_FILES
    ),
    "bundle_checkpoint_hash_matches_lock": (
        phase9_lock.get(
            "bundled_checkpoint_sha256"
        )
        == sha256_file(
            ARTIFACT_ROOT
            / "model_int8_checkpoint.pt"
        )
        == bundle_lock.get(
            "checkpoint_sha256"
        )
    ),
    "bundle_manifest_hash_matches_lock": (
        bundle_lock.get(
            "artifact_manifest_sha256"
        )
        == sha256_file(
            ARTIFACT_ROOT
            / "artifact_manifest.json"
        )
    ),
    "bundle_contract_hash_matches_lock": (
        bundle_lock.get(
            "inference_contract_sha256"
        )
        == sha256_file(
            ARTIFACT_ROOT
            / "inference_contract.json"
        )
    ),
    "release_outputs_absent": (
        not PLANNED_STAGING_ROOT.exists()
        and not PLANNED_RELEASE_DIRECTORY.exists()
        and not PLANNED_ARCHIVE_PATH.exists()
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
        "Phase 10 release-archive preflight "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

bundle_inventory = [
    file_record(
        path,
        relative_to=ARTIFACT_ROOT,
    )
    for path in sorted(
        ARTIFACT_ROOT.iterdir()
    )
    if path.is_file()
]

audit_files = (
    PHASE8_FAMILY_LOCK,
    PHASE9_ARTIFACT_LOCK,
    PHASE9_LOCK_MANIFEST,
    PHASE9_VERIFICATION,
    PHASE9_PROTOCOL,
)

audit_inventory = [
    file_record(path)
    for path in audit_files
]

planned_release_entries = []

for record in bundle_inventory:
    planned_release_entries.append(
        {
            "archive_path": (
                "deployment_artifact/"
                + record["relative_path"].replace(
                    "\\",
                    "/",
                )
            ),
            "source_path": record["path"],
            "source_sha256": record["sha256"],
            "source_size_bytes": (
                record["size_bytes"]
            ),
            "entry_role": (
                "locked_deployment_artifact"
            ),
        }
    )

for path in audit_files:
    planned_release_entries.append(
        {
            "archive_path": (
                "audit/"
                + path.name
            ),
            "source_path": str(path),
            "source_sha256": (
                sha256_file(path)
            ),
            "source_size_bytes": int(
                path.stat().st_size
            ),
            "entry_role": (
                "locked_audit_evidence"
            ),
        }
    )

for generated_name in sorted(
    PLANNED_GENERATED_RELEASE_FILES
):
    planned_release_entries.append(
        {
            "archive_path": generated_name,
            "source_path": "",
            "source_sha256": "",
            "source_size_bytes": "",
            "entry_role": (
                "generated_release_metadata"
            ),
        }
    )

planned_archive_paths = [
    entry["archive_path"]
    for entry in planned_release_entries
]

if (
    len(planned_archive_paths)
    != len(set(planned_archive_paths))
):
    raise RuntimeError(
        "Duplicate planned archive paths detected."
    )

protocol = {
    "status": "locked",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "release_identity": {
        "release_name": (
            "TinyML_IDS_IEEE_release_v3_2"
        ),
        "model_family": (
            EXPECTED_MODEL_FAMILY
        ),
        "representation": (
            EXPECTED_REPRESENTATION
        ),
        "canonical_seed": (
            EXPECTED_CANONICAL_SEED
        ),
    },
    "planned_outputs": {
        "staging_root": str(
            PLANNED_STAGING_ROOT
        ),
        "release_directory": str(
            PLANNED_RELEASE_DIRECTORY
        ),
        "zip_archive": str(
            PLANNED_ARCHIVE_PATH
        ),
    },
    "archive_policy": {
        "format": "ZIP",
        "compression": "ZIP_DEFLATED",
        "entry_order": (
            "lexicographic by normalized "
            "forward-slash archive path"
        ),
        "fixed_entry_timestamp": (
            FIXED_ZIP_TIMESTAMP
        ),
        "fixed_external_permissions": (
            "0644 for files"
        ),
        "UTF8_names": True,
        "duplicate_paths_allowed": False,
        "overwrite_existing_outputs": False,
        "source_hash_verification_before_copy": (
            True
        ),
        "release_directory_hash_manifest": (
            True
        ),
        "archive_hash_recording": True,
        "independent_archive_verification_required": (
            True
        ),
    },
    "included_content": {
        "locked_deployment_bundle_file_count": (
            len(bundle_inventory)
        ),
        "locked_audit_file_count": (
            len(audit_inventory)
        ),
        "generated_release_metadata_file_count": (
            len(
                PLANNED_GENERATED_RELEASE_FILES
            )
        ),
        "planned_total_file_count": (
            len(planned_release_entries)
        ),
        "planned_entries": (
            planned_release_entries
        ),
    },
    "excluded_content": {
        "raw_dataset": True,
        "processed_dataset": True,
        "train_validation_test_arrays": True,
        "database_files": True,
        "nonselected_checkpoints": True,
        "temporary_files": True,
        "logs": True,
        "reason": (
            "The release archive distributes only "
            "the locked deployment artifact and "
            "minimum audit evidence. Dataset files "
            "and nonselected checkpoints are excluded."
        ),
    },
    "release_documentation_policy": {
        "README_RELEASE.md": (
            "Describe model family, canonical seed, "
            "runtime contract, file layout, and "
            "integrity verification."
        ),
        "DATA_NOT_INCLUDED.md": (
            "State that N-BaIoT data and derived "
            "train/validation/test arrays are not "
            "redistributed."
        ),
        "release_manifest.json": (
            "Record every released file's relative "
            "path, size, and SHA-256 before ZIP creation."
        ),
        "release_lock.json": (
            "Record final directory and ZIP hashes "
            "after independent verification."
        ),
    },
    "source_artifacts": {
        "phase8_model_family_lock": (
            file_record(
                PHASE8_FAMILY_LOCK
            )
        ),
        "phase9_final_artifact_lock": (
            file_record(
                PHASE9_ARTIFACT_LOCK
            )
        ),
        "phase9_lock_manifest": (
            file_record(
                PHASE9_LOCK_MANIFEST
            )
        ),
        "phase9_verification": (
            file_record(
                PHASE9_VERIFICATION
            )
        ),
        "phase9_protocol": (
            file_record(
                PHASE9_PROTOCOL
            )
        ),
        "deployment_artifact_inventory": (
            bundle_inventory
        ),
    },
    "release_construction_performed": False,
    "zip_archive_created": False,
    "release_locked": False,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

source_manifest = {
    "status": "planned",
    "phase": 10,
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "created_at_utc": utc_now(),
    "release_name": (
        "TinyML_IDS_IEEE_release_v3_2"
    ),
    "planned_entry_count": (
        len(planned_release_entries)
    ),
    "planned_entries": (
        planned_release_entries
    ),
    "raw_or_processed_dataset_included": (
        False
    ),
    "nonselected_checkpoint_included": (
        False
    ),
    "release_construction_performed": (
        False
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SOURCE_MANIFEST,
    source_manifest,
)

preflight_checks = {
    "bundle_file_count_8": (
        len(bundle_inventory) == 8
    ),
    "audit_file_count_5": (
        len(audit_inventory) == 5
    ),
    "generated_metadata_count_4": (
        len(
            PLANNED_GENERATED_RELEASE_FILES
        )
        == 4
    ),
    "planned_total_file_count_17": (
        len(planned_release_entries)
        == 17
    ),
    "archive_paths_unique": (
        len(planned_archive_paths)
        == len(set(planned_archive_paths))
    ),
    "raw_dataset_excluded": (
        protocol[
            "excluded_content"
        ]["raw_dataset"]
        is True
    ),
    "processed_dataset_excluded": (
        protocol[
            "excluded_content"
        ]["processed_dataset"]
        is True
    ),
    "nonselected_checkpoints_excluded": (
        protocol[
            "excluded_content"
        ][
            "nonselected_checkpoints"
        ]
        is True
    ),
    "release_outputs_not_created": (
        not PLANNED_STAGING_ROOT.exists()
        and not PLANNED_RELEASE_DIRECTORY.exists()
        and not PLANNED_ARCHIVE_PATH.exists()
    ),
    "release_construction_not_performed": (
        protocol[
            "release_construction_performed"
        ]
        is False
    ),
    "zip_not_created": (
        protocol[
            "zip_archive_created"
        ]
        is False
    ),
    "release_not_locked": (
        protocol[
            "release_locked"
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
        "Phase 10 release-archive preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_"
        "protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "protocol": (
        file_record(
            OUTPUT_PROTOCOL
        )
    ),
    "source_manifest": (
        file_record(
            OUTPUT_SOURCE_MANIFEST
        )
    ),
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "release_locked": False,
    "ready_for_release_construction": (
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
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_"
        "protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "release_name": (
        "TinyML_IDS_IEEE_release_v3_2"
    ),
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "source_manifest": str(
        OUTPUT_SOURCE_MANIFEST
    ),
    "source_manifest_sha256": (
        sha256_file(
            OUTPUT_SOURCE_MANIFEST
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
    "planned_release_directory": str(
        PLANNED_RELEASE_DIRECTORY
    ),
    "planned_zip_archive": str(
        PLANNED_ARCHIVE_PATH
    ),
    "planned_file_count": (
        len(planned_release_entries)
    ),
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "release_locked": False,
    "ready_for_release_construction": (
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
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_"
        "protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE8_FAMILY_LOCK
        ),
        file_record(
            PHASE9_ARTIFACT_LOCK
        ),
        file_record(
            PHASE9_LOCK_MANIFEST
        ),
        file_record(
            PHASE9_VERIFICATION
        ),
        file_record(
            PHASE9_PROTOCOL
        ),
        *bundle_inventory,
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_PROTOCOL
        ),
        file_record(
            OUTPUT_SOURCE_MANIFEST
        ),
        file_record(
            OUTPUT_PREFLIGHT
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "planned_file_count": (
        len(planned_release_entries)
    ),
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "release_locked": False,
    "ready_for_release_construction": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 10 DETERMINISTIC RELEASE ARCHIVE PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Phase 8 model family status     : LOCKED"
)
print(
    "Phase 9 deployment artifact     : LOCKED"
)
print(
    "Model family                    : tinyml_mlp::P50-QAT"
)
print(
    "Representation                  : static_int8"
)
print(
    "Canonical checkpoint seed       : 2026"
)
print(
    "Deployment bundle files         : 8"
)
print(
    "Audit evidence files            : 5"
)
print(
    "Generated release metadata      : 4"
)
print(
    "Planned release files           : 17"
)
print(
    "Raw dataset included            : False"
)
print(
    "Processed dataset included      : False"
)
print(
    "Nonselected checkpoints included: False"
)
print(
    "Deterministic ZIP timestamp     : 1980-01-01 00:00:00"
)
print(
    "Release construction performed  : False"
)
print(
    "ZIP archive created             : False"
)
print(
    "Release locked                  : False"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for release construction  : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 10 RELEASE ARCHIVE PROTOCOL LOCKED"
)
