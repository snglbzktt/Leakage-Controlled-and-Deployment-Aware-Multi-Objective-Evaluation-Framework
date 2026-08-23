from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

OLD_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase10_release_archive_protocol_v3_2.json"
)

OLD_PREFLIGHT_LOCK_PATH = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_locked_v3_2.json"
)

OLD_SOURCE_MANIFEST_PATH = (
    AUDIT
    / "phase10_release_archive_source_manifest_v3_2.json"
)

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

RELEASE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "release"
)

PLANNED_STAGING_ROOT = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.building"
)

PLANNED_RELEASE_DIRECTORY = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3"
)

PLANNED_ARCHIVE_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.zip"
)

PLANNED_EXTERNAL_FINAL_LOCK = (
    AUDIT
    / "phase10_release_archive_locked_v3_3.json"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase10_release_archive_protocol_v3_3.json"
)

OUTPUT_SOURCE_MANIFEST = (
    AUDIT
    / "phase10_release_archive_source_manifest_v3_3.json"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_lock_manifest_v3_3.json"
)

PROTOCOL_VERSION = "phase10_release_archive_v3_3"
SUPERSEDED_PROTOCOL_VERSION = "phase10_release_archive_v3_2"

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

GENERATED_PAYLOAD_DOCUMENTS = {
    "README_RELEASE.md",
    "DATA_NOT_INCLUDED.md",
}

MANIFEST_FILE_NAME = "release_manifest.json"

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
    OLD_PROTOCOL_PATH,
    OLD_PREFLIGHT_LOCK_PATH,
    OLD_SOURCE_MANIFEST_PATH,
    PHASE8_FAMILY_LOCK,
    PHASE9_ARTIFACT_LOCK,
    PHASE9_LOCK_MANIFEST,
    PHASE9_VERIFICATION,
    PHASE9_PROTOCOL,
    ARTIFACT_ROOT,
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
            "Corrected Phase 10 protocol artifact "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

for planned_path in (
    PLANNED_STAGING_ROOT,
    PLANNED_RELEASE_DIRECTORY,
    PLANNED_ARCHIVE_PATH,
    PLANNED_EXTERNAL_FINAL_LOCK,
):
    if planned_path.exists():
        raise FileExistsError(
            "Corrected release output already exists: "
            f"{planned_path}"
        )

old_protocol = read_json(
    OLD_PROTOCOL_PATH
)

old_preflight_lock = read_json(
    OLD_PREFLIGHT_LOCK_PATH
)

phase8_lock = read_json(
    PHASE8_FAMILY_LOCK
)

phase9_lock = read_json(
    PHASE9_ARTIFACT_LOCK
)

entry_checks = {
    "old_protocol_locked": (
        old_protocol.get("status")
        == "locked"
        and old_protocol.get(
            "protocol_version"
        )
        == SUPERSEDED_PROTOCOL_VERSION
        and old_protocol.get(
            "all_checks_passed"
        )
        is True
    ),
    "old_preflight_locked": (
        old_preflight_lock.get("status")
        == "locked"
        and old_preflight_lock.get(
            "protocol_version"
        )
        == SUPERSEDED_PROTOCOL_VERSION
        and old_preflight_lock.get(
            "ready_for_release_construction"
        )
        is True
        and old_preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "old_protocol_hash_matches": (
        old_preflight_lock.get(
            "protocol_sha256"
        )
        == sha256_file(
            OLD_PROTOCOL_PATH
        )
    ),
    "old_source_manifest_hash_matches": (
        old_preflight_lock.get(
            "source_manifest_sha256"
        )
        == sha256_file(
            OLD_SOURCE_MANIFEST_PATH
        )
    ),
    "phase8_model_family_locked": (
        phase8_lock.get("status")
        == "locked"
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
            "model_family"
        )
        == EXPECTED_MODEL_FAMILY
        and phase9_lock.get(
            "representation"
        )
        == EXPECTED_REPRESENTATION
        and int(
            phase9_lock.get(
                "canonical_seed"
            )
        )
        == EXPECTED_CANONICAL_SEED
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
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Corrected Phase 10 protocol entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

observed_bundle_files = {
    path.name
    for path in ARTIFACT_ROOT.iterdir()
    if path.is_file()
}

if observed_bundle_files != EXPECTED_BUNDLE_FILES:
    raise RuntimeError(
        "Locked deployment bundle file set mismatch."
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

planned_payload_entries: list[
    dict[str, Any]
] = []

for record in bundle_inventory:
    planned_payload_entries.append(
        {
            "archive_path": (
                "deployment_artifact/"
                + record[
                    "relative_path"
                ].replace(
                    "\\",
                    "/",
                )
            ),
            "source_path": (
                record["path"]
            ),
            "source_sha256": (
                record["sha256"]
            ),
            "source_size_bytes": (
                record["size_bytes"]
            ),
            "entry_role": (
                "locked_deployment_artifact"
            ),
        }
    )

for path in audit_files:
    planned_payload_entries.append(
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
    GENERATED_PAYLOAD_DOCUMENTS
):
    planned_payload_entries.append(
        {
            "archive_path": (
                generated_name
            ),
            "source_path": "",
            "source_sha256": "",
            "source_size_bytes": "",
            "entry_role": (
                "generated_release_documentation"
            ),
        }
    )

payload_archive_paths = [
    entry["archive_path"]
    for entry in planned_payload_entries
]

if (
    len(payload_archive_paths)
    != len(set(payload_archive_paths))
):
    raise RuntimeError(
        "Duplicate payload archive paths detected."
    )

planned_archive_entries = [
    *planned_payload_entries,
    {
        "archive_path": (
            MANIFEST_FILE_NAME
        ),
        "source_path": "",
        "source_sha256": "",
        "source_size_bytes": "",
        "entry_role": (
            "generated_payload_manifest"
        ),
    },
]

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
    "supersedes": {
        "protocol_version": (
            SUPERSEDED_PROTOCOL_VERSION
        ),
        "protocol_path": str(
            OLD_PROTOCOL_PATH
        ),
        "protocol_sha256": (
            sha256_file(
                OLD_PROTOCOL_PATH
            )
        ),
        "reason": (
            "The superseded protocol placed "
            "release_lock.json inside the ZIP while "
            "requiring that same file to contain the "
            "ZIP hash, and required a manifest to hash "
            "itself. Both requirements create circular "
            "hash dependencies. Version 3.3 removes "
            "those circularities without changing the "
            "selected model or deployment artifact."
        ),
    },
    "release_identity": {
        "release_name": (
            "TinyML_IDS_IEEE_release_v3_3"
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
        "external_final_lock": str(
            PLANNED_EXTERNAL_FINAL_LOCK
        ),
    },
    "archive_policy": {
        "format": "ZIP",
        "compression": (
            "ZIP_DEFLATED"
        ),
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
        "independent_archive_verification_required": (
            True
        ),
    },
    "manifest_policy": {
        "manifest_file": (
            MANIFEST_FILE_NAME
        ),
        "manifest_inside_archive": True,
        "manifest_records_payload_files": (
            True
        ),
        "manifest_records_itself": False,
        "reason_manifest_excludes_itself": (
            "A file cannot contain its own final "
            "SHA-256 without a circular dependency."
        ),
        "external_lock_records_manifest_hash": (
            True
        ),
    },
    "final_lock_policy": {
        "final_lock_inside_archive": False,
        "external_final_lock": str(
            PLANNED_EXTERNAL_FINAL_LOCK
        ),
        "external_lock_records": [
            "release directory tree hash",
            "release manifest SHA-256",
            "ZIP archive SHA-256",
            "ZIP size",
            "independent verification report SHA-256",
        ],
        "reason_lock_is_external": (
            "A lock file inside the ZIP cannot contain "
            "the final ZIP SHA-256 without changing the "
            "archive whose hash it records."
        ),
    },
    "included_content": {
        "locked_deployment_bundle_file_count": (
            len(bundle_inventory)
        ),
        "locked_audit_file_count": (
            len(audit_files)
        ),
        "generated_documentation_file_count": (
            len(
                GENERATED_PAYLOAD_DOCUMENTS
            )
        ),
        "payload_file_count_excluding_manifest": (
            len(
                planned_payload_entries
            )
        ),
        "archive_file_count_including_manifest": (
            len(
                planned_archive_entries
            )
        ),
        "planned_payload_entries": (
            planned_payload_entries
        ),
        "planned_archive_entries": (
            planned_archive_entries
        ),
    },
    "excluded_content": {
        "raw_dataset": True,
        "processed_dataset": True,
        "train_validation_test_arrays": (
            True
        ),
        "database_files": True,
        "nonselected_checkpoints": True,
        "temporary_files": True,
        "logs": True,
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
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "external_final_lock_created": (
        False
    ),
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
        "TinyML_IDS_IEEE_release_v3_3"
    ),
    "payload_file_count_excluding_manifest": (
        len(
            planned_payload_entries
        )
    ),
    "archive_file_count_including_manifest": (
        len(
            planned_archive_entries
        )
    ),
    "planned_payload_entries": (
        planned_payload_entries
    ),
    "planned_archive_entries": (
        planned_archive_entries
    ),
    "external_final_lock": str(
        PLANNED_EXTERNAL_FINAL_LOCK
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
    "deployment_bundle_files_8": (
        len(bundle_inventory)
        == 8
    ),
    "audit_evidence_files_5": (
        len(audit_files)
        == 5
    ),
    "generated_documentation_files_2": (
        len(
            GENERATED_PAYLOAD_DOCUMENTS
        )
        == 2
    ),
    "payload_files_excluding_manifest_15": (
        len(
            planned_payload_entries
        )
        == 15
    ),
    "archive_files_including_manifest_16": (
        len(
            planned_archive_entries
        )
        == 16
    ),
    "payload_paths_unique": (
        len(payload_archive_paths)
        == len(set(payload_archive_paths))
    ),
    "manifest_does_not_hash_itself": (
        protocol[
            "manifest_policy"
        ][
            "manifest_records_itself"
        ]
        is False
    ),
    "final_lock_external": (
        protocol[
            "final_lock_policy"
        ][
            "final_lock_inside_archive"
        ]
        is False
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
    "corrected_outputs_absent": (
        not PLANNED_STAGING_ROOT.exists()
        and not PLANNED_RELEASE_DIRECTORY.exists()
        and not PLANNED_ARCHIVE_PATH.exists()
        and not PLANNED_EXTERNAL_FINAL_LOCK.exists()
    ),
    "release_construction_not_performed": (
        protocol[
            "release_construction_performed"
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
        "Corrected Phase 10 preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_"
        "protocol_circularity_fix_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": (
        entry_checks
    ),
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
    "external_final_lock_created": (
        False
    ),
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
        "TinyML_IDS_IEEE_release_v3_3"
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
    "planned_external_final_lock": str(
        PLANNED_EXTERNAL_FINAL_LOCK
    ),
    "payload_file_count_excluding_manifest": (
        len(
            planned_payload_entries
        )
    ),
    "archive_file_count_including_manifest": (
        len(
            planned_archive_entries
        )
    ),
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "external_final_lock_created": (
        False
    ),
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
    "superseded_protocol": (
        file_record(
            OLD_PROTOCOL_PATH
        )
    ),
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
    "payload_file_count_excluding_manifest": (
        len(
            planned_payload_entries
        )
    ),
    "archive_file_count_including_manifest": (
        len(
            planned_archive_entries
        )
    ),
    "release_construction_performed": (
        False
    ),
    "zip_archive_created": False,
    "external_final_lock_created": (
        False
    ),
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
print("PHASE 10 RELEASE ARCHIVE PROTOCOL CIRCULARITY FIX")
print("=" * 92)
print(
    "Superseded protocol             : phase10_release_archive_v3_2"
)
print(
    "Corrected protocol              : phase10_release_archive_v3_3"
)
print(
    "Reason                          : removed circular hash dependencies"
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
    "Generated documentation files   : 2"
)
print(
    "Payload files before manifest   : 15"
)
print(
    "Archive files with manifest     : 16"
)
print(
    "Manifest hashes itself          : False"
)
print(
    "Final release lock inside ZIP   : False"
)
print(
    "External final lock planned     : True"
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
    "Release construction performed  : False"
)
print(
    "ZIP archive created             : False"
)
print(
    "Release locked                  : False"
)
print(
    "Corrected preflight status      : LOCKED"
)
print(
    "Ready for release construction  : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 10 CORRECTED RELEASE PROTOCOL LOCKED"
)
