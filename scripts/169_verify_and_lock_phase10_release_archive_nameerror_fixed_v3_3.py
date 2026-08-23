from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase10_release_archive_protocol_v3_3.json"
)

PREFLIGHT_LOCK_PATH = (
    AUDIT
    / "phase10_release_archive_protocol_preflight_locked_v3_3.json"
)

SOURCE_MANIFEST_PATH = (
    AUDIT
    / "phase10_release_archive_source_manifest_v3_3.json"
)

CONSTRUCTION_REPORT_PATH = (
    AUDIT
    / "phase10_release_archive_construction_v3_3.json"
)

CONSTRUCTION_COMPLETION_PATH = (
    AUDIT
    / "phase10_release_archive_constructed_v3_3.json"
)

PHASE9_FINAL_ARTIFACT_LOCK_PATH = (
    AUDIT
    / "phase9_final_deployment_artifact_locked_v3_2.json"
)

RELEASE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "release"
)

RELEASE_DIRECTORY = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3"
)

ZIP_ARCHIVE_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.zip"
)

REBUILD_ZIP_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.verification_rebuild.zip"
)

RELEASE_MANIFEST_PATH = (
    RELEASE_DIRECTORY
    / "release_manifest.json"
)

OUTPUT_VERIFICATION = (
    AUDIT
    / "phase10_release_archive_verification_v3_3.json"
)

OUTPUT_FINAL_LOCK = (
    AUDIT
    / "phase10_release_archive_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase10_release_archive_lock_manifest_v3_3.json"
)

PROTOCOL_VERSION = "phase10_release_archive_v3_3"

RELEASE_NAME = "TinyML_IDS_IEEE_release_v3_3"
README_NAME = "README_RELEASE.md"
DATA_NOTICE_NAME = "DATA_NOT_INCLUDED.md"
MODEL_FAMILY = "tinyml_mlp::P50-QAT"
REPRESENTATION = "static_int8"
CANONICAL_SEED = 2026

EXPECTED_CLASS_LABELS = [
    "benign",
    "gafgyt",
    "mirai",
]

EXPECTED_PAYLOAD_FILE_COUNT = 15
EXPECTED_ARCHIVE_FILE_COUNT = 16

EXPECTED_DIRECTORY_FILES = {
    "DATA_NOT_INCLUDED.md",
    "README_RELEASE.md",
    "release_manifest.json",
    "audit/phase8_final_model_family_locked_v3_2.json",
    "audit/phase9_final_artifact_protocol_v3_2.json",
    "audit/phase9_final_artifact_verification_v3_2.json",
    "audit/phase9_final_deployment_artifact_lock_manifest_v3_2.json",
    "audit/phase9_final_deployment_artifact_locked_v3_2.json",
    "deployment_artifact/artifact_lock.json",
    "deployment_artifact/artifact_manifest.json",
    "deployment_artifact/inference_contract.json",
    "deployment_artifact/model_int8_checkpoint.pt",
    "deployment_artifact/nbaiot_models.py",
    "deployment_artifact/phase5_physical_pruning_engine_v3_2.py",
    "deployment_artifact/phase5_qat_engine_v3_2.py",
    "deployment_artifact/train_only_standard_scaler_v3_2.npz",
}

EXPECTED_PAYLOAD_FILES = (
    EXPECTED_DIRECTORY_FILES
    - {"release_manifest.json"}
)

FIXED_ZIP_TIMESTAMP = (
    1980,
    1,
    1,
    0,
    0,
    0,
)

EXPECTED_FILE_MODE = 0o100644

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


README_TEXT = """# TinyML IDS IEEE Release v3.3

## Locked deployment artifact

- Model family: `tinyml_mlp::P50-QAT`
- Representation: `static_int8`
- Canonical deployment checkpoint seed: `2026`
- Input shape: `[batch, 115]`
- Input dtype: `float32`
- Preprocessing: locked train-only StandardScaler
- Output shape: `[batch, 3]`
- Output semantics: logits
- Class order: `0=benign`, `1=gafgyt`, `2=mirai`
- Runtime target: CPU
- Quantization backend recorded by the artifact: `onednn`

The model family was selected from aggregate five-seed evidence. The concrete deployment checkpoint was fixed to the canonical project seed 2026 and was not selected by ranking test, validation, or deployment metrics.

## Directory layout

- `deployment_artifact/`: locked model checkpoint, source modules, scaler, inference contract, artifact manifest, and artifact lock.
- `audit/`: minimum locked evidence for model-family and deployment-artifact provenance.
- `release_manifest.json`: SHA-256 and byte size for every payload file except the manifest itself.
- `DATA_NOT_INCLUDED.md`: dataset redistribution statement.

## Integrity

Verify each payload file against `release_manifest.json`. The final ZIP SHA-256 and release-directory tree hash are stored in the external Phase 10 release lock after independent verification.

## Scientific scope

This bundle contains the locked deployment artifact only. It does not replace the full experimental repository, raw data, processed split arrays, or nonselected checkpoints.
"""

DATA_NOTICE_TEXT = """# Data Not Included

This release does not redistribute the N-BaIoT dataset, raw CSV files, processed databases, canonical train/validation/test arrays, cached feature arrays, or nonselected checkpoints.

The archive contains only the locked deployment artifact and minimum audit evidence needed to identify and verify that artifact.

Users must obtain any source dataset under its applicable license and reproduce preprocessing and evaluation from the research repository and documented protocol.
"""


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


def sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


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


def release_tree_sha256(
    root: Path,
    relative_paths: list[str],
) -> str:
    digest = hashlib.sha256()

    digest.update(
        b"phase10_release_tree_v1\0"
    )

    for relative_path in sorted(
        relative_paths
    ):
        path = (
            root
            / Path(relative_path)
        )

        size_bytes = int(
            path.stat().st_size
        )

        file_hash = sha256_file(path)

        digest.update(
            relative_path.encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
        digest.update(
            str(size_bytes).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(
            file_hash.encode(
                "ascii"
            )
        )
        digest.update(b"\n")

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
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


def remove_with_retry(
    path: Path,
) -> None:
    if not path.exists():
        return

    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        f"Windows kept the path locked: {path}"
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
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
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
        record["relative_path"] = (
            path.relative_to(
                relative_to
            )
            .as_posix()
        )

    return record


def deterministic_zip_info(
    archive_path: str,
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        filename=archive_path,
        date_time=FIXED_ZIP_TIMESTAMP,
    )

    info.compress_type = (
        zipfile.ZIP_DEFLATED
    )

    info.create_system = 3
    info.external_attr = (
        EXPECTED_FILE_MODE << 16
    )

    info.flag_bits |= 0x800

    return info


def deterministic_zip(
    source_root: Path,
    destination: Path,
    archive_paths: list[str],
) -> None:
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for archive_path in sorted(
            archive_paths
        ):
            source_path = (
                source_root
                / Path(archive_path)
            )

            data = source_path.read_bytes()

            archive.writestr(
                deterministic_zip_info(
                    archive_path
                ),
                data,
                compress_type=(
                    zipfile.ZIP_DEFLATED
                ),
                compresslevel=9,
            )


required_paths = (
    PROTOCOL_PATH,
    PREFLIGHT_LOCK_PATH,
    SOURCE_MANIFEST_PATH,
    CONSTRUCTION_REPORT_PATH,
    CONSTRUCTION_COMPLETION_PATH,
    PHASE9_FINAL_ARTIFACT_LOCK_PATH,
    RELEASE_DIRECTORY,
    ZIP_ARCHIVE_PATH,
    RELEASE_MANIFEST_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_VERIFICATION,
    OUTPUT_FINAL_LOCK,
    OUTPUT_LOCK_MANIFEST,
    REBUILD_ZIP_PATH,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 10 verification output "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

protocol = read_json(
    PROTOCOL_PATH
)

preflight_lock = read_json(
    PREFLIGHT_LOCK_PATH
)

source_manifest = read_json(
    SOURCE_MANIFEST_PATH
)

construction_report = read_json(
    CONSTRUCTION_REPORT_PATH
)

construction_completion = read_json(
    CONSTRUCTION_COMPLETION_PATH
)

phase9_lock = read_json(
    PHASE9_FINAL_ARTIFACT_LOCK_PATH
)

release_manifest = read_json(
    RELEASE_MANIFEST_PATH
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and protocol.get(
            "all_checks_passed"
        )
        is True
    ),
    "preflight_locked": (
        preflight_lock.get("status")
        == "locked"
        and preflight_lock.get(
            "ready_for_release_construction"
        )
        is True
        and preflight_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "construction_report_completed": (
        construction_report.get(
            "status"
        )
        == "constructed"
        and construction_report.get(
            "ready_for_independent_verification"
        )
        is True
        and construction_report.get(
            "all_checks_passed"
        )
        is True
    ),
    "construction_completion_completed": (
        construction_completion.get(
            "status"
        )
        == "constructed"
        and construction_completion.get(
            "ready_for_independent_verification"
        )
        is True
        and construction_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "protocol_hash_matches": (
        preflight_lock.get(
            "protocol_sha256"
        )
        == sha256_file(
            PROTOCOL_PATH
        )
    ),
    "source_manifest_hash_matches": (
        preflight_lock.get(
            "source_manifest_sha256"
        )
        == sha256_file(
            SOURCE_MANIFEST_PATH
        )
    ),
    "construction_report_hash_matches": (
        construction_completion.get(
            "construction_report_sha256"
        )
        == sha256_file(
            CONSTRUCTION_REPORT_PATH
        )
    ),
    "release_manifest_hash_matches": (
        construction_completion.get(
            "release_manifest_sha256"
        )
        == sha256_file(
            RELEASE_MANIFEST_PATH
        )
    ),
    "zip_hash_matches_completion": (
        construction_completion.get(
            "zip_archive_sha256"
        )
        == sha256_file(
            ZIP_ARCHIVE_PATH
        )
    ),
    "phase9_final_artifact_locked": (
        phase9_lock.get("status")
        == "locked"
        and phase9_lock.get(
            "model_family"
        )
        == MODEL_FAMILY
        and phase9_lock.get(
            "representation"
        )
        == REPRESENTATION
        and int(
            phase9_lock.get(
                "canonical_seed"
            )
        )
        == CANONICAL_SEED
        and phase9_lock.get(
            "final_deployment_artifact_locked"
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
        "Phase 10 independent verification "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

directory_files = {
    path.relative_to(
        RELEASE_DIRECTORY
    ).as_posix()
    for path in RELEASE_DIRECTORY.rglob("*")
    if path.is_file()
}

directory_checks = {
    "directory_file_count_16": (
        len(directory_files)
        == EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "directory_file_set_exact": (
        directory_files
        == EXPECTED_DIRECTORY_FILES
    ),
    "readme_exact": (
        (
            RELEASE_DIRECTORY
            / README_NAME
        ).read_text(
            encoding="utf-8"
        )
        == README_TEXT
    ),
    "data_notice_exact": (
        (
            RELEASE_DIRECTORY
            / DATA_NOTICE_NAME
        ).read_text(
            encoding="utf-8"
        )
        == DATA_NOTICE_TEXT
    ),
}

failed_directory_checks = [
    name
    for name, passed
    in directory_checks.items()
    if not passed
]

if failed_directory_checks:
    raise RuntimeError(
        "Release-directory verification failed: "
        + ", ".join(
            failed_directory_checks
        )
    )

manifest_checks = {
    "manifest_status_pending_verification": (
        release_manifest.get("status")
        == (
            "constructed_pending_"
            "independent_verification"
        )
    ),
    "manifest_protocol_matches": (
        release_manifest.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "manifest_release_identity": (
        release_manifest.get(
            "release_name"
        )
        == RELEASE_NAME
        and release_manifest.get(
            "model_family"
        )
        == MODEL_FAMILY
        and release_manifest.get(
            "representation"
        )
        == REPRESENTATION
        and int(
            release_manifest.get(
                "canonical_seed"
            )
        )
        == CANONICAL_SEED
    ),
    "manifest_class_order": (
        release_manifest.get(
            "class_labels_in_output_index_order"
        )
        == EXPECTED_CLASS_LABELS
    ),
    "manifest_does_not_record_itself": (
        release_manifest.get(
            "manifest_records_itself"
        )
        is False
    ),
    "manifest_payload_count_15": (
        int(
            release_manifest.get(
                "payload_file_count"
            )
        )
        == EXPECTED_PAYLOAD_FILE_COUNT
    ),
    "manifest_ready": (
        release_manifest.get(
            "ready_for_independent_verification"
        )
        is True
        and release_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "dataset_excluded": (
        release_manifest.get(
            "raw_dataset_included"
        )
        is False
        and release_manifest.get(
            "processed_dataset_included"
        )
        is False
    ),
    "nonselected_checkpoints_excluded": (
        release_manifest.get(
            "nonselected_checkpoints_included"
        )
        is False
    ),
}

failed_manifest_checks = [
    name
    for name, passed
    in manifest_checks.items()
    if not passed
]

if failed_manifest_checks:
    raise RuntimeError(
        "Release-manifest identity verification "
        "failed: "
        + ", ".join(
            failed_manifest_checks
        )
    )

payload_records = list(
    release_manifest[
        "payload_files"
    ]
)

if len(payload_records) != (
    EXPECTED_PAYLOAD_FILE_COUNT
):
    raise RuntimeError(
        "Release-manifest payload record "
        "count mismatch."
    )

payload_record_paths = [
    record["relative_path"]
    for record in payload_records
]

if (
    len(payload_record_paths)
    != len(set(payload_record_paths))
):
    raise RuntimeError(
        "Duplicate release-manifest payload paths."
    )

if set(payload_record_paths) != (
    EXPECTED_PAYLOAD_FILES
):
    raise RuntimeError(
        "Release-manifest payload path set mismatch."
    )

payload_integrity_checks: dict[
    str,
    bool,
] = {}

for record in payload_records:
    relative_path = record[
        "relative_path"
    ]

    path = (
        RELEASE_DIRECTORY
        / Path(relative_path)
    )

    payload_integrity_checks[
        relative_path
    ] = (
        path.is_file()
        and int(
            record[
                "size_bytes"
            ]
        )
        == int(
            path.stat().st_size
        )
        and record["sha256"]
        == sha256_file(path)
    )

failed_payload_checks = [
    name
    for name, passed
    in payload_integrity_checks.items()
    if not passed
]

if failed_payload_checks:
    raise RuntimeError(
        "Release-manifest payload integrity "
        "failed: "
        + ", ".join(
            failed_payload_checks
        )
    )

planned_entries = list(
    source_manifest[
        "planned_payload_entries"
    ]
)

planned_entry_lookup = {
    entry["archive_path"]: entry
    for entry in planned_entries
}

if set(
    planned_entry_lookup.keys()
) != EXPECTED_PAYLOAD_FILES:
    raise RuntimeError(
        "Source-manifest planned payload "
        "set mismatch."
    )

source_copy_checks: dict[
    str,
    bool,
] = {}

for relative_path in sorted(
    EXPECTED_PAYLOAD_FILES
):
    planned = planned_entry_lookup[
        relative_path
    ]

    actual_path = (
        RELEASE_DIRECTORY
        / Path(relative_path)
    )

    role = planned[
        "entry_role"
    ]

    if role in {
        "locked_deployment_artifact",
        "locked_audit_evidence",
    }:
        source_path = Path(
            planned["source_path"]
        )

        source_copy_checks[
            relative_path
        ] = (
            source_path.is_file()
            and planned[
                "source_sha256"
            ]
            == sha256_file(
                source_path
            )
            == sha256_file(
                actual_path
            )
            and int(
                planned[
                    "source_size_bytes"
                ]
            )
            == int(
                source_path.stat().st_size
            )
            == int(
                actual_path.stat().st_size
            )
        )

    elif role == (
        "generated_release_documentation"
    ):
        source_copy_checks[
            relative_path
        ] = (
            relative_path
            in {
                README_NAME,
                DATA_NOTICE_NAME,
            }
        )

    else:
        source_copy_checks[
            relative_path
        ] = False

failed_source_copy_checks = [
    name
    for name, passed
    in source_copy_checks.items()
    if not passed
]

if failed_source_copy_checks:
    raise RuntimeError(
        "Release source-copy verification failed: "
        + ", ".join(
            failed_source_copy_checks
        )
    )

print("=" * 92)
print("PHASE 10 RELEASE ARCHIVE INDEPENDENT VERIFICATION")
print("=" * 92)
print(
    "Release name                    : TinyML_IDS_IEEE_release_v3_3"
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
    "Release directory files         : 16"
)
print(
    "Manifest payload files          : 15"
)
print(
    "Manifest payload hashes         : verified"
)
print(
    "Locked source copies            : verified"
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
print()

with zipfile.ZipFile(
    ZIP_ARCHIVE_PATH,
    mode="r",
) as archive:
    infos = archive.infolist()

    names = [
        info.filename
        for info in infos
    ]

    zip_entry_checks = {
        "entry_count_16": (
            len(infos)
            == EXPECTED_ARCHIVE_FILE_COUNT
        ),
        "entry_names_exact": (
            set(names)
            == EXPECTED_DIRECTORY_FILES
        ),
        "entry_names_unique": (
            len(names)
            == len(set(names))
        ),
        "entry_order_lexicographic": (
            names
            == sorted(names)
        ),
        "entry_timestamps_fixed": all(
            info.date_time
            == FIXED_ZIP_TIMESTAMP
            for info in infos
        ),
        "entry_compression_deflated": all(
            info.compress_type
            == zipfile.ZIP_DEFLATED
            for info in infos
        ),
        "entry_create_system_unix": all(
            info.create_system == 3
            for info in infos
        ),
        "entry_permissions_fixed": all(
            (
                info.external_attr
                >> 16
            )
            == EXPECTED_FILE_MODE
            for info in infos
        ),
        "testzip_passed": (
            archive.testzip()
            is None
        ),
    }

    zip_content_checks: dict[
        str,
        bool,
    ] = {}

    for info in infos:
        directory_path = (
            RELEASE_DIRECTORY
            / Path(info.filename)
        )

        zip_content_checks[
            info.filename
        ] = (
            sha256_bytes(
                archive.read(
                    info.filename
                )
            )
            == sha256_file(
                directory_path
            )
        )

failed_zip_entry_checks = [
    name
    for name, passed
    in zip_entry_checks.items()
    if not passed
]

failed_zip_content_checks = [
    name
    for name, passed
    in zip_content_checks.items()
    if not passed
]

if (
    failed_zip_entry_checks
    or failed_zip_content_checks
):
    raise RuntimeError(
        "ZIP independent verification failed: "
        + ", ".join(
            failed_zip_entry_checks
            + failed_zip_content_checks
        )
    )

rebuild_succeeded = False

try:
    deterministic_zip(
        RELEASE_DIRECTORY,
        REBUILD_ZIP_PATH,
        sorted(
            EXPECTED_DIRECTORY_FILES
        ),
    )

    rebuild_checks = {
        "rebuild_exists": (
            REBUILD_ZIP_PATH.is_file()
        ),
        "rebuild_size_exact": (
            int(
                REBUILD_ZIP_PATH.stat().st_size
            )
            == int(
                ZIP_ARCHIVE_PATH.stat().st_size
            )
        ),
        "rebuild_sha256_exact": (
            sha256_file(
                REBUILD_ZIP_PATH
            )
            == sha256_file(
                ZIP_ARCHIVE_PATH
            )
        ),
        "rebuild_bytes_exact": (
            REBUILD_ZIP_PATH.read_bytes()
            == ZIP_ARCHIVE_PATH.read_bytes()
        ),
    }

    failed_rebuild_checks = [
        name
        for name, passed
        in rebuild_checks.items()
        if not passed
    ]

    if failed_rebuild_checks:
        raise RuntimeError(
            "Deterministic ZIP rebuild failed: "
            + ", ".join(
                failed_rebuild_checks
            )
        )

    rebuild_succeeded = True

finally:
    if REBUILD_ZIP_PATH.exists():
        remove_with_retry(
            REBUILD_ZIP_PATH
        )

if not rebuild_succeeded:
    raise RuntimeError(
        "Deterministic ZIP rebuild did not complete."
    )

tree_hash = release_tree_sha256(
    RELEASE_DIRECTORY,
    sorted(
        EXPECTED_DIRECTORY_FILES
    ),
)

zip_hash = sha256_file(
    ZIP_ARCHIVE_PATH
)

manifest_hash = sha256_file(
    RELEASE_MANIFEST_PATH
)

verification_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "directory_checks_passed": all(
        directory_checks.values()
    ),
    "manifest_checks_passed": all(
        manifest_checks.values()
    ),
    "payload_integrity_passed": all(
        payload_integrity_checks.values()
    ),
    "source_copy_checks_passed": all(
        source_copy_checks.values()
    ),
    "zip_entry_checks_passed": all(
        zip_entry_checks.values()
    ),
    "zip_content_checks_passed": all(
        zip_content_checks.values()
    ),
    "deterministic_rebuild_passed": all(
        rebuild_checks.values()
    ),
    "release_tree_hash_computed": (
        len(tree_hash) == 64
    ),
    "zip_hash_computed": (
        len(zip_hash) == 64
    ),
    "manifest_hash_computed": (
        len(manifest_hash) == 64
    ),
    "external_final_lock_not_yet_written": (
        not OUTPUT_FINAL_LOCK.exists()
    ),
}

failed_verification_checks = [
    name
    for name, passed
    in verification_checks.items()
    if not passed
]

if failed_verification_checks:
    raise RuntimeError(
        "Phase 10 release verification failed: "
        + ", ".join(
            failed_verification_checks
        )
    )

verification = {
    "status": "passed",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive_"
        "independent_verification"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "release_name": (
        RELEASE_NAME
    ),
    "release_directory": str(
        RELEASE_DIRECTORY
    ),
    "zip_archive": str(
        ZIP_ARCHIVE_PATH
    ),
    "model_family": (
        MODEL_FAMILY
    ),
    "representation": (
        REPRESENTATION
    ),
    "canonical_seed": (
        CANONICAL_SEED
    ),
    "class_labels_in_output_index_order": (
        EXPECTED_CLASS_LABELS
    ),
    "entry_checks": (
        entry_checks
    ),
    "directory_checks": (
        directory_checks
    ),
    "manifest_checks": (
        manifest_checks
    ),
    "payload_integrity_checks": (
        payload_integrity_checks
    ),
    "source_copy_checks": (
        source_copy_checks
    ),
    "zip_entry_checks": (
        zip_entry_checks
    ),
    "zip_content_checks": (
        zip_content_checks
    ),
    "deterministic_rebuild_checks": (
        rebuild_checks
    ),
    "verification_checks": (
        verification_checks
    ),
    "release_directory_tree_hash_algorithm": (
        "sha256 over sorted records: "
        "relative_path NUL size NUL file_sha256 LF, "
        "prefixed by phase10_release_tree_v1 NUL"
    ),
    "release_directory_tree_sha256": (
        tree_hash
    ),
    "release_manifest_sha256": (
        manifest_hash
    ),
    "zip_archive_sha256": (
        zip_hash
    ),
    "zip_archive_size_bytes": int(
        ZIP_ARCHIVE_PATH.stat().st_size
    ),
    "payload_file_count": (
        EXPECTED_PAYLOAD_FILE_COUNT
    ),
    "archive_file_count": (
        EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "raw_dataset_included": False,
    "processed_dataset_included": (
        False
    ),
    "nonselected_checkpoints_included": (
        False
    ),
    "external_final_lock_written": (
        False
    ),
    "ready_to_lock_release": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_VERIFICATION,
    verification,
)

final_lock = {
    "status": "locked",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "release_name": (
        RELEASE_NAME
    ),
    "release_directory": str(
        RELEASE_DIRECTORY
    ),
    "zip_archive": str(
        ZIP_ARCHIVE_PATH
    ),
    "model_family": (
        MODEL_FAMILY
    ),
    "representation": (
        REPRESENTATION
    ),
    "canonical_seed": (
        CANONICAL_SEED
    ),
    "class_labels_in_output_index_order": (
        EXPECTED_CLASS_LABELS
    ),
    "payload_file_count": (
        EXPECTED_PAYLOAD_FILE_COUNT
    ),
    "archive_file_count": (
        EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "release_directory_tree_hash_algorithm": (
        verification[
            "release_directory_tree_hash_algorithm"
        ]
    ),
    "release_directory_tree_sha256": (
        tree_hash
    ),
    "release_manifest_sha256": (
        manifest_hash
    ),
    "zip_archive_sha256": (
        zip_hash
    ),
    "zip_archive_size_bytes": int(
        ZIP_ARCHIVE_PATH.stat().st_size
    ),
    "independent_verification_report": str(
        OUTPUT_VERIFICATION
    ),
    "independent_verification_report_sha256": (
        sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
    "deterministic_rebuild_exact": True,
    "raw_dataset_included": False,
    "processed_dataset_included": (
        False
    ),
    "nonselected_checkpoints_included": (
        False
    ),
    "release_locked": True,
    "ready_for_distribution": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_FINAL_LOCK,
    final_lock,
)

postlock_checks = {
    "external_final_lock_created": (
        OUTPUT_FINAL_LOCK.is_file()
    ),
    "external_final_lock_status_locked": (
        read_json(
            OUTPUT_FINAL_LOCK
        ).get(
            "status"
        )
        == "locked"
    ),
    "external_lock_zip_hash_matches": (
        read_json(
            OUTPUT_FINAL_LOCK
        ).get(
            "zip_archive_sha256"
        )
        == sha256_file(
            ZIP_ARCHIVE_PATH
        )
    ),
    "external_lock_tree_hash_matches": (
        read_json(
            OUTPUT_FINAL_LOCK
        ).get(
            "release_directory_tree_sha256"
        )
        == release_tree_sha256(
            RELEASE_DIRECTORY,
            sorted(
                EXPECTED_DIRECTORY_FILES
            ),
        )
    ),
    "external_lock_verification_hash_matches": (
        read_json(
            OUTPUT_FINAL_LOCK
        ).get(
            "independent_verification_report_sha256"
        )
        == sha256_file(
            OUTPUT_VERIFICATION
        )
    ),
}

failed_postlock_checks = [
    name
    for name, passed
    in postlock_checks.items()
    if not passed
]

if failed_postlock_checks:
    raise RuntimeError(
        "Phase 10 external final-lock "
        "verification failed: "
        + ", ".join(
            failed_postlock_checks
        )
    )

lock_manifest = {
    "status": "locked",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PROTOCOL_PATH
        ),
        file_record(
            PREFLIGHT_LOCK_PATH
        ),
        file_record(
            SOURCE_MANIFEST_PATH
        ),
        file_record(
            CONSTRUCTION_REPORT_PATH
        ),
        file_record(
            CONSTRUCTION_COMPLETION_PATH
        ),
        file_record(
            PHASE9_FINAL_ARTIFACT_LOCK_PATH
        ),
        file_record(
            RELEASE_MANIFEST_PATH
        ),
        file_record(
            ZIP_ARCHIVE_PATH
        ),
    ],
    "release_directory_inventory": [
        file_record(
            RELEASE_DIRECTORY
            / Path(relative_path),
            relative_to=RELEASE_DIRECTORY,
        )
        for relative_path in sorted(
            EXPECTED_DIRECTORY_FILES
        )
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_VERIFICATION
        ),
        file_record(
            OUTPUT_FINAL_LOCK
        ),
    ],
    "postlock_checks": (
        postlock_checks
    ),
    "release_directory_tree_sha256": (
        tree_hash
    ),
    "zip_archive_sha256": (
        zip_hash
    ),
    "release_locked": True,
    "ready_for_distribution": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print()
print("=" * 92)
print("PHASE 10 RELEASE ARCHIVE VERIFICATION SUMMARY")
print("=" * 92)
print(
    "Release directory files         : 16"
)
print(
    "Manifest payload files          : 15"
)
print(
    "Payload hashes verified         : True"
)
print(
    "Locked source copies verified   : True"
)
print(
    "ZIP entry names exact           : True"
)
print(
    "ZIP entry order lexicographic   : True"
)
print(
    "ZIP timestamps fixed            : True"
)
print(
    "ZIP permissions fixed           : True"
)
print(
    "ZIP contents match directory    : True"
)
print(
    "Deterministic ZIP rebuild exact : True"
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
    "External final lock created     : True"
)
print(
    "Release status                  : LOCKED"
)
print(
    "Ready for distribution          : True"
)
print(
    "All checks passed               : True"
)
print(
    "ZIP SHA-256                     : "
    f"{zip_hash}"
)
print(
    "Release tree SHA-256            : "
    f"{tree_hash}"
)
print(
    "PHASE 10 RELEASE ARCHIVE VERIFIED AND LOCKED"
)
