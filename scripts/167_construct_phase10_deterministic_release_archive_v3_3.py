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

RELEASE_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "release"
)

STAGING_ROOT = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.building"
)

RELEASE_DIRECTORY = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3"
)

ZIP_ARCHIVE_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.zip"
)

ZIP_TEMP_PATH = (
    RELEASE_ROOT
    / "TinyML_IDS_IEEE_release_v3_3.zip.building"
)

EXTERNAL_FINAL_LOCK_PATH = (
    AUDIT
    / "phase10_release_archive_locked_v3_3.json"
)

OUTPUT_CONSTRUCTION_REPORT = (
    AUDIT
    / "phase10_release_archive_construction_v3_3.json"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase10_release_archive_constructed_v3_3.json"
)

PROTOCOL_VERSION = "phase10_release_archive_v3_3"

RELEASE_NAME = "TinyML_IDS_IEEE_release_v3_3"
MODEL_FAMILY = "tinyml_mlp::P50-QAT"
REPRESENTATION = "static_int8"
CANONICAL_SEED = 2026
CLASS_LABELS = [
    "benign",
    "gafgyt",
    "mirai",
]

README_NAME = "README_RELEASE.md"
DATA_NOTICE_NAME = "DATA_NOT_INCLUDED.md"
MANIFEST_NAME = "release_manifest.json"

FIXED_ZIP_TIMESTAMP = (
    1980,
    1,
    1,
    0,
    0,
    0,
)

EXPECTED_PAYLOAD_COUNT = 15
EXPECTED_ARCHIVE_FILE_COUNT = 16

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


def remove_path_with_retry(
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
        "Windows kept the path locked: "
        f"{path}"
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


def atomic_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        text,
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


def normalize_archive_path(
    value: str,
) -> str:
    normalized = (
        value.replace(
            "\\",
            "/",
        )
        .lstrip("/")
    )

    if (
        not normalized
        or normalized.endswith("/")
        or ".." in Path(
            normalized
        ).parts
    ):
        raise RuntimeError(
            "Invalid archive path: "
            f"{value}"
        )

    return normalized


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
        0o100644 << 16
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

            if not source_path.is_file():
                raise FileNotFoundError(
                    source_path
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
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_CONSTRUCTION_REPORT,
    OUTPUT_COMPLETION,
    EXTERNAL_FINAL_LOCK_PATH,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 10 construction or final-lock "
            "artifact already exists; refusing "
            f"to overwrite: {output_path}"
        )

for planned_path in (
    STAGING_ROOT,
    RELEASE_DIRECTORY,
    ZIP_ARCHIVE_PATH,
    ZIP_TEMP_PATH,
):
    if planned_path.exists():
        raise FileExistsError(
            "Phase 10 release output already "
            f"exists: {planned_path}"
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
            "protocol_version"
        )
        == PROTOCOL_VERSION
        and preflight_lock.get(
            "ready_for_release_construction"
        )
        is True
        and preflight_lock.get(
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
    "payload_count_15": (
        int(
            source_manifest[
                "payload_file_count_excluding_manifest"
            ]
        )
        == EXPECTED_PAYLOAD_COUNT
    ),
    "archive_count_16": (
        int(
            source_manifest[
                "archive_file_count_including_manifest"
            ]
        )
        == EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "release_outputs_absent": (
        not STAGING_ROOT.exists()
        and not RELEASE_DIRECTORY.exists()
        and not ZIP_ARCHIVE_PATH.exists()
        and not ZIP_TEMP_PATH.exists()
    ),
    "external_final_lock_absent": (
        not EXTERNAL_FINAL_LOCK_PATH.exists()
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
        "Phase 10 release construction "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

payload_entries = list(
    source_manifest[
        "planned_payload_entries"
    ]
)

if len(payload_entries) != EXPECTED_PAYLOAD_COUNT:
    raise RuntimeError(
        "Payload entry count mismatch."
    )

archive_paths = [
    normalize_archive_path(
        entry["archive_path"]
    )
    for entry in payload_entries
]

if (
    len(archive_paths)
    != len(set(archive_paths))
):
    raise RuntimeError(
        "Duplicate payload archive paths."
    )

generated_names = {
    README_NAME,
    DATA_NOTICE_NAME,
}

observed_generated_names = {
    entry["archive_path"]
    for entry in payload_entries
    if entry["entry_role"]
    == "generated_release_documentation"
}

if observed_generated_names != generated_names:
    raise RuntimeError(
        "Generated documentation plan mismatch."
    )

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

RELEASE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

construction_started_at = utc_now()
construction_succeeded = False

try:
    STAGING_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_count = 0

    for entry in payload_entries:
        archive_path = normalize_archive_path(
            entry["archive_path"]
        )

        destination = (
            STAGING_ROOT
            / Path(archive_path)
        )

        role = entry[
            "entry_role"
        ]

        if role in {
            "locked_deployment_artifact",
            "locked_audit_evidence",
        }:
            source_path = Path(
                entry["source_path"]
            )

            if not source_path.is_file():
                raise FileNotFoundError(
                    source_path
                )

            expected_sha256 = entry[
                "source_sha256"
            ]

            expected_size = int(
                entry[
                    "source_size_bytes"
                ]
            )

            if (
                sha256_file(source_path)
                != expected_sha256
            ):
                raise RuntimeError(
                    "Source SHA-256 mismatch "
                    f"before copy: {source_path}"
                )

            if (
                int(
                    source_path.stat().st_size
                )
                != expected_size
            ):
                raise RuntimeError(
                    "Source byte-size mismatch "
                    f"before copy: {source_path}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copyfile(
                source_path,
                destination,
            )

            if (
                sha256_file(destination)
                != expected_sha256
            ):
                raise RuntimeError(
                    "Copied file hash mismatch: "
                    f"{archive_path}"
                )

            copied_count += 1

        elif (
            role
            == "generated_release_documentation"
        ):
            if archive_path == README_NAME:
                atomic_text(
                    destination,
                    README_TEXT,
                )
            elif (
                archive_path
                == DATA_NOTICE_NAME
            ):
                atomic_text(
                    destination,
                    DATA_NOTICE_TEXT,
                )
            else:
                raise RuntimeError(
                    "Unexpected generated "
                    f"documentation: {archive_path}"
                )

        else:
            raise RuntimeError(
                "Unsupported payload role: "
                f"{role}"
            )

    payload_records = [
        file_record(
            STAGING_ROOT
            / Path(archive_path),
            relative_to=STAGING_ROOT,
        )
        for archive_path in sorted(
            archive_paths
        )
    ]

    if (
        len(payload_records)
        != EXPECTED_PAYLOAD_COUNT
    ):
        raise RuntimeError(
            "Constructed payload count mismatch."
        )

    release_manifest = {
        "status": (
            "constructed_pending_"
            "independent_verification"
        ),
        "phase": 10,
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "release_name": (
            RELEASE_NAME
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
            CLASS_LABELS
        ),
        "manifest_records_itself": False,
        "payload_file_count": (
            EXPECTED_PAYLOAD_COUNT
        ),
        "payload_files": (
            payload_records
        ),
        "raw_dataset_included": False,
        "processed_dataset_included": (
            False
        ),
        "nonselected_checkpoints_included": (
            False
        ),
        "ready_for_independent_verification": (
            True
        ),
        "all_checks_passed": True,
    }

    manifest_path = (
        STAGING_ROOT
        / MANIFEST_NAME
    )

    atomic_json(
        manifest_path,
        release_manifest,
    )

    constructed_files = {
        path.relative_to(
            STAGING_ROOT
        ).as_posix()
        for path in STAGING_ROOT.rglob("*")
        if path.is_file()
    }

    expected_constructed_files = {
        *archive_paths,
        MANIFEST_NAME,
    }

    if (
        constructed_files
        != expected_constructed_files
    ):
        raise RuntimeError(
            "Constructed release file set "
            "does not match protocol."
        )

    replace_with_retry(
        STAGING_ROOT,
        RELEASE_DIRECTORY,
    )

    deterministic_zip(
        RELEASE_DIRECTORY,
        ZIP_TEMP_PATH,
        sorted(
            expected_constructed_files
        ),
    )

    replace_with_retry(
        ZIP_TEMP_PATH,
        ZIP_ARCHIVE_PATH,
    )

    construction_succeeded = True

finally:
    if not construction_succeeded:
        remove_path_with_retry(
            STAGING_ROOT
        )
        remove_path_with_retry(
            ZIP_TEMP_PATH
        )

if not RELEASE_DIRECTORY.is_dir():
    raise RuntimeError(
        "Release directory was not created."
    )

if not ZIP_ARCHIVE_PATH.is_file():
    raise RuntimeError(
        "Release ZIP was not created."
    )

directory_files = {
    path.relative_to(
        RELEASE_DIRECTORY
    ).as_posix()
    for path in RELEASE_DIRECTORY.rglob("*")
    if path.is_file()
}

expected_directory_files = {
    *archive_paths,
    MANIFEST_NAME,
}

if directory_files != expected_directory_files:
    raise RuntimeError(
        "Release directory file set mismatch."
    )

with zipfile.ZipFile(
    ZIP_ARCHIVE_PATH,
    mode="r",
) as archive:
    zip_infos = archive.infolist()

    zip_names = [
        info.filename
        for info in zip_infos
    ]

    zip_name_set = set(
        zip_names
    )

    zip_checks = {
        "entry_count_16": (
            len(zip_infos)
            == EXPECTED_ARCHIVE_FILE_COUNT
        ),
        "entry_names_exact": (
            zip_name_set
            == expected_directory_files
        ),
        "entry_order_lexicographic": (
            zip_names
            == sorted(zip_names)
        ),
        "entry_names_unique": (
            len(zip_names)
            == len(set(zip_names))
        ),
        "timestamps_fixed": all(
            info.date_time
            == FIXED_ZIP_TIMESTAMP
            for info in zip_infos
        ),
        "all_entries_deflated": all(
            info.compress_type
            == zipfile.ZIP_DEFLATED
            for info in zip_infos
        ),
        "testzip_passed": (
            archive.testzip()
            is None
        ),
    }

    extracted_hash_checks: dict[
        str,
        bool,
    ] = {}

    for info in zip_infos:
        release_file = (
            RELEASE_DIRECTORY
            / Path(
                info.filename
            )
        )

        extracted_hash_checks[
            info.filename
        ] = (
            sha256_bytes(
                archive.read(
                    info.filename
                )
            )
            == sha256_file(
                release_file
            )
        )

failed_zip_checks = [
    name
    for name, passed
    in zip_checks.items()
    if not passed
]

failed_extracted_hash_checks = [
    name
    for name, passed
    in extracted_hash_checks.items()
    if not passed
]

if (
    failed_zip_checks
    or failed_extracted_hash_checks
):
    raise RuntimeError(
        "Phase 10 ZIP construction checks "
        "failed: "
        + ", ".join(
            failed_zip_checks
            + failed_extracted_hash_checks
        )
    )

release_manifest_path = (
    RELEASE_DIRECTORY
    / MANIFEST_NAME
)

construction_checks = {
    "copied_locked_source_files_13": (
        copied_count == 13
    ),
    "payload_files_15": (
        len(archive_paths)
        == EXPECTED_PAYLOAD_COUNT
    ),
    "release_directory_files_16": (
        len(directory_files)
        == EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "manifest_present": (
        release_manifest_path.is_file()
    ),
    "zip_present": (
        ZIP_ARCHIVE_PATH.is_file()
    ),
    "zip_checks_passed": all(
        zip_checks.values()
    ),
    "zip_content_hashes_match_directory": all(
        extracted_hash_checks.values()
    ),
    "external_final_lock_absent": (
        not EXTERNAL_FINAL_LOCK_PATH.exists()
    ),
    "release_not_finally_locked": True,
}

failed_construction_checks = [
    name
    for name, passed
    in construction_checks.items()
    if not passed
]

if failed_construction_checks:
    raise RuntimeError(
        "Phase 10 release construction "
        "checks failed: "
        + ", ".join(
            failed_construction_checks
        )
    )

construction_report = {
    "status": "constructed",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "construction_started_at_utc": (
        construction_started_at
    ),
    "constructed_at_utc": utc_now(),
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
    "payload_file_count": (
        EXPECTED_PAYLOAD_COUNT
    ),
    "archive_file_count": (
        EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "release_manifest": (
        file_record(
            release_manifest_path
        )
    ),
    "zip_archive_record": (
        file_record(
            ZIP_ARCHIVE_PATH
        )
    ),
    "release_directory_inventory": [
        file_record(
            RELEASE_DIRECTORY
            / Path(relative_path),
            relative_to=RELEASE_DIRECTORY,
        )
        for relative_path in sorted(
            directory_files
        )
    ],
    "zip_checks": (
        zip_checks
    ),
    "zip_content_hash_checks": (
        extracted_hash_checks
    ),
    "construction_checks": (
        construction_checks
    ),
    "raw_dataset_included": False,
    "processed_dataset_included": (
        False
    ),
    "nonselected_checkpoints_included": (
        False
    ),
    "external_final_lock_created": (
        False
    ),
    "release_locked": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_CONSTRUCTION_REPORT,
    construction_report,
)

completion = {
    "status": "constructed",
    "phase": 10,
    "artifact_name": (
        "deterministic_release_archive"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "constructed_at_utc": utc_now(),
    "release_directory": str(
        RELEASE_DIRECTORY
    ),
    "zip_archive": str(
        ZIP_ARCHIVE_PATH
    ),
    "zip_archive_sha256": (
        sha256_file(
            ZIP_ARCHIVE_PATH
        )
    ),
    "zip_archive_size_bytes": int(
        ZIP_ARCHIVE_PATH.stat().st_size
    ),
    "release_manifest": str(
        release_manifest_path
    ),
    "release_manifest_sha256": (
        sha256_file(
            release_manifest_path
        )
    ),
    "construction_report": str(
        OUTPUT_CONSTRUCTION_REPORT
    ),
    "construction_report_sha256": (
        sha256_file(
            OUTPUT_CONSTRUCTION_REPORT
        )
    ),
    "payload_file_count": (
        EXPECTED_PAYLOAD_COUNT
    ),
    "archive_file_count": (
        EXPECTED_ARCHIVE_FILE_COUNT
    ),
    "external_final_lock_created": (
        False
    ),
    "release_locked": False,
    "ready_for_independent_verification": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

print("=" * 92)
print("PHASE 10 DETERMINISTIC RELEASE ARCHIVE CONSTRUCTION SUMMARY")
print("=" * 92)
print(
    "Release directory               : "
    f"{RELEASE_DIRECTORY}"
)
print(
    "ZIP archive                     : "
    f"{ZIP_ARCHIVE_PATH}"
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
    "Locked source files copied      : 13"
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
    "ZIP entry order lexicographic   : True"
)
print(
    "ZIP timestamps fixed            : True"
)
print(
    "ZIP content matches directory   : True"
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
    "External final lock created     : False"
)
print(
    "Release locked                  : False"
)
print(
    "Ready for independent verify    : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 10 RELEASE ARCHIVE CONSTRUCTED"
)
