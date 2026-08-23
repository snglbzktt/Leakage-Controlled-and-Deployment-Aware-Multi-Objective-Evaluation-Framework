from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

DEVICE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "devices"
)

DEVICE_LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_device_ledger_v2.csv"
)

CLEANUP_LEDGER_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "early_lodo"
    / "early_lodo_cache_cleanup_ledger_v2.csv"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "early_lodo_fold_cache_v2"
)

RETAINED_CACHE_FILES = {
    "cache_summary.json",
    "scaler.npz",
}

DELETE_SUFFIXES = {
    ".npy",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_name(
    value: str,
) -> str:
    result = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return result.strip("_")


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def save_json_atomic(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            restval="",
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .casefold()
        == "true"
    )


def unlink_with_retry(
    path: Path,
    attempts: int = 5,
) -> None:
    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            path.unlink()
            return

        except FileNotFoundError:
            return

        except PermissionError as error:
            last_error = error

            gc.collect()

            time.sleep(
                0.5 * attempt
            )

    if last_error is not None:
        raise last_error


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="Danmini_Doorbell",
)

args = parser.parse_args()

held_out_device = str(
    args.device
)

device_safe_name = safe_name(
    held_out_device
)


device_directory = (
    DEVICE_ROOT
    / device_safe_name
)

device_summary_file = (
    device_directory
    / "device_summary.json"
)

cache_manifest_file = (
    device_directory
    / "cache_release_manifest.csv"
)

device_release_manifest_file = (
    device_directory
    / "device_release_manifest.csv"
)

cleanup_record_file = (
    device_directory
    / "cache_cleanup_record.json"
)

cache_directory = (
    CACHE_ROOT
    / device_safe_name
)


required_files = [
    device_summary_file,
    cache_manifest_file,
    device_release_manifest_file,
    DEVICE_LEDGER_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


device_summary = load_json(
    device_summary_file
)

device_ledger_rows = load_csv(
    DEVICE_LEDGER_FILE
)

cache_manifest_rows = load_csv(
    cache_manifest_file
)


matching_device_ledger_rows = [
    row
    for row in device_ledger_rows
    if row["held_out_device"]
    == held_out_device
]


manifest_paths: list[Path] = []
manifest_hashes_match = True
manifest_sizes_match = True
manifest_paths_inside_cache = True


for row in cache_manifest_rows:
    path = (
        PROJECT_ROOT
        / row["relative_path"]
    )

    path = path.resolve()

    manifest_paths.append(path)

    try:
        path.relative_to(
            cache_directory.resolve()
        )

    except ValueError:
        manifest_paths_inside_cache = False

    if path.exists():
        if path.stat().st_size != int(
            row["size_bytes"]
        ):
            manifest_sizes_match = False

        if sha256_file(path) != row["sha256"]:
            manifest_hashes_match = False

    else:
        manifest_sizes_match = False
        manifest_hashes_match = False


actual_cache_files = (
    sorted(
        path.resolve()
        for path
        in cache_directory.iterdir()
        if path.is_file()
    )
    if cache_directory.exists()
    else []
)

manifest_file_set = set(
    manifest_paths
)

actual_file_set = set(
    actual_cache_files
)


removable_files = [
    path
    for path in manifest_paths
    if path.name
    not in RETAINED_CACHE_FILES
    and path.suffix.casefold()
    in DELETE_SUFFIXES
]

retained_files = [
    cache_directory / filename
    for filename in sorted(
        RETAINED_CACHE_FILES
    )
]


already_completed = False

if cleanup_record_file.exists():
    previous_record = load_json(
        cleanup_record_file
    )

    already_completed = (
        previous_record.get("status")
        == "completed"
        and all(
            not path.exists()
            for path in removable_files
        )
        and all(
            path.exists()
            for path in retained_files
        )
    )


if already_completed:
    print(
        "Existing completed cache cleanup "
        "record was found."
    )

    print(
        "EARLY LODO DEVICE CACHE "
        "CLEANUP ALREADY COMPLETED"
    )

    sys.exit(0)


pre_cleanup_checks = {
    "device_summary_complete":
        device_summary.get("status")
        == "complete",

    "device_summary_checks_passed":
        device_summary.get(
            "all_checks_passed"
        )
        is True,

    "three_models_completed":
        int(
            device_summary.get(
                "completed_model_count",
                -1,
            )
        )
        == 3,

    "cache_cleanup_eligible":
        device_summary.get(
            "cache",
            {},
        ).get(
            "cleanup_eligible_after_lock"
        )
        is True,

    "one_device_ledger_row":
        len(
            matching_device_ledger_rows
        )
        == 1,

    "device_ledger_complete":
        len(
            matching_device_ledger_rows
        )
        == 1
        and matching_device_ledger_rows[
            0
        ].get("status")
        == "complete",

    "device_ledger_cleanup_eligible":
        len(
            matching_device_ledger_rows
        )
        == 1
        and parse_bool(
            matching_device_ledger_rows[
                0
            ].get(
                "cache_cleanup_eligible",
                False,
            )
        ),

    "cache_directory_exists":
        cache_directory.exists(),

    "cache_manifest_nonempty":
        len(cache_manifest_rows)
        > 0,

    "manifest_paths_inside_cache":
        manifest_paths_inside_cache,

    "manifest_matches_cache_directory":
        manifest_file_set
        == actual_file_set,

    "manifest_sizes_match":
        manifest_sizes_match,

    "manifest_hashes_match":
        manifest_hashes_match,

    "removable_files_found":
        len(removable_files)
        > 0,

    "retained_files_present":
        all(
            path.exists()
            for path in retained_files
        ),
}


all_pre_cleanup_checks_passed = all(
    pre_cleanup_checks.values()
)


print("=" * 86)
print("EARLY LODO DEVICE CACHE CLEANUP")
print("=" * 86)
print(
    f"Held-out device : "
    f"{held_out_device}"
)
print(
    f"Cache directory : "
    f"{cache_directory}"
)
print(
    f"Manifest files  : "
    f"{len(cache_manifest_rows)}"
)
print(
    f"Files to remove : "
    f"{len(removable_files)}"
)

print()
print("PRE-CLEANUP VALIDATION CHECKS")

for name, passed in (
    pre_cleanup_checks.items()
):
    print(f"{name}: {passed}")


if not all_pre_cleanup_checks_passed:
    print()
    print(
        "EARLY LODO DEVICE CACHE "
        "CLEANUP PREFLIGHT FAILED"
    )

    sys.exit(1)


removable_manifest_rows = [
    row
    for row, path in zip(
        cache_manifest_rows,
        manifest_paths,
    )
    if path in removable_files
]

reclaimed_bytes_expected = sum(
    int(row["size_bytes"])
    for row in removable_manifest_rows
)


pending_record = {
    "protocol_version":
        "early_lodo_cache_cleanup_v2_1",

    "status":
        "verified_pending_removal",

    "verified_at":
        utc_now(),

    "held_out_device":
        held_out_device,

    "cache_directory":
        str(cache_directory),

    "retained_files": [
        str(path)
        for path in retained_files
    ],

    "removed_files": [
        {
            "relative_path":
                row["relative_path"],

            "size_bytes":
                int(
                    row["size_bytes"]
                ),

            "sha256":
                row["sha256"],
        }
        for row in removable_manifest_rows
    ],

    "expected_reclaimed_bytes":
        reclaimed_bytes_expected,

    "device_summary_sha256":
        sha256_file(
            device_summary_file
        ),

    "cache_manifest_sha256":
        sha256_file(
            cache_manifest_file
        ),

    "device_release_manifest_sha256":
        sha256_file(
            device_release_manifest_file
        ),

    "pre_cleanup_validation_checks":
        pre_cleanup_checks,

    "all_pre_cleanup_checks_passed":
        all_pre_cleanup_checks_passed,
}


save_json_atomic(
    cleanup_record_file,
    pending_record,
)


print()
print("Removing rebuildable NPY arrays...")


for index, path in enumerate(
    removable_files,
    start=1,
):
    print(
        f"[{index}/{len(removable_files)}] "
        f"{path.name}"
    )

    unlink_with_retry(path)


post_cleanup_checks = {
    "all_removable_files_removed":
        all(
            not path.exists()
            for path in removable_files
        ),

    "retained_files_still_present":
        all(
            path.exists()
            for path in retained_files
        ),

    "cache_directory_retained":
        cache_directory.exists(),

    "device_summary_unchanged":
        sha256_file(
            device_summary_file
        )
        == pending_record[
            "device_summary_sha256"
        ],

    "cache_manifest_unchanged":
        sha256_file(
            cache_manifest_file
        )
        == pending_record[
            "cache_manifest_sha256"
        ],

    "device_release_manifest_unchanged":
        sha256_file(
            device_release_manifest_file
        )
        == pending_record[
            "device_release_manifest_sha256"
        ],
}


all_post_cleanup_checks_passed = all(
    post_cleanup_checks.values()
)


completed_at = utc_now()


cleanup_record = {
    **pending_record,

    "status":
        (
            "completed"
            if all_post_cleanup_checks_passed
            else "failed"
        ),

    "completed_at":
        completed_at,

    "removed_file_count":
        len(removable_files),

    "reclaimed_bytes":
        reclaimed_bytes_expected,

    "reclaimed_gb":
        (
            reclaimed_bytes_expected
            / (1024 ** 3)
        ),

    "post_cleanup_validation_checks":
        post_cleanup_checks,

    "all_post_cleanup_checks_passed":
        all_post_cleanup_checks_passed,
}


save_json_atomic(
    cleanup_record_file,
    cleanup_record,
)


cleanup_ledger_fieldnames = [
    "held_out_device",
    "status",
    "completed_at",
    "removed_file_count",
    "reclaimed_bytes",
    "reclaimed_gb",
    "retained_files",
    "device_summary_file",
    "cache_manifest_file",
    "cleanup_record_file",
]


cleanup_ledger_rows = (
    load_csv(
        CLEANUP_LEDGER_FILE
    )
    if CLEANUP_LEDGER_FILE.exists()
    else []
)

cleanup_ledger_rows = [
    row
    for row in cleanup_ledger_rows
    if row["held_out_device"]
    != held_out_device
]

cleanup_ledger_rows.append(
    {
        "held_out_device":
            held_out_device,

        "status":
            cleanup_record["status"],

        "completed_at":
            completed_at,

        "removed_file_count":
            len(removable_files),

        "reclaimed_bytes":
            reclaimed_bytes_expected,

        "reclaimed_gb":
            (
                reclaimed_bytes_expected
                / (1024 ** 3)
            ),

        "retained_files":
            "|".join(
                path.name
                for path in retained_files
            ),

        "device_summary_file":
            str(
                device_summary_file.relative_to(
                    PROJECT_ROOT
                )
            ),

        "cache_manifest_file":
            str(
                cache_manifest_file.relative_to(
                    PROJECT_ROOT
                )
            ),

        "cleanup_record_file":
            str(
                cleanup_record_file.relative_to(
                    PROJECT_ROOT
                )
            ),
    }
)

cleanup_ledger_rows.sort(
    key=lambda row:
        row["held_out_device"]
)

write_csv_atomic(
    CLEANUP_LEDGER_FILE,
    cleanup_ledger_rows,
    cleanup_ledger_fieldnames,
)


print()
print("=" * 86)
print("CACHE CLEANUP SUMMARY")
print("=" * 86)
print(
    f"Held-out device : "
    f"{held_out_device}"
)
print(
    f"Removed files   : "
    f"{len(removable_files)}"
)
print(
    f"Reclaimed bytes : "
    f"{reclaimed_bytes_expected:,}"
)
print(
    f"Reclaimed GB    : "
    f"{reclaimed_bytes_expected / (1024 ** 3):.3f}"
)
print(
    "Retained files  : "
    + " | ".join(
        path.name
        for path in retained_files
    )
)

print()
print("POST-CLEANUP VALIDATION CHECKS")

for name, passed in (
    post_cleanup_checks.items()
):
    print(f"{name}: {passed}")

print()
print(
    f"Cleanup record : "
    f"{cleanup_record_file}"
)
print(
    f"Cleanup ledger : "
    f"{CLEANUP_LEDGER_FILE}"
)

if not all_post_cleanup_checks_passed:
    print()
    print(
        "EARLY LODO DEVICE CACHE "
        "CLEANUP FAILED"
    )

    sys.exit(1)

print()
print(
    "EARLY LODO DEVICE CACHE "
    "CLEANUP PASSED"
)
