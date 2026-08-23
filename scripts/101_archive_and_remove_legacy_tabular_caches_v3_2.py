from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

FINAL_MANIFEST = (
    FINAL_CACHE
    / "manifest.json"
)

FINAL_VERIFICATION = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

LEGACY_DIRECTORIES = [
    (
        ROOT
        / "data"
        / "cache"
        / "tabular_baseline_family3_v3_1.building"
    ),
    (
        ROOT
        / "data"
        / "cache"
        / "tabular_baseline_family3_v3_2.building"
    ),
]

ARCHIVE_FINAL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "legacy_cache_evidence_v3_2"
)

ARCHIVE_BUILDING = ARCHIVE_FINAL.with_name(
    ARCHIVE_FINAL.name + ".building"
)

CLEANUP_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "legacy_cache_cleanup_summary_v3_2.json"
)

MAX_EVIDENCE_COPY_BYTES = 128 * 1024 * 1024

EVIDENCE_SUFFIXES = {
    ".json",
    ".sqlite",
    ".csv",
    ".log",
}

EVIDENCE_NAMES = {
    "scaler_mean.npy",
    "scaler_scale.npy",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


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

    os.replace(
        temporary,
        path,
    )


def directory_size_bytes(
    path: Path,
) -> int:
    if not path.exists():
        return 0

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


required_paths = [
    FINAL_CACHE,
    FINAL_MANIFEST,
    FINAL_VERIFICATION,
    *LEGACY_DIRECTORIES,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (
    ARCHIVE_FINAL,
    ARCHIVE_BUILDING,
    CLEANUP_REPORT,
):
    if path.exists():
        raise FileExistsError(
            "Cleanup artifact already exists; "
            f"refusing to overwrite: {path}"
        )

verification = json.loads(
    FINAL_VERIFICATION.read_text(
        encoding="utf-8"
    )
)

final_manifest = json.loads(
    FINAL_MANIFEST.read_text(
        encoding="utf-8"
    )
)

preflight_checks = {
    "verification_status_passed": (
        verification.get("status")
        == "passed"
    ),
    "verification_all_checks_passed": (
        verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "verification_failed_checks_empty": (
        verification.get(
            "failed_checks"
        )
        == []
    ),
    "final_manifest_completed": (
        final_manifest.get("status")
        == "completed"
    ),
    "final_manifest_all_checks_passed": (
        final_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "final_cache_path_matches": (
        Path(
            verification["final_cache"]
        ).resolve()
        == FINAL_CACHE.resolve()
    ),
    "final_manifest_hash_matches_verification": (
        verification.get(
            "cache_manifest_sha256"
        )
        == sha256_file(
            FINAL_MANIFEST
        )
    ),
    "final_scaled_overlap_zero": (
        int(
            final_manifest[
                "scaled_overlap_audit"
            ][
                "cross_split_model_input_fingerprint_count"
            ]
        )
        == 0
    ),
    "final_family_conflicts_zero": (
        int(
            final_manifest[
                "scaled_overlap_audit"
            ][
                "global_family_conflict_count"
            ]
        )
        == 0
    ),
}

failed_preflight = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight:
    raise RuntimeError(
        "Legacy cleanup preflight failed: "
        + ", ".join(
            failed_preflight
        )
    )

free_before = shutil.disk_usage(
    ROOT
).free

legacy_sizes_before = {
    str(path): directory_size_bytes(
        path
    )
    for path in LEGACY_DIRECTORIES
}

ARCHIVE_BUILDING.mkdir(
    parents=True,
)

source_inventory: list[
    dict[str, Any]
] = []

copied_evidence: list[
    dict[str, Any]
] = []

print("=" * 92)
print("ARCHIVE LEGACY CACHE EVIDENCE")
print("=" * 92)

for legacy_index, legacy_directory in enumerate(
    LEGACY_DIRECTORIES,
    start=1,
):
    archive_subdirectory = (
        ARCHIVE_BUILDING
        / legacy_directory.name
    )

    archive_subdirectory.mkdir(
        parents=True,
    )

    files = sorted(
        path
        for path in legacy_directory.rglob("*")
        if path.is_file()
    )

    print(
        f"[inventory] {legacy_index}/"
        f"{len(LEGACY_DIRECTORIES)} | "
        f"{legacy_directory.name} | "
        f"files={len(files):,}"
    )

    for position, source_path in enumerate(
        files,
        start=1,
    ):
        relative_path = source_path.relative_to(
            legacy_directory
        )

        size_bytes = int(
            source_path.stat().st_size
        )

        digest = sha256_file(
            source_path
        )

        should_copy = (
            (
                source_path.suffix.lower()
                in EVIDENCE_SUFFIXES
                or source_path.name
                in EVIDENCE_NAMES
            )
            and size_bytes
            <= MAX_EVIDENCE_COPY_BYTES
        )

        source_inventory.append(
            {
                "legacy_directory": str(
                    legacy_directory
                ),
                "relative_path": str(
                    relative_path
                ),
                "size_bytes": (
                    size_bytes
                ),
                "sha256": digest,
                "copied_to_evidence_archive": (
                    should_copy
                ),
            }
        )

        if should_copy:
            destination_path = (
                archive_subdirectory
                / relative_path
            )

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

            copied_digest = sha256_file(
                destination_path
            )

            if copied_digest != digest:
                raise RuntimeError(
                    "Copied evidence hash mismatch: "
                    f"{relative_path}"
                )

            copied_evidence.append(
                {
                    "source": str(
                        source_path
                    ),
                    "destination_relative_path": str(
                        destination_path.relative_to(
                            ARCHIVE_BUILDING
                        )
                    ),
                    "size_bytes": (
                        size_bytes
                    ),
                    "sha256": digest,
                }
            )

        print(
            f"  [{position}/{len(files)}] "
            f"{relative_path} | "
            f"{size_bytes / (1024**2):.3f} MiB | "
            f"copied={should_copy}",
            flush=True,
        )

archive_manifest = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "purpose": (
        "Preserve compact audit evidence and "
        "cryptographic inventory before deleting "
        "failed/intermediate cache directories."
    ),
    "final_cache": str(
        FINAL_CACHE
    ),
    "final_cache_manifest": str(
        FINAL_MANIFEST
    ),
    "final_cache_manifest_sha256": (
        sha256_file(
            FINAL_MANIFEST
        )
    ),
    "final_verification_report": str(
        FINAL_VERIFICATION
    ),
    "final_verification_report_sha256": (
        sha256_file(
            FINAL_VERIFICATION
        )
    ),
    "legacy_directories": [
        str(path)
        for path in LEGACY_DIRECTORIES
    ],
    "legacy_sizes_before_bytes": (
        legacy_sizes_before
    ),
    "source_file_inventory": (
        source_inventory
    ),
    "copied_evidence": (
        copied_evidence
    ),
    "copied_evidence_count": len(
        copied_evidence
    ),
    "preflight_checks": (
        preflight_checks
    ),
    "all_checks_passed": True,
}

atomic_json(
    ARCHIVE_BUILDING
    / "evidence_manifest.json",
    archive_manifest,
)

ARCHIVE_BUILDING.rename(
    ARCHIVE_FINAL
)

print()
print("=" * 92)
print("DELETE VERIFIED LEGACY CACHE DIRECTORIES")
print("=" * 92)

for legacy_directory in LEGACY_DIRECTORIES:
    size_gib = (
        legacy_sizes_before[
            str(legacy_directory)
        ]
        / (1024**3)
    )

    print(
        f"[delete] {legacy_directory} | "
        f"{size_gib:.3f} GiB",
        flush=True,
    )

    shutil.rmtree(
        legacy_directory
    )

    if legacy_directory.exists():
        raise RuntimeError(
            "Legacy directory still exists "
            "after deletion: "
            f"{legacy_directory}"
        )

free_after = shutil.disk_usage(
    ROOT
).free

freed_bytes = (
    free_after - free_before
)

post_checks = {
    "final_cache_still_exists": (
        FINAL_CACHE.exists()
    ),
    "final_manifest_still_exists": (
        FINAL_MANIFEST.exists()
    ),
    "evidence_archive_exists": (
        ARCHIVE_FINAL.exists()
    ),
    "evidence_manifest_exists": (
        (
            ARCHIVE_FINAL
            / "evidence_manifest.json"
        ).exists()
    ),
    "all_legacy_directories_removed": all(
        not path.exists()
        for path in LEGACY_DIRECTORIES
    ),
    "final_manifest_hash_unchanged": (
        verification.get(
            "cache_manifest_sha256"
        )
        == sha256_file(
            FINAL_MANIFEST
        )
    ),
}

failed_post = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post:
    raise RuntimeError(
        "Legacy cleanup post-check failed: "
        + ", ".join(
            failed_post
        )
    )

cleanup_report = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "final_cache": str(
        FINAL_CACHE
    ),
    "evidence_archive": str(
        ARCHIVE_FINAL
    ),
    "legacy_directories_removed": [
        str(path)
        for path in LEGACY_DIRECTORIES
    ],
    "legacy_sizes_before_bytes": (
        legacy_sizes_before
    ),
    "free_disk_before_bytes": (
        free_before
    ),
    "free_disk_after_bytes": (
        free_after
    ),
    "observed_freed_bytes": (
        freed_bytes
    ),
    "preflight_checks": (
        preflight_checks
    ),
    "post_checks": (
        post_checks
    ),
    "all_checks_passed": True,
}

atomic_json(
    CLEANUP_REPORT,
    cleanup_report,
)

print()
print("=" * 92)
print("LEGACY CACHE CLEANUP SUMMARY")
print("=" * 92)
print(
    "Evidence files archived : "
    f"{len(copied_evidence):,}"
)
print(
    "Legacy directories      : "
    f"{len(LEGACY_DIRECTORIES):,}"
)
print(
    "Legacy size removed     : "
    f"{sum(legacy_sizes_before.values()) / (1024**3):.3f} GiB"
)
print(
    "Observed free-space gain: "
    f"{freed_bytes / (1024**3):.3f} GiB"
)
print(
    "Free disk before       : "
    f"{free_before / (1024**3):.3f} GiB"
)
print(
    "Free disk after        : "
    f"{free_after / (1024**3):.3f} GiB"
)
print(
    "Final cache preserved  : True"
)
print(
    "Evidence archive       : "
    f"{ARCHIVE_FINAL}"
)
print(
    "Cleanup report         : "
    f"{CLEANUP_REPORT}"
)
print(
    "All checks passed      : True"
)
print(
    "LEGACY CACHE CLEANUP COMPLETED"
)
