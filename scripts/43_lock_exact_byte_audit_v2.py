from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path.cwd()

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

SUMMARY_FILE = (
    AUDIT_DIRECTORY
    / "nbaiot_exact_byte_audit_summary_v2.json"
)

CONFLICT_FILE = (
    AUDIT_DIRECTORY
    / "nbaiot_exact_byte_conflicts_v2.csv"
)

DATABASE_FILE = (
    AUDIT_DIRECTORY
    / "nbaiot_exact_byte_audit_v2.sqlite"
)

LOG_FILE = (
    AUDIT_DIRECTORY
    / "exact_byte_audit_run_v2.log"
)

MANIFEST_FILE = (
    AUDIT_DIRECTORY
    / "exact_byte_audit_release_manifest_v2.csv"
)

COMPLETION_FILE = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "PHASE_1B_EXACT_BYTE_AUDIT_COMPLETE.md"
)


def sha256_file(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


required_files = [
    SUMMARY_FILE,
    CONFLICT_FILE,
    DATABASE_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Eksik exact-audit dosyaları:")
    for missing in missing_files:
        print(missing)
    sys.exit(1)

summary = json.loads(
    SUMMARY_FILE.read_text(
        encoding="utf-8",
    )
)

required_checks = {
    "status_completed":
        summary.get("status") == "completed",

    "all_checks_passed":
        summary.get("all_checks_passed") is True,

    "file_count":
        summary.get("completed_file_count") == 89,

    "row_count":
        summary.get("completed_row_count")
        == 7_062_606,

    "exact_group_count":
        summary.get("exact_group_count")
        == 2_482_676,

    "occurrence_count":
        summary.get("exact_occurrence_count")
        == 7_062_606,

    "byte_conflicts":
        summary.get("byte_conflict_group_count")
        == 0,

    "sha256_conflicts":
        summary.get("sha256_conflict_group_count")
        == 0,

    "canonical_byte_length":
        (
            summary.get(
                "minimum_canonical_byte_length"
            )
            == 920
            and
            summary.get(
                "maximum_canonical_byte_length"
            )
            == 920
        ),
}

with CONFLICT_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as handle:
    conflict_reader = csv.reader(handle)
    conflict_rows = list(conflict_reader)

conflict_data_row_count = max(
    0,
    len(conflict_rows) - 1,
)

required_checks[
    "empty_conflict_report"
] = conflict_data_row_count == 0

failed_checks = [
    name
    for name, passed
    in required_checks.items()
    if not passed
]

if failed_checks:
    print("Exact audit kilitlenemedi.")
    print("Başarısız kontroller:")

    for check in failed_checks:
        print(f"- {check}")

    sys.exit(1)

files_to_manifest = [
    DATABASE_FILE,
    SUMMARY_FILE,
    CONFLICT_FILE,
]

if LOG_FILE.exists():
    files_to_manifest.append(LOG_FILE)

manifest_rows = []

print("SHA-256 manifesti oluşturuluyor...")

for file_path in files_to_manifest:
    print(f"Hash hesaplanıyor: {file_path.name}")

    manifest_rows.append(
        {
            "relative_path": str(
                file_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "size_bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
    )

with MANIFEST_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "relative_path",
            "size_bytes",
            "sha256",
        ],
    )

    writer.writeheader()
    writer.writerows(manifest_rows)

completed_at = datetime.now(
    timezone.utc
).isoformat()

completion_text = f"""# Phase 1B — Exact Byte Audit Completed

- Completion time: {completed_at}
- Raw records: 7,062,606
- Source-precision exact vectors: 2,482,676
- Feature count: 115
- Canonical bytes per vector: 920
- Byte-conflict groups: 0
- SHA-256 conflict groups: 0
- Maximum occurrence count: {summary.get("maximum_occurrence_count")}
- All validation checks passed: true

## Interpretation

The original pair of 64-bit vector hashes was independently
validated against canonical little-endian float64 byte sequences
and SHA-256 digests.

No case was found where records assigned to the same original
hash pair contained different canonical bytes.

This result validates source-precision exact-vector grouping.
It does not replace the separate float32 model-input grouping
analysis.

## Artifacts

- `{DATABASE_FILE.relative_to(PROJECT_ROOT)}`
- `{SUMMARY_FILE.relative_to(PROJECT_ROOT)}`
- `{CONFLICT_FILE.relative_to(PROJECT_ROOT)}`
- `{MANIFEST_FILE.relative_to(PROJECT_ROOT)}`
"""

COMPLETION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

COMPLETION_FILE.write_text(
    completion_text,
    encoding="utf-8",
)

print()
print("=" * 72)
print("FAZ 1B KİLİTLENDİ")
print("=" * 72)

for name, passed in required_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Manifest : {MANIFEST_FILE}")
print(f"Tamamlama: {COMPLETION_FILE}")
print("Exact byte audit yapıtları değişmez olarak kaydedildi.")
