from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path.cwd()

TARGET_FILES = [
    PROJECT_ROOT
    / "scripts"
    / "44_audit_task_aware_device_overlap_v2.py",

    PROJECT_ROOT
    / "scripts"
    / "08_materialize_primary_dataset.py",

    PROJECT_ROOT
    / "scripts"
    / "05_audit_nbaiot_duplicates.py",

    PROJECT_ROOT
    / "scripts"
    / "42_audit_exact_fingerprints_v2.py",

    PROJECT_ROOT
    / "scripts"
    / "37_build_float32_leakage_ablation_splits.py",
]

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "lodo_membership_logic_extract_v2.txt"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "lodo_membership_logic_extract_v2.json"
)


SEARCH_TERMS = [
    "source_device",
    "source_device_count",
    "device_id",
    "device_name",
    "device_membership",
    "membership",
    "device_mask",
    "device_set",
    "device_ids",
    "file_id",
    "file_to_device",
    "device_to_id",
    "source_file",
    "source_files",
    "group_concat",
    "distinct",
    "fingerprint_count",
    "novel_fingerprint",
    "train_exposed",
    "held_out",
    "lodo",
    "group_exact",
    "original_occurrence_count",
]


SEARCH_PATTERN = re.compile(
    "|".join(
        re.escape(term)
        for term in SEARCH_TERMS
    ),
    flags=re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


missing_files = [
    str(path)
    for path in TARGET_FILES
    if not path.exists()
]

if missing_files:
    print("Eksik hedef betikler:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


results = []

for path in TARGET_FILES:
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    matched_indexes = [
        index
        for index, line in enumerate(lines)
        if SEARCH_PATTERN.search(line)
    ]

    selected_indexes: set[int] = set()

    for index in matched_indexes:
        start = max(
            0,
            index - 8,
        )

        end = min(
            len(lines),
            index + 9,
        )

        selected_indexes.update(
            range(start, end)
        )

    extracted_lines = []

    previous_index: int | None = None

    for index in sorted(
        selected_indexes
    ):
        if (
            previous_index is not None
            and index > previous_index + 1
        ):
            extracted_lines.append(
                "... omitted ..."
            )

        extracted_lines.append(
            f"{index + 1:05d}: "
            f"{lines[index]}"
        )

        previous_index = index

    results.append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "match_count":
                len(matched_indexes),

            "extracted_line_count":
                len(extracted_lines),

            "lines":
                extracted_lines,
        }
    )


report_lines = [
    "=" * 94,
    "PHASE 2B — LODO MEMBERSHIP LOGIC EXTRACT",
    "=" * 94,
]

for result in results:
    report_lines.extend(
        [
            "",
            "=" * 94,
            (
                f"{result['relative_path']} | "
                f"matches={result['match_count']} | "
                f"extracted="
                f"{result['extracted_line_count']}"
            ),
            "=" * 94,
        ]
    )

    report_lines.extend(
        result["lines"]
    )


OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_REPORT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


OUTPUT_JSON.write_text(
    json.dumps(
        {
            "protocol_version":
                "lodo_membership_logic_extract_v2_1",

            "status":
                "completed",

            "completed_at":
                utc_now(),

            "search_terms":
                SEARCH_TERMS,

            "results":
                results,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("=" * 78)
print("PHASE 2B LODO MEMBERSHIP LOGIC EXTRACT")
print("=" * 78)

for result in results:
    print(
        f"{result['relative_path']} | "
        f"matches={result['match_count']} | "
        f"extracted_lines="
        f"{result['extracted_line_count']}"
    )

print()
print(f"Report: {OUTPUT_REPORT}")
print(f"JSON  : {OUTPUT_JSON}")
print()
print("PHASE 2B LODO MEMBERSHIP LOGIC EXTRACT COMPLETE")
