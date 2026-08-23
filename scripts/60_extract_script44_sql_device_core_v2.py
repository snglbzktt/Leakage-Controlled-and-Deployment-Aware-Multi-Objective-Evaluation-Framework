from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

TARGET_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "44_audit_task_aware_device_overlap_v2.py"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "script44_sql_device_core_v2.txt"
)


PATTERNS = [
    re.compile(
        r"CREATE\s+(?:TEMP\s+)?TABLE",
        re.IGNORECASE,
    ),
    re.compile(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO",
        re.IGNORECASE,
    ),
    re.compile(
        r"SELECT\s+DISTINCT",
        re.IGNORECASE,
    ),
    re.compile(
        r"GROUP\s+BY",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdevice_id\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdevice\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfile_id\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfile_rows\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdevice_rows\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brelative_path\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhash_forward\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhash_reverse\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bgroup_exact\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmembership\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdevice_count\b",
        re.IGNORECASE,
    ),
]


if not TARGET_FILE.exists():
    print(
        f"Target file not found: {TARGET_FILE}"
    )
    sys.exit(1)


source = TARGET_FILE.read_text(
    encoding="utf-8-sig",
    errors="replace",
)

lines = source.splitlines()


matching_indexes = []

for index, line in enumerate(lines):
    if any(
        pattern.search(line)
        for pattern in PATTERNS
    ):
        matching_indexes.append(index)


windows = []

for index in matching_indexes:
    windows.append(
        [
            max(
                0,
                index - 10,
            ),
            min(
                len(lines) - 1,
                index + 16,
            ),
        ]
    )


merged_windows = []

for start, end in sorted(windows):
    if not merged_windows:
        merged_windows.append(
            [start, end]
        )
        continue

    previous = merged_windows[-1]

    if start <= previous[1] + 3:
        previous[1] = max(
            previous[1],
            end,
        )
    else:
        merged_windows.append(
            [start, end]
        )


ranked_blocks = []

for start, end in merged_windows:
    block_text = "\n".join(
        lines[start:end + 1]
    )

    score = 0

    score += 8 * len(
        re.findall(
            r"CREATE\s+(?:TEMP\s+)?TABLE",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 8 * len(
        re.findall(
            r"INSERT\s+(?:OR\s+\w+\s+)?INTO",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 5 * len(
        re.findall(
            r"\bfile_id\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 5 * len(
        re.findall(
            r"\bdevice_id\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 4 * len(
        re.findall(
            r"\bhash_forward\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 4 * len(
        re.findall(
            r"\bhash_reverse\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 4 * len(
        re.findall(
            r"\bgroup_exact\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 3 * len(
        re.findall(
            r"\brelative_path\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 2 * len(
        re.findall(
            r"\bdevice_count\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    ranked_blocks.append(
        {
            "start": start,
            "end": end,
            "score": score,
        }
    )


ranked_blocks.sort(
    key=lambda item: (
        -item["score"],
        item["start"],
    )
)


selected_blocks = ranked_blocks[:12]

selected_blocks.sort(
    key=lambda item: item["start"]
)


report = [
    "=" * 96,
    "SCRIPT 44 SQL / DEVICE MEMBERSHIP CORE",
    "=" * 96,
    "",
    (
        f"Source line count: "
        f"{len(lines)}"
    ),
    (
        f"Matching lines   : "
        f"{len(matching_indexes)}"
    ),
    (
        f"Selected blocks  : "
        f"{len(selected_blocks)}"
    ),
]


for block_number, block in enumerate(
    selected_blocks,
    start=1,
):
    start = block["start"]
    end = block["end"]

    report.extend(
        [
            "",
            "=" * 96,
            (
                f"BLOCK {block_number} | "
                f"LINES {start + 1}-{end + 1} | "
                f"SCORE {block['score']}"
            ),
            "=" * 96,
        ]
    )

    for index in range(
        start,
        end + 1,
    ):
        report.append(
            f"{index + 1:05d}: "
            f"{lines[index]}"
        )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("=" * 78)
print("SCRIPT 44 SQL / DEVICE CORE EXTRACT")
print("=" * 78)
print(
    f"Source lines   : {len(lines)}"
)
print(
    f"Matching lines : {len(matching_indexes)}"
)
print(
    f"Selected blocks: {len(selected_blocks)}"
)

print()
print("SELECTED BLOCKS")

for number, block in enumerate(
    selected_blocks,
    start=1,
):
    print(
        f"Block {number}: "
        f"lines={block['start'] + 1}-"
        f"{block['end'] + 1} | "
        f"score={block['score']}"
    )

print()
print(f"Output: {OUTPUT_FILE}")
print()
print("SCRIPT 44 SQL / DEVICE CORE EXTRACT COMPLETE")
