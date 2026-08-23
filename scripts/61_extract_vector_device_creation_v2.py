from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

SOURCE_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "44_audit_task_aware_device_overlap_v2.py"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "vector_device_creation_core_v2.txt"
)


TARGET_PATTERN = re.compile(
    r"\b("
    r"vector_device|"
    r"per_device_group|"
    r"source_files|"
    r"file_id|"
    r"device_count"
    r")\b",
    flags=re.IGNORECASE,
)


if not SOURCE_FILE.exists():
    print(
        f"Source file not found: {SOURCE_FILE}"
    )
    sys.exit(1)


lines = SOURCE_FILE.read_text(
    encoding="utf-8-sig",
    errors="replace",
).splitlines()


match_indexes = [
    index
    for index, line in enumerate(lines)
    if TARGET_PATTERN.search(line)
]


windows = []

for index in match_indexes:
    windows.append(
        [
            max(0, index - 28),
            min(
                len(lines) - 1,
                index + 35,
            ),
        ]
    )


merged = []

for start, end in windows:
    if not merged:
        merged.append(
            [start, end]
        )
        continue

    previous = merged[-1]

    if start <= previous[1] + 3:
        previous[1] = max(
            previous[1],
            end,
        )
    else:
        merged.append(
            [start, end]
        )


selected = []

for start, end in merged:
    block_text = "\n".join(
        lines[start:end + 1]
    )

    score = 0

    if re.search(
        r"CREATE\s+(?:TEMP\s+)?TABLE\s+vector_device",
        block_text,
        flags=re.IGNORECASE,
    ):
        score += 100

    if re.search(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+vector_device",
        block_text,
        flags=re.IGNORECASE,
    ):
        score += 100

    if re.search(
        r"CREATE\s+(?:TEMP\s+)?TABLE\s+per_device_group",
        block_text,
        flags=re.IGNORECASE,
    ):
        score += 80

    if re.search(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+per_device_group",
        block_text,
        flags=re.IGNORECASE,
    ):
        score += 80

    score += 10 * len(
        re.findall(
            r"\bvector_device\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 8 * len(
        re.findall(
            r"\bper_device_group\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 4 * len(
        re.findall(
            r"\bfile_id\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    score += 3 * len(
        re.findall(
            r"\bdevice\b",
            block_text,
            flags=re.IGNORECASE,
        )
    )

    selected.append(
        {
            "start": start,
            "end": end,
            "score": score,
        }
    )


selected.sort(
    key=lambda item: (
        -item["score"],
        item["start"],
    )
)

selected = selected[:8]

selected.sort(
    key=lambda item: item["start"]
)


report = [
    "=" * 96,
    "VECTOR_DEVICE / PER_DEVICE_GROUP CREATION CORE",
    "=" * 96,
]


for block_number, block in enumerate(
    selected,
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
print("VECTOR_DEVICE CREATION EXTRACT")
print("=" * 78)
print(f"Source matches : {len(match_indexes)}")
print(f"Selected blocks: {len(selected)}")

for number, block in enumerate(
    selected,
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
print("VECTOR_DEVICE CREATION EXTRACT COMPLETE")
