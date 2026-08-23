from __future__ import annotations

import ast
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
    / "script44_membership_core_v2.txt"
)


PATTERN = re.compile(
    r"("
    r"CREATE\s+(?:TEMP\s+)?TABLE|"
    r"INSERT\s+INTO|"
    r"SELECT\s+DISTINCT|"
    r"GROUP\s+BY|"
    r"JOIN\s+|"
    r"file_id|"
    r"device_id|"
    r"device_count|"
    r"device_name|"
    r"source_device|"
    r"membership|"
    r"member|"
    r"fingerprint|"
    r"hash_forward|"
    r"hash_reverse|"
    r"first_file_id|"
    r"relative_path|"
    r"class_label|"
    r"file_rows|"
    r"device_rows|"
    r"device_map|"
    r"file_map|"
    r"setdefault|"
    r"Counter|"
    r"defaultdict|"
    r"bitmask|"
    r"mask"
    r")",
    flags=re.IGNORECASE,
)


if not TARGET_FILE.exists():
    print(
        f"Target file not found: "
        f"{TARGET_FILE}"
    )
    sys.exit(1)


source = TARGET_FILE.read_text(
    encoding="utf-8-sig",
    errors="replace",
)

lines = source.splitlines()


# Function inventory
tree = ast.parse(
    source,
    filename=str(TARGET_FILE),
)

functions = []

for node in ast.walk(tree):
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        functions.append(
            {
                "name":
                    node.name,

                "start":
                    node.lineno,

                "end":
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),
            }
        )


# Relevant line windows
matching_indexes = [
    index
    for index, line in enumerate(lines)
    if PATTERN.search(line)
]

windows = []

for index in matching_indexes:
    windows.append(
        (
            max(0, index - 12),
            min(
                len(lines) - 1,
                index + 14,
            ),
        )
    )


# Merge overlapping windows
merged_windows = []

for start, end in sorted(windows):
    if not merged_windows:
        merged_windows.append(
            [start, end]
        )
        continue

    previous = merged_windows[-1]

    if start <= previous[1] + 2:
        previous[1] = max(
            previous[1],
            end,
        )
    else:
        merged_windows.append(
            [start, end]
        )


report = []

report.append(
    "=" * 94
)

report.append(
    "SCRIPT 44 FUNCTION INVENTORY"
)

report.append(
    "=" * 94
)

for function in sorted(
    functions,
    key=lambda item: item["start"],
):
    report.append(
        (
            f"{function['start']:05d}-"
            f"{function['end']:05d} | "
            f"{function['name']}"
        )
    )


report.append("")
report.append(
    "=" * 94
)

report.append(
    "SCRIPT 44 MEMBERSHIP CORE"
)

report.append(
    "=" * 94
)


for block_number, (
    start,
    end,
) in enumerate(
    merged_windows,
    start=1,
):
    report.append("")
    report.append(
        (
            f"--- BLOCK {block_number} | "
            f"LINES {start + 1}-"
            f"{end + 1} ---"
        )
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
print("SCRIPT 44 MEMBERSHIP CORE EXTRACT")
print("=" * 78)
print(
    f"Matching source lines : "
    f"{len(matching_indexes)}"
)
print(
    f"Merged blocks         : "
    f"{len(merged_windows)}"
)
print(
    f"Functions found       : "
    f"{len(functions)}"
)
print()
print(f"Output: {OUTPUT_FILE}")
print()
print("SCRIPT 44 MEMBERSHIP CORE EXTRACT COMPLETE")
