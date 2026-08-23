from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
    / "metadata.parquet"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "nbaiot_exact_byte_audit_v2.sqlite"
)

SCRIPTS_DIRECTORY = (
    PROJECT_ROOT
    / "scripts"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "lodo_membership_source_trace_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "lodo_membership_source_trace_v2.txt"
)


SEARCH_PATTERN = re.compile(
    r"("
    r"source_device_count|"
    r"source_device|"
    r"device_lodo_compact_summary|"
    r"device_family_support|"
    r"device_membership|"
    r"membership|"
    r"device_id|"
    r"source_file_id|"
    r"source_file|"
    r"file_id|"
    r"group_concat|"
    r"json_group|"
    r"distinct.*device"
    r")",
    flags=re.IGNORECASE,
)

CANDIDATE_FILE_PATTERN = re.compile(
    r"(device|membership|source|exact|fingerprint|lodo)",
    flags=re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_value(
    value: Any,
    maximum_length: int = 240,
) -> str:
    if isinstance(value, bytes):
        return (
            f"<bytes length={len(value)}>"
        )

    text = repr(value)

    if len(text) > maximum_length:
        text = (
            text[:maximum_length]
            + "..."
        )

    return text


def quote_identifier(
    name: str,
) -> str:
    return (
        '"'
        + name.replace('"', '""')
        + '"'
    )


required_files = [
    METADATA_FILE,
    DATABASE_FILE,
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


# -------------------------------------------------------------------------
# 1. Metadata schema and samples
# -------------------------------------------------------------------------

parquet_file = pq.ParquetFile(
    METADATA_FILE
)

metadata_schema = [
    {
        "column":
            field.name,

        "type":
            str(field.type),
    }
    for field in parquet_file.schema_arrow
]

metadata_columns = [
    item["column"]
    for item in metadata_schema
]

first_batch = next(
    parquet_file.iter_batches(
        batch_size=10,
        columns=metadata_columns,
    )
)

sample_frame = (
    first_batch
    .to_pandas()
)

metadata_samples = []

for _, row in sample_frame.iterrows():
    metadata_samples.append(
        {
            column:
                safe_value(
                    row[column]
                )
            for column in metadata_columns
        }
    )


# -------------------------------------------------------------------------
# 2. Full SQLite object inventory
# -------------------------------------------------------------------------

database_uri = (
    DATABASE_FILE
    .resolve()
    .as_uri()
    + "?mode=ro"
)

connection = sqlite3.connect(
    database_uri,
    uri=True,
)

sqlite_objects = []

try:
    objects = connection.execute(
        """
        SELECT
            type,
            name,
            sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()

    for object_type, object_name, create_sql in objects:
        quoted_name = quote_identifier(
            str(object_name)
        )

        columns = connection.execute(
            f"PRAGMA table_info({quoted_name})"
        ).fetchall()

        column_details = [
            {
                "position":
                    int(row[0]),

                "name":
                    str(row[1]),

                "type":
                    str(row[2]),

                "not_null":
                    bool(row[3]),

                "primary_key_position":
                    int(row[5]),
            }
            for row in columns
        ]

        indexes = connection.execute(
            f"PRAGMA index_list({quoted_name})"
        ).fetchall()

        index_details = []

        for index_row in indexes:
            index_name = str(
                index_row[1]
            )

            quoted_index = quote_identifier(
                index_name
            )

            index_columns = connection.execute(
                f"PRAGMA index_info({quoted_index})"
            ).fetchall()

            index_details.append(
                {
                    "name":
                        index_name,

                    "unique":
                        bool(index_row[2]),

                    "columns": [
                        str(index_column[2])
                        for index_column
                        in index_columns
                    ],
                }
            )

        samples = []

        try:
            cursor = connection.execute(
                f"SELECT * FROM {quoted_name} LIMIT 3"
            )

            sample_columns = [
                description[0]
                for description
                in cursor.description
            ]

            for sample_row in cursor.fetchall():
                samples.append(
                    {
                        sample_columns[index]:
                            safe_value(value)
                        for index, value
                        in enumerate(sample_row)
                    }
                )

        except Exception as error:
            samples.append(
                {
                    "sample_error":
                        repr(error)
                }
            )

        sqlite_objects.append(
            {
                "type":
                    str(object_type),

                "name":
                    str(object_name),

                "create_sql":
                    create_sql,

                "columns":
                    column_details,

                "indexes":
                    index_details,

                "samples":
                    samples,
            }
        )

finally:
    connection.close()


# -------------------------------------------------------------------------
# 3. Search all project scripts for membership construction
# -------------------------------------------------------------------------

code_locations = []

for script_path in sorted(
    SCRIPTS_DIRECTORY.glob("*.py")
):
    text = script_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    matching_indexes = [
        index
        for index, line in enumerate(lines)
        if SEARCH_PATTERN.search(line)
    ]

    if not matching_indexes:
        continue

    selected_indexes: set[int] = set()

    for index in matching_indexes:
        start = max(
            0,
            index - 4,
        )

        end = min(
            len(lines),
            index + 5,
        )

        selected_indexes.update(
            range(start, end)
        )

    context_lines = []

    previous_index: int | None = None

    for index in sorted(
        selected_indexes
    ):
        if (
            previous_index is not None
            and index > previous_index + 1
        ):
            context_lines.append(
                "... omitted ..."
            )

        context_lines.append(
            f"{index + 1:05d}: "
            f"{lines[index]}"
        )

        previous_index = index

    code_locations.append(
        {
            "relative_path":
                str(
                    script_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "match_count":
                len(matching_indexes),

            "context":
                context_lines,
        }
    )


# -------------------------------------------------------------------------
# 4. Candidate project files
# -------------------------------------------------------------------------

candidate_files = []

search_roots = [
    PROJECT_ROOT
    / "data"
    / "processed",

    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit",

    PROJECT_ROOT
    / "models"
    / "preprocessing",
]

allowed_suffixes = {
    ".csv",
    ".json",
    ".parquet",
    ".sqlite",
    ".db",
    ".npz",
    ".npy",
    ".txt",
}

for root in search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if (
            path.suffix.casefold()
            not in allowed_suffixes
        ):
            continue

        relative_path = str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

        if not CANDIDATE_FILE_PATTERN.search(
            relative_path
        ):
            continue

        candidate_files.append(
            {
                "relative_path":
                    relative_path,

                "size_bytes":
                    path.stat().st_size,
            }
        )


summary = {
    "protocol_version":
        "lodo_membership_source_trace_v2_1",

    "status":
        "completed",

    "completed_at":
        utc_now(),

    "metadata": {
        "path":
            str(METADATA_FILE),

        "row_count":
            int(
                parquet_file
                .metadata
                .num_rows
            ),

        "schema":
            metadata_schema,

        "samples":
            metadata_samples,
    },

    "sqlite": {
        "path":
            str(DATABASE_FILE),

        "object_count":
            len(sqlite_objects),

        "objects":
            sqlite_objects,
    },

    "code_locations":
        code_locations,

    "candidate_files":
        candidate_files,
}


OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report_lines = [
    "=" * 90,
    "PHASE 2B LODO MEMBERSHIP SOURCE TRACE",
    "=" * 90,
    "",
    "METADATA SCHEMA",
]

for item in metadata_schema:
    report_lines.append(
        f"- {item['column']}: "
        f"{item['type']}"
    )

report_lines.extend(
    [
        "",
        "METADATA SAMPLES",
    ]
)

for sample in metadata_samples:
    report_lines.append(
        json.dumps(
            sample,
            ensure_ascii=False,
        )
    )

report_lines.extend(
    [
        "",
        "SQLITE OBJECTS",
    ]
)

for item in sqlite_objects:
    report_lines.append(
        (
            f"- {item['type']} "
            f"{item['name']}"
        )
    )

    report_lines.append(
        "  columns: "
        + " | ".join(
            (
                f"{column['name']}:"
                f"{column['type']}"
            )
            for column
            in item["columns"]
        )
    )

    report_lines.append(
        "  indexes: "
        + json.dumps(
            item["indexes"],
            ensure_ascii=False,
        )
    )

    report_lines.append(
        "  samples: "
        + json.dumps(
            item["samples"],
            ensure_ascii=False,
        )
    )

report_lines.extend(
    [
        "",
        "CODE LOCATIONS",
    ]
)

for item in code_locations:
    report_lines.append(
        (
            f"--- {item['relative_path']} "
            f"| matches={item['match_count']} ---"
        )
    )

    report_lines.extend(
        item["context"]
    )

report_lines.extend(
    [
        "",
        "CANDIDATE FILES",
    ]
)

for item in candidate_files:
    report_lines.append(
        (
            f"- {item['relative_path']} | "
            f"{item['size_bytes']} bytes"
        )
    )


OUTPUT_REPORT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print("=" * 90)
print("PHASE 2B LODO MEMBERSHIP SOURCE TRACE")
print("=" * 90)

print()
print("METADATA SCHEMA")

for item in metadata_schema:
    print(
        f"{item['column']}: "
        f"{item['type']}"
    )

print()
print("SQLITE OBJECTS")

for item in sqlite_objects:
    print(
        f"{item['type']} {item['name']}"
    )

    print(
        "  columns: "
        + " | ".join(
            column["name"]
            for column
            in item["columns"]
        )
    )

print()
print("CODE LOCATIONS")

for item in code_locations:
    print(
        f"{item['relative_path']} | "
        f"matches={item['match_count']}"
    )

print()
print("CANDIDATE FILES")

for item in candidate_files:
    print(
        f"{item['relative_path']} | "
        f"{item['size_bytes']} bytes"
    )

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")
print()
print("PHASE 2B LODO MEMBERSHIP SOURCE TRACE COMPLETE")
