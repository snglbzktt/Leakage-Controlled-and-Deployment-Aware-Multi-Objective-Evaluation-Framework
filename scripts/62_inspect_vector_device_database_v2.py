from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

SCRIPT_44 = (
    PROJECT_ROOT
    / "scripts"
    / "44_audit_task_aware_device_overlap_v2.py"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "vector_device_database_inspection_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "vector_device_database_inspection_v2.txt"
)


EXPECTED_COLUMNS = {
    "hash_forward",
    "hash_reverse",
    "device",
    "class_label",
    "occurrence_count",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".venv",
    ".git",
    "__pycache__",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def quote_identifier(
    name: str,
) -> str:
    return (
        '"'
        + name.replace('"', '""')
        + '"'
    )


def safe_value(
    value: Any,
    maximum_length: int = 180,
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


if not SCRIPT_44.exists():
    print(
        f"Script 44 bulunamadı: {SCRIPT_44}"
    )
    sys.exit(1)


script_text = SCRIPT_44.read_text(
    encoding="utf-8-sig",
    errors="replace",
)

script_lines = script_text.splitlines()


source_database_context = []

for index, line in enumerate(
    script_lines
):
    if "SOURCE_DATABASE" not in line:
        continue

    start = max(
        0,
        index - 5,
    )

    end = min(
        len(script_lines),
        index + 8,
    )

    source_database_context.append(
        [
            f"{line_index + 1:05d}: "
            f"{script_lines[line_index]}"
            for line_index in range(
                start,
                end,
            )
        ]
    )


database_files = []

for suffix in (
    "*.sqlite",
    "*.sqlite3",
    "*.db",
):
    for path in PROJECT_ROOT.rglob(
        suffix
    ):
        if not path.is_file():
            continue

        relative_parts = set(
            path.relative_to(
                PROJECT_ROOT
            ).parts
        )

        if (
            relative_parts
            & EXCLUDED_DIRECTORY_NAMES
        ):
            continue

        database_files.append(path)


database_files = sorted(
    set(database_files)
)


inspection_rows = []

for database_path in database_files:
    database_result: dict[str, Any] = {
        "relative_path":
            str(
                database_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "size_bytes":
            database_path.stat().st_size,

        "open_succeeded":
            False,

        "contains_vector_device":
            False,

        "error":
            None,
    }

    try:
        database_uri = (
            database_path.resolve().as_uri()
            + "?mode=ro"
        )

        connection = sqlite3.connect(
            database_uri,
            uri=True,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        try:
            objects = connection.execute(
                """
                SELECT
                    type,
                    name,
                    sql
                FROM sqlite_master
                WHERE type IN (
                    'table',
                    'view'
                )
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            ).fetchall()

            object_names = {
                str(row["name"])
                for row in objects
            }

            database_result[
                "open_succeeded"
            ] = True

            database_result[
                "object_names"
            ] = sorted(
                object_names
            )

            if (
                "vector_device"
                not in object_names
            ):
                inspection_rows.append(
                    database_result
                )

                continue

            database_result[
                "contains_vector_device"
            ] = True

            table_details = []

            for object_row in objects:
                object_name = str(
                    object_row["name"]
                )

                quoted_name = (
                    quote_identifier(
                        object_name
                    )
                )

                columns = (
                    connection.execute(
                        f"PRAGMA table_info("
                        f"{quoted_name})"
                    ).fetchall()
                )

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

                indexes = (
                    connection.execute(
                        f"PRAGMA index_list("
                        f"{quoted_name})"
                    ).fetchall()
                )

                index_details = []

                for index_row in indexes:
                    index_name = str(
                        index_row[1]
                    )

                    quoted_index = (
                        quote_identifier(
                            index_name
                        )
                    )

                    index_columns = (
                        connection.execute(
                            f"PRAGMA index_info("
                            f"{quoted_index})"
                        ).fetchall()
                    )

                    index_details.append(
                        {
                            "name":
                                index_name,

                            "unique":
                                bool(
                                    index_row[2]
                                ),

                            "columns": [
                                str(
                                    index_column[2]
                                )
                                for index_column
                                in index_columns
                            ],
                        }
                    )

                table_details.append(
                    {
                        "type":
                            str(
                                object_row[
                                    "type"
                                ]
                            ),

                        "name":
                            object_name,

                        "columns":
                            column_details,

                        "indexes":
                            index_details,
                    }
                )

            database_result[
                "table_details"
            ] = table_details

            vector_columns = {
                str(row[1])
                for row in connection.execute(
                    """
                    PRAGMA table_info(
                        "vector_device"
                    )
                    """
                ).fetchall()
            }

            database_result[
                "vector_device_columns"
            ] = sorted(
                vector_columns
            )

            database_result[
                "expected_columns_present"
            ] = (
                EXPECTED_COLUMNS
                .issubset(
                    vector_columns
                )
            )

            vector_row_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM vector_device
                    """
                ).fetchone()[0]
            )

            database_result[
                "vector_device_row_count"
            ] = vector_row_count

            devices = [
                str(row[0])
                for row
                in connection.execute(
                    """
                    SELECT DISTINCT device
                    FROM vector_device
                    ORDER BY device
                    """
                ).fetchall()
            ]

            database_result[
                "devices"
            ] = devices

            database_result[
                "device_count"
            ] = len(devices)

            labels = [
                str(row[0])
                for row
                in connection.execute(
                    """
                    SELECT DISTINCT class_label
                    FROM vector_device
                    ORDER BY class_label
                    """
                ).fetchall()
            ]

            database_result[
                "class_labels"
            ] = labels

            sample_cursor = (
                connection.execute(
                    """
                    SELECT *
                    FROM vector_device
                    ORDER BY
                        hash_forward,
                        hash_reverse,
                        device,
                        class_label
                    LIMIT 8
                    """
                )
            )

            sample_columns = [
                item[0]
                for item
                in sample_cursor.description
            ]

            database_result[
                "vector_device_samples"
            ] = [
                {
                    sample_columns[index]:
                        safe_value(value)
                    for index, value
                    in enumerate(row)
                }
                for row in sample_cursor.fetchall()
            ]

            if (
                "vector_groups"
                in object_names
            ):
                database_result[
                    "vector_groups_row_count"
                ] = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM vector_groups
                        """
                    ).fetchone()[0]
                )

                database_result[
                    "vector_groups_columns"
                ] = [
                    str(row[1])
                    for row
                    in connection.execute(
                        """
                        PRAGMA table_info(
                            "vector_groups"
                        )
                        """
                    ).fetchall()
                ]

            likely_source_tables = [
                name
                for name in object_names
                if any(
                    term in name.casefold()
                    for term in (
                        "file",
                        "source",
                        "manifest",
                        "catalog",
                    )
                )
            ]

            source_table_details = []

            for table_name in sorted(
                likely_source_tables
            ):
                quoted_name = (
                    quote_identifier(
                        table_name
                    )
                )

                columns = [
                    str(row[1])
                    for row
                    in connection.execute(
                        f"PRAGMA table_info("
                        f"{quoted_name})"
                    ).fetchall()
                ]

                samples = []

                try:
                    cursor = (
                        connection.execute(
                            f"SELECT * "
                            f"FROM {quoted_name} "
                            f"LIMIT 5"
                        )
                    )

                    cursor_columns = [
                        item[0]
                        for item
                        in cursor.description
                    ]

                    samples = [
                        {
                            cursor_columns[index]:
                                safe_value(value)
                            for index, value
                            in enumerate(row)
                        }
                        for row
                        in cursor.fetchall()
                    ]

                except Exception as error:
                    samples = [
                        {
                            "error":
                                repr(error)
                        }
                    ]

                source_table_details.append(
                    {
                        "name":
                            table_name,

                        "columns":
                            columns,

                        "samples":
                            samples,
                    }
                )

            database_result[
                "likely_source_tables"
            ] = source_table_details

            inspection_rows.append(
                database_result
            )

        finally:
            connection.close()

    except Exception as error:
        database_result[
            "error"
        ] = repr(error)

        inspection_rows.append(
            database_result
        )


matching_databases = [
    row
    for row in inspection_rows
    if row.get(
        "contains_vector_device"
    )
]


validation_checks = {
    "script44_source_database_context_found":
        len(
            source_database_context
        )
        > 0,

    "database_files_found":
        len(database_files)
        > 0,

    "vector_device_database_found":
        len(matching_databases)
        >= 1,

    "expected_vector_device_columns":
        all(
            row.get(
                "expected_columns_present"
            )
            is True
            for row in matching_databases
        ),

    "nine_devices_detected":
        all(
            int(
                row.get(
                    "device_count",
                    -1,
                )
            )
            == 9
            for row in matching_databases
        ),

    "vector_groups_present":
        all(
            "vector_groups"
            in row.get(
                "object_names",
                []
            )
            for row in matching_databases
        ),
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "vector_device_database_inspection_v2_1",

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "source_database_context":
        source_database_context,

    "database_file_count":
        len(database_files),

    "databases":
        inspection_rows,

    "matching_database_count":
        len(matching_databases),

    "matching_databases":
        matching_databases,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
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


report = [
    "=" * 94,
    "VECTOR_DEVICE DATABASE INSPECTION",
    "=" * 94,
    "",
    "SCRIPT 44 SOURCE_DATABASE CONTEXT",
]

for block in source_database_context:
    report.extend(block)
    report.append("")


report.extend(
    [
        "DATABASE INVENTORY",
    ]
)

for row in inspection_rows:
    report.append(
        (
            f"- {row['relative_path']} | "
            f"size="
            f"{row['size_bytes']} | "
            f"open="
            f"{row['open_succeeded']} | "
            f"vector_device="
            f"{row['contains_vector_device']} | "
            f"error={row['error']}"
        )
    )


for row in matching_databases:
    report.extend(
        [
            "",
            "=" * 94,
            (
                "MATCHING DATABASE: "
                f"{row['relative_path']}"
            ),
            "=" * 94,
            (
                "vector_device rows: "
                f"{row['vector_device_row_count']:,}"
            ),
            (
                "vector_groups rows: "
                f"{row.get('vector_groups_row_count'):,}"
            ),
            (
                "devices: "
                + " | ".join(
                    row["devices"]
                )
            ),
            (
                "class labels: "
                + " | ".join(
                    row["class_labels"]
                )
            ),
            (
                "vector_device columns: "
                + " | ".join(
                    row[
                        "vector_device_columns"
                    ]
                )
            ),
            "",
            "TABLE DETAILS",
        ]
    )

    for table in row[
        "table_details"
    ]:
        report.append(
            (
                f"- {table['type']} "
                f"{table['name']} | "
                "columns="
                + " | ".join(
                    column["name"]
                    for column
                    in table["columns"]
                )
            )
        )

        report.append(
            "  indexes="
            + json.dumps(
                table["indexes"],
                ensure_ascii=False,
            )
        )

    report.extend(
        [
            "",
            "VECTOR_DEVICE SAMPLES",
        ]
    )

    for sample in row[
        "vector_device_samples"
    ]:
        report.append(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
        )

    report.extend(
        [
            "",
            "LIKELY SOURCE TABLES",
        ]
    )

    for table in row[
        "likely_source_tables"
    ]:
        report.append(
            (
                f"- {table['name']} | "
                "columns="
                + " | ".join(
                    table["columns"]
                )
            )
        )

        for sample in table[
            "samples"
        ]:
            report.append(
                "  "
                + json.dumps(
                    sample,
                    ensure_ascii=False,
                )
            )


report.extend(
    [
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    report.append(
        f"{name}: {passed}"
    )


OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("=" * 78)
print("VECTOR_DEVICE DATABASE INSPECTION")
print("=" * 78)
print(
    f"Database files scanned : "
    f"{len(database_files)}"
)
print(
    f"Matching databases     : "
    f"{len(matching_databases)}"
)

print()
print("MATCHING DATABASES")

for row in matching_databases:
    print(
        f"{row['relative_path']} | "
        f"size_gb="
        f"{row['size_bytes'] / (1024 ** 3):.3f} | "
        f"vector_device_rows="
        f"{row['vector_device_row_count']:,} | "
        f"vector_groups_rows="
        f"{row.get('vector_groups_row_count'):,} | "
        f"devices="
        f"{row['device_count']}"
    )

print()
print("VECTOR_DEVICE COLUMNS")

for row in matching_databases:
    print(
        f"{row['relative_path']}:"
    )

    for column in row[
        "vector_device_columns"
    ]:
        print(f"- {column}")

print()
print("DEVICES")

for row in matching_databases:
    for device in row[
        "devices"
    ]:
        print(f"- {device}")

print()
print("CLASS LABELS")

for row in matching_databases:
    for label in row[
        "class_labels"
    ]:
        print(f"- {label}")

print()
print("LIKELY SOURCE TABLES")

for row in matching_databases:
    for table in row[
        "likely_source_tables"
    ]:
        print(
            f"{table['name']} | "
            + " | ".join(
                table["columns"]
            )
        )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")

if not all_checks_passed:
    print()
    print(
        "VECTOR_DEVICE DATABASE "
        "INSPECTION FAILED"
    )
    sys.exit(1)

print()
print(
    "VECTOR_DEVICE DATABASE "
    "INSPECTION PASSED"
)
