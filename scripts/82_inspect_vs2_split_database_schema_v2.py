from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_seed2026.sqlite"
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def compact(value: Any) -> str:
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"

    text = repr(value)

    if len(text) > 180:
        return text[:177] + "..."

    return text


print("=" * 88)
print("VS2 SPLIT DATABASE SCHEMA INSPECTION")
print("=" * 88)
print(f"Database: {DATABASE}")
print(f"Exists  : {DATABASE.exists()}")

if not DATABASE.exists():
    raise FileNotFoundError(DATABASE)

print(f"Bytes   : {DATABASE.stat().st_size:,}")

with sqlite3.connect(DATABASE) as connection:
    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    print(f"Check   : {quick_check}")

    if quick_check.lower() != "ok":
        raise RuntimeError(
            f"SQLite quick_check failed: {quick_check}"
        )

    tables = connection.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    print("")
    print("=" * 88)
    print("TABLE INVENTORY")
    print("=" * 88)
    print(f"Table count: {len(tables)}")

    table_names = []

    for table_name, create_sql in tables:
        table_names.append(str(table_name))
        identifier = quote_identifier(str(table_name))

        row_count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {identifier}"
            ).fetchone()[0]
        )

        columns = connection.execute(
            f"PRAGMA table_info({identifier})"
        ).fetchall()

        indexes = connection.execute(
            f"PRAGMA index_list({identifier})"
        ).fetchall()

        print("")
        print("-" * 88)
        print(f"TABLE: {table_name}")
        print(f"ROWS : {row_count:,}")
        print("CREATE SQL:")
        print(create_sql)

        print("COLUMNS:")
        for column in columns:
            print(
                f"  cid={column[0]} | name={column[1]} "
                f"| type={column[2]} | notnull={column[3]} "
                f"| default={column[4]} | pk={column[5]}"
            )

        print("INDEXES:")
        if indexes:
            for index in indexes:
                print(f"  {index}")
        else:
            print("  none")

        print("SAMPLE ROWS:")
        sample_rows = connection.execute(
            f"SELECT * FROM {identifier} LIMIT 3"
        ).fetchall()

        if not sample_rows:
            print("  none")

        for row in sample_rows:
            print(
                "  ("
                + ", ".join(compact(value) for value in row)
                + ")"
            )

    if "metadata" in table_names:
        print("")
        print("=" * 88)
        print("METADATA")
        print("=" * 88)

        metadata_rows = connection.execute(
            """
            SELECT *
            FROM metadata
            ORDER BY 1
            """
        ).fetchall()

        for row in metadata_rows:
            print(
                " | ".join(
                    compact(value)
                    for value in row
                )
            )

print("")
print("VS2 SPLIT DATABASE SCHEMA INSPECTION COMPLETED")
