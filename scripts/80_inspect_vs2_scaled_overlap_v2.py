from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path.cwd()

AUDIT_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vs2_float32_leakage_ablation_cache_v2"
    / "grouped_float32"
    / "scaled_model_input_overlap_audit.sqlite"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vs2_float32_leakage_ablation_cache_v2"
)

STAGE38_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "38_run_b0_matched_leakage_ablation.py"
)


def quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


print("=" * 88)
print("VS2 SCALED OVERLAP INSPECTION")
print("=" * 88)

print(f"Audit database : {AUDIT_DATABASE}")
print(f"Database exists: {AUDIT_DATABASE.exists()}")

if not AUDIT_DATABASE.exists():
    raise FileNotFoundError(
        f"Scaled overlap audit database was not found: {AUDIT_DATABASE}"
    )

print(f"Database bytes : {AUDIT_DATABASE.stat().st_size:,}")

with sqlite3.connect(AUDIT_DATABASE) as connection:
    quick_check = connection.execute(
        "PRAGMA quick_check"
    ).fetchone()[0]

    print(f"SQLite check   : {quick_check}")

    tables = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
    ]

    print("")
    print("=" * 88)
    print("DATABASE TABLES")
    print("=" * 88)
    print(f"Table count: {len(tables)}")

    for table in tables:
        identifier = quoted_identifier(table)

        columns = connection.execute(
            f"PRAGMA table_info({identifier})"
        ).fetchall()

        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {identifier}"
        ).fetchone()[0]

        print("")
        print(f"TABLE: {table}")
        print(f"Rows : {row_count:,}")
        print("Columns:")

        for column in columns:
            print(
                f"  name={column[1]} | type={column[2]} "
                f"| notnull={column[3]} | pk={column[5]}"
            )

        sample_rows = connection.execute(
            f"SELECT * FROM {identifier} LIMIT 10"
        ).fetchall()

        print("Sample rows:")

        for row in sample_rows:
            print(f"  {row}")

print("")
print("=" * 88)
print("CACHE MANIFESTS")
print("=" * 88)

for arm in ("naive_row", "grouped_float32"):
    arm_directory = CACHE_ROOT / arm
    manifest = arm_directory / "manifest.json"

    print("")
    print(f"ARM: {arm}")
    print(f"Directory exists: {arm_directory.exists()}")
    print(f"Manifest exists : {manifest.exists()}")

    if arm_directory.exists():
        files = sorted(
            path
            for path in arm_directory.rglob("*")
            if path.is_file()
        )

        print(f"File count      : {len(files)}")
        print(
            "Total bytes     : "
            f"{sum(path.stat().st_size for path in files):,}"
        )

        for path in files:
            relative = path.relative_to(PROJECT_ROOT)
            print(
                f"  {relative} | bytes={path.stat().st_size:,}"
            )

    if manifest.exists():
        manifest_data = json.loads(
            manifest.read_text(encoding="utf-8")
        )

        print("Manifest JSON:")
        print(
            json.dumps(
                manifest_data,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
        )

print("")
print("=" * 88)
print("STAGE 38 OVERLAP FUNCTION")
print("=" * 88)

source_lines = STAGE38_SCRIPT.read_text(
    encoding="utf-8"
).splitlines()

start_line = 500
end_line = 615

for line_number in range(start_line, min(end_line, len(source_lines)) + 1):
    print(
        f"{line_number:04d}: "
        f"{source_lines[line_number - 1]}"
    )

print("")
print("VS2 SCALED OVERLAP INSPECTION COMPLETED")
