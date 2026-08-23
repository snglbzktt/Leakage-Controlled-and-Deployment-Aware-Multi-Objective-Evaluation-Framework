from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def get_tables(connection: sqlite3.Connection) -> list[str]:
    query = (
        "SELECT name "
        "FROM sqlite_master "
        "WHERE type = 'table' "
        "ORDER BY name"
    )
    return [str(row[0]) for row in connection.execute(query)]


def get_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    query = f'PRAGMA table_info("{table_name}")'
    return [str(row[1]) for row in connection.execute(query)]


project_root = Path.cwd()
data_root = project_root / "data"

if not data_root.exists():
    print(f"HATA: data klasörü bulunamadı: {data_root}")
    sys.exit(1)

patterns = ("*.sqlite", "*.sqlite3", "*.db")
candidates: list[Path] = []

for pattern in patterns:
    candidates.extend(data_root.rglob(pattern))

candidates = sorted(set(path.resolve() for path in candidates))

if not candidates:
    print("HATA: data klasörü altında SQLite veritabanı bulunamadı.")
    sys.exit(1)

print("=" * 78)
print("FAZ 1 VERİTABANI TANILAMA")
print("=" * 78)
print(f"Proje kökü: {project_root}")
print(f"Bulunan veritabanı sayısı: {len(candidates)}")
print()

database_summaries: list[dict[str, object]] = []
database_with_files_table: Path | None = None

for database_path in candidates:
    size_gb = database_path.stat().st_size / (1024 ** 3)

    print("-" * 78)
    print(f"Veritabanı: {database_path}")
    print(f"Boyut: {size_gb:.3f} GB")

    try:
        connection = sqlite3.connect(str(database_path))
        tables = get_tables(connection)

        print(f"Tablolar: {tables}")

        table_details: dict[str, list[str]] = {}

        for table_name in tables:
            columns = get_columns(connection, table_name)
            table_details[table_name] = columns
            print(f"  {table_name}: {columns}")

        summary: dict[str, object] = {
            "database": str(database_path),
            "size_gb": round(size_gb, 6),
            "tables": tables,
            "table_columns": table_details,
        }

        if "files" in tables:
            database_with_files_table = database_path

            file_columns = table_details["files"]
            file_row_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]

            summary["files_table_row_count"] = int(file_row_count)
            print(f"files tablosu kayıt sayısı: {file_row_count}")

            if "status" in file_columns:
                status_rows = connection.execute(
                    "SELECT status, COUNT(*) "
                    "FROM files "
                    "GROUP BY status "
                    "ORDER BY status"
                ).fetchall()

                status_summary = {
                    str(status): int(count)
                    for status, count in status_rows
                }

                summary["file_statuses"] = status_summary
                print(f"Durumlar: {status_summary}")

            possible_count_columns = (
                "expected_row_count",
                "expected_rows",
                "row_count",
                "inserted_row_count",
                "inserted_rows",
                "processed_row_count",
            )

            for column_name in possible_count_columns:
                if column_name in file_columns:
                    query = (
                        f'SELECT SUM("{column_name}") '
                        "FROM files"
                    )
                    value = connection.execute(query).fetchone()[0]
                    summary[f"sum_{column_name}"] = int(value or 0)
                    print(f"SUM({column_name}): {int(value or 0)}")

        database_summaries.append(summary)
        connection.close()

    except sqlite3.Error as error:
        print(f"SQLite okuma hatası: {error}")

print()
print("=" * 78)

if database_with_files_table is None:
    print("SONUÇ: 'files' tablosu bulunan veritabanı tespit edilemedi.")
    print("Bir sonraki adımda mevcut tablo yapısına göre yol belirlenecek.")
else:
    print("ADAY DENETİM VERİTABANI:")
    print(database_with_files_table)

print("=" * 78)

output_directory = project_root / "results" / "v2" / "audit"
output_directory.mkdir(parents=True, exist_ok=True)

output_file = output_directory / "database_schema_diagnostic_v2.json"

output_file.write_text(
    json.dumps(
        {
            "project_root": str(project_root),
            "candidate_count": len(candidates),
            "database_with_files_table": (
                str(database_with_files_table)
                if database_with_files_table is not None
                else None
            ),
            "databases": database_summaries,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"Tanılama çıktısı: {output_file}")
print("FAZ 1 VERİTABANI TANILAMA TAMAMLANDI")
