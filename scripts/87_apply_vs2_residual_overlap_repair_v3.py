"""Apply the residual VS2 scaled-overlap repair plan to a new protocol-v2.3 database.

Safety properties:
- never modifies the protocol-v2.2 parent database,
- validates the residual plan before writing,
- copies the parent database through SQLite backup,
- changes only fingerprint_stats.grouped_split for the four planned groups,
- preserves global and per-family split counts exactly,
- keeps the earlier v2.2 repair provenance,
- records the residual repair in a separate audit table,
- writes the final database atomically,
- does not rebuild cache and does not start training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v2.sqlite"
)
DEFAULT_OUTPUT_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v3.sqlite"
)
DEFAULT_PLAN_CSV = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_residual_overlap_repair_plan_v3.csv"
)
DEFAULT_PLAN_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "vs2_residual_overlap_repair_plan_summary_v3.json"
)
DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "vs2_ablation_repair_v3"
)
DEFAULT_NOTE = (
    PROJECT_ROOT
    / "docs"
    / "v2"
    / "vs2"
    / "VS2_RESIDUAL_SCALED_OVERLAP_REPAIR_V3.md"
)

PROTOCOL_NAME = "vs2_float32_leakage_ablation"
PARENT_PROTOCOL_VERSION = "2.2"
REPAIRED_PROTOCOL_VERSION = "2.3"

EXPECTED_COMPONENT_COUNT = 4
EXPECTED_PLAN_GROUP_COUNT = 12
EXPECTED_AFFECTED_RAW_ROWS = 12
EXPECTED_MOVED_GROUP_COUNT = 4
EXPECTED_MOVED_RAW_ROWS = 4

SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("benign", "gafgyt", "mirai")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except Exception as exc:
        raise RuntimeError(
            f"Invalid integer in residual repair plan: {key}={row.get(key)!r}"
        ) from exc


def safe_unlink(path: Path) -> None:
    if not path.exists():
        return

    last_error: Exception | None = None
    for attempt in range(10):
        try:
            path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))

    if last_error is not None:
        raise last_error


def load_and_validate_plan(
    plan_csv: Path,
    plan_summary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(plan_summary_path)

    expected_summary = {
        "status": "completed",
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "planned_protocol_version": REPAIRED_PROTOCOL_VERSION,
        "residual_scaled_collision_component_count": EXPECTED_COMPONENT_COUNT,
        "affected_original_fingerprint_group_count": EXPECTED_PLAN_GROUP_COUNT,
        "affected_raw_row_count": EXPECTED_AFFECTED_RAW_ROWS,
        "label_conflict_component_count": 0,
        "minimum_moved_raw_rows": EXPECTED_MOVED_RAW_ROWS,
        "moved_original_fingerprint_group_count": EXPECTED_MOVED_GROUP_COUNT,
        "planned_cross_split_scaled_overlap_count": 0,
    }
    for key, expected in expected_summary.items():
        observed = summary.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Residual repair summary mismatch: "
                f"{key}={observed!r}, expected={expected!r}"
            )

    rows: list[dict[str, Any]] = []
    with plan_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "scaled_hash_forward",
            "scaled_hash_reverse",
            "original_hash_forward",
            "original_hash_reverse",
            "family_label",
            "class_name",
            "raw_row_count",
            "old_grouped_split_code",
            "new_grouped_split_code",
            "rows_moved",
        }
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"Residual repair plan columns missing: {sorted(missing)}"
            )

        for source_row in reader:
            rows.append(
                {
                    "scaled_hash_forward": parse_int(
                        source_row, "scaled_hash_forward"
                    ),
                    "scaled_hash_reverse": parse_int(
                        source_row, "scaled_hash_reverse"
                    ),
                    "original_hash_forward": parse_int(
                        source_row, "original_hash_forward"
                    ),
                    "original_hash_reverse": parse_int(
                        source_row, "original_hash_reverse"
                    ),
                    "family_label": parse_int(source_row, "family_label"),
                    "class_name": str(source_row["class_name"]),
                    "raw_row_count": parse_int(source_row, "raw_row_count"),
                    "old_grouped_split_code": parse_int(
                        source_row, "old_grouped_split_code"
                    ),
                    "new_grouped_split_code": parse_int(
                        source_row, "new_grouped_split_code"
                    ),
                    "rows_moved": parse_int(source_row, "rows_moved"),
                }
            )

    if len(rows) != EXPECTED_PLAN_GROUP_COUNT:
        raise RuntimeError(
            f"Residual repair plan row count mismatch: "
            f"{len(rows)} != {EXPECTED_PLAN_GROUP_COUNT}"
        )

    original_pairs = {
        (row["original_hash_forward"], row["original_hash_reverse"])
        for row in rows
    }
    if len(original_pairs) != len(rows):
        raise RuntimeError(
            "Residual repair plan contains duplicate original fingerprint pairs."
        )

    components: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["family_label"] not in (0, 1, 2):
            raise RuntimeError(f"Invalid family label: {row}")
        if row["class_name"] != CLASS_NAMES[row["family_label"]]:
            raise RuntimeError(f"Class name/family mismatch: {row}")
        if row["raw_row_count"] <= 0:
            raise RuntimeError(f"Non-positive raw_row_count: {row}")
        if row["old_grouped_split_code"] not in (0, 1, 2):
            raise RuntimeError(f"Invalid old split: {row}")
        if row["new_grouped_split_code"] not in (0, 1, 2):
            raise RuntimeError(f"Invalid new split: {row}")

        expected_moved = (
            row["raw_row_count"]
            if row["old_grouped_split_code"]
            != row["new_grouped_split_code"]
            else 0
        )
        if row["rows_moved"] != expected_moved:
            raise RuntimeError(f"rows_moved is inconsistent: {row}")

        component_key = (
            row["scaled_hash_forward"],
            row["scaled_hash_reverse"],
        )
        components.setdefault(component_key, []).append(row)

    if len(components) != EXPECTED_COMPONENT_COUNT:
        raise RuntimeError(
            f"Residual scaled component count mismatch: "
            f"{len(components)} != {EXPECTED_COMPONENT_COUNT}"
        )

    for component_key, component_rows in components.items():
        new_splits = {
            row["new_grouped_split_code"] for row in component_rows
        }
        old_splits = {
            row["old_grouped_split_code"] for row in component_rows
        }
        family_labels = {row["family_label"] for row in component_rows}

        if len(new_splits) != 1:
            raise RuntimeError(
                f"Residual component has multiple planned splits: "
                f"{component_key} -> {new_splits}"
            )
        if len(old_splits) <= 1:
            raise RuntimeError(
                f"Residual component was not cross-split: "
                f"{component_key} -> {old_splits}"
            )
        if len(family_labels) != 1:
            raise RuntimeError(
                f"Residual component has label conflict: "
                f"{component_key} -> {family_labels}"
            )

    affected_rows = sum(row["raw_row_count"] for row in rows)
    moved_rows = sum(row["rows_moved"] for row in rows)
    moved_groups = sum(1 for row in rows if row["rows_moved"] > 0)

    if affected_rows != EXPECTED_AFFECTED_RAW_ROWS:
        raise RuntimeError(
            f"Residual affected row count mismatch: "
            f"{affected_rows} != {EXPECTED_AFFECTED_RAW_ROWS}"
        )
    if moved_rows != EXPECTED_MOVED_RAW_ROWS:
        raise RuntimeError(
            f"Residual moved row count mismatch: "
            f"{moved_rows} != {EXPECTED_MOVED_RAW_ROWS}"
        )
    if moved_groups != EXPECTED_MOVED_GROUP_COUNT:
        raise RuntimeError(
            f"Residual moved group count mismatch: "
            f"{moved_groups} != {EXPECTED_MOVED_GROUP_COUNT}"
        )

    old_by_family_split: dict[tuple[int, int], int] = {}
    new_by_family_split: dict[tuple[int, int], int] = {}

    for row in rows:
        old_key = (
            row["family_label"],
            row["old_grouped_split_code"],
        )
        new_key = (
            row["family_label"],
            row["new_grouped_split_code"],
        )
        old_by_family_split[old_key] = (
            old_by_family_split.get(old_key, 0)
            + row["raw_row_count"]
        )
        new_by_family_split[new_key] = (
            new_by_family_split.get(new_key, 0)
            + row["raw_row_count"]
        )

    if old_by_family_split != new_by_family_split:
        raise RuntimeError(
            "Residual repair does not preserve affected per-family split totals: "
            f"old={old_by_family_split}, new={new_by_family_split}"
        )

    return rows, summary


def read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key,value FROM metadata").fetchall()
    return {str(key): str(value) for key, value in rows}


def grouped_counts(connection: sqlite3.Connection) -> dict[int, int]:
    return {
        int(code): int(count)
        for code, count in connection.execute(
            """
            SELECT grouped_split, SUM(raw_row_count)
            FROM fingerprint_stats
            GROUP BY grouped_split
            ORDER BY grouped_split
            """
        ).fetchall()
    }


def grouped_family_counts(
    connection: sqlite3.Connection,
) -> dict[tuple[int, int], int]:
    return {
        (int(split_code), int(family)): int(count)
        for split_code, family, count in connection.execute(
            """
            SELECT grouped_split, family_label_min, SUM(raw_row_count)
            FROM fingerprint_stats
            GROUP BY grouped_split, family_label_min
            ORDER BY grouped_split, family_label_min
            """
        ).fetchall()
    }


def backup_database(source: Path, temporary: Path) -> None:
    safe_unlink(temporary)
    for suffix in ("-wal", "-shm"):
        safe_unlink(temporary.with_name(temporary.name + suffix))

    copied_pages = 0
    started = time.perf_counter()

    def progress(status: int, remaining: int, total: int) -> None:
        nonlocal copied_pages
        completed = total - remaining
        if completed - copied_pages >= 20_000 or remaining == 0:
            percentage = 100.0 * completed / total if total else 100.0
            print(
                f"[database copy] {completed:,}/{total:,} pages "
                f"({percentage:.1f}%)",
                flush=True,
            )
            copied_pages = completed

    with closing(sqlite3.connect(source)) as source_connection:
        with closing(sqlite3.connect(temporary)) as target_connection:
            source_connection.backup(
                target_connection,
                pages=10_000,
                progress=progress,
                sleep=0.05,
            )

    elapsed = time.perf_counter() - started
    print(
        f"Database copy completed in {elapsed:.1f} seconds.",
        flush=True,
    )


def apply_repair(
    temporary_database: Path,
    plan_rows: list[dict[str, Any]],
    source_database_sha256: str,
    plan_csv_sha256: str,
    plan_summary_sha256: str,
    repair_script_sha256: str,
) -> dict[str, Any]:
    with closing(
        sqlite3.connect(temporary_database, timeout=120.0)
    ) as connection:
        connection.execute("PRAGMA foreign_keys=ON")

        quick_check_before = str(
            connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        )
        if quick_check_before.lower() != "ok":
            raise RuntimeError(
                f"Copied database quick_check failed: "
                f"{quick_check_before}"
            )

        metadata = read_metadata(connection)
        required_metadata = {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PARENT_PROTOCOL_VERSION,
            "feature_dtype": "float32",
            "feature_count": "115",
            "task": "family_3",
            "seed": "2026",
            "validation_status": "passed",
            "source_file_count": "89",
            "scaled_overlap_repair_status": "applied_and_validated",
        }
        for key, expected in required_metadata.items():
            observed = metadata.get(key)
            if observed != expected:
                raise RuntimeError(
                    f"Parent database metadata mismatch: "
                    f"{key}={observed!r}, expected={expected!r}"
                )

        earlier_repair_table_exists = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type='table'
                  AND name='scaled_overlap_repair_plan'
                """
            ).fetchone()[0]
            == 1
        )
        if not earlier_repair_table_exists:
            raise RuntimeError(
                "Parent v2.2 repair audit table is missing."
            )

        earlier_repair_row_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM scaled_overlap_repair_plan"
            ).fetchone()[0]
        )
        if earlier_repair_row_count != 1036:
            raise RuntimeError(
                f"Parent v2.2 repair audit row count mismatch: "
                f"{earlier_repair_row_count} != 1036"
            )

        old_grouped_counts = grouped_counts(connection)
        old_family_counts = grouped_family_counts(connection)

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                scaled_overlap_residual_repair_plan_v3 (
                    scaled_hash_forward INTEGER NOT NULL,
                    scaled_hash_reverse INTEGER NOT NULL,
                    original_hash_forward INTEGER NOT NULL,
                    original_hash_reverse INTEGER NOT NULL,
                    family_label INTEGER NOT NULL,
                    raw_row_count INTEGER NOT NULL,
                    old_grouped_split INTEGER NOT NULL,
                    new_grouped_split INTEGER NOT NULL,
                    rows_moved INTEGER NOT NULL,
                    PRIMARY KEY(
                        original_hash_forward,
                        original_hash_reverse
                    )
                ) WITHOUT ROWID
                """
            )

            existing_rows = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM scaled_overlap_residual_repair_plan_v3
                    """
                ).fetchone()[0]
            )
            if existing_rows != 0:
                raise RuntimeError(
                    "Residual repair audit table is already non-empty."
                )

            connection.executemany(
                """
                INSERT INTO scaled_overlap_residual_repair_plan_v3(
                    scaled_hash_forward,
                    scaled_hash_reverse,
                    original_hash_forward,
                    original_hash_reverse,
                    family_label,
                    raw_row_count,
                    old_grouped_split,
                    new_grouped_split,
                    rows_moved
                )
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        row["scaled_hash_forward"],
                        row["scaled_hash_reverse"],
                        row["original_hash_forward"],
                        row["original_hash_reverse"],
                        row["family_label"],
                        row["raw_row_count"],
                        row["old_grouped_split_code"],
                        row["new_grouped_split_code"],
                        row["rows_moved"],
                    )
                    for row in plan_rows
                ],
            )

            joined_count, mismatch_count = connection.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(
                        CASE
                            WHEN f.raw_row_count != p.raw_row_count
                              OR f.family_label_min != p.family_label
                              OR f.family_label_max != p.family_label
                              OR f.grouped_split != p.old_grouped_split
                            THEN 1 ELSE 0
                        END
                    )
                FROM scaled_overlap_residual_repair_plan_v3 p
                JOIN fingerprint_stats f
                  ON f.hash_forward = p.original_hash_forward
                 AND f.hash_reverse = p.original_hash_reverse
                """
            ).fetchone()

            joined_count = int(joined_count)
            mismatch_count = int(mismatch_count or 0)

            if joined_count != EXPECTED_PLAN_GROUP_COUNT:
                raise RuntimeError(
                    f"Residual plan/database join count mismatch: "
                    f"{joined_count} != {EXPECTED_PLAN_GROUP_COUNT}"
                )
            if mismatch_count != 0:
                raise RuntimeError(
                    f"Residual plan/database mismatches found: "
                    f"{mismatch_count}"
                )

            update_cursor = connection.execute(
                """
                UPDATE fingerprint_stats
                SET grouped_split = (
                    SELECT p.new_grouped_split
                    FROM scaled_overlap_residual_repair_plan_v3 p
                    WHERE
                        p.original_hash_forward =
                            fingerprint_stats.hash_forward
                        AND p.original_hash_reverse =
                            fingerprint_stats.hash_reverse
                )
                WHERE EXISTS (
                    SELECT 1
                    FROM scaled_overlap_residual_repair_plan_v3 p
                    WHERE
                        p.original_hash_forward =
                            fingerprint_stats.hash_forward
                        AND p.original_hash_reverse =
                            fingerprint_stats.hash_reverse
                        AND p.old_grouped_split !=
                            p.new_grouped_split
                )
                """
            )

            updated_group_count = int(update_cursor.rowcount)
            if updated_group_count != EXPECTED_MOVED_GROUP_COUNT:
                raise RuntimeError(
                    f"Updated residual fingerprint group count mismatch: "
                    f"{updated_group_count} != "
                    f"{EXPECTED_MOVED_GROUP_COUNT}"
                )

            post_plan_mismatch_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM scaled_overlap_residual_repair_plan_v3 p
                    JOIN fingerprint_stats f
                      ON f.hash_forward = p.original_hash_forward
                     AND f.hash_reverse = p.original_hash_reverse
                    WHERE f.grouped_split != p.new_grouped_split
                    """
                ).fetchone()[0]
            )
            if post_plan_mismatch_count != 0:
                raise RuntimeError(
                    f"Post-repair residual plan mismatches: "
                    f"{post_plan_mismatch_count}"
                )

            repaired_component_overlap_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT
                            p.scaled_hash_forward,
                            p.scaled_hash_reverse
                        FROM scaled_overlap_residual_repair_plan_v3 p
                        JOIN fingerprint_stats f
                          ON f.hash_forward =
                                p.original_hash_forward
                         AND f.hash_reverse =
                                p.original_hash_reverse
                        GROUP BY
                            p.scaled_hash_forward,
                            p.scaled_hash_reverse
                        HAVING COUNT(
                            DISTINCT f.grouped_split
                        ) > 1
                    )
                    """
                ).fetchone()[0]
            )
            if repaired_component_overlap_count != 0:
                raise RuntimeError(
                    "Residual scaled components still span multiple "
                    f"splits: {repaired_component_overlap_count}"
                )

            new_grouped_counts = grouped_counts(connection)
            new_family_counts = grouped_family_counts(connection)

            if new_grouped_counts != old_grouped_counts:
                raise RuntimeError(
                    f"Global grouped split totals changed: "
                    f"old={old_grouped_counts}, "
                    f"new={new_grouped_counts}"
                )
            if new_family_counts != old_family_counts:
                raise RuntimeError(
                    f"Grouped split/family totals changed: "
                    f"old={old_family_counts}, "
                    f"new={new_family_counts}"
                )

            invalid_grouped = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM fingerprint_stats
                    WHERE grouped_split IS NULL
                       OR grouped_split NOT IN (0,1,2)
                    """
                ).fetchone()[0]
            )
            if invalid_grouped != 0:
                raise RuntimeError(
                    f"Invalid grouped_split values after residual repair: "
                    f"{invalid_grouped}"
                )

            largest_size, largest_split = connection.execute(
                """
                SELECT raw_row_count, grouped_split
                FROM fingerprint_stats
                ORDER BY raw_row_count DESC
                LIMIT 1
                """
            ).fetchone()
            if int(largest_split) != 0:
                raise RuntimeError(
                    f"Largest fingerprint group moved out of train: "
                    f"size={largest_size}, split={largest_split}"
                )

            repair_metadata = {
                "protocol_version": REPAIRED_PROTOCOL_VERSION,
                "validation_status": "passed",
                "generated_at_utc": utc_now(),
                "grouped_assignment_method": (
                    "family-stratified largest-first raw-row balancing; "
                    "protocol-v2.2 post-scaling collision repair; "
                    "protocol-v2.3 residual post-scaling collision repair "
                    "with exact per-family split-total preservation and "
                    "minimum moved rows"
                ),
                "residual_scaled_overlap_repair_status": (
                    "applied_and_validated"
                ),
                "residual_scaled_overlap_repair_parent_database_sha256": (
                    source_database_sha256
                ),
                "residual_scaled_overlap_repair_plan_csv_sha256": (
                    plan_csv_sha256
                ),
                "residual_scaled_overlap_repair_plan_summary_sha256": (
                    plan_summary_sha256
                ),
                "residual_scaled_overlap_repair_script_sha256": (
                    repair_script_sha256
                ),
                "residual_scaled_overlap_repair_component_count": (
                    str(EXPECTED_COMPONENT_COUNT)
                ),
                "residual_scaled_overlap_repair_affected_group_count": (
                    str(EXPECTED_PLAN_GROUP_COUNT)
                ),
                "residual_scaled_overlap_repair_affected_raw_row_count": (
                    str(EXPECTED_AFFECTED_RAW_ROWS)
                ),
                "residual_scaled_overlap_repair_moved_group_count": (
                    str(EXPECTED_MOVED_GROUP_COUNT)
                ),
                "residual_scaled_overlap_repair_moved_raw_row_count": (
                    str(EXPECTED_MOVED_RAW_ROWS)
                ),
                "residual_scaled_overlap_repair_planned_overlap_count": "0",
            }
            connection.executemany(
                """
                INSERT INTO metadata(key,value)
                VALUES (?,?)
                ON CONFLICT(key)
                DO UPDATE SET value=excluded.value
                """,
                list(repair_metadata.items()),
            )

            connection.commit()
        except Exception:
            connection.rollback()
            raise

        quick_check_after = str(
            connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        )
        if quick_check_after.lower() != "ok":
            raise RuntimeError(
                f"Residual-repaired database quick_check failed: "
                f"{quick_check_after}"
            )

        integrity_check = str(
            connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
        )
        if integrity_check.lower() != "ok":
            raise RuntimeError(
                f"Residual-repaired database integrity_check failed: "
                f"{integrity_check}"
            )

        final_metadata = read_metadata(connection)
        if (
            final_metadata.get("protocol_version")
            != REPAIRED_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Residual-repaired protocol_version was not persisted."
            )

        residual_repair_table_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM scaled_overlap_residual_repair_plan_v3
                """
            ).fetchone()[0]
        )

    return {
        "quick_check_before": quick_check_before,
        "quick_check_after": quick_check_after,
        "integrity_check": integrity_check,
        "old_grouped_split_counts": {
            SPLITS[code]: old_grouped_counts[code]
            for code in range(3)
        },
        "new_grouped_split_counts": {
            SPLITS[code]: new_grouped_counts[code]
            for code in range(3)
        },
        "old_grouped_split_family_counts": {
            f"{SPLITS[split_code]}:{CLASS_NAMES[family]}": count
            for (split_code, family), count
            in sorted(old_family_counts.items())
        },
        "new_grouped_split_family_counts": {
            f"{SPLITS[split_code]}:{CLASS_NAMES[family]}": count
            for (split_code, family), count
            in sorted(new_family_counts.items())
        },
        "updated_fingerprint_group_count": updated_group_count,
        "residual_repair_plan_table_row_count": (
            residual_repair_table_count
        ),
        "parent_v2_2_repair_table_row_count": (
            earlier_repair_row_count
        ),
        "post_repair_plan_mismatch_count": (
            post_plan_mismatch_count
        ),
        "planned_residual_component_overlap_count": (
            repaired_component_overlap_count
        ),
        "largest_group_raw_row_count": int(largest_size),
        "largest_group_split": SPLITS[int(largest_split)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the VS2 residual scaled-overlap repair "
            "to a new protocol-v2.3 SQLite database."
        )
    )
    parser.add_argument(
        "--source-database",
        type=Path,
        default=DEFAULT_SOURCE_DATABASE,
    )
    parser.add_argument(
        "--output-database",
        type=Path,
        default=DEFAULT_OUTPUT_DATABASE,
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=DEFAULT_PLAN_CSV,
    )
    parser.add_argument(
        "--plan-summary",
        type=Path,
        default=DEFAULT_PLAN_SUMMARY,
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )
    parser.add_argument(
        "--note",
        type=Path,
        default=DEFAULT_NOTE,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_database = args.source_database.resolve()
    output_database = args.output_database.resolve()
    plan_csv = args.plan_csv.resolve()
    plan_summary_path = args.plan_summary.resolve()
    report_directory = args.report_directory.resolve()
    note_path = args.note.resolve()
    repair_script = Path(__file__).resolve()

    print("=" * 88)
    print("VS2 RESIDUAL SCALED OVERLAP REPAIR DATABASE")
    print("=" * 88)
    print(f"Source database : {source_database}")
    print(f"Output database : {output_database}")
    print(f"Residual plan   : {plan_csv}")

    for required in (
        source_database,
        plan_csv,
        plan_summary_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if source_database == output_database:
        raise RuntimeError(
            "Source and output database paths must differ."
        )

    output_database.parent.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    if output_database.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output database already exists: {output_database}. "
                "Use --overwrite only after reviewing it."
            )
        safe_unlink(output_database)

    temporary_database = output_database.with_suffix(
        output_database.suffix + ".building"
    )
    safe_unlink(temporary_database)

    minimum_free_bytes = (
        source_database.stat().st_size
        + 512 * 1024 * 1024
    )
    free_bytes = shutil.disk_usage(
        output_database.parent
    ).free

    print(f"Free disk bytes : {free_bytes:,}")
    print(f"Minimum needed  : {minimum_free_bytes:,}")

    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            "Insufficient disk space for an atomic residual-repaired "
            "database copy."
        )

    plan_rows, plan_summary = load_and_validate_plan(
        plan_csv=plan_csv,
        plan_summary_path=plan_summary_path,
    )
    print(
        "Residual repair plan validation passed.",
        flush=True,
    )

    source_database_sha256 = sha256(source_database)
    plan_csv_sha256 = sha256(plan_csv)
    plan_summary_sha256 = sha256(plan_summary_path)
    repair_script_sha256 = sha256(repair_script)

    try:
        backup_database(
            source_database,
            temporary_database,
        )
        validation = apply_repair(
            temporary_database=temporary_database,
            plan_rows=plan_rows,
            source_database_sha256=source_database_sha256,
            plan_csv_sha256=plan_csv_sha256,
            plan_summary_sha256=plan_summary_sha256,
            repair_script_sha256=repair_script_sha256,
        )

        os.replace(
            temporary_database,
            output_database,
        )
    except Exception:
        safe_unlink(temporary_database)
        raise

    output_database_sha256 = sha256(output_database)

    report_json = (
        report_directory
        / "vs2_residual_scaled_overlap_repair_database_v3.json"
    )
    report_txt = (
        report_directory
        / "vs2_residual_scaled_overlap_repair_database_v3.txt"
    )

    report = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "protocol": PROTOCOL_NAME,
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "repaired_protocol_version": REPAIRED_PROTOCOL_VERSION,
        "repair_role": (
            "Versioned residual split-database repair before VS2 "
            "training; protocol-v2.2 parent database preserved."
        ),
        "source_database": str(source_database),
        "source_database_sha256": source_database_sha256,
        "output_database": str(output_database),
        "output_database_sha256": output_database_sha256,
        "output_database_bytes": output_database.stat().st_size,
        "plan_csv": str(plan_csv),
        "plan_csv_sha256": plan_csv_sha256,
        "plan_summary": str(plan_summary_path),
        "plan_summary_sha256": plan_summary_sha256,
        "repair_script": str(repair_script),
        "repair_script_sha256": repair_script_sha256,
        "repair_counts": {
            "residual_scaled_collision_components": (
                EXPECTED_COMPONENT_COUNT
            ),
            "affected_original_fingerprint_groups": (
                EXPECTED_PLAN_GROUP_COUNT
            ),
            "affected_raw_rows": EXPECTED_AFFECTED_RAW_ROWS,
            "moved_original_fingerprint_groups": (
                EXPECTED_MOVED_GROUP_COUNT
            ),
            "moved_raw_rows": EXPECTED_MOVED_RAW_ROWS,
        },
        "validation": validation,
        "source_plan_summary": plan_summary,
    }
    atomic_json(report_json, report)

    checks = {
        "parent_v2_2_database_preserved": (
            source_database.exists()
        ),
        "repaired_v2_3_database_exists": (
            output_database.exists()
        ),
        "output_differs_from_parent_sha256": (
            output_database_sha256
            != source_database_sha256
        ),
        "quick_check_ok": (
            validation["quick_check_after"] == "ok"
        ),
        "integrity_check_ok": (
            validation["integrity_check"] == "ok"
        ),
        "exactly_4_groups_updated": (
            validation["updated_fingerprint_group_count"]
            == EXPECTED_MOVED_GROUP_COUNT
        ),
        "residual_repair_table_has_12_rows": (
            validation[
                "residual_repair_plan_table_row_count"
            ]
            == EXPECTED_PLAN_GROUP_COUNT
        ),
        "parent_repair_table_preserved": (
            validation["parent_v2_2_repair_table_row_count"]
            == 1036
        ),
        "global_split_counts_preserved": (
            validation["old_grouped_split_counts"]
            == validation["new_grouped_split_counts"]
        ),
        "split_family_counts_preserved": (
            validation["old_grouped_split_family_counts"]
            == validation["new_grouped_split_family_counts"]
        ),
        "planned_residual_overlap_zero": (
            validation[
                "planned_residual_component_overlap_count"
            ]
            == 0
        ),
        "largest_group_remains_train": (
            validation["largest_group_split"]
            == "train"
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]
    if failed:
        raise RuntimeError(
            f"Final residual repair checks failed: {failed}"
        )

    text_lines = [
        "=" * 88,
        "VS2 RESIDUAL SCALED OVERLAP REPAIR DATABASE",
        "=" * 88,
        "Status                            : completed",
        f"Parent protocol version           : "
        f"{PARENT_PROTOCOL_VERSION}",
        f"Repaired protocol version         : "
        f"{REPAIRED_PROTOCOL_VERSION}",
        f"Residual collision components     : "
        f"{EXPECTED_COMPONENT_COUNT:,}",
        f"Affected fingerprint groups       : "
        f"{EXPECTED_PLAN_GROUP_COUNT:,}",
        f"Moved fingerprint groups          : "
        f"{EXPECTED_MOVED_GROUP_COUNT:,}",
        f"Moved raw rows                    : "
        f"{EXPECTED_MOVED_RAW_ROWS:,}",
        "Planned residual overlap after fix: 0",
        "",
        "GROUPED SPLIT COUNTS (OLD -> NEW)",
    ]

    for split in SPLITS:
        old_value = validation[
            "old_grouped_split_counts"
        ][split]
        new_value = validation[
            "new_grouped_split_counts"
        ][split]
        text_lines.append(
            f"{split}: {old_value:,} -> {new_value:,}"
        )

    text_lines.extend(["", "VALIDATION CHECKS"])
    for key, value in checks.items():
        text_lines.append(f"{key}: {value}")

    text_lines.extend(
        [
            "",
            f"Parent DB : {source_database}",
            f"Output DB : {output_database}",
            f"Report    : {report_json}",
            "",
            "VS2 RESIDUAL SCALED OVERLAP REPAIR "
            "DATABASE COMPLETED",
        ]
    )

    report_txt.write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    note_lines = [
        "# VS2 Residual Scaled-Overlap Repair V3",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Parent split protocol version: "
        f"`{PARENT_PROTOCOL_VERSION}`",
        f"- Repaired split protocol version: "
        f"`{REPAIRED_PROTOCOL_VERSION}`",
        f"- Parent database SHA-256: "
        f"`{source_database_sha256}`",
        f"- Repaired database SHA-256: "
        f"`{output_database_sha256}`",
        f"- Residual collision components repaired: "
        f"`{EXPECTED_COMPONENT_COUNT}`",
        f"- Original fingerprint groups affected: "
        f"`{EXPECTED_PLAN_GROUP_COUNT}`",
        f"- Raw rows moved: `{EXPECTED_MOVED_RAW_ROWS}`",
        "- No cross-family label conflict was present.",
        "- Global split counts were preserved exactly.",
        "- Per-family split counts were preserved exactly.",
        "- The protocol-v2.2 parent database was not modified.",
        "- Cache regeneration and a full final-float32 overlap "
        "verification remain required.",
        "",
        f"Machine-readable report: `{report_json}`",
    ]

    note_path.write_text(
        "\n".join(note_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    for line in text_lines:
        print(line)


if __name__ == "__main__":
    main()
