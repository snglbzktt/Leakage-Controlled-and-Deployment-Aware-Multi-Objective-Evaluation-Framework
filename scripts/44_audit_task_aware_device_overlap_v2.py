from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "task_aware_device_overlap_v2_1"

EXPECTED_SOURCE_GROUPS = 2_482_676
EXPECTED_RAW_RECORDS = 7_062_606
EXPECTED_DEVICE_COUNT = 9

PROJECT_ROOT = Path.cwd()

SOURCE_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

EXACT_AUDIT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "nbaiot_exact_byte_audit_summary_v2.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

TASK_SUMMARY_CSV = (
    OUTPUT_DIRECTORY
    / "task_aware_conflict_summary_v2.csv"
)

TASK_DETAILS_CSV = (
    OUTPUT_DIRECTORY
    / "task_aware_conflict_details_v2.csv"
)

DEVICE_SUMMARY_CSV = (
    OUTPUT_DIRECTORY
    / "device_fingerprint_lodo_feasibility_v2.csv"
)

DEVICE_CLASS_CSV = (
    OUTPUT_DIRECTORY
    / "device_class_support_v2.csv"
)

DEVICE_PAIR_CSV = (
    OUTPUT_DIRECTORY
    / "device_pair_fingerprint_overlap_v2.csv"
)

DEVICE_COUNT_DISTRIBUTION_CSV = (
    OUTPUT_DIRECTORY
    / "device_count_distribution_v2.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIRECTORY
    / "task_aware_device_audit_summary_v2.json"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_rate(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def map_family(
    original_label: str,
) -> tuple[str, str, str]:
    normalized = (
        original_label
        .strip()
        .casefold()
    )

    if "benign" in normalized:
        binary_label = "benign"
        family_label = "benign"

    elif "gafgyt" in normalized:
        binary_label = "attack"
        family_label = "gafgyt"

    elif "mirai" in normalized:
        binary_label = "attack"
        family_label = "mirai"

    else:
        raise ValueError(
            "Bilinmeyen sınıf etiketi: "
            f"{original_label}"
        )

    multiclass_label = normalized

    return (
        binary_label,
        family_label,
        multiclass_label,
    )


if not SOURCE_DATABASE.exists():
    print(
        "HATA: Kaynak duplicate veritabanı bulunamadı:"
    )
    print(SOURCE_DATABASE)
    sys.exit(1)

if not EXACT_AUDIT_SUMMARY.exists():
    print(
        "HATA: Kilitlenmiş exact audit özeti bulunamadı:"
    )
    print(EXACT_AUDIT_SUMMARY)
    sys.exit(1)

exact_summary = json.loads(
    EXACT_AUDIT_SUMMARY.read_text(
        encoding="utf-8",
    )
)

if exact_summary.get("all_checks_passed") is not True:
    print(
        "HATA: Exact byte audit başarılı değil."
    )
    sys.exit(1)

if (
    exact_summary.get("exact_group_count")
    != EXPECTED_SOURCE_GROUPS
):
    print(
        "HATA: Exact grup sayısı beklenen değerle uyuşmuyor."
    )
    sys.exit(1)

if (
    exact_summary.get("exact_occurrence_count")
    != EXPECTED_RAW_RECORDS
):
    print(
        "HATA: Exact kayıt sayısı beklenen değerle uyuşmuyor."
    )
    sys.exit(1)

print("=" * 78)
print("FAZ 1C — GÖREV-DUYARLI ÇATIŞMA VE CİHAZ ÖRTÜŞME DENETİMİ")
print("=" * 78)
print(f"Kaynak DB       : {SOURCE_DATABASE}")
print(f"Exact audit     : {EXACT_AUDIT_SUMMARY}")
print(f"Beklenen grup   : {EXPECTED_SOURCE_GROUPS:,}")
print(f"Beklenen kayıt  : {EXPECTED_RAW_RECORDS:,}")

started_at = time.perf_counter()

connection = sqlite3.connect(
    str(SOURCE_DATABASE)
)

connection.row_factory = sqlite3.Row

labels = [
    str(row[0])
    for row in connection.execute(
        """
        SELECT DISTINCT class_label
        FROM vector_device
        ORDER BY class_label
        """
    ).fetchall()
]

print()
print("Kaynak sınıf etiketleri:")

for label in labels:
    print(f"- {label}")

connection.execute(
    """
    CREATE TEMP TABLE label_map (
        original_label TEXT PRIMARY KEY,
        binary_label TEXT NOT NULL,
        family3_label TEXT NOT NULL,
        multiclass11_label TEXT NOT NULL
    )
    """
)

for label in labels:
    binary_label, family_label, multiclass_label = (
        map_family(label)
    )

    connection.execute(
        """
        INSERT INTO label_map (
            original_label,
            binary_label,
            family3_label,
            multiclass11_label
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            label,
            binary_label,
            family_label,
            multiclass_label,
        ),
    )

connection.commit()

source_group_summary = connection.execute(
    """
    SELECT
        COUNT(*) AS group_count,
        COALESCE(
            SUM(occurrence_count),
            0
        ) AS occurrence_count,
        MAX(device_count) AS maximum_device_count,
        MAX(class_count) AS maximum_class_count
    FROM vector_groups
    """
).fetchone()

source_group_count = int(
    source_group_summary["group_count"]
)

source_occurrence_count = int(
    source_group_summary["occurrence_count"]
)

maximum_device_count = int(
    source_group_summary["maximum_device_count"]
)

maximum_class_count = int(
    source_group_summary["maximum_class_count"]
)

if source_group_count != EXPECTED_SOURCE_GROUPS:
    raise RuntimeError(
        "vector_groups grup sayısı uyuşmuyor."
    )

if source_occurrence_count != EXPECTED_RAW_RECORDS:
    raise RuntimeError(
        "vector_groups occurrence toplamı uyuşmuyor."
    )

task_specs = {
    "binary": "binary_label",
    "family_3": "family3_label",
    "multiclass_11": "multiclass11_label",
}

task_summary_rows: list[dict[str, Any]] = []
task_detail_rows: list[dict[str, Any]] = []

print()
print("Görev-duyarlı etiket çatışmaları hesaplanıyor...")

for task_name, mapped_column in task_specs.items():
    print(f"- {task_name}")

    summary_query = f"""
        SELECT
            COUNT(*) AS group_count,

            COALESCE(
                SUM(
                    CASE
                        WHEN target_label_count > 1
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS conflict_group_count,

            COALESCE(
                SUM(occurrence_count),
                0
            ) AS occurrence_count,

            COALESCE(
                SUM(
                    CASE
                        WHEN target_label_count > 1
                        THEN occurrence_count
                        ELSE 0
                    END
                ),
                0
            ) AS conflict_occurrence_count,

            MAX(target_label_count)
                AS maximum_target_labels_per_group

        FROM (
            SELECT
                vd.hash_forward,
                vd.hash_reverse,

                COUNT(
                    DISTINCT lm.{mapped_column}
                ) AS target_label_count,

                SUM(
                    vd.occurrence_count
                ) AS occurrence_count

            FROM vector_device AS vd

            INNER JOIN label_map AS lm
                ON
                    vd.class_label
                    = lm.original_label

            GROUP BY
                vd.hash_forward,
                vd.hash_reverse
        )
    """

    task_row = connection.execute(
        summary_query
    ).fetchone()

    group_count = int(
        task_row["group_count"]
    )

    conflict_group_count = int(
        task_row["conflict_group_count"]
    )

    occurrence_count = int(
        task_row["occurrence_count"]
    )

    conflict_occurrence_count = int(
        task_row["conflict_occurrence_count"]
    )

    maximum_target_labels = int(
        task_row[
            "maximum_target_labels_per_group"
        ]
    )

    task_summary_rows.append(
        {
            "task": task_name,
            "group_count": group_count,
            "conflict_group_count":
                conflict_group_count,
            "conflict_group_rate":
                safe_rate(
                    conflict_group_count,
                    group_count,
                ),
            "occurrence_count":
                occurrence_count,
            "conflict_occurrence_count":
                conflict_occurrence_count,
            "conflict_occurrence_rate":
                safe_rate(
                    conflict_occurrence_count,
                    occurrence_count,
                ),
            "maximum_target_labels_per_group":
                maximum_target_labels,
        }
    )

    detail_query = f"""
        SELECT
            vd.hash_forward,
            vd.hash_reverse,

            SUM(
                vd.occurrence_count
            ) AS occurrence_count,

            COUNT(
                DISTINCT vd.device
            ) AS device_count,

            COUNT(
                DISTINCT vd.class_label
            ) AS original_label_count,

            COUNT(
                DISTINCT lm.{mapped_column}
            ) AS target_label_count,

            GROUP_CONCAT(
                DISTINCT vd.class_label
            ) AS original_labels,

            GROUP_CONCAT(
                DISTINCT lm.{mapped_column}
            ) AS target_labels

        FROM vector_device AS vd

        INNER JOIN label_map AS lm
            ON
                vd.class_label
                = lm.original_label

        GROUP BY
            vd.hash_forward,
            vd.hash_reverse

        HAVING
            COUNT(
                DISTINCT lm.{mapped_column}
            ) > 1

        ORDER BY
            occurrence_count DESC,
            vd.hash_forward,
            vd.hash_reverse
    """

    detail_results = connection.execute(
        detail_query
    ).fetchall()

    for detail in detail_results:
        task_detail_rows.append(
            {
                "task": task_name,
                "hash_forward":
                    int(detail["hash_forward"]),
                "hash_reverse":
                    int(detail["hash_reverse"]),
                "occurrence_count":
                    int(detail["occurrence_count"]),
                "device_count":
                    int(detail["device_count"]),
                "original_label_count":
                    int(
                        detail[
                            "original_label_count"
                        ]
                    ),
                "target_label_count":
                    int(
                        detail[
                            "target_label_count"
                        ]
                    ),
                "original_labels":
                    str(
                        detail["original_labels"]
                    ),
                "target_labels":
                    str(
                        detail["target_labels"]
                    ),
            }
        )

write_csv(
    TASK_SUMMARY_CSV,
    task_summary_rows,
    [
        "task",
        "group_count",
        "conflict_group_count",
        "conflict_group_rate",
        "occurrence_count",
        "conflict_occurrence_count",
        "conflict_occurrence_rate",
        "maximum_target_labels_per_group",
    ],
)

write_csv(
    TASK_DETAILS_CSV,
    task_detail_rows,
    [
        "task",
        "hash_forward",
        "hash_reverse",
        "occurrence_count",
        "device_count",
        "original_label_count",
        "target_label_count",
        "original_labels",
        "target_labels",
    ],
)

print()
print("Cihaz bazlı LODO uygunluğu hesaplanıyor...")

device_query = """
    WITH per_device_group AS (
        SELECT
            hash_forward,
            hash_reverse,
            device,
            SUM(
                occurrence_count
            ) AS occurrence_count
        FROM vector_device
        GROUP BY
            hash_forward,
            hash_reverse,
            device
    )

    SELECT
        pdg.device,

        COUNT(*) AS fingerprint_count,

        SUM(
            pdg.occurrence_count
        ) AS raw_occurrence_count,

        SUM(
            CASE
                WHEN vg.device_count = 1
                THEN 1
                ELSE 0
            END
        ) AS novel_fingerprint_count,

        SUM(
            CASE
                WHEN vg.device_count = 1
                THEN pdg.occurrence_count
                ELSE 0
            END
        ) AS novel_occurrence_count,

        SUM(
            CASE
                WHEN vg.device_count > 1
                THEN 1
                ELSE 0
            END
        ) AS train_exposed_fingerprint_count,

        SUM(
            CASE
                WHEN vg.device_count > 1
                THEN pdg.occurrence_count
                ELSE 0
            END
        ) AS train_exposed_occurrence_count,

        MAX(
            vg.device_count
        ) AS maximum_group_device_count

    FROM per_device_group AS pdg

    INNER JOIN vector_groups AS vg
        ON
            pdg.hash_forward
                = vg.hash_forward
            AND
            pdg.hash_reverse
                = vg.hash_reverse

    GROUP BY
        pdg.device

    ORDER BY
        pdg.device
"""

device_rows_raw = connection.execute(
    device_query
).fetchall()

device_summary_rows: list[
    dict[str, Any]
] = []

for row in device_rows_raw:
    fingerprint_count = int(
        row["fingerprint_count"]
    )

    raw_occurrence_count = int(
        row["raw_occurrence_count"]
    )

    novel_fingerprint_count = int(
        row["novel_fingerprint_count"]
    )

    novel_occurrence_count = int(
        row["novel_occurrence_count"]
    )

    train_exposed_fingerprint_count = int(
        row[
            "train_exposed_fingerprint_count"
        ]
    )

    train_exposed_occurrence_count = int(
        row[
            "train_exposed_occurrence_count"
        ]
    )

    device_summary_rows.append(
        {
            "device": str(row["device"]),
            "fingerprint_count":
                fingerprint_count,
            "raw_occurrence_count":
                raw_occurrence_count,
            "novel_fingerprint_count":
                novel_fingerprint_count,
            "novel_fingerprint_rate":
                safe_rate(
                    novel_fingerprint_count,
                    fingerprint_count,
                ),
            "novel_occurrence_count":
                novel_occurrence_count,
            "novel_occurrence_rate":
                safe_rate(
                    novel_occurrence_count,
                    raw_occurrence_count,
                ),
            "train_exposed_fingerprint_count":
                train_exposed_fingerprint_count,
            "train_exposed_fingerprint_rate":
                safe_rate(
                    train_exposed_fingerprint_count,
                    fingerprint_count,
                ),
            "train_exposed_occurrence_count":
                train_exposed_occurrence_count,
            "train_exposed_occurrence_rate":
                safe_rate(
                    train_exposed_occurrence_count,
                    raw_occurrence_count,
                ),
            "maximum_group_device_count":
                int(
                    row[
                        "maximum_group_device_count"
                    ]
                ),
        }
    )

write_csv(
    DEVICE_SUMMARY_CSV,
    device_summary_rows,
    [
        "device",
        "fingerprint_count",
        "raw_occurrence_count",
        "novel_fingerprint_count",
        "novel_fingerprint_rate",
        "novel_occurrence_count",
        "novel_occurrence_rate",
        "train_exposed_fingerprint_count",
        "train_exposed_fingerprint_rate",
        "train_exposed_occurrence_count",
        "train_exposed_occurrence_rate",
        "maximum_group_device_count",
    ],
)

print("Cihaz ve sınıf destekleri hesaplanıyor...")

device_class_query = """
    WITH device_class_group AS (
        SELECT
            device,
            class_label,
            hash_forward,
            hash_reverse,
            SUM(
                occurrence_count
            ) AS occurrence_count
        FROM vector_device
        GROUP BY
            device,
            class_label,
            hash_forward,
            hash_reverse
    )

    SELECT
        device,
        class_label,
        COUNT(*) AS fingerprint_count,
        SUM(
            occurrence_count
        ) AS raw_occurrence_count

    FROM device_class_group

    GROUP BY
        device,
        class_label

    ORDER BY
        device,
        class_label
"""

device_class_rows = [
    {
        "device": str(row["device"]),
        "class_label":
            str(row["class_label"]),
        "fingerprint_count":
            int(row["fingerprint_count"]),
        "raw_occurrence_count":
            int(row["raw_occurrence_count"]),
    }
    for row in connection.execute(
        device_class_query
    ).fetchall()
]

write_csv(
    DEVICE_CLASS_CSV,
    device_class_rows,
    [
        "device",
        "class_label",
        "fingerprint_count",
        "raw_occurrence_count",
    ],
)

print("Cihaz sayısı dağılımı hesaplanıyor...")

device_count_distribution_rows = [
    {
        "device_count":
            int(row["device_count"]),
        "fingerprint_count":
            int(row["fingerprint_count"]),
        "raw_occurrence_count":
            int(row["raw_occurrence_count"]),
    }
    for row in connection.execute(
        """
        SELECT
            device_count,
            COUNT(*) AS fingerprint_count,
            SUM(
                occurrence_count
            ) AS raw_occurrence_count
        FROM vector_groups
        GROUP BY device_count
        ORDER BY device_count
        """
    ).fetchall()
]

write_csv(
    DEVICE_COUNT_DISTRIBUTION_CSV,
    device_count_distribution_rows,
    [
        "device_count",
        "fingerprint_count",
        "raw_occurrence_count",
    ],
)

print("Cihaz çiftleri arasındaki fingerprint örtüşmesi hesaplanıyor...")

device_pair_query = """
    WITH per_device_group AS (
        SELECT
            hash_forward,
            hash_reverse,
            device,
            SUM(
                occurrence_count
            ) AS occurrence_count
        FROM vector_device
        GROUP BY
            hash_forward,
            hash_reverse,
            device
    )

    SELECT
        left_side.device AS device_a,
        right_side.device AS device_b,

        COUNT(*) AS shared_fingerprint_count,

        SUM(
            left_side.occurrence_count
        ) AS device_a_occurrences_on_shared,

        SUM(
            right_side.occurrence_count
        ) AS device_b_occurrences_on_shared

    FROM per_device_group AS left_side

    INNER JOIN per_device_group AS right_side
        ON
            left_side.hash_forward
                = right_side.hash_forward
            AND
            left_side.hash_reverse
                = right_side.hash_reverse
            AND
            left_side.device
                < right_side.device

    GROUP BY
        left_side.device,
        right_side.device

    ORDER BY
        left_side.device,
        right_side.device
"""

device_pair_rows = [
    {
        "device_a": str(
            row["device_a"]
        ),
        "device_b": str(
            row["device_b"]
        ),
        "shared_fingerprint_count":
            int(
                row[
                    "shared_fingerprint_count"
                ]
            ),
        "device_a_occurrences_on_shared":
            int(
                row[
                    "device_a_occurrences_on_shared"
                ]
            ),
        "device_b_occurrences_on_shared":
            int(
                row[
                    "device_b_occurrences_on_shared"
                ]
            ),
    }
    for row in connection.execute(
        device_pair_query
    ).fetchall()
]

write_csv(
    DEVICE_PAIR_CSV,
    device_pair_rows,
    [
        "device_a",
        "device_b",
        "shared_fingerprint_count",
        "device_a_occurrences_on_shared",
        "device_b_occurrences_on_shared",
    ],
)

distinct_device_count = int(
    connection.execute(
        """
        SELECT COUNT(
            DISTINCT device
        )
        FROM vector_device
        """
    ).fetchone()[0]
)

multi_device_summary = connection.execute(
    """
    SELECT
        COUNT(*) AS multi_device_group_count,

        COALESCE(
            SUM(occurrence_count),
            0
        ) AS multi_device_occurrence_count

    FROM vector_groups

    WHERE device_count > 1
    """
).fetchone()

single_device_summary = connection.execute(
    """
    SELECT
        COUNT(*) AS single_device_group_count,

        COALESCE(
            SUM(occurrence_count),
            0
        ) AS single_device_occurrence_count

    FROM vector_groups

    WHERE device_count = 1
    """
).fetchone()

device_raw_total = sum(
    int(row["raw_occurrence_count"])
    for row in device_summary_rows
)

binary_conflicts = next(
    row
    for row in task_summary_rows
    if row["task"] == "binary"
)

family_conflicts = next(
    row
    for row in task_summary_rows
    if row["task"] == "family_3"
)

multiclass_conflicts = next(
    row
    for row in task_summary_rows
    if row["task"] == "multiclass_11"
)

elapsed_seconds = (
    time.perf_counter()
    - started_at
)

summary = {
    "protocol_version":
        PROTOCOL_VERSION,

    "status":
        "completed",

    "completed_at":
        utc_now(),

    "elapsed_seconds":
        round(
            elapsed_seconds,
            3,
        ),

    "exact_audit_all_checks_passed":
        exact_summary.get(
            "all_checks_passed"
        ),

    "source_group_count":
        source_group_count,

    "source_occurrence_count":
        source_occurrence_count,

    "distinct_device_count":
        distinct_device_count,

    "maximum_device_count_per_group":
        maximum_device_count,

    "maximum_original_labels_per_group":
        maximum_class_count,

    "single_device_group_count":
        int(
            single_device_summary[
                "single_device_group_count"
            ]
        ),

    "single_device_occurrence_count":
        int(
            single_device_summary[
                "single_device_occurrence_count"
            ]
        ),

    "multi_device_group_count":
        int(
            multi_device_summary[
                "multi_device_group_count"
            ]
        ),

    "multi_device_occurrence_count":
        int(
            multi_device_summary[
                "multi_device_occurrence_count"
            ]
        ),

    "binary_conflict_group_count":
        int(
            binary_conflicts[
                "conflict_group_count"
            ]
        ),

    "family3_conflict_group_count":
        int(
            family_conflicts[
                "conflict_group_count"
            ]
        ),

    "multiclass11_conflict_group_count":
        int(
            multiclass_conflicts[
                "conflict_group_count"
            ]
        ),

    "device_summary_raw_occurrence_total":
        device_raw_total,

    "task_summary_csv":
        str(TASK_SUMMARY_CSV),

    "task_details_csv":
        str(TASK_DETAILS_CSV),

    "device_summary_csv":
        str(DEVICE_SUMMARY_CSV),

    "device_class_csv":
        str(DEVICE_CLASS_CSV),

    "device_pair_csv":
        str(DEVICE_PAIR_CSV),

    "device_count_distribution_csv":
        str(
            DEVICE_COUNT_DISTRIBUTION_CSV
        ),
}

validation_checks = {
    "source_group_count":
        source_group_count
        == EXPECTED_SOURCE_GROUPS,

    "source_occurrence_count":
        source_occurrence_count
        == EXPECTED_RAW_RECORDS,

    "distinct_device_count":
        distinct_device_count
        == EXPECTED_DEVICE_COUNT,

    "device_raw_occurrence_total":
        device_raw_total
        == EXPECTED_RAW_RECORDS,

    "binary_task_group_count":
        int(
            binary_conflicts[
                "group_count"
            ]
        )
        == EXPECTED_SOURCE_GROUPS,

    "family3_task_group_count":
        int(
            family_conflicts[
                "group_count"
            ]
        )
        == EXPECTED_SOURCE_GROUPS,

    "multiclass11_task_group_count":
        int(
            multiclass_conflicts[
                "group_count"
            ]
        )
        == EXPECTED_SOURCE_GROUPS,

    "exact_audit_valid":
        exact_summary.get(
            "all_checks_passed"
        )
        is True,
}

summary["validation_checks"] = (
    validation_checks
)

summary["all_checks_passed"] = all(
    validation_checks.values()
)

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

connection.close()

print()
print("=" * 78)
print("FAZ 1C ÖZETİ")
print("=" * 78)
print(
    f"Binary conflict groups     : "
    f"{summary['binary_conflict_group_count']:,}"
)
print(
    f"Family-3 conflict groups   : "
    f"{summary['family3_conflict_group_count']:,}"
)
print(
    f"11-class conflict groups   : "
    f"{summary['multiclass11_conflict_group_count']:,}"
)
print(
    f"Single-device groups       : "
    f"{summary['single_device_group_count']:,}"
)
print(
    f"Multi-device groups        : "
    f"{summary['multi_device_group_count']:,}"
)
print(
    f"Distinct devices           : "
    f"{summary['distinct_device_count']}"
)
print(
    f"Elapsed seconds            : "
    f"{summary['elapsed_seconds']}"
)
print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Task summary : {TASK_SUMMARY_CSV}")
print(f"Task details : {TASK_DETAILS_CSV}")
print(f"Device LODO  : {DEVICE_SUMMARY_CSV}")
print(f"Device pairs : {DEVICE_PAIR_CSV}")
print(f"Summary JSON : {SUMMARY_JSON}")

if not summary["all_checks_passed"]:
    print()
    print("FAZ 1C BAŞARISIZ")
    sys.exit(1)

print()
print("FAZ 1C BAŞARILI")
