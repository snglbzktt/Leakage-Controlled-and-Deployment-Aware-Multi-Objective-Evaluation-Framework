"""
N-BaIoT duplicate yapısının cihaz bazlı analizi.

Bu betik mevcut SQLite duplicate denetim veritabanını kullanarak:

- Özellik vektörlerinin kaç cihazda görüldüğünü hesaplar.
- Her cihazdaki benzersiz özellik vektörü sayısını belirler.
- Yalnızca tek bir cihazda görülen cihaz-özel vektörleri belirler.
- Diğer cihazlarla paylaşılan vektörleri belirler.
- Cihaz ve sınıf bazında strict holdout uygulanabilirliğini ölçer.
- Cihaz çiftleri arasındaki vektör örtüşmesini hesaplar.
- Jaccard ve overlap katsayılarını üretir.

Ham CSV dosyalarını yeniden okumaz.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT duplicate yapısını "
            "cihaz ve sınıf bazında analiz eder."
        )
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="Duplicate denetim SQLite veritabanı.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Raporların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Önceden oluşturulmuş analiz tablolarını "
            "silerek yeniden oluşturur."
        ),
    )

    return parser.parse_args()


# ==========================================================
# JSON HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path değerlerini JSON uyumlu hâle getirir."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} JSON ile uyumlu değil."
    )


def write_json_atomic(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    """JSON raporunu güvenli biçimde kaydeder."""

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            data,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )

    temporary_file.replace(output_file)


# ==========================================================
# DATABASE HELPERS
# ==========================================================

def configure_database(
    connection: sqlite3.Connection,
) -> None:
    """SQLite performans seçeneklerini ayarlar."""

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA temp_store=FILE"
    )

    connection.execute(
        "PRAGMA cache_size=-200000"
    )


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """Bir SQLite tablosunun varlığını kontrol eder."""

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return bool(
        result is not None
        and int(result[0]) > 0
    )


def scalar_query(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
) -> int:
    """Tek bir sayısal SQL sonucu döndürür."""

    result = connection.execute(
        query,
        parameters,
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


def validate_database(
    connection: sqlite3.Connection,
) -> None:
    """Duplicate denetim veritabanını doğrular."""

    required_tables = {
        "files",
        "fingerprints",
    }

    missing_tables = [
        table_name
        for table_name in required_tables
        if not table_exists(
            connection,
            table_name,
        )
    ]

    if missing_tables:
        raise RuntimeError(
            "SQLite veritabanında gerekli tablolar eksik: "
            + ", ".join(sorted(missing_tables))
        )

    incomplete_files = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM files
        WHERE status != 'done'
        """,
    )

    if incomplete_files > 0:
        raise RuntimeError(
            "Duplicate denetiminde tamamlanmamış "
            f"{incomplete_files} dosya bulunuyor."
        )

    expected_rows = scalar_query(
        connection,
        """
        SELECT SUM(expected_row_count)
        FROM files
        """,
    )

    fingerprint_rows = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM fingerprints
        """,
    )

    if expected_rows != fingerprint_rows:
        raise RuntimeError(
            "SQLite satır sayısı uyuşmuyor: "
            f"beklenen={expected_rows:,}, "
            f"parmak_izi={fingerprint_rows:,}"
        )


# ==========================================================
# MATERIALIZED TABLES
# ==========================================================

def drop_analysis_tables(
    connection: sqlite3.Connection,
) -> None:
    """Önceden oluşturulmuş analiz tablolarını siler."""

    connection.executescript(
        """
        DROP TABLE IF EXISTS vector_device;
        DROP TABLE IF EXISTS vector_groups;
        """
    )

    connection.commit()


def create_vector_groups(
    connection: sqlite3.Connection,
) -> None:
    """
    Her farklı özellik vektörü için global grup istatistiği oluşturur.
    """

    if table_exists(
        connection,
        "vector_groups",
    ):
        print(
            "vector_groups tablosu mevcut; "
            "yeniden kullanılacak."
        )
        return

    print(
        "Global özellik vektörü grupları oluşturuluyor..."
    )

    connection.execute(
        """
        CREATE TABLE vector_groups AS
        SELECT
            p.hash_forward AS hash_forward,
            p.hash_reverse AS hash_reverse,
            COUNT(*) AS occurrence_count,
            COUNT(
                DISTINCT f.device
            ) AS device_count,
            COUNT(
                DISTINCT f.class_label
            ) AS class_count
        FROM fingerprints AS p
        JOIN files AS f
            ON f.file_id = p.file_id
        GROUP BY
            p.hash_forward,
            p.hash_reverse
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX
        idx_vector_groups_hash
        ON vector_groups (
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX
        idx_vector_groups_device_count
        ON vector_groups (
            device_count
        )
        """
    )

    connection.commit()


def create_vector_device(
    connection: sqlite3.Connection,
) -> None:
    """
    Her özellik vektörünün cihaz ve sınıf bazındaki oluşumunu oluşturur.
    """

    if table_exists(
        connection,
        "vector_device",
    ):
        print(
            "vector_device tablosu mevcut; "
            "yeniden kullanılacak."
        )
        return

    print(
        "Cihaz bazlı özellik vektörü tablosu oluşturuluyor..."
    )

    connection.execute(
        """
        CREATE TABLE vector_device AS
        SELECT
            p.hash_forward AS hash_forward,
            p.hash_reverse AS hash_reverse,
            f.device AS device,
            f.class_label AS class_label,
            COUNT(*) AS occurrence_count
        FROM fingerprints AS p
        JOIN files AS f
            ON f.file_id = p.file_id
        GROUP BY
            p.hash_forward,
            p.hash_reverse,
            f.device,
            f.class_label
        """
    )

    connection.execute(
        """
        CREATE INDEX
        idx_vector_device_device_hash
        ON vector_device (
            device,
            hash_forward,
            hash_reverse
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX
        idx_vector_device_hash_device
        ON vector_device (
            hash_forward,
            hash_reverse,
            device
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX
        idx_vector_device_device_class
        ON vector_device (
            device,
            class_label
        )
        """
    )

    connection.commit()


# ==========================================================
# REPORT QUERIES
# ==========================================================

def create_group_span_report(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Vektörlerin kaç cihazda görüldüğünü raporlar."""

    report = pd.read_sql_query(
        """
        SELECT
            device_count,
            COUNT(*) AS unique_vector_count,
            SUM(
                occurrence_count
            ) AS row_occurrence_count,
            SUM(
                occurrence_count - 1
            ) AS duplicate_rows_beyond_first
        FROM vector_groups
        GROUP BY device_count
        ORDER BY device_count
        """,
        connection,
    )

    total_vectors = int(
        report["unique_vector_count"].sum()
    )

    total_rows = int(
        report["row_occurrence_count"].sum()
    )

    report[
        "unique_vector_percentage"
    ] = (
        report["unique_vector_count"]
        / total_vectors
        * 100
    )

    report[
        "row_occurrence_percentage"
    ] = (
        report["row_occurrence_count"]
        / total_rows
        * 100
    )

    return report


def create_device_report(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Cihaz bazlı duplicate ve özgünlük raporu oluşturur."""

    report = pd.read_sql_query(
        """
        SELECT
            vd.device AS device,

            SUM(
                vd.occurrence_count
            ) AS total_rows,

            COUNT(*) AS unique_vectors,

            SUM(
                CASE
                    WHEN vg.device_count = 1
                    THEN 1
                    ELSE 0
                END
            ) AS device_exclusive_vectors,

            SUM(
                CASE
                    WHEN vg.device_count > 1
                    THEN 1
                    ELSE 0
                END
            ) AS cross_device_shared_vectors,

            SUM(
                CASE
                    WHEN vg.device_count = 1
                    THEN vd.occurrence_count
                    ELSE 0
                END
            ) AS rows_from_device_exclusive_vectors,

            SUM(
                CASE
                    WHEN vg.device_count > 1
                    THEN vd.occurrence_count
                    ELSE 0
                END
            ) AS rows_from_shared_vectors

        FROM vector_device AS vd

        JOIN vector_groups AS vg
            ON vg.hash_forward = vd.hash_forward
           AND vg.hash_reverse = vd.hash_reverse

        GROUP BY vd.device
        ORDER BY vd.device
        """,
        connection,
    )

    report[
        "exclusive_vector_percentage"
    ] = (
        report["device_exclusive_vectors"]
        / report["unique_vectors"]
        * 100
    )

    report[
        "shared_vector_percentage"
    ] = (
        report["cross_device_shared_vectors"]
        / report["unique_vectors"]
        * 100
    )

    report[
        "exclusive_row_percentage"
    ] = (
        report[
            "rows_from_device_exclusive_vectors"
        ]
        / report["total_rows"]
        * 100
    )

    report[
        "shared_row_percentage"
    ] = (
        report["rows_from_shared_vectors"]
        / report["total_rows"]
        * 100
    )

    return report


def create_device_class_report(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Cihaz ve sınıf bazlı özgünlük raporu oluşturur."""

    report = pd.read_sql_query(
        """
        SELECT
            vd.device AS device,
            vd.class_label AS class_label,

            SUM(
                vd.occurrence_count
            ) AS total_rows,

            COUNT(*) AS unique_vectors,

            SUM(
                CASE
                    WHEN vg.device_count = 1
                    THEN 1
                    ELSE 0
                END
            ) AS device_exclusive_vectors,

            SUM(
                CASE
                    WHEN vg.device_count > 1
                    THEN 1
                    ELSE 0
                END
            ) AS cross_device_shared_vectors,

            SUM(
                CASE
                    WHEN vg.device_count = 1
                    THEN vd.occurrence_count
                    ELSE 0
                END
            ) AS rows_from_device_exclusive_vectors,

            SUM(
                CASE
                    WHEN vg.device_count > 1
                    THEN vd.occurrence_count
                    ELSE 0
                END
            ) AS rows_from_shared_vectors

        FROM vector_device AS vd

        JOIN vector_groups AS vg
            ON vg.hash_forward = vd.hash_forward
           AND vg.hash_reverse = vd.hash_reverse

        GROUP BY
            vd.device,
            vd.class_label

        ORDER BY
            vd.device,
            vd.class_label
        """,
        connection,
    )

    report[
        "exclusive_vector_percentage"
    ] = (
        report["device_exclusive_vectors"]
        / report["unique_vectors"]
        * 100
    )

    report[
        "exclusive_row_percentage"
    ] = (
        report[
            "rows_from_device_exclusive_vectors"
        ]
        / report["total_rows"]
        * 100
    )

    return report


def create_holdout_feasibility_report(
    device_report: pd.DataFrame,
    device_class_report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her cihaz için strict leave-one-device-out uygulanabilirliğini özetler.
    """

    rows: list[dict[str, Any]] = []

    for device_name in sorted(
        device_report["device"].tolist()
    ):
        device_row = device_report[
            device_report["device"]
            == device_name
        ].iloc[0]

        class_rows = device_class_report[
            device_class_report["device"]
            == device_name
        ]

        present_class_count = int(
            class_rows["class_label"].nunique()
        )

        novel_class_count = int(
            (
                class_rows[
                    "device_exclusive_vectors"
                ] > 0
            ).sum()
        )

        classes_without_novel_vectors = (
            class_rows.loc[
                class_rows[
                    "device_exclusive_vectors"
                ] == 0,
                "class_label",
            ]
            .astype(str)
            .tolist()
        )

        minimum_novel_vectors = int(
            class_rows[
                "device_exclusive_vectors"
            ].min()
        )

        rows.append(
            {
                "device": device_name,
                "present_class_count": (
                    present_class_count
                ),
                "novel_class_count": (
                    novel_class_count
                ),
                "all_present_classes_have_novel_vectors": bool(
                    present_class_count
                    == novel_class_count
                ),
                "classes_without_novel_vectors": (
                    ", ".join(
                        classes_without_novel_vectors
                    )
                ),
                "total_rows": int(
                    device_row["total_rows"]
                ),
                "unique_vectors": int(
                    device_row["unique_vectors"]
                ),
                "strict_holdout_novel_vectors": int(
                    device_row[
                        "device_exclusive_vectors"
                    ]
                ),
                "strict_holdout_vector_retention_percentage": float(
                    device_row[
                        "exclusive_vector_percentage"
                    ]
                ),
                "strict_holdout_novel_rows": int(
                    device_row[
                        "rows_from_device_exclusive_vectors"
                    ]
                ),
                "strict_holdout_row_retention_percentage": float(
                    device_row[
                        "exclusive_row_percentage"
                    ]
                ),
                "minimum_novel_vectors_in_any_present_class": (
                    minimum_novel_vectors
                ),
            }
        )

    return pd.DataFrame(rows)


def create_device_overlap_reports(
    connection: sqlite3.Connection,
    device_report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cihaz çiftleri arasındaki özellik vektörü örtüşmesini hesaplar."""

    devices = sorted(
        device_report["device"]
        .astype(str)
        .tolist()
    )

    unique_counts = {
        str(row.device): int(row.unique_vectors)
        for row in device_report.itertuples(
            index=False
        )
    }

    overlap_count_matrix = pd.DataFrame(
        0,
        index=devices,
        columns=devices,
        dtype=np.int64,
    )

    jaccard_matrix = pd.DataFrame(
        0.0,
        index=devices,
        columns=devices,
        dtype=np.float64,
    )

    pair_records: list[dict[str, Any]] = []

    for device in devices:
        count = unique_counts[device]

        overlap_count_matrix.loc[
            device,
            device,
        ] = count

        jaccard_matrix.loc[
            device,
            device,
        ] = 1.0

    device_pairs = [
        (devices[first_index], devices[second_index])
        for first_index in range(len(devices))
        for second_index in range(
            first_index + 1,
            len(devices),
        )
    ]

    for device_a, device_b in tqdm(
        device_pairs,
        desc="Cihaz çiftleri karşılaştırılıyor",
        unit="çift",
    ):
        intersection_count = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM vector_device AS a
            JOIN vector_device AS b
              ON a.hash_forward = b.hash_forward
             AND a.hash_reverse = b.hash_reverse
            WHERE a.device = ?
              AND b.device = ?
            """,
            (
                device_a,
                device_b,
            ),
        )

        count_a = unique_counts[device_a]
        count_b = unique_counts[device_b]

        union_count = (
            count_a
            + count_b
            - intersection_count
        )

        jaccard = (
            intersection_count / union_count
            if union_count > 0
            else 0.0
        )

        smaller_device_count = min(
            count_a,
            count_b,
        )

        overlap_coefficient = (
            intersection_count
            / smaller_device_count
            if smaller_device_count > 0
            else 0.0
        )

        overlap_count_matrix.loc[
            device_a,
            device_b,
        ] = intersection_count

        overlap_count_matrix.loc[
            device_b,
            device_a,
        ] = intersection_count

        jaccard_matrix.loc[
            device_a,
            device_b,
        ] = jaccard

        jaccard_matrix.loc[
            device_b,
            device_a,
        ] = jaccard

        pair_records.append(
            {
                "device_a": device_a,
                "device_b": device_b,
                "device_a_unique_vectors": count_a,
                "device_b_unique_vectors": count_b,
                "shared_unique_vectors": (
                    intersection_count
                ),
                "union_unique_vectors": (
                    union_count
                ),
                "jaccard_similarity": (
                    jaccard
                ),
                "overlap_coefficient": (
                    overlap_coefficient
                ),
            }
        )

    pair_report = pd.DataFrame(
        pair_records
    ).sort_values(
        "shared_unique_vectors",
        ascending=False,
    )

    return (
        pair_report,
        overlap_count_matrix,
        jaccard_matrix,
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    database_file = args.database_file.resolve()
    output_dir = args.output_dir.resolve()

    if not database_file.exists():
        raise FileNotFoundError(
            "Duplicate denetim veritabanı bulunamadı: "
            f"{database_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    free_space_gb = (
        shutil.disk_usage(
            database_file.parent
        ).free
        / (1024**3)
    )

    print("=" * 78)
    print("N-BaIoT Duplicate Yapı Analizi")
    print("=" * 78)
    print(f"Veritabanı      : {database_file}")
    print(f"Çıktı klasörü   : {output_dir}")
    print(f"Boş disk alanı  : {free_space_gb:.2f} GB")
    print(f"Tabloları yenile: {args.rebuild}")
    print("=" * 78)

    if free_space_gb < 3:
        print(
            "UYARI: Boş disk alanı 3 GB altında."
        )

    connection = sqlite3.connect(
        database_file,
        timeout=120,
    )

    try:
        configure_database(
            connection
        )

        validate_database(
            connection
        )

        if args.rebuild:
            drop_analysis_tables(
                connection
            )

        create_vector_groups(
            connection
        )

        create_vector_device(
            connection
        )

        cross_class_groups = scalar_query(
            connection,
            """
            SELECT COUNT(*)
            FROM vector_groups
            WHERE class_count > 1
            """,
        )

        if cross_class_groups > 0:
            raise RuntimeError(
                "Sınıflar arası parmak izi grupları bulundu: "
                f"{cross_class_groups:,}"
            )

        print(
            "Cihaz ve sınıf raporları hazırlanıyor..."
        )

        group_span_report = (
            create_group_span_report(
                connection
            )
        )

        device_report = (
            create_device_report(
                connection
            )
        )

        device_class_report = (
            create_device_class_report(
                connection
            )
        )

        holdout_report = (
            create_holdout_feasibility_report(
                device_report=device_report,
                device_class_report=(
                    device_class_report
                ),
            )
        )

        (
            pair_report,
            overlap_count_matrix,
            jaccard_matrix,
        ) = create_device_overlap_reports(
            connection=connection,
            device_report=device_report,
        )

        # --------------------------------------------------
        # Output paths
        # --------------------------------------------------

        group_span_file = (
            output_dir
            / "nbaiot_vector_device_span.csv"
        )

        device_file = (
            output_dir
            / "nbaiot_device_duplicate_structure.csv"
        )

        device_class_file = (
            output_dir
            / "nbaiot_device_class_duplicate_structure.csv"
        )

        holdout_file = (
            output_dir
            / "nbaiot_device_holdout_feasibility.csv"
        )

        pair_file = (
            output_dir
            / "nbaiot_device_pair_overlap.csv"
        )

        overlap_matrix_file = (
            output_dir
            / "nbaiot_device_overlap_count_matrix.csv"
        )

        jaccard_matrix_file = (
            output_dir
            / "nbaiot_device_jaccard_matrix.csv"
        )

        summary_file = (
            output_dir
            / "nbaiot_duplicate_structure_summary.json"
        )

        # --------------------------------------------------
        # Save CSV reports
        # --------------------------------------------------

        group_span_report.to_csv(
            group_span_file,
            index=False,
            encoding="utf-8",
        )

        device_report.to_csv(
            device_file,
            index=False,
            encoding="utf-8",
        )

        device_class_report.to_csv(
            device_class_file,
            index=False,
            encoding="utf-8",
        )

        holdout_report.to_csv(
            holdout_file,
            index=False,
            encoding="utf-8",
        )

        pair_report.to_csv(
            pair_file,
            index=False,
            encoding="utf-8",
        )

        overlap_count_matrix.to_csv(
            overlap_matrix_file,
            encoding="utf-8",
        )

        jaccard_matrix.to_csv(
            jaccard_matrix_file,
            encoding="utf-8",
        )

        strict_full_class_devices = (
            holdout_report.loc[
                (
                    holdout_report[
                        "present_class_count"
                    ] == 11
                )
                & (
                    holdout_report[
                        "all_present_classes_have_novel_vectors"
                    ]
                ),
                "device",
            ]
            .astype(str)
            .tolist()
        )

        summary: dict[str, Any] = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "database_file": str(
                database_file
            ),
            "global_unique_vector_count": int(
                scalar_query(
                    connection,
                    """
                    SELECT COUNT(*)
                    FROM vector_groups
                    """,
                )
            ),
            "global_row_count": int(
                scalar_query(
                    connection,
                    """
                    SELECT SUM(occurrence_count)
                    FROM vector_groups
                    """,
                )
            ),
            "cross_class_group_count": int(
                cross_class_groups
            ),
            "device_count": int(
                len(device_report)
            ),
            "strict_full_11_class_holdout_device_count": int(
                len(strict_full_class_devices)
            ),
            "strict_full_11_class_holdout_devices": (
                strict_full_class_devices
            ),
            "device_summary": (
                device_report.to_dict(
                    orient="records"
                )
            ),
            "holdout_feasibility": (
                holdout_report.to_dict(
                    orient="records"
                )
            ),
            "report_files": {
                "group_span": str(
                    group_span_file
                ),
                "device_structure": str(
                    device_file
                ),
                "device_class_structure": str(
                    device_class_file
                ),
                "holdout_feasibility": str(
                    holdout_file
                ),
                "pair_overlap": str(
                    pair_file
                ),
                "overlap_count_matrix": str(
                    overlap_matrix_file
                ),
                "jaccard_matrix": str(
                    jaccard_matrix_file
                ),
            },
        }

        write_json_atomic(
            data=summary,
            output_file=summary_file,
        )

        connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )

        print()
        print("=" * 78)
        print("Duplicate yapı analizi tamamlandı")
        print("=" * 78)

        display_columns = [
            "device",
            "total_rows",
            "unique_vectors",
            "device_exclusive_vectors",
            "exclusive_vector_percentage",
            "exclusive_row_percentage",
        ]

        print(
            device_report[
                display_columns
            ].to_string(
                index=False,
                formatters={
                    "exclusive_vector_percentage": (
                        lambda value: f"{value:.4f}"
                    ),
                    "exclusive_row_percentage": (
                        lambda value: f"{value:.4f}"
                    ),
                },
            )
        )

        print()
        print(
            "Strict 11 sınıflı holdout için "
            "uygun cihaz sayısı: "
            f"{len(strict_full_class_devices)}"
        )

        print(
            "Uygun cihazlar: "
            + (
                ", ".join(
                    strict_full_class_devices
                )
                if strict_full_class_devices
                else "Yok"
            )
        )

        print()
        print(f"Cihaz raporu   : {device_file}")
        print(f"Holdout raporu : {holdout_file}")
        print(f"Örtüşme raporu : {pair_file}")
        print(f"JSON özet      : {summary_file}")
        print("=" * 78)

    finally:
        connection.close()


if __name__ == "__main__":
    main()