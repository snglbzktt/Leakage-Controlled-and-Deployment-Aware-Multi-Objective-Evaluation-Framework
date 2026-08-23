"""
N-BaIoT ana deney split indeksini oluşturur.

Amaç:
- Global olarak benzersiz özellik vektörlerini kullanmak.
- Aynı vektörün train, validation ve test arasında bulunmasını engellemek.
- Deterministik ve tekrar üretilebilir 70/15/15 bölme oluşturmak.
- Ham CSV verilerini bu aşamada yeniden okumamak.
- Bölme sonuçlarını sınıf bazında doğrulamak.

Kaynak:
data/cache/nbaiot_duplicate_audit.sqlite

Çıktılar:
data/splits/nbaiot_primary_unique_split_seed2026.csv
results/reports/nbaiot_primary_split_summary.json
results/reports/nbaiot_primary_split_class_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
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

DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


# ==========================================================
# CONSTANTS
# ==========================================================

UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT benzersiz özellik vektörleri için "
            "deterministik ana deney split indeksi oluşturur."
        )
    )

    parser.add_argument(
        "--database-file",
        type=Path,
        default=DEFAULT_DATABASE_FILE,
        help="Duplicate denetim SQLite veritabanı.",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Split indeksinin kaydedileceği klasör.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Özet raporların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministik split tohumu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="SQLite sorgusundan alınacak kayıt sayısı.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut split dosyasının üzerine yazar.",
    )

    return parser.parse_args()


# ==========================================================
# JSON HELPERS
# ==========================================================

def json_default(value: object) -> object:
    """NumPy ve Path nesnelerini JSON uyumlu hâle getirir."""

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
    """JSON dosyasını geçici dosya üzerinden güvenli biçimde yazar."""

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

def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """SQLite tablosunun varlığını kontrol eder."""

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
) -> int:
    """Tek sayısal değer döndüren SQL sorgusunu çalıştırır."""

    result = connection.execute(
        query
    ).fetchone()

    if result is None or result[0] is None:
        return 0

    return int(result[0])


def validate_database(
    connection: sqlite3.Connection,
) -> tuple[int, int]:
    """
    Duplicate veritabanını ve analiz tablolarını doğrular.

    Dönüş:
        global_unique_vector_count,
        global_raw_row_count
    """

    required_tables = {
        "files",
        "fingerprints",
        "vector_groups",
        "vector_device",
    }

    missing_tables = [
        table_name
        for table_name in sorted(required_tables)
        if not table_exists(
            connection,
            table_name,
        )
    ]

    if missing_tables:
        raise RuntimeError(
            "SQLite veritabanında gerekli tablolar eksik: "
            + ", ".join(missing_tables)
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
            "Sınıflar arası çelişkili özellik vektörleri var: "
            f"{cross_class_groups:,}"
        )

    unique_vector_count = scalar_query(
        connection,
        """
        SELECT COUNT(*)
        FROM vector_groups
        """,
    )

    raw_row_count = scalar_query(
        connection,
        """
        SELECT SUM(occurrence_count)
        FROM vector_groups
        """,
    )

    return (
        unique_vector_count,
        raw_row_count,
    )


# ==========================================================
# DETERMINISTIC HASH SPLIT
# ==========================================================

def splitmix64(
    values: np.ndarray,
) -> np.ndarray:
    """
    uint64 değerlerini deterministik biçimde karıştırır.

    Bu fonksiyon kriptografik amaçla değil, tekrar üretilebilir
    split ataması için kullanılmaktadır.
    """

    values = values.astype(
        np.uint64,
        copy=False,
    )

    with np.errstate(over="ignore"):
        values = (
            values
            + np.uint64(0x9E3779B97F4A7C15)
        )

        values = (
            values
            ^ (values >> np.uint64(30))
        ) * np.uint64(0xBF58476D1CE4E5B9)

        values = (
            values
            ^ (values >> np.uint64(27))
        ) * np.uint64(0x94D049BB133111EB)

        values = (
            values
            ^ (values >> np.uint64(31))
        )

    return values


def calculate_split_codes(
    hash_forward: np.ndarray,
    hash_reverse: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    İki 64-bit parmak izinden deterministik split kodu üretir.

    Kodlar:
        0 -> train
        1 -> validation
        2 -> test
    """

    forward_unsigned = (
        hash_forward
        .astype(np.int64, copy=False)
        .view(np.uint64)
    )

    reverse_unsigned = (
        hash_reverse
        .astype(np.int64, copy=False)
        .view(np.uint64)
    )

    rotated_reverse = (
        reverse_unsigned << np.uint64(32)
    ) | (
        reverse_unsigned >> np.uint64(32)
    )

    seed_unsigned = np.uint64(
        seed & UINT64_MASK
    )

    mixed_input = (
        forward_unsigned
        ^ rotated_reverse
        ^ seed_unsigned
    )

    random_values = splitmix64(
        mixed_input
    )

    train_threshold = np.uint64(
        (UINT64_RANGE * 70) // 100
    )

    validation_threshold = np.uint64(
        (UINT64_RANGE * 85) // 100
    )

    split_codes = np.full(
        len(random_values),
        2,
        dtype=np.uint8,
    )

    split_codes[
        random_values < validation_threshold
    ] = 1

    split_codes[
        random_values < train_threshold
    ] = 0

    return split_codes


# ==========================================================
# REPORT HELPERS
# ==========================================================

def nested_counter_to_dataframe(
    unique_counts: dict[str, Counter[str]],
    occurrence_counts: dict[str, Counter[str]],
) -> pd.DataFrame:
    """Sınıf ve split sayaçlarını tabloya dönüştürür."""

    classes = sorted(
        set(unique_counts)
        | set(occurrence_counts)
    )

    records: list[dict[str, Any]] = []

    for class_label in classes:
        class_unique_total = int(
            sum(
                unique_counts[
                    class_label
                ].values()
            )
        )

        class_occurrence_total = int(
            sum(
                occurrence_counts[
                    class_label
                ].values()
            )
        )

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            unique_count = int(
                unique_counts[
                    class_label
                ][split_name]
            )

            occurrence_count = int(
                occurrence_counts[
                    class_label
                ][split_name]
            )

            records.append(
                {
                    "class_label": class_label,
                    "split": split_name,
                    "unique_vector_count": (
                        unique_count
                    ),
                    "unique_vector_percentage_within_class": (
                        unique_count
                        / class_unique_total
                        * 100
                        if class_unique_total > 0
                        else 0.0
                    ),
                    "raw_occurrence_count": (
                        occurrence_count
                    ),
                    "raw_occurrence_percentage_within_class": (
                        occurrence_count
                        / class_occurrence_total
                        * 100
                        if class_occurrence_total > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Ana program akışı."""

    args = parse_arguments()

    database_file = args.database_file.resolve()
    split_dir = args.split_dir.resolve()
    report_dir = args.report_dir.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size sıfırdan büyük olmalıdır."
        )

    if not database_file.exists():
        raise FileNotFoundError(
            "Duplicate SQLite veritabanı bulunamadı: "
            f"{database_file}"
        )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_file = (
        split_dir
        / (
            "nbaiot_primary_unique_split_"
            f"seed{args.seed}.csv"
        )
    )

    temporary_split_file = (
        split_file.with_suffix(
            split_file.suffix + ".tmp"
        )
    )

    class_report_file = (
        report_dir
        / "nbaiot_primary_split_class_distribution.csv"
    )

    summary_file = (
        report_dir
        / "nbaiot_primary_split_summary.json"
    )

    if split_file.exists() and not args.overwrite:
        raise FileExistsError(
            "Split dosyası zaten mevcut. Üzerine yazmak için "
            "--overwrite kullan: "
            f"{split_file}"
        )

    if temporary_split_file.exists():
        temporary_split_file.unlink()

    connection = sqlite3.connect(
        database_file,
        timeout=120,
    )

    try:
        (
            expected_unique_vectors,
            expected_raw_rows,
        ) = validate_database(
            connection
        )

        print("=" * 78)
        print("N-BaIoT Ana Deney Split İndeksi")
        print("=" * 78)
        print(f"Veritabanı          : {database_file}")
        print(f"Split dosyası       : {split_file}")
        print(f"Seed                : {args.seed}")
        print("Bölme oranı         : %70 / %15 / %15")
        print(
            "Benzersiz vektör    : "
            f"{expected_unique_vectors:,}"
        )
        print(
            "Ham kayıt karşılığı : "
            f"{expected_raw_rows:,}"
        )
        print(f"Batch boyutu        : {args.batch_size:,}")
        print("=" * 78)

        query = """
            SELECT
                vd.hash_forward,
                vd.hash_reverse,
                MIN(vd.class_label) AS class_label,
                SUM(vd.occurrence_count) AS occurrence_count,
                COUNT(*) AS device_count
            FROM vector_device AS vd
            GROUP BY
                vd.hash_forward,
                vd.hash_reverse
            ORDER BY
                vd.hash_forward,
                vd.hash_reverse
        """

        cursor = connection.execute(
            query
        )

        split_unique_counts: Counter[str] = (
            Counter()
        )

        split_occurrence_counts: Counter[str] = (
            Counter()
        )

        class_unique_counts: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        class_occurrence_counts: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        device_span_counts: Counter[int] = (
            Counter()
        )

        processed_unique_vectors = 0
        processed_raw_occurrences = 0

        progress = tqdm(
            total=expected_unique_vectors,
            desc="Split indeksi oluşturuluyor",
            unit="vektör",
        )

        with temporary_split_file.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file_handle:
            writer = csv.writer(
                file_handle
            )

            writer.writerow(
                [
                    "hash_forward",
                    "hash_reverse",
                    "class_label",
                    "occurrence_count",
                    "device_count",
                    "split",
                ]
            )

            while True:
                rows = cursor.fetchmany(
                    args.batch_size
                )

                if not rows:
                    break

                hash_forward = np.asarray(
                    [
                        int(row[0])
                        for row in rows
                    ],
                    dtype=np.int64,
                )

                hash_reverse = np.asarray(
                    [
                        int(row[1])
                        for row in rows
                    ],
                    dtype=np.int64,
                )

                split_codes = (
                    calculate_split_codes(
                        hash_forward=hash_forward,
                        hash_reverse=hash_reverse,
                        seed=args.seed,
                    )
                )

                output_rows: list[
                    tuple[
                        int,
                        int,
                        str,
                        int,
                        int,
                        str,
                    ]
                ] = []

                for row, split_code in zip(
                    rows,
                    split_codes,
                    strict=True,
                ):
                    (
                        forward_value,
                        reverse_value,
                        class_label,
                        occurrence_count,
                        device_count,
                    ) = row

                    split_name = SPLIT_NAMES[
                        int(split_code)
                    ]

                    class_label = str(
                        class_label
                    )

                    occurrence_count = int(
                        occurrence_count
                    )

                    device_count = int(
                        device_count
                    )

                    output_rows.append(
                        (
                            int(forward_value),
                            int(reverse_value),
                            class_label,
                            occurrence_count,
                            device_count,
                            split_name,
                        )
                    )

                    split_unique_counts[
                        split_name
                    ] += 1

                    split_occurrence_counts[
                        split_name
                    ] += occurrence_count

                    class_unique_counts[
                        class_label
                    ][split_name] += 1

                    class_occurrence_counts[
                        class_label
                    ][split_name] += (
                        occurrence_count
                    )

                    device_span_counts[
                        device_count
                    ] += 1

                    processed_unique_vectors += 1
                    processed_raw_occurrences += (
                        occurrence_count
                    )

                writer.writerows(
                    output_rows
                )

                progress.update(
                    len(rows)
                )

        progress.close()

        if (
            processed_unique_vectors
            != expected_unique_vectors
        ):
            raise RuntimeError(
                "Benzersiz vektör sayısı uyuşmuyor: "
                f"beklenen={expected_unique_vectors:,}, "
                f"işlenen={processed_unique_vectors:,}"
            )

        if (
            processed_raw_occurrences
            != expected_raw_rows
        ):
            raise RuntimeError(
                "Ham oluşum sayısı uyuşmuyor: "
                f"beklenen={expected_raw_rows:,}, "
                f"işlenen={processed_raw_occurrences:,}"
            )

        required_splits = {
            "train",
            "validation",
            "test",
        }

        for class_label, counts in (
            class_unique_counts.items()
        ):
            missing_splits = (
                required_splits
                - set(
                    split_name
                    for split_name, count
                    in counts.items()
                    if count > 0
                )
            )

            if missing_splits:
                raise RuntimeError(
                    f"{class_label} sınıfında eksik split var: "
                    + ", ".join(
                        sorted(missing_splits)
                    )
                )

        temporary_split_file.replace(
            split_file
        )

        class_distribution = (
            nested_counter_to_dataframe(
                unique_counts=class_unique_counts,
                occurrence_counts=(
                    class_occurrence_counts
                ),
            )
        )

        class_distribution.to_csv(
            class_report_file,
            index=False,
            encoding="utf-8",
        )

        overall_split_records: list[
            dict[str, Any]
        ] = []

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            unique_count = int(
                split_unique_counts[
                    split_name
                ]
            )

            occurrence_count = int(
                split_occurrence_counts[
                    split_name
                ]
            )

            overall_split_records.append(
                {
                    "split": split_name,
                    "unique_vector_count": (
                        unique_count
                    ),
                    "unique_vector_percentage": (
                        unique_count
                        / expected_unique_vectors
                        * 100
                    ),
                    "raw_occurrence_count": (
                        occurrence_count
                    ),
                    "raw_occurrence_percentage": (
                        occurrence_count
                        / expected_raw_rows
                        * 100
                    ),
                }
            )

        summary: dict[str, Any] = {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "database_file": str(
                database_file
            ),
            "split_file": str(
                split_file
            ),
            "seed": int(
                args.seed
            ),
            "assignment_method": (
                "Deterministic uint64 SplitMix64 assignment "
                "from the composite feature-vector fingerprint"
            ),
            "ratios": {
                "train": TRAIN_RATIO,
                "validation": VALIDATION_RATIO,
                "test": TEST_RATIO,
            },
            "global_unique_vector_count": int(
                expected_unique_vectors
            ),
            "global_raw_occurrence_count": int(
                expected_raw_rows
            ),
            "class_count": int(
                len(class_unique_counts)
            ),
            "overall_split_distribution": (
                overall_split_records
            ),
            "device_span_distribution": {
                str(device_count): int(count)
                for device_count, count
                in sorted(
                    device_span_counts.items()
                )
            },
            "leakage_guarantee": (
                "Each exact feature-vector fingerprint occurs "
                "in exactly one split."
            ),
            "primary_evaluation_weighting": (
                "Each globally unique feature vector receives "
                "equal weight."
            ),
            "class_distribution_report": str(
                class_report_file
            ),
        }

        write_json_atomic(
            data=summary,
            output_file=summary_file,
        )

        print()
        print("=" * 78)
        print("Ana deney split indeksi tamamlandı")
        print("=" * 78)

        for record in overall_split_records:
            print(
                f"{record['split']:10s}: "
                f"{record['unique_vector_count']:,} "
                "benzersiz vektör "
                f"(%{record['unique_vector_percentage']:.4f})"
            )

        print()
        print(
            "Toplam benzersiz vektör : "
            f"{processed_unique_vectors:,}"
        )
        print(
            "Ham kayıt karşılığı     : "
            f"{processed_raw_occurrences:,}"
        )
        print(f"Sınıf sayısı            : {len(class_unique_counts)}")
        print()
        print(f"Split indeksi : {split_file}")
        print(f"Sınıf raporu  : {class_report_file}")
        print(f"JSON özet     : {summary_file}")
        print("=" * 78)

    except Exception:
        if temporary_split_file.exists():
            temporary_split_file.unlink()

        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()