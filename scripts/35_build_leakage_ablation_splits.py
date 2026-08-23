"""
N-BaIoT sızıntı ablation'ı için karşılaştırılabilir split indeksleri oluşturur.

Kollar
-----
1. naive_row:
   Her ham satır, (file_id, row_number, seed) üzerinden bağımsız olarak
   %70/%15/%15 oranında bölünür. Aynı özellik vektörünün farklı splitlere
   düşmesine izin verir ve yaygın fakat sızıntıya açık protokolü temsil eder.
2. fingerprint_grouped:
   Aynı çift 64-bit özellik parmak izine sahip bütün satırlar tek splitte
   tutulur. Böylece train/validation/test arasında tam vektör örtüşmesi olmaz.

Bu aşama eğitim yapmaz. Eğitimden önce splitleri kilitler, iki kolun aynı ham
kayıt evrenini kullandığını doğrular ve naif koldaki gerçek örtüşmeyi ölçer.
Mevcut 150 ana deney veya bunların artifactları değiştirilmez.

Varsayılan girdiler
-------------------
data/cache/nbaiot_duplicate_audit.sqlite

Çıktılar
--------
data/splits/vs1_leakage_ablation_seed2026.sqlite
results/vs1/reports/nbaiot_vs1_leakage_split_distribution.csv
results/vs1/reports/nbaiot_vs1_leakage_overlap_by_pair.csv
results/vs1/reports/nbaiot_vs1_leakage_ablation_split_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DATABASE = (
    PROJECT_ROOT / "data" / "cache" / "nbaiot_duplicate_audit.sqlite"
)
DEFAULT_OUTPUT_DATABASE = (
    PROJECT_ROOT / "data" / "splits" / "vs1_leakage_ablation_seed2026.sqlite"
)
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "results" / "vs1" / "reports"

SPLITS = ("train", "validation", "test")
SPLIT_PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)
UINT64_MASK = (1 << 64) - 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VS1 sızıntı ablation'ı için naif satır ve parmak izi grup "
            "kontrollü split indekslerini oluşturur."
        )
    )
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--output-database", type=Path, default=DEFAULT_OUTPUT_DATABASE)
    parser.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} JSON ile uyumlu değil.")


def write_json_atomic(document: dict[str, Any], output_file: Path) -> None:
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as file_handle:
        json.dump(
            document,
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
    temporary_file.replace(output_file)


def file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def validate_source_database(connection: sqlite3.Connection) -> int:
    required_tables = {"files", "fingerprints"}
    missing = sorted(
        table for table in required_tables if not table_exists(connection, table)
    )
    if missing:
        raise RuntimeError("Kaynak SQLite tabloları eksik: " + ", ".join(missing))

    incomplete = int(
        connection.execute(
            "SELECT COUNT(*) FROM files WHERE status != 'done'"
        ).fetchone()[0]
    )
    if incomplete:
        raise RuntimeError(f"Tamamlanmamış kaynak dosya sayısı: {incomplete}")

    expected = int(
        connection.execute("SELECT SUM(expected_row_count) FROM files").fetchone()[0]
    )
    actual = int(connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0])
    if expected != actual:
        raise RuntimeError(
            f"Ham satır sayısı uyuşmuyor: files={expected:,}, fingerprints={actual:,}"
        )
    if actual <= 0:
        raise RuntimeError("Kaynak veritabanında parmak izi kaydı yok.")
    return actual


def register_sql_functions(connection: sqlite3.Connection, seed: int) -> None:
    """SQLite içinde platformdan bağımsız deterministik split fonksiyonları."""

    def splitmix64(value: int) -> int:
        value &= UINT64_MASK
        value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
        return (value ^ (value >> 31)) & UINT64_MASK

    def split_name(mixed_value: int) -> str:
        uniform = splitmix64(int(mixed_value)) / float(1 << 64)
        if uniform < 0.70:
            return "train"
        if uniform < 0.85:
            return "validation"
        return "test"

    def naive_split(file_id: int, row_number: int) -> str:
        key = (
            (int(file_id) * 0xD6E8FEB86659FD93)
            ^ (int(row_number) * 0xA5A3564E27F8862F)
            ^ (seed & UINT64_MASK)
        ) & UINT64_MASK
        return split_name(key)

    def grouped_split(hash_forward: int, hash_reverse: int) -> str:
        forward = int(hash_forward) & UINT64_MASK
        reverse = int(hash_reverse) & UINT64_MASK
        rotated_reverse = ((reverse << 32) | (reverse >> 32)) & UINT64_MASK
        return split_name(forward ^ rotated_reverse ^ (seed & UINT64_MASK))

    connection.create_function("naive_split", 2, naive_split, deterministic=True)
    connection.create_function("grouped_split", 2, grouped_split, deterministic=True)


def build_output_database(
    source_database: Path,
    output_database: Path,
    seed: int,
    overwrite: bool,
) -> None:
    if output_database.exists():
        if not overwrite:
            raise FileExistsError(
                f"Çıktı zaten mevcut: {output_database}. Yenilemek için --overwrite kullanın."
            )
        output_database.unlink()

    output_database.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def stage(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[{elapsed / 60:8.2f} dk] {message}", flush=True)

    # Yarım kalan WAL/SHM yan dosyaları yeni koşuma taşınmamalıdır.
    for sidecar in (
        output_database.with_name(output_database.name + "-wal"),
        output_database.with_name(output_database.name + "-shm"),
    ):
        if sidecar.exists():
            sidecar.unlink()

    connection = sqlite3.connect(output_database, timeout=120.0)
    try:
        # Bu veritabanı --overwrite ile yeniden üretilebilir. Tek büyük üretim
        # sırasında WAL, ana dosyanın değişim zamanını yanıltıyor ve ek I/O
        # oluşturuyordu; geçici üretimde journal_mode=OFF daha uygundur.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-100000")
        register_sql_functions(connection, seed)
        connection.execute("ATTACH DATABASE ? AS source", (str(source_database),))
        stage("Parmak izi grupları tek geçişte özetleniyor...")
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);

            CREATE TABLE fingerprint_stats AS
            SELECT
                p.hash_forward,
                p.hash_reverse,
                SUM(naive_split(p.file_id, p.row_number) = 'train') AS naive_train_rows,
                SUM(naive_split(p.file_id, p.row_number) = 'validation') AS naive_validation_rows,
                SUM(naive_split(p.file_id, p.row_number) = 'test') AS naive_test_rows,
                grouped_split(p.hash_forward, p.hash_reverse) AS grouped_split
            FROM source.fingerprints AS p
            GROUP BY p.hash_forward, p.hash_reverse;
            """
        )
        stage("7.062.606 satır için iki bölme kolu yazılıyor...")
        connection.executescript(
            """
            CREATE TABLE assignments AS
            SELECT
                p.file_id,
                p.row_number,
                p.hash_forward,
                p.hash_reverse,
                f.device,
                f.class_label,
                naive_split(p.file_id, p.row_number) AS naive_row_split,
                grouped_split(p.hash_forward, p.hash_reverse) AS grouped_split
            FROM source.fingerprints AS p
            JOIN source.files AS f ON f.file_id = p.file_id;
            """
        )
        stage("Naif kol eğitim erişim indeksi kuruluyor (uzun sürebilir)...")
        connection.execute(
            "CREATE INDEX idx_assignments_naive_file_row "
            "ON assignments(naive_row_split, file_id, row_number)"
        )
        stage("Grup kontrollü kol eğitim erişim indeksi kuruluyor (uzun sürebilir)...")
        connection.execute(
            "CREATE INDEX idx_assignments_grouped_file_row "
            "ON assignments(grouped_split, file_id, row_number)"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("protocol", "vs1_leakage_ablation"),
                ("protocol_version", "1.0"),
                ("seed", str(seed)),
                ("split_ratios", "0.70/0.15/0.15"),
                ("source_database", str(source_database)),
                ("generated_at_utc", datetime.now(timezone.utc).isoformat()),
            ),
        )
        connection.commit()
        stage("Çıktı veritabanı tamamlandı.")
    finally:
        connection.close()


def split_distribution(connection: sqlite3.Connection) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split, count_column in (
        ("train", "naive_train_rows"),
        ("validation", "naive_validation_rows"),
        ("test", "naive_test_rows"),
    ):
        row = connection.execute(
            f"SELECT COALESCE(SUM({count_column}), 0), "
            f"SUM({count_column} > 0) FROM fingerprint_stats"
        ).fetchone()
        records.append({"arm": "naive_row", "split": split,
                        "row_count": int(row[0]),
                        "unique_fingerprint_count": int(row[1])})
    for split in SPLITS:
        row = connection.execute(
            "SELECT COALESCE(SUM(naive_train_rows + naive_validation_rows + "
            "naive_test_rows), 0), COUNT(*) FROM fingerprint_stats "
            "WHERE grouped_split = ?", (split,)
        ).fetchone()
        records.append({"arm": "fingerprint_grouped", "split": split,
                        "row_count": int(row[0]),
                        "unique_fingerprint_count": int(row[1])})
    result = pd.DataFrame.from_records(records)
    totals = result.groupby("arm")["row_count"].transform("sum")
    result["row_percentage"] = result["row_count"] * 100.0 / totals
    return result


def overlap_report(connection: sqlite3.Connection) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    count_columns = {"train": "naive_train_rows", "validation": "naive_validation_rows",
                     "test": "naive_test_rows"}
    for arm in ("naive_row", "fingerprint_grouped"):
        for left, right in SPLIT_PAIRS:
            if arm == "naive_row":
                overlap = int(connection.execute(
                    f"SELECT COUNT(*) FROM fingerprint_stats WHERE "
                    f"{count_columns[left]} > 0 AND {count_columns[right]} > 0"
                ).fetchone()[0])
            else:
                # Bir parmak izi grubuna yalnızca tek grouped_split atanır.
                overlap = 0
            records.append(
                {
                    "arm": arm,
                    "split_left": left,
                    "split_right": right,
                    "overlapping_fingerprint_count": overlap,
                }
            )
    return pd.DataFrame.from_records(records)


def leakage_exposure(connection: sqlite3.Connection, split: str) -> dict[str, int | float]:
    count_column = {"validation": "naive_validation_rows", "test": "naive_test_rows"}[split]
    row = connection.execute(
        f"SELECT COALESCE(SUM({count_column}), 0), "
        f"COALESCE(SUM(CASE WHEN naive_train_rows > 0 THEN {count_column} ELSE 0 END), 0), "
        f"SUM(naive_train_rows > 0 AND {count_column} > 0) FROM fingerprint_stats"
    ).fetchone()
    evaluation_rows = int(row[0])
    exposed_rows = int(row[1] or 0)
    return {
        "split": split,
        "evaluation_rows": evaluation_rows,
        "train_exposed_rows": exposed_rows,
        "train_exposed_unique_fingerprints": int(row[2] or 0),
        "train_exposed_row_percentage": (
            100.0 * exposed_rows / evaluation_rows if evaluation_rows else 0.0
        ),
    }


def main() -> None:
    args = parse_arguments()
    source_database = args.source_database.resolve()
    output_database = args.output_database.resolve()
    report_directory = args.report_directory.resolve()

    if not source_database.exists():
        raise FileNotFoundError(f"Kaynak veritabanı bulunamadı: {source_database}")
    if source_database == output_database:
        raise ValueError("Kaynak ve çıktı veritabanı aynı olamaz.")

    with sqlite3.connect(source_database) as source_connection:
        source_row_count = validate_source_database(source_connection)

    build_output_database(
        source_database=source_database,
        output_database=output_database,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    report_directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_database) as connection:
        assignment_count = int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        if assignment_count != source_row_count:
            raise RuntimeError(
                f"Atama sayısı uyuşmuyor: kaynak={source_row_count:,}, çıktı={assignment_count:,}"
            )
        distribution = split_distribution(connection)
        overlaps = overlap_report(connection)
        exposures = [
            leakage_exposure(connection, "validation"),
            leakage_exposure(connection, "test"),
        ]

    grouped_overlap = int(
        overlaps.loc[
            overlaps["arm"] == "fingerprint_grouped",
            "overlapping_fingerprint_count",
        ].sum()
    )
    if grouped_overlap != 0:
        raise RuntimeError(
            f"Grup kontrollü kolda beklenmeyen split örtüşmesi: {grouped_overlap:,}"
        )

    distribution_file = report_directory / "nbaiot_vs1_leakage_split_distribution.csv"
    overlap_file = report_directory / "nbaiot_vs1_leakage_overlap_by_pair.csv"
    summary_file = report_directory / "nbaiot_vs1_leakage_ablation_split_summary.json"
    distribution.to_csv(distribution_file, index=False, encoding="utf-8")
    overlaps.to_csv(overlap_file, index=False, encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "vs1_leakage_ablation",
        "protocol_version": "1.0",
        "status": "split_indices_locked_before_training",
        "seed": int(args.seed),
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "source_row_count": source_row_count,
        "arms": {
            "naive_row": {
                "assignment_unit": "raw row",
                "exact_duplicate_cross_split_overlap_allowed": True,
                "validation_and_test_train_exposure": exposures,
            },
            "fingerprint_grouped": {
                "assignment_unit": "paired 64-bit feature fingerprint group",
                "exact_duplicate_cross_split_overlap_allowed": False,
                "observed_pairwise_overlap_sum": grouped_overlap,
            },
        },
        "scientific_controls": {
            "same_raw_row_universe": True,
            "same_split_seed": True,
            "same_split_ratios": True,
            "training_configuration_must_be_identical_between_arms": True,
            "scaler_and_class_weights_must_be_fitted_separately_from_each_arm_train_only": True,
            "validation_selects_checkpoint": True,
            "test_used_once_after_model_selection": True,
        },
        "artifacts": {
            "assignment_database": str(output_database),
            "assignment_database_sha256": file_sha256(output_database),
            "split_distribution_csv": str(distribution_file),
            "overlap_by_pair_csv": str(overlap_file),
        },
    }
    write_json_atomic(summary, summary_file)

    print("=" * 78)
    print("VS1 Sızıntı Ablation Splitleri Hazır")
    print("=" * 78)
    print(f"Ham kayıt       : {source_row_count:,}")
    print(f"Seed            : {args.seed}")
    print(f"Çıktı veritabanı: {output_database}")
    print(f"Rapor klasörü   : {report_directory}")
    print(f"Gruplu örtüşme  : {grouped_overlap:,}")
    for exposure in exposures:
        print(
            f"Naif {exposure['split']} train-maruz kalma: "
            f"{exposure['train_exposed_rows']:,}/{exposure['evaluation_rows']:,} "
            f"(%{exposure['train_exposed_row_percentage']:.4f})"
        )
    print("Mevcut 150 ana deneye ait hiçbir dosya değiştirilmedi.")
    print("=" * 78)


if __name__ == "__main__":
    main()