"""Build leakage-ablation splits in the model's actual float32 input space.

This stage creates two deterministic, family-balance-audited 70/15/15 arms over
the same raw N-BaIoT row universe:

* ``naive_row`` assigns each raw row independently and therefore permits exact
  float32 feature vectors to cross train/validation/test boundaries.
* ``grouped_float32`` assigns every identical 115-feature float32 vector to one
  split.  Groups are allocated largest-first against per-family raw-row targets
  so that very large duplicate groups cannot silently distort the split ratios.
  Grouping is performed after the same float32 conversion used by the PyTorch
  model.

The existing primary experiments and the older VS1 leakage-ablation artifacts
are never modified.  Put this file in ``scripts/`` inside TinyML_IDS_IEEE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATABASE = PROJECT_ROOT / "data" / "cache" / "nbaiot_duplicate_audit.sqlite"
DEFAULT_CSV_ROOT = PROJECT_ROOT / "data" / "interim" / "nbaiot_unpacked"
DEFAULT_OUTPUT_DATABASE = PROJECT_ROOT / "data" / "splits" / "vs2_float32_leakage_ablation_seed2026.sqlite"
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "results" / "vs2" / "reports"

SPLITS = ("train", "validation", "test")
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_CODES.items()}
SPLIT_PAIRS = ((0, 1), (0, 2), (1, 2))
FAMILY_NAMES = ("benign", "gafgyt", "mirai")
UINT64_MASK = (1 << 64) - 1
TRAIN_LIMIT = int(0.70 * (1 << 64))
VALIDATION_LIMIT = int(0.85 * (1 << 64))
EXPECTED_FEATURE_COUNT = 115
EXPECTED_RAW_ROW_COUNT = 7_062_606
EXPECTED_FLOAT32_GROUP_COUNT = 2_278_176
EXPECTED_SOURCE_FILE_COUNT = 89
LOCKED_SPLIT_SEED = 2026
PROTOCOL_VERSION = "2.1"
MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS = 0.01
MAX_NAIVE_RATIO_ERROR_PERCENTAGE_POINTS = 0.25
MINIMUM_STAGE37_FREE_BYTES = 5 * 1024**3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 37: float32 leakage-ablation splits")
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=DEFAULT_CSV_ROOT,
        help="files tablosundaki göreli CSV yollarının bağlandığı N-BaIoT kök klasörü",
    )
    parser.add_argument("--output-database", type=Path, default=DEFAULT_OUTPUT_DATABASE)
    parser.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
    os.replace(temporary, path)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def source_files(source_database: Path, csv_root: Path) -> list[tuple[int, Path, str, int]]:
    """Return file id, resolved CSV path, raw class label, expected rows."""
    with sqlite3.connect(source_database) as connection:
        if not table_exists(connection, "files"):
            raise RuntimeError("Kaynak veritabanında files tablosu yok.")
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(files)")]
        path_candidates = ("file_path", "csv_path", "path", "relative_path", "source_file")
        path_column = next((name for name in path_candidates if name in columns), None)
        required = {"file_id", "class_label", "expected_row_count"}
        if path_column is None or not required.issubset(columns):
            raise RuntimeError(f"Desteklenmeyen files şeması: {columns}")
        if "status" in columns:
            incomplete = int(
                connection.execute("SELECT COUNT(*) FROM files WHERE status != 'done'").fetchone()[0]
            )
            if incomplete:
                raise RuntimeError(f"Tamamlanmamış kaynak dosya sayısı: {incomplete}")
        rows = connection.execute(
            f'SELECT file_id, "{path_column}", class_label, expected_row_count '
            "FROM files ORDER BY file_id"
        ).fetchall()

    discovered: list[tuple[int, Path, str, int]] = []
    for file_id, raw_path, class_label, expected_rows in rows:
        path = Path(str(raw_path))
        candidates = [path] if path.is_absolute() else [
            csv_root / path,
            PROJECT_ROOT / path,
            source_database.parent / path,
            path,
        ]
        resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            attempted = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
            raise FileNotFoundError(
                f"CSV bulunamadı (file_id={file_id}): {raw_path}\n"
                f"Denenen yollar:\n  - {attempted}\n"
                "Veri farklı klasördeyse --csv-root ile N-BaIoT kökünü belirtin."
            )
        discovered.append((int(file_id), resolved, str(class_label), int(expected_rows)))
    return discovered


def family_index(raw_label: str) -> int:
    value = raw_label.lower()
    if "benign" in value:
        return 0
    if "gafgyt" in value:
        return 1
    if "mirai" in value:
        return 2
    raise ValueError(f"Bilinmeyen family_3 etiketi: {raw_label!r}")


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def split_codes(keys: np.ndarray) -> np.ndarray:
    mixed = splitmix64(keys)
    return np.where(
        mixed < np.uint64(TRAIN_LIMIT),
        np.uint8(0),
        np.where(mixed < np.uint64(VALIDATION_LIMIT), np.uint8(1), np.uint8(2)),
    )


def float32_fingerprints(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create an order-sensitive paired uint64 fingerprint of float32 rows.

    Equal model-input vectors always receive equal fingerprints.  Paired
    64-bit hashes are used only to co-locate rows; a collision would merge
    additional rows rather than allow an exact duplicate to cross splits.
    """
    matrix = np.array(values, dtype="<f4", order="C", copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Beklenen (*, 115) matris, bulunan {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Float32 özellik matrisinde sonlu olmayan değer var.")
    # Make +0.0 and -0.0 identical, as they are numerically identical inputs.
    matrix[matrix == 0.0] = 0.0
    words = matrix.view("<u4").reshape(matrix.shape)
    row_count = matrix.shape[0]
    forward = np.full(row_count, np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    reverse = np.full(row_count, np.uint64(0x84222325CBF29CE4), dtype=np.uint64)
    prime_forward = np.uint64(0x100000001B3)
    prime_reverse = np.uint64(0x9E3779B185EBCA87)
    for column in range(matrix.shape[1]):
        word_forward = words[:, column].astype(np.uint64, copy=False)
        word_reverse = words[:, matrix.shape[1] - 1 - column].astype(np.uint64, copy=False)
        forward = (forward ^ (word_forward + np.uint64(column + 1))) * prime_forward
        reverse = (reverse ^ (word_reverse + np.uint64(column + 1))) * prime_reverse
        forward ^= forward >> np.uint64(32)
        reverse ^= reverse >> np.uint64(29)
    return forward, reverse


def naive_keys(file_id: int, row_numbers: np.ndarray, seed: int) -> np.ndarray:
    file_component = np.uint64((int(file_id) * 0xD6E8FEB86659FD93) & UINT64_MASK)
    return (
        file_component
        ^ row_numbers.astype(np.uint64) * np.uint64(0xA5A3564E27F8862F)
        ^ np.uint64(seed & UINT64_MASK)
    )


def grouped_keys(forward: np.ndarray, reverse: np.ndarray, seed: int) -> np.ndarray:
    rotated = (reverse << np.uint64(32)) | (reverse >> np.uint64(32))
    return forward ^ rotated ^ np.uint64(seed & UINT64_MASK)


def split_targets(total_rows: int) -> np.ndarray:
    """Return deterministic integer 70/15/15 targets that sum to total_rows."""
    train = total_rows * 70 // 100
    validation = total_rows * 15 // 100
    return np.asarray((train, validation, total_rows - train - validation), dtype=np.int64)


def choose_balanced_split(
    remaining: np.ndarray,
    targets: np.ndarray,
    group_size: int,
    order_key: int,
) -> int:
    """Largest-first bin choice with deterministic seed-derived tie-breaking."""
    rotation = int(order_key & UINT64_MASK) % len(SPLITS)
    tie_order = tuple((rotation + offset) % len(SPLITS) for offset in range(len(SPLITS)))
    feasible = [code for code in tie_order if int(remaining[code]) >= group_size]
    if feasible:
        # Fill the split with the largest proportional deficit first.
        return max(feasible, key=lambda code: float(remaining[code]) / float(targets[code]))
    # A group larger than every remaining capacity is put where normalized
    # overshoot is smallest. Descending group-size order bounds this deviation.
    return min(
        tie_order,
        key=lambda code: (
            max(0, group_size - int(remaining[code])) / float(targets[code]),
            code,
        ),
    )


def allocate_grouped_splits(
    connection: sqlite3.Connection,
    seed: int,
    progress: Any,
) -> list[dict[str, Any]]:
    """Allocate float32 groups to balanced family-stratified raw-row targets."""
    allocation_audit: list[dict[str, Any]] = []
    connection.execute(
        "CREATE TABLE group_split_assignments ("
        "hash_forward INTEGER NOT NULL, hash_reverse INTEGER NOT NULL, "
        "grouped_split INTEGER NOT NULL, "
        "PRIMARY KEY(hash_forward,hash_reverse)) WITHOUT ROWID"
    )
    insert_sql = (
        "INSERT INTO group_split_assignments(hash_forward,hash_reverse,grouped_split) "
        "VALUES (?,?,?)"
    )
    for family in range(3):
        total_rows, group_count, largest_group = connection.execute(
            "SELECT COALESCE(SUM(raw_row_count),0), COUNT(*), "
            "COALESCE(MAX(raw_row_count),0) FROM fingerprint_stats WHERE family_label_min=?",
            (family,),
        ).fetchone()
        total_rows = int(total_rows)
        group_count = int(group_count)
        largest_group = int(largest_group)
        if total_rows <= 0 or group_count <= 0:
            raise RuntimeError(f"Family {family} için grup bulunamadı.")
        targets = split_targets(total_rows)
        assigned = np.zeros(3, dtype=np.int64)
        remaining = targets.copy()
        cursor = connection.execute(
            "SELECT hash_forward,hash_reverse,raw_row_count,group_order_key "
            "FROM fingerprint_stats WHERE family_label_min=? "
            "ORDER BY raw_row_count DESC, group_order_key, hash_forward, hash_reverse",
            (family,),
        )
        batch: list[tuple[int, int, int]] = []
        processed = 0
        last_progress = time.perf_counter()
        for hash_forward, hash_reverse, raw_row_count, group_order_key in cursor:
            group_size = int(raw_row_count)
            split_code = choose_balanced_split(
                remaining=remaining,
                targets=targets,
                group_size=group_size,
                order_key=int(group_order_key),
            )
            assigned[split_code] += group_size
            remaining[split_code] -= group_size
            batch.append((int(hash_forward), int(hash_reverse), split_code))
            processed += 1
            if len(batch) >= 50_000:
                connection.executemany(insert_sql, batch)
                batch.clear()
            now = time.perf_counter()
            if now - last_progress >= 60.0:
                progress(
                    f"Family {family}: {processed:,}/{group_count:,} float32 grup dengelendi"
                )
                last_progress = now
        if batch:
            connection.executemany(insert_sql, batch)
        deviations = assigned - targets
        percentage_point_errors = np.abs(deviations) * 100.0 / total_rows
        maximum_error = float(percentage_point_errors.max())
        if maximum_error > MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS:
            raise RuntimeError(
                f"Family {family} split dengesi tolerans dışında: "
                f"maksimum hata={maximum_error:.6f} yüzde puan"
            )
        allocation_audit.append(
            {
                "family_label": family,
                "raw_row_count": total_rows,
                "float32_group_count": group_count,
                "largest_group_raw_row_count": largest_group,
                "target_row_counts": targets.tolist(),
                "assigned_row_counts": assigned.tolist(),
                "deviation_rows": deviations.tolist(),
                "maximum_ratio_error_percentage_points": maximum_error,
            }
        )
        progress(
            f"Family {family} dengeleme tamamlandı: hedef={targets.tolist()}, "
            f"gerçek={assigned.tolist()}"
        )
    assigned_group_count = int(
        connection.execute("SELECT COUNT(*) FROM group_split_assignments").fetchone()[0]
    )
    expected_group_count = int(
        connection.execute("SELECT COUNT(*) FROM fingerprint_stats").fetchone()[0]
    )
    if assigned_group_count != expected_group_count:
        raise RuntimeError(
            f"Grouped atama sayısı geçersiz: {assigned_group_count:,} != {expected_group_count:,}"
        )
    progress("Dengeli grouped split kodları ana grup tablosuna işleniyor...")
    connection.execute(
        "UPDATE fingerprint_stats SET grouped_split=("
        "SELECT grouped_split FROM group_split_assignments g "
        "WHERE g.hash_forward=fingerprint_stats.hash_forward "
        "AND g.hash_reverse=fingerprint_stats.hash_reverse)"
    )
    unassigned = int(
        connection.execute(
            "SELECT COUNT(*) FROM fingerprint_stats WHERE grouped_split IS NULL"
        ).fetchone()[0]
    )
    if unassigned:
        raise RuntimeError(f"Atanmamış float32 grup sayısı: {unassigned:,}")
    connection.execute("DROP TABLE group_split_assignments")
    progress("Dengeli grouped split kodları doğrulandı.")
    connection.commit()
    return allocation_audit


def to_sql_int64(values: np.ndarray) -> list[int]:
    return values.view(np.int64).tolist()


def create_database(
    source_database: Path,
    output_database: Path,
    files: list[tuple[int, Path, str, int]],
    seed: int,
    chunk_size: int,
    overwrite: bool,
) -> None:
    if output_database.exists():
        if not overwrite:
            raise FileExistsError(
                f"Çıktı zaten var: {output_database}. Yeniden üretmek için --overwrite kullanın."
            )
        output_database.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = output_database.with_name(output_database.name + suffix)
        if sidecar.exists():
            sidecar.unlink()

    output_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_database, timeout=120.0)
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[{(time.perf_counter() - started) / 60:8.2f} dk] {message}", flush=True)

    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-100000")
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE source_file_integrity (
                file_id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                class_label TEXT NOT NULL,
                expected_row_count INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE assignments (
                file_id INTEGER NOT NULL,
                row_number INTEGER NOT NULL,
                hash_forward INTEGER NOT NULL,
                hash_reverse INTEGER NOT NULL,
                family_label INTEGER NOT NULL,
                naive_row_split INTEGER NOT NULL,
                group_order_key INTEGER NOT NULL,
                PRIMARY KEY (file_id, row_number)
            ) WITHOUT ROWID;
            """
        )
        insert_sql = (
            "INSERT INTO assignments(file_id,row_number,hash_forward,hash_reverse,"
            "family_label,naive_row_split,group_order_key) VALUES (?,?,?,?,?,?,?)"
        )
        total_rows = 0
        for file_number, (file_id, csv_path, raw_label, expected_rows) in enumerate(files, start=1):
            stat_before = csv_path.stat()
            family = family_index(raw_label)
            base = 0
            last_file_progress = time.perf_counter()
            for frame in pd.read_csv(csv_path, chunksize=chunk_size):
                values = frame.to_numpy(dtype=np.float32, copy=False)
                if values.shape[1] != EXPECTED_FEATURE_COUNT:
                    raise RuntimeError(f"Özellik sayısı 115 değil: {csv_path}")
                forward, reverse = float32_fingerprints(values)
                row_numbers = np.arange(base, base + len(values), dtype=np.uint64)
                naive = split_codes(naive_keys(file_id, row_numbers, seed))
                order_keys = grouped_keys(forward, reverse, seed)
                records: Iterator[tuple[int, int, int, int, int, int, int]] = zip(
                    [file_id] * len(values),
                    row_numbers.astype(np.int64).tolist(),
                    to_sql_int64(forward),
                    to_sql_int64(reverse),
                    [family] * len(values),
                    naive.astype(np.int64).tolist(),
                    to_sql_int64(order_keys),
                )
                connection.executemany(insert_sql, records)
                base += len(values)
                total_rows += len(values)
                now = time.perf_counter()
                if now - last_file_progress >= 60.0:
                    progress(
                        f"Dosya {file_number}/{len(files)} işleniyor: {csv_path.name} "
                        f"({base:,}/{expected_rows:,} satır)"
                    )
                    last_file_progress = now
            if base != expected_rows:
                raise RuntimeError(
                    f"Satır uyuşmazlığı ({csv_path.name}): okunan={base:,}, beklenen={expected_rows:,}"
                )
            stat_after = csv_path.stat()
            if (
                stat_before.st_size != stat_after.st_size
                or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            ):
                raise RuntimeError(f"CSV işleme sırasında değişti: {csv_path}")
            progress(f"Dosya {file_number}/{len(files)} SHA-256 özeti alınıyor: {csv_path.name}")
            file_sha256 = sha256(csv_path)
            connection.execute(
                "INSERT INTO source_file_integrity("
                "file_id,file_path,class_label,expected_row_count,file_size_bytes,file_mtime_ns,sha256"
                ") VALUES (?,?,?,?,?,?,?)",
                (
                    file_id,
                    str(csv_path),
                    raw_label,
                    expected_rows,
                    int(stat_after.st_size),
                    int(stat_after.st_mtime_ns),
                    file_sha256,
                ),
            )
            connection.commit()
            progress(f"Dosya {file_number}/{len(files)} tamamlandı: {csv_path.name} ({base:,} satır)")

        if total_rows != EXPECTED_RAW_ROW_COUNT:
            raise RuntimeError(
                f"Toplam ham kayıt beklenenden farklı: {total_rows:,} != {EXPECTED_RAW_ROW_COUNT:,}"
            )

        progress("Float32 parmak izi grupları özetleniyor...")
        connection.execute(
            "CREATE INDEX idx_assignments_hash ON assignments(hash_forward, hash_reverse)"
        )
        connection.executescript(
            """
            CREATE TABLE fingerprint_stats AS
            SELECT
                hash_forward,
                hash_reverse,
                MIN(group_order_key) AS group_order_key,
                COUNT(*) AS raw_row_count,
                SUM(naive_row_split = 0) AS naive_train_rows,
                SUM(naive_row_split = 1) AS naive_validation_rows,
                SUM(naive_row_split = 2) AS naive_test_rows,
                CAST(NULL AS INTEGER) AS grouped_split,
                MIN(family_label) AS family_label_min,
                MAX(family_label) AS family_label_max
            FROM assignments
            GROUP BY hash_forward, hash_reverse;
            CREATE UNIQUE INDEX idx_fingerprint_stats_hash
            ON fingerprint_stats(hash_forward, hash_reverse);
            """
        )
        progress("Eğitim erişim indeksleri oluşturuluyor...")
        connection.execute(
            "CREATE INDEX idx_assignments_naive_file_row "
            "ON assignments(naive_row_split, file_id, row_number)"
        )
        cross_family = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats WHERE family_label_min != family_label_max"
            ).fetchone()[0]
        )
        if cross_family:
            raise RuntimeError(
                f"Aynı float32 girdiye birden fazla family_3 etiketi atanmış: {cross_family:,} grup"
            )
        progress("Float32 grupları family_3 katmanlı ham-kayıt hedeflerine dengeleniyor...")
        allocation_audit = allocate_grouped_splits(connection, seed, progress)
        connection.execute(
            "CREATE INDEX idx_fingerprint_stats_grouped "
            "ON fingerprint_stats(grouped_split, hash_forward, hash_reverse)"
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (
                ("protocol", "vs2_float32_leakage_ablation"),
                ("protocol_version", PROTOCOL_VERSION),
                ("seed", str(seed)),
                ("split_ratios", "0.70/0.15/0.15"),
                ("source_database", str(source_database)),
                ("source_database_sha256", sha256(source_database)),
                ("source_file_count", str(len(files))),
                ("feature_dtype", "float32"),
                ("feature_count", str(EXPECTED_FEATURE_COUNT)),
                ("task", "family_3"),
                ("grouped_assignment_method", "family-stratified largest-first raw-row balancing"),
                ("validation_status", "pending"),
                ("allocation_audit", json.dumps(allocation_audit, ensure_ascii=False)),
                ("builder_script_sha256", sha256(Path(__file__).resolve())),
                ("generated_at_utc", utc_now()),
            ),
        )
        connection.commit()
        progress("Yeni split veritabanı tamamlandı.")
    finally:
        connection.close()


def build_reports(database: Path, report_directory: Path, seed: int) -> None:
    records: list[dict[str, Any]] = []
    overlap_records: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    with sqlite3.connect(database) as connection:
        assignment_count = int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        if assignment_count != EXPECTED_RAW_ROW_COUNT:
            raise RuntimeError(
                f"Atama tablosu satır sayısı geçersiz: {assignment_count:,} != "
                f"{EXPECTED_RAW_ROW_COUNT:,}"
            )
        source_file_count = int(
            connection.execute("SELECT COUNT(*) FROM source_file_integrity").fetchone()[0]
        )
        if source_file_count != EXPECTED_SOURCE_FILE_COUNT:
            raise RuntimeError(
                f"Kaynak dosya sayısı geçersiz: {source_file_count} != {EXPECTED_SOURCE_FILE_COUNT}"
            )
        group_count = int(connection.execute("SELECT COUNT(*) FROM fingerprint_stats").fetchone()[0])
        if group_count != EXPECTED_FLOAT32_GROUP_COUNT:
            raise RuntimeError(
                "Float32 grup sayısı önceki tam-korpus denetimiyle uyuşmuyor: "
                f"{group_count:,} != {EXPECTED_FLOAT32_GROUP_COUNT:,}"
            )
        invalid_grouped_assignment = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats "
                "WHERE grouped_split IS NULL OR grouped_split NOT IN (0,1,2)"
            ).fetchone()[0]
        )
        cross_family = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats WHERE family_label_min != family_label_max"
            ).fetchone()[0]
        )
        if invalid_grouped_assignment:
            raise RuntimeError(
                f"Geçersiz/atanmamış grouped split sayısı: {invalid_grouped_assignment:,}"
            )
        # fingerprint_stats has exactly one row and one split per paired
        # float32 fingerprint; an identical model input cannot span splits.
        grouped_overlap = 0
        if cross_family:
            raise RuntimeError(
                f"Aynı float32 girdiye birden fazla family_3 etiketi atanmış: {cross_family:,} grup"
            )

        naive_columns = ("naive_train_rows", "naive_validation_rows", "naive_test_rows")
        for split_code, column in enumerate(naive_columns):
            row_count, unique_count = connection.execute(
                f"SELECT COALESCE(SUM({column}),0), SUM({column}>0) FROM fingerprint_stats"
            ).fetchone()
            records.append(
                {
                    "arm": "naive_row",
                    "split": CODE_TO_SPLIT[split_code],
                    "row_count": int(row_count),
                    "unique_float32_fingerprint_count": int(unique_count),
                }
            )
        for split_code in range(3):
            row_count, unique_count = connection.execute(
                "SELECT COALESCE(SUM(raw_row_count),0), COUNT(*) FROM fingerprint_stats "
                "WHERE grouped_split=?",
                (split_code,),
            ).fetchone()
            records.append(
                {
                    "arm": "grouped_float32",
                    "split": CODE_TO_SPLIT[split_code],
                    "row_count": int(row_count),
                    "unique_float32_fingerprint_count": int(unique_count),
                }
            )

        for family in range(3):
            family_total = int(
                connection.execute(
                    "SELECT COALESCE(SUM(raw_row_count),0) FROM fingerprint_stats "
                    "WHERE family_label_min=?",
                    (family,),
                ).fetchone()[0]
            )
            targets = split_targets(family_total)
            for split_code, split_name in enumerate(SPLITS):
                naive_column = naive_columns[split_code]
                naive_count = int(
                    connection.execute(
                        f"SELECT COALESCE(SUM({naive_column}),0) FROM fingerprint_stats "
                        "WHERE family_label_min=?",
                        (family,),
                    ).fetchone()[0]
                )
                grouped_count = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(raw_row_count),0) FROM fingerprint_stats "
                        "WHERE family_label_min=? AND grouped_split=?",
                        (family, split_code),
                    ).fetchone()[0]
                )
                for arm, count in (("naive_row", naive_count), ("grouped_float32", grouped_count)):
                    actual_pct = 100.0 * count / family_total
                    target_pct = 100.0 * int(targets[split_code]) / family_total
                    error = abs(actual_pct - target_pct)
                    class_records.append(
                        {
                            "arm": arm,
                            "family": FAMILY_NAMES[family],
                            "family_label": family,
                            "split": split_name,
                            "row_count": count,
                            "target_row_count": int(targets[split_code]),
                            "row_percentage_within_family": actual_pct,
                            "target_percentage_within_family": target_pct,
                            "absolute_ratio_error_percentage_points": error,
                        }
                    )
                    allowed_error = (
                        MAX_NAIVE_RATIO_ERROR_PERCENTAGE_POINTS
                        if arm == "naive_row"
                        else MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS
                    )
                    if error > allowed_error:
                        raise RuntimeError(
                            f"{arm}/family{family}/{split_name} oran hatası tolerans dışında: "
                            f"{error:.6f} yüzde puan"
                        )

        for arm in ("naive_row", "grouped_float32"):
            for left, right in SPLIT_PAIRS:
                if arm == "naive_row":
                    columns = naive_columns
                    count = int(
                        connection.execute(
                            f"SELECT COUNT(*) FROM fingerprint_stats WHERE "
                            f"{columns[left]}>0 AND {columns[right]}>0"
                        ).fetchone()[0]
                    )
                else:
                    count = 0
                overlap_records.append(
                    {
                        "arm": arm,
                        "split_left": CODE_TO_SPLIT[left],
                        "split_right": CODE_TO_SPLIT[right],
                        "overlapping_float32_fingerprint_count": count,
                    }
                )

        for split_code, column in ((1, "naive_validation_rows"), (2, "naive_test_rows")):
            evaluation_rows, exposed_rows, exposed_groups = connection.execute(
                f"SELECT COALESCE(SUM({column}),0), "
                f"COALESCE(SUM(CASE WHEN naive_train_rows>0 THEN {column} ELSE 0 END),0), "
                f"SUM(naive_train_rows>0 AND {column}>0) FROM fingerprint_stats"
            ).fetchone()
            evaluation_rows = int(evaluation_rows)
            exposed_rows = int(exposed_rows)
            exposures.append(
                {
                    "split": CODE_TO_SPLIT[split_code],
                    "evaluation_rows": evaluation_rows,
                    "train_exposed_rows": exposed_rows,
                    "train_exposed_unique_float32_fingerprints": int(exposed_groups),
                    "train_exposed_row_percentage": 100.0 * exposed_rows / evaluation_rows,
                }
            )

    distribution = pd.DataFrame(records)
    distribution["row_percentage"] = (
        distribution["row_count"]
        * 100.0
        / distribution.groupby("arm")["row_count"].transform("sum")
    )
    overlap = pd.DataFrame(overlap_records)
    class_distribution = pd.DataFrame(class_records)
    report_directory.mkdir(parents=True, exist_ok=True)
    distribution_file = report_directory / "nbaiot_vs2_float32_leakage_split_distribution.csv"
    overlap_file = report_directory / "nbaiot_vs2_float32_leakage_overlap_by_pair.csv"
    class_distribution_file = (
        report_directory / "nbaiot_vs2_float32_leakage_family_split_distribution.csv"
    )
    summary_file = report_directory / "nbaiot_vs2_float32_leakage_split_summary.json"
    distribution.to_csv(distribution_file, index=False)
    overlap.to_csv(overlap_file, index=False)
    class_distribution.to_csv(class_distribution_file, index=False)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE metadata SET value='passed' WHERE key='validation_status'"
        )
        if connection.total_changes != 1:
            raise RuntimeError("validation_status metadata satırı güncellenemedi.")
        allocation_json = connection.execute(
            "SELECT value FROM metadata WHERE key='allocation_audit'"
        ).fetchone()
        connection.commit()
    allocation_audit = json.loads(str(allocation_json[0])) if allocation_json else []
    summary = {
        "generated_at_utc": utc_now(),
        "protocol": "vs2_float32_leakage_ablation",
        "protocol_version": PROTOCOL_VERSION,
        "status": "split_indices_locked_before_training",
        "seed": seed,
        "task": "family_3",
        "feature_space_used_for_grouping": "115-feature float32 model input",
        "source_row_count": assignment_count,
        "source_file_count": source_file_count,
        "source_files_sha256_locked": True,
        "builder_script_sha256": sha256(Path(__file__).resolve()),
        "unique_float32_fingerprint_count": group_count,
        "cross_family_label_fingerprint_count": cross_family,
        "maximum_allowed_ratio_error_percentage_points": {
            "naive_row": MAX_NAIVE_RATIO_ERROR_PERCENTAGE_POINTS,
            "grouped_float32": MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS,
        },
        "family_stratified_allocation_audit": allocation_audit,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "arms": {
            "naive_row": {
                "assignment_unit": "raw row",
                "cross_split_exact_float32_overlap_allowed": True,
                "validation_and_test_train_exposure": exposures,
            },
            "grouped_float32": {
                "assignment_unit": "paired fingerprint of float32 model-input vector",
                "allocation_method": "family-stratified largest-first raw-row balancing",
                "cross_split_exact_float32_overlap_allowed": False,
                "observed_pairwise_overlap_sum": 0,
            },
        },
        "scientific_controls": {
            "same_raw_row_universe": True,
            "same_split_seed": True,
            "same_split_ratios": True,
            "primary_experimental_factor": "assignment unit: raw row vs float32 group",
            "family_split_ratios_audited_with_fail_closed_tolerances": True,
            "training_configuration_must_be_identical_between_arms": True,
            "train_only_scaler_and_class_weight_fit_per_arm": True,
        },
        "artifacts": {
            "assignment_database": str(database),
            "assignment_database_sha256": sha256(database),
            "split_distribution_csv": str(distribution_file),
            "overlap_by_pair_csv": str(overlap_file),
            "family_split_distribution_csv": str(class_distribution_file),
        },
    }
    atomic_json(summary_file, summary)

    print("=" * 78)
    print("37. aşama tamamlandı: float32 sızıntı-ablasyon splitleri hazır")
    print(f"Ham kayıt                         : {assignment_count:,}")
    print(f"Benzersiz float32 parmak izi      : {group_count:,}")
    print(f"Gruplu kolda split örtüşmesi      : {grouped_overlap:,}")
    print(f"Family_3 etiket çakışmalı grup    : {cross_family:,}")
    print(f"SHA-256 ile kilitlenen kaynak CSV : {source_file_count:,}")
    for arm in ("naive_row", "grouped_float32"):
        arm_rows = distribution[distribution["arm"] == arm]
        formatted = ", ".join(
            f"{row.split}={int(row.row_count):,} (%{float(row.row_percentage):.4f})"
            for row in arm_rows.itertuples(index=False)
        )
        print(f"{arm:32s}: {formatted}")
    for item in exposures:
        print(
            f"Naif {item['split']} train-maruz kalma: "
            f"{item['train_exposed_rows']:,}/{item['evaluation_rows']:,} "
            f"(%{item['train_exposed_row_percentage']:.4f})"
        )
    print("Eski deney ve sonuç dosyaları değiştirilmedi.")
    print("=" * 78)


def main() -> None:
    args = parse_arguments()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size pozitif olmalıdır.")
    if args.seed != LOCKED_SPLIT_SEED:
        raise ValueError(f"Split seed kilitlidir: {LOCKED_SPLIT_SEED}")
    source_database = args.source_database.resolve()
    csv_root = args.csv_root.resolve()
    output_database = args.output_database.resolve()
    report_directory = args.report_directory.resolve()
    if not source_database.exists():
        raise FileNotFoundError(f"Kaynak veritabanı bulunamadı: {source_database}")
    if source_database == output_database:
        raise ValueError("Kaynak ve çıktı veritabanı aynı olamaz.")
    output_database.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_database.parent).free
    if free_bytes < MINIMUM_STAGE37_FREE_BYTES:
        raise RuntimeError(
            "37. aşama için en az 5 GiB boş alan gerekir; "
            f"bulunan={free_bytes / 1024**3:.2f} GiB"
        )
    files = source_files(source_database, csv_root)
    if len(files) != EXPECTED_SOURCE_FILE_COUNT:
        raise RuntimeError(
            f"Kaynak CSV sayısı geçersiz: {len(files)} != {EXPECTED_SOURCE_FILE_COUNT}"
        )
    expected_total = sum(item[3] for item in files)
    if expected_total != EXPECTED_RAW_ROW_COUNT:
        raise RuntimeError(
            f"files tablosundaki toplam kayıt beklenenden farklı: "
            f"{expected_total:,} != {EXPECTED_RAW_ROW_COUNT:,}"
        )
    create_database(
        source_database=source_database,
        output_database=output_database,
        files=files,
        seed=args.seed,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    build_reports(output_database, report_directory, args.seed)


if __name__ == "__main__":
    main()