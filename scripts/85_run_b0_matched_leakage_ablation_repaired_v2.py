"""Run the corrected leakage ablation with the locked B0 training protocol.

The two arms produced by stage 37 contain the same raw N-BaIoT rows and differ
only in the split assignment unit.  Both arms use the B0 ``tinyml_mlp`` model,
AdamW, inverse-square-root class weights, train-only standardization, identical
early stopping, and the five locked seeds.

The experiment is an auxiliary raw-frequency leakage-sensitivity analysis.  It
does not replace or overwrite the 150 primary compression experiments.
Put this file in ``scripts/`` inside TinyML_IDS_IEEE.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nbaiot_models import create_model  # noqa: E402


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "splits" / "vs2_float32_leakage_ablation_seed2026.sqlite"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "protocols" / "nbaiot_family3_fp32_baseline_protocol_v1.json"
DEFAULT_CSV_ROOT = PROJECT_ROOT / "data" / "interim" / "nbaiot_unpacked"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "processed" / "vs2_float32_leakage_ablation"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "vs2" / "leakage_ablation_b0_matched"

ARMS = ("naive_row", "grouped_float32")
SPLITS = ("train", "validation", "test")
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
SEEDS = (42, 123, 2026, 3407, 8192)
CLASS_NAMES = ("benign", "gafgyt", "mirai")
EXPECTED_FEATURE_COUNT = 115
EXPECTED_PARAMETER_COUNT = 9_603
EXPECTED_RAW_ROW_COUNT = 7_062_606
EXPECTED_FLOAT32_GROUP_COUNT = 2_278_176
EXPECTED_LARGEST_FLOAT32_GROUP_RAW_ROWS = 965_536
EXPECTED_SOURCE_FILE_COUNT = 89
LOCKED_SPLIT_SEED = 2026
PROTOCOL_VERSION = "2.2"
SCALER_AUDIT_TOLERANCE = 1e-4
MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS = 0.01
MAX_NAIVE_RATIO_ERROR_PERCENTAGE_POINTS = 0.25
MINIMUM_CACHE_SAFETY_BYTES = 2 * 1024**3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 38: B0-matched leakage ablation")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=DEFAULT_CSV_ROOT,
        help="N-BaIoT cihaz klasÃƒÂ¶rlerini iÃƒÂ§eren kÃƒÂ¶k dizin.",
    )
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="YalnÃ„Â±z tamamlanmamÃ„Â±Ã…Å¸ VS2 run klasÃƒÂ¶rlerini silip yeniden baÃ…Å¸latÃ„Â±r.",
    )
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} JSON ile uyumlu deÃ„Å¸il.")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"JSON kÃƒÂ¶kÃƒÂ¼ sÃƒÂ¶zlÃƒÂ¼k deÃ„Å¸il: {path}")
    return document


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def database_metadata(database: Path) -> dict[str, str]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT key,value FROM metadata").fetchall()
    metadata = {str(key): str(value) for key, value in rows}
    expected = {
        "protocol": "vs2_float32_leakage_ablation",
        "protocol_version": PROTOCOL_VERSION,
        "feature_dtype": "float32",
        "feature_count": "115",
        "task": "family_3",
        "seed": str(LOCKED_SPLIT_SEED),
        "validation_status": "passed",
        "source_file_count": str(EXPECTED_SOURCE_FILE_COUNT),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Split veritabanÃ„Â± metadata uyuÃ…Å¸mazlÃ„Â±Ã„Å¸Ã„Â±: {key}={metadata.get(key)!r}")
    return metadata


def source_database_path(database: Path) -> Path:
    metadata = database_metadata(database)
    path = Path(metadata["source_database"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Kaynak parmak izi veritabanÃ„Â± bulunamadÃ„Â±: {path}")
    return path.resolve()


def discover_files(source_database: Path, csv_root: Path) -> list[tuple[int, Path, str, int]]:
    with sqlite3.connect(source_database) as connection:
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(files)")]
        path_candidates = ("file_path", "csv_path", "path", "relative_path", "source_file")
        path_column = next((name for name in path_candidates if name in columns), None)
        required = {"file_id", "class_label", "expected_row_count"}
        if path_column is None or not required.issubset(columns):
            raise RuntimeError(f"Desteklenmeyen source.files Ã…Å¸emasÃ„Â±: {columns}")
        rows = connection.execute(
            f'SELECT file_id,"{path_column}",class_label,expected_row_count '
            "FROM files ORDER BY file_id"
        ).fetchall()
    result: list[tuple[int, Path, str, int]] = []
    for file_id, raw_path, label, expected_rows in rows:
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
                f"CSV bulunamadÃ„Â± (file_id={file_id}): {raw_path}\n"
                f"Denenen yollar:\n  - {attempted}\n"
                "Veri farklÃ„Â± klasÃƒÂ¶rdeyse --csv-root ile N-BaIoT kÃƒÂ¶kÃƒÂ¼nÃƒÂ¼ belirtin."
            )
        result.append((int(file_id), resolved, str(label), int(expected_rows)))
    return result


def family_label(raw: str) -> int:
    value = raw.lower()
    if "benign" in value:
        return 0
    if "gafgyt" in value:
        return 1
    if "mirai" in value:
        return 2
    raise ValueError(f"Bilinmeyen family_3 etiketi: {raw!r}")


def split_column(arm: str) -> str:
    if arm == "naive_row":
        return "naive_row_split"
    if arm == "grouped_float32":
        return "grouped_split"
    raise ValueError(f"Bilinmeyen kol: {arm}")


def split_counts(database: Path, arm: str) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        if arm == "naive_row":
            rows = connection.execute(
                "SELECT naive_row_split,COUNT(*) FROM assignments GROUP BY naive_row_split"
            ).fetchall()
        elif arm == "grouped_float32":
            rows = connection.execute(
                "SELECT grouped_split,SUM(raw_row_count) FROM fingerprint_stats "
                "GROUP BY grouped_split"
            ).fetchall()
        else:
            raise ValueError(f"Bilinmeyen kol: {arm}")
    by_code = {int(code): int(count) for code, count in rows}
    if set(by_code) != {0, 1, 2}:
        raise RuntimeError(f"{arm} split kodlarÃ„Â± geÃƒÂ§ersiz: {by_code}")
    result = {name: by_code[code] for name, code in SPLIT_CODES.items()}
    if sum(result.values()) != EXPECTED_RAW_ROW_COUNT:
        raise RuntimeError(f"{arm} toplam kayÃ„Â±t sayÃ„Â±sÃ„Â± geÃƒÂ§ersiz: {sum(result.values()):,}")
    return result


def split_class_counts(database: Path, arm: str) -> dict[str, list[int]]:
    with sqlite3.connect(database) as connection:
        if arm == "naive_row":
            rows = connection.execute(
                "SELECT naive_row_split,family_label,COUNT(*) FROM assignments "
                "GROUP BY naive_row_split,family_label"
            ).fetchall()
        elif arm == "grouped_float32":
            rows = connection.execute(
                "SELECT grouped_split,family_label_min,SUM(raw_row_count) "
                "FROM fingerprint_stats GROUP BY grouped_split,family_label_min"
            ).fetchall()
        else:
            raise ValueError(f"Bilinmeyen kol: {arm}")
    result = {split: [0, 0, 0] for split in SPLITS}
    for split_code, family, count in rows:
        result[SPLITS[int(split_code)]][int(family)] = int(count)
    for split, values in result.items():
        if any(value <= 0 for value in values):
            raise RuntimeError(f"{arm}/{split}: eksik family_3 sÃ„Â±nÃ„Â±fÃ„Â±: {values}")
    return result


def selected_rows(database: Path, file_id: int, arm: str, split: str) -> np.ndarray:
    split_code = SPLIT_CODES[split]
    with sqlite3.connect(database) as connection:
        if arm == "naive_row":
            rows = connection.execute(
                "SELECT row_number FROM assignments "
                "WHERE naive_row_split=? AND file_id=? ORDER BY row_number",
                (split_code, file_id),
            ).fetchall()
        elif arm == "grouped_float32":
            rows = connection.execute(
                "SELECT a.row_number FROM assignments a "
                "JOIN fingerprint_stats f "
                "ON f.hash_forward=a.hash_forward AND f.hash_reverse=a.hash_reverse "
                "WHERE f.grouped_split=? AND a.file_id=? ORDER BY a.row_number",
                (split_code, file_id),
            ).fetchall()
        else:
            raise ValueError(f"Bilinmeyen kol: {arm}")
    return np.fromiter((int(row[0]) for row in rows), dtype=np.int64, count=len(rows))


def validate_database(database: Path) -> dict[str, Any]:
    """Fail closed unless the stage-37 database proves every scientific lock."""
    metadata = database_metadata(database)
    ratio_audit: list[dict[str, Any]] = []
    with sqlite3.connect(database) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Split veritabanÃ„Â± quick_check baÃ…Å¸arÃ„Â±sÃ„Â±z: {quick_check}")
        assignment_count = int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        group_count, grouped_raw_rows = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(raw_row_count),0) FROM fingerprint_stats"
        ).fetchone()
        group_count = int(group_count)
        grouped_raw_rows = int(grouped_raw_rows)
        cross_family = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats WHERE family_label_min!=family_label_max"
            ).fetchone()[0]
        )
        invalid_grouped = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats "
                "WHERE grouped_split IS NULL OR grouped_split NOT IN (0,1,2)"
            ).fetchone()[0]
        )
        largest_size, largest_split = connection.execute(
            "SELECT raw_row_count,grouped_split FROM fingerprint_stats "
            "ORDER BY raw_row_count DESC LIMIT 1"
        ).fetchone()
        naive_overlap = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprint_stats WHERE "
                "(naive_train_rows>0 AND naive_validation_rows>0) OR "
                "(naive_train_rows>0 AND naive_test_rows>0) OR "
                "(naive_validation_rows>0 AND naive_test_rows>0)"
            ).fetchone()[0]
        )
        grouped_counts = {
            int(code): int(count)
            for code, count in connection.execute(
                "SELECT grouped_split,SUM(raw_row_count) FROM fingerprint_stats "
                "GROUP BY grouped_split"
            ).fetchall()
        }
        naive_counts = {
            int(code): int(count)
            for code, count in connection.execute(
                "SELECT naive_row_split,COUNT(*) FROM assignments GROUP BY naive_row_split"
            ).fetchall()
        }
        for family in range(3):
            family_total = int(
                connection.execute(
                    "SELECT SUM(raw_row_count) FROM fingerprint_stats WHERE family_label_min=?",
                    (family,),
                ).fetchone()[0]
            )
            targets = (
                family_total * 70 // 100,
                family_total * 15 // 100,
                family_total - family_total * 70 // 100 - family_total * 15 // 100,
            )
            for split_code, split in enumerate(SPLITS):
                naive_column = ("naive_train_rows", "naive_validation_rows", "naive_test_rows")[
                    split_code
                ]
                naive_value = int(
                    connection.execute(
                        f"SELECT SUM({naive_column}) FROM fingerprint_stats WHERE family_label_min=?",
                        (family,),
                    ).fetchone()[0]
                )
                grouped_value = int(
                    connection.execute(
                        "SELECT SUM(raw_row_count) FROM fingerprint_stats "
                        "WHERE family_label_min=? AND grouped_split=?",
                        (family, split_code),
                    ).fetchone()[0]
                )
                for arm, value, tolerance in (
                    ("naive_row", naive_value, MAX_NAIVE_RATIO_ERROR_PERCENTAGE_POINTS),
                    (
                        "grouped_float32",
                        grouped_value,
                        MAX_GROUPED_RATIO_ERROR_PERCENTAGE_POINTS,
                    ),
                ):
                    error = abs(value - targets[split_code]) * 100.0 / family_total
                    ratio_audit.append(
                        {
                            "arm": arm,
                            "family": CLASS_NAMES[family],
                            "split": split,
                            "row_count": value,
                            "target_row_count": targets[split_code],
                            "absolute_ratio_error_percentage_points": error,
                            "tolerance_percentage_points": tolerance,
                            "passed": error <= tolerance,
                        }
                    )
    locks = {
        "assignment_count": assignment_count == EXPECTED_RAW_ROW_COUNT,
        "group_count": group_count == EXPECTED_FLOAT32_GROUP_COUNT,
        "grouped_raw_rows": grouped_raw_rows == EXPECTED_RAW_ROW_COUNT,
        "cross_family_zero": cross_family == 0,
        "all_grouped_assignments_valid": invalid_grouped == 0,
        "all_grouped_splits_present": set(grouped_counts) == {0, 1, 2},
        "all_naive_splits_present": set(naive_counts) == {0, 1, 2},
        "grouped_counts_sum": sum(grouped_counts.values()) == EXPECTED_RAW_ROW_COUNT,
        "naive_counts_sum": sum(naive_counts.values()) == EXPECTED_RAW_ROW_COUNT,
        "largest_group_count": int(largest_size) == EXPECTED_LARGEST_FLOAT32_GROUP_RAW_ROWS,
        "largest_group_kept_in_train": int(largest_split) == SPLIT_CODES["train"],
        "naive_cross_split_overlap_observed": naive_overlap > 0,
        "family_split_ratio_tolerances": all(item["passed"] for item in ratio_audit),
    }
    failed = [name for name, passed in locks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage-37 bilimsel kilitleri baÃ…Å¸arÃ„Â±sÃ„Â±z: {failed}")
    return {
        "metadata": metadata,
        "quick_check": quick_check,
        "assignment_count": assignment_count,
        "float32_group_count": group_count,
        "cross_family_group_count": cross_family,
        "invalid_grouped_assignment_count": invalid_grouped,
        "largest_group_raw_row_count": int(largest_size),
        "largest_group_split": "train",
        "naive_cross_split_overlapping_group_count": naive_overlap,
        "naive_split_counts": {SPLITS[code]: count for code, count in naive_counts.items()},
        "grouped_split_counts": {SPLITS[code]: count for code, count in grouped_counts.items()},
        "family_split_ratio_audit": ratio_audit,
        "locks": locks,
    }


def validate_source_files(database: Path, csv_root: Path) -> dict[str, Any]:
    """Verify that stage 38 will read the exact CSV bytes audited by stage 37."""
    metadata = database_metadata(database)
    source_database = source_database_path(database)
    observed_source_database_sha256 = sha256(source_database)
    expected_source_database_sha256 = metadata.get("source_database_sha256")
    if observed_source_database_sha256 != expected_source_database_sha256:
        raise RuntimeError(
            "Kaynak denetim veritabanÃ„Â± SHA-256 deÃ„Å¸eri stage 37 sonrasÃ„Â±nda deÃ„Å¸iÃ…Å¸miÃ…Å¸."
        )
    discovered = {
        file_id: (path, label, rows)
        for file_id, path, label, rows in discover_files(source_database, csv_root)
    }
    with sqlite3.connect(database) as connection:
        integrity_rows = connection.execute(
            "SELECT file_id,file_path,class_label,expected_row_count,file_size_bytes,sha256 "
            "FROM source_file_integrity ORDER BY file_id"
        ).fetchall()
    expected_file_count = int(metadata.get("source_file_count", "-1"))
    if len(integrity_rows) != expected_file_count or set(discovered) != {
        int(row[0]) for row in integrity_rows
    }:
        raise RuntimeError("Kaynak CSV envanteri stage 37 ile uyuÃ…Å¸muyor.")
    total_bytes = 0
    for position, row in enumerate(integrity_rows, start=1):
        file_id, stored_path, stored_label, stored_rows, stored_size, stored_sha256 = row
        current_path, current_label, current_rows = discovered[int(file_id)]
        if current_path.resolve() != Path(str(stored_path)).resolve():
            raise RuntimeError(f"Kaynak CSV yolu deÃ„Å¸iÃ…Å¸miÃ…Å¸ (file_id={file_id}).")
        if current_label != str(stored_label) or current_rows != int(stored_rows):
            raise RuntimeError(f"Kaynak CSV etiketi/satÃ„Â±r sayÃ„Â±sÃ„Â± deÃ„Å¸iÃ…Å¸miÃ…Å¸ (file_id={file_id}).")
        stat = current_path.stat()
        if stat.st_size != int(stored_size):
            raise RuntimeError(f"Kaynak CSV boyutu deÃ„Å¸iÃ…Å¸miÃ…Å¸: {current_path}")
        observed_sha256 = sha256(current_path)
        if observed_sha256 != str(stored_sha256):
            raise RuntimeError(f"Kaynak CSV SHA-256 uyuÃ…Å¸muyor: {current_path}")
        total_bytes += int(stat.st_size)
        print(
            f"[kaynak doÃ„Å¸rulama] {position}/{len(integrity_rows)}: {current_path.name}",
            flush=True,
        )
    return {
        "source_database": str(source_database),
        "source_database_sha256": observed_source_database_sha256,
        "source_file_count": len(integrity_rows),
        "source_total_bytes": total_bytes,
        "all_source_file_sha256_values_match": True,
    }


def inverse_sqrt_mean1(class_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.shape != (3,) or np.any(counts <= 0):
        raise RuntimeError(f"GeÃƒÂ§ersiz eÃ„Å¸itim sÃ„Â±nÃ„Â±f sayÃ„Â±larÃ„Â±: {counts.tolist()}")
    raw = 1.0 / np.sqrt(counts)
    return (raw / raw.mean()).astype(np.float32)


def float32_fingerprints(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paired order-sensitive fingerprints of the exact model-facing float32 rows."""
    matrix = np.array(values, dtype="<f4", order="C", copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"Model girdisi boyutu geÃƒÂ§ersiz: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Model girdisinde sonlu olmayan deÃ„Å¸er var.")
    matrix[matrix == 0.0] = 0.0
    words = matrix.view("<u4").reshape(matrix.shape)
    forward = np.full(matrix.shape[0], np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    reverse = np.full(matrix.shape[0], np.uint64(0x84222325CBF29CE4), dtype=np.uint64)
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


def to_sql_int64(values: np.ndarray) -> list[int]:
    return values.view(np.int64).tolist()


def audit_scaled_grouped_overlap(
    arm_directory: Path,
    chunk_size: int,
) -> dict[str, Any]:
    """Prove zero overlap after StandardScaler and final float32 conversion."""
    audit_database = arm_directory / "scaled_model_input_overlap_audit.sqlite"
    for path in (
        audit_database,
        audit_database.with_name(audit_database.name + "-wal"),
        audit_database.with_name(audit_database.name + "-shm"),
    ):
        if path.exists():
            path.unlink()
    mean = np.load(arm_directory / "mean.npy")
    scale = np.load(arm_directory / "std.npy")
    connection = sqlite3.connect(audit_database, timeout=120.0)
    started = time.perf_counter()
    processed_rows = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE scaled_fingerprints ("
            "hash_forward INTEGER NOT NULL, hash_reverse INTEGER NOT NULL, "
            "split_code INTEGER NOT NULL, "
            "PRIMARY KEY(hash_forward,hash_reverse,split_code)) WITHOUT ROWID"
        )
        insert_sql = (
            "INSERT OR IGNORE INTO scaled_fingerprints(hash_forward,hash_reverse,split_code) "
            "VALUES (?,?,?)"
        )
        for split_code, split in enumerate(SPLITS):
            x = np.load(arm_directory / f"X_{split}.npy", mmap_mode="r")
            last_progress = time.perf_counter()
            for start in range(0, len(x), chunk_size):
                raw = np.asarray(x[start : start + chunk_size], dtype=np.float64)
                scaled = ((raw - mean) / scale).astype(np.float32)
                forward, reverse = float32_fingerprints(scaled)
                connection.executemany(
                    insert_sql,
                    zip(
                        to_sql_int64(forward),
                        to_sql_int64(reverse),
                        [split_code] * len(scaled),
                    ),
                )
                processed_rows += len(scaled)
                now = time.perf_counter()
                if now - last_progress >= 60.0:
                    print(
                        f"[grouped_float32 gerÃƒÂ§ek model-girdisi denetimi] {split}: "
                        f"{min(start + chunk_size, len(x)):,}/{len(x):,}",
                        flush=True,
                    )
                    last_progress = now
            connection.commit()
        unique_by_split = {
            SPLITS[int(code)]: int(count)
            for code, count in connection.execute(
                "SELECT split_code,COUNT(*) FROM scaled_fingerprints GROUP BY split_code"
            ).fetchall()
        }
        overlap_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT hash_forward,hash_reverse FROM scaled_fingerprints "
                "GROUP BY hash_forward,hash_reverse HAVING COUNT(*)>1)"
            ).fetchone()[0]
        )
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Ãƒâ€“lÃƒÂ§eklenmiÃ…Å¸ girdi denetim DB quick_check baÃ…Å¸arÃ„Â±sÃ„Â±z: {quick_check}")
    finally:
        connection.close()
    if processed_rows != EXPECTED_RAW_ROW_COUNT:
        raise RuntimeError(
            f"Ãƒâ€“lÃƒÂ§eklenmiÃ…Å¸ girdi denetim satÃ„Â±r sayÃ„Â±sÃ„Â± geÃƒÂ§ersiz: "
            f"{processed_rows:,} != {EXPECTED_RAW_ROW_COUNT:,}"
        )
    if set(unique_by_split) != set(SPLITS):
        raise RuntimeError(f"Ãƒâ€“lÃƒÂ§eklenmiÃ…Å¸ girdi split envanteri eksik: {unique_by_split}")
    if overlap_count != 0:
        raise RuntimeError(
            "Grouped kolda StandardScaler ve float32 dÃƒÂ¶nÃƒÂ¼Ã…Å¸ÃƒÂ¼mÃƒÂ¼ sonrasÃ„Â±nda splitler arasÃ„Â± "
            f"{overlap_count:,} model-girdisi ÃƒÂ§akÃ„Â±Ã…Å¸masÃ„Â± bulundu; eÃ„Å¸itim gÃƒÂ¼venli biÃƒÂ§imde durduruldu."
        )
    return {
        "feature_space": "StandardScaler(float64 parameters) then float32 model input",
        "full_raw_row_scan": True,
        "processed_raw_rows": processed_rows,
        "unique_fingerprint_count_by_split": unique_by_split,
        "cross_split_model_input_fingerprint_count": overlap_count,
        "audit_database": str(audit_database),
        "audit_database_sha256": sha256(audit_database),
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        "passed": True,
    }


def ensure_cache_disk_space(database: Path, cache: Path, arms: list[str]) -> None:
    missing_bytes = 0
    for arm in arms:
        manifest_file = cache / arm / "manifest.json"
        completed = False
        if manifest_file.exists():
            try:
                completed = read_json(manifest_file).get("status") == "completed"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                completed = False
        if not completed:
            counts = split_counts(database, arm)
            row_count = sum(counts.values())
            missing_bytes += row_count * (EXPECTED_FEATURE_COUNT * 4 + 8)
    if missing_bytes == 0:
        return
    cache.mkdir(parents=True, exist_ok=True)
    required = missing_bytes + MINIMUM_CACHE_SAFETY_BYTES
    free = shutil.disk_usage(cache).free
    if free < required:
        raise RuntimeError(
            "VS2 ÃƒÂ¶nbelleÃ„Å¸i iÃƒÂ§in boÃ…Å¸ alan yetersiz: "
            f"gereken yaklaÃ…Å¸Ã„Â±k={required / 1024**3:.2f} GiB, "
            f"bulunan={free / 1024**3:.2f} GiB"
        )


def validate_completed_cache(
    arm_directory: Path,
    manifest: dict[str, Any],
    arm: str,
    counts: dict[str, int],
    expected_class_counts: dict[str, list[int]],
    database_sha256: str,
    stage38_script_sha256: str,
) -> None:
    expected_manifest = {
        "status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "assignment_database_sha256": database_sha256,
        "stage38_script_sha256": stage38_script_sha256,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "feature_dtype": "float32",
        "scaler_parameter_dtype": "float64",
        "split_counts": counts,
        "split_class_counts": expected_class_counts,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Eski/uyumsuz VS2 ÃƒÂ¶nbelleÃ„Å¸i bulundu ({arm}): {mismatches}")
    for split in SPLITS:
        x = np.load(arm_directory / f"X_{split}.npy", mmap_mode="r")
        y = np.load(arm_directory / f"y_{split}.npy", mmap_mode="r")
        if x.shape != (counts[split], EXPECTED_FEATURE_COUNT) or x.dtype != np.float32:
            raise RuntimeError(f"GeÃƒÂ§ersiz ÃƒÂ¶nbellek X dizisi: {arm}/{split} {x.shape} {x.dtype}")
        if y.shape != (counts[split],) or y.dtype != np.int64:
            raise RuntimeError(f"GeÃƒÂ§ersiz ÃƒÂ¶nbellek y dizisi: {arm}/{split} {y.shape} {y.dtype}")
        observed_counts = np.bincount(y, minlength=3).astype(np.int64).tolist()
        if observed_counts != expected_class_counts[split]:
            raise RuntimeError(
                f"Ãƒâ€“nbellek sÃ„Â±nÃ„Â±f sayÃ„Â±larÃ„Â± uyuÃ…Å¸muyor: {arm}/{split} "
                f"{observed_counts} != {expected_class_counts[split]}"
            )
    mean = np.load(arm_directory / "mean.npy")
    scale = np.load(arm_directory / "std.npy")
    weights = np.load(arm_directory / "class_weights.npy")
    if mean.shape != (EXPECTED_FEATURE_COUNT,) or mean.dtype != np.float64:
        raise RuntimeError(f"GeÃƒÂ§ersiz scaler mean: {arm} {mean.shape} {mean.dtype}")
    if scale.shape != (EXPECTED_FEATURE_COUNT,) or scale.dtype != np.float64:
        raise RuntimeError(f"GeÃƒÂ§ersiz scaler scale: {arm} {scale.shape} {scale.dtype}")
    if weights.shape != (3,) or weights.dtype != np.float32:
        raise RuntimeError(f"GeÃƒÂ§ersiz sÃ„Â±nÃ„Â±f aÃ„Å¸Ã„Â±rlÃ„Â±Ã„Å¸Ã„Â±: {arm} {weights.shape} {weights.dtype}")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError(f"Sonlu/pozitif olmayan scaler parametresi: {arm}")
    split_audit = manifest.get("scaled_split_full_scan_audit")
    if not isinstance(split_audit, dict):
        raise RuntimeError(f"Ãƒâ€“nbellekte ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ split tam-tarama denetimi yok: {arm}")
    for split in SPLITS:
        item = split_audit.get(split)
        if (
            not isinstance(item, dict)
            or item.get("processed_rows") != counts[split]
            or item.get("all_model_input_values_finite") is not True
            or item.get("model_input_dtype") != "float32"
        ):
            raise RuntimeError(f"Ãƒâ€“lÃƒÂ§eklenmiÃ…Å¸ split denetimi geÃƒÂ§ersiz: {arm}/{split}: {item}")
    overlap_audit = manifest.get("scaled_model_input_overlap_audit")
    if arm == "grouped_float32":
        if not isinstance(overlap_audit, dict):
            raise RuntimeError("Grouped ÃƒÂ¶nbellekte ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ model-girdisi denetimi yok.")
        if (
            overlap_audit.get("passed") is not True
            or overlap_audit.get("cross_split_model_input_fingerprint_count") != 0
            or overlap_audit.get("processed_raw_rows") != EXPECTED_RAW_ROW_COUNT
        ):
            raise RuntimeError(f"Grouped ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ model-girdisi denetimi geÃƒÂ§ersiz: {overlap_audit}")
        audit_database = Path(str(overlap_audit.get("audit_database", "")))
        if not audit_database.exists():
            raise RuntimeError(f"Grouped ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ girdi denetim DB bulunamadÃ„Â±: {audit_database}")
        if sha256(audit_database) != overlap_audit.get("audit_database_sha256"):
            raise RuntimeError("Grouped ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ girdi denetim DB SHA-256 uyuÃ…Å¸muyor.")


def materialize_arm(
    database: Path,
    csv_root: Path,
    arm: str,
    cache: Path,
    chunk_size: int,
    database_sha256: str,
    stage38_script_sha256: str,
) -> dict[str, Any]:
    arm_directory = cache / arm
    manifest_file = arm_directory / "manifest.json"
    counts = split_counts(database, arm)
    expected_class_counts = split_class_counts(database, arm)
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        if manifest.get("status") == "completed":
            validate_completed_cache(
                arm_directory=arm_directory,
                manifest=manifest,
                arm=arm,
                counts=counts,
                expected_class_counts=expected_class_counts,
                database_sha256=database_sha256,
                stage38_script_sha256=stage38_script_sha256,
            )
            print(f"[{arm}] TamamlanmÃ„Â±Ã…Å¸ VS2 ÃƒÂ¶nbelleÃ„Å¸i kullanÃ„Â±lÃ„Â±yor.", flush=True)
            return manifest

    arm_directory.mkdir(parents=True, exist_ok=True)
    source_database = source_database_path(database)
    files = discover_files(source_database, csv_root)
    arrays: dict[str, tuple[np.memmap, np.memmap]] = {}
    offsets = {split: 0 for split in SPLITS}
    for split in SPLITS:
        x_array = np.lib.format.open_memmap(
            arm_directory / f"X_{split}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(counts[split], EXPECTED_FEATURE_COUNT),
        )
        y_array = np.lib.format.open_memmap(
            arm_directory / f"y_{split}.npy",
            mode="w+",
            dtype=np.int64,
            shape=(counts[split],),
        )
        arrays[split] = (x_array, y_array)

    for file_number, (file_id, csv_path, raw_label, expected_rows) in enumerate(files, start=1):
        row_sets = {split: selected_rows(database, file_id, arm, split) for split in SPLITS}
        base = 0
        target = family_label(raw_label)
        last_file_progress = time.perf_counter()
        for frame in pd.read_csv(csv_path, chunksize=chunk_size):
            values = frame.to_numpy(dtype=np.float32, copy=False)
            if values.shape[1] != EXPECTED_FEATURE_COUNT or not np.isfinite(values).all():
                raise RuntimeError(f"GeÃƒÂ§ersiz CSV ÃƒÂ¶zellikleri: {csv_path}")
            end = base + len(values)
            for split in SPLITS:
                chosen = row_sets[split]
                left = int(np.searchsorted(chosen, base, side="left"))
                right = int(np.searchsorted(chosen, end, side="left"))
                local = chosen[left:right] - base
                amount = len(local)
                if amount:
                    output_start = offsets[split]
                    arrays[split][0][output_start : output_start + amount] = values[local]
                    arrays[split][1][output_start : output_start + amount] = target
                    offsets[split] += amount
            base = end
            now = time.perf_counter()
            if now - last_file_progress >= 60.0:
                print(
                    f"[{arm}] Dosya {file_number}/{len(files)} iÃ…Å¸leniyor: {csv_path.name} "
                    f"({base:,}/{expected_rows:,} satÃ„Â±r)",
                    flush=True,
                )
                last_file_progress = now
        if base != expected_rows:
            raise RuntimeError(
                f"CSV satÃ„Â±r sayÃ„Â±sÃ„Â± uyuÃ…Å¸muyor ({csv_path.name}): {base:,} != {expected_rows:,}"
            )
        print(f"[{arm}] Dosya {file_number}/{len(files)} hazÃ„Â±rlandÃ„Â±: {csv_path.name}", flush=True)

    for split in SPLITS:
        if offsets[split] != counts[split]:
            raise RuntimeError(f"{arm}/{split}: {offsets[split]:,} != {counts[split]:,}")
        arrays[split][0].flush()
        arrays[split][1].flush()
        observed_counts = np.bincount(arrays[split][1], minlength=3).astype(np.int64).tolist()
        if observed_counts != expected_class_counts[split]:
            raise RuntimeError(
                f"{arm}/{split}: materyalize sÃ„Â±nÃ„Â±f sayÃ„Â±larÃ„Â± uyuÃ…Å¸muyor: "
                f"{observed_counts} != {expected_class_counts[split]}"
            )

    x_train = np.load(arm_directory / "X_train.npy", mmap_mode="r")
    y_train = np.load(arm_directory / "y_train.npy", mmap_mode="r")
    scaler = StandardScaler(with_mean=True, with_std=True)
    for start in range(0, len(x_train), chunk_size):
        scaler.partial_fit(np.asarray(x_train[start : start + chunk_size], dtype=np.float32))
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    standard_deviation = np.asarray(scaler.scale_, dtype=np.float64)
    if mean.shape != (EXPECTED_FEATURE_COUNT,) or standard_deviation.shape != (
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(f"{arm}: StandardScaler parametre boyutu geÃƒÂ§ersiz.")
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise RuntimeError(f"{arm}: StandardScaler sonlu olmayan parametre ÃƒÂ¼retti.")
    if np.any(standard_deviation <= 0):
        raise RuntimeError(f"{arm}: StandardScaler pozitif olmayan ÃƒÂ¶lÃƒÂ§ek ÃƒÂ¼retti.")
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = inverse_sqrt_mean1(class_counts)
    np.save(arm_directory / "mean.npy", mean)
    np.save(arm_directory / "std.npy", standard_deviation)
    np.save(arm_directory / "class_weights.npy", class_weights)

    scaled_sum = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float64)
    scaled_square = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float64)
    scaled_finite = True
    for start in range(0, len(x_train), chunk_size):
        raw = np.asarray(x_train[start : start + chunk_size], dtype=np.float64)
        scaled = (raw - mean) / standard_deviation
        scaled_finite = scaled_finite and bool(np.isfinite(scaled).all())
        scaled_sum += scaled.sum(axis=0)
        scaled_square += np.square(scaled).sum(axis=0)
    scaled_mean = scaled_sum / len(x_train)
    scaled_variance = np.maximum(
        scaled_square / len(x_train) - np.square(scaled_mean), 0.0
    )
    scaled_std = np.sqrt(scaled_variance)
    max_abs_scaled_mean = float(np.max(np.abs(scaled_mean)))
    max_scaled_std_error = float(np.max(np.abs(scaled_std - 1.0)))
    scaler_audit_passed = (
        scaled_finite
        and max_abs_scaled_mean <= SCALER_AUDIT_TOLERANCE
        and max_scaled_std_error <= SCALER_AUDIT_TOLERANCE
    )
    if not scaler_audit_passed:
        raise RuntimeError(
            f"{arm}: StandardScaler tam tarama denetimi baÃ…Å¸arÃ„Â±sÃ„Â±z; "
            f"mean={max_abs_scaled_mean:.3e}, std_error={max_scaled_std_error:.3e}"
        )
    scaled_split_audit: dict[str, Any] = {}
    for split in SPLITS:
        x_split = np.load(arm_directory / f"X_{split}.npy", mmap_mode="r")
        split_min = math.inf
        split_max = -math.inf
        split_finite = True
        processed = 0
        for start in range(0, len(x_split), chunk_size):
            raw = np.asarray(x_split[start : start + chunk_size], dtype=np.float64)
            scaled32 = ((raw - mean) / standard_deviation).astype(np.float32)
            split_finite = split_finite and bool(np.isfinite(scaled32).all())
            split_min = min(split_min, float(np.min(scaled32)))
            split_max = max(split_max, float(np.max(scaled32)))
            processed += len(scaled32)
        if not split_finite or processed != counts[split]:
            raise RuntimeError(
                f"{arm}/{split}: ÃƒÂ¶lÃƒÂ§eklenmiÃ…Å¸ model-girdisi sonluluk/satÃ„Â±r denetimi baÃ…Å¸arÃ„Â±sÃ„Â±z."
            )
        scaled_split_audit[split] = {
            "processed_rows": processed,
            "all_model_input_values_finite": split_finite,
            "minimum_model_input_value": split_min,
            "maximum_model_input_value": split_max,
            "model_input_dtype": "float32",
        }
    if arm == "grouped_float32":
        print(
            "[grouped_float32] StandardScaler sonrasÃ„Â± gerÃƒÂ§ek float32 model-girdisi "
            "ÃƒÂ¶rtÃƒÂ¼Ã…Å¸mesi denetleniyor...",
            flush=True,
        )
        scaled_overlap_audit = audit_scaled_grouped_overlap(arm_directory, chunk_size)
    else:
        scaled_overlap_audit = {
            "feature_space": "StandardScaler(float64 parameters) then float32 model input",
            "full_raw_row_scan": False,
            "reason": "Naive arm intentionally permits duplicate exposure; grouped arm is the zero-overlap lock.",
            "passed": None,
        }

    manifest = {
        "status": "completed",
        "created_at_utc": utc_now(),
        "protocol": "vs2_float32_leakage_ablation",
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "assignment_database_sha256": database_sha256,
        "stage38_script_sha256": stage38_script_sha256,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "feature_dtype": "float32",
        "scaler_method": "sklearn.preprocessing.StandardScaler.partial_fit",
        "scaler_parameter_dtype": "float64",
        "scaled_model_input_dtype": "float32",
        "class_names": list(CLASS_NAMES),
        "split_counts": counts,
        "split_class_counts": expected_class_counts,
        "total_raw_rows": sum(counts.values()),
        "train_class_counts": class_counts.tolist(),
        "scaler_fit_split": "train_only",
        "class_weights_fit_split": "train_only",
        "class_weight_scheme": "inverse_square_root_frequency_mean1",
        "class_weights": class_weights.tolist(),
        "scaler_full_scan_audit": {
            "all_scaled_values_finite": scaled_finite,
            "maximum_absolute_scaled_train_mean": max_abs_scaled_mean,
            "maximum_scaled_train_standard_deviation_error": max_scaled_std_error,
            "tolerance": SCALER_AUDIT_TOLERANCE,
            "passed": scaler_audit_passed,
        },
        "scaled_split_full_scan_audit": scaled_split_audit,
        "scaled_model_input_overlap_audit": scaled_overlap_audit,
    }
    atomic_json(manifest_file, manifest)
    return manifest


class MemmapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, directory: Path, split: str) -> None:
        self.x = np.load(directory / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(directory / f"y_{split}.npy", mmap_mode="r")
        self.mean = np.load(directory / "mean.npy")
        self.std = np.load(directory / "std.npy")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw = np.asarray(self.x[index], dtype=np.float64)
        scaled = ((raw - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(scaled), torch.tensor(int(self.y[index]), dtype=torch.long)


def set_reproducible(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def multiclass_mcc(confusion: np.ndarray) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    true_sum = matrix.sum(axis=1)
    predicted_sum = matrix.sum(axis=0)
    sample_count = matrix.sum()
    correct = np.trace(matrix)
    numerator = correct * sample_count - np.dot(true_sum, predicted_sum)
    denominator = math.sqrt(
        (sample_count**2 - np.dot(predicted_sum, predicted_sum))
        * (sample_count**2 - np.dot(true_sum, true_sum))
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "mcc": multiclass_mcc(confusion),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            CLASS_NAMES[index]: {
                "support": int(support[index]),
                "predicted_count": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "false_negative_rate": float(1.0 - recall[index]),
            }
            for index in range(3)
        },
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0
    for features, targets in loader:
        logits = model(features)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        amount = len(targets)
        loss_sum += float(loss) * amount
        sample_count += amount
        np.add.at(confusion, (targets.numpy(), predictions.numpy()), 1)
    return {
        "loss": loss_sum / sample_count,
        "sample_count": sample_count,
        **confusion_metrics(confusion),
    }


def protocol_settings(path: Path) -> tuple[dict[str, Any], str]:
    protocol = read_json(path)
    if tuple(int(seed) for seed in protocol.get("training_seeds", ())) != SEEDS:
        raise RuntimeError(f"B0 protokol seed kÃƒÂ¼mesi uyuÃ…Å¸muyor: {protocol.get('training_seeds')!r}")
    if str(protocol.get("task")) != "family_3":
        raise RuntimeError(f"B0 gÃƒÂ¶rev adÃ„Â± uyuÃ…Å¸muyor: {protocol.get('task')!r}")
    common = protocol["common_training_configuration"]
    model = protocol["model_specific_configuration"]["tinyml_mlp"]
    expected = {
        "optimizer": "AdamW",
        "class_weight_scheme": "inverse_square_root_frequency_mean1",
    }
    for key, value in expected.items():
        if str(common.get(key)) != value:
            raise RuntimeError(f"B0 protokolÃƒÂ¼ uyuÃ…Å¸muyor: {key}={common.get(key)!r}")
    settings = {
        "optimizer": "AdamW",
        "model_name": "tinyml_mlp",
        "batch_size": int(common["batch_size"]),
        "num_workers": int(common["num_workers"]),
        "max_epochs": int(common["max_epochs"]),
        "learning_rate": float(model["learning_rate"]),
        "weight_decay": float(common["weight_decay"]),
        "gradient_clip_norm": float(common["gradient_clip_norm"]),
        "patience": int(common["early_stopping_patience"]),
        "min_delta": float(common["early_stopping_min_delta"]),
        "torch_threads": int(common["torch_threads"]),
        "class_weight_scheme": str(common["class_weight_scheme"]),
    }
    locked = {
        "batch_size": 4096,
        "num_workers": 0,
        "max_epochs": 15,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "patience": 4,
        "min_delta": 0.0002,
        "torch_threads": 4,
    }
    for key, expected_value in locked.items():
        if settings[key] != expected_value:
            raise RuntimeError(
                f"Kilitli B0 ayarÃ„Â± deÃ„Å¸iÃ…Å¸miÃ…Å¸: {key}={settings[key]!r}, beklenen={expected_value!r}"
            )
    return settings, sha256(path)


def train_one(
    arm_directory: Path,
    run_directory: Path,
    arm: str,
    seed: int,
    cfg: dict[str, Any],
    protocol_sha256: str,
    database_sha256: str,
    cache_manifest_sha256: str,
    stage38_script_sha256: str,
    restart_incomplete: bool,
) -> dict[str, Any]:
    result_file = run_directory / "result.json"
    manifest_file = run_directory / "run_manifest.json"
    if result_file.exists():
        result = read_json(result_file)
        expected_result = {
            "status": "completed",
            "arm": arm,
            "seed": seed,
            "model_name": "tinyml_mlp",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "protocol_sha256": protocol_sha256,
            "assignment_database_sha256": database_sha256,
            "cache_manifest_sha256": cache_manifest_sha256,
            "stage38_script_sha256": stage38_script_sha256,
        }
        mismatches = {
            key: (result.get(key), value)
            for key, value in expected_result.items()
            if result.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"TamamlanmÃ„Â±Ã…Å¸ VS2 sonucu uyumsuz: {result_file}: {mismatches}")
        print(f"[{arm}/seed{seed}] TamamlanmÃ„Â±Ã…Å¸ VS2 sonucu kullanÃ„Â±lÃ„Â±yor.", flush=True)
        return result
    if run_directory.exists() and any(run_directory.iterdir()):
        if not restart_incomplete:
            raise RuntimeError(
                f"YarÃ„Â±m kalmÃ„Â±Ã…Å¸ run bulundu: {run_directory}. "
                "Ã„Â°nceledikten sonra yalnÃ„Â±z bu VS2 run iÃƒÂ§in --restart-incomplete kullanÃ„Â±n."
            )
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)

    set_reproducible(seed, cfg["torch_threads"])
    train_dataset = MemmapDataset(arm_directory, "train")
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=train_generator,
        num_workers=cfg["num_workers"],
    )
    validation_loader = DataLoader(
        MemmapDataset(arm_directory, "validation"),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        MemmapDataset(arm_directory, "test"),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    model = create_model(
        model_name="tinyml_mlp",
        input_features=EXPECTED_FEATURE_COUNT,
        num_classes=3,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"B0 tinyml_mlp parametre sayÃ„Â±sÃ„Â± deÃ„Å¸iÃ…Å¸miÃ…Å¸: {parameter_count:,} != {EXPECTED_PARAMETER_COUNT:,}"
        )
    weights = torch.from_numpy(np.load(arm_directory / "class_weights.npy"))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    run_config = {
        "status": "training_started",
        "created_at_utc": utc_now(),
        "arm": arm,
        "seed": seed,
        "model_name": "tinyml_mlp",
        "parameter_count": parameter_count,
        "protocol_sha256": protocol_sha256,
        "assignment_database_sha256": database_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "stage38_script_sha256": stage38_script_sha256,
        "training_configuration": cfg,
        "test_evaluation_attempt_count": 0,
        "test_used_for_selection": False,
    }
    atomic_json(manifest_file, run_config)

    best_macro_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_file = run_directory / "best_checkpoint.pt"
    training_started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            amount = len(targets)
            running_loss += float(loss.detach()) * amount
            sample_count += amount
        validation = evaluate(model, validation_loader, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / sample_count,
                "validation_loss": validation["loss"],
                "validation_macro_f1": validation["macro_f1"],
                "validation_accuracy": validation["accuracy"],
            }
        )
        print(
            f"[{arm}] seed={seed} epoch={epoch}: "
            f"val_macro_f1={validation['macro_f1']:.6f}",
            flush=True,
        )
        if validation["macro_f1"] > best_macro_f1 + cfg["min_delta"]:
            best_macro_f1 = float(validation["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "tinyml_mlp",
                    "task_name": "family_3",
                    "class_names": CLASS_NAMES,
                    "arm": arm,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "protocol_sha256": protocol_sha256,
                    "assignment_database_sha256": database_sha256,
                    "cache_manifest_sha256": cache_manifest_sha256,
                    "stage38_script_sha256": stage38_script_sha256,
                },
                checkpoint_file,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg["patience"]:
                break

    training_seconds = time.perf_counter() - training_started

    manifest = {
        **run_config,
        "status": "test_started",
        "training_completed_at_utc": utc_now(),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test_evaluation_attempt_count": 1,
    }
    atomic_json(manifest_file, manifest)
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test = evaluate(model, test_loader, criterion)
    result = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "arm": arm,
        "seed": seed,
        "model_name": "tinyml_mlp",
        "parameter_count": parameter_count,
        "protocol_sha256": protocol_sha256,
        "assignment_database_sha256": database_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "stage38_script_sha256": stage38_script_sha256,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_macro_f1": best_macro_f1,
        "training_seconds": training_seconds,
        "test": test,
        "history": history,
        "test_evaluation_attempt_count": 1,
        "test_used_for_selection": False,
    }
    atomic_json(result_file, result)
    atomic_json(
        manifest_file,
        {
            **manifest,
            "status": "completed",
            "completed_at_utc": result["completed_at_utc"],
            "result_file": str(result_file),
        },
    )
    return result


def summary_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
    critical = float(student_t.ppf(0.975, df=count - 1)) if count > 1 else 0.0
    margin = critical * standard_deviation / math.sqrt(count) if count > 1 else 0.0
    return {
        "sample_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def exact_wilcoxon_signed_rank(differences: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.abs(differences) > 0.0]
    if len(differences) == 0:
        return {"nonzero_pairs": 0, "statistic_w": 0.0, "two_sided_exact_p": 1.0}
    ranks = pd.Series(np.abs(differences)).rank(method="average").to_numpy(dtype=np.float64)
    observed_plus = float(ranks[differences > 0].sum())
    observed_minus = float(ranks[differences < 0].sum())
    observed_w = min(observed_plus, observed_minus)
    extreme = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(ranks)):
        signed = np.asarray(signs, dtype=np.int8)
        plus = float(ranks[signed > 0].sum())
        minus = float(ranks[signed < 0].sum())
        if min(plus, minus) <= observed_w + 1e-12:
            extreme += 1
        total += 1
    return {
        "nonzero_pairs": int(len(differences)),
        "statistic_w": observed_w,
        "rank_sum_positive": observed_plus,
        "rank_sum_negative": observed_minus,
        "two_sided_exact_p": extreme / total,
        "enumerated_sign_assignments": total,
    }


def write_reports(
    frame: pd.DataFrame,
    output_directory: Path,
    manifests: dict[str, Any],
    cfg: dict[str, Any],
    protocol_sha256: str,
    database_sha256: str,
    database_validation: dict[str, Any],
    stage38_script_sha256: str,
    elapsed_minutes: float,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    run_file = output_directory / "leakage_ablation_b0_matched_runs.csv"
    frame.to_csv(run_file, index=False)
    metrics = (
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_f1",
        "test_mcc",
    ) + tuple(
        f"test_{class_name}_{metric}"
        for class_name in CLASS_NAMES
        for metric in ("precision", "recall", "f1", "false_negative_rate")
    )
    aggregate_records: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_frame = frame[frame["arm"] == arm]
        for metric in metrics:
            aggregate_records.append(
                {"arm": arm, "metric": metric, **summary_statistics(arm_frame[metric].to_numpy())}
            )
    aggregate = pd.DataFrame(aggregate_records)
    aggregate_file = output_directory / "leakage_ablation_b0_matched_aggregate.csv"
    aggregate.to_csv(aggregate_file, index=False)

    paired_records: list[dict[str, Any]] = []
    exact_tests: dict[str, Any] = {}
    for metric in metrics:
        pivot = frame.pivot(index="seed", columns="arm", values=metric).loc[list(SEEDS)]
        differences = (
            pivot["naive_row"].to_numpy(dtype=np.float64)
            - pivot["grouped_float32"].to_numpy(dtype=np.float64)
        )
        for seed, naive, grouped, difference in zip(
            pivot.index,
            pivot["naive_row"],
            pivot["grouped_float32"],
            differences,
        ):
            paired_records.append(
                {
                    "metric": metric,
                    "seed": int(seed),
                    "naive_row": float(naive),
                    "grouped_float32": float(grouped),
                    "naive_minus_grouped": float(difference),
                }
            )
        exact_tests[metric] = {
            "difference_direction": "naive_row_minus_grouped_float32",
            "difference_summary": summary_statistics(differences),
            "same_direction_pair_count": int(np.sum(differences > 0)),
            "exact_wilcoxon_signed_rank": exact_wilcoxon_signed_rank(differences),
        }
    paired = pd.DataFrame(paired_records)
    paired_file = output_directory / "leakage_ablation_b0_matched_paired_deltas.csv"
    paired.to_csv(paired_file, index=False)
    exact_file = output_directory / "leakage_ablation_b0_matched_exact_inference.json"
    atomic_json(
        exact_file,
        {
            "generated_at_utc": utc_now(),
            "paired_unit": "locked training seed",
            "fixed_split_seed": LOCKED_SPLIT_SEED,
            "inference_scope": (
                "Conditional on the single locked split seed; split-to-split uncertainty is not estimated."
            ),
            "pair_count": len(SEEDS),
            "interpretation_rule": (
                "With five nonzero pairs the smallest attainable two-sided exact p-value is 0.0625; "
                "results must therefore be interpreted with effect sizes and consistency, not as p<0.05 proof."
            ),
            "tests": exact_tests,
        },
    )
    summary_file = output_directory / "leakage_ablation_b0_matched_summary.json"
    atomic_json(
        summary_file,
        {
            "generated_at_utc": utc_now(),
            "status": "completed",
            "protocol": "vs2_float32_leakage_ablation_b0_matched",
            "protocol_version": PROTOCOL_VERSION,
            "analysis_role": "auxiliary raw-frequency leakage-sensitivity analysis",
            "does_not_replace_primary_experiments": True,
            "model": "tinyml_mlp",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "training_seeds": list(SEEDS),
            "fixed_split_seed": LOCKED_SPLIT_SEED,
            "inference_scope": (
                "Training-seed variability conditional on split seed 2026; no split-seed generalization claim."
            ),
            "arms": list(ARMS),
            "total_training_run_count": len(ARMS) * len(SEEDS),
            "same_raw_row_universe_between_arms": True,
            "raw_row_count_per_arm": EXPECTED_RAW_ROW_COUNT,
            "primary_experimental_factor": (
                "split assignment unit: raw row vs identical float32 vector group"
            ),
            "split_ratio_control": (
                "70/15/15 family-level raw-row ratios are audited with fail-closed tolerances"
            ),
            "training_configuration": cfg,
            "locked_b0_protocol_sha256": protocol_sha256,
            "assignment_database_sha256": database_sha256,
            "assignment_database_validation": database_validation,
            "stage38_script_sha256": stage38_script_sha256,
            "data_manifests": manifests,
            "test_evaluations_per_arm_seed": 1,
            "test_used_for_selection": False,
            "elapsed_minutes": elapsed_minutes,
            "artifacts": {
                "runs_csv": str(run_file),
                "aggregate_csv": str(aggregate_file),
                "paired_deltas_csv": str(paired_file),
                "exact_inference_json": str(exact_file),
            },
        },
    )


def main() -> None:
    args = parse_arguments()
    started = time.perf_counter()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size pozitif olmalÃ„Â±dÃ„Â±r.")
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"Seed kÃƒÂ¼mesi kilitli olmalÃ„Â±dÃ„Â±r: {SEEDS}")
    if tuple(args.arms) != ARMS:
        print("UYARI: YalnÃ„Â±z seÃƒÂ§ili kollar ÃƒÂ§alÃ„Â±Ã…Å¸tÃ„Â±rÃ„Â±lÃ„Â±yor; nihai eÃ…Å¸leÃ…Å¸tirilmiÃ…Å¸ rapor oluÃ…Å¸mayabilir.")
    database = args.database.resolve()
    protocol = args.protocol.resolve()
    csv_root = args.csv_root.resolve()
    cache = args.cache_directory.resolve()
    output = args.output_directory.resolve()
    if not database.exists():
        raise FileNotFoundError(database)
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    database_validation = validate_database(database)
    print("Stage-37 bilimsel kilitleri doÃ„Å¸rulandÃ„Â±; kaynak CSV ÃƒÂ¶zetleri sÃ„Â±nanÃ„Â±yor...", flush=True)
    database_validation["source_integrity"] = validate_source_files(database, csv_root)
    print("Kaynak CSV bÃƒÂ¼tÃƒÂ¼nlÃƒÂ¼Ã„Å¸ÃƒÂ¼ doÃ„Å¸rulandÃ„Â±; veritabanÃ„Â± ÃƒÂ¶zeti alÃ„Â±nÃ„Â±yor...", flush=True)
    database_sha256 = sha256(database)
    stage38_script_sha256 = sha256(Path(__file__).resolve())
    cfg, protocol_sha256 = protocol_settings(protocol)
    ensure_cache_disk_space(database, cache, list(args.arms))

    manifests: dict[str, Any] = {}
    for arm in args.arms:
        manifests[arm] = materialize_arm(
            database=database,
            csv_root=csv_root,
            arm=arm,
            cache=cache,
            chunk_size=args.chunk_size,
            database_sha256=database_sha256,
            stage38_script_sha256=stage38_script_sha256,
        )
    if args.prepare_only:
        print("VS2 ÃƒÂ¶nbellek hazÃ„Â±rlama tamamlandÃ„Â±; --prepare-only nedeniyle eÃ„Å¸itim baÃ…Å¸latÃ„Â±lmadÃ„Â±.")
        return

    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for seed in args.seeds:
            run_directory = output / arm / f"seed{seed}"
            print(f"[{arm}/seed{seed}] B0-eÃ…Å¸lenmiÃ…Å¸ eÃ„Å¸itim baÃ…Å¸lÃ„Â±yor...", flush=True)
            result = train_one(
                arm_directory=cache / arm,
                run_directory=run_directory,
                arm=arm,
                seed=seed,
                cfg=cfg,
                protocol_sha256=protocol_sha256,
                database_sha256=database_sha256,
                cache_manifest_sha256=sha256(cache / arm / "manifest.json"),
                stage38_script_sha256=stage38_script_sha256,
                restart_incomplete=args.restart_incomplete,
            )
            row = {
                "arm": arm,
                "seed": seed,
                "parameter_count": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                "epochs_completed": result["epochs_completed"],
                "best_validation_macro_f1": result["best_validation_macro_f1"],
                "training_seconds": result["training_seconds"],
                "test_evaluation_attempt_count": result["test_evaluation_attempt_count"],
                "test_used_for_selection": result["test_used_for_selection"],
            }
            for key, value in result["test"].items():
                if isinstance(value, (int, float)):
                    row[f"test_{key}"] = value
            for class_name, class_metrics in result["test"]["per_class"].items():
                for metric, value in class_metrics.items():
                    if metric not in ("support", "predicted_count"):
                        row[f"test_{class_name}_{metric}"] = float(value)
            rows.append(row)
            gc.collect()

    frame = pd.DataFrame(rows)
    if set(args.arms) == set(ARMS) and set(args.seeds) == set(SEEDS):
        write_reports(
            frame=frame,
            output_directory=output,
            manifests=manifests,
            cfg=cfg,
            protocol_sha256=protocol_sha256,
            database_sha256=database_sha256,
            database_validation=database_validation,
            stage38_script_sha256=stage38_script_sha256,
            elapsed_minutes=(time.perf_counter() - started) / 60.0,
        )
        macro = frame.pivot(index="seed", columns="arm", values="test_macro_f1").loc[list(SEEDS)]
        differences = macro["naive_row"] - macro["grouped_float32"]
        print("=" * 78)
        print("38. aÃ…Å¸ama tamamlandÃ„Â±: B0-eÃ…Å¸lenmiÃ…Å¸ float32 sÃ„Â±zÃ„Â±ntÃ„Â± ablasyonu")
        print(f"Naif Macro-F1 ortalamasÃ„Â± : {macro['naive_row'].mean():.9f}")
        print(f"Gruplu Macro-F1 ortalamasÃ„Â±: {macro['grouped_float32'].mean():.9f}")
        print(f"Naif - gruplu farkÃ„Â±       : {differences.mean():.9f}")
        print("Eski 150 ana koÃ…Å¸u ve VS1 sonuÃƒÂ§larÃ„Â± deÃ„Å¸iÃ…Å¸tirilmedi.")
        print("=" * 78)


if __name__ == "__main__":
    main()