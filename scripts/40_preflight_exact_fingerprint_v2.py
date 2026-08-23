from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FILE_COUNT = 89
EXPECTED_TOTAL_ROWS = 7_062_606
EXPECTED_FEATURE_COUNT = 115

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data"

DATABASE_PATH = (
    DATA_ROOT
    / "cache"
    / "nbaiot_duplicate_audit.sqlite"
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


def canonical_float64_bytes(values: np.ndarray) -> bytes:
    """
    Kaynak hassasiyetindeki bir vektörü kanonik byte dizisine çevirir.

    Kurallar:
    - float64
    - little-endian
    - C-contiguous
    - -0.0 ve +0.0 aynı gösterim
    - NaN/inf kabul edilmez
    """
    array = np.asarray(
        values,
        dtype=np.dtype("<f8"),
    ).reshape(-1)

    if array.size != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Beklenen {EXPECTED_FEATURE_COUNT} özellik, "
            f"bulunan {array.size}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Vektörde NaN veya sonsuz değer bulundu."
        )

    array = np.ascontiguousarray(array)

    # -0.0 değerlerini +0.0 biçimine getir.
    zero_mask = array == 0.0

    if zero_mask.any():
        array = array.copy()
        array[zero_mask] = 0.0

    return array.tobytes(order="C")


def sha256_digest(values: np.ndarray) -> str:
    canonical_bytes = canonical_float64_bytes(values)

    return hashlib.sha256(
        canonical_bytes
    ).hexdigest()


def normalize_relative_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()


def build_csv_index() -> list[Path]:
    return sorted(
        path.resolve()
        for path in DATA_ROOT.rglob("*.csv")
        if path.is_file()
    )


def resolve_source_file(
    stored_path: str,
    csv_files: list[Path],
) -> tuple[Path | None, str]:
    """
    SQLite içindeki relative_path değerini gerçek dosyaya bağlar.
    """
    raw_path = Path(stored_path)

    direct_candidates = []

    if raw_path.is_absolute():
        direct_candidates.append(raw_path)
    else:
        direct_candidates.extend(
            [
                PROJECT_ROOT / raw_path,
                DATA_ROOT / raw_path,
                DATA_ROOT / "raw" / raw_path,
                DATA_ROOT / "interim" / raw_path,
            ]
        )

    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(), "direct"

    stored_normalized = normalize_relative_path(
        stored_path
    )

    suffix_matches = [
        path
        for path in csv_files
        if normalize_relative_path(
            str(path.relative_to(PROJECT_ROOT))
        ).endswith(stored_normalized)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0], "suffix"

    basename_matches = [
        path
        for path in csv_files
        if path.name.lower() == raw_path.name.lower()
    ]

    if len(basename_matches) == 1:
        return basename_matches[0], "basename"

    return None, "unresolved"


def determine_row_numbering(
    minimum: int,
    maximum: int,
    expected_rows: int,
) -> str:
    if (
        minimum == 0
        and maximum == expected_rows - 1
    ):
        return "zero_based"

    if (
        minimum == 1
        and maximum == expected_rows
    ):
        return "one_based"

    return "unexpected"


if not DATABASE_PATH.exists():
    print(
        f"HATA: Denetim veritabanı bulunamadı: "
        f"{DATABASE_PATH}"
    )
    sys.exit(1)

csv_files = build_csv_index()

print("=" * 78)
print("FAZ 1A — EXACT FINGERPRINT ÖN HAZIRLIK")
print("=" * 78)
print(f"Proje kökü       : {PROJECT_ROOT}")
print(f"Veritabanı       : {DATABASE_PATH}")
print(f"Data CSV sayısı  : {len(csv_files)}")

disk_usage = shutil.disk_usage(PROJECT_ROOT)

free_gb = disk_usage.free / (1024 ** 3)
total_gb = disk_usage.total / (1024 ** 3)

print(f"Disk kapasitesi  : {total_gb:.2f} GB")
print(f"Boş disk alanı   : {free_gb:.2f} GB")

if free_gb < 8.0:
    print(
        "UYARI: Tam exact fingerprint denetimi için "
        "en az 8 GB boş alan öneriliyor."
    )

connection = sqlite3.connect(
    str(DATABASE_PATH)
)

connection.row_factory = sqlite3.Row

metadata_rows = connection.execute(
    """
    SELECT key, value
    FROM metadata
    ORDER BY key
    """
).fetchall()

metadata = {
    str(row["key"]): str(row["value"])
    for row in metadata_rows
}

file_rows = connection.execute(
    """
    SELECT
        file_id,
        relative_path,
        device,
        class_label,
        expected_row_count,
        inserted_row_count,
        status,
        error
    FROM files
    ORDER BY file_id
    """
).fetchall()

fingerprint_ranges = connection.execute(
    """
    SELECT
        file_id,
        MIN(row_number) AS minimum_row_number,
        MAX(row_number) AS maximum_row_number,
        COUNT(*) AS fingerprint_row_count
    FROM fingerprints
    GROUP BY file_id
    ORDER BY file_id
    """
).fetchall()

connection.close()

range_by_file = {
    int(row["file_id"]): {
        "minimum": int(row["minimum_row_number"]),
        "maximum": int(row["maximum_row_number"]),
        "count": int(row["fingerprint_row_count"]),
    }
    for row in fingerprint_ranges
}

results: list[dict[str, object]] = []

unresolved_count = 0
schema_error_count = 0
row_numbering_error_count = 0
database_count_error_count = 0

print()
print("Kaynak dosyalar kontrol ediliyor...")

for index, row in enumerate(file_rows, start=1):
    file_id = int(row["file_id"])
    stored_path = str(row["relative_path"])
    expected_rows = int(row["expected_row_count"])
    inserted_rows = int(row["inserted_row_count"])

    resolved_path, resolution_method = (
        resolve_source_file(
            stored_path,
            csv_files,
        )
    )

    range_info = range_by_file.get(file_id)

    result: dict[str, object] = {
        "file_id": file_id,
        "stored_relative_path": stored_path,
        "resolved_path": (
            str(resolved_path)
            if resolved_path is not None
            else ""
        ),
        "resolution_method": resolution_method,
        "device": str(row["device"]),
        "class_label": str(row["class_label"]),
        "expected_row_count": expected_rows,
        "inserted_row_count": inserted_rows,
        "database_status": str(row["status"]),
        "database_error": str(row["error"] or ""),
        "minimum_row_number": "",
        "maximum_row_number": "",
        "fingerprint_row_count": "",
        "row_numbering": "",
        "sample_column_count": "",
        "sample_all_finite": "",
        "sample_sha256": "",
        "file_size_bytes": "",
        "status": "pending",
        "error": "",
    }

    if range_info is None:
        result["status"] = "error"
        result["error"] = (
            "fingerprints tablosunda file_id bulunamadı"
        )
        database_count_error_count += 1
        results.append(result)
        continue

    minimum_row = range_info["minimum"]
    maximum_row = range_info["maximum"]
    fingerprint_count = range_info["count"]

    row_numbering = determine_row_numbering(
        minimum_row,
        maximum_row,
        expected_rows,
    )

    result["minimum_row_number"] = minimum_row
    result["maximum_row_number"] = maximum_row
    result["fingerprint_row_count"] = fingerprint_count
    result["row_numbering"] = row_numbering

    if (
        fingerprint_count != expected_rows
        or inserted_rows != expected_rows
    ):
        database_count_error_count += 1
        result["status"] = "error"
        result["error"] = (
            "SQLite satır sayıları beklenen sayıyla uyuşmuyor"
        )

    if row_numbering == "unexpected":
        row_numbering_error_count += 1
        result["status"] = "error"
        result["error"] = (
            str(result["error"])
            + "; beklenmeyen row_number aralığı"
        ).strip("; ")

    if resolved_path is None:
        unresolved_count += 1
        result["status"] = "error"
        result["error"] = (
            str(result["error"])
            + "; kaynak CSV çözümlenemedi"
        ).strip("; ")

        results.append(result)
        continue

    result["file_size_bytes"] = (
        resolved_path.stat().st_size
    )

    try:
        sample = pd.read_csv(
            resolved_path,
            header=None,
            nrows=3,
            dtype=np.float64,
        )

        column_count = int(sample.shape[1])
        sample_array = sample.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        all_finite = bool(
            np.isfinite(sample_array).all()
        )

        result["sample_column_count"] = column_count
        result["sample_all_finite"] = all_finite

        if column_count != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Beklenen {EXPECTED_FEATURE_COUNT} sütun, "
                f"bulunan {column_count}"
            )

        if not all_finite:
            raise ValueError(
                "Örnek satırlarda NaN veya sonsuz değer bulundu"
            )

        result["sample_sha256"] = sha256_digest(
            sample_array[0]
        )

        if result["status"] == "pending":
            result["status"] = "ok"

    except Exception as error:
        schema_error_count += 1
        result["status"] = "error"
        result["error"] = (
            str(result["error"])
            + f"; CSV örnekleme hatası: {error}"
        ).strip("; ")

    results.append(result)

    if (
        index == 1
        or index % 10 == 0
        or index == len(file_rows)
    ):
        print(
            f"  {index:>2}/{len(file_rows)} dosya kontrol edildi"
        )

result_frame = pd.DataFrame(results)

csv_output = (
    OUTPUT_DIRECTORY
    / "exact_fingerprint_preflight_files_v2.csv"
)

result_frame.to_csv(
    csv_output,
    index=False,
    encoding="utf-8-sig",
)

total_expected_rows = sum(
    int(row["expected_row_count"])
    for row in file_rows
)

total_inserted_rows = sum(
    int(row["inserted_row_count"])
    for row in file_rows
)

ok_count = int(
    (result_frame["status"] == "ok").sum()
)

error_count = int(
    (result_frame["status"] != "ok").sum()
)

summary = {
    "protocol": "exact_fingerprint_preflight_v2",
    "project_root": str(PROJECT_ROOT),
    "database_path": str(DATABASE_PATH),
    "database_size_bytes": DATABASE_PATH.stat().st_size,
    "metadata": metadata,
    "expected_file_count": EXPECTED_FILE_COUNT,
    "observed_file_count": len(file_rows),
    "expected_total_rows": EXPECTED_TOTAL_ROWS,
    "observed_expected_rows": total_expected_rows,
    "observed_inserted_rows": total_inserted_rows,
    "data_csv_file_count": len(csv_files),
    "resolved_file_count": (
        len(file_rows) - unresolved_count
    ),
    "unresolved_file_count": unresolved_count,
    "schema_error_count": schema_error_count,
    "row_numbering_error_count": (
        row_numbering_error_count
    ),
    "database_count_error_count": (
        database_count_error_count
    ),
    "ok_file_count": ok_count,
    "error_file_count": error_count,
    "free_disk_gb": round(free_gb, 3),
    "output_csv": str(csv_output),
}

json_output = (
    OUTPUT_DIRECTORY
    / "exact_fingerprint_preflight_summary_v2.json"
)

json_output.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 78)
print("FAZ 1A ÖZETİ")
print("=" * 78)

for key, value in summary.items():
    if key != "metadata":
        print(f"{key}: {value}")

fatal_conditions = [
    len(file_rows) != EXPECTED_FILE_COUNT,
    total_expected_rows != EXPECTED_TOTAL_ROWS,
    total_inserted_rows != EXPECTED_TOTAL_ROWS,
    unresolved_count != 0,
    schema_error_count != 0,
    row_numbering_error_count != 0,
    database_count_error_count != 0,
    error_count != 0,
]

if any(fatal_conditions):
    print()
    print("FAZ 1A BAŞARISIZ")
    print(
        "Ayrıntılar için şu dosyayı inceleyin:"
    )
    print(csv_output)
    sys.exit(1)

print()
print("FAZ 1A BAŞARILI")
print(f"Dosya raporu : {csv_output}")
print(f"Özet raporu  : {json_output}")
