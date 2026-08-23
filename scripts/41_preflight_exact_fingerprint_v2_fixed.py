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

RAW_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nbaiot_unpacked"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
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


def normalize_path(value: str) -> str:
    return (
        value
        .replace("\\", "/")
        .strip()
        .lstrip("./")
        .casefold()
    )


def canonical_float64_bytes(
    values: np.ndarray,
) -> bytes:
    array = np.asarray(
        values,
        dtype=np.dtype("<f8"),
    ).reshape(-1)

    if array.size != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Beklenen özellik sayısı "
            f"{EXPECTED_FEATURE_COUNT}, "
            f"bulunan {array.size}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "NaN veya sonsuz değer bulundu."
        )

    array = np.ascontiguousarray(array)

    zero_mask = array == 0.0

    if zero_mask.any():
        array = array.copy()
        array[zero_mask] = 0.0

    return array.tobytes(order="C")


def sample_sha256(
    values: np.ndarray,
) -> str:
    return hashlib.sha256(
        canonical_float64_bytes(values)
    ).hexdigest()


if not RAW_DATA_ROOT.exists():
    print(
        "HATA: Ham veri klasörü bulunamadı:"
    )
    print(RAW_DATA_ROOT)
    sys.exit(1)

if not DATABASE_PATH.exists():
    print(
        "HATA: Denetim veritabanı bulunamadı:"
    )
    print(DATABASE_PATH)
    sys.exit(1)

raw_csv_files = sorted(
    path.resolve()
    for path in RAW_DATA_ROOT.rglob("*.csv")
    if path.is_file()
)

relative_file_map = {
    normalize_path(
        path.relative_to(
            RAW_DATA_ROOT
        ).as_posix()
    ): path
    for path in raw_csv_files
}

print("=" * 78)
print("FAZ 1A — DÜZELTİLMİŞ EXACT FINGERPRINT ÖN KONTROL")
print("=" * 78)
print(f"Proje kökü       : {PROJECT_ROOT}")
print(f"Ham veri kökü    : {RAW_DATA_ROOT}")
print(f"Ham CSV sayısı   : {len(raw_csv_files)}")
print(f"Veritabanı       : {DATABASE_PATH}")

disk_usage = shutil.disk_usage(PROJECT_ROOT)
free_disk_gb = disk_usage.free / (1024 ** 3)

print(f"Boş disk alanı   : {free_disk_gb:.3f} GB")

connection = sqlite3.connect(
    str(DATABASE_PATH)
)

connection.row_factory = sqlite3.Row

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

range_rows = connection.execute(
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
        "minimum": int(
            row["minimum_row_number"]
        ),
        "maximum": int(
            row["maximum_row_number"]
        ),
        "count": int(
            row["fingerprint_row_count"]
        ),
    }
    for row in range_rows
}

results: list[dict[str, object]] = []

for index, row in enumerate(
    file_rows,
    start=1,
):
    file_id = int(row["file_id"])
    stored_path = str(row["relative_path"])

    expected_rows = int(
        row["expected_row_count"]
    )

    inserted_rows = int(
        row["inserted_row_count"]
    )

    normalized_stored_path = normalize_path(
        stored_path
    )

    resolved_path = relative_file_map.get(
        normalized_stored_path
    )

    resolution_method = "exact_relative_path"

    candidate_paths: list[Path] = []

    if resolved_path is None:
        candidate_paths = [
            path
            for relative_path, path
            in relative_file_map.items()
            if (
                relative_path.endswith(
                    normalized_stored_path
                )
                or normalized_stored_path.endswith(
                    relative_path
                )
            )
        ]

        if len(candidate_paths) == 1:
            resolved_path = candidate_paths[0]
            resolution_method = "unique_suffix"
        elif len(candidate_paths) > 1:
            resolution_method = "ambiguous"
        else:
            resolution_method = "unresolved"

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
        "candidate_count": len(candidate_paths),
        "candidate_paths": " | ".join(
            str(path)
            for path in candidate_paths
        ),
        "device": str(row["device"]),
        "class_label": str(
            row["class_label"]
        ),
        "expected_row_count": expected_rows,
        "inserted_row_count": inserted_rows,
        "minimum_row_number": "",
        "maximum_row_number": "",
        "fingerprint_row_count": "",
        "row_numbering": "",
        "sample_column_count": "",
        "sample_all_numeric": "",
        "sample_all_finite": "",
        "sample_sha256": "",
        "status": "pending",
        "error": "",
    }

    errors: list[str] = []

    if range_info is None:
        errors.append(
            "fingerprints tablosunda "
            "file_id bulunamadı"
        )
    else:
        minimum_row = range_info["minimum"]
        maximum_row = range_info["maximum"]
        fingerprint_count = range_info["count"]

        result["minimum_row_number"] = (
            minimum_row
        )
        result["maximum_row_number"] = (
            maximum_row
        )
        result["fingerprint_row_count"] = (
            fingerprint_count
        )

        if (
            minimum_row == 0
            and maximum_row == expected_rows - 1
        ):
            row_numbering = "zero_based"
        elif (
            minimum_row == 1
            and maximum_row == expected_rows
        ):
            row_numbering = "one_based"
        else:
            row_numbering = "unexpected"
            errors.append(
                "beklenmeyen row_number aralığı"
            )

        result["row_numbering"] = (
            row_numbering
        )

        if fingerprint_count != expected_rows:
            errors.append(
                "fingerprint satır sayısı "
                "expected_row_count ile uyuşmuyor"
            )

    if inserted_rows != expected_rows:
        errors.append(
            "inserted_row_count "
            "expected_row_count ile uyuşmuyor"
        )

    if resolved_path is None:
        errors.append(
            "kaynak CSV çözümlenemedi"
        )
    else:
        try:
            # İlk satır sütun başlıklarıdır.
            sample_frame = pd.read_csv(
                resolved_path,
                header=0,
                nrows=3,
                low_memory=False,
            )

            column_count = int(
                sample_frame.shape[1]
            )

            result["sample_column_count"] = (
                column_count
            )

            if (
                column_count
                != EXPECTED_FEATURE_COUNT
            ):
                raise ValueError(
                    f"Beklenen "
                    f"{EXPECTED_FEATURE_COUNT} sütun, "
                    f"bulunan {column_count}"
                )

            numeric_frame = sample_frame.apply(
                pd.to_numeric,
                errors="raise",
            )

            sample_array = (
                numeric_frame.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

            result["sample_all_numeric"] = True

            all_finite = bool(
                np.isfinite(
                    sample_array
                ).all()
            )

            result["sample_all_finite"] = (
                all_finite
            )

            if not all_finite:
                raise ValueError(
                    "Örnek veride NaN veya "
                    "sonsuz değer bulundu"
                )

            result["sample_sha256"] = (
                sample_sha256(
                    sample_array[0]
                )
            )

        except Exception as error:
            errors.append(
                f"CSV örnekleme hatası: {error}"
            )

    if errors:
        result["status"] = "error"
        result["error"] = "; ".join(errors)
    else:
        result["status"] = "ok"

    results.append(result)

    if (
        index == 1
        or index % 10 == 0
        or index == len(file_rows)
    ):
        print(
            f"{index:>2}/{len(file_rows)} "
            "dosya kontrol edildi"
        )

result_frame = pd.DataFrame(results)

csv_output = (
    OUTPUT_DIRECTORY
    / "exact_fingerprint_preflight_files_v2_fixed.csv"
)

result_frame.to_csv(
    csv_output,
    index=False,
    encoding="utf-8-sig",
)

status_counts = {
    str(key): int(value)
    for key, value in (
        result_frame["status"]
        .value_counts()
        .to_dict()
        .items()
    )
}

resolution_counts = {
    str(key): int(value)
    for key, value in (
        result_frame["resolution_method"]
        .value_counts()
        .to_dict()
        .items()
    )
}

total_expected_rows = sum(
    int(row["expected_row_count"])
    for row in file_rows
)

total_inserted_rows = sum(
    int(row["inserted_row_count"])
    for row in file_rows
)

summary = {
    "protocol": (
        "exact_fingerprint_preflight_v2_fixed"
    ),
    "project_root": str(PROJECT_ROOT),
    "raw_data_root": str(RAW_DATA_ROOT),
    "database_path": str(DATABASE_PATH),
    "raw_csv_file_count": len(
        raw_csv_files
    ),
    "database_file_count": len(
        file_rows
    ),
    "expected_total_rows": (
        EXPECTED_TOTAL_ROWS
    ),
    "observed_expected_rows": (
        total_expected_rows
    ),
    "observed_inserted_rows": (
        total_inserted_rows
    ),
    "status_counts": status_counts,
    "resolution_counts": resolution_counts,
    "free_disk_gb": round(
        free_disk_gb,
        3,
    ),
    "output_csv": str(csv_output),
}

json_output = (
    OUTPUT_DIRECTORY
    / "exact_fingerprint_preflight_summary_v2_fixed.json"
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
print("FAZ 1A DÜZELTİLMİŞ ÖZET")
print("=" * 78)

for key, value in summary.items():
    print(f"{key}: {value}")

error_rows = result_frame[
    result_frame["status"] != "ok"
]

if not error_rows.empty:
    print()
    print("HATALI DOSYALAR")
    print("-" * 78)

    for _, error_row in error_rows.iterrows():
        print(
            f"file_id={error_row['file_id']} | "
            f"path={error_row['stored_relative_path']} | "
            f"resolution="
            f"{error_row['resolution_method']} | "
            f"error={error_row['error']}"
        )

fatal_conditions = [
    len(raw_csv_files) != EXPECTED_FILE_COUNT,
    len(file_rows) != EXPECTED_FILE_COUNT,
    total_expected_rows
        != EXPECTED_TOTAL_ROWS,
    total_inserted_rows
        != EXPECTED_TOTAL_ROWS,
    not error_rows.empty,
]

print()

if any(fatal_conditions):
    print("FAZ 1A DÜZELTİLMİŞ KONTROL BAŞARISIZ")
    print(f"Rapor: {csv_output}")
    sys.exit(1)

print("FAZ 1A DÜZELTİLMİŞ KONTROL BAŞARILI")
print(f"Dosya raporu : {csv_output}")
print(f"Özet raporu  : {json_output}")

if free_disk_gb < 12.0:
    print()
    print(
        "UYARI: Ön kontrol başarılı olsa bile "
        "tam denetim başlatılmayacak."
    )
    print(
        "Gerekli güvenli boş alan: en az 12 GB"
    )
