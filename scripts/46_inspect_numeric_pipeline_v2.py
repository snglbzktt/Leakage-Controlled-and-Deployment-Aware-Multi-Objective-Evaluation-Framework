from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "numeric_pipeline_inventory_v2.json"
)

CODE_HITS_CSV = (
    OUTPUT_DIR
    / "numeric_pipeline_code_hits_v2.csv"
)

ARTIFACTS_CSV = (
    OUTPUT_DIR
    / "numeric_pipeline_artifacts_v2.csv"
)

PARQUET_CSV = (
    OUTPUT_DIR
    / "numeric_pipeline_parquet_inventory_v2.csv"
)


KEYWORDS = [
    "float32",
    "float64",
    "StandardScaler",
    "standardizer",
    "scaler",
    "fit_transform",
    ".fit(",
    ".transform(",
    "astype(np.float32",
    "astype(np.float64",
    "to_numpy(dtype=np.float32",
    "to_numpy(dtype=np.float64",
    "nbaiot_float32_canonical",
    "nbaiot_primary_seed2026",
]


SEARCH_ROOTS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
]


ARTIFACT_ROOTS = [
    PROJECT_ROOT / "models" / "preprocessing",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "splits",
    PROJECT_ROOT / "results" / "reports",
]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except ValueError:
        return str(path)


print("=" * 78)
print("FAZ 1D — SAYISAL VERİ HATTI ENVANTERİ")
print("=" * 78)
print(f"Proje kökü: {PROJECT_ROOT}")

# ------------------------------------------------------------------
# 1. Kod içinde float32, scaler ve standardizasyon kullanımını tara
# ------------------------------------------------------------------
code_hits: list[dict[str, Any]] = []

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue

    for file_path in sorted(
        search_root.rglob("*.py")
    ):
        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            matched_keywords = [
                keyword
                for keyword in KEYWORDS
                if keyword.casefold()
                in line.casefold()
            ]

            if matched_keywords:
                code_hits.append(
                    {
                        "relative_path":
                            relative_path(
                                file_path
                            ),

                        "line_number":
                            line_number,

                        "keywords":
                            "|".join(
                                matched_keywords
                            ),

                        "code":
                            line.strip(),
                    }
                )

write_csv(
    CODE_HITS_CSV,
    code_hits,
    [
        "relative_path",
        "line_number",
        "keywords",
        "code",
    ],
)

# ------------------------------------------------------------------
# 2. Scaler, standardizer ve işlenmiş veri yapıtlarını bul
# ------------------------------------------------------------------
artifact_rows: list[dict[str, Any]] = []

candidate_name_pattern = re.compile(
    r"(scaler|standard|preprocess|normaliz|float32|float64|split|class_weight)",
    flags=re.IGNORECASE,
)

for root in ARTIFACT_ROOTS:
    if not root.exists():
        continue

    for file_path in sorted(
        root.rglob("*")
    ):
        if not file_path.is_file():
            continue

        if not candidate_name_pattern.search(
            file_path.name
        ):
            continue

        artifact_rows.append(
            {
                "relative_path":
                    relative_path(file_path),

                "suffix":
                    file_path.suffix.lower(),

                "size_bytes":
                    file_path.stat().st_size,

                "size_mb":
                    round(
                        file_path.stat().st_size
                        / (1024 ** 2),
                        6,
                    ),
            }
        )

write_csv(
    ARTIFACTS_CSV,
    artifact_rows,
    [
        "relative_path",
        "suffix",
        "size_bytes",
        "size_mb",
    ],
)

# ------------------------------------------------------------------
# 3. Parquet dosyalarını ve satır sayılarını incele
# ------------------------------------------------------------------
parquet_rows: list[dict[str, Any]] = []
pyarrow_error = ""

try:
    import pyarrow.parquet as pq

    processed_root = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    if processed_root.exists():
        for parquet_path in sorted(
            processed_root.rglob("*.parquet")
        ):
            try:
                parquet_file = pq.ParquetFile(
                    parquet_path
                )

                schema_names = (
                    parquet_file.schema.names
                )

                parquet_rows.append(
                    {
                        "relative_path":
                            relative_path(
                                parquet_path
                            ),

                        "row_count":
                            parquet_file.metadata.num_rows,

                        "row_group_count":
                            parquet_file.metadata.num_row_groups,

                        "column_count":
                            len(schema_names),

                        "first_columns":
                            "|".join(
                                schema_names[:10]
                            ),

                        "last_columns":
                            "|".join(
                                schema_names[-10:]
                            ),

                        "size_bytes":
                            parquet_path.stat().st_size,
                    }
                )

            except Exception as error:
                parquet_rows.append(
                    {
                        "relative_path":
                            relative_path(
                                parquet_path
                            ),

                        "row_count":
                            "",

                        "row_group_count":
                            "",

                        "column_count":
                            "",

                        "first_columns":
                            "",

                        "last_columns":
                            "",

                        "size_bytes":
                            parquet_path.stat().st_size,

                        "error":
                            str(error),
                    }
                )

except Exception as error:
    pyarrow_error = str(error)

write_csv(
    PARQUET_CSV,
    parquet_rows,
    [
        "relative_path",
        "row_count",
        "row_group_count",
        "column_count",
        "first_columns",
        "last_columns",
        "size_bytes",
        "error",
    ],
)

# ------------------------------------------------------------------
# 4. Özellikle önemli dosyaları bul
# ------------------------------------------------------------------
important_candidates: list[str] = []

patterns = [
    "nbaiot_pipeline.py",
    "*standardizer*",
    "*scaler*",
    "*preprocess*",
    "*class_weight*",
    "*task_definition*",
]

for pattern in patterns:
    for path in PROJECT_ROOT.rglob(
        pattern
    ):
        if path.is_file():
            important_candidates.append(
                relative_path(path)
            )

important_candidates = sorted(
    set(important_candidates)
)

code_files_with_float32 = sorted(
    {
        row["relative_path"]
        for row in code_hits
        if "float32" in row["keywords"].casefold()
    }
)

code_files_with_scaler = sorted(
    {
        row["relative_path"]
        for row in code_hits
        if (
            "scaler" in row["keywords"].casefold()
            or
            "standard" in row["keywords"].casefold()
        )
    }
)

summary = {
    "status":
        "completed",

    "project_root":
        str(PROJECT_ROOT),

    "code_hit_count":
        len(code_hits),

    "artifact_count":
        len(artifact_rows),

    "parquet_count":
        len(parquet_rows),

    "pyarrow_error":
        pyarrow_error,

    "important_candidates":
        important_candidates,

    "code_files_with_float32":
        code_files_with_float32,

    "code_files_with_scaler":
        code_files_with_scaler,

    "code_hits_csv":
        str(CODE_HITS_CSV),

    "artifacts_csv":
        str(ARTIFACTS_CSV),

    "parquet_inventory_csv":
        str(PARQUET_CSV),
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("ÖNEMLİ DOSYA ADAYLARI")
print("-" * 78)

for path in important_candidates:
    print(path)

print()
print("FLOAT32 KULLANAN DOSYALAR")
print("-" * 78)

for path in code_files_with_float32:
    print(path)

print()
print("SCALER/STANDARDIZER KULLANAN DOSYALAR")
print("-" * 78)

for path in code_files_with_scaler:
    print(path)

print()
print("PARQUET ENVANTERİ")
print("-" * 78)

for row in parquet_rows:
    print(
        f"{row['relative_path']} | "
        f"rows={row.get('row_count', '')} | "
        f"columns={row.get('column_count', '')}"
    )

print()
print("=" * 78)
print("FAZ 1D ENVANTERİ TAMAMLANDI")
print("=" * 78)
print(f"Kod taraması : {CODE_HITS_CSV}")
print(f"Yapıt listesi: {ARTIFACTS_CSV}")
print(f"Parquet listesi: {PARQUET_CSV}")
print(f"Özet JSON    : {SUMMARY_JSON}")
