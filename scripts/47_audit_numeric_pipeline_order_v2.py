from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


PROTOCOL_VERSION = "numeric_pipeline_order_audit_v2_1"

EXPECTED_FEATURE_COUNT = 115

EXPECTED_SPLIT_ROWS = {
    "train": 1_738_133,
    "validation": 371_884,
    "test": 372_659,
}

BATCH_SIZE = 20_000

# Önceden belirlenen karar eşikleri
UNIQUE_COUNT_RELATIVE_THRESHOLD = 0.0001
ROW_DIFFERENCE_RATE_THRESHOLD = 0.01
MAX_ABSOLUTE_DIFFERENCE_THRESHOLD = 1e-4
MEAN_ABSOLUTE_DIFFERENCE_THRESHOLD = 1e-7


PROJECT_ROOT = Path.cwd()

SOURCE_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
)

CURRENT_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_float32_canonical_seed2026"
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

SPLIT_METRICS_CSV = (
    OUTPUT_DIRECTORY
    / "numeric_pipeline_order_split_metrics_v2.csv"
)

OVERLAP_CSV = (
    OUTPUT_DIRECTORY
    / "numeric_pipeline_order_overlap_v2.csv"
)

FEATURE_STATS_CSV = (
    OUTPUT_DIRECTORY
    / "numeric_pipeline_order_feature_stats_v2.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIRECTORY
    / "numeric_pipeline_order_summary_v2.json"
)


PAIR_DTYPE = np.dtype(
    [
        ("forward", "<u8"),
        ("reverse", "<u8"),
    ]
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def get_feature_columns(
    parquet_path: Path,
) -> list[str]:
    parquet_file = pq.ParquetFile(
        parquet_path
    )

    schema = parquet_file.schema_arrow

    feature_columns = [
        field.name
        for field in schema
        if pa.types.is_floating(
            field.type
        )
    ]

    if (
        len(feature_columns)
        != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"{parquet_path.name}: "
            f"beklenen {EXPECTED_FEATURE_COUNT} "
            f"float özellik, bulunan "
            f"{len(feature_columns)}"
        )

    return feature_columns


def iter_feature_batches(
    parquet_path: Path,
    feature_columns: list[str],
):
    parquet_file = pq.ParquetFile(
        parquet_path
    )

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=feature_columns,
    ):
        frame = batch.to_pandas()

        matrix = frame.to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if (
            matrix.ndim != 2
            or matrix.shape[1]
            != EXPECTED_FEATURE_COUNT
        ):
            raise RuntimeError(
                "Beklenmeyen batch boyutu: "
                f"{matrix.shape}"
            )

        if not np.isfinite(
            matrix
        ).all():
            raise RuntimeError(
                f"{parquet_path}: "
                "NaN veya sonsuz değer bulundu."
            )

        yield np.asarray(
            matrix,
            dtype=np.float64,
            order="C",
        )


def transform_pipeline_a(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Mevcut sıranın ortak evren karşılığı:

    float64 kaynak
    → float32
    → float64 hesaplama
    → StandardScaler
    → float32 model girdisi
    """
    float32_source = values.astype(
        np.float32
    )

    promoted = float32_source.astype(
        np.float64
    )

    return (
        (promoted - mean) / scale
    ).astype(
        np.float32
    )


def transform_pipeline_b(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Alternatif sıra:

    float64 kaynak
    → StandardScaler
    → float32 model girdisi
    """
    return (
        (values - mean) / scale
    ).astype(
        np.float32
    )


def fingerprint_pair(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        values,
        copy=False,
    )

    forward = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64,
        copy=False,
    )

    reverse = pd.util.hash_pandas_object(
        frame.iloc[:, ::-1],
        index=False,
        categorize=False,
    ).to_numpy(
        dtype=np.uint64,
        copy=False,
    )

    return forward, reverse


def overlap_count(
    left: np.ndarray,
    right: np.ndarray,
) -> int:
    return int(
        np.intersect1d(
            left,
            right,
            assume_unique=True,
        ).size
    )


for split_name, expected_rows in (
    EXPECTED_SPLIT_ROWS.items()
):
    split_file = (
        SOURCE_DATA_DIRECTORY
        / f"{split_name}.parquet"
    )

    if not split_file.exists():
        print(
            f"HATA: Kaynak split bulunamadı: "
            f"{split_file}"
        )
        sys.exit(1)

    observed_rows = (
        pq.ParquetFile(
            split_file
        ).metadata.num_rows
    )

    if observed_rows != expected_rows:
        print(
            f"HATA: {split_name} satır sayısı "
            f"beklenen değerle uyuşmuyor. "
            f"Beklenen={expected_rows}, "
            f"bulunan={observed_rows}"
        )
        sys.exit(1)


feature_columns = get_feature_columns(
    SOURCE_DATA_DIRECTORY
    / "train.parquet"
)

print("=" * 78)
print("FAZ 1D — SAYISAL İŞLEM SIRASI DENETİMİ")
print("=" * 78)
print(f"Kaynak evren : {SOURCE_DATA_DIRECTORY}")
print(f"Mevcut evren : {CURRENT_DATA_DIRECTORY}")
print(f"Özellik sayısı: {len(feature_columns)}")
print(f"Batch boyutu : {BATCH_SIZE:,}")
print()
print("Hat A: float64 -> float32 -> scaler -> float32")
print("Hat B: float64 -> scaler -> float32")

started_at = time.perf_counter()

# ------------------------------------------------------------------
# 1. Aynı primary train üzerinde iki scaler öğren
# ------------------------------------------------------------------
print()
print("İki scaler yalnız primary train üzerinde öğreniliyor...")

scaler_a = StandardScaler(
    with_mean=True,
    with_std=True,
)

scaler_b = StandardScaler(
    with_mean=True,
    with_std=True,
)

fit_row_count = 0

for batch_index, values in enumerate(
    iter_feature_batches(
        SOURCE_DATA_DIRECTORY
        / "train.parquet",
        feature_columns,
    ),
    start=1,
):
    pipeline_a_input = values.astype(
        np.float32
    ).astype(
        np.float64
    )

    scaler_a.partial_fit(
        pipeline_a_input
    )

    scaler_b.partial_fit(
        values
    )

    fit_row_count += int(
        values.shape[0]
    )

    if (
        batch_index == 1
        or batch_index % 10 == 0
    ):
        print(
            f"  scaler fit: "
            f"{fit_row_count:,}/"
            f"{EXPECTED_SPLIT_ROWS['train']:,}"
        )


if (
    fit_row_count
    != EXPECTED_SPLIT_ROWS["train"]
):
    raise RuntimeError(
        "Scaler fit satır sayısı uyuşmuyor."
    )


mean_a = np.asarray(
    scaler_a.mean_,
    dtype=np.float64,
)

scale_a = np.asarray(
    scaler_a.scale_,
    dtype=np.float64,
)

mean_b = np.asarray(
    scaler_b.mean_,
    dtype=np.float64,
)

scale_b = np.asarray(
    scaler_b.scale_,
    dtype=np.float64,
)


if (
    not np.isfinite(mean_a).all()
    or not np.isfinite(scale_a).all()
    or not np.isfinite(mean_b).all()
    or not np.isfinite(scale_b).all()
):
    raise RuntimeError(
        "Scaler parametrelerinde "
        "sonlu olmayan değer bulundu."
    )


if (
    np.any(scale_a <= 0)
    or np.any(scale_b <= 0)
):
    raise RuntimeError(
        "Scaler içinde pozitif olmayan "
        "scale değeri bulundu."
    )


feature_stat_rows: list[
    dict[str, Any]
] = []

for feature_index, feature_name in enumerate(
    feature_columns
):
    feature_stat_rows.append(
        {
            "feature_index":
                feature_index,

            "feature_name":
                feature_name,

            "pipeline_a_mean":
                float(mean_a[feature_index]),

            "pipeline_b_mean":
                float(mean_b[feature_index]),

            "absolute_mean_difference":
                float(
                    abs(
                        mean_a[feature_index]
                        - mean_b[feature_index]
                    )
                ),

            "pipeline_a_scale":
                float(scale_a[feature_index]),

            "pipeline_b_scale":
                float(scale_b[feature_index]),

            "absolute_scale_difference":
                float(
                    abs(
                        scale_a[feature_index]
                        - scale_b[feature_index]
                    )
                ),
        }
    )


write_csv(
    FEATURE_STATS_CSV,
    feature_stat_rows,
    [
        "feature_index",
        "feature_name",
        "pipeline_a_mean",
        "pipeline_b_mean",
        "absolute_mean_difference",
        "pipeline_a_scale",
        "pipeline_b_scale",
        "absolute_scale_difference",
    ],
)


# ------------------------------------------------------------------
# 2. Üç split üzerinde sayısal ve yapısal karşılaştırma
# ------------------------------------------------------------------
split_metric_rows: list[
    dict[str, Any]
] = []

unique_pairs_a: dict[
    str,
    np.ndarray,
] = {}

unique_pairs_b: dict[
    str,
    np.ndarray,
] = {}

global_row_count = 0
global_different_row_count = 0
global_element_count = 0
global_different_element_count = 0
global_absolute_difference_sum = 0.0
global_max_absolute_difference = 0.0


for split_name, expected_rows in (
    EXPECTED_SPLIT_ROWS.items()
):
    print()
    print("-" * 78)
    print(
        f"{split_name.upper()} "
        f"karşılaştırılıyor "
        f"({expected_rows:,} satır)"
    )

    pair_array_a = np.empty(
        expected_rows,
        dtype=PAIR_DTYPE,
    )

    pair_array_b = np.empty(
        expected_rows,
        dtype=PAIR_DTYPE,
    )

    row_offset = 0
    split_different_rows = 0
    split_different_elements = 0
    split_absolute_difference_sum = 0.0
    split_max_absolute_difference = 0.0
    split_element_count = 0

    for batch_index, values in enumerate(
        iter_feature_batches(
            SOURCE_DATA_DIRECTORY
            / f"{split_name}.parquet",
            feature_columns,
        ),
        start=1,
    ):
        transformed_a = (
            transform_pipeline_a(
                values,
                mean_a,
                scale_a,
            )
        )

        transformed_b = (
            transform_pipeline_b(
                values,
                mean_b,
                scale_b,
            )
        )

        batch_rows = int(
            values.shape[0]
        )

        end_offset = (
            row_offset + batch_rows
        )

        difference_mask = (
            transformed_a
            != transformed_b
        )

        different_rows = int(
            np.any(
                difference_mask,
                axis=1,
            ).sum()
        )

        different_elements = int(
            difference_mask.sum()
        )

        absolute_difference = np.abs(
            transformed_a.astype(
                np.float64
            )
            - transformed_b.astype(
                np.float64
            )
        )

        split_different_rows += (
            different_rows
        )

        split_different_elements += (
            different_elements
        )

        split_absolute_difference_sum += (
            float(
                absolute_difference.sum(
                    dtype=np.float64
                )
            )
        )

        if absolute_difference.size > 0:
            split_max_absolute_difference = max(
                split_max_absolute_difference,
                float(
                    absolute_difference.max()
                ),
            )

        split_element_count += int(
            absolute_difference.size
        )

        forward_a, reverse_a = (
            fingerprint_pair(
                transformed_a
            )
        )

        forward_b, reverse_b = (
            fingerprint_pair(
                transformed_b
            )
        )

        pair_array_a[
            "forward"
        ][
            row_offset:end_offset
        ] = forward_a

        pair_array_a[
            "reverse"
        ][
            row_offset:end_offset
        ] = reverse_a

        pair_array_b[
            "forward"
        ][
            row_offset:end_offset
        ] = forward_b

        pair_array_b[
            "reverse"
        ][
            row_offset:end_offset
        ] = reverse_b

        row_offset = end_offset

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or row_offset == expected_rows
        ):
            print(
                f"  {row_offset:,}/"
                f"{expected_rows:,}"
            )

    if row_offset != expected_rows:
        raise RuntimeError(
            f"{split_name}: işlenen satır "
            "sayısı uyuşmuyor."
        )

    unique_a = np.unique(
        pair_array_a
    )

    unique_b = np.unique(
        pair_array_b
    )

    unique_pairs_a[
        split_name
    ] = unique_a

    unique_pairs_b[
        split_name
    ] = unique_b

    del pair_array_a
    del pair_array_b

    mean_absolute_difference = (
        split_absolute_difference_sum
        / split_element_count
        if split_element_count > 0
        else 0.0
    )

    split_metric_rows.append(
        {
            "split":
                split_name,

            "row_count":
                expected_rows,

            "element_count":
                split_element_count,

            "different_row_count":
                split_different_rows,

            "different_row_rate":
                (
                    split_different_rows
                    / expected_rows
                ),

            "different_element_count":
                split_different_elements,

            "different_element_rate":
                (
                    split_different_elements
                    / split_element_count
                ),

            "mean_absolute_difference":
                mean_absolute_difference,

            "maximum_absolute_difference":
                split_max_absolute_difference,

            "pipeline_a_unique_input_count":
                int(unique_a.size),

            "pipeline_b_unique_input_count":
                int(unique_b.size),

            "unique_count_difference_b_minus_a":
                int(
                    unique_b.size
                    - unique_a.size
                ),

            "absolute_unique_count_difference":
                int(
                    abs(
                        unique_b.size
                        - unique_a.size
                    )
                ),

            "relative_unique_count_difference":
                (
                    abs(
                        unique_b.size
                        - unique_a.size
                    )
                    / expected_rows
                ),

            "pipeline_a_duplicate_row_count":
                int(
                    expected_rows
                    - unique_a.size
                ),

            "pipeline_b_duplicate_row_count":
                int(
                    expected_rows
                    - unique_b.size
                ),
        }
    )

    global_row_count += expected_rows

    global_different_row_count += (
        split_different_rows
    )

    global_element_count += (
        split_element_count
    )

    global_different_element_count += (
        split_different_elements
    )

    global_absolute_difference_sum += (
        split_absolute_difference_sum
    )

    global_max_absolute_difference = max(
        global_max_absolute_difference,
        split_max_absolute_difference,
    )


write_csv(
    SPLIT_METRICS_CSV,
    split_metric_rows,
    [
        "split",
        "row_count",
        "element_count",
        "different_row_count",
        "different_row_rate",
        "different_element_count",
        "different_element_rate",
        "mean_absolute_difference",
        "maximum_absolute_difference",
        "pipeline_a_unique_input_count",
        "pipeline_b_unique_input_count",
        "unique_count_difference_b_minus_a",
        "absolute_unique_count_difference",
        "relative_unique_count_difference",
        "pipeline_a_duplicate_row_count",
        "pipeline_b_duplicate_row_count",
    ],
)


# ------------------------------------------------------------------
# 3. Splitler arası final model-girdisi örtüşmesi
# ------------------------------------------------------------------
split_pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

overlap_rows: list[
    dict[str, Any]
] = []

for left_split, right_split in split_pairs:
    overlap_a = overlap_count(
        unique_pairs_a[left_split],
        unique_pairs_a[right_split],
    )

    overlap_b = overlap_count(
        unique_pairs_b[left_split],
        unique_pairs_b[right_split],
    )

    overlap_rows.append(
        {
            "split_a":
                left_split,

            "split_b":
                right_split,

            "pipeline_a_overlap_count":
                overlap_a,

            "pipeline_b_overlap_count":
                overlap_b,

            "overlap_difference_b_minus_a":
                overlap_b - overlap_a,
        }
    )


write_csv(
    OVERLAP_CSV,
    overlap_rows,
    [
        "split_a",
        "split_b",
        "pipeline_a_overlap_count",
        "pipeline_b_overlap_count",
        "overlap_difference_b_minus_a",
    ],
)


# ------------------------------------------------------------------
# 4. Önceden belirlenmiş karar kuralları
# ------------------------------------------------------------------
global_different_row_rate = (
    global_different_row_count
    / global_row_count
)

global_different_element_rate = (
    global_different_element_count
    / global_element_count
)

global_mean_absolute_difference = (
    global_absolute_difference_sum
    / global_element_count
)


decision_reasons: list[str] = []

for row in split_metric_rows:
    if (
        row[
            "relative_unique_count_difference"
        ]
        >= UNIQUE_COUNT_RELATIVE_THRESHOLD
    ):
        decision_reasons.append(
            f"{row['split']}: final float32 "
            "benzersiz girdi sayısı farkı "
            "önceden belirlenen eşiği geçti."
        )


for row in overlap_rows:
    if (
        row[
            "pipeline_a_overlap_count"
        ]
        != row[
            "pipeline_b_overlap_count"
        ]
    ):
        decision_reasons.append(
            f"{row['split_a']}-"
            f"{row['split_b']}: "
            "splitler arası final float32 "
            "örtüşme sayısı iki hat arasında "
            "farklıdır."
        )


numeric_trigger = (
    global_different_row_rate
    >= ROW_DIFFERENCE_RATE_THRESHOLD
    and
    (
        global_max_absolute_difference
        >= MAX_ABSOLUTE_DIFFERENCE_THRESHOLD
        or
        global_mean_absolute_difference
        >= MEAN_ABSOLUTE_DIFFERENCE_THRESHOLD
    )
)

if numeric_trigger:
    decision_reasons.append(
        "Satır fark oranı ve sayısal fark "
        "önceden belirlenen ortak eşiği geçti."
    )


requires_training_pilot = bool(
    decision_reasons
)


current_dataset_rows = {}

for split_name in (
    "train",
    "validation",
    "test",
):
    current_file = (
        CURRENT_DATA_DIRECTORY
        / f"{split_name}.parquet"
    )

    current_dataset_rows[
        split_name
    ] = (
        int(
            pq.ParquetFile(
                current_file
            ).metadata.num_rows
        )
        if current_file.exists()
        else None
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

    "source_universe":
        str(
            SOURCE_DATA_DIRECTORY
        ),

    "diagnostic_design":
        (
            "Both processing orders were "
            "evaluated on the same "
            "source-precision primary splits."
        ),

    "important_limitation":
        (
            "This is a processing-order "
            "diagnostic. Pipeline A here "
            "uses the primary source-precision "
            "split, while the historical main "
            "experiments used the separate "
            "float32-canonical split."
        ),

    "pipeline_a":
        (
            "source float64 -> float32 -> "
            "train-only StandardScaler with "
            "float64 parameters -> float32"
        ),

    "pipeline_b":
        (
            "source float64 -> train-only "
            "StandardScaler in float64 -> "
            "float32"
        ),

    "fit_row_count":
        fit_row_count,

    "expected_split_rows":
        EXPECTED_SPLIT_ROWS,

    "historical_float32_canonical_rows":
        current_dataset_rows,

    "maximum_scaler_mean_difference":
        float(
            np.max(
                np.abs(
                    mean_a - mean_b
                )
            )
        ),

    "maximum_scaler_scale_difference":
        float(
            np.max(
                np.abs(
                    scale_a - scale_b
                )
            )
        ),

    "global_row_count":
        global_row_count,

    "global_different_row_count":
        global_different_row_count,

    "global_different_row_rate":
        global_different_row_rate,

    "global_element_count":
        global_element_count,

    "global_different_element_count":
        global_different_element_count,

    "global_different_element_rate":
        global_different_element_rate,

    "global_mean_absolute_difference":
        global_mean_absolute_difference,

    "global_maximum_absolute_difference":
        global_max_absolute_difference,

    "thresholds": {
        "unique_count_relative_threshold":
            UNIQUE_COUNT_RELATIVE_THRESHOLD,

        "row_difference_rate_threshold":
            ROW_DIFFERENCE_RATE_THRESHOLD,

        "maximum_absolute_difference_threshold":
            MAX_ABSOLUTE_DIFFERENCE_THRESHOLD,

        "mean_absolute_difference_threshold":
            MEAN_ABSOLUTE_DIFFERENCE_THRESHOLD,
    },

    "requires_training_pilot":
        requires_training_pilot,

    "decision_reasons":
        decision_reasons,

    "split_metrics_csv":
        str(SPLIT_METRICS_CSV),

    "overlap_csv":
        str(OVERLAP_CSV),

    "feature_stats_csv":
        str(FEATURE_STATS_CSV),
}


validation_checks = {
    "train_fit_count":
        fit_row_count
        == EXPECTED_SPLIT_ROWS["train"],

    "global_row_count":
        global_row_count
        == sum(
            EXPECTED_SPLIT_ROWS.values()
        ),

    "feature_count":
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT,

    "scaler_a_finite":
        bool(
            np.isfinite(mean_a).all()
            and np.isfinite(scale_a).all()
        ),

    "scaler_b_finite":
        bool(
            np.isfinite(mean_b).all()
            and np.isfinite(scale_b).all()
        ),

    "three_split_metric_rows":
        len(split_metric_rows) == 3,

    "three_overlap_rows":
        len(overlap_rows) == 3,
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


print()
print("=" * 78)
print("FAZ 1D ÖZETİ")
print("=" * 78)

print(
    f"Scaler max mean farkı : "
    f"{summary['maximum_scaler_mean_difference']:.12g}"
)

print(
    f"Scaler max scale farkı: "
    f"{summary['maximum_scaler_scale_difference']:.12g}"
)

print(
    f"Farklı satır oranı    : "
    f"{100 * global_different_row_rate:.6f}%"
)

print(
    f"Farklı eleman oranı   : "
    f"{100 * global_different_element_rate:.6f}%"
)

print(
    f"Ortalama mutlak fark  : "
    f"{global_mean_absolute_difference:.12g}"
)

print(
    f"Maksimum mutlak fark  : "
    f"{global_max_absolute_difference:.12g}"
)

print()
print("SPLIT METRİKLERİ")

for row in split_metric_rows:
    print(
        f"{row['split']} | "
        f"A_unique="
        f"{row['pipeline_a_unique_input_count']:,} | "
        f"B_unique="
        f"{row['pipeline_b_unique_input_count']:,} | "
        f"delta="
        f"{row['unique_count_difference_b_minus_a']:,} | "
        f"different_rows="
        f"{100 * row['different_row_rate']:.6f}%"
    )

print()
print("SPLIT ÖRTÜŞMELERİ")

for row in overlap_rows:
    print(
        f"{row['split_a']}-"
        f"{row['split_b']} | "
        f"A={row['pipeline_a_overlap_count']:,} | "
        f"B={row['pipeline_b_overlap_count']:,}"
    )

print()
print(
    "8 koşuluk eğitim pilotu gerekli mi: "
    f"{requires_training_pilot}"
)

if decision_reasons:
    print("Karar nedenleri:")

    for reason in decision_reasons:
        print(f"- {reason}")
else:
    print(
        "Önceden belirlenen eşikler "
        "aşılmadı."
    )

print()
print("VALIDATION CHECKS")

for name, passed in validation_checks.items():
    print(f"{name}: {passed}")

print()
print(f"Split metrikleri: {SPLIT_METRICS_CSV}")
print(f"Örtüşme raporu : {OVERLAP_CSV}")
print(f"Özellik raporu : {FEATURE_STATS_CSV}")
print(f"Özet JSON      : {SUMMARY_JSON}")

if not summary["all_checks_passed"]:
    print()
    print("FAZ 1D BAŞARISIZ")
    sys.exit(1)

print()
print("FAZ 1D BAŞARILI")
