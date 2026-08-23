from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


PROTOCOL_VERSION = "numeric_order_pilot_cache_v2_1"

FEATURE_COUNT = 115
BATCH_SIZE = 20_000

EXPECTED_ROWS = {
    "train": 1_738_133,
    "validation": 371_884,
    "test": 372_659,
}

CLASS_NAMES = [
    "benign",
    "gafgyt",
    "mirai",
]


PROJECT_ROOT = Path.cwd()

SOURCE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "numeric_order_pilot_v2"
)

SHARED_DIRECTORY = (
    CACHE_ROOT
    / "shared"
)

PIPELINE_A_DIRECTORY = (
    CACHE_ROOT
    / "pipeline_a"
)

PIPELINE_B_DIRECTORY = (
    CACHE_ROOT
    / "pipeline_b"
)

STATE_DIRECTORY = (
    CACHE_ROOT
    / "state"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
)

PHASE_1D_SUMMARY = (
    AUDIT_DIRECTORY
    / "numeric_pipeline_order_summary_v2.json"
)

PILOT_PROTOCOL = (
    PROJECT_ROOT
    / "configs"
    / "protocols"
    / "nbaiot_numeric_order_training_pilot_v2.json"
)

CACHE_SUMMARY = (
    AUDIT_DIRECTORY
    / "numeric_order_pilot_cache_summary_v2.json"
)

CACHE_MANIFEST = (
    AUDIT_DIRECTORY
    / "numeric_order_pilot_cache_manifest_v2.json"
)

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "52_build_numeric_order_pilot_cache_v2.py"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_remove(
    path: Path,
) -> None:
    if path.exists():
        path.unlink()


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def save_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def discover_schema() -> tuple[
    list[str],
    str,
]:
    train_file = (
        SOURCE_DIRECTORY
        / "train.parquet"
    )

    schema = pq.ParquetFile(
        train_file
    ).schema_arrow

    feature_columns = [
        field.name
        for field in schema
        if pa.types.is_floating(
            field.type
        )
    ]

    if len(feature_columns) != FEATURE_COUNT:
        raise RuntimeError(
            "Beklenen özellik sayısı "
            f"{FEATURE_COUNT}, bulunan "
            f"{len(feature_columns)}."
        )

    non_feature_columns = [
        field.name
        for field in schema
        if field.name not in feature_columns
    ]

    preferred_names = {
        "class_label",
        "label",
        "target",
        "attack_label",
        "attack_type",
    }

    preferred_matches = [
        name
        for name in non_feature_columns
        if name.casefold()
        in preferred_names
    ]

    if len(preferred_matches) == 1:
        label_column = (
            preferred_matches[0]
        )

    elif len(non_feature_columns) == 1:
        label_column = (
            non_feature_columns[0]
        )

    else:
        raise RuntimeError(
            "Etiket sütunu otomatik "
            "belirlenemedi. Özellik dışı "
            f"sütunlar: {non_feature_columns}"
        )

    expected_names = [
        field.name
        for field in schema
    ]

    for split_name in EXPECTED_ROWS:
        split_file = (
            SOURCE_DIRECTORY
            / f"{split_name}.parquet"
        )

        split_parquet = pq.ParquetFile(
            split_file
        )

        observed_rows = int(
            split_parquet.metadata.num_rows
        )

        if (
            observed_rows
            != EXPECTED_ROWS[split_name]
        ):
            raise RuntimeError(
                f"{split_name}: satır sayısı "
                "uyuşmuyor. "
                f"Beklenen={EXPECTED_ROWS[split_name]}, "
                f"bulunan={observed_rows}"
            )

        observed_names = [
            field.name
            for field
            in split_parquet.schema_arrow
        ]

        if observed_names != expected_names:
            raise RuntimeError(
                f"{split_name}: sütun sırası "
                "train ile uyuşmuyor."
            )

    return (
        feature_columns,
        label_column,
    )


def iter_frames(
    parquet_path: Path,
    columns: list[str],
):
    parquet_file = pq.ParquetFile(
        parquet_path
    )

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns,
    ):
        yield batch.to_pandas()


def feature_matrix(
    frame,
    feature_columns: list[str],
) -> np.ndarray:
    values = frame[
        feature_columns
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
        order="C",
    )

    if (
        values.ndim != 2
        or values.shape[1]
        != FEATURE_COUNT
    ):
        raise RuntimeError(
            "Geçersiz özellik matrisi: "
            f"{values.shape}"
        )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Kaynak özelliklerde NaN veya "
            "sonsuz değer bulundu."
        )

    return values


def encode_labels(
    values: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(
        values,
        dtype=str,
    )

    normalized = np.char.lower(
        np.char.strip(labels)
    )

    encoded = np.full(
        normalized.shape[0],
        -1,
        dtype=np.int8,
    )

    encoded[
        normalized == "benign"
    ] = 0

    encoded[
        np.char.startswith(
            normalized,
            "gafgyt",
        )
    ] = 1

    encoded[
        np.char.startswith(
            normalized,
            "mirai",
        )
    ] = 2

    invalid_mask = encoded < 0

    if np.any(invalid_mask):
        invalid_values = np.unique(
            normalized[
                invalid_mask
            ]
        )

        raise RuntimeError(
            "Bilinmeyen sınıf etiketleri: "
            f"{invalid_values.tolist()}"
        )

    return encoded


def fit_scalers(
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_file = (
        SOURCE_DIRECTORY
        / "train.parquet"
    )

    scaler_a = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    scaler_b = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    processed_rows = 0

    print()
    print(
        "Pipeline A ve B scalerları "
        "yalnız train üzerinde öğreniliyor..."
    )

    for batch_index, frame in enumerate(
        iter_frames(
            train_file,
            feature_columns,
        ),
        start=1,
    ):
        raw = feature_matrix(
            frame,
            feature_columns,
        )

        pipeline_a_source = (
            raw.astype(
                np.float32
            )
            .astype(
                np.float64
            )
        )

        scaler_a.partial_fit(
            pipeline_a_source
        )

        scaler_b.partial_fit(
            raw
        )

        processed_rows += int(
            raw.shape[0]
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or processed_rows
            == EXPECTED_ROWS["train"]
        ):
            print(
                f"  scaler fit: "
                f"{processed_rows:,}/"
                f"{EXPECTED_ROWS['train']:,}"
            )

    if (
        processed_rows
        != EXPECTED_ROWS["train"]
    ):
        raise RuntimeError(
            "Scaler fit satır sayısı "
            "uyuşmuyor."
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

    for name, values in {
        "mean_a": mean_a,
        "scale_a": scale_a,
        "mean_b": mean_b,
        "scale_b": scale_b,
    }.items():
        if values.shape != (
            FEATURE_COUNT,
        ):
            raise RuntimeError(
                f"{name}: boyut hatası "
                f"{values.shape}"
            )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{name}: sonlu olmayan "
                "değer bulundu."
            )

    if (
        np.any(scale_a <= 0)
        or np.any(scale_b <= 0)
    ):
        raise RuntimeError(
            "Scaler scale değerlerinden "
            "biri pozitif değil."
        )

    return (
        mean_a,
        scale_a,
        mean_b,
        scale_b,
    )


def transform_a(
    raw: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    rounded = (
        raw.astype(
            np.float32
        )
        .astype(
            np.float64
        )
    )

    return (
        (rounded - mean)
        / scale
    ).astype(
        np.float32
    )


def transform_b(
    raw: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (
        (raw - mean)
        / scale
    ).astype(
        np.float32
    )


def validate_npy(
    path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> bool:
    if not path.exists():
        return False

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return (
            array.shape == expected_shape
            and array.dtype
            == np.dtype(expected_dtype)
        )

    except Exception:
        return False


def process_split(
    split_name: str,
    feature_columns: list[str],
    label_column: str,
    mean_a: np.ndarray,
    scale_a: np.ndarray,
    mean_b: np.ndarray,
    scale_b: np.ndarray,
) -> dict[str, Any]:
    expected_rows = (
        EXPECTED_ROWS[split_name]
    )

    source_file = (
        SOURCE_DIRECTORY
        / f"{split_name}.parquet"
    )

    output_a = (
        PIPELINE_A_DIRECTORY
        / f"x_{split_name}.npy"
    )

    output_b = (
        PIPELINE_B_DIRECTORY
        / f"x_{split_name}.npy"
    )

    output_y = (
        SHARED_DIRECTORY
        / f"y_{split_name}.npy"
    )

    partial_a = output_a.with_name(
        output_a.stem
        + ".partial.npy"
    )

    partial_b = output_b.with_name(
        output_b.stem
        + ".partial.npy"
    )

    partial_y = output_y.with_name(
        output_y.stem
        + ".partial.npy"
    )

    state_file = (
        STATE_DIRECTORY
        / f"{split_name}_complete.json"
    )

    if state_file.exists():
        state = load_json(
            state_file
        )

        existing_valid = (
            state.get(
                "protocol_version"
            )
            == PROTOCOL_VERSION
            and validate_npy(
                output_a,
                (
                    expected_rows,
                    FEATURE_COUNT,
                ),
                np.float32,
            )
            and validate_npy(
                output_b,
                (
                    expected_rows,
                    FEATURE_COUNT,
                ),
                np.float32,
            )
            and validate_npy(
                output_y,
                (
                    expected_rows,
                ),
                np.int8,
            )
        )

        if existing_valid:
            print(
                f"{split_name}: tamamlanmış "
                "önbellek doğrulandı, atlanıyor."
            )

            return state

    for path in (
        partial_a,
        partial_b,
        partial_y,
    ):
        safe_remove(path)

    if state_file.exists():
        safe_remove(state_file)

    for path in (
        output_a,
        output_b,
        output_y,
    ):
        safe_remove(path)

    print()
    print("-" * 78)
    print(
        f"{split_name.upper()} önbelleği "
        f"oluşturuluyor ({expected_rows:,} satır)"
    )

    memmap_a = np.lib.format.open_memmap(
        partial_a,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected_rows,
            FEATURE_COUNT,
        ),
    )

    memmap_b = np.lib.format.open_memmap(
        partial_b,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected_rows,
            FEATURE_COUNT,
        ),
    )

    memmap_y = np.lib.format.open_memmap(
        partial_y,
        mode="w+",
        dtype=np.int8,
        shape=(
            expected_rows,
        ),
    )

    row_offset = 0
    different_row_count = 0
    different_element_count = 0
    absolute_difference_sum = 0.0
    maximum_absolute_difference = 0.0

    class_counts = np.zeros(
        len(CLASS_NAMES),
        dtype=np.int64,
    )

    selected_columns = (
        feature_columns
        + [label_column]
    )

    for batch_index, frame in enumerate(
        iter_frames(
            source_file,
            selected_columns,
        ),
        start=1,
    ):
        raw = feature_matrix(
            frame,
            feature_columns,
        )

        labels = encode_labels(
            frame[
                label_column
            ].to_numpy()
        )

        transformed_a = transform_a(
            raw,
            mean_a,
            scale_a,
        )

        transformed_b = transform_b(
            raw,
            mean_b,
            scale_b,
        )

        if (
            not np.isfinite(
                transformed_a
            ).all()
            or not np.isfinite(
                transformed_b
            ).all()
        ):
            raise RuntimeError(
                f"{split_name}: dönüşüm "
                "sonrasında sonlu olmayan "
                "değer bulundu."
            )

        batch_rows = int(
            raw.shape[0]
        )

        end_offset = (
            row_offset
            + batch_rows
        )

        memmap_a[
            row_offset:end_offset
        ] = transformed_a

        memmap_b[
            row_offset:end_offset
        ] = transformed_b

        memmap_y[
            row_offset:end_offset
        ] = labels

        difference_mask = (
            transformed_a
            != transformed_b
        )

        different_row_count += int(
            np.any(
                difference_mask,
                axis=1,
            ).sum()
        )

        different_element_count += int(
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

        absolute_difference_sum += float(
            absolute_difference.sum(
                dtype=np.float64
            )
        )

        if absolute_difference.size:
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(
                    absolute_difference.max()
                ),
            )

        class_counts += np.bincount(
            labels.astype(
                np.int64
            ),
            minlength=len(
                CLASS_NAMES
            ),
        )

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

    memmap_a.flush()
    memmap_b.flush()
    memmap_y.flush()

    del memmap_a
    del memmap_b
    del memmap_y

    os.replace(
        partial_a,
        output_a,
    )

    os.replace(
        partial_b,
        output_b,
    )

    os.replace(
        partial_y,
        output_y,
    )

    element_count = (
        expected_rows
        * FEATURE_COUNT
    )

    state = {
        "protocol_version":
            PROTOCOL_VERSION,

        "status":
            "completed",

        "completed_at":
            utc_now(),

        "split":
            split_name,

        "row_count":
            expected_rows,

        "element_count":
            element_count,

        "different_row_count":
            different_row_count,

        "different_row_rate":
            (
                different_row_count
                / expected_rows
            ),

        "different_element_count":
            different_element_count,

        "different_element_rate":
            (
                different_element_count
                / element_count
            ),

        "absolute_difference_sum":
            absolute_difference_sum,

        "mean_absolute_difference":
            (
                absolute_difference_sum
                / element_count
            ),

        "maximum_absolute_difference":
            maximum_absolute_difference,

        "class_counts": {
            CLASS_NAMES[index]:
                int(class_counts[index])
            for index in range(
                len(CLASS_NAMES)
            )
        },

        "files": {
            "pipeline_a":
                str(output_a),

            "pipeline_b":
                str(output_b),

            "labels":
                str(output_y),
        },
    }

    save_json(
        state_file,
        state,
    )

    return state


required_inputs = [
    PHASE_1D_SUMMARY,
    PILOT_PROTOCOL,
    SCRIPT_PATH,
]

for split_name in EXPECTED_ROWS:
    required_inputs.append(
        SOURCE_DIRECTORY
        / f"{split_name}.parquet"
    )

missing_inputs = [
    str(path)
    for path in required_inputs
    if not path.exists()
]

if missing_inputs:
    print("Eksik giriş dosyaları:")

    for path in missing_inputs:
        print(f"- {path}")

    sys.exit(1)


phase_1d = load_json(
    PHASE_1D_SUMMARY
)

pilot_protocol = load_json(
    PILOT_PROTOCOL
)

if (
    phase_1d.get(
        "all_checks_passed"
    )
    is not True
    or phase_1d.get(
        "requires_training_pilot"
    )
    is not True
):
    raise RuntimeError(
        "Faz 1D pilot kararı geçerli değil."
    )

if (
    pilot_protocol.get(
        "status"
    )
    != "locked_before_training"
    or pilot_protocol.get(
        "run_count"
    )
    != 8
):
    raise RuntimeError(
        "Kilitli pilot protokolü "
        "beklenen durumda değil."
    )


for directory in (
    CACHE_ROOT,
    SHARED_DIRECTORY,
    PIPELINE_A_DIRECTORY,
    PIPELINE_B_DIRECTORY,
    STATE_DIRECTORY,
    AUDIT_DIRECTORY,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


estimated_cache_bytes = (
    2
    * sum(
        EXPECTED_ROWS.values()
    )
    * FEATURE_COUNT
    * np.dtype(
        np.float32
    ).itemsize
    +
    sum(
        EXPECTED_ROWS.values()
    )
    * np.dtype(
        np.int8
    ).itemsize
)

free_disk_bytes = shutil.disk_usage(
    CACHE_ROOT
).free

print("=" * 78)
print("FAZ 1E-A — SAYISAL HAT PİLOT ÖNBELLEĞİ")
print("=" * 78)
print(f"Kaynak       : {SOURCE_DIRECTORY}")
print(f"Çıktı        : {CACHE_ROOT}")
print(
    "Tahmini toplam önbellek: "
    f"{estimated_cache_bytes / (1024 ** 3):.3f} GiB"
)
print(
    "Boş disk              : "
    f"{free_disk_bytes / (1024 ** 3):.3f} GiB"
)

if (
    free_disk_bytes
    < estimated_cache_bytes
    + 1 * 1024 ** 3
):
    raise RuntimeError(
        "Önbellek ve 1 GiB güvenlik "
        "payı için yeterli disk yok."
    )


feature_columns, label_column = (
    discover_schema()
)

print(f"Özellik sayısı: {len(feature_columns)}")
print(f"Etiket sütunu : {label_column}")


started_at = time.perf_counter()

(
    mean_a,
    scale_a,
    mean_b,
    scale_b,
) = fit_scalers(
    feature_columns
)


np.savez_compressed(
    PIPELINE_A_DIRECTORY
    / "scaler.npz",
    mean=mean_a,
    scale=scale_a,
    feature_names=np.asarray(
        feature_columns,
        dtype="U",
    ),
    fit_split=np.asarray(
        ["train"],
        dtype="U",
    ),
    fit_row_count=np.asarray(
        [
            EXPECTED_ROWS["train"]
        ],
        dtype=np.int64,
    ),
    source_dtype=np.asarray(
        ["float64_to_float32"],
        dtype="U",
    ),
)

np.savez_compressed(
    PIPELINE_B_DIRECTORY
    / "scaler.npz",
    mean=mean_b,
    scale=scale_b,
    feature_names=np.asarray(
        feature_columns,
        dtype="U",
    ),
    fit_split=np.asarray(
        ["train"],
        dtype="U",
    ),
    fit_row_count=np.asarray(
        [
            EXPECTED_ROWS["train"]
        ],
        dtype=np.int64,
    ),
    source_dtype=np.asarray(
        ["float64"],
        dtype="U",
    ),
)


save_json(
    SHARED_DIRECTORY
    / "schema.json",
    {
        "protocol_version":
            PROTOCOL_VERSION,

        "feature_count":
            FEATURE_COUNT,

        "feature_names":
            feature_columns,

        "label_column":
            label_column,

        "class_names":
            CLASS_NAMES,

        "class_mapping": {
            "benign": 0,
            "gafgyt_*": 1,
            "mirai_*": 2,
        },

        "model_input_dtype":
            "float32",

        "stored_label_dtype":
            "int8",
    },
)


split_states = []

for split_name in (
    "train",
    "validation",
    "test",
):
    split_states.append(
        process_split(
            split_name,
            feature_columns,
            label_column,
            mean_a,
            scale_a,
            mean_b,
            scale_b,
        )
    )


train_state = next(
    state
    for state in split_states
    if state["split"] == "train"
)

train_counts = np.asarray(
    [
        train_state[
            "class_counts"
        ][class_name]
        for class_name
        in CLASS_NAMES
    ],
    dtype=np.float64,
)

if np.any(train_counts <= 0):
    raise RuntimeError(
        "Train splitinde sıfır destekli "
        "sınıf bulundu."
    )

raw_weights = (
    1.0
    / np.sqrt(
        train_counts
    )
)

class_weights = (
    raw_weights
    / raw_weights.mean()
).astype(
    np.float32
)

np.save(
    SHARED_DIRECTORY
    / "class_weights.npy",
    class_weights,
    allow_pickle=False,
)


global_row_count = sum(
    int(state["row_count"])
    for state in split_states
)

global_element_count = sum(
    int(state["element_count"])
    for state in split_states
)

global_different_row_count = sum(
    int(
        state[
            "different_row_count"
        ]
    )
    for state in split_states
)

global_different_element_count = sum(
    int(
        state[
            "different_element_count"
        ]
    )
    for state in split_states
)

global_absolute_difference_sum = sum(
    float(
        state[
            "absolute_difference_sum"
        ]
    )
    for state in split_states
)

global_mean_absolute_difference = (
    global_absolute_difference_sum
    / global_element_count
)

global_maximum_absolute_difference = max(
    float(
        state[
            "maximum_absolute_difference"
        ]
    )
    for state in split_states
)


phase_1d_match_checks = {
    "row_count":
        global_row_count
        == int(
            phase_1d[
                "global_row_count"
            ]
        ),

    "different_row_count":
        global_different_row_count
        == int(
            phase_1d[
                "global_different_row_count"
            ]
        ),

    "element_count":
        global_element_count
        == int(
            phase_1d[
                "global_element_count"
            ]
        ),

    "different_element_count":
        global_different_element_count
        == int(
            phase_1d[
                "global_different_element_count"
            ]
        ),

    "mean_absolute_difference":
        math.isclose(
            global_mean_absolute_difference,
            float(
                phase_1d[
                    "global_mean_absolute_difference"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),

    "maximum_absolute_difference":
        math.isclose(
            global_maximum_absolute_difference,
            float(
                phase_1d[
                    "global_maximum_absolute_difference"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
}


cache_validation_checks = {
    "feature_count":
        len(feature_columns)
        == FEATURE_COUNT,

    "global_row_count":
        global_row_count
        == sum(
            EXPECTED_ROWS.values()
        ),

    "all_phase_1d_metrics_match":
        all(
            phase_1d_match_checks.values()
        ),

    "train_class_count_total":
        int(
            train_counts.sum()
        )
        == EXPECTED_ROWS["train"],

    "class_weights_finite":
        bool(
            np.isfinite(
                class_weights
            ).all()
        ),

    "class_weights_positive":
        bool(
            np.all(
                class_weights > 0
            )
        ),
}


for split_name, rows in (
    EXPECTED_ROWS.items()
):
    cache_validation_checks[
        f"{split_name}_pipeline_a"
    ] = validate_npy(
        PIPELINE_A_DIRECTORY
        / f"x_{split_name}.npy",
        (
            rows,
            FEATURE_COUNT,
        ),
        np.float32,
    )

    cache_validation_checks[
        f"{split_name}_pipeline_b"
    ] = validate_npy(
        PIPELINE_B_DIRECTORY
        / f"x_{split_name}.npy",
        (
            rows,
            FEATURE_COUNT,
        ),
        np.float32,
    )

    cache_validation_checks[
        f"{split_name}_labels"
    ] = validate_npy(
        SHARED_DIRECTORY
        / f"y_{split_name}.npy",
        (
            rows,
        ),
        np.int8,
    )


all_checks_passed = all(
    cache_validation_checks.values()
)


elapsed_seconds = (
    time.perf_counter()
    - started_at
)


summary = {
    "protocol_version":
        PROTOCOL_VERSION,

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "elapsed_seconds":
        round(
            elapsed_seconds,
            3,
        ),

    "cache_root":
        str(CACHE_ROOT),

    "source_directory":
        str(SOURCE_DIRECTORY),

    "feature_count":
        FEATURE_COUNT,

    "feature_names":
        feature_columns,

    "label_column":
        label_column,

    "class_names":
        CLASS_NAMES,

    "class_counts_train": {
        CLASS_NAMES[index]:
            int(train_counts[index])
        for index in range(
            len(CLASS_NAMES)
        )
    },

    "class_weights": {
        CLASS_NAMES[index]:
            float(
                class_weights[index]
            )
        for index in range(
            len(CLASS_NAMES)
        )
    },

    "split_states":
        split_states,

    "global_row_count":
        global_row_count,

    "global_different_row_count":
        global_different_row_count,

    "global_different_row_rate":
        (
            global_different_row_count
            / global_row_count
        ),

    "global_element_count":
        global_element_count,

    "global_different_element_count":
        global_different_element_count,

    "global_different_element_rate":
        (
            global_different_element_count
            / global_element_count
        ),

    "global_mean_absolute_difference":
        global_mean_absolute_difference,

    "global_maximum_absolute_difference":
        global_maximum_absolute_difference,

    "scaler_maximum_mean_difference":
        float(
            np.max(
                np.abs(
                    mean_a - mean_b
                )
            )
        ),

    "scaler_maximum_scale_difference":
        float(
            np.max(
                np.abs(
                    scale_a - scale_b
                )
            )
        ),

    "phase_1d_match_checks":
        phase_1d_match_checks,

    "validation_checks":
        cache_validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


save_json(
    CACHE_SUMMARY,
    summary,
)


manifest_paths = [
    PIPELINE_A_DIRECTORY
    / "scaler.npz",

    PIPELINE_B_DIRECTORY
    / "scaler.npz",

    SHARED_DIRECTORY
    / "schema.json",

    SHARED_DIRECTORY
    / "class_weights.npy",

    CACHE_SUMMARY,
]

for split_name in EXPECTED_ROWS:
    manifest_paths.extend(
        [
            PIPELINE_A_DIRECTORY
            / f"x_{split_name}.npy",

            PIPELINE_B_DIRECTORY
            / f"x_{split_name}.npy",

            SHARED_DIRECTORY
            / f"y_{split_name}.npy",

            STATE_DIRECTORY
            / f"{split_name}_complete.json",
        ]
    )


manifest = {
    "protocol_version":
        PROTOCOL_VERSION,

    "created_at":
        utc_now(),

    "source_dependencies": [
        {
            "relative_path":
                str(
                    PHASE_1D_SUMMARY.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    PHASE_1D_SUMMARY
                ),
        },
        {
            "relative_path":
                str(
                    PILOT_PROTOCOL.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    PILOT_PROTOCOL
                ),
        },
        {
            "relative_path":
                str(
                    SCRIPT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    SCRIPT_PATH
                ),
        },
    ],

    "cache_files": [],
}


print()
print("Önbellek SHA-256 manifesti oluşturuluyor...")

for path in manifest_paths:
    print(
        f"  hash: "
        f"{path.relative_to(PROJECT_ROOT)}"
    )

    manifest[
        "cache_files"
    ].append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(path),
        }
    )


save_json(
    CACHE_MANIFEST,
    manifest,
)


final_cache_size = sum(
    path.stat().st_size
    for path in manifest_paths
    if path.exists()
)


print()
print("=" * 78)
print("FAZ 1E-A ÖNBELLEK ÖZETİ")
print("=" * 78)

print(
    f"Toplam satır               : "
    f"{global_row_count:,}"
)

print(
    f"Özellik sayısı             : "
    f"{FEATURE_COUNT}"
)

print(
    f"Farklı satır oranı         : "
    f"{100 * summary['global_different_row_rate']:.6f}%"
)

print(
    f"Ortalama mutlak fark       : "
    f"{global_mean_absolute_difference:.12g}"
)

print(
    f"Maksimum mutlak fark       : "
    f"{global_maximum_absolute_difference:.12g}"
)

print(
    f"Önbellek büyüklüğü         : "
    f"{final_cache_size / (1024 ** 3):.3f} GiB"
)

print()
print("TRAIN SINIF SAYILARI")

for class_name in CLASS_NAMES:
    print(
        f"{class_name}: "
        f"{summary['class_counts_train'][class_name]:,}"
    )

print()
print("TRAIN SINIF AĞIRLIKLARI")

for class_name in CLASS_NAMES:
    print(
        f"{class_name}: "
        f"{summary['class_weights'][class_name]:.9f}"
    )

print()
print("FAZ 1D UYUMLULUK KONTROLLERİ")

for name, passed in (
    phase_1d_match_checks.items()
):
    print(f"{name}: {passed}")

print()
print("ÖNBELLEK KONTROLLERİ")

for name, passed in (
    cache_validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"Önbellek : {CACHE_ROOT}")
print(f"Özet     : {CACHE_SUMMARY}")
print(f"Manifest : {CACHE_MANIFEST}")

if not all_checks_passed:
    print()
    print("FAZ 1E-A ÖNBELLEK BAŞARISIZ")
    sys.exit(1)

print()
print("FAZ 1E-A ÖNBELLEK BAŞARILI")
