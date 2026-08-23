from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nbaiot_primary_seed2026"
    / "metadata.parquet"
)

DEVICE_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "device_lodo_compact_summary_v2.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_membership_inspection_v2.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "audit"
    / "early_lodo_membership_inspection_v2.txt"
)

BATCH_SIZE = 100_000
SAMPLE_LIMIT = 8
UNIQUE_VALUE_LIMIT = 200


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_text(
    value: Any,
    maximum_length: int = 300,
) -> str:
    if value is None:
        return ""

    text = str(value)

    if len(text) > maximum_length:
        return (
            text[:maximum_length]
            + "..."
        )

    return text


def load_device_summary() -> tuple[
    list[str],
    dict[str, int],
]:
    with DEVICE_SUMMARY_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    devices = [
        row["device"].strip()
        for row in rows
    ]

    expected_counts = {
        row["device"].strip():
            int(row["fingerprint_count"])
        for row in rows
    }

    return (
        devices,
        expected_counts,
    )


required_files = [
    METADATA_FILE,
    DEVICE_SUMMARY_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print("Missing input files:")

    for path in missing_files:
        print(f"- {path}")

    sys.exit(1)


devices, expected_device_counts = (
    load_device_summary()
)

parquet_file = pq.ParquetFile(
    METADATA_FILE
)

schema = parquet_file.schema_arrow

column_names = [
    field.name
    for field in schema
]

column_types = {
    field.name:
        str(field.type)
    for field in schema
}


source_device_column = next(
    (
        name
        for name in column_names
        if name.casefold()
        == "source_device"
    ),
    None,
)

source_device_count_column = next(
    (
        name
        for name in column_names
        if name.casefold()
        == "source_device_count"
    ),
    None,
)


label_candidate_columns = [
    name
    for name in column_names
    if any(
        term in name.casefold()
        for term in (
            "family",
            "label",
            "class",
            "attack",
            "target",
        )
    )
]


hash_columns = [
    name
    for name in column_names
    if any(
        term in name.casefold()
        for term in (
            "hash",
            "fingerprint",
            "sha",
            "canonical",
        )
    )
]


occurrence_columns = [
    name
    for name in column_names
    if any(
        term in name.casefold()
        for term in (
            "occurrence",
            "frequency",
            "count",
        )
    )
]


if source_device_column is None:
    raise RuntimeError(
        "source_device column was not found."
    )

if source_device_count_column is None:
    raise RuntimeError(
        "source_device_count column was not found."
    )


selected_columns = []

for name in (
    [
        source_device_column,
        source_device_count_column,
    ]
    + label_candidate_columns
    + hash_columns
    + occurrence_columns
):
    if (
        name in column_names
        and name not in selected_columns
    ):
        selected_columns.append(name)


row_count = int(
    parquet_file.metadata.num_rows
)

device_observed_counts = {
    device: 0
    for device in devices
}

device_shared_counts = {
    device: 0
    for device in devices
}

device_count_distribution: Counter[int] = (
    Counter()
)

label_unique_values = {
    column: set()
    for column in label_candidate_columns
}

single_device_samples = []
multi_device_samples = []

processed_rows = 0
invalid_device_count_rows = 0
maximum_source_device_count = 0


for batch_index, batch in enumerate(
    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=selected_columns,
    ),
    start=1,
):
    frame = batch.to_pandas()

    device_text = (
        frame[source_device_column]
        .astype("string")
        .fillna("")
    )

    device_counts = np.asarray(
        frame[
            source_device_count_column
        ],
        dtype=np.int64,
    )

    invalid_device_count_rows += int(
        np.sum(
            device_counts < 1
        )
    )

    if device_counts.size:
        maximum_source_device_count = max(
            maximum_source_device_count,
            int(device_counts.max()),
        )

    unique_counts, count_frequencies = (
        np.unique(
            device_counts,
            return_counts=True,
        )
    )

    for value, frequency in zip(
        unique_counts,
        count_frequencies,
    ):
        device_count_distribution[
            int(value)
        ] += int(frequency)

    for device in devices:
        membership_mask = (
            device_text.str.contains(
                device,
                regex=False,
                na=False,
            )
        ).to_numpy(
            dtype=bool
        )

        device_observed_counts[
            device
        ] += int(
            membership_mask.sum()
        )

        device_shared_counts[
            device
        ] += int(
            np.logical_and(
                membership_mask,
                device_counts > 1,
            ).sum()
        )

    for column in label_candidate_columns:
        if (
            len(
                label_unique_values[
                    column
                ]
            )
            >= UNIQUE_VALUE_LIMIT
        ):
            continue

        values = (
            frame[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        label_unique_values[
            column
        ].update(values)

    if (
        len(single_device_samples)
        < SAMPLE_LIMIT
    ):
        single_indexes = np.flatnonzero(
            device_counts == 1
        )

        for index in single_indexes:
            sample = {
                column:
                    safe_text(
                        frame.iloc[
                            int(index)
                        ][column]
                    )
                for column in selected_columns
            }

            single_device_samples.append(
                sample
            )

            if (
                len(single_device_samples)
                >= SAMPLE_LIMIT
            ):
                break

    if (
        len(multi_device_samples)
        < SAMPLE_LIMIT
    ):
        multi_indexes = np.flatnonzero(
            device_counts > 1
        )

        for index in multi_indexes:
            sample = {
                column:
                    safe_text(
                        frame.iloc[
                            int(index)
                        ][column]
                    )
                for column in selected_columns
            }

            multi_device_samples.append(
                sample
            )

            if (
                len(multi_device_samples)
                >= SAMPLE_LIMIT
            ):
                break

    processed_rows += len(frame)

    if (
        batch_index == 1
        or batch_index % 5 == 0
        or processed_rows == row_count
    ):
        print(
            f"Scanned rows: "
            f"{processed_rows:,}/"
            f"{row_count:,}"
        )


device_count_matches = {
    device:
        (
            device_observed_counts[device]
            == expected_device_counts[device]
        )
    for device in devices
}


likely_family3_columns = []

for column, values in (
    label_unique_values.items()
):
    normalized = {
        str(value)
        .strip()
        .casefold()
        for value in values
    }

    family_hits = {
        "benign",
        "gafgyt",
        "mirai",
    }

    if family_hits.issubset(
        normalized
    ):
        likely_family3_columns.append(
            column
        )


validation_checks = {
    "metadata_row_count":
        processed_rows
        == row_count,

    "device_count":
        len(devices)
        == 9,

    "source_device_column":
        source_device_column
        is not None,

    "source_device_count_column":
        source_device_count_column
        is not None,

    "all_device_fingerprint_counts_match":
        all(
            device_count_matches.values()
        ),

    "device_count_distribution_complete":
        sum(
            device_count_distribution.values()
        )
        == row_count,

    "no_invalid_source_device_count":
        invalid_device_count_rows
        == 0,

    "multi_device_rows_exist":
        any(
            count > 1
            and frequency > 0
            for count, frequency
            in device_count_distribution.items()
        ),

    "family3_column_detected":
        len(
            likely_family3_columns
        )
        >= 1,

    "single_device_samples_collected":
        len(single_device_samples)
        > 0,

    "multi_device_samples_collected":
        len(multi_device_samples)
        > 0,
}


all_checks_passed = all(
    validation_checks.values()
)


summary = {
    "protocol_version":
        "early_lodo_membership_inspection_v2_1",

    "status":
        (
            "completed"
            if all_checks_passed
            else "failed"
        ),

    "completed_at":
        utc_now(),

    "metadata_file":
        str(METADATA_FILE),

    "metadata_row_count":
        row_count,

    "schema": [
        {
            "column":
                name,

            "type":
                column_types[name],
        }
        for name in column_names
    ],

    "source_device_column":
        source_device_column,

    "source_device_count_column":
        source_device_count_column,

    "label_candidate_columns":
        label_candidate_columns,

    "likely_family3_columns":
        likely_family3_columns,

    "hash_columns":
        hash_columns,

    "occurrence_columns":
        occurrence_columns,

    "selected_columns":
        selected_columns,

    "source_device_count_distribution": {
        str(key):
            int(value)
        for key, value in sorted(
            device_count_distribution.items()
        )
    },

    "maximum_source_device_count":
        maximum_source_device_count,

    "device_membership": {
        device: {
            "expected_fingerprint_count":
                expected_device_counts[
                    device
                ],

            "observed_fingerprint_count":
                device_observed_counts[
                    device
                ],

            "shared_fingerprint_count":
                device_shared_counts[
                    device
                ],

            "count_matches":
                device_count_matches[
                    device
                ],
        }
        for device in devices
    },

    "label_unique_values": {
        column:
            sorted(
                safe_text(value)
                for value in values
            )[:UNIQUE_VALUE_LIMIT]
        for column, values
        in label_unique_values.items()
    },

    "single_device_samples":
        single_device_samples,

    "multi_device_samples":
        multi_device_samples,

    "validation_checks":
        validation_checks,

    "all_checks_passed":
        all_checks_passed,
}


OUTPUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


lines = [
    "=" * 78,
    "PHASE 2B LODO MEMBERSHIP INSPECTION",
    "=" * 78,
    "",
    f"Metadata rows              : {row_count:,}",
    f"Source device column       : {source_device_column}",
    (
        "Source device count column : "
        f"{source_device_count_column}"
    ),
    (
        "Likely family-3 columns    : "
        + ", ".join(
            likely_family3_columns
        )
    ),
    (
        "Maximum device count       : "
        f"{maximum_source_device_count}"
    ),
    "",
    "SCHEMA",
]

for name in column_names:
    lines.append(
        f"- {name}: {column_types[name]}"
    )

lines.extend(
    [
        "",
        "SOURCE DEVICE COUNT DISTRIBUTION",
    ]
)

for count, frequency in sorted(
    device_count_distribution.items()
):
    lines.append(
        f"- {count}: {frequency:,}"
    )

lines.extend(
    [
        "",
        "DEVICE COUNT MATCH",
    ]
)

for device in devices:
    lines.append(
        (
            f"- {device} | "
            f"expected="
            f"{expected_device_counts[device]:,} | "
            f"observed="
            f"{device_observed_counts[device]:,} | "
            f"shared="
            f"{device_shared_counts[device]:,} | "
            f"match="
            f"{device_count_matches[device]}"
        )
    )

lines.extend(
    [
        "",
        "LABEL CANDIDATES",
    ]
)

for column, values in (
    label_unique_values.items()
):
    lines.append(
        f"- {column}: "
        + " | ".join(
            sorted(
                safe_text(value)
                for value in values
            )[:30]
        )
    )

lines.extend(
    [
        "",
        "SINGLE DEVICE SAMPLES",
    ]
)

for sample in single_device_samples:
    lines.append(
        json.dumps(
            sample,
            ensure_ascii=False,
        )
    )

lines.extend(
    [
        "",
        "MULTI DEVICE SAMPLES",
    ]
)

for sample in multi_device_samples:
    lines.append(
        json.dumps(
            sample,
            ensure_ascii=False,
        )
    )

lines.extend(
    [
        "",
        "VALIDATION CHECKS",
    ]
)

for name, passed in (
    validation_checks.items()
):
    lines.append(
        f"{name}: {passed}"
    )

lines.append("")
lines.append(
    f"Preflight passed: "
    f"{all_checks_passed}"
)


OUTPUT_REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print("=" * 78)
print("PHASE 2B LODO MEMBERSHIP INSPECTION")
print("=" * 78)
print(f"Metadata rows              : {row_count:,}")
print(f"Source device column       : {source_device_column}")
print(
    "Source device count column : "
    f"{source_device_count_column}"
)
print(
    "Likely family-3 columns    : "
    + ", ".join(
        likely_family3_columns
    )
)
print(
    "Maximum device count       : "
    f"{maximum_source_device_count}"
)

print()
print("SOURCE DEVICE COUNT DISTRIBUTION")

for count, frequency in sorted(
    device_count_distribution.items()
):
    print(
        f"{count}: {frequency:,}"
    )

print()
print("DEVICE COUNT MATCH")

for device in devices:
    print(
        f"{device} | "
        f"expected="
        f"{expected_device_counts[device]:,} | "
        f"observed="
        f"{device_observed_counts[device]:,} | "
        f"shared="
        f"{device_shared_counts[device]:,} | "
        f"match="
        f"{device_count_matches[device]}"
    )

print()
print("LABEL CANDIDATES")

for column, values in (
    label_unique_values.items()
):
    print(
        f"{column}: "
        + " | ".join(
            sorted(
                safe_text(value)
                for value in values
            )[:30]
        )
    )

print()
print("SINGLE DEVICE SAMPLES")

for sample in single_device_samples:
    print(
        json.dumps(
            sample,
            ensure_ascii=False,
        )
    )

print()
print("MULTI DEVICE SAMPLES")

for sample in multi_device_samples:
    print(
        json.dumps(
            sample,
            ensure_ascii=False,
        )
    )

print()
print("VALIDATION CHECKS")

for name, passed in (
    validation_checks.items()
):
    print(f"{name}: {passed}")

print()
print(f"JSON   : {OUTPUT_JSON}")
print(f"Report : {OUTPUT_REPORT}")

if not all_checks_passed:
    print()
    print("PHASE 2B LODO MEMBERSHIP INSPECTION FAILED")
    sys.exit(1)

print()
print("PHASE 2B LODO MEMBERSHIP INSPECTION PASSED")
