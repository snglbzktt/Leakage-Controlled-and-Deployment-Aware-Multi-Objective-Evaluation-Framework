from __future__ import annotations

import csv
import itertools
import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path.cwd()

BUILD_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2.building"
)

OVERLAP_DATABASE = (
    BUILD_CACHE
    / "scaled_model_input_overlap_audit.sqlite"
)

FAILURE_REPORT = (
    BUILD_CACHE
    / "scaled_overlap_failure_iteration1.json"
)

ITERATION_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration1_summary_v3_2.json"
)

ADDENDUM = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_v3_2.json"
)

ADDENDUM_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_manifest_v3_2.json"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_plan_v3_2.json"
)

OUTPUT_COMPONENTS_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_components_v3_2.csv"
)

OUTPUT_MOVES_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_moves_v3_2.csv"
)

FEATURE_COUNT = 115
CHUNK_SIZE = 100_000

EXPECTED_CROSS_SPLIT_COMPONENTS = 3
EXPECTED_TOTAL_FINGERPRINTS = 2_278_176
EXPECTED_TOTAL_RAW_ROWS = 7_062_606

SPLITS = (
    "train",
    "validation",
    "test",
)

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}

EXPECTED_SPLIT_COUNTS = {
    "train": {
        "fingerprints": 1_534_583,
        "raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_796,
        "raw_rows": 1_059_389,
    },
    "test": {
        "fingerprints": 371_797,
        "raw_rows": 1_059_394,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        path,
    )


def to_signed_int64(
    values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        values,
        dtype=np.uint64,
    ).view(np.int64)


def float32_fingerprints(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        values,
        dtype="<f4",
        order="C",
        copy=True,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[1] != FEATURE_COUNT
    ):
        raise RuntimeError(
            f"Expected (*, {FEATURE_COUNT}); "
            f"found {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Non-finite model input found."
        )

    matrix[matrix == 0.0] = 0.0

    words = matrix.view("<u4").reshape(
        matrix.shape
    )

    forward = np.full(
        matrix.shape[0],
        np.uint64(0xCBF29CE484222325),
        dtype=np.uint64,
    )

    reverse = np.full(
        matrix.shape[0],
        np.uint64(0x84222325CBF29CE4),
        dtype=np.uint64,
    )

    prime_forward = np.uint64(
        0x100000001B3
    )

    prime_reverse = np.uint64(
        0x9E3779B185EBCA87
    )

    with np.errstate(over="ignore"):
        for column in range(
            matrix.shape[1]
        ):
            word_forward = words[
                :,
                column,
            ].astype(
                np.uint64,
                copy=False,
            )

            word_reverse = words[
                :,
                matrix.shape[1] - 1 - column,
            ].astype(
                np.uint64,
                copy=False,
            )

            forward = (
                forward
                ^ (
                    word_forward
                    + np.uint64(column + 1)
                )
            ) * prime_forward

            reverse = (
                reverse
                ^ (
                    word_reverse
                    + np.uint64(column + 1)
                )
            ) * prime_reverse

            forward ^= (
                forward
                >> np.uint64(32)
            )

            reverse ^= (
                reverse
                >> np.uint64(29)
            )

    return (
        to_signed_int64(forward),
        to_signed_int64(reverse),
    )


required_paths = [
    BUILD_CACHE,
    OVERLAP_DATABASE,
    FAILURE_REPORT,
    ITERATION_REPORT,
    ADDENDUM,
    ADDENDUM_MANIFEST,
    BUILD_CACHE / "scaler_mean.npy",
    BUILD_CACHE / "scaler_scale.npy",
]

for split_name in SPLITS:
    required_paths.extend(
        [
            BUILD_CACHE / f"X_{split_name}.npy",
            BUILD_CACHE / f"y_{split_name}.npy",
            BUILD_CACHE / f"raw_row_count_{split_name}.npy",
            BUILD_CACHE / f"hash_forward_{split_name}.npy",
            BUILD_CACHE / f"hash_reverse_{split_name}.npy",
        ]
    )

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (
    OUTPUT_JSON,
    OUTPUT_COMPONENTS_CSV,
    OUTPUT_MOVES_CSV,
):
    if path.exists():
        raise FileExistsError(
            "Iteration-2 plan artifact already exists; "
            f"refusing to overwrite: {path}"
        )

failure = json.loads(
    FAILURE_REPORT.read_text(
        encoding="utf-8"
    )
)

iteration_report = json.loads(
    ITERATION_REPORT.read_text(
        encoding="utf-8"
    )
)

addendum = json.loads(
    ADDENDUM.read_text(
        encoding="utf-8"
    )
)

addendum_manifest = json.loads(
    ADDENDUM_MANIFEST.read_text(
        encoding="utf-8"
    )
)

input_checks = {
    "failure_report_failed": (
        failure.get("passed") is False
    ),
    "failure_report_component_count_matches": (
        int(
            failure[
                "cross_split_model_input_fingerprint_count"
            ]
        )
        == EXPECTED_CROSS_SPLIT_COMPONENTS
    ),
    "failure_report_family_conflicts_zero": (
        int(
            failure[
                "global_family_conflict_count"
            ]
        )
        == 0
        and int(
            failure[
                "within_split_family_conflict_count"
            ]
        )
        == 0
    ),
    "iteration_report_blocks_training": (
        iteration_report.get(
            "all_model_training_blocked"
        )
        is True
    ),
    "addendum_locked": (
        addendum.get("status")
        == "locked"
    ),
    "addendum_iteration_policy_present": (
        "repeat the same locked deterministic"
        in str(
            addendum[
                "repair_policy"
            ][
                "iteration_policy"
            ]
        ).lower()
    ),
    "addendum_manifest_locked": (
        addendum_manifest.get("status")
        == "locked"
    ),
}

failed_input_checks = [
    name
    for name, passed in input_checks.items()
    if not passed
]

if failed_input_checks:
    raise RuntimeError(
        "Iteration-2 input validation failed: "
        + ", ".join(failed_input_checks)
    )

database_uri = (
    OVERLAP_DATABASE.resolve().as_uri()
    + "?mode=ro"
)

with sqlite3.connect(
    database_uri,
    uri=True,
    timeout=120.0,
) as connection:
    quick_check = str(
        connection.execute(
            "PRAGMA quick_check"
        ).fetchone()[0]
    )

    if quick_check.lower() != "ok":
        raise RuntimeError(
            "Overlap database quick_check failed: "
            f"{quick_check}"
        )

    target_rows = connection.execute(
        """
        SELECT
            hash_forward,
            hash_reverse
        FROM scaled_fingerprints
        GROUP BY
            hash_forward,
            hash_reverse
        HAVING COUNT(
            DISTINCT split_code
        ) > 1
        ORDER BY
            hash_forward,
            hash_reverse
        """
    ).fetchall()

target_hashes = {
    (
        int(hash_forward),
        int(hash_reverse),
    )
    for hash_forward, hash_reverse
    in target_rows
}

if (
    len(target_hashes)
    != EXPECTED_CROSS_SPLIT_COMPONENTS
):
    raise RuntimeError(
        "Unexpected iteration-2 target count: "
        f"{len(target_hashes):,} != "
        f"{EXPECTED_CROSS_SPLIT_COMPONENTS:,}"
    )

mean = np.load(
    BUILD_CACHE / "scaler_mean.npy"
)

scale = np.load(
    BUILD_CACHE / "scaler_scale.npy"
)

if (
    mean.shape != (FEATURE_COUNT,)
    or scale.shape != (FEATURE_COUNT,)
    or mean.dtype != np.float64
    or scale.dtype != np.float64
):
    raise RuntimeError(
        "Invalid iteration-1 scaler artifacts."
    )

component_rows: dict[
    tuple[int, int],
    list[dict[str, Any]],
] = defaultdict(list)

processed_rows = 0
matched_rows = 0
started = time.perf_counter()

print("=" * 92)
print("SCALED COLLISION ITERATION-2 PLAN")
print("=" * 92)
print(
    "Target components: "
    f"{len(target_hashes):,}"
)

for split_code, split_name in enumerate(
    SPLITS
):
    x = np.load(
        BUILD_CACHE / f"X_{split_name}.npy",
        mmap_mode="r",
    )

    y = np.load(
        BUILD_CACHE / f"y_{split_name}.npy",
        mmap_mode="r",
    )

    raw_counts = np.load(
        BUILD_CACHE
        / f"raw_row_count_{split_name}.npy",
        mmap_mode="r",
    )

    original_forward = np.load(
        BUILD_CACHE
        / f"hash_forward_{split_name}.npy",
        mmap_mode="r",
    )

    original_reverse = np.load(
        BUILD_CACHE
        / f"hash_reverse_{split_name}.npy",
        mmap_mode="r",
    )

    for start in range(
        0,
        len(x),
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            len(x),
        )

        raw = np.asarray(
            x[start:end],
            dtype=np.float64,
        )

        scaled = (
            (
                raw - mean
            )
            / scale
        ).astype(
            np.float32
        )

        scaled_forward, scaled_reverse = (
            float32_fingerprints(
                scaled
            )
        )

        target_mask = np.zeros(
            end - start,
            dtype=bool,
        )

        for (
            target_forward,
            target_reverse,
        ) in target_hashes:
            target_mask |= (
                (
                    scaled_forward
                    == target_forward
                )
                & (
                    scaled_reverse
                    == target_reverse
                )
            )

        selected_local_indices = np.flatnonzero(
            target_mask
        )

        for local_index in (
            selected_local_indices.tolist()
        ):
            key = (
                int(
                    scaled_forward[
                        local_index
                    ]
                ),
                int(
                    scaled_reverse[
                        local_index
                    ]
                ),
            )

            row_index = (
                start + local_index
            )

            component_rows[key].append(
                {
                    "scaled_hash_forward": (
                        key[0]
                    ),
                    "scaled_hash_reverse": (
                        key[1]
                    ),
                    "source_split_code": (
                        split_code
                    ),
                    "source_split": (
                        split_name
                    ),
                    "source_row_index": int(
                        row_index
                    ),
                    "family": int(
                        y[row_index]
                    ),
                    "raw_row_count": int(
                        raw_counts[
                            row_index
                        ]
                    ),
                    "original_hash_forward": int(
                        original_forward[
                            row_index
                        ]
                    ),
                    "original_hash_reverse": int(
                        original_reverse[
                            row_index
                        ]
                    ),
                }
            )

            matched_rows += 1

        processed_rows += (
            end - start
        )

    print(
        f"[scan] {split_name:<10} "
        f"rows={len(x):,}",
        flush=True,
    )

if (
    processed_rows
    != EXPECTED_TOTAL_FINGERPRINTS
):
    raise RuntimeError(
        "Processed fingerprint count mismatch: "
        f"{processed_rows:,}"
    )

if set(component_rows) != target_hashes:
    missing = (
        target_hashes
        - set(component_rows)
    )

    raise RuntimeError(
        "Not all iteration-2 components were recovered: "
        f"missing={len(missing):,}"
    )

component_keys = sorted(
    component_rows
)

component_summaries: list[
    dict[str, Any]
] = []

candidate_destinations: list[
    tuple[int, ...]
] = []

for component_number, key in enumerate(
    component_keys,
    start=1,
):
    rows = component_rows[key]

    families = {
        int(row["family"])
        for row in rows
    }

    if len(families) != 1:
        raise RuntimeError(
            "Iteration-2 cross-family component found: "
            f"{key} -> {families}"
        )

    present_splits = tuple(
        sorted(
            {
                int(
                    row[
                        "source_split_code"
                    ]
                )
                for row in rows
            }
        )
    )

    if len(present_splits) < 2:
        raise RuntimeError(
            "Recovered component does not span splits: "
            f"{key}"
        )

    candidate_destinations.append(
        present_splits
    )

    by_split_count = [
        sum(
            int(
                row[
                    "source_split_code"
                ]
            )
            == split_code
            for row in rows
        )
        for split_code in range(3)
    ]

    by_split_raw = [
        sum(
            int(row["raw_row_count"])
            for row in rows
            if int(
                row[
                    "source_split_code"
                ]
            )
            == split_code
        )
        for split_code in range(3)
    ]

    component_summaries.append(
        {
            "component_number": (
                component_number
            ),
            "scaled_hash_forward": (
                key[0]
            ),
            "scaled_hash_reverse": (
                key[1]
            ),
            "family": int(
                next(iter(families))
            ),
            "split_span": (
                len(present_splits)
            ),
            "source_fingerprint_count": (
                len(rows)
            ),
            "source_raw_row_count": sum(
                int(row["raw_row_count"])
                for row in rows
            ),
            "train_fingerprint_count": (
                by_split_count[0]
            ),
            "validation_fingerprint_count": (
                by_split_count[1]
            ),
            "test_fingerprint_count": (
                by_split_count[2]
            ),
            "train_raw_row_count": (
                by_split_raw[0]
            ),
            "validation_raw_row_count": (
                by_split_raw[1]
            ),
            "test_raw_row_count": (
                by_split_raw[2]
            ),
        }
    )

best: tuple[
    tuple[Any, ...],
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
] | None = None

for destinations in itertools.product(
    *candidate_destinations
):
    fingerprint_delta = np.zeros(
        3,
        dtype=np.int64,
    )

    raw_row_delta = np.zeros(
        3,
        dtype=np.int64,
    )

    moved_fingerprints = 0
    moved_raw_rows = 0

    for (
        key,
        destination_code,
    ) in zip(
        component_keys,
        destinations,
    ):
        for row in component_rows[key]:
            source_code = int(
                row["source_split_code"]
            )

            if source_code == destination_code:
                continue

            raw_weight = int(
                row["raw_row_count"]
            )

            moved_fingerprints += 1
            moved_raw_rows += raw_weight

            fingerprint_delta[
                source_code
            ] -= 1

            fingerprint_delta[
                destination_code
            ] += 1

            raw_row_delta[
                source_code
            ] -= raw_weight

            raw_row_delta[
                destination_code
            ] += raw_weight

    objective = (
        moved_fingerprints,
        moved_raw_rows,
        int(
            np.abs(
                raw_row_delta
            ).sum()
        ),
        int(
            np.abs(
                fingerprint_delta
            ).sum()
        ),
        tuple(
            int(value)
            for value in destinations
        ),
    )

    candidate = (
        objective,
        tuple(
            int(value)
            for value in destinations
        ),
        fingerprint_delta.copy(),
        raw_row_delta.copy(),
    )

    if (
        best is None
        or candidate[0] < best[0]
    ):
        best = candidate

if best is None:
    raise RuntimeError(
        "No deterministic repair candidate found."
    )

(
    objective,
    chosen_destinations,
    fingerprint_delta,
    raw_row_delta,
) = best

move_records: list[
    dict[str, Any]
] = []

component_records: list[
    dict[str, Any]
] = []

for (
    component_number,
    key,
    destination_code,
    summary_row,
) in zip(
    range(
        1,
        len(component_keys) + 1,
    ),
    component_keys,
    chosen_destinations,
    component_summaries,
):
    rows = component_rows[key]

    moved_rows = [
        row
        for row in rows
        if int(
            row[
                "source_split_code"
            ]
        )
        != destination_code
    ]

    component_records.append(
        {
            **summary_row,
            "destination_split_code": (
                destination_code
            ),
            "destination_split": (
                SPLIT_NAMES[
                    destination_code
                ]
            ),
            "moved_fingerprint_count": (
                len(moved_rows)
            ),
            "moved_raw_row_count": sum(
                int(row["raw_row_count"])
                for row in moved_rows
            ),
        }
    )

    for row in moved_rows:
        move_records.append(
            {
                "component_number": (
                    component_number
                ),
                "scaled_hash_forward": (
                    key[0]
                ),
                "scaled_hash_reverse": (
                    key[1]
                ),
                "family": int(
                    row["family"]
                ),
                "source_split_code": int(
                    row[
                        "source_split_code"
                    ]
                ),
                "source_split": str(
                    row["source_split"]
                ),
                "destination_split_code": (
                    destination_code
                ),
                "destination_split": (
                    SPLIT_NAMES[
                        destination_code
                    ]
                ),
                "source_row_index": int(
                    row[
                        "source_row_index"
                    ]
                ),
                "raw_row_count": int(
                    row[
                        "raw_row_count"
                    ]
                ),
                "original_hash_forward": int(
                    row[
                        "original_hash_forward"
                    ]
                ),
                "original_hash_reverse": int(
                    row[
                        "original_hash_reverse"
                    ]
                ),
            }
        )

original_fingerprint_counts = np.asarray(
    [
        EXPECTED_SPLIT_COUNTS[
            split_name
        ]["fingerprints"]
        for split_name in SPLITS
    ],
    dtype=np.int64,
)

original_raw_counts = np.asarray(
    [
        EXPECTED_SPLIT_COUNTS[
            split_name
        ]["raw_rows"]
        for split_name in SPLITS
    ],
    dtype=np.int64,
)

proposed_fingerprint_counts = (
    original_fingerprint_counts
    + fingerprint_delta
)

proposed_raw_counts = (
    original_raw_counts
    + raw_row_delta
)

moved_fingerprint_count = len(
    move_records
)

moved_raw_row_count = sum(
    int(row["raw_row_count"])
    for row in move_records
)

checks = {
    **input_checks,
    "all_components_recovered": (
        len(component_rows)
        == EXPECTED_CROSS_SPLIT_COMPONENTS
    ),
    "all_components_single_family": all(
        len(
            {
                int(row["family"])
                for row in rows
            }
        )
        == 1
        for rows in component_rows.values()
    ),
    "all_components_span_multiple_splits": all(
        len(
            {
                int(
                    row[
                        "source_split_code"
                    ]
                )
                for row in rows
            }
        )
        >= 2
        for rows in component_rows.values()
    ),
    "fingerprint_total_preserved": (
        int(
            proposed_fingerprint_counts.sum()
        )
        == EXPECTED_TOTAL_FINGERPRINTS
    ),
    "raw_row_total_preserved": (
        int(
            proposed_raw_counts.sum()
        )
        == EXPECTED_TOTAL_RAW_ROWS
    ),
    "all_proposed_split_counts_positive": (
        bool(
            np.all(
                proposed_fingerprint_counts
                > 0
            )
        )
        and bool(
            np.all(
                proposed_raw_counts
                > 0
            )
        )
    ),
    "objective_move_count_matches_records": (
        moved_fingerprint_count
        == int(objective[0])
    ),
    "objective_raw_rows_match_records": (
        moved_raw_row_count
        == int(objective[1])
    ),
}

failed_checks = [
    name
    for name, passed in checks.items()
    if not passed
]

if failed_checks:
    raise RuntimeError(
        "Iteration-2 repair-plan validation failed: "
        + ", ".join(failed_checks)
    )

component_fieldnames = [
    "component_number",
    "scaled_hash_forward",
    "scaled_hash_reverse",
    "family",
    "split_span",
    "source_fingerprint_count",
    "source_raw_row_count",
    "train_fingerprint_count",
    "validation_fingerprint_count",
    "test_fingerprint_count",
    "train_raw_row_count",
    "validation_raw_row_count",
    "test_raw_row_count",
    "destination_split_code",
    "destination_split",
    "moved_fingerprint_count",
    "moved_raw_row_count",
]

move_fieldnames = [
    "component_number",
    "scaled_hash_forward",
    "scaled_hash_reverse",
    "family",
    "source_split_code",
    "source_split",
    "destination_split_code",
    "destination_split",
    "source_row_index",
    "raw_row_count",
    "original_hash_forward",
    "original_hash_reverse",
]

atomic_csv(
    OUTPUT_COMPONENTS_CSV,
    component_records,
    component_fieldnames,
)

atomic_csv(
    OUTPUT_MOVES_CSV,
    move_records,
    move_fieldnames,
)

split_plan = []

for split_code, split_name in enumerate(
    SPLITS
):
    split_plan.append(
        {
            "split_code": (
                split_code
            ),
            "split": split_name,
            "original_fingerprint_count": int(
                original_fingerprint_counts[
                    split_code
                ]
            ),
            "fingerprint_delta": int(
                fingerprint_delta[
                    split_code
                ]
            ),
            "proposed_fingerprint_count": int(
                proposed_fingerprint_counts[
                    split_code
                ]
            ),
            "original_raw_row_count": int(
                original_raw_counts[
                    split_code
                ]
            ),
            "raw_row_delta": int(
                raw_row_delta[
                    split_code
                ]
            ),
            "proposed_raw_row_count": int(
                proposed_raw_counts[
                    split_code
                ]
            ),
        }
    )

summary = {
    "status": "plan_only_not_applied",
    "repair_iteration": 2,
    "generated_at_utc": utc_now(),
    "protocol_version": (
        "tabular_baseline_protocol_v3_2"
    ),
    "source_derived_build_cache": str(
        BUILD_CACHE
    ),
    "cross_split_scaled_component_count": (
        len(component_records)
    ),
    "affected_original_fingerprint_row_count": (
        matched_rows
    ),
    "proposed_moved_fingerprint_count": (
        moved_fingerprint_count
    ),
    "proposed_moved_raw_row_count": (
        moved_raw_row_count
    ),
    "global_objective": {
        "moved_fingerprint_count": int(
            objective[0]
        ),
        "moved_raw_row_count": int(
            objective[1]
        ),
        "sum_absolute_raw_row_split_delta": int(
            objective[2]
        ),
        "sum_absolute_fingerprint_split_delta": int(
            objective[3]
        ),
        "destination_split_codes": list(
            objective[4]
        ),
    },
    "split_plan": split_plan,
    "component_span_distribution": {
        str(span): sum(
            int(
                row["split_span"]
            )
            == span
            for row in component_records
        )
        for span in (2, 3)
    },
    "family_distribution": {
        str(family): sum(
            int(row["family"])
            == family
            for row in component_records
        )
        for family in (0, 1, 2)
    },
    "components_csv": str(
        OUTPUT_COMPONENTS_CSV
    ),
    "moves_csv": str(
        OUTPUT_MOVES_CSV
    ),
    "checks": checks,
    "all_checks_passed": True,
    "elapsed_seconds": (
        time.perf_counter()
        - started
    ),
    "scientific_note": (
        "Iteration 2 follows the already locked v3_2 "
        "deterministic iteration policy. It is a plan only "
        "and does not mutate cache arrays, the split database, "
        "the locked protocol, or model results."
    ),
}

atomic_json(
    OUTPUT_JSON,
    summary,
)

print()
print("=" * 92)
print(
    "ITERATION-2 MINIMUM-MOVEMENT PLAN SUMMARY"
)
print("=" * 92)
print(
    "Scaled collision components      : "
    f"{len(component_records):,}"
)
print(
    "Affected original fingerprints   : "
    f"{matched_rows:,}"
)
print(
    "Minimum fingerprint moves        : "
    f"{moved_fingerprint_count:,}"
)
print(
    "Moved represented raw rows       : "
    f"{moved_raw_row_count:,}"
)

print()
print("FAMILY DISTRIBUTION")

for family in range(3):
    print(
        f"  family={family} | "
        f"components="
        f"{summary['family_distribution'][str(family)]:,}"
    )

print()
print("PROPOSED SPLIT DELTAS")

for row in split_plan:
    print(
        f"  {row['split']:<10} | "
        f"fingerprints "
        f"{row['original_fingerprint_count']:,} "
        f"{row['fingerprint_delta']:+,} -> "
        f"{row['proposed_fingerprint_count']:,} | "
        f"raw_rows "
        f"{row['original_raw_row_count']:,} "
        f"{row['raw_row_delta']:+,} -> "
        f"{row['proposed_raw_row_count']:,}"
    )

print()
print("PLAN ARTIFACTS")
print(f"  Summary    : {OUTPUT_JSON}")
print(f"  Components : {OUTPUT_COMPONENTS_CSV}")
print(f"  Moves      : {OUTPUT_MOVES_CSV}")
print()
print("Plan applied : False")
print("All checks passed: True")
print(
    "SCALED COLLISION ITERATION-2 PLAN COMPLETED"
)
