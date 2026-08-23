from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()

BASE_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_1.building"
)

DERIVED_FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

DERIVED_BUILD_CACHE = DERIVED_FINAL_CACHE.with_name(
    DERIVED_FINAL_CACHE.name + ".building"
)

BASE_FAILURE_REPORT = (
    BASE_CACHE
    / "scaled_overlap_failure.json"
)

BASE_OVERLAP_DATABASE = (
    BASE_CACHE
    / "scaled_model_input_overlap_audit.sqlite"
)

BASE_PROTOCOL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

BASE_PROTOCOL_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
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

REPAIR_PLAN = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_plan_v3.json"
)

COMPONENTS_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_components_v3.csv"
)

MOVES_CSV = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_moves_v3.csv"
)

SUCCESS_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration1_summary_v3_2.json"
)

PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"
BASE_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_1"

FEATURE_COUNT = 115
CHUNK_SIZE = 100_000
SCALER_AUDIT_TOLERANCE = 1e-4
MINIMUM_FREE_DISK_GIB = 2.5

EXPECTED_TOTAL_FINGERPRINTS = 2_278_176
EXPECTED_TOTAL_RAW_ROWS = 7_062_606
EXPECTED_MOVES = 353
EXPECTED_MOVED_RAW_ROWS = 353
EXPECTED_COMPONENTS = 280

SPLITS = (
    "train",
    "validation",
    "test",
)

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

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

EXPECTED_FAMILY_SPLIT_COUNTS = {
    ("train", 0): (359_439, 389_152),
    ("train", 1): (27_221, 1_986_790),
    ("train", 2): (1_147_923, 2_567_881),
    ("validation", 0): (77_021, 83_389),
    ("validation", 1): (48_793, 425_740),
    ("validation", 2): (245_982, 550_260),
    ("test", 0): (77_022, 83_391),
    ("test", 1): (48_793, 425_742),
    ("test", 2): (245_982, 550_261),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


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


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_move_rows() -> list[dict[str, Any]]:
    with MOVES_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    parsed: list[dict[str, Any]] = []

    for row in rows:
        parsed.append(
            {
                "component_number": int(
                    row["component_number"]
                ),
                "family": int(
                    row["family"]
                ),
                "source_split_code": int(
                    row["source_split_code"]
                ),
                "source_split": str(
                    row["source_split"]
                ),
                "destination_split_code": int(
                    row["destination_split_code"]
                ),
                "destination_split": str(
                    row["destination_split"]
                ),
                "source_row_index": int(
                    row["source_row_index"]
                ),
                "raw_row_count": int(
                    row["raw_row_count"]
                ),
                "original_hash_forward": int(
                    row["original_hash_forward"]
                ),
                "original_hash_reverse": int(
                    row["original_hash_reverse"]
                ),
                "scaled_hash_forward": int(
                    row["scaled_hash_forward"]
                ),
                "scaled_hash_reverse": int(
                    row["scaled_hash_reverse"]
                ),
            }
        )

    return parsed


def validate_inputs(
    move_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_paths = [
        BASE_CACHE,
        BASE_FAILURE_REPORT,
        BASE_OVERLAP_DATABASE,
        BASE_PROTOCOL,
        BASE_PROTOCOL_MANIFEST,
        ADDENDUM,
        ADDENDUM_MANIFEST,
        REPAIR_PLAN,
        COMPONENTS_CSV,
        MOVES_CSV,
    ]

    for split_name in SPLITS:
        required_paths.extend(
            [
                BASE_CACHE
                / f"X_{split_name}.npy",
                BASE_CACHE
                / f"y_{split_name}.npy",
                BASE_CACHE
                / f"raw_row_count_{split_name}.npy",
                BASE_CACHE
                / f"hash_forward_{split_name}.npy",
                BASE_CACHE
                / f"hash_reverse_{split_name}.npy",
            ]
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if DERIVED_FINAL_CACHE.exists():
        raise FileExistsError(
            "Derived final cache already exists; "
            f"refusing to overwrite: {DERIVED_FINAL_CACHE}"
        )

    if DERIVED_BUILD_CACHE.exists():
        raise FileExistsError(
            "Derived build directory already exists; "
            f"inspect before retrying: {DERIVED_BUILD_CACHE}"
        )

    base_protocol = read_json(
        BASE_PROTOCOL
    )

    base_manifest = read_json(
        BASE_PROTOCOL_MANIFEST
    )

    addendum = read_json(
        ADDENDUM
    )

    addendum_manifest = read_json(
        ADDENDUM_MANIFEST
    )

    repair_plan = read_json(
        REPAIR_PLAN
    )

    base_failure = read_json(
        BASE_FAILURE_REPORT
    )

    checks = {
        "base_protocol_locked": (
            base_protocol.get("status")
            == "locked"
        ),
        "base_protocol_version_matches": (
            base_protocol.get(
                "protocol_version"
            )
            == BASE_PROTOCOL_VERSION
        ),
        "base_protocol_hash_matches_manifest": (
            base_manifest.get(
                "protocol_file_sha256"
            )
            == sha256_file(
                BASE_PROTOCOL
            )
        ),
        "addendum_locked": (
            addendum.get("status")
            == "locked"
        ),
        "addendum_version_matches": (
            addendum.get(
                "protocol_version"
            )
            == PROTOCOL_VERSION
        ),
        "addendum_hash_matches_manifest": (
            addendum_manifest.get(
                "addendum_sha256"
            )
            == sha256_file(
                ADDENDUM
            )
        ),
        "repair_plan_hash_matches_addendum": (
            addendum[
                "provenance"
            ][
                "repair_plan_sha256"
            ]
            == sha256_file(
                REPAIR_PLAN
            )
        ),
        "moves_hash_matches_addendum": (
            addendum[
                "provenance"
            ][
                "moves_csv_sha256"
            ]
            == sha256_file(
                MOVES_CSV
            )
        ),
        "components_hash_matches_addendum": (
            addendum[
                "provenance"
            ][
                "components_csv_sha256"
            ]
            == sha256_file(
                COMPONENTS_CSV
            )
        ),
        "base_failure_has_280_components": (
            int(
                base_failure[
                    "cross_split_model_input_fingerprint_count"
                ]
            )
            == EXPECTED_COMPONENTS
        ),
        "move_count_matches": (
            len(move_rows)
            == EXPECTED_MOVES
        ),
        "moved_raw_rows_match": (
            sum(
                int(row["raw_row_count"])
                for row in move_rows
            )
            == EXPECTED_MOVED_RAW_ROWS
        ),
        "all_moves_benign": all(
            int(row["family"]) == 0
            for row in move_rows
        ),
        "all_moves_have_unit_raw_weight": all(
            int(row["raw_row_count"])
            == 1
            for row in move_rows
        ),
        "all_moves_cross_splits": all(
            int(row["source_split_code"])
            != int(
                row[
                    "destination_split_code"
                ]
            )
            for row in move_rows
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Derived repair input validation failed: "
            + ", ".join(failed)
        )

    return {
        "base_protocol": base_protocol,
        "base_manifest": base_manifest,
        "addendum": addendum,
        "addendum_manifest": addendum_manifest,
        "repair_plan": repair_plan,
        "base_failure": base_failure,
        "checks": checks,
    }


def load_moved_payloads(
    move_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[
        str,
        list[dict[str, Any]],
    ] = {
        split_name: []
        for split_name in SPLITS
    }

    for row in move_rows:
        by_source[
            str(row["source_split"])
        ].append(row)

    payloads: list[dict[str, Any]] = []

    for split_name in SPLITS:
        split_rows = sorted(
            by_source[split_name],
            key=lambda row: int(
                row["source_row_index"]
            ),
        )

        if not split_rows:
            continue

        x = np.load(
            BASE_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            BASE_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            BASE_CACHE
            / f"raw_row_count_{split_name}.npy",
            mmap_mode="r",
        )

        hash_forward = np.load(
            BASE_CACHE
            / f"hash_forward_{split_name}.npy",
            mmap_mode="r",
        )

        hash_reverse = np.load(
            BASE_CACHE
            / f"hash_reverse_{split_name}.npy",
            mmap_mode="r",
        )

        observed_indices: set[int] = set()

        for row in split_rows:
            index = int(
                row["source_row_index"]
            )

            if index in observed_indices:
                raise RuntimeError(
                    "Duplicate move source index: "
                    f"{split_name}/{index}"
                )

            observed_indices.add(index)

            if (
                index < 0
                or index >= len(x)
            ):
                raise RuntimeError(
                    "Move source index out of range: "
                    f"{split_name}/{index}"
                )

            observed_family = int(
                y[index]
            )

            observed_raw_count = int(
                raw_counts[index]
            )

            observed_forward = int(
                hash_forward[index]
            )

            observed_reverse = int(
                hash_reverse[index]
            )

            expected_values = {
                "family": int(
                    row["family"]
                ),
                "raw_row_count": int(
                    row["raw_row_count"]
                ),
                "hash_forward": int(
                    row[
                        "original_hash_forward"
                    ]
                ),
                "hash_reverse": int(
                    row[
                        "original_hash_reverse"
                    ]
                ),
            }

            observed_values = {
                "family": observed_family,
                "raw_row_count": (
                    observed_raw_count
                ),
                "hash_forward": (
                    observed_forward
                ),
                "hash_reverse": (
                    observed_reverse
                ),
            }

            if observed_values != expected_values:
                raise RuntimeError(
                    "Move source identity mismatch: "
                    f"{split_name}/{index} | "
                    f"observed={observed_values} | "
                    f"expected={expected_values}"
                )

            features = np.array(
                x[index],
                dtype=np.float32,
                copy=True,
            )

            actual_forward, actual_reverse = (
                float32_fingerprints(
                    features.reshape(
                        1,
                        FEATURE_COUNT,
                    )
                )
            )

            if (
                int(actual_forward[0])
                != observed_forward
                or int(actual_reverse[0])
                != observed_reverse
            ):
                raise RuntimeError(
                    "Moved-row feature hash mismatch: "
                    f"{split_name}/{index}"
                )

            payloads.append(
                {
                    **row,
                    "features": features,
                    "observed_family": (
                        observed_family
                    ),
                    "observed_raw_row_count": (
                        observed_raw_count
                    ),
                    "observed_hash_forward": (
                        observed_forward
                    ),
                    "observed_hash_reverse": (
                        observed_reverse
                    ),
                }
            )

    if len(payloads) != EXPECTED_MOVES:
        raise RuntimeError(
            "Moved payload count mismatch: "
            f"{len(payloads):,} != "
            f"{EXPECTED_MOVES:,}"
        )

    return payloads


def create_output_arrays(
    split_name: str,
) -> dict[str, np.memmap]:
    count = EXPECTED_SPLIT_COUNTS[
        split_name
    ]["fingerprints"]

    return {
        "X": np.lib.format.open_memmap(
            DERIVED_BUILD_CACHE
            / f"X_{split_name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                count,
                FEATURE_COUNT,
            ),
        ),
        "y": np.lib.format.open_memmap(
            DERIVED_BUILD_CACHE
            / f"y_{split_name}.npy",
            mode="w+",
            dtype=np.int8,
            shape=(count,),
        ),
        "raw_row_count": (
            np.lib.format.open_memmap(
                DERIVED_BUILD_CACHE
                / (
                    "raw_row_count_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            )
        ),
        "hash_forward": (
            np.lib.format.open_memmap(
                DERIVED_BUILD_CACHE
                / (
                    "hash_forward_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            )
        ),
        "hash_reverse": (
            np.lib.format.open_memmap(
                DERIVED_BUILD_CACHE
                / (
                    "hash_reverse_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            )
        ),
    }


def copy_segment(
    source: dict[str, np.ndarray],
    target: dict[str, np.memmap],
    source_start: int,
    source_end: int,
    target_start: int,
) -> int:
    amount = source_end - source_start

    if amount <= 0:
        return target_start

    target_end = target_start + amount

    target["X"][
        target_start:target_end
    ] = source["X"][
        source_start:source_end
    ]

    target["y"][
        target_start:target_end
    ] = source["y"][
        source_start:source_end
    ]

    target["raw_row_count"][
        target_start:target_end
    ] = source["raw_row_count"][
        source_start:source_end
    ]

    target["hash_forward"][
        target_start:target_end
    ] = source["hash_forward"][
        source_start:source_end
    ]

    target["hash_reverse"][
        target_start:target_end
    ] = source["hash_reverse"][
        source_start:source_end
    ]

    return target_end


def materialize_derived_cache(
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    outgoing_by_split: dict[
        str,
        list[int],
    ] = {
        split_name: []
        for split_name in SPLITS
    }

    incoming_by_split: dict[
        str,
        list[dict[str, Any]],
    ] = {
        split_name: []
        for split_name in SPLITS
    }

    for payload in payloads:
        outgoing_by_split[
            str(payload["source_split"])
        ].append(
            int(
                payload[
                    "source_row_index"
                ]
            )
        )

        incoming_by_split[
            str(payload["destination_split"])
        ].append(payload)

    split_records: list[
        dict[str, Any]
    ] = []

    for split_name in SPLITS:
        outgoing = sorted(
            outgoing_by_split[
                split_name
            ]
        )

        incoming = sorted(
            incoming_by_split[
                split_name
            ],
            key=lambda row: (
                int(
                    row[
                        "component_number"
                    ]
                ),
                int(
                    row[
                        "source_split_code"
                    ]
                ),
                int(
                    row[
                        "source_row_index"
                    ]
                ),
            ),
        )

        source = {
            "X": np.load(
                BASE_CACHE
                / f"X_{split_name}.npy",
                mmap_mode="r",
            ),
            "y": np.load(
                BASE_CACHE
                / f"y_{split_name}.npy",
                mmap_mode="r",
            ),
            "raw_row_count": np.load(
                BASE_CACHE
                / (
                    "raw_row_count_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
            "hash_forward": np.load(
                BASE_CACHE
                / (
                    "hash_forward_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
            "hash_reverse": np.load(
                BASE_CACHE
                / (
                    "hash_reverse_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
        }

        expected_count = int(
            EXPECTED_SPLIT_COUNTS[
                split_name
            ][
                "fingerprints"
            ]
        )

        if (
            len(source["X"])
            != expected_count
        ):
            raise RuntimeError(
                "Base split size mismatch: "
                f"{split_name}"
            )

        if (
            expected_count
            - len(outgoing)
            + len(incoming)
            != expected_count
        ):
            raise RuntimeError(
                "Derived split count would change: "
                f"{split_name} | "
                f"outgoing={len(outgoing)} | "
                f"incoming={len(incoming)}"
            )

        target = create_output_arrays(
            split_name
        )

        source_position = 0
        target_position = 0

        for outgoing_index in outgoing:
            if outgoing_index < source_position:
                raise RuntimeError(
                    "Outgoing indices are not strictly increasing: "
                    f"{split_name}"
                )

            target_position = copy_segment(
                source=source,
                target=target,
                source_start=source_position,
                source_end=outgoing_index,
                target_start=target_position,
            )

            source_position = (
                outgoing_index + 1
            )

        target_position = copy_segment(
            source=source,
            target=target,
            source_start=source_position,
            source_end=expected_count,
            target_start=target_position,
        )

        for payload in incoming:
            if target_position >= expected_count:
                raise RuntimeError(
                    "Derived split overflow while appending moves: "
                    f"{split_name}"
                )

            target["X"][
                target_position
            ] = payload["features"]

            target["y"][
                target_position
            ] = int(
                payload[
                    "observed_family"
                ]
            )

            target["raw_row_count"][
                target_position
            ] = int(
                payload[
                    "observed_raw_row_count"
                ]
            )

            target["hash_forward"][
                target_position
            ] = int(
                payload[
                    "observed_hash_forward"
                ]
            )

            target["hash_reverse"][
                target_position
            ] = int(
                payload[
                    "observed_hash_reverse"
                ]
            )

            target_position += 1

        if target_position != expected_count:
            raise RuntimeError(
                "Derived split final offset mismatch: "
                f"{split_name} | "
                f"{target_position:,} != "
                f"{expected_count:,}"
            )

        for array in target.values():
            array.flush()

        split_records.append(
            {
                "split": split_name,
                "source_fingerprint_count": (
                    expected_count
                ),
                "outgoing_fingerprint_count": (
                    len(outgoing)
                ),
                "incoming_fingerprint_count": (
                    len(incoming)
                ),
                "derived_fingerprint_count": (
                    target_position
                ),
            }
        )

        print(
            f"[repair materialization] "
            f"{split_name:<10} | "
            f"out={len(outgoing):,} | "
            f"in={len(incoming):,} | "
            f"rows={target_position:,}",
            flush=True,
        )

        del source
        del target

    return {
        "splits": split_records,
        "total_outgoing": sum(
            len(values)
            for values
            in outgoing_by_split.values()
        ),
        "total_incoming": sum(
            len(values)
            for values
            in incoming_by_split.values()
        ),
    }


def validate_derived_unscaled_cache() -> dict[str, Any]:
    split_records: list[
        dict[str, Any]
    ] = []

    total_fingerprints = 0
    total_raw_rows = 0

    for split_name in SPLITS:
        expected = EXPECTED_SPLIT_COUNTS[
            split_name
        ]

        x = np.load(
            DERIVED_BUILD_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            DERIVED_BUILD_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            DERIVED_BUILD_CACHE
            / (
                "raw_row_count_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        hash_forward = np.load(
            DERIVED_BUILD_CACHE
            / (
                "hash_forward_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        hash_reverse = np.load(
            DERIVED_BUILD_CACHE
            / (
                "hash_reverse_"
                f"{split_name}.npy"
            ),
            mmap_mode="r",
        )

        expected_count = int(
            expected["fingerprints"]
        )

        shape_checks = {
            "X": (
                x.shape
                == (
                    expected_count,
                    FEATURE_COUNT,
                )
                and x.dtype == np.float32
            ),
            "y": (
                y.shape
                == (expected_count,)
                and y.dtype == np.int8
            ),
            "raw_row_count": (
                raw_counts.shape
                == (expected_count,)
                and raw_counts.dtype
                == np.int64
            ),
            "hash_forward": (
                hash_forward.shape
                == (expected_count,)
                and hash_forward.dtype
                == np.int64
            ),
            "hash_reverse": (
                hash_reverse.shape
                == (expected_count,)
                and hash_reverse.dtype
                == np.int64
            ),
        }

        if not all(
            shape_checks.values()
        ):
            raise RuntimeError(
                "Derived shape/dtype check failed: "
                f"{split_name} -> {shape_checks}"
            )

        class_fingerprint_counts = (
            np.bincount(
                np.asarray(
                    y,
                    dtype=np.int64,
                ),
                minlength=3,
            )
            .astype(np.int64)
        )

        class_raw_row_counts = np.bincount(
            np.asarray(
                y,
                dtype=np.int64,
            ),
            weights=np.asarray(
                raw_counts,
                dtype=np.float64,
            ),
            minlength=3,
        ).astype(np.int64)

        expected_class_fingerprints = np.asarray(
            [
                EXPECTED_FAMILY_SPLIT_COUNTS[
                    (
                        split_name,
                        family,
                    )
                ][0]
                for family in range(3)
            ],
            dtype=np.int64,
        )

        expected_class_raw_rows = np.asarray(
            [
                EXPECTED_FAMILY_SPLIT_COUNTS[
                    (
                        split_name,
                        family,
                    )
                ][1]
                for family in range(3)
            ],
            dtype=np.int64,
        )

        if not np.array_equal(
            class_fingerprint_counts,
            expected_class_fingerprints,
        ):
            raise RuntimeError(
                "Derived class fingerprint counts changed: "
                f"{split_name}"
            )

        if not np.array_equal(
            class_raw_row_counts,
            expected_class_raw_rows,
        ):
            raise RuntimeError(
                "Derived class raw-row counts changed: "
                f"{split_name}"
            )

        observed_raw_rows = int(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            ).sum()
        )

        if observed_raw_rows != int(
            expected["raw_rows"]
        ):
            raise RuntimeError(
                "Derived split raw-row sum changed: "
                f"{split_name}"
            )

        if np.any(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            )
            <= 0
        ):
            raise RuntimeError(
                "Derived non-positive raw-row count: "
                f"{split_name}"
            )

        finite = True
        hash_mismatch_count = 0
        minimum_value = math.inf
        maximum_value = -math.inf

        for start in range(
            0,
            expected_count,
            CHUNK_SIZE,
        ):
            end = min(
                start + CHUNK_SIZE,
                expected_count,
            )

            values = np.asarray(
                x[start:end],
                dtype=np.float32,
            )

            finite = (
                finite
                and bool(
                    np.isfinite(
                        values
                    ).all()
                )
            )

            minimum_value = min(
                minimum_value,
                float(
                    np.min(values)
                ),
            )

            maximum_value = max(
                maximum_value,
                float(
                    np.max(values)
                ),
            )

            actual_forward, actual_reverse = (
                float32_fingerprints(
                    values
                )
            )

            hash_mismatch_count += int(
                np.count_nonzero(
                    actual_forward
                    != np.asarray(
                        hash_forward[
                            start:end
                        ],
                        dtype=np.int64,
                    )
                )
            )

            hash_mismatch_count += int(
                np.count_nonzero(
                    actual_reverse
                    != np.asarray(
                        hash_reverse[
                            start:end
                        ],
                        dtype=np.int64,
                    )
                )
            )

        if not finite:
            raise RuntimeError(
                "Derived non-finite feature value: "
                f"{split_name}"
            )

        if hash_mismatch_count != 0:
            raise RuntimeError(
                "Derived unscaled hash mismatch: "
                f"{split_name} -> "
                f"{hash_mismatch_count:,}"
            )

        split_records.append(
            {
                "split": split_name,
                "fingerprint_count": (
                    expected_count
                ),
                "raw_row_count": (
                    observed_raw_rows
                ),
                "class_fingerprint_counts": (
                    class_fingerprint_counts.tolist()
                ),
                "class_raw_row_counts": (
                    class_raw_row_counts.tolist()
                ),
                "all_values_finite": finite,
                "minimum_value": (
                    minimum_value
                ),
                "maximum_value": (
                    maximum_value
                ),
                "hash_mismatch_count": (
                    hash_mismatch_count
                ),
                "shape_dtype_checks": (
                    shape_checks
                ),
            }
        )

        total_fingerprints += (
            expected_count
        )

        total_raw_rows += (
            observed_raw_rows
        )

        print(
            f"[unscaled validation] "
            f"{split_name:<10} | "
            f"rows={expected_count:,} | "
            f"raw_rows={observed_raw_rows:,}",
            flush=True,
        )

    checks = {
        "total_fingerprint_count_matches": (
            total_fingerprints
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "total_raw_row_count_matches": (
            total_raw_rows
            == EXPECTED_TOTAL_RAW_ROWS
        ),
        "all_hashes_match": all(
            row["hash_mismatch_count"]
            == 0
            for row in split_records
        ),
        "all_values_finite": all(
            row["all_values_finite"]
            for row in split_records
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Derived unscaled validation failed: "
            + ", ".join(failed)
        )

    return {
        "splits": split_records,
        "total_fingerprint_count": (
            total_fingerprints
        ),
        "total_raw_row_count": (
            total_raw_rows
        ),
        "checks": checks,
    }


def audit_unscaled_hash_overlap() -> dict[str, Any]:
    database = (
        DERIVED_BUILD_CACHE
        / "unscaled_hash_overlap_audit.sqlite"
    )

    if database.exists():
        database.unlink()

    connection = sqlite3.connect(
        database,
        timeout=120.0,
    )

    started = time.perf_counter()

    try:
        connection.execute(
            "PRAGMA journal_mode=OFF"
        )

        connection.execute(
            "PRAGMA synchronous=OFF"
        )

        connection.execute(
            "PRAGMA temp_store=FILE"
        )

        connection.execute(
            """
            CREATE TABLE hashes (
                hash_forward INTEGER NOT NULL,
                hash_reverse INTEGER NOT NULL,
                split_code INTEGER NOT NULL,

                PRIMARY KEY (
                    hash_forward,
                    hash_reverse
                )
            ) WITHOUT ROWID
            """
        )

        insert_sql = """
            INSERT INTO hashes (
                hash_forward,
                hash_reverse,
                split_code
            )
            VALUES (?, ?, ?)
        """

        processed = 0

        for split_code, split_name in enumerate(
            SPLITS
        ):
            forward = np.load(
                DERIVED_BUILD_CACHE
                / (
                    "hash_forward_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            )

            reverse = np.load(
                DERIVED_BUILD_CACHE
                / (
                    "hash_reverse_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            )

            for start in range(
                0,
                len(forward),
                CHUNK_SIZE,
            ):
                end = min(
                    start + CHUNK_SIZE,
                    len(forward),
                )

                try:
                    connection.executemany(
                        insert_sql,
                        zip(
                            np.asarray(
                                forward[
                                    start:end
                                ],
                                dtype=np.int64,
                            ).tolist(),
                            np.asarray(
                                reverse[
                                    start:end
                                ],
                                dtype=np.int64,
                            ).tolist(),
                            [split_code]
                            * (
                                end - start
                            ),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise RuntimeError(
                        "Duplicate original float32 hash "
                        "found after repair."
                    ) from error

                processed += (
                    end - start
                )

            connection.commit()

        stored = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM hashes
                """
            ).fetchone()[0]
        )

        quick_check = str(
            connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        )

    finally:
        connection.close()

    checks = {
        "processed_count_matches": (
            processed
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "stored_unique_count_matches": (
            stored
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "quick_check_ok": (
            quick_check.lower() == "ok"
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Derived unscaled overlap audit failed: "
            + ", ".join(failed)
        )

    return {
        "processed_fingerprint_rows": (
            processed
        ),
        "stored_unique_fingerprints": (
            stored
        ),
        "cross_split_original_hash_overlap_count": 0,
        "database": str(database),
        "database_quick_check": (
            quick_check
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "checks": checks,
        "passed": True,
    }


def fit_and_audit_scaler() -> dict[str, Any]:
    x_train = np.load(
        DERIVED_BUILD_CACHE
        / "X_train.npy",
        mmap_mode="r",
    )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    for start in range(
        0,
        len(x_train),
        CHUNK_SIZE,
    ):
        scaler.partial_fit(
            np.asarray(
                x_train[
                    start:start + CHUNK_SIZE
                ],
                dtype=np.float32,
            )
        )

    mean = np.asarray(
        scaler.mean_,
        dtype=np.float64,
    )

    scale = np.asarray(
        scaler.scale_,
        dtype=np.float64,
    )

    if (
        mean.shape != (FEATURE_COUNT,)
        or scale.shape
        != (FEATURE_COUNT,)
    ):
        raise RuntimeError(
            "Repaired scaler parameter shape mismatch."
        )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError(
            "Invalid repaired scaler parameters."
        )

    np.save(
        DERIVED_BUILD_CACHE
        / "scaler_mean.npy",
        mean,
    )

    np.save(
        DERIVED_BUILD_CACHE
        / "scaler_scale.npy",
        scale,
    )

    scaled_sum = np.zeros(
        FEATURE_COUNT,
        dtype=np.float64,
    )

    scaled_square = np.zeros(
        FEATURE_COUNT,
        dtype=np.float64,
    )

    all_finite = True

    for start in range(
        0,
        len(x_train),
        CHUNK_SIZE,
    ):
        raw = np.asarray(
            x_train[
                start:start + CHUNK_SIZE
            ],
            dtype=np.float64,
        )

        scaled = (
            raw - mean
        ) / scale

        all_finite = (
            all_finite
            and bool(
                np.isfinite(
                    scaled
                ).all()
            )
        )

        scaled_sum += scaled.sum(
            axis=0
        )

        scaled_square += np.square(
            scaled
        ).sum(
            axis=0
        )

    scaled_mean = (
        scaled_sum / len(x_train)
    )

    scaled_variance = np.maximum(
        scaled_square / len(x_train)
        - np.square(
            scaled_mean
        ),
        0.0,
    )

    scaled_std = np.sqrt(
        scaled_variance
    )

    maximum_absolute_mean = float(
        np.max(
            np.abs(
                scaled_mean
            )
        )
    )

    maximum_std_error = float(
        np.max(
            np.abs(
                scaled_std - 1.0
            )
        )
    )

    passed = bool(
        all_finite
        and maximum_absolute_mean
        <= SCALER_AUDIT_TOLERANCE
        and maximum_std_error
        <= SCALER_AUDIT_TOLERANCE
    )

    if not passed:
        raise RuntimeError(
            "Repaired train-only scaler audit failed: "
            f"mean={maximum_absolute_mean:.3e}, "
            f"std_error={maximum_std_error:.3e}"
        )

    return {
        "fit_split": "train",
        "fit_sample_count": int(
            len(x_train)
        ),
        "fit_unit": (
            "one repaired representative per "
            "VS2 float32 fingerprint"
        ),
        "sample_weight": None,
        "method": (
            "sklearn.preprocessing."
            "StandardScaler.partial_fit"
        ),
        "parameter_dtype": "float64",
        "scaled_model_input_dtype": (
            "float32"
        ),
        "all_scaled_values_finite": (
            all_finite
        ),
        "maximum_absolute_scaled_mean": (
            maximum_absolute_mean
        ),
        "maximum_scaled_standard_deviation_error": (
            maximum_std_error
        ),
        "tolerance": (
            SCALER_AUDIT_TOLERANCE
        ),
        "passed": passed,
    }


def audit_scaled_overlap() -> dict[str, Any]:
    database = (
        DERIVED_BUILD_CACHE
        / "scaled_model_input_overlap_audit.sqlite"
    )

    if database.exists():
        database.unlink()

    mean = np.load(
        DERIVED_BUILD_CACHE
        / "scaler_mean.npy"
    )

    scale = np.load(
        DERIVED_BUILD_CACHE
        / "scaler_scale.npy"
    )

    connection = sqlite3.connect(
        database,
        timeout=120.0,
    )

    started = time.perf_counter()

    try:
        connection.execute(
            "PRAGMA journal_mode=OFF"
        )

        connection.execute(
            "PRAGMA synchronous=OFF"
        )

        connection.execute(
            "PRAGMA temp_store=FILE"
        )

        connection.execute(
            """
            CREATE TABLE scaled_fingerprints (
                hash_forward INTEGER NOT NULL,
                hash_reverse INTEGER NOT NULL,
                split_code INTEGER NOT NULL,
                family_min INTEGER NOT NULL,
                family_max INTEGER NOT NULL,
                row_count INTEGER NOT NULL,

                PRIMARY KEY (
                    hash_forward,
                    hash_reverse,
                    split_code
                )
            ) WITHOUT ROWID
            """
        )

        insert_sql = """
            INSERT INTO scaled_fingerprints (
                hash_forward,
                hash_reverse,
                split_code,
                family_min,
                family_max,
                row_count
            )
            VALUES (?, ?, ?, ?, ?, 1)

            ON CONFLICT (
                hash_forward,
                hash_reverse,
                split_code
            )
            DO UPDATE SET
                family_min = MIN(
                    family_min,
                    excluded.family_min
                ),
                family_max = MAX(
                    family_max,
                    excluded.family_max
                ),
                row_count = row_count + 1
        """

        processed_rows = 0
        unique_by_split: dict[
            str,
            int
        ] = {}

        for split_code, split_name in enumerate(
            SPLITS
        ):
            x = np.load(
                DERIVED_BUILD_CACHE
                / f"X_{split_name}.npy",
                mmap_mode="r",
            )

            y = np.load(
                DERIVED_BUILD_CACHE
                / f"y_{split_name}.npy",
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

                if not np.isfinite(
                    scaled
                ).all():
                    raise RuntimeError(
                        "Non-finite repaired scaled input: "
                        f"{split_name}"
                    )

                forward, reverse = (
                    float32_fingerprints(
                        scaled
                    )
                )

                families = np.asarray(
                    y[start:end],
                    dtype=np.int64,
                )

                connection.executemany(
                    insert_sql,
                    zip(
                        forward.tolist(),
                        reverse.tolist(),
                        [split_code]
                        * len(scaled),
                        families.tolist(),
                        families.tolist(),
                    ),
                )

                processed_rows += len(
                    scaled
                )

            connection.commit()

            unique_by_split[
                split_name
            ] = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM scaled_fingerprints
                    WHERE split_code = ?
                    """,
                    (split_code,),
                ).fetchone()[0]
            )

            print(
                "[repaired scaled audit] "
                f"{split_name:<10} | "
                f"rows={len(x):,} | "
                f"unique_scaled="
                f"{unique_by_split[split_name]:,}",
                flush=True,
            )

        cross_split_overlap = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
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
                )
                """
            ).fetchone()[0]
        )

        within_split_family_conflicts = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM scaled_fingerprints
                WHERE family_min
                      != family_max
                """
            ).fetchone()[0]
        )

        global_family_conflicts = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        hash_forward,
                        hash_reverse,
                        MIN(family_min)
                            AS minimum_family,
                        MAX(family_max)
                            AS maximum_family
                    FROM scaled_fingerprints
                    GROUP BY
                        hash_forward,
                        hash_reverse
                    HAVING minimum_family
                           != maximum_family
                )
                """
            ).fetchone()[0]
        )

        unique_total = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM scaled_fingerprints
                """
            ).fetchone()[0]
        )

        collapsed_within_split_rows = (
            processed_rows - unique_total
        )

        quick_check = str(
            connection.execute(
                "PRAGMA quick_check"
            ).fetchone()[0]
        )

    finally:
        connection.close()

    checks = {
        "processed_rows_match": (
            processed_rows
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "cross_split_overlap_zero": (
            cross_split_overlap == 0
        ),
        "within_split_family_conflicts_zero": (
            within_split_family_conflicts
            == 0
        ),
        "global_family_conflicts_zero": (
            global_family_conflicts == 0
        ),
        "quick_check_ok": (
            quick_check.lower() == "ok"
        ),
    }

    result = {
        "iteration": 1,
        "feature_space": (
            "repaired train-only StandardScaler "
            "float64 parameters then final float32 input"
        ),
        "processed_fingerprint_rows": (
            processed_rows
        ),
        "unique_scaled_fingerprints_by_split": (
            unique_by_split
        ),
        "collapsed_within_split_rows": (
            collapsed_within_split_rows
        ),
        "cross_split_model_input_fingerprint_count": (
            cross_split_overlap
        ),
        "within_split_family_conflict_count": (
            within_split_family_conflicts
        ),
        "global_family_conflict_count": (
            global_family_conflicts
        ),
        "database": str(database),
        "database_quick_check": (
            quick_check
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "checks": checks,
        "passed": all(
            checks.values()
        ),
    }

    if not result["passed"]:
        atomic_json(
            DERIVED_BUILD_CACHE
            / (
                "scaled_overlap_failure_"
                "iteration1.json"
            ),
            result,
        )

    return result


def artifact_inventory() -> list[
    dict[str, Any]
]:
    excluded = {
        "manifest.json",
        "build_status.json",
    }

    paths = sorted(
        path
        for path in DERIVED_BUILD_CACHE.iterdir()
        if path.is_file()
        and path.name not in excluded
    )

    records: list[
        dict[str, Any]
    ] = []

    for position, path in enumerate(
        paths,
        start=1,
    ):
        print(
            f"[artifact hash] "
            f"{position}/{len(paths)} | "
            f"{path.name}",
            flush=True,
        )

        records.append(
            {
                "relative_path": (
                    path.name
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    return records


def main() -> None:
    started = time.perf_counter()

    print("=" * 92)
    print(
        "APPLY SCALED-COLLISION REPAIR "
        "TO DERIVED TABULAR CACHE"
    )
    print("=" * 92)
    print(f"Base cache    : {BASE_CACHE}")
    print(
        f"Derived build: {DERIVED_BUILD_CACHE}"
    )
    print(
        f"Derived final: {DERIVED_FINAL_CACHE}"
    )

    move_rows = load_move_rows()

    input_validation = (
        validate_inputs(
            move_rows
        )
    )

    free_disk_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        / (1024**3)
    )

    if free_disk_gib < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(
            "Insufficient free disk space: "
            f"{free_disk_gib:.3f} GiB < "
            f"{MINIMUM_FREE_DISK_GIB:.3f} GiB"
        )

    DERIVED_BUILD_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DERIVED_BUILD_CACHE.mkdir()

    atomic_json(
        DERIVED_BUILD_CACHE
        / "build_status.json",
        {
            "status": "building",
            "iteration": 1,
            "started_at_utc": utc_now(),
            "builder_script": str(
                Path(__file__).resolve()
            ),
            "builder_script_sha256": (
                sha256_file(
                    Path(__file__).resolve()
                )
            ),
            "base_cache": str(
                BASE_CACHE
            ),
            "addendum": str(
                ADDENDUM
            ),
            "repair_plan": str(
                REPAIR_PLAN
            ),
            "moves_csv": str(
                MOVES_CSV
            ),
        },
    )

    print(
        "Loading and validating 353 moved rows...",
        flush=True,
    )

    payloads = load_moved_payloads(
        move_rows
    )

    print(
        "Materializing derived cache...",
        flush=True,
    )

    materialization = (
        materialize_derived_cache(
            payloads
        )
    )

    print(
        "Validating repaired unscaled cache...",
        flush=True,
    )

    unscaled_validation = (
        validate_derived_unscaled_cache()
    )

    print(
        "Auditing original float32 hash uniqueness...",
        flush=True,
    )

    unscaled_overlap_audit = (
        audit_unscaled_hash_overlap()
    )

    print(
        "Refitting train-only scaler...",
        flush=True,
    )

    scaler_audit = (
        fit_and_audit_scaler()
    )

    print(
        "Auditing repaired final float32 model input...",
        flush=True,
    )

    scaled_overlap_audit = (
        audit_scaled_overlap()
    )

    if not scaled_overlap_audit["passed"]:
        failure = {
            "status": (
                "iteration_1_requires_"
                "additional_repair"
            ),
            "generated_at_utc": utc_now(),
            "protocol_version": (
                PROTOCOL_VERSION
            ),
            "derived_build_cache": str(
                DERIVED_BUILD_CACHE
            ),
            "applied_move_count": (
                EXPECTED_MOVES
            ),
            "scaled_overlap_audit": (
                scaled_overlap_audit
            ),
            "next_action": (
                "Plan the next deterministic "
                "minimum-movement iteration from "
                "the derived build cache."
            ),
            "all_model_training_blocked": True,
        }

        atomic_json(
            SUCCESS_REPORT,
            failure,
        )

        print()
        print("=" * 92)
        print(
            "REPAIRED CACHE STILL HAS "
            "SCALED CROSS-SPLIT COLLISIONS"
        )
        print("=" * 92)
        print(
            "Cross-split overlap: "
            f"{scaled_overlap_audit['cross_split_model_input_fingerprint_count']:,}"
        )
        print(
            "Family conflicts   : "
            f"{scaled_overlap_audit['global_family_conflict_count']:,}"
        )
        print(
            "Derived build kept : "
            f"{DERIVED_BUILD_CACHE}"
        )
        print(
            "No model training was started."
        )

        raise RuntimeError(
            "Repair iteration 1 did not reach "
            "zero scaled cross-split overlap."
        )

    artifacts = artifact_inventory()

    completed_at = utc_now()

    manifest = {
        "status": "completed",
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "created_at_utc": completed_at,
        "repair_iteration": 1,
        "cache_role": (
            "family_3 classical tabular "
            "baseline derived cache"
        ),
        "cache_directory": str(
            DERIVED_FINAL_CACHE
        ),
        "base_cache": str(
            BASE_CACHE
        ),
        "base_cache_mutated": False,
        "locked_split_database_mutated": False,
        "assignment_unit": (
            "one representative row per "
            "paired VS2 float32 fingerprint"
        ),
        "feature_count": FEATURE_COUNT,
        "feature_dtype": "float32",
        "label_dtype": "int8",
        "raw_row_count_dtype": "int64",
        "hash_dtype": "int64",
        "classes": [
            "benign",
            "gafgyt",
            "mirai",
        ],
        "applied_move_count": (
            EXPECTED_MOVES
        ),
        "applied_moved_raw_row_count": (
            EXPECTED_MOVED_RAW_ROWS
        ),
        "training_sample_weight": None,
        "primary_evaluation_view": (
            "fingerprint_level"
        ),
        "secondary_evaluation_view": (
            "raw-record weighted by "
            "raw_row_count"
        ),
        "provenance": {
            "base_protocol": str(
                BASE_PROTOCOL
            ),
            "base_protocol_sha256": (
                sha256_file(
                    BASE_PROTOCOL
                )
            ),
            "addendum": str(
                ADDENDUM
            ),
            "addendum_sha256": (
                sha256_file(
                    ADDENDUM
                )
            ),
            "repair_plan": str(
                REPAIR_PLAN
            ),
            "repair_plan_sha256": (
                sha256_file(
                    REPAIR_PLAN
                )
            ),
            "components_csv": str(
                COMPONENTS_CSV
            ),
            "components_csv_sha256": (
                sha256_file(
                    COMPONENTS_CSV
                )
            ),
            "moves_csv": str(
                MOVES_CSV
            ),
            "moves_csv_sha256": (
                sha256_file(
                    MOVES_CSV
                )
            ),
            "builder_script": str(
                Path(__file__).resolve()
            ),
            "builder_script_sha256": (
                sha256_file(
                    Path(__file__).resolve()
                )
            ),
        },
        "input_validation": (
            input_validation["checks"]
        ),
        "materialization": (
            materialization
        ),
        "unscaled_validation": (
            unscaled_validation
        ),
        "unscaled_overlap_audit": (
            unscaled_overlap_audit
        ),
        "scaler_audit": (
            scaler_audit
        ),
        "scaled_overlap_audit": (
            scaled_overlap_audit
        ),
        "artifacts": artifacts,
        "free_disk_before_build_gib": (
            free_disk_gib
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        DERIVED_BUILD_CACHE
        / "manifest.json",
        manifest,
    )

    (
        DERIVED_BUILD_CACHE
        / "build_status.json"
    ).unlink()

    DERIVED_BUILD_CACHE.rename(
        DERIVED_FINAL_CACHE
    )

    cache_size_bytes = sum(
        path.stat().st_size
        for path in DERIVED_FINAL_CACHE.iterdir()
        if path.is_file()
    )

    report = {
        "status": "completed",
        "generated_at_utc": (
            completed_at
        ),
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "repair_iteration": 1,
        "derived_cache": str(
            DERIVED_FINAL_CACHE
        ),
        "manifest": str(
            DERIVED_FINAL_CACHE
            / "manifest.json"
        ),
        "manifest_sha256": (
            sha256_file(
                DERIVED_FINAL_CACHE
                / "manifest.json"
            )
        ),
        "applied_move_count": (
            EXPECTED_MOVES
        ),
        "fingerprint_count": (
            EXPECTED_TOTAL_FINGERPRINTS
        ),
        "represented_raw_row_count": (
            EXPECTED_TOTAL_RAW_ROWS
        ),
        "scaled_cross_split_overlap_count": (
            scaled_overlap_audit[
                "cross_split_model_input_fingerprint_count"
            ]
        ),
        "scaled_global_family_conflict_count": (
            scaled_overlap_audit[
                "global_family_conflict_count"
            ]
        ),
        "cache_size_bytes": (
            cache_size_bytes
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "all_checks_passed": True,
    }

    atomic_json(
        SUCCESS_REPORT,
        report,
    )

    print()
    print("=" * 92)
    print(
        "DERIVED TABULAR CACHE REPAIR SUMMARY"
    )
    print("=" * 92)
    print(
        "Applied fingerprint moves      : "
        f"{EXPECTED_MOVES:,}"
    )
    print(
        "Moved represented raw rows     : "
        f"{EXPECTED_MOVED_RAW_ROWS:,}"
    )
    print(
        "Fingerprint rows               : "
        f"{EXPECTED_TOTAL_FINGERPRINTS:,}"
    )
    print(
        "Represented raw rows           : "
        f"{EXPECTED_TOTAL_RAW_ROWS:,}"
    )
    print(
        "Original-hash cross-split      : "
        f"{unscaled_overlap_audit['cross_split_original_hash_overlap_count']:,}"
    )
    print(
        "Scaled cross-split overlap     : "
        f"{scaled_overlap_audit['cross_split_model_input_fingerprint_count']:,}"
    )
    print(
        "Scaled global family conflicts : "
        f"{scaled_overlap_audit['global_family_conflict_count']:,}"
    )
    print(
        "Derived cache size             : "
        f"{cache_size_bytes / (1024**3):.3f} GiB"
    )
    print(
        "Derived cache                  : "
        f"{DERIVED_FINAL_CACHE}"
    )
    print(
        "Manifest                       : "
        f"{DERIVED_FINAL_CACHE / 'manifest.json'}"
    )
    print(
        "Report                         : "
        f"{SUCCESS_REPORT}"
    )
    print(
        "Elapsed                        : "
        f"{(time.perf_counter() - started) / 60.0:.2f} minutes"
    )
    print(
        "All checks passed              : True"
    )
    print(
        "DERIVED TABULAR CACHE REPAIR COMPLETED"
    )


if __name__ == "__main__":
    main()
