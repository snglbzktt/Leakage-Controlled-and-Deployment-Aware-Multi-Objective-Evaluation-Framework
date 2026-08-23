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

SOURCE_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2.building"
)

WORK_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2_iteration2.building"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
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

ITERATION1_FAILURE = (
    SOURCE_CACHE
    / "scaled_overlap_failure_iteration1.json"
)

ITERATION1_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration1_summary_v3_2.json"
)

ITERATION2_PLAN = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_plan_v3_2.json"
)

ITERATION2_COMPONENTS = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_components_v3_2.csv"
)

ITERATION2_MOVES = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_moves_v3_2.csv"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_scaled_repair_iteration2_summary_v3_2.json"
)

PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"

FEATURE_COUNT = 115
CHUNK_SIZE = 100_000
SCALER_AUDIT_TOLERANCE = 1e-4
MINIMUM_FREE_DISK_GIB = 2.0

EXPECTED_TOTAL_FINGERPRINTS = 2_278_176
EXPECTED_TOTAL_RAW_ROWS = 7_062_606
EXPECTED_ITERATION1_MOVES = 353
EXPECTED_ITERATION2_COMPONENTS = 3
EXPECTED_ITERATION2_MOVES = 3
EXPECTED_ITERATION2_MOVED_RAW_ROWS = 3

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


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def load_moves() -> list[dict[str, Any]]:
    with ITERATION2_MOVES.open(
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


def inspect_source_cache() -> dict[str, Any]:
    split_records: dict[
        str,
        dict[str, Any],
    ] = {}

    total_fingerprints = 0
    total_raw_rows = 0

    for split_name in SPLITS:
        x = np.load(
            SOURCE_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            SOURCE_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            SOURCE_CACHE
            / f"raw_row_count_{split_name}.npy",
            mmap_mode="r",
        )

        forward = np.load(
            SOURCE_CACHE
            / f"hash_forward_{split_name}.npy",
            mmap_mode="r",
        )

        reverse = np.load(
            SOURCE_CACHE
            / f"hash_reverse_{split_name}.npy",
            mmap_mode="r",
        )

        count = int(len(x))

        shape_checks = {
            "X": (
                x.shape
                == (
                    count,
                    FEATURE_COUNT,
                )
                and x.dtype == np.float32
            ),
            "y": (
                y.shape == (count,)
                and y.dtype == np.int8
            ),
            "raw_row_count": (
                raw_counts.shape == (count,)
                and raw_counts.dtype
                == np.int64
            ),
            "hash_forward": (
                forward.shape == (count,)
                and forward.dtype
                == np.int64
            ),
            "hash_reverse": (
                reverse.shape == (count,)
                and reverse.dtype
                == np.int64
            ),
        }

        if not all(
            shape_checks.values()
        ):
            raise RuntimeError(
                "Invalid source cache arrays: "
                f"{split_name} -> {shape_checks}"
            )

        class_fingerprint_counts = np.bincount(
            np.asarray(
                y,
                dtype=np.int64,
            ),
            minlength=3,
        ).astype(np.int64)

        class_raw_counts = np.bincount(
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

        observed_raw_rows = int(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            ).sum()
        )

        split_records[
            split_name
        ] = {
            "fingerprint_count": count,
            "raw_row_count": (
                observed_raw_rows
            ),
            "class_fingerprint_counts": (
                class_fingerprint_counts
            ),
            "class_raw_row_counts": (
                class_raw_counts
            ),
        }

        total_fingerprints += count
        total_raw_rows += (
            observed_raw_rows
        )

    if (
        total_fingerprints
        != EXPECTED_TOTAL_FINGERPRINTS
    ):
        raise RuntimeError(
            "Source fingerprint total mismatch: "
            f"{total_fingerprints:,}"
        )

    if (
        total_raw_rows
        != EXPECTED_TOTAL_RAW_ROWS
    ):
        raise RuntimeError(
            "Source raw-row total mismatch: "
            f"{total_raw_rows:,}"
        )

    return {
        "splits": split_records,
        "total_fingerprint_count": (
            total_fingerprints
        ),
        "total_raw_row_count": (
            total_raw_rows
        ),
    }


def validate_inputs(
    moves: list[dict[str, Any]],
    source_inventory: dict[str, Any],
) -> dict[str, Any]:
    required_paths = [
        SOURCE_CACHE,
        ADDENDUM,
        ADDENDUM_MANIFEST,
        ITERATION1_FAILURE,
        ITERATION1_REPORT,
        ITERATION2_PLAN,
        ITERATION2_COMPONENTS,
        ITERATION2_MOVES,
    ]

    for split_name in SPLITS:
        required_paths.extend(
            [
                SOURCE_CACHE
                / f"X_{split_name}.npy",
                SOURCE_CACHE
                / f"y_{split_name}.npy",
                SOURCE_CACHE
                / f"raw_row_count_{split_name}.npy",
                SOURCE_CACHE
                / f"hash_forward_{split_name}.npy",
                SOURCE_CACHE
                / f"hash_reverse_{split_name}.npy",
            ]
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if WORK_CACHE.exists():
        raise FileExistsError(
            "Iteration-2 work cache already exists; "
            f"refusing to overwrite: {WORK_CACHE}"
        )

    if FINAL_CACHE.exists():
        raise FileExistsError(
            "Final cache already exists; "
            f"refusing to overwrite: {FINAL_CACHE}"
        )

    addendum = read_json(
        ADDENDUM
    )

    addendum_manifest = read_json(
        ADDENDUM_MANIFEST
    )

    iteration1_failure = read_json(
        ITERATION1_FAILURE
    )

    iteration1_report = read_json(
        ITERATION1_REPORT
    )

    plan = read_json(
        ITERATION2_PLAN
    )

    proposed_counts = {
        str(row["split"]): {
            "fingerprints": int(
                row[
                    "proposed_fingerprint_count"
                ]
            ),
            "raw_rows": int(
                row[
                    "proposed_raw_row_count"
                ]
            ),
        }
        for row in plan[
            "split_plan"
        ]
    }

    checks = {
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
        "iteration_policy_allows_repeat": (
            "repeat the same locked deterministic"
            in str(
                addendum[
                    "repair_policy"
                ][
                    "iteration_policy"
                ]
            ).lower()
        ),
        "iteration1_failure_has_three_components": (
            int(
                iteration1_failure[
                    "cross_split_model_input_fingerprint_count"
                ]
            )
            == EXPECTED_ITERATION2_COMPONENTS
        ),
        "iteration1_family_conflicts_zero": (
            int(
                iteration1_failure[
                    "global_family_conflict_count"
                ]
            )
            == 0
        ),
        "iteration1_training_blocked": (
            iteration1_report.get(
                "all_model_training_blocked"
            )
            is True
        ),
        "iteration2_plan_not_applied": (
            plan.get("status")
            == "plan_only_not_applied"
        ),
        "iteration2_plan_number_matches": (
            int(
                plan[
                    "repair_iteration"
                ]
            )
            == 2
        ),
        "iteration2_plan_checks_passed": (
            plan.get(
                "all_checks_passed"
            )
            is True
        ),
        "iteration2_component_count_matches": (
            int(
                plan[
                    "cross_split_scaled_component_count"
                ]
            )
            == EXPECTED_ITERATION2_COMPONENTS
        ),
        "iteration2_move_count_matches": (
            len(moves)
            == EXPECTED_ITERATION2_MOVES
            == int(
                plan[
                    "proposed_moved_fingerprint_count"
                ]
            )
        ),
        "iteration2_moved_raw_rows_match": (
            sum(
                int(row["raw_row_count"])
                for row in moves
            )
            == EXPECTED_ITERATION2_MOVED_RAW_ROWS
            == int(
                plan[
                    "proposed_moved_raw_row_count"
                ]
            )
        ),
        "moves_hash_matches_plan": (
            sha256_file(
                ITERATION2_MOVES
            )
            == sha256_file(
                Path(
                    plan[
                        "moves_csv"
                    ]
                )
            )
        ),
        "components_hash_matches_plan": (
            sha256_file(
                ITERATION2_COMPONENTS
            )
            == sha256_file(
                Path(
                    plan[
                        "components_csv"
                    ]
                )
            )
        ),
        "all_moves_cross_splits": all(
            int(
                row[
                    "source_split_code"
                ]
            )
            != int(
                row[
                    "destination_split_code"
                ]
            )
            for row in moves
        ),
        "all_moves_benign": all(
            int(row["family"]) == 0
            for row in moves
        ),
        "all_moves_have_unit_weight": all(
            int(row["raw_row_count"])
            == 1
            for row in moves
        ),
        "proposed_total_fingerprints_preserved": (
            sum(
                row["fingerprints"]
                for row
                in proposed_counts.values()
            )
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "proposed_total_raw_rows_preserved": (
            sum(
                row["raw_rows"]
                for row
                in proposed_counts.values()
            )
            == EXPECTED_TOTAL_RAW_ROWS
        ),
        "source_total_fingerprints_match": (
            source_inventory[
                "total_fingerprint_count"
            ]
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "source_total_raw_rows_match": (
            source_inventory[
                "total_raw_row_count"
            ]
            == EXPECTED_TOTAL_RAW_ROWS
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Iteration-2 input validation failed: "
            + ", ".join(failed)
        )

    return {
        "addendum": addendum,
        "addendum_manifest": (
            addendum_manifest
        ),
        "iteration1_failure": (
            iteration1_failure
        ),
        "iteration1_report": (
            iteration1_report
        ),
        "plan": plan,
        "proposed_counts": (
            proposed_counts
        ),
        "checks": checks,
    }


def load_payloads(
    moves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, int]
    ] = set()

    for row in moves:
        split_name = str(
            row["source_split"]
        )

        index = int(
            row["source_row_index"]
        )

        identity = (
            split_name,
            index,
        )

        if identity in seen:
            raise RuntimeError(
                "Duplicate move source: "
                f"{identity}"
            )

        seen.add(identity)

        x = np.load(
            SOURCE_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            SOURCE_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            SOURCE_CACHE
            / f"raw_row_count_{split_name}.npy",
            mmap_mode="r",
        )

        forward = np.load(
            SOURCE_CACHE
            / f"hash_forward_{split_name}.npy",
            mmap_mode="r",
        )

        reverse = np.load(
            SOURCE_CACHE
            / f"hash_reverse_{split_name}.npy",
            mmap_mode="r",
        )

        if (
            index < 0
            or index >= len(x)
        ):
            raise RuntimeError(
                "Move source index out of range: "
                f"{split_name}/{index}"
            )

        observed = {
            "family": int(
                y[index]
            ),
            "raw_row_count": int(
                raw_counts[index]
            ),
            "original_hash_forward": int(
                forward[index]
            ),
            "original_hash_reverse": int(
                reverse[index]
            ),
        }

        expected = {
            "family": int(
                row["family"]
            ),
            "raw_row_count": int(
                row["raw_row_count"]
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

        if observed != expected:
            raise RuntimeError(
                "Move source identity mismatch: "
                f"{split_name}/{index} | "
                f"observed={observed} | "
                f"expected={expected}"
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
            != observed[
                "original_hash_forward"
            ]
            or int(actual_reverse[0])
            != observed[
                "original_hash_reverse"
            ]
        ):
            raise RuntimeError(
                "Move source feature/hash mismatch: "
                f"{split_name}/{index}"
            )

        payloads.append(
            {
                **row,
                "features": features,
            }
        )

    return payloads


def create_arrays(
    split_name: str,
    row_count: int,
) -> dict[str, np.memmap]:
    return {
        "X": np.lib.format.open_memmap(
            WORK_CACHE
            / f"X_{split_name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                row_count,
                FEATURE_COUNT,
            ),
        ),
        "y": np.lib.format.open_memmap(
            WORK_CACHE
            / f"y_{split_name}.npy",
            mode="w+",
            dtype=np.int8,
            shape=(row_count,),
        ),
        "raw_row_count": (
            np.lib.format.open_memmap(
                WORK_CACHE
                / (
                    "raw_row_count_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(row_count,),
            )
        ),
        "hash_forward": (
            np.lib.format.open_memmap(
                WORK_CACHE
                / (
                    "hash_forward_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(row_count,),
            )
        ),
        "hash_reverse": (
            np.lib.format.open_memmap(
                WORK_CACHE
                / (
                    "hash_reverse_"
                    f"{split_name}.npy"
                ),
                mode="w+",
                dtype=np.int64,
                shape=(row_count,),
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
    amount = (
        source_end - source_start
    )

    if amount <= 0:
        return target_start

    target_end = (
        target_start + amount
    )

    for key in source:
        target[key][
            target_start:target_end
        ] = source[key][
            source_start:source_end
        ]

    return target_end


def materialize(
    payloads: list[dict[str, Any]],
    source_inventory: dict[str, Any],
    proposed_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    outgoing: dict[
        str,
        list[int],
    ] = {
        split_name: []
        for split_name in SPLITS
    }

    incoming: dict[
        str,
        list[dict[str, Any]],
    ] = {
        split_name: []
        for split_name in SPLITS
    }

    for payload in payloads:
        outgoing[
            str(payload["source_split"])
        ].append(
            int(
                payload[
                    "source_row_index"
                ]
            )
        )

        incoming[
            str(
                payload[
                    "destination_split"
                ]
            )
        ].append(payload)

    records: list[
        dict[str, Any]
    ] = []

    for split_name in SPLITS:
        outgoing_indices = sorted(
            outgoing[split_name]
        )

        incoming_rows = sorted(
            incoming[split_name],
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
                SOURCE_CACHE
                / f"X_{split_name}.npy",
                mmap_mode="r",
            ),
            "y": np.load(
                SOURCE_CACHE
                / f"y_{split_name}.npy",
                mmap_mode="r",
            ),
            "raw_row_count": np.load(
                SOURCE_CACHE
                / (
                    "raw_row_count_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
            "hash_forward": np.load(
                SOURCE_CACHE
                / (
                    "hash_forward_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
            "hash_reverse": np.load(
                SOURCE_CACHE
                / (
                    "hash_reverse_"
                    f"{split_name}.npy"
                ),
                mmap_mode="r",
            ),
        }

        source_count = int(
            source_inventory[
                "splits"
            ][
                split_name
            ][
                "fingerprint_count"
            ]
        )

        target_count = int(
            proposed_counts[
                split_name
            ][
                "fingerprints"
            ]
        )

        calculated_target = (
            source_count
            - len(outgoing_indices)
            + len(incoming_rows)
        )

        if (
            calculated_target
            != target_count
        ):
            raise RuntimeError(
                "Proposed split count mismatch: "
                f"{split_name} | "
                f"{calculated_target:,} != "
                f"{target_count:,}"
            )

        target = create_arrays(
            split_name,
            target_count,
        )

        source_position = 0
        target_position = 0

        for outgoing_index in (
            outgoing_indices
        ):
            if outgoing_index < source_position:
                raise RuntimeError(
                    "Outgoing indices are not "
                    "strictly increasing: "
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
            source_end=source_count,
            target_start=target_position,
        )

        for payload in incoming_rows:
            if (
                target_position
                >= target_count
            ):
                raise RuntimeError(
                    "Target split overflow: "
                    f"{split_name}"
                )

            target["X"][
                target_position
            ] = payload["features"]

            target["y"][
                target_position
            ] = int(
                payload["family"]
            )

            target["raw_row_count"][
                target_position
            ] = int(
                payload["raw_row_count"]
            )

            target["hash_forward"][
                target_position
            ] = int(
                payload[
                    "original_hash_forward"
                ]
            )

            target["hash_reverse"][
                target_position
            ] = int(
                payload[
                    "original_hash_reverse"
                ]
            )

            target_position += 1

        if (
            target_position
            != target_count
        ):
            raise RuntimeError(
                "Final target offset mismatch: "
                f"{split_name} | "
                f"{target_position:,} != "
                f"{target_count:,}"
            )

        for array in target.values():
            array.flush()

        records.append(
            {
                "split": split_name,
                "source_count": (
                    source_count
                ),
                "outgoing_count": (
                    len(outgoing_indices)
                ),
                "incoming_count": (
                    len(incoming_rows)
                ),
                "target_count": (
                    target_count
                ),
            }
        )

        print(
            "[iteration-2 materialization] "
            f"{split_name:<10} | "
            f"out={len(outgoing_indices):,} | "
            f"in={len(incoming_rows):,} | "
            f"rows={target_count:,}",
            flush=True,
        )

        del source
        del target

    return {
        "splits": records,
        "total_outgoing": sum(
            len(values)
            for values in outgoing.values()
        ),
        "total_incoming": sum(
            len(values)
            for values in incoming.values()
        ),
    }


def validate_unscaled(
    source_inventory: dict[str, Any],
    moves: list[dict[str, Any]],
    proposed_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    expected_class_fingerprints: dict[
        str,
        np.ndarray,
    ] = {
        split_name: np.array(
            source_inventory[
                "splits"
            ][
                split_name
            ][
                "class_fingerprint_counts"
            ],
            dtype=np.int64,
            copy=True,
        )
        for split_name in SPLITS
    }

    expected_class_raw_rows: dict[
        str,
        np.ndarray,
    ] = {
        split_name: np.array(
            source_inventory[
                "splits"
            ][
                split_name
            ][
                "class_raw_row_counts"
            ],
            dtype=np.int64,
            copy=True,
        )
        for split_name in SPLITS
    }

    for row in moves:
        source_split = str(
            row["source_split"]
        )

        destination_split = str(
            row["destination_split"]
        )

        family = int(
            row["family"]
        )

        raw_weight = int(
            row["raw_row_count"]
        )

        expected_class_fingerprints[
            source_split
        ][family] -= 1

        expected_class_fingerprints[
            destination_split
        ][family] += 1

        expected_class_raw_rows[
            source_split
        ][family] -= raw_weight

        expected_class_raw_rows[
            destination_split
        ][family] += raw_weight

    records: list[
        dict[str, Any]
    ] = []

    total_fingerprints = 0
    total_raw_rows = 0

    for split_name in SPLITS:
        expected_count = int(
            proposed_counts[
                split_name
            ][
                "fingerprints"
            ]
        )

        expected_raw_rows = int(
            proposed_counts[
                split_name
            ][
                "raw_rows"
            ]
        )

        x = np.load(
            WORK_CACHE
            / f"X_{split_name}.npy",
            mmap_mode="r",
        )

        y = np.load(
            WORK_CACHE
            / f"y_{split_name}.npy",
            mmap_mode="r",
        )

        raw_counts = np.load(
            WORK_CACHE
            / f"raw_row_count_{split_name}.npy",
            mmap_mode="r",
        )

        forward = np.load(
            WORK_CACHE
            / f"hash_forward_{split_name}.npy",
            mmap_mode="r",
        )

        reverse = np.load(
            WORK_CACHE
            / f"hash_reverse_{split_name}.npy",
            mmap_mode="r",
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
                y.shape == (expected_count,)
                and y.dtype == np.int8
            ),
            "raw_row_count": (
                raw_counts.shape
                == (expected_count,)
                and raw_counts.dtype
                == np.int64
            ),
            "hash_forward": (
                forward.shape
                == (expected_count,)
                and forward.dtype
                == np.int64
            ),
            "hash_reverse": (
                reverse.shape
                == (expected_count,)
                and reverse.dtype
                == np.int64
            ),
        }

        if not all(
            shape_checks.values()
        ):
            raise RuntimeError(
                "Iteration-2 shape/dtype "
                "validation failed: "
                f"{split_name} -> {shape_checks}"
            )

        observed_class_fingerprints = np.bincount(
            np.asarray(
                y,
                dtype=np.int64,
            ),
            minlength=3,
        ).astype(np.int64)

        observed_class_raw_rows = np.bincount(
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

        observed_raw_rows = int(
            np.asarray(
                raw_counts,
                dtype=np.int64,
            ).sum()
        )

        if not np.array_equal(
            observed_class_fingerprints,
            expected_class_fingerprints[
                split_name
            ],
        ):
            raise RuntimeError(
                "Iteration-2 class fingerprint "
                "counts mismatch: "
                f"{split_name}"
            )

        if not np.array_equal(
            observed_class_raw_rows,
            expected_class_raw_rows[
                split_name
            ],
        ):
            raise RuntimeError(
                "Iteration-2 class raw-row "
                "counts mismatch: "
                f"{split_name}"
            )

        if (
            observed_raw_rows
            != expected_raw_rows
        ):
            raise RuntimeError(
                "Iteration-2 split raw-row "
                "count mismatch: "
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
                "Non-positive raw-row weight: "
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
                        forward[
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
                        reverse[
                            start:end
                        ],
                        dtype=np.int64,
                    )
                )
            )

        if not finite:
            raise RuntimeError(
                "Non-finite iteration-2 cache "
                f"value: {split_name}"
            )

        if hash_mismatch_count != 0:
            raise RuntimeError(
                "Iteration-2 hash mismatch: "
                f"{split_name} -> "
                f"{hash_mismatch_count:,}"
            )

        records.append(
            {
                "split": split_name,
                "fingerprint_count": (
                    expected_count
                ),
                "raw_row_count": (
                    observed_raw_rows
                ),
                "class_fingerprint_counts": (
                    observed_class_fingerprints.tolist()
                ),
                "class_raw_row_counts": (
                    observed_class_raw_rows.tolist()
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
            "[iteration-2 unscaled validation] "
            f"{split_name:<10} | "
            f"rows={expected_count:,} | "
            f"raw_rows={observed_raw_rows:,}",
            flush=True,
        )

    checks = {
        "total_fingerprints_preserved": (
            total_fingerprints
            == EXPECTED_TOTAL_FINGERPRINTS
        ),
        "total_raw_rows_preserved": (
            total_raw_rows
            == EXPECTED_TOTAL_RAW_ROWS
        ),
        "all_values_finite": all(
            row["all_values_finite"]
            for row in records
        ),
        "all_hashes_match": all(
            row["hash_mismatch_count"]
            == 0
            for row in records
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Iteration-2 unscaled validation failed: "
            + ", ".join(failed)
        )

    return {
        "splits": records,
        "total_fingerprint_count": (
            total_fingerprints
        ),
        "total_raw_row_count": (
            total_raw_rows
        ),
        "checks": checks,
    }


def audit_original_hash_uniqueness() -> dict[str, Any]:
    database = (
        WORK_CACHE
        / "unscaled_hash_overlap_audit.sqlite"
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
                WORK_CACHE
                / f"hash_forward_{split_name}.npy",
                mmap_mode="r",
            )

            reverse = np.load(
                WORK_CACHE
                / f"hash_reverse_{split_name}.npy",
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
                        "Duplicate original hash found "
                        "after iteration-2 repair."
                    ) from error

                processed += (
                    end - start
                )

            connection.commit()

        stored = int(
            connection.execute(
                "SELECT COUNT(*) FROM hashes"
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
            "Iteration-2 original-hash audit failed: "
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
        WORK_CACHE / "X_train.npy",
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
            "Iteration-2 scaler shape mismatch."
        )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError(
            "Invalid iteration-2 scaler values."
        )

    np.save(
        WORK_CACHE / "scaler_mean.npy",
        mean,
    )

    np.save(
        WORK_CACHE / "scaler_scale.npy",
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
        - np.square(scaled_mean),
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
            "Iteration-2 scaler audit failed: "
            f"mean={maximum_absolute_mean:.3e}, "
            f"std_error={maximum_std_error:.3e}"
        )

    return {
        "fit_split": "train",
        "fit_sample_count": int(
            len(x_train)
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
        WORK_CACHE
        / "scaled_model_input_overlap_audit.sqlite"
    )

    mean = np.load(
        WORK_CACHE / "scaler_mean.npy"
    )

    scale = np.load(
        WORK_CACHE / "scaler_scale.npy"
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
                WORK_CACHE
                / f"X_{split_name}.npy",
                mmap_mode="r",
            )

            y = np.load(
                WORK_CACHE
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
                        "Non-finite final scaled input: "
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
                "[iteration-2 scaled audit] "
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
        "repair_iteration": 2,
        "feature_space": (
            "iteration-2 repaired train-only "
            "StandardScaler float64 parameters "
            "then final float32 model input"
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
            WORK_CACHE
            / (
                "scaled_overlap_failure_"
                "iteration2.json"
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
        for path in WORK_CACHE.iterdir()
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
        "APPLY SCALED-COLLISION REPAIR ITERATION 2"
    )
    print("=" * 92)
    print(
        f"Source cache : {SOURCE_CACHE}"
    )
    print(
        f"Work cache   : {WORK_CACHE}"
    )
    print(
        f"Final cache  : {FINAL_CACHE}"
    )

    source_inventory = (
        inspect_source_cache()
    )

    moves = load_moves()

    validation = validate_inputs(
        moves=moves,
        source_inventory=source_inventory,
    )

    free_disk_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        / (1024**3)
    )

    if (
        free_disk_gib
        < MINIMUM_FREE_DISK_GIB
    ):
        raise RuntimeError(
            "Insufficient free disk space: "
            f"{free_disk_gib:.3f} GiB < "
            f"{MINIMUM_FREE_DISK_GIB:.3f} GiB"
        )

    WORK_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_CACHE.mkdir()

    atomic_json(
        WORK_CACHE
        / "build_status.json",
        {
            "status": "building",
            "repair_iteration": 2,
            "started_at_utc": utc_now(),
            "source_cache": str(
                SOURCE_CACHE
            ),
            "iteration2_plan": str(
                ITERATION2_PLAN
            ),
            "iteration2_moves": str(
                ITERATION2_MOVES
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
    )

    print(
        "Loading and validating 3 moved rows...",
        flush=True,
    )

    payloads = load_payloads(
        moves
    )

    print(
        "Materializing iteration-2 derived cache...",
        flush=True,
    )

    materialization = materialize(
        payloads=payloads,
        source_inventory=source_inventory,
        proposed_counts=validation[
            "proposed_counts"
        ],
    )

    print(
        "Validating iteration-2 unscaled cache...",
        flush=True,
    )

    unscaled_validation = (
        validate_unscaled(
            source_inventory=source_inventory,
            moves=moves,
            proposed_counts=validation[
                "proposed_counts"
            ],
        )
    )

    print(
        "Auditing original float32 hash uniqueness...",
        flush=True,
    )

    original_hash_audit = (
        audit_original_hash_uniqueness()
    )

    print(
        "Refitting train-only scaler...",
        flush=True,
    )

    scaler_audit = (
        fit_and_audit_scaler()
    )

    print(
        "Auditing final scaled float32 model inputs...",
        flush=True,
    )

    scaled_audit = (
        audit_scaled_overlap()
    )

    if not scaled_audit["passed"]:
        failure_report = {
            "status": (
                "iteration_2_requires_"
                "additional_repair"
            ),
            "generated_at_utc": utc_now(),
            "protocol_version": (
                PROTOCOL_VERSION
            ),
            "repair_iteration": 2,
            "work_cache": str(
                WORK_CACHE
            ),
            "iteration2_move_count": (
                EXPECTED_ITERATION2_MOVES
            ),
            "cumulative_move_count": (
                EXPECTED_ITERATION1_MOVES
                + EXPECTED_ITERATION2_MOVES
            ),
            "scaled_overlap_audit": (
                scaled_audit
            ),
            "all_model_training_blocked": True,
        }

        atomic_json(
            OUTPUT_REPORT,
            failure_report,
        )

        print()
        print("=" * 92)
        print(
            "ITERATION-2 CACHE STILL HAS "
            "SCALED CROSS-SPLIT COLLISIONS"
        )
        print("=" * 92)
        print(
            "Cross-split overlap: "
            f"{scaled_audit['cross_split_model_input_fingerprint_count']:,}"
        )
        print(
            "Family conflicts   : "
            f"{scaled_audit['global_family_conflict_count']:,}"
        )
        print(
            "Work cache kept    : "
            f"{WORK_CACHE}"
        )
        print(
            "No model training was started."
        )

        raise RuntimeError(
            "Repair iteration 2 did not reach "
            "zero scaled cross-split overlap."
        )

    artifacts = artifact_inventory()

    completed_at = utc_now()

    manifest = {
        "status": "completed",
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "created_at_utc": (
            completed_at
        ),
        "repair_iterations_completed": 2,
        "iteration1_move_count": (
            EXPECTED_ITERATION1_MOVES
        ),
        "iteration2_move_count": (
            EXPECTED_ITERATION2_MOVES
        ),
        "cumulative_move_count": (
            EXPECTED_ITERATION1_MOVES
            + EXPECTED_ITERATION2_MOVES
        ),
        "cumulative_moved_raw_row_count": (
            EXPECTED_ITERATION1_MOVES
            + EXPECTED_ITERATION2_MOVED_RAW_ROWS
        ),
        "cache_role": (
            "final family_3 classical "
            "tabular baseline cache"
        ),
        "cache_directory": str(
            FINAL_CACHE
        ),
        "source_iteration1_cache": str(
            SOURCE_CACHE
        ),
        "source_iteration1_cache_mutated": False,
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
        "split_counts": (
            validation[
                "proposed_counts"
            ]
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
            "addendum": str(
                ADDENDUM
            ),
            "addendum_sha256": (
                sha256_file(
                    ADDENDUM
                )
            ),
            "iteration2_plan": str(
                ITERATION2_PLAN
            ),
            "iteration2_plan_sha256": (
                sha256_file(
                    ITERATION2_PLAN
                )
            ),
            "iteration2_components": str(
                ITERATION2_COMPONENTS
            ),
            "iteration2_components_sha256": (
                sha256_file(
                    ITERATION2_COMPONENTS
                )
            ),
            "iteration2_moves": str(
                ITERATION2_MOVES
            ),
            "iteration2_moves_sha256": (
                sha256_file(
                    ITERATION2_MOVES
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
            validation["checks"]
        ),
        "materialization": (
            materialization
        ),
        "unscaled_validation": (
            unscaled_validation
        ),
        "original_hash_audit": (
            original_hash_audit
        ),
        "scaler_audit": (
            scaler_audit
        ),
        "scaled_overlap_audit": (
            scaled_audit
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
        WORK_CACHE / "manifest.json",
        manifest,
    )

    (
        WORK_CACHE
        / "build_status.json"
    ).unlink()

    WORK_CACHE.rename(
        FINAL_CACHE
    )

    cache_size_bytes = sum(
        path.stat().st_size
        for path in FINAL_CACHE.iterdir()
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
        "repair_iterations_completed": 2,
        "iteration2_move_count": (
            EXPECTED_ITERATION2_MOVES
        ),
        "cumulative_move_count": (
            EXPECTED_ITERATION1_MOVES
            + EXPECTED_ITERATION2_MOVES
        ),
        "fingerprint_count": (
            EXPECTED_TOTAL_FINGERPRINTS
        ),
        "represented_raw_row_count": (
            EXPECTED_TOTAL_RAW_ROWS
        ),
        "final_split_counts": (
            validation[
                "proposed_counts"
            ]
        ),
        "scaled_cross_split_overlap_count": (
            scaled_audit[
                "cross_split_model_input_fingerprint_count"
            ]
        ),
        "scaled_global_family_conflict_count": (
            scaled_audit[
                "global_family_conflict_count"
            ]
        ),
        "final_cache": str(
            FINAL_CACHE
        ),
        "manifest": str(
            FINAL_CACHE
            / "manifest.json"
        ),
        "manifest_sha256": (
            sha256_file(
                FINAL_CACHE
                / "manifest.json"
            )
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
        OUTPUT_REPORT,
        report,
    )

    print()
    print("=" * 92)
    print(
        "FINAL TABULAR CACHE REPAIR SUMMARY"
    )
    print("=" * 92)
    print(
        "Iteration-2 fingerprint moves  : "
        f"{EXPECTED_ITERATION2_MOVES:,}"
    )
    print(
        "Cumulative fingerprint moves   : "
        f"{EXPECTED_ITERATION1_MOVES + EXPECTED_ITERATION2_MOVES:,}"
    )
    print(
        "Fingerprint rows               : "
        f"{EXPECTED_TOTAL_FINGERPRINTS:,}"
    )
    print(
        "Represented raw rows           : "
        f"{EXPECTED_TOTAL_RAW_ROWS:,}"
    )

    for split_name in SPLITS:
        counts = validation[
            "proposed_counts"
        ][split_name]

        print(
            f"{split_name:<10} fingerprints       : "
            f"{counts['fingerprints']:,}"
        )

    print(
        "Original-hash cross-split      : "
        f"{original_hash_audit['cross_split_original_hash_overlap_count']:,}"
    )
    print(
        "Scaled cross-split overlap     : "
        f"{scaled_audit['cross_split_model_input_fingerprint_count']:,}"
    )
    print(
        "Scaled global family conflicts : "
        f"{scaled_audit['global_family_conflict_count']:,}"
    )
    print(
        "Final cache size               : "
        f"{cache_size_bytes / (1024**3):.3f} GiB"
    )
    print(
        "Final cache                    : "
        f"{FINAL_CACHE}"
    )
    print(
        "Manifest                       : "
        f"{FINAL_CACHE / 'manifest.json'}"
    )
    print(
        "Report                         : "
        f"{OUTPUT_REPORT}"
    )
    print(
        "Elapsed                        : "
        f"{(time.perf_counter() - started) / 60.0:.2f} minutes"
    )
    print(
        "All checks passed              : True"
    )
    print(
        "FINAL TABULAR CACHE REPAIR COMPLETED"
    )


if __name__ == "__main__":
    main()
