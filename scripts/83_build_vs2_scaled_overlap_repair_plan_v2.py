"""Build a deterministic repair plan for VS2 post-scaling overlap.

Diagnostic/planning only:
- reads the locked Stage-37 assignment database and grouped cache,
- maps the 371 final float32 collisions back to original fingerprint groups,
- solves an exact split-count-preserving reassignment,
- writes a repair plan,
- does not modify any database, cache, manifest, or training result.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "vs2_float32_leakage_ablation_seed2026.sqlite"
)
CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vs2_float32_leakage_ablation_cache_v2"
    / "grouped_float32"
)
AUDIT_DATABASE = CACHE_DIRECTORY / "scaled_model_input_overlap_audit.sqlite"
STAGE38_SCRIPT = PROJECT_ROOT / "scripts" / "38_run_b0_matched_leakage_ablation.py"
CSV_ROOT = PROJECT_ROOT / "data" / "interim" / "nbaiot_unpacked"

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "audit"
PLAN_CSV = OUTPUT_DIRECTORY / "vs2_scaled_overlap_repair_plan_v2.csv"
COMPONENT_CSV = OUTPUT_DIRECTORY / "vs2_scaled_overlap_repair_components_v2.csv"
SUMMARY_JSON = OUTPUT_DIRECTORY / "vs2_scaled_overlap_repair_plan_summary_v2.json"
REPORT_TXT = OUTPUT_DIRECTORY / "vs2_scaled_overlap_repair_plan_report_v2.txt"

SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("benign", "gafgyt", "mirai")
EXPECTED_SCALED_COMPONENT_COUNT = 371
EXPECTED_AFFECTED_RAW_ROWS = 1036
CHUNK_SIZE = 100_000
INF = np.int32(1_000_000_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
    temporary.replace(path)


def load_stage38_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage38_repair_runtime", STAGE38_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 38: {STAGE38_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_combo(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    reverse_rotated = np.left_shift(reverse, np.uint64(1)) | np.right_shift(
        reverse, np.uint64(63)
    )
    return np.bitwise_xor(forward, reverse_rotated)


def split_name(code: int) -> str:
    return SPLITS[int(code)]


def load_target_scaled_pairs() -> set[tuple[int, int]]:
    with sqlite3.connect(AUDIT_DATABASE) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Audit database quick_check failed: {quick_check}")

        rows = connection.execute(
            """
            SELECT hash_forward, hash_reverse
            FROM scaled_fingerprints
            GROUP BY hash_forward, hash_reverse
            HAVING COUNT(*) > 1
            ORDER BY hash_forward, hash_reverse
            """
        ).fetchall()

    pairs = {(int(row[0]), int(row[1])) for row in rows}
    if len(pairs) != EXPECTED_SCALED_COMPONENT_COUNT:
        raise RuntimeError(
            f"Unexpected scaled collision count: "
            f"{len(pairs)} != {EXPECTED_SCALED_COMPONENT_COUNT}"
        )
    return pairs


def file_assignment_arrays(
    connection: sqlite3.Connection,
    file_id: int,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = connection.execute(
        """
        SELECT row_number, hash_forward, hash_reverse, family_label
        FROM assignments
        WHERE file_id = ?
        ORDER BY row_number
        """,
        (file_id,),
    ).fetchall()

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Assignment row count mismatch for file {file_id}: "
            f"{len(rows)} != {expected_rows}"
        )

    row_numbers = np.fromiter((int(row[0]) for row in rows), dtype=np.int64)
    expected = np.arange(expected_rows, dtype=np.int64)
    if not np.array_equal(row_numbers, expected):
        raise RuntimeError(
            f"Assignment row numbers are not contiguous for file {file_id}"
        )

    hash_forward = np.fromiter((int(row[1]) for row in rows), dtype=np.int64)
    hash_reverse = np.fromiter((int(row[2]) for row in rows), dtype=np.int64)
    family_label = np.fromiter((int(row[3]) for row in rows), dtype=np.int8)
    return hash_forward, hash_reverse, family_label


def exact_count_preserving_assignment(
    components: list[dict[str, Any]],
    target_train_rows: int,
    target_validation_rows: int,
) -> tuple[list[int], int]:
    """Assign each scaled component to one split with exact row-count preservation.

    The objective is the minimum number of moved raw rows. Ties are resolved
    deterministically by trying train, then validation, then test.
    """

    train_limit = int(target_train_rows)
    validation_limit = int(target_validation_rows)

    cost = np.full((train_limit + 1, validation_limit + 1), INF, dtype=np.int32)
    cost[0, 0] = 0

    choices = np.full(
        (len(components), train_limit + 1, validation_limit + 1),
        255,
        dtype=np.uint8,
    )

    for component_index, component in enumerate(components):
        weight = int(component["raw_row_count"])
        current_rows = component["current_rows_by_split"]
        move_costs = [
            weight - int(current_rows[0]),
            weight - int(current_rows[1]),
            weight - int(current_rows[2]),
        ]

        new_cost = np.full_like(cost, INF)
        chosen = np.full(cost.shape, 255, dtype=np.uint8)

        # Train assignment.
        if weight <= train_limit:
            candidate = cost[: train_limit + 1 - weight, :] + np.int32(move_costs[0])
            destination = new_cost[weight:, :]
            mask = candidate < destination
            destination[mask] = candidate[mask]
            chosen_destination = chosen[weight:, :]
            chosen_destination[mask] = 0

        # Validation assignment.
        if weight <= validation_limit:
            candidate = cost[:, : validation_limit + 1 - weight] + np.int32(
                move_costs[1]
            )
            destination = new_cost[:, weight:]
            mask = candidate < destination
            destination[mask] = candidate[mask]
            chosen_destination = chosen[:, weight:]
            chosen_destination[mask] = 1

        # Test assignment.
        candidate = cost + np.int32(move_costs[2])
        mask = candidate < new_cost
        new_cost[mask] = candidate[mask]
        chosen[mask] = 2

        if int(new_cost.min()) >= int(INF):
            raise RuntimeError(
                f"No reachable DP state after component {component_index + 1}"
            )

        choices[component_index] = chosen
        cost = new_cost

    final_cost = int(cost[train_limit, validation_limit])
    if final_cost >= int(INF):
        raise RuntimeError(
            "No exact split-count-preserving repair assignment exists."
        )

    assignments = [0] * len(components)
    train_rows = train_limit
    validation_rows = validation_limit

    for component_index in range(len(components) - 1, -1, -1):
        choice = int(choices[component_index, train_rows, validation_rows])
        if choice not in (0, 1, 2):
            raise RuntimeError(
                f"Invalid DP backtracking choice at component {component_index}: {choice}"
            )

        assignments[component_index] = choice
        weight = int(components[component_index]["raw_row_count"])

        if choice == 0:
            train_rows -= weight
        elif choice == 1:
            validation_rows -= weight

        if train_rows < 0 or validation_rows < 0:
            raise RuntimeError("DP backtracking produced a negative state.")

    if train_rows != 0 or validation_rows != 0:
        raise RuntimeError(
            f"DP backtracking did not return to origin: "
            f"train={train_rows}, validation={validation_rows}"
        )

    return assignments, final_cost


def main() -> None:
    print("=" * 88)
    print("VS2 SCALED OVERLAP REPAIR PLAN")
    print("=" * 88)

    for required in (
        DATABASE,
        CACHE_DIRECTORY / "X_train.npy",
        CACHE_DIRECTORY / "X_validation.npy",
        CACHE_DIRECTORY / "X_test.npy",
        CACHE_DIRECTORY / "y_train.npy",
        CACHE_DIRECTORY / "y_validation.npy",
        CACHE_DIRECTORY / "y_test.npy",
        CACHE_DIRECTORY / "mean.npy",
        CACHE_DIRECTORY / "std.npy",
        AUDIT_DATABASE,
        STAGE38_SCRIPT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stage38 = load_stage38_module()
    target_pairs = load_target_scaled_pairs()

    target_forward = np.asarray(
        [pair[0] for pair in target_pairs], dtype=np.int64
    ).view(np.uint64)
    target_reverse = np.asarray(
        [pair[1] for pair in target_pairs], dtype=np.int64
    ).view(np.uint64)
    target_combos = np.unique(pair_combo(target_forward, target_reverse))

    mean = np.load(CACHE_DIRECTORY / "mean.npy")
    scale = np.load(CACHE_DIRECTORY / "std.npy")

    x_arrays = {
        split: np.load(CACHE_DIRECTORY / f"X_{split}.npy", mmap_mode="r")
        for split in SPLITS
    }
    y_arrays = {
        split: np.load(CACHE_DIRECTORY / f"y_{split}.npy", mmap_mode="r")
        for split in SPLITS
    }
    cursors = {split: 0 for split in SPLITS}

    source_database = stage38.source_database_path(DATABASE)
    files = stage38.discover_files(source_database, CSV_ROOT)

    # scaled_pair -> original_pair -> record
    component_groups: dict[
        tuple[int, int],
        dict[tuple[int, int], dict[str, Any]],
    ] = defaultdict(dict)

    matched_rows = 0

    with sqlite3.connect(DATABASE) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Assignment database quick_check failed: {quick_check}")

        for file_number, (file_id, csv_path, _raw_label, expected_rows) in enumerate(
            files, start=1
        ):
            print(
                f"[mapping] file {file_number}/"
                f"{len(files)}: {Path(csv_path).name}",
                flush=True,
            )

            original_forward, original_reverse, family_labels = file_assignment_arrays(
                connection=connection,
                file_id=int(file_id),
                expected_rows=int(expected_rows),
            )

            for split_code, split in enumerate(SPLITS):
                selected = np.asarray(
                    stage38.selected_rows(DATABASE, int(file_id), "grouped_float32", split),
                    dtype=np.int64,
                )
                count = len(selected)
                start_cursor = cursors[split]
                stop_cursor = start_cursor + count

                if stop_cursor > len(x_arrays[split]):
                    raise RuntimeError(
                        f"Cache cursor exceeded {split}: "
                        f"{stop_cursor} > {len(x_arrays[split])}"
                    )

                for local_start in range(0, count, CHUNK_SIZE):
                    local_stop = min(local_start + CHUNK_SIZE, count)

                    raw = np.asarray(
                        x_arrays[split][
                            start_cursor + local_start : start_cursor + local_stop
                        ],
                        dtype=np.float64,
                    )
                    scaled = ((raw - mean) / scale).astype(np.float32)
                    forward_u, reverse_u = stage38.float32_fingerprints(scaled)
                    combos = pair_combo(forward_u, reverse_u)
                    candidate_indices = np.flatnonzero(np.isin(combos, target_combos))

                    labels = np.asarray(
                        y_arrays[split][
                            start_cursor + local_start : start_cursor + local_stop
                        ],
                        dtype=np.int64,
                    )

                    for local_index in candidate_indices.tolist():
                        scaled_pair = (
                            int(forward_u[local_index].view(np.int64)),
                            int(reverse_u[local_index].view(np.int64)),
                        )
                        if scaled_pair not in target_pairs:
                            continue

                        selected_position = local_start + local_index
                        row_number = int(selected[selected_position])

                        original_pair = (
                            int(original_forward[row_number]),
                            int(original_reverse[row_number]),
                        )
                        family_label = int(family_labels[row_number])
                        observed_label = int(labels[local_index])

                        if observed_label != family_label:
                            raise RuntimeError(
                                "Cache/assignment family label mismatch: "
                                f"file={file_id}, row={row_number}, "
                                f"cache={observed_label}, assignment={family_label}"
                            )

                        record = component_groups[scaled_pair].get(original_pair)
                        if record is None:
                            record = {
                                "family_label": family_label,
                                "rows_by_split": [0, 0, 0],
                                "file_ids": set(),
                            }
                            component_groups[scaled_pair][original_pair] = record

                        if int(record["family_label"]) != family_label:
                            raise RuntimeError(
                                f"Original fingerprint label conflict: {original_pair}"
                            )

                        record["rows_by_split"][split_code] += 1
                        record["file_ids"].add(int(file_id))
                        matched_rows += 1

                cursors[split] = stop_cursor

    for split in SPLITS:
        if cursors[split] != len(x_arrays[split]):
            raise RuntimeError(
                f"Cache cursor mismatch for {split}: "
                f"{cursors[split]} != {len(x_arrays[split])}"
            )

    if set(component_groups) != target_pairs:
        missing = target_pairs - set(component_groups)
        extra = set(component_groups) - target_pairs
        raise RuntimeError(
            f"Scaled component mapping mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    components: list[dict[str, Any]] = []
    original_group_count = 0
    old_rows_by_split = [0, 0, 0]

    for scaled_pair in sorted(component_groups):
        groups = component_groups[scaled_pair]
        current_rows = [0, 0, 0]
        group_records: list[dict[str, Any]] = []

        for original_pair in sorted(groups):
            record = groups[original_pair]
            rows_by_split = [int(value) for value in record["rows_by_split"]]
            nonzero_splits = [code for code, value in enumerate(rows_by_split) if value > 0]

            if len(nonzero_splits) != 1:
                raise RuntimeError(
                    f"Original fingerprint appears in multiple grouped splits: "
                    f"{original_pair} -> {rows_by_split}"
                )

            old_split = int(nonzero_splits[0])
            raw_row_count = int(sum(rows_by_split))
            family_label = int(record["family_label"])

            if family_label != 0:
                raise RuntimeError(
                    f"Expected only benign affected groups, got label={family_label}"
                )

            current_rows[old_split] += raw_row_count
            old_rows_by_split[old_split] += raw_row_count
            original_group_count += 1

            group_records.append(
                {
                    "original_hash_forward": int(original_pair[0]),
                    "original_hash_reverse": int(original_pair[1]),
                    "family_label": family_label,
                    "raw_row_count": raw_row_count,
                    "old_grouped_split": old_split,
                    "file_ids": sorted(int(value) for value in record["file_ids"]),
                }
            )

        if sum(value > 0 for value in current_rows) <= 1:
            raise RuntimeError(
                f"Scaled collision component is not cross-split: {scaled_pair}"
            )

        components.append(
            {
                "scaled_hash_forward": int(scaled_pair[0]),
                "scaled_hash_reverse": int(scaled_pair[1]),
                "raw_row_count": int(sum(current_rows)),
                "current_rows_by_split": current_rows,
                "groups": group_records,
            }
        )

    if len(components) != EXPECTED_SCALED_COMPONENT_COUNT:
        raise RuntimeError(
            f"Unexpected component count: "
            f"{len(components)} != {EXPECTED_SCALED_COMPONENT_COUNT}"
        )

    if matched_rows != EXPECTED_AFFECTED_RAW_ROWS:
        raise RuntimeError(
            f"Unexpected affected row count: "
            f"{matched_rows} != {EXPECTED_AFFECTED_RAW_ROWS}"
        )

    assignments, minimum_moved_rows = exact_count_preserving_assignment(
        components=components,
        target_train_rows=old_rows_by_split[0],
        target_validation_rows=old_rows_by_split[1],
    )

    new_rows_by_split = [0, 0, 0]
    component_count_by_new_split = [0, 0, 0]
    moved_group_count = 0
    moved_rows_check = 0
    plan_rows: list[dict[str, Any]] = []

    for component, new_split in zip(components, assignments):
        component["new_grouped_split"] = int(new_split)
        new_rows_by_split[new_split] += int(component["raw_row_count"])
        component_count_by_new_split[new_split] += 1

        for group in component["groups"]:
            old_split = int(group["old_grouped_split"])
            raw_row_count = int(group["raw_row_count"])
            moved = old_split != new_split
            moved_rows = raw_row_count if moved else 0

            moved_group_count += int(moved)
            moved_rows_check += moved_rows

            plan_rows.append(
                {
                    "scaled_hash_forward": component["scaled_hash_forward"],
                    "scaled_hash_reverse": component["scaled_hash_reverse"],
                    "original_hash_forward": group["original_hash_forward"],
                    "original_hash_reverse": group["original_hash_reverse"],
                    "family_label": group["family_label"],
                    "class_name": CLASS_NAMES[int(group["family_label"])],
                    "raw_row_count": raw_row_count,
                    "old_grouped_split_code": old_split,
                    "old_grouped_split": split_name(old_split),
                    "new_grouped_split_code": new_split,
                    "new_grouped_split": split_name(new_split),
                    "rows_moved": moved_rows,
                    "file_ids": "|".join(str(value) for value in group["file_ids"]),
                }
            )

    if new_rows_by_split != old_rows_by_split:
        raise RuntimeError(
            f"Repaired affected split totals differ: "
            f"old={old_rows_by_split}, new={new_rows_by_split}"
        )

    if moved_rows_check != minimum_moved_rows:
        raise RuntimeError(
            f"Moved-row objective mismatch: "
            f"{moved_rows_check} != {minimum_moved_rows}"
        )

    with PLAN_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scaled_hash_forward",
            "scaled_hash_reverse",
            "original_hash_forward",
            "original_hash_reverse",
            "family_label",
            "class_name",
            "raw_row_count",
            "old_grouped_split_code",
            "old_grouped_split",
            "new_grouped_split_code",
            "new_grouped_split",
            "rows_moved",
            "file_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plan_rows)

    with COMPONENT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scaled_hash_forward",
            "scaled_hash_reverse",
            "original_group_count",
            "raw_row_count",
            "old_train_rows",
            "old_validation_rows",
            "old_test_rows",
            "new_grouped_split_code",
            "new_grouped_split",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for component in components:
            writer.writerow(
                {
                    "scaled_hash_forward": component["scaled_hash_forward"],
                    "scaled_hash_reverse": component["scaled_hash_reverse"],
                    "original_group_count": len(component["groups"]),
                    "raw_row_count": component["raw_row_count"],
                    "old_train_rows": component["current_rows_by_split"][0],
                    "old_validation_rows": component["current_rows_by_split"][1],
                    "old_test_rows": component["current_rows_by_split"][2],
                    "new_grouped_split_code": component["new_grouped_split"],
                    "new_grouped_split": split_name(component["new_grouped_split"]),
                }
            )

    summary = {
        "status": "completed",
        "analysis_role": "repair_plan_only_no_input_artifact_mutation",
        "repair_strategy": (
            "Merge original float32 fingerprint groups that collapse to the same "
            "StandardScaler-plus-float32 model input, assign each merged component "
            "to one split, preserve affected train/validation/test raw-row totals "
            "exactly, and minimize moved raw rows."
        ),
        "scaled_collision_component_count": len(components),
        "affected_original_fingerprint_group_count": original_group_count,
        "affected_raw_row_count": matched_rows,
        "affected_class_names": ["benign"],
        "old_affected_rows_by_split": {
            split_name(code): int(old_rows_by_split[code]) for code in range(3)
        },
        "new_affected_rows_by_split": {
            split_name(code): int(new_rows_by_split[code]) for code in range(3)
        },
        "component_count_by_new_split": {
            split_name(code): int(component_count_by_new_split[code])
            for code in range(3)
        },
        "minimum_moved_raw_rows": int(minimum_moved_rows),
        "moved_original_fingerprint_group_count": int(moved_group_count),
        "planned_cross_split_scaled_overlap_count": 0,
        "validation_checks": {
            "database_quick_check_ok": True,
            "exactly_371_scaled_components": len(components)
            == EXPECTED_SCALED_COMPONENT_COUNT,
            "exactly_1036_affected_raw_rows": matched_rows
            == EXPECTED_AFFECTED_RAW_ROWS,
            "only_benign_affected": True,
            "each_original_group_currently_in_one_split": True,
            "affected_split_totals_preserved_exactly": new_rows_by_split
            == old_rows_by_split,
            "minimum_move_objective_consistent": moved_rows_check
            == minimum_moved_rows,
            "planned_zero_scaled_overlap": True,
        },
        "source_sha256": {
            "assignment_database": sha256(DATABASE),
            "scaled_overlap_audit_database": sha256(AUDIT_DATABASE),
            "stage38_script": sha256(STAGE38_SCRIPT),
        },
        "outputs": {
            "repair_plan_csv": str(PLAN_CSV),
            "component_csv": str(COMPONENT_CSV),
            "summary_json": str(SUMMARY_JSON),
            "report_txt": str(REPORT_TXT),
        },
    }
    atomic_json(SUMMARY_JSON, summary)

    report_lines = [
        "=" * 88,
        "VS2 SCALED OVERLAP REPAIR PLAN",
        "=" * 88,
        f"Scaled collision components          : {len(components):,}",
        f"Affected original fingerprint groups : {original_group_count:,}",
        f"Affected raw rows                    : {matched_rows:,}",
        f"Minimum moved raw rows               : {minimum_moved_rows:,}",
        f"Moved original fingerprint groups    : {moved_group_count:,}",
        "",
        "AFFECTED SPLIT TOTALS (OLD -> PLANNED)",
    ]

    for code, split in enumerate(SPLITS):
        report_lines.append(
            f"{split}: {old_rows_by_split[code]:,} -> {new_rows_by_split[code]:,}"
        )

    report_lines.extend(
        [
            "",
            "VALIDATION CHECKS",
        ]
    )
    for key, value in summary["validation_checks"].items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend(
        [
            "",
            f"Plan       : {PLAN_CSV}",
            f"Components : {COMPONENT_CSV}",
            f"Summary    : {SUMMARY_JSON}",
            "",
            "VS2 SCALED OVERLAP REPAIR PLAN COMPLETED",
        ]
    )

    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
