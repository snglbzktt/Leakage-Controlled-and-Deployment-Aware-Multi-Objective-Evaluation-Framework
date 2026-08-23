"""Build a deterministic repair plan for residual VS2 post-scaling overlap.

This is a planning-only step for the repaired protocol v2.2 cache. It:
- reads the repaired split database and incomplete grouped cache,
- discovers the current residual cross-split final-float32 collisions,
- maps them back to original fingerprint groups,
- checks class consistency,
- attempts an exact per-family split-count-preserving reassignment,
- minimizes moved raw rows,
- writes audit artifacts,
- does not modify any database, cache, manifest, or training output.
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
    / "vs2_float32_leakage_ablation_scaled_repaired_seed2026_v2.sqlite"
)
CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vs2_float32_leakage_ablation_repaired_cache_v2"
    / "grouped_float32"
)
AUDIT_DATABASE = CACHE_DIRECTORY / "scaled_model_input_overlap_audit.sqlite"
RUNNER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "85_run_b0_matched_leakage_ablation_repaired_v2.py"
)
CSV_ROOT = PROJECT_ROOT / "data" / "interim" / "nbaiot_unpacked"

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "audit"
PLAN_CSV = OUTPUT_DIRECTORY / "vs2_residual_overlap_repair_plan_v3.csv"
COMPONENT_CSV = OUTPUT_DIRECTORY / "vs2_residual_overlap_components_v3.csv"
SUMMARY_JSON = OUTPUT_DIRECTORY / "vs2_residual_overlap_repair_plan_summary_v3.json"
REPORT_TXT = OUTPUT_DIRECTORY / "vs2_residual_overlap_repair_plan_report_v3.txt"

SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("benign", "gafgyt", "mirai")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
    temporary.replace(path)


def load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage85_runtime", RUNNER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner: {RUNNER_SCRIPT}")
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
    if not np.array_equal(row_numbers, np.arange(expected_rows, dtype=np.int64)):
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

        if weight <= train_limit:
            candidate = cost[: train_limit + 1 - weight, :] + np.int32(move_costs[0])
            destination = new_cost[weight:, :]
            mask = candidate < destination
            destination[mask] = candidate[mask]
            chosen_destination = chosen[weight:, :]
            chosen_destination[mask] = 0

        if weight <= validation_limit:
            candidate = cost[:, : validation_limit + 1 - weight] + np.int32(
                move_costs[1]
            )
            destination = new_cost[:, weight:]
            mask = candidate < destination
            destination[mask] = candidate[mask]
            chosen_destination = chosen[:, weight:]
            chosen_destination[mask] = 1

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
            "No exact affected-row split-count-preserving assignment exists."
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

    if train_rows != 0 or validation_rows != 0:
        raise RuntimeError(
            f"DP backtracking did not return to origin: "
            f"train={train_rows}, validation={validation_rows}"
        )

    return assignments, final_cost


def main() -> None:
    print("=" * 88)
    print("VS2 RESIDUAL SCALED OVERLAP REPAIR PLAN")
    print("=" * 88)

    for required in (
        DATABASE,
        AUDIT_DATABASE,
        RUNNER_SCRIPT,
        CACHE_DIRECTORY / "X_train.npy",
        CACHE_DIRECTORY / "X_validation.npy",
        CACHE_DIRECTORY / "X_test.npy",
        CACHE_DIRECTORY / "y_train.npy",
        CACHE_DIRECTORY / "y_validation.npy",
        CACHE_DIRECTORY / "y_test.npy",
        CACHE_DIRECTORY / "mean.npy",
        CACHE_DIRECTORY / "std.npy",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runner = load_runner_module()

    with sqlite3.connect(AUDIT_DATABASE) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Audit database quick_check failed: {quick_check}")

        overlap_rows = connection.execute(
            """
            SELECT hash_forward, hash_reverse
            FROM scaled_fingerprints
            GROUP BY hash_forward, hash_reverse
            HAVING COUNT(*) > 1
            ORDER BY hash_forward, hash_reverse
            """
        ).fetchall()

    target_pairs = {(int(row[0]), int(row[1])) for row in overlap_rows}
    overlap_count = len(target_pairs)
    print(f"Residual scaled collision components: {overlap_count:,}")

    if overlap_count == 0:
        summary = {
            "status": "completed_no_repair_needed",
            "residual_scaled_collision_component_count": 0,
            "planned_cross_split_scaled_overlap_count": 0,
        }
        atomic_json(SUMMARY_JSON, summary)
        REPORT_TXT.write_text(
            "VS2 RESIDUAL SCALED OVERLAP REPAIR PLAN\n"
            "No residual overlap exists; no repair is needed.\n",
            encoding="utf-8",
        )
        print("No residual overlap exists; no repair is needed.")
        print("VS2 RESIDUAL SCALED OVERLAP REPAIR PLAN COMPLETED")
        return

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

    source_database = runner.source_database_path(DATABASE)
    files = runner.discover_files(source_database, CSV_ROOT)

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
                f"[mapping] file {file_number}/{len(files)}: {Path(csv_path).name}",
                flush=True,
            )

            original_forward, original_reverse, family_labels = file_assignment_arrays(
                connection=connection,
                file_id=int(file_id),
                expected_rows=int(expected_rows),
            )

            for split_code, split in enumerate(SPLITS):
                selected = np.asarray(
                    runner.selected_rows(
                        DATABASE,
                        int(file_id),
                        "grouped_float32",
                        split,
                    ),
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
                    forward_u, reverse_u = runner.float32_fingerprints(scaled)
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

                        row_number = int(selected[local_start + local_index])
                        original_pair = (
                            int(original_forward[row_number]),
                            int(original_reverse[row_number]),
                        )
                        family_label = int(family_labels[row_number])
                        observed_label = int(labels[local_index])

                        if observed_label != family_label:
                            raise RuntimeError(
                                "Cache/assignment label mismatch: "
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
            f"Scaled component mapping mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    components: list[dict[str, Any]] = []
    original_group_count = 0
    label_conflict_component_count = 0

    for scaled_pair in sorted(component_groups):
        groups = component_groups[scaled_pair]
        current_rows = [0, 0, 0]
        family_labels_in_component: set[int] = set()
        group_records: list[dict[str, Any]] = []

        for original_pair in sorted(groups):
            record = groups[original_pair]
            rows_by_split = [int(value) for value in record["rows_by_split"]]
            nonzero_splits = [
                code for code, value in enumerate(rows_by_split) if value > 0
            ]
            if len(nonzero_splits) != 1:
                raise RuntimeError(
                    f"Original fingerprint spans multiple grouped splits: "
                    f"{original_pair} -> {rows_by_split}"
                )

            old_split = int(nonzero_splits[0])
            raw_row_count = int(sum(rows_by_split))
            family_label = int(record["family_label"])

            family_labels_in_component.add(family_label)
            current_rows[old_split] += raw_row_count
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

        label_conflict = len(family_labels_in_component) > 1
        label_conflict_component_count += int(label_conflict)

        components.append(
            {
                "scaled_hash_forward": int(scaled_pair[0]),
                "scaled_hash_reverse": int(scaled_pair[1]),
                "raw_row_count": int(sum(current_rows)),
                "current_rows_by_split": current_rows,
                "family_labels": sorted(family_labels_in_component),
                "label_conflict": label_conflict,
                "groups": group_records,
            }
        )

    if label_conflict_component_count != 0:
        raise RuntimeError(
            "Residual scaled collisions include cross-family label conflicts: "
            f"{label_conflict_component_count}"
        )

    components_by_family: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        family = int(component["family_labels"][0])
        components_by_family[family].append(component)

    assignments_by_component: dict[tuple[int, int], int] = {}
    minimum_moved_rows = 0
    affected_rows_by_family_split: dict[int, list[int]] = {}

    for family, family_components in sorted(components_by_family.items()):
        target_rows = [0, 0, 0]
        for component in family_components:
            for split_code in range(3):
                target_rows[split_code] += int(
                    component["current_rows_by_split"][split_code]
                )

        affected_rows_by_family_split[family] = target_rows
        assignments, family_move_cost = exact_count_preserving_assignment(
            components=family_components,
            target_train_rows=target_rows[0],
            target_validation_rows=target_rows[1],
        )
        minimum_moved_rows += int(family_move_cost)

        for component, new_split in zip(family_components, assignments):
            key = (
                int(component["scaled_hash_forward"]),
                int(component["scaled_hash_reverse"]),
            )
            assignments_by_component[key] = int(new_split)

    plan_rows: list[dict[str, Any]] = []
    moved_group_count = 0
    moved_rows_check = 0
    new_rows_by_family_split: dict[int, list[int]] = {
        family: [0, 0, 0] for family in components_by_family
    }

    for component in components:
        component_key = (
            int(component["scaled_hash_forward"]),
            int(component["scaled_hash_reverse"]),
        )
        new_split = assignments_by_component[component_key]
        family = int(component["family_labels"][0])
        new_rows_by_family_split[family][new_split] += int(
            component["raw_row_count"]
        )

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

    if moved_rows_check != minimum_moved_rows:
        raise RuntimeError(
            f"Moved-row objective mismatch: "
            f"{moved_rows_check} != {minimum_moved_rows}"
        )

    for family in components_by_family:
        if (
            new_rows_by_family_split[family]
            != affected_rows_by_family_split[family]
        ):
            raise RuntimeError(
                f"Per-family affected split totals changed for family={family}: "
                f"old={affected_rows_by_family_split[family]}, "
                f"new={new_rows_by_family_split[family]}"
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
            "family_label",
            "class_name",
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
            family = int(component["family_labels"][0])
            component_key = (
                int(component["scaled_hash_forward"]),
                int(component["scaled_hash_reverse"]),
            )
            new_split = assignments_by_component[component_key]
            writer.writerow(
                {
                    "scaled_hash_forward": component["scaled_hash_forward"],
                    "scaled_hash_reverse": component["scaled_hash_reverse"],
                    "family_label": family,
                    "class_name": CLASS_NAMES[family],
                    "original_group_count": len(component["groups"]),
                    "raw_row_count": component["raw_row_count"],
                    "old_train_rows": component["current_rows_by_split"][0],
                    "old_validation_rows": component["current_rows_by_split"][1],
                    "old_test_rows": component["current_rows_by_split"][2],
                    "new_grouped_split_code": new_split,
                    "new_grouped_split": split_name(new_split),
                }
            )

    summary = {
        "status": "completed",
        "analysis_role": "repair_plan_only_no_input_artifact_mutation",
        "parent_protocol_version": "2.2",
        "planned_protocol_version": "2.3",
        "residual_scaled_collision_component_count": overlap_count,
        "affected_original_fingerprint_group_count": original_group_count,
        "affected_raw_row_count": matched_rows,
        "label_conflict_component_count": label_conflict_component_count,
        "affected_class_names": [
            CLASS_NAMES[family] for family in sorted(components_by_family)
        ],
        "minimum_moved_raw_rows": int(minimum_moved_rows),
        "moved_original_fingerprint_group_count": int(moved_group_count),
        "planned_cross_split_scaled_overlap_count": 0,
        "affected_rows_by_family_and_split": {
            CLASS_NAMES[family]: {
                SPLITS[split_code]: int(values[split_code])
                for split_code in range(3)
            }
            for family, values in sorted(affected_rows_by_family_split.items())
        },
        "validation_checks": {
            "audit_database_quick_check_ok": True,
            "all_residual_components_mapped": len(components) == overlap_count,
            "no_cross_family_label_conflicts": label_conflict_component_count == 0,
            "each_original_group_currently_in_one_split": True,
            "per_family_affected_split_totals_preserved_exactly": True,
            "minimum_move_objective_consistent": (
                moved_rows_check == minimum_moved_rows
            ),
            "planned_zero_scaled_overlap_for_current_components": True,
        },
        "source_sha256": {
            "parent_assignment_database": sha256(DATABASE),
            "residual_scaled_overlap_audit_database": sha256(AUDIT_DATABASE),
            "runner_script": sha256(RUNNER_SCRIPT),
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
        "VS2 RESIDUAL SCALED OVERLAP REPAIR PLAN",
        "=" * 88,
        f"Residual scaled collision components : {overlap_count:,}",
        f"Affected original fingerprint groups : {original_group_count:,}",
        f"Affected raw rows                    : {matched_rows:,}",
        f"Label-conflict components            : {label_conflict_component_count:,}",
        f"Minimum moved raw rows               : {minimum_moved_rows:,}",
        f"Moved original fingerprint groups    : {moved_group_count:,}",
        "",
        "AFFECTED ROWS BY FAMILY AND SPLIT",
    ]

    for family, values in sorted(affected_rows_by_family_split.items()):
        report_lines.append(
            f"{CLASS_NAMES[family]}: "
            f"train={values[0]:,}, validation={values[1]:,}, test={values[2]:,}"
        )

    report_lines.extend(["", "VALIDATION CHECKS"])
    for key, value in summary["validation_checks"].items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend(
        [
            "",
            f"Plan       : {PLAN_CSV}",
            f"Components : {COMPONENT_CSV}",
            f"Summary    : {SUMMARY_JSON}",
            "",
            "VS2 RESIDUAL SCALED OVERLAP REPAIR PLAN COMPLETED",
        ]
    )

    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
