"""Audit the 371 cross-split collisions after VS2 scaling and float32 casting.

This script is diagnostic only. It does not modify splits, caches, manifests,
or training outputs.
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
STAGE38_SCRIPT = PROJECT_ROOT / "scripts" / "38_run_b0_matched_leakage_ablation.py"
ARM_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vs2_float32_leakage_ablation_cache_v2"
    / "grouped_float32"
)
AUDIT_DATABASE = ARM_DIRECTORY / "scaled_model_input_overlap_audit.sqlite"
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "audit"
SUMMARY_FILE = OUTPUT_DIRECTORY / "vs2_scaled_overlap_detail_summary_v2.json"
DETAIL_FILE = OUTPUT_DIRECTORY / "vs2_scaled_overlap_detail_rows_v2.csv"
TEXT_REPORT_FILE = OUTPUT_DIRECTORY / "vs2_scaled_overlap_detail_report_v2.txt"

SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("benign", "gafgyt", "mirai")
EXPECTED_HASH_OVERLAP_COUNT = 371
CHUNK_SIZE = 100_000


def load_stage38_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage38_runtime", STAGE38_SCRIPT)
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
    temporary.replace(path)


def split_signature(split_codes: set[int]) -> str:
    return "|".join(SPLITS[code] for code in sorted(split_codes))


def main() -> None:
    print("=" * 88)
    print("VS2 SCALED OVERLAP DETAIL AUDIT")
    print("=" * 88)

    for required in (
        STAGE38_SCRIPT,
        AUDIT_DATABASE,
        ARM_DIRECTORY / "mean.npy",
        ARM_DIRECTORY / "std.npy",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    stage38 = load_stage38_module()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(AUDIT_DATABASE) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"Audit database quick_check failed: {quick_check}")

        overlap_rows = connection.execute(
            """
            SELECT
                hash_forward,
                hash_reverse,
                COUNT(*) AS split_count,
                GROUP_CONCAT(split_code, '|') AS split_codes
            FROM scaled_fingerprints
            GROUP BY hash_forward, hash_reverse
            HAVING COUNT(*) > 1
            ORDER BY hash_forward, hash_reverse
            """
        ).fetchall()

    hash_pairs = {(int(row[0]), int(row[1])) for row in overlap_rows}
    if len(hash_pairs) != EXPECTED_HASH_OVERLAP_COUNT:
        raise RuntimeError(
            "Unexpected Stage-38 overlap count: "
            f"{len(hash_pairs)} != {EXPECTED_HASH_OVERLAP_COUNT}"
        )

    target_forward = np.asarray([pair[0] for pair in hash_pairs], dtype=np.int64).view(
        np.uint64
    )
    target_reverse = np.asarray([pair[1] for pair in hash_pairs], dtype=np.int64).view(
        np.uint64
    )
    target_combos = np.unique(pair_combo(target_forward, target_reverse))

    mean = np.load(ARM_DIRECTORY / "mean.npy")
    scale = np.load(ARM_DIRECTORY / "std.npy")

    # pair -> exact SHA values observed for that dual-hash pair
    pair_to_exact_hashes: dict[tuple[int, int], set[str]] = defaultdict(set)

    # (dual-hash pair, exact SHA) -> counts and split/class coverage
    exact_records: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}

    scanned_rows = 0
    matched_rows = 0

    for split_code, split in enumerate(SPLITS):
        x = np.load(ARM_DIRECTORY / f"X_{split}.npy", mmap_mode="r")
        y = np.load(ARM_DIRECTORY / f"y_{split}.npy", mmap_mode="r")

        if len(x) != len(y):
            raise RuntimeError(f"X/y row mismatch for {split}: {len(x)} != {len(y)}")

        print("")
        print(f"Scanning {split}: {len(x):,} rows")

        for start in range(0, len(x), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(x))
            raw = np.asarray(x[start:stop], dtype=np.float64)
            scaled = ((raw - mean) / scale).astype(np.float32)

            forward_u, reverse_u = stage38.float32_fingerprints(scaled)
            combos = pair_combo(forward_u, reverse_u)
            candidate_mask = np.isin(combos, target_combos)
            candidate_indices = np.flatnonzero(candidate_mask)

            labels = np.asarray(y[start:stop], dtype=np.int64)

            for local_index in candidate_indices.tolist():
                pair = (
                    int(forward_u[local_index].view(np.int64)),
                    int(reverse_u[local_index].view(np.int64)),
                )
                if pair not in hash_pairs:
                    continue

                exact_bytes = scaled[local_index].tobytes(order="C")
                exact_sha256 = hashlib.sha256(exact_bytes).hexdigest()
                label = int(labels[local_index])

                pair_to_exact_hashes[pair].add(exact_sha256)
                key = (pair, exact_sha256)

                record = exact_records.get(key)
                if record is None:
                    record = {
                        "hash_forward": pair[0],
                        "hash_reverse": pair[1],
                        "exact_scaled_row_sha256": exact_sha256,
                        "split_codes": set(),
                        "row_count_by_split": [0, 0, 0],
                        "row_count_by_class": [0, 0, 0],
                        "row_count_by_split_class": [[0, 0, 0] for _ in SPLITS],
                    }
                    exact_records[key] = record

                record["split_codes"].add(split_code)
                record["row_count_by_split"][split_code] += 1
                if label < 0 or label >= len(CLASS_NAMES):
                    raise RuntimeError(f"Unexpected label {label} in {split}")
                record["row_count_by_class"][label] += 1
                record["row_count_by_split_class"][split_code][label] += 1
                matched_rows += 1

            scanned_rows += stop - start

            if stop == len(x) or stop % 500_000 == 0:
                print(
                    f"  {split}: {stop:,}/{len(x):,} | matched rows so far={matched_rows:,}",
                    flush=True,
                )

    exact_cross_records: list[dict[str, Any]] = []
    for record in exact_records.values():
        split_codes = set(record["split_codes"])
        if len(split_codes) <= 1:
            continue

        class_codes = {
            class_code
            for class_code, count in enumerate(record["row_count_by_class"])
            if count > 0
        }

        exact_cross_records.append(
            {
                **record,
                "split_codes": sorted(split_codes),
                "split_signature": split_signature(split_codes),
                "total_raw_rows": int(sum(record["row_count_by_split"])),
                "class_codes": sorted(class_codes),
                "class_names": [CLASS_NAMES[code] for code in sorted(class_codes)],
                "label_conflict": len(class_codes) > 1,
            }
        )

    exact_cross_records.sort(
        key=lambda item: (
            item["hash_forward"],
            item["hash_reverse"],
            item["exact_scaled_row_sha256"],
        )
    )

    hash_collision_pair_count = sum(
        1 for hashes in pair_to_exact_hashes.values() if len(hashes) > 1
    )

    split_signature_counts: dict[str, int] = defaultdict(int)
    affected_rows_by_split = [0, 0, 0]
    affected_rows_by_class = [0, 0, 0]
    affected_rows_by_split_class = [[0, 0, 0] for _ in SPLITS]
    label_conflict_count = 0

    for record in exact_cross_records:
        split_signature_counts[record["split_signature"]] += 1
        label_conflict_count += int(record["label_conflict"])

        for split_code in range(len(SPLITS)):
            affected_rows_by_split[split_code] += record["row_count_by_split"][split_code]
            for class_code in range(len(CLASS_NAMES)):
                affected_rows_by_split_class[split_code][class_code] += record[
                    "row_count_by_split_class"
                ][split_code][class_code]

        for class_code in range(len(CLASS_NAMES)):
            affected_rows_by_class[class_code] += record["row_count_by_class"][class_code]

    with DETAIL_FILE.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "hash_forward",
            "hash_reverse",
            "exact_scaled_row_sha256",
            "split_signature",
            "total_raw_rows",
            "train_rows",
            "validation_rows",
            "test_rows",
            "benign_rows",
            "gafgyt_rows",
            "mirai_rows",
            "class_names",
            "label_conflict",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in exact_cross_records:
            writer.writerow(
                {
                    "hash_forward": record["hash_forward"],
                    "hash_reverse": record["hash_reverse"],
                    "exact_scaled_row_sha256": record["exact_scaled_row_sha256"],
                    "split_signature": record["split_signature"],
                    "total_raw_rows": record["total_raw_rows"],
                    "train_rows": record["row_count_by_split"][0],
                    "validation_rows": record["row_count_by_split"][1],
                    "test_rows": record["row_count_by_split"][2],
                    "benign_rows": record["row_count_by_class"][0],
                    "gafgyt_rows": record["row_count_by_class"][1],
                    "mirai_rows": record["row_count_by_class"][2],
                    "class_names": "|".join(record["class_names"]),
                    "label_conflict": record["label_conflict"],
                }
            )

    summary = {
        "status": "completed",
        "analysis_role": "diagnostic_only_no_artifact_mutation",
        "stage38_overlap_hash_pair_count": len(hash_pairs),
        "exact_cross_split_scaled_row_count": len(exact_cross_records),
        "dual_hash_pairs_with_multiple_exact_sha256_values": hash_collision_pair_count,
        "exact_cross_split_label_conflict_count": label_conflict_count,
        "scanned_raw_rows": scanned_rows,
        "matched_raw_rows": matched_rows,
        "split_signature_counts": dict(sorted(split_signature_counts.items())),
        "affected_raw_rows_by_split": {
            SPLITS[index]: int(value)
            for index, value in enumerate(affected_rows_by_split)
        },
        "affected_raw_rows_by_class": {
            CLASS_NAMES[index]: int(value)
            for index, value in enumerate(affected_rows_by_class)
        },
        "affected_raw_rows_by_split_and_class": {
            SPLITS[split_code]: {
                CLASS_NAMES[class_code]: int(
                    affected_rows_by_split_class[split_code][class_code]
                )
                for class_code in range(len(CLASS_NAMES))
            }
            for split_code in range(len(SPLITS))
        },
        "source_artifacts": {
            "stage38_script": str(STAGE38_SCRIPT),
            "grouped_cache": str(ARM_DIRECTORY),
            "scaled_overlap_audit_database": str(AUDIT_DATABASE),
        },
        "outputs": {
            "detail_csv": str(DETAIL_FILE),
            "summary_json": str(SUMMARY_FILE),
            "text_report": str(TEXT_REPORT_FILE),
        },
    }
    atomic_json(SUMMARY_FILE, summary)

    report_lines = [
        "=" * 88,
        "VS2 SCALED OVERLAP DETAIL AUDIT",
        "=" * 88,
        f"Stage-38 overlap hash pairs           : {len(hash_pairs):,}",
        f"Exact cross-split scaled rows         : {len(exact_cross_records):,}",
        f"Hash pairs with multiple exact rows   : {hash_collision_pair_count:,}",
        f"Exact cross-split label conflicts     : {label_conflict_count:,}",
        f"Scanned raw rows                      : {scanned_rows:,}",
        f"Matched affected raw rows             : {matched_rows:,}",
        "",
        "SPLIT SIGNATURE COUNTS",
    ]
    for key, value in sorted(split_signature_counts.items()):
        report_lines.append(f"{key}: {value:,}")

    report_lines.extend(["", "AFFECTED RAW ROWS BY SPLIT"])
    for index, split in enumerate(SPLITS):
        report_lines.append(f"{split}: {affected_rows_by_split[index]:,}")

    report_lines.extend(["", "AFFECTED RAW ROWS BY CLASS"])
    for index, class_name in enumerate(CLASS_NAMES):
        report_lines.append(f"{class_name}: {affected_rows_by_class[index]:,}")

    report_lines.extend(
        [
            "",
            f"Summary: {SUMMARY_FILE}",
            f"Details: {DETAIL_FILE}",
            "",
            "VS2 SCALED OVERLAP DETAIL AUDIT COMPLETED",
        ]
    )
    TEXT_REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
