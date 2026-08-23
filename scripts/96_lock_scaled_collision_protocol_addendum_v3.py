from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

BASE_PROTOCOL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_locked_v3.json"
)

BASE_LOCK_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_lock_manifest_v3.json"
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

FAILURE_REPORT = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_1.building"
    / "scaled_overlap_failure.json"
)

BUILD_DIRECTORY = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_1.building"
)

OUTPUT_ADDENDUM = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_v3_2.json"
)

OUTPUT_MANIFEST = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "tabular_baseline_protocol_scaled_collision_addendum_manifest_v3_2.json"
)

BASE_VERSION = "tabular_baseline_protocol_v3_1"
ADDENDUM_VERSION = "tabular_baseline_protocol_v3_2"

EXPECTED_COMPONENTS = 280
EXPECTED_AFFECTED_FINGERPRINTS = 938
EXPECTED_MOVES = 353
EXPECTED_MOVED_RAW_ROWS = 353
EXPECTED_TWO_SPLIT_COMPONENTS = 263
EXPECTED_THREE_SPLIT_COMPONENTS = 17


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


required_paths = [
    BASE_PROTOCOL,
    BASE_LOCK_MANIFEST,
    REPAIR_PLAN,
    COMPONENTS_CSV,
    MOVES_CSV,
    FAILURE_REPORT,
    BUILD_DIRECTORY,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (
    OUTPUT_ADDENDUM,
    OUTPUT_MANIFEST,
):
    if path.exists():
        raise FileExistsError(
            "Locked addendum artifact already exists; "
            f"refusing to overwrite: {path}"
        )

base_protocol = json.loads(
    BASE_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

base_manifest = json.loads(
    BASE_LOCK_MANIFEST.read_text(
        encoding="utf-8"
    )
)

repair_plan = json.loads(
    REPAIR_PLAN.read_text(
        encoding="utf-8"
    )
)

failure_report = json.loads(
    FAILURE_REPORT.read_text(
        encoding="utf-8"
    )
)

with COMPONENTS_CSV.open(
    "r",
    newline="",
    encoding="utf-8",
) as handle:
    component_rows = list(
        csv.DictReader(handle)
    )

with MOVES_CSV.open(
    "r",
    newline="",
    encoding="utf-8",
) as handle:
    move_rows = list(
        csv.DictReader(handle)
    )

base_protocol_sha256 = sha256_file(
    BASE_PROTOCOL
)

base_manifest_sha256 = sha256_file(
    BASE_LOCK_MANIFEST
)

repair_plan_sha256 = sha256_file(
    REPAIR_PLAN
)

components_sha256 = sha256_file(
    COMPONENTS_CSV
)

moves_sha256 = sha256_file(
    MOVES_CSV
)

failure_report_sha256 = sha256_file(
    FAILURE_REPORT
)

move_raw_row_sum = sum(
    int(row["raw_row_count"])
    for row in move_rows
)

component_family_values = {
    int(row["family"])
    for row in component_rows
}

move_family_values = {
    int(row["family"])
    for row in move_rows
}

two_split_components = sum(
    int(row["split_span"]) == 2
    for row in component_rows
)

three_split_components = sum(
    int(row["split_span"]) == 3
    for row in component_rows
)

split_plan = repair_plan.get(
    "split_plan"
)

if not isinstance(
    split_plan,
    list,
):
    raise RuntimeError(
        "Repair plan split_plan is missing."
    )

fingerprint_delta_sum = sum(
    int(row["fingerprint_delta"])
    for row in split_plan
)

raw_row_delta_sum = sum(
    int(row["raw_row_delta"])
    for row in split_plan
)

all_individual_deltas_zero = all(
    int(row["fingerprint_delta"]) == 0
    and int(row["raw_row_delta"]) == 0
    for row in split_plan
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
        == BASE_VERSION
    ),
    "base_manifest_locked": (
        base_manifest.get("status")
        == "locked"
    ),
    "base_protocol_hash_matches_manifest": (
        base_manifest.get(
            "protocol_file_sha256"
        )
        == base_protocol_sha256
    ),
    "repair_plan_not_applied": (
        repair_plan.get("status")
        == "plan_only_not_applied"
    ),
    "repair_plan_checks_passed": (
        repair_plan.get(
            "all_checks_passed"
        )
        is True
    ),
    "failure_report_failed_only_for_overlap": (
        failure_report.get("passed")
        is False
        and int(
            failure_report[
                "cross_split_model_input_fingerprint_count"
            ]
        )
        == EXPECTED_COMPONENTS
        and int(
            failure_report[
                "within_split_family_conflict_count"
            ]
        )
        == 0
        and int(
            failure_report[
                "global_family_conflict_count"
            ]
        )
        == 0
    ),
    "component_count_matches": (
        len(component_rows)
        == EXPECTED_COMPONENTS
        == int(
            repair_plan[
                "cross_split_scaled_component_count"
            ]
        )
    ),
    "affected_fingerprint_count_matches": (
        int(
            repair_plan[
                "affected_original_fingerprint_row_count"
            ]
        )
        == EXPECTED_AFFECTED_FINGERPRINTS
    ),
    "move_count_matches": (
        len(move_rows)
        == EXPECTED_MOVES
        == int(
            repair_plan[
                "proposed_moved_fingerprint_count"
            ]
        )
    ),
    "moved_raw_rows_match": (
        move_raw_row_sum
        == EXPECTED_MOVED_RAW_ROWS
        == int(
            repair_plan[
                "proposed_moved_raw_row_count"
            ]
        )
    ),
    "all_moved_fingerprints_have_unit_raw_weight": (
        all(
            int(row["raw_row_count"])
            == 1
            for row in move_rows
        )
    ),
    "component_family_is_benign_only": (
        component_family_values == {0}
    ),
    "move_family_is_benign_only": (
        move_family_values == {0}
    ),
    "component_span_distribution_matches": (
        two_split_components
        == EXPECTED_TWO_SPLIT_COMPONENTS
        and three_split_components
        == EXPECTED_THREE_SPLIT_COMPONENTS
    ),
    "no_identity_moves": all(
        int(row["source_split_code"])
        != int(
            row["destination_split_code"]
        )
        for row in move_rows
    ),
    "global_fingerprint_delta_zero": (
        fingerprint_delta_sum == 0
    ),
    "global_raw_row_delta_zero": (
        raw_row_delta_sum == 0
    ),
    "each_split_delta_zero": (
        all_individual_deltas_zero
    ),
}

failed_checks = [
    name
    for name, passed in checks.items()
    if not passed
]

if failed_checks:
    raise RuntimeError(
        "Scaled-collision addendum lock failed: "
        + ", ".join(failed_checks)
    )

locked_at = utc_now()

addendum = {
    "status": "locked",
    "protocol_version": (
        ADDENDUM_VERSION
    ),
    "locked_at_utc": locked_at,
    "amends_protocol_version": (
        BASE_VERSION
    ),
    "amendment_scope": (
        "Pre-training deterministic repair of "
        "cross-split collisions introduced only after "
        "train-only StandardScaler transformation and "
        "final float32 conversion."
    ),
    "roadmap_impact": (
        "No phase, model, seed, metric, or test-use "
        "change. This addendum remains inside Phase 4B "
        "cache integrity preparation."
    ),
    "scientific_trigger": {
        "detected_before_any_phase_4_model_training": True,
        "processed_fingerprint_rows": int(
            failure_report[
                "processed_fingerprint_rows"
            ]
        ),
        "cross_split_scaled_component_count": (
            EXPECTED_COMPONENTS
        ),
        "affected_original_fingerprint_count": (
            EXPECTED_AFFECTED_FINGERPRINTS
        ),
        "within_split_family_conflict_count": 0,
        "global_family_conflict_count": 0,
        "affected_family": "benign",
    },
    "repair_policy": {
        "input_used_for_collision_detection": (
            "features only after train-only scaler and "
            "final float32 conversion"
        ),
        "model_metrics_used": False,
        "validation_performance_used": False,
        "test_performance_used": False,
        "labels_used_for_destination_selection": False,
        "labels_used_only_for_safety_conflict_check": True,
        "component_constraint": (
            "Every scaled collision component must be "
            "assigned wholly to exactly one split."
        ),
        "destination_objective_order": [
            "minimize moved original fingerprint rows",
            (
                "among fingerprint-minimal choices, minimize "
                "moved represented raw rows"
            ),
            (
                "among remaining ties, minimize cumulative "
                "absolute split raw-row deltas"
            ),
            (
                "then minimize cumulative absolute split "
                "fingerprint deltas"
            ),
            "then choose the lowest split code",
        ],
        "post_repair_scaler_policy": (
            "Refit StandardScaler on the repaired train "
            "fingerprint set only."
        ),
        "post_repair_audit_policy": (
            "Recompute final float32 fingerprints over all "
            "splits and fail closed unless cross-split "
            "collision count and family-conflict count are zero."
        ),
        "iteration_policy": (
            "If refitting the scaler creates new cross-split "
            "collisions, repeat the same locked deterministic "
            "repair rule and retain a complete iteration audit."
        ),
        "locked_split_database_mutation": False,
        "base_cache_mutation": False,
        "derived_cache_required": True,
    },
    "locked_repair_plan": {
        "scaled_collision_components": (
            EXPECTED_COMPONENTS
        ),
        "two_split_components": (
            EXPECTED_TWO_SPLIT_COMPONENTS
        ),
        "three_split_components": (
            EXPECTED_THREE_SPLIT_COMPONENTS
        ),
        "affected_original_fingerprints": (
            EXPECTED_AFFECTED_FINGERPRINTS
        ),
        "moved_original_fingerprints": (
            EXPECTED_MOVES
        ),
        "moved_represented_raw_rows": (
            EXPECTED_MOVED_RAW_ROWS
        ),
        "all_moved_fingerprints_raw_row_count_is_one": True,
        "per_split_fingerprint_delta": {
            str(row["split"]): int(
                row["fingerprint_delta"]
            )
            for row in split_plan
        },
        "per_split_raw_row_delta": {
            str(row["split"]): int(
                row["raw_row_delta"]
            )
            for row in split_plan
        },
        "all_split_counts_preserved_exactly": True,
    },
    "unchanged_locks": {
        "task": "family_3",
        "classes": [
            "benign",
            "gafgyt",
            "mirai",
        ],
        "feature_count": 115,
        "source_raw_row_count": 7_062_606,
        "source_fingerprint_count": 2_278_176,
        "model_count": 4,
        "expected_run_count": 12,
        "model_hyperparameters_changed": False,
        "seed_set_changed": False,
        "primary_evaluation_view_changed": False,
        "secondary_evaluation_view_changed": False,
        "test_used_for_selection": False,
    },
    "provenance": {
        "base_protocol": str(
            BASE_PROTOCOL
        ),
        "base_protocol_sha256": (
            base_protocol_sha256
        ),
        "base_lock_manifest": str(
            BASE_LOCK_MANIFEST
        ),
        "base_lock_manifest_sha256": (
            base_manifest_sha256
        ),
        "failure_report": str(
            FAILURE_REPORT
        ),
        "failure_report_sha256": (
            failure_report_sha256
        ),
        "repair_plan": str(
            REPAIR_PLAN
        ),
        "repair_plan_sha256": (
            repair_plan_sha256
        ),
        "components_csv": str(
            COMPONENTS_CSV
        ),
        "components_csv_sha256": (
            components_sha256
        ),
        "moves_csv": str(
            MOVES_CSV
        ),
        "moves_csv_sha256": (
            moves_sha256
        ),
        "source_build_directory": str(
            BUILD_DIRECTORY
        ),
        "locker_script": str(
            Path(__file__).resolve()
        ),
        "locker_script_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
    },
    "validation_checks": checks,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_ADDENDUM,
    addendum,
)

manifest = {
    "status": "locked",
    "protocol_version": (
        ADDENDUM_VERSION
    ),
    "locked_at_utc": locked_at,
    "addendum": str(
        OUTPUT_ADDENDUM
    ),
    "addendum_sha256": sha256_file(
        OUTPUT_ADDENDUM
    ),
    "base_protocol_sha256": (
        base_protocol_sha256
    ),
    "repair_plan_sha256": (
        repair_plan_sha256
    ),
    "components_csv_sha256": (
        components_sha256
    ),
    "moves_csv_sha256": (
        moves_sha256
    ),
    "expected_first_iteration_moves": (
        EXPECTED_MOVES
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_MANIFEST,
    manifest,
)

print("=" * 92)
print("SCALED-COLLISION PROTOCOL ADDENDUM LOCK")
print("=" * 92)
print(f"Base protocol       : {BASE_VERSION}")
print(f"Addendum version    : {ADDENDUM_VERSION}")
print("Status              : locked")
print(
    "Collision components: "
    f"{EXPECTED_COMPONENTS:,}"
)
print(
    "Affected fingerprints: "
    f"{EXPECTED_AFFECTED_FINGERPRINTS:,}"
)
print(
    "Locked moves        : "
    f"{EXPECTED_MOVES:,}"
)
print(
    "Moved raw rows      : "
    f"{EXPECTED_MOVED_RAW_ROWS:,}"
)
print("All split deltas    : 0")
print("Model/seed changes  : none")
print("Test metrics used   : False")
print()
print("LOCKED FILES")
print(f"  Addendum : {OUTPUT_ADDENDUM}")
print(f"  Manifest : {OUTPUT_MANIFEST}")
print()
print("VALIDATION CHECKS")

for name, passed in checks.items():
    print(f"  {name}: {passed}")

print()
print("All checks passed: True")
print(
    "SCALED-COLLISION PROTOCOL ADDENDUM LOCK COMPLETED"
)
