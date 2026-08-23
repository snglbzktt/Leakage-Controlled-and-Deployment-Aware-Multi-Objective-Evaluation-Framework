from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

RUNNER_SCRIPT = (
    ROOT
    / "scripts"
    / "104_run_decision_tree_b0_five_seeds_v3_2.py"
)

RUN_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "runs"
)

SEEDS = [42, 123, 2026, 3407, 8192]

RUN_DIRECTORIES = [
    RUN_ROOT
    / f"decision_tree_b0__seed_{seed}"
    for seed in SEEDS
]

TEMPORARY_SCALED_TRAIN = (
    ROOT
    / "results"
    / "v2"
    / "tabular_baselines"
    / "decision_tree_b0_scaled_train_float32.tmp.npy"
)

POSSIBLE_AGGREGATES = [
    (
        ROOT
        / "results"
        / "v2"
        / "audit"
        / "decision_tree_b0_aggregate_v3_2.json"
    ),
    (
        ROOT
        / "results"
        / "v2"
        / "audit"
        / "decision_tree_b0_runs_v3_2.csv"
    ),
]

ARCHIVE_FINAL = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "invalid_runs"
    / "decision_tree_b0_input_space_parser_bug_v3_2"
)

ARCHIVE_BUILDING = ARCHIVE_FINAL.with_name(
    ARCHIVE_FINAL.name + ".building"
)

INVALIDATION_REPORT = (
    ROOT
    / "results"
    / "v2"
    / "audit"
    / "decision_tree_b0_invalidation_v3_2.json"
)

EXPECTED_MODEL_ID = "decision_tree_b0"
EXPECTED_INPUT_SPACE = "canonical_float32_unscaled"
EXPECTED_PROTOCOL_VERSION = "tabular_baseline_protocol_v3_2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


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

    os.replace(temporary, path)


def directory_size_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


required_paths = [
    RUNNER_SCRIPT,
    *RUN_DIRECTORIES,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for path in (
    ARCHIVE_FINAL,
    ARCHIVE_BUILDING,
    INVALIDATION_REPORT,
):
    if path.exists():
        raise FileExistsError(
            "Invalidation artifact already exists; "
            f"refusing to overwrite: {path}"
        )

runner_text = RUNNER_SCRIPT.read_text(
    encoding="utf-8"
)

bug_expression_present = (
    '"scaled" in input_space'
    in runner_text
)

if not bug_expression_present:
    raise RuntimeError(
        "Expected input-space parser bug expression "
        "was not found in the runner script."
    )

run_inventory: list[dict[str, Any]] = []

for expected_seed, run_directory in zip(
    SEEDS,
    RUN_DIRECTORIES,
):
    metrics_path = (
        run_directory / "metrics.json"
    )

    manifest_path = (
        run_directory / "run_manifest.json"
    )

    status_path = (
        run_directory / "run_status.json"
    )

    for path in (
        metrics_path,
        manifest_path,
        status_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    status = read_json(status_path)

    checks = {
        "model_id_matches": (
            metrics.get("model_id")
            == EXPECTED_MODEL_ID
            and manifest.get("model_id")
            == EXPECTED_MODEL_ID
            and status.get("model_id")
            == EXPECTED_MODEL_ID
        ),
        "seed_matches": (
            int(metrics.get("seed"))
            == expected_seed
            and int(manifest.get("seed"))
            == expected_seed
            and int(status.get("seed"))
            == expected_seed
        ),
        "protocol_version_matches": (
            metrics.get("protocol_version")
            == EXPECTED_PROTOCOL_VERSION
            and manifest.get(
                "protocol_version"
            )
            == EXPECTED_PROTOCOL_VERSION
        ),
        "declared_input_space_is_unscaled": (
            str(
                manifest.get("input_space")
            ).strip().lower()
            == EXPECTED_INPUT_SPACE
        ),
        "run_completed": (
            manifest.get("status")
            == "completed"
            and status.get("status")
            == "completed"
            and status.get("stage")
            == "completed"
        ),
        "test_evaluation_count_is_one": (
            int(
                metrics.get(
                    "test_evaluation_count"
                )
            )
            == 1
            and int(
                manifest.get(
                    "observed_test_evaluation_count"
                )
            )
            == 1
        ),
    }

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            f"Run validation failed for seed "
            f"{expected_seed}: {failed}"
        )

    run_inventory.append(
        {
            "run_id": metrics["run_id"],
            "seed": expected_seed,
            "source_directory": str(
                run_directory
            ),
            "size_bytes": (
                directory_size_bytes(
                    run_directory
                )
            ),
            "metrics_sha256": (
                sha256_file(metrics_path)
            ),
            "run_manifest_sha256": (
                sha256_file(manifest_path)
            ),
            "run_status_sha256": (
                sha256_file(status_path)
            ),
            "declared_input_space": (
                manifest["input_space"]
            ),
            "test_fingerprint_macro_f1": (
                metrics["test"][
                    "primary_fingerprint_level"
                ]["macro_f1"]
            ),
            "checks": checks,
        }
    )

temporary_file_inventory: dict[str, Any] = {
    "path": str(TEMPORARY_SCALED_TRAIN),
    "exists": TEMPORARY_SCALED_TRAIN.exists(),
    "size_bytes": 0,
    "sha256": None,
}

if TEMPORARY_SCALED_TRAIN.exists():
    temporary_file_inventory[
        "size_bytes"
    ] = int(
        TEMPORARY_SCALED_TRAIN.stat().st_size
    )

    print(
        "Hashing stale temporary scaled matrix...",
        flush=True,
    )

    temporary_file_inventory[
        "sha256"
    ] = sha256_file(
        TEMPORARY_SCALED_TRAIN
    )

aggregate_inventory = []

for path in POSSIBLE_AGGREGATES:
    if path.exists():
        aggregate_inventory.append(
            {
                "path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(path),
            }
        )

free_before = shutil.disk_usage(ROOT).free

ARCHIVE_BUILDING.mkdir(
    parents=True,
    exist_ok=False,
)

archived_runs_directory = (
    ARCHIVE_BUILDING / "runs"
)

archived_runs_directory.mkdir()

archived_aggregates_directory = (
    ARCHIVE_BUILDING
    / "partial_aggregate_artifacts"
)

archived_aggregates_directory.mkdir()

print("=" * 92)
print("QUARANTINE INVALID DECISION TREE B0 RUNS")
print("=" * 92)

for run_directory in RUN_DIRECTORIES:
    destination = (
        archived_runs_directory
        / run_directory.name
    )

    print(
        f"[move] {run_directory.name}",
        flush=True,
    )

    shutil.move(
        str(run_directory),
        str(destination),
    )

for path in POSSIBLE_AGGREGATES:
    if path.exists():
        shutil.move(
            str(path),
            str(
                archived_aggregates_directory
                / path.name
            ),
        )

if TEMPORARY_SCALED_TRAIN.exists():
    print(
        "[delete] stale temporary scaled matrix",
        flush=True,
    )

    TEMPORARY_SCALED_TRAIN.unlink()

archive_manifest = {
    "status": "invalidated_and_quarantined",
    "generated_at_utc": utc_now(),
    "model_id": EXPECTED_MODEL_ID,
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "scientific_reason": (
        "The locked run matrix declared "
        "'canonical_float32_unscaled', but the runner "
        "used substring parsing: 'scaled' in input_space. "
        "Because 'unscaled' contains 'scaled', all five "
        "runs were trained and evaluated in the scaled "
        "feature space. These results therefore do not "
        "conform to the locked unscaled Decision Tree "
        "protocol and must not be used."
    ),
    "runner_script": str(
        RUNNER_SCRIPT
    ),
    "runner_script_sha256": (
        sha256_file(RUNNER_SCRIPT)
    ),
    "bug_expression_confirmed": (
        bug_expression_present
    ),
    "run_inventory": run_inventory,
    "temporary_scaled_train": (
        temporary_file_inventory
    ),
    "partial_aggregate_inventory": (
        aggregate_inventory
    ),
    "test_metrics_must_not_be_reported": True,
    "rerun_required": True,
    "all_quarantine_prechecks_passed": True,
}

atomic_json(
    ARCHIVE_BUILDING
    / "invalidation_manifest.json",
    archive_manifest,
)

ARCHIVE_BUILDING.rename(
    ARCHIVE_FINAL
)

post_checks = {
    "archive_exists": (
        ARCHIVE_FINAL.exists()
    ),
    "invalidation_manifest_exists": (
        (
            ARCHIVE_FINAL
            / "invalidation_manifest.json"
        ).exists()
    ),
    "all_active_run_directories_removed": all(
        not path.exists()
        for path in RUN_DIRECTORIES
    ),
    "temporary_scaled_train_removed": (
        not TEMPORARY_SCALED_TRAIN.exists()
    ),
    "active_aggregate_files_removed": all(
        not path.exists()
        for path in POSSIBLE_AGGREGATES
    ),
}

failed_post = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post:
    raise RuntimeError(
        "Decision-tree quarantine post-check failed: "
        + ", ".join(failed_post)
    )

free_after = shutil.disk_usage(ROOT).free

report = {
    "status": "completed",
    "generated_at_utc": utc_now(),
    "model_id": EXPECTED_MODEL_ID,
    "protocol_version": (
        EXPECTED_PROTOCOL_VERSION
    ),
    "invalid_run_count": len(
        run_inventory
    ),
    "invalid_seeds": SEEDS,
    "archive": str(ARCHIVE_FINAL),
    "temporary_file_deleted": (
        temporary_file_inventory["exists"]
    ),
    "temporary_file_size_bytes": (
        temporary_file_inventory[
            "size_bytes"
        ]
    ),
    "free_disk_before_bytes": (
        free_before
    ),
    "free_disk_after_bytes": (
        free_after
    ),
    "post_checks": post_checks,
    "rerun_required": True,
    "all_checks_passed": True,
}

atomic_json(
    INVALIDATION_REPORT,
    report,
)

print()
print("=" * 92)
print("DECISION TREE B0 INVALIDATION SUMMARY")
print("=" * 92)
print(
    "Invalid runs quarantined : "
    f"{len(run_inventory)}"
)
print(
    "Seeds                    : "
    f"{SEEDS}"
)
print(
    "Protocol input space     : "
    f"{EXPECTED_INPUT_SPACE}"
)
print(
    "Bug                      : "
    "'scaled' matched inside 'unscaled'"
)
print(
    "Temporary file removed   : "
    f"{temporary_file_inventory['exists']}"
)
print(
    "Archive                  : "
    f"{ARCHIVE_FINAL}"
)
print(
    "Invalidation report      : "
    f"{INVALIDATION_REPORT}"
)
print(
    "Rerun required           : True"
)
print(
    "All checks passed        : True"
)
print(
    "DECISION TREE B0 INVALIDATION COMPLETED"
)
