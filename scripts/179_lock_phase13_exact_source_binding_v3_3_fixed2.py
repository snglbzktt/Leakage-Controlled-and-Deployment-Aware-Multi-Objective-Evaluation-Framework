from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"
RESULTS_V2 = ROOT / "results" / "v2"

DISCOVERY_PATH = (
    AUDIT
    / "phase13_targeted_exact_source_discovery_v3_3.json"
)

ASSET_PLAN_PATH = (
    ROOT
    / "configs"
    / "protocols"
    / "phase13_manuscript_asset_plan_v3_3.json"
)

ASSET_PLAN_LOCK_PATH = (
    AUDIT
    / "phase13_manuscript_asset_plan_locked_v3_3.json"
)

OUTPUT_BINDING_JSON = (
    ROOT
    / "configs"
    / "protocols"
    / "phase13_exact_source_binding_v3_3.json"
)

OUTPUT_BINDING_CSV = (
    AUDIT
    / "phase13_exact_source_binding_v3_3.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase13_exact_source_binding_preflight_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase13_exact_source_binding_locked_v3_3.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase13_exact_source_binding_lock_manifest_v3_3.json"
)

PROTOCOL_VERSION = (
    "phase13_exact_source_binding_v3_3"
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


SOURCE_PATHS = {
    "lodo_device_model_matrix": (
        RESULTS_V2
        / "early_lodo"
        / "global"
        / "early_lodo_device_model_matrix_v2.csv"
    ),
    "lodo_global_summary": (
        RESULTS_V2
        / "early_lodo"
        / "global"
        / "early_lodo_global_summary_v2.json"
    ),
    "phase4_summary": (
        AUDIT
        / "phase4_classical_baselines_summary_v3_2.json"
    ),
    "phase4_model_summary": (
        AUDIT
        / "phase4_classical_baselines_model_summary_v3_2.csv"
    ),
    "phase5_group_summary": (
        AUDIT
        / "phase5_compression_group_summary_v3_2.csv"
    ),
    "phase5_master_matrix": (
        AUDIT
        / "phase5_compression_master_run_matrix_v3_2.csv"
    ),
    "phase5_closure": (
        AUDIT
        / "phase5_compression_closure_v3_2.json"
    ),
    "phase6_group_summary": (
        AUDIT
        / "phase6_deployment_benchmark_group_summary_v3_2.csv"
    ),
    "phase6_locked_matrix": (
        AUDIT
        / "phase6_deployment_benchmark_locked_matrix_v3_2.csv"
    ),
    "phase6_summary": (
        AUDIT
        / "phase6_deployment_benchmark_summary_v3_2.json"
    ),
    "phase7_omnibus": (
        AUDIT
        / "phase7_statistical_omnibus_results_v3_2.csv"
    ),
    "phase7_posthoc_verified": (
        AUDIT
        / "phase7_statistical_posthoc_verified_v3_2.csv"
    ),
    "phase7_summary": (
        AUDIT
        / "phase7_statistical_analysis_summary_v3_2.json"
    ),
    "phase8_input_matrix": (
        AUDIT
        / "phase8_multi_objective_decision_input_matrix_v3_2.csv"
    ),
    "phase8_pareto_results": (
        AUDIT
        / "phase8_multi_objective_pareto_results_v3_2.csv"
    ),
    "phase8_selection_trace": (
        AUDIT
        / "phase8_multi_objective_selection_trace_v3_2.csv"
    ),
    "phase8_final_model_lock": (
        AUDIT
        / "phase8_final_model_family_locked_v3_2.json"
    ),
    "phase9_artifact_lock": (
        AUDIT
        / "phase9_final_deployment_artifact_locked_v3_2.json"
    ),
    "phase10_release_lock": (
        AUDIT
        / "phase10_release_archive_locked_v3_3.json"
    ),
    "phase11_all_runs": (
        RESULTS_V2
        / "phase11_input_robustness_v3_3"
        / "phase11_input_robustness_all_runs_v3_3.csv"
    ),
    "phase11_summary": (
        RESULTS_V2
        / "phase11_input_robustness_v3_3"
        / "phase11_input_robustness_summary_v3_3.json"
    ),
    "phase11_lock": (
        AUDIT
        / "phase11_input_robustness_locked_v3_3.json"
    ),
}


ASSET_BINDINGS = {
    "figure_01_methodology_overview": [
        "lodo_global_summary",
        "phase5_closure",
        "phase6_summary",
        "phase8_final_model_lock",
        "phase9_artifact_lock",
        "phase10_release_lock",
        "phase11_lock",
    ],
    "figure_02_lodo_generalization": [
        "lodo_device_model_matrix",
        "lodo_global_summary",
    ],
    "figure_03_compression_tradeoff": [
        "phase5_group_summary",
        "phase6_group_summary",
    ],
    "figure_04_pareto_selection": [
        "phase8_input_matrix",
        "phase8_pareto_results",
        "phase8_selection_trace",
        "phase8_final_model_lock",
    ],
    "figure_05_input_robustness": [
        "phase11_all_runs",
        "phase11_summary",
        "phase11_lock",
    ],
    "table_01_dataset_protocol": [
        "lodo_global_summary",
        "phase4_summary",
        "phase5_closure",
    ],
    "table_02_classical_baselines": [
        "phase4_model_summary",
        "phase4_summary",
    ],
    "table_03_compression_results": [
        "phase5_group_summary",
        "phase5_master_matrix",
        "phase5_closure",
    ],
    "table_04_deployment_benchmarks": [
        "phase6_group_summary",
        "phase6_locked_matrix",
        "phase6_summary",
    ],
    "table_05_statistical_analysis": [
        "phase7_omnibus",
        "phase7_posthoc_verified",
        "phase7_summary",
    ],
    "table_06_robustness_results": [
        "phase11_all_runs",
        "phase11_summary",
        "phase11_lock",
    ],
    "table_07_reproducibility_release": [
        "phase8_final_model_lock",
        "phase9_artifact_lock",
        "phase10_release_lock",
        "phase11_lock",
    ],
}


EXPECTED_CSV_ROWS = {
    "lodo_device_model_matrix": 27,
    "phase4_model_summary": 4,
    "phase5_group_summary": 22,
    "phase5_master_matrix": 110,
    "phase6_group_summary": 22,
    "phase6_locked_matrix": 110,
    "phase7_omnibus": 16,
    "phase7_posthoc_verified": 160,
    "phase8_input_matrix": 22,
    "phase8_pareto_results": 54,
    "phase8_selection_trace": 3,
    "phase11_all_runs": 27,
}


REQUIRED_HEADER_GROUPS = {
    "lodo_device_model_matrix": [
        ("held_out_device",),
        ("model_id",),
        ("test_fingerprint_macro_f1",),
    ],
    "phase4_model_summary": [
        ("model_id",),
        ("test_fingerprint_macro_f1", "mean_test_fingerprint_macro_f1"),
    ],
    "phase5_group_summary": [
        ("architecture",),
        ("variant",),
        ("mean_test_fingerprint_macro_f1", "test_fingerprint_macro_f1_mean"),
    ],
    "phase6_group_summary": [
        ("architecture",),
        ("variant",),
        ("latency", "batch1"),
        ("throughput", "batch32"),
    ],
    "phase7_omnibus": [
        ("architecture",),
        ("metric",),
        ("monte_carlo", "p"),
    ],
    "phase7_posthoc_verified": [
        ("architecture",),
        ("metric",),
        ("Holm_adjusted_p",),
    ],
    "phase8_input_matrix": [
        ("architecture",),
        ("variant",),
    ],
    "phase8_pareto_results": [
        ("architecture",),
        ("variant",),
        ("pareto",),
    ],
    "phase8_selection_trace": [
        ("profile",),
        ("selected_architecture",),
        ("selected_variant",),
        ("selection_rule",),
    ],
    "phase11_all_runs": [
        ("model_id",),
        ("condition_id",),
        ("fingerprint_macro_f1",),
        ("severe_fragility",),
    ],
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
            digest.update(block)

    return digest.hexdigest()


def replace_with_retry(
    source: Path,
    destination: Path,
) -> None:
    last_error: OSError | None = None

    for attempt in range(
        1,
        WINDOWS_FILE_RETRY_COUNT + 1,
    ):
        try:
            os.replace(
                source,
                destination,
            )
            return
        except PermissionError as error:
            last_error = error

            if attempt == WINDOWS_FILE_RETRY_COUNT:
                break

            time.sleep(
                WINDOWS_FILE_RETRY_DELAY_SECONDS
            )

    raise RuntimeError(
        "Windows kept the destination locked "
        f"after {WINDOWS_FILE_RETRY_COUNT} attempts: "
        f"{destination}"
    ) from last_error


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
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    replace_with_retry(
        temporary,
        path,
    )


def write_csv(
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

    replace_with_retry(
        temporary,
        path,
    )


def inspect_csv(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        headers = list(
            reader.fieldnames
            or []
        )

        row_count = sum(
            1
            for _ in reader
        )

    return {
        "row_count": row_count,
        "headers": headers,
    }


def inspect_json(
    path: Path,
) -> dict[str, Any]:
    value = read_json(path)

    return {
        "top_level_keys": sorted(
            str(key)
            for key in value.keys()
        ),
        "status": value.get("status"),
        "all_checks_passed": (
            value.get(
                "all_checks_passed"
            )
        ),
    }


def file_record(
    source_id: str,
    path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_id": source_id,
        "path": str(path),
        "relative_path": (
            path.relative_to(ROOT)
            .as_posix()
        ),
        "suffix": path.suffix.lower(),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }

    if path.suffix.lower() == ".csv":
        record["schema"] = inspect_csv(
            path
        )
    else:
        record["schema"] = inspect_json(
            path
        )

    return record


def normalize_header_token(
    value: str,
) -> str:
    return "".join(
        character
        for character in value.lower()
        if character.isalnum()
    )


def header_group_present(
    headers: list[str],
    group: tuple[str, ...],
) -> bool:
    normalized_headers = [
        normalize_header_token(
            header
        )
        for header in headers
    ]

    normalized_tokens = [
        normalize_header_token(
            token
        )
        for token in group
    ]

    return any(
        any(
            token in header
            for token in normalized_tokens
        )
        for header in normalized_headers
    )


for path in (
    DISCOVERY_PATH,
    ASSET_PLAN_PATH,
    ASSET_PLAN_LOCK_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_BINDING_JSON,
    OUTPUT_BINDING_CSV,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 exact-source binding output "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

discovery = read_json(
    DISCOVERY_PATH
)

asset_plan = read_json(
    ASSET_PLAN_PATH
)

asset_plan_lock = read_json(
    ASSET_PLAN_LOCK_PATH
)

entry_checks = {
    "discovery_completed": (
        discovery.get("status")
        == "completed"
        and discovery.get(
            "ready_for_exact_source_binding"
        )
        is True
        and discovery.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_locked": (
        asset_plan.get("status")
        == "locked"
        and asset_plan.get(
            "ready_for_asset_generation"
        )
        is True
        and asset_plan.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_lock_valid": (
        asset_plan_lock.get("status")
        == "locked"
        and asset_plan_lock.get(
            "ready_for_asset_generation"
        )
        is True
        and asset_plan_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "asset_plan_hash_matches": (
        asset_plan_lock[
            "plan"
        ][
            "sha256"
        ]
        == sha256_file(
            ASSET_PLAN_PATH
        )
    ),
    "asset_binding_count_12": (
        len(ASSET_BINDINGS) == 12
    ),
}

failed_entry_checks = [
    name
    for name, passed
    in entry_checks.items()
    if not passed
]

if failed_entry_checks:
    raise RuntimeError(
        "Phase 13 exact-source binding "
        "entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

source_records = {}

for source_id, path in (
    SOURCE_PATHS.items()
):
    if not path.exists():
        raise FileNotFoundError(path)

    record = file_record(
        source_id,
        path,
    )

    if source_id in EXPECTED_CSV_ROWS:
        observed_rows = record[
            "schema"
        ][
            "row_count"
        ]

        expected_rows = (
            EXPECTED_CSV_ROWS[
                source_id
            ]
        )

        if observed_rows != expected_rows:
            raise RuntimeError(
                f"{source_id} row-count mismatch: "
                f"{observed_rows} != {expected_rows}"
            )

    if source_id in REQUIRED_HEADER_GROUPS:
        headers = record[
            "schema"
        ][
            "headers"
        ]

        missing_groups = [
            group
            for group in (
                REQUIRED_HEADER_GROUPS[
                    source_id
                ]
            )
            if not header_group_present(
                headers,
                group,
            )
        ]

        if missing_groups:
            raise RuntimeError(
                f"{source_id} missing required "
                f"header groups: {missing_groups}"
            )

    source_records[
        source_id
    ] = record

planned_asset_ids = {
    asset["asset_id"]
    for asset in asset_plan[
        "assets"
    ]
}

if set(
    ASSET_BINDINGS.keys()
) != planned_asset_ids:
    raise RuntimeError(
        "Exact binding asset IDs do not match "
        "the locked asset plan."
    )

binding_rows: list[
    dict[str, Any]
] = []

resolved_assets: list[
    dict[str, Any]
] = []

for asset_order, asset in enumerate(
    asset_plan[
        "assets"
    ],
    start=1,
):
    asset_id = asset[
        "asset_id"
    ]

    source_ids = (
        ASSET_BINDINGS[
            asset_id
        ]
    )

    bound_sources = [
        source_records[
            source_id
        ]
        for source_id in source_ids
    ]

    resolved_assets.append(
        {
            "asset_order": asset_order,
            "asset_id": asset_id,
            "asset_type": asset[
                "asset_type"
            ],
            "title": asset["title"],
            "source_ids": source_ids,
            "bound_sources": (
                bound_sources
            ),
            "binding_status": "locked",
            "model_inference_required": (
                False
            ),
            "test_array_access_required": (
                False
            ),
            "source_mutation_allowed": (
                False
            ),
        }
    )

    for source_order, record in enumerate(
        bound_sources,
        start=1,
    ):
        binding_rows.append(
            {
                "asset_order": asset_order,
                "asset_id": asset_id,
                "asset_type": asset[
                    "asset_type"
                ],
                "source_order": source_order,
                "source_id": record[
                    "source_id"
                ],
                "relative_path": record[
                    "relative_path"
                ],
                "suffix": record[
                    "suffix"
                ],
                "size_bytes": record[
                    "size_bytes"
                ],
                "sha256": record[
                    "sha256"
                ],
                "binding_status": (
                    "locked"
                ),
            }
        )

binding = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "exact_source_binding"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "phase12_external_dataset": (
        "deferred_optional"
    ),
    "asset_count": 12,
    "unique_source_count": len(
        source_records
    ),
    "binding_row_count": len(
        binding_rows
    ),
    "assets": resolved_assets,
    "source_registry": (
        source_records
    ),
    "binding_rules": {
        "only_locked_scientific_outputs": (
            True
        ),
        "smoke_test_outputs_excluded_from_results": (
            True
        ),
        "device_level_lodo_matrix_used": (
            True
        ),
        "verified_phase7_outputs_used": (
            True
        ),
        "no_model_inference": True,
        "no_test_array_access": True,
        "no_source_mutation": True,
    },
    "source_discovery": {
        "path": str(
            DISCOVERY_PATH
        ),
        "sha256": sha256_file(
            DISCOVERY_PATH
        ),
    },
    "asset_plan": {
        "path": str(
            ASSET_PLAN_PATH
        ),
        "sha256": sha256_file(
            ASSET_PLAN_PATH
        ),
    },
    "asset_generation_performed": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_BINDING_JSON,
    binding,
)

write_csv(
    OUTPUT_BINDING_CSV,
    binding_rows,
    fieldnames=list(
        binding_rows[0].keys()
    ),
)

preflight_checks = {
    "entry_checks_passed": all(
        entry_checks.values()
    ),
    "all_assets_bound": (
        len(resolved_assets) == 12
    ),
    "all_bindings_locked": all(
        asset[
            "binding_status"
        ]
        == "locked"
        for asset in resolved_assets
    ),
    "all_assets_have_sources": all(
        len(
            asset[
                "bound_sources"
            ]
        )
        > 0
        for asset in resolved_assets
    ),
    "lodo_uses_27_row_matrix": (
        source_records[
            "lodo_device_model_matrix"
        ][
            "schema"
        ][
            "row_count"
        ]
        == 27
    ),
    "statistics_use_16_omnibus_rows": (
        source_records[
            "phase7_omnibus"
        ][
            "schema"
        ][
            "row_count"
        ]
        == 16
    ),
    "statistics_use_160_posthoc_rows": (
        source_records[
            "phase7_posthoc_verified"
        ][
            "schema"
        ][
            "row_count"
        ]
        == 160
    ),
    "smoke_results_not_bound": all(
        "smoke"
        not in row[
            "relative_path"
        ].lower()
        for row in binding_rows
    ),
    "generation_not_performed": (
        binding[
            "asset_generation_performed"
        ]
        is False
    ),
    "model_inference_not_performed": (
        binding[
            "model_inference_performed"
        ]
        is False
    ),
    "test_arrays_not_accessed": (
        binding[
            "test_arrays_accessed"
        ]
        is False
    ),
    "source_files_not_mutated": (
        binding[
            "source_files_mutated"
        ]
        is False
    ),
}

failed_preflight_checks = [
    name
    for name, passed
    in preflight_checks.items()
    if not passed
]

if failed_preflight_checks:
    raise RuntimeError(
        "Phase 13 exact-source binding "
        "preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 13,
    "artifact_name": (
        "exact_source_binding_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "binding": {
        "path": str(
            OUTPUT_BINDING_JSON
        ),
        "sha256": sha256_file(
            OUTPUT_BINDING_JSON
        ),
    },
    "binding_csv": {
        "path": str(
            OUTPUT_BINDING_CSV
        ),
        "sha256": sha256_file(
            OUTPUT_BINDING_CSV
        ),
    },
    "asset_count": 12,
    "unique_source_count": len(
        source_records
    ),
    "binding_row_count": len(
        binding_rows
    ),
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "exact_source_binding"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "binding": {
        "path": str(
            OUTPUT_BINDING_JSON
        ),
        "sha256": sha256_file(
            OUTPUT_BINDING_JSON
        ),
    },
    "binding_csv": {
        "path": str(
            OUTPUT_BINDING_CSV
        ),
        "sha256": sha256_file(
            OUTPUT_BINDING_CSV
        ),
    },
    "preflight": {
        "path": str(
            OUTPUT_PREFLIGHT
        ),
        "sha256": sha256_file(
            OUTPUT_PREFLIGHT
        ),
    },
    "asset_count": 12,
    "unique_source_count": len(
        source_records
    ),
    "binding_row_count": len(
        binding_rows
    ),
    "asset_generation_performed": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 13,
    "artifact_name": (
        "exact_source_binding"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        {
            "path": record[
                "path"
            ],
            "size_bytes": record[
                "size_bytes"
            ],
            "sha256": record[
                "sha256"
            ],
        }
        for record in sorted(
            source_records.values(),
            key=lambda item: (
                item["source_id"]
            ),
        )
    ],
    "generated_artifacts": [
        {
            "path": str(
                OUTPUT_BINDING_JSON
            ),
            "size_bytes": int(
                OUTPUT_BINDING_JSON
                .stat()
                .st_size
            ),
            "sha256": sha256_file(
                OUTPUT_BINDING_JSON
            ),
        },
        {
            "path": str(
                OUTPUT_BINDING_CSV
            ),
            "size_bytes": int(
                OUTPUT_BINDING_CSV
                .stat()
                .st_size
            ),
            "sha256": sha256_file(
                OUTPUT_BINDING_CSV
            ),
        },
        {
            "path": str(
                OUTPUT_PREFLIGHT
            ),
            "size_bytes": int(
                OUTPUT_PREFLIGHT
                .stat()
                .st_size
            ),
            "sha256": sha256_file(
                OUTPUT_PREFLIGHT
            ),
        },
        {
            "path": str(
                OUTPUT_LOCK
            ),
            "size_bytes": int(
                OUTPUT_LOCK
                .stat()
                .st_size
            ),
            "sha256": sha256_file(
                OUTPUT_LOCK
            ),
        },
    ],
    "asset_count": 12,
    "unique_source_count": len(
        source_records
    ),
    "binding_row_count": len(
        binding_rows
    ),
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 96)
print("PHASE 13 EXACT SOURCE BINDING")
print("=" * 96)
print(
    "Assets bound                    : 12"
)
print(
    "Unique scientific sources       : "
    f"{len(source_records)}"
)
print(
    "Asset-source binding rows       : "
    f"{len(binding_rows)}"
)
print(
    "LODO source                     : "
    "early_lodo_device_model_matrix_v2.csv"
)
print(
    "LODO rows                       : 27"
)
print(
    "Phase 7 omnibus source          : "
    "phase7_statistical_omnibus_results_v3_2.csv"
)
print(
    "Phase 7 omnibus rows            : 16"
)
print(
    "Phase 7 post-hoc source         : "
    "phase7_statistical_posthoc_verified_v3_2.csv"
)
print(
    "Phase 7 post-hoc rows           : 160"
)
print(
    "Smoke-test result sources used  : False"
)
print()

for asset in resolved_assets:
    print(
        f"{asset['asset_order']:02d}. "
        f"{asset['asset_id']:<38} | "
        f"sources={len(asset['source_ids'])}"
    )

print()
print("=" * 96)
print("EXACT SOURCE BINDING PREFLIGHT")
print("=" * 96)
print(
    "All 12 assets bound             : True"
)
print(
    "Bindings locked                 : True"
)
print(
    "Model inference performed       : False"
)
print(
    "Test arrays accessed            : False"
)
print(
    "Source files mutated            : False"
)
print(
    "Asset generation performed      : False"
)
print(
    "Ready for asset generation      : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 13 EXACT SOURCE BINDING LOCKED"
)
