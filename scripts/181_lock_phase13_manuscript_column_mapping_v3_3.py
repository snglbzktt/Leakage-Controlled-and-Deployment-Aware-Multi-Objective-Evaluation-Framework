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
CONFIG = ROOT / "configs" / "protocols"

BINDING_PATH = (
    CONFIG
    / "phase13_exact_source_binding_v3_3.json"
)

BINDING_LOCK_PATH = (
    AUDIT
    / "phase13_exact_source_binding_locked_v3_3.json"
)

FIELD_INSPECTION_PATH = (
    AUDIT
    / "phase13_bound_source_field_inspection_v3_3.json"
)

OUTPUT_MAPPING_JSON = (
    CONFIG
    / "phase13_manuscript_column_mapping_v3_3.json"
)

OUTPUT_MAPPING_CSV = (
    AUDIT
    / "phase13_manuscript_column_mapping_v3_3.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase13_manuscript_column_mapping_preflight_v3_3.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase13_manuscript_column_mapping_locked_v3_3.json"
)

PROTOCOL_VERSION = (
    "phase13_manuscript_column_mapping_v3_3"
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


COLUMN_MAPPING: dict[str, list[dict[str, Any]]] = {
    "figure_01_methodology_overview": [
        {
            "source_id": "lodo_global_summary",
            "fields": [
                "scope.device_count",
                "scope.model_count",
                "scope.scientific_run_count",
                "gate.passed",
            ],
            "role": "LODO scope and gate",
        },
        {
            "source_id": "phase5_closure",
            "fields": [
                "architecture_count",
                "variant_count",
                "seed_count",
                "final_run_count",
                "group_count",
            ],
            "role": "Compression experiment scope",
        },
        {
            "source_id": "phase6_summary",
            "fields": [
                "run_count",
                "runtime_policy.cpu_threads",
                "runtime_policy.interop_threads",
                "data_access.validation_data_access",
                "data_access.test_data_access",
            ],
            "role": "Deployment benchmark protocol",
        },
        {
            "source_id": "phase8_final_model_lock",
            "fields": [
                "selected_model_family.candidate_id",
                "selected_model_family.representation",
                "final_model_family_selected",
            ],
            "role": "Model-family selection",
        },
        {
            "source_id": "phase9_artifact_lock",
            "fields": [
                "canonical_seed",
                "source_bundle_checkpoint_exact",
                "serialization_roundtrip_exact",
                "final_deployment_artifact_locked",
            ],
            "role": "Final deployment artifact",
        },
        {
            "source_id": "phase10_release_lock",
            "fields": [
                "payload_file_count",
                "archive_file_count",
                "deterministic_rebuild_exact",
                "release_locked",
            ],
            "role": "Deterministic release",
        },
        {
            "source_id": "phase11_lock",
            "fields": [
                "model_count",
                "condition_count",
                "locked_evaluation_count",
                "phase11_locked",
            ],
            "role": "Robustness evaluation scope",
        },
    ],
    "figure_02_lodo_generalization": [
        {
            "source_id": "lodo_device_model_matrix",
            "fields": [
                "held_out_device",
                "model_id",
                "test_fingerprint_macro_f1",
                "benign_fnr",
                "gafgyt_fnr",
                "mirai_fnr",
                "local_severe_signal",
            ],
            "role": "Per-device LODO performance",
        },
        {
            "source_id": "lodo_global_summary",
            "fields": [
                "model_ranking",
                "global_statistics.mean_cell_test_macro_f1",
                "global_statistics.minimum_cell_test_macro_f1",
                "gate.status",
            ],
            "role": "Global LODO summary",
        },
    ],
    "figure_03_compression_tradeoff": [
        {
            "source_id": "phase5_group_summary",
            "fields": [
                "architecture",
                "variant",
                "representation",
                "mean_test_fingerprint_macro_f1",
                "mean_state_size_ratio_vs_float",
                "fragility_triggered_in_any_run",
            ],
            "role": "Accuracy and compression characteristics",
        },
        {
            "source_id": "phase6_group_summary",
            "fields": [
                "architecture",
                "variant",
                "representation",
                "mean_serialized_state_dict_bytes",
                "mean_peak_inference_RSS_delta_bytes",
                "mean_batch_1_median_ms",
                "mean_batch_32_samples_per_second",
            ],
            "role": "Deployment measurements",
        },
    ],
    "figure_04_pareto_selection": [
        {
            "source_id": "phase8_input_matrix",
            "fields": [
                "architecture",
                "variant",
                "candidate_role",
                "representation",
                "mean_test_fingerprint_macro_f1",
                "mean_serialized_state_dict_bytes",
                "mean_batch_1_median_ms",
                "mean_batch_32_samples_per_second",
                "fragility_triggered_in_any_run",
            ],
            "role": "Candidate decision space",
        },
        {
            "source_id": "phase8_pareto_results",
            "fields": [
                "profile",
                "primary_profile",
                "architecture",
                "variant",
                "eligible",
                "Pareto_nondominated",
                "dominated_by_count",
                "deployment_improvement_count",
                "failed_gates",
            ],
            "role": "Pareto eligibility and dominance",
        },
        {
            "source_id": "phase8_selection_trace",
            "fields": [
                "profile",
                "primary_profile",
                "eligible_candidate_count",
                "Pareto_candidate_count",
                "selected_architecture",
                "selected_variant",
                "selected_representation",
                "selection_rule",
            ],
            "role": "Profile-specific selection",
        },
        {
            "source_id": "phase8_final_model_lock",
            "fields": [
                "primary_profile",
                "selected_model_family.candidate_id",
                "sensitivity_profile_selections",
                "selection_stable_across_profiles",
            ],
            "role": "Locked final family",
        },
    ],
    "figure_05_input_robustness": [
        {
            "source_id": "phase11_all_runs",
            "fields": [
                "model_id",
                "condition_id",
                "condition_family",
                "severity",
                "fingerprint_macro_f1",
                "macro_f1_drop_from_model_clean",
                "macro_f1_retention_ratio",
                "severe_fragility",
                "material_sensitivity",
            ],
            "role": "Condition-level robustness",
        },
        {
            "source_id": "phase11_summary",
            "fields": [
                "model_summaries",
                "severe_fragility_evaluation_count",
                "material_sensitivity_evaluation_count",
            ],
            "role": "Robustness summary",
        },
    ],
    "table_01_dataset_protocol": [
        {
            "source_id": "lodo_global_summary",
            "fields": [
                "scope.device_count",
                "scope.model_count",
                "scope.scientific_run_count",
                "scope.two_class_test_devices",
                "locked_thresholds",
                "interpretation.class_coverage_note",
            ],
            "role": "LODO dataset and evaluation protocol",
        },
        {
            "source_id": "phase4_summary",
            "fields": [
                "model_count",
                "total_run_count",
                "interpretation_policy",
                "scientific_caveats",
            ],
            "role": "Classical baseline protocol",
        },
        {
            "source_id": "phase5_closure",
            "fields": [
                "architecture_count",
                "variant_count",
                "seed_count",
                "final_run_count",
                "test_evaluation_count_total",
                "final_checks",
            ],
            "role": "Compression protocol",
        },
    ],
    "table_02_classical_baselines": [
        {
            "source_id": "phase4_model_summary",
            "fields": [
                "descriptive_macro_f1_rank",
                "model_id",
                "display_name",
                "run_count",
                "seed_policy",
                "input_space",
                "mean_test_fingerprint_macro_f1",
                "std_test_fingerprint_macro_f1",
                "mean_test_raw_weighted_macro_f1",
                "mean_test_gafgyt_fnr",
                "mean_test_mirai_fnr",
                "scientific_status",
                "important_caveat",
            ],
            "role": "Baseline performance table",
        },
    ],
    "table_03_compression_results": [
        {
            "source_id": "phase5_group_summary",
            "fields": [
                "architecture",
                "variant",
                "representation",
                "run_count",
                "mean_test_fingerprint_macro_f1",
                "std_test_fingerprint_macro_f1",
                "mean_test_raw_weighted_macro_f1",
                "mean_state_size_ratio_vs_float",
                "fragility_triggered_in_any_run",
            ],
            "role": "Grouped compression results",
        },
        {
            "source_id": "phase5_closure",
            "fields": [
                "final_run_count",
                "group_count",
                "fragility_groups",
                "final_checks",
            ],
            "role": "Compression closure notes",
        },
    ],
    "table_04_deployment_benchmarks": [
        {
            "source_id": "phase6_group_summary",
            "fields": [
                "architecture",
                "variant",
                "representation",
                "run_count",
                "mean_serialized_state_dict_bytes",
                "mean_peak_inference_RSS_delta_bytes",
                "mean_batch_1_median_ms",
                "std_batch_1_median_ms",
                "mean_batch_1_p95_ms",
                "mean_batch_32_samples_per_second",
                "validation_data_access",
                "test_data_access",
            ],
            "role": "Grouped deployment measurements",
        },
        {
            "source_id": "phase6_summary",
            "fields": [
                "runtime_policy",
                "data_access",
                "global_checks",
            ],
            "role": "Benchmark execution protocol",
        },
    ],
    "table_05_statistical_analysis": [
        {
            "source_id": "phase7_omnibus",
            "fields": [
                "endpoint_class",
                "architecture",
                "metric",
                "block_count",
                "variant_count",
                "friedman_statistic",
                "kendalls_W",
                "monte_carlo_p_plus_one",
                "monte_carlo_reject_nominal",
            ],
            "role": "Omnibus statistical results",
        },
        {
            "source_id": "phase7_posthoc_verified",
            "fields": [
                "endpoint_class",
                "architecture",
                "metric",
                "reference_variant",
                "candidate_variant",
                "mean_improvement_delta",
                "paired_rank_biserial_improvement",
                "exact_two_sided_p_raw",
                "Holm_adjusted_p",
                "Holm_reject_alpha_0_05",
            ],
            "role": "Verified post-hoc comparisons",
        },
        {
            "source_id": "phase7_summary",
            "fields": [
                "omnibus_analysis_count",
                "posthoc_comparison_count",
                "confirmatory_omnibus_rejections_nominal",
                "exploratory_omnibus_rejections_nominal",
                "confirmatory_posthoc_Holm_rejections",
                "exploratory_posthoc_Holm_rejections",
                "minimum_raw_posthoc_p",
                "minimum_Holm_adjusted_p",
                "small_sample_interpretation",
            ],
            "role": "Statistical interpretation",
        },
    ],
    "table_06_robustness_results": [
        {
            "source_id": "phase11_all_runs",
            "fields": [
                "model_id",
                "condition_id",
                "condition_family",
                "severity",
                "fingerprint_macro_f1",
                "raw_weighted_macro_f1",
                "macro_f1_drop_from_model_clean",
                "severe_fragility",
                "material_sensitivity",
            ],
            "role": "All robustness evaluations",
        },
        {
            "source_id": "phase11_summary",
            "fields": [
                "model_summaries",
                "severe_fragility_evaluation_count",
                "material_sensitivity_evaluation_count",
                "model_retraining_performed",
                "test_used_for_model_selection",
                "final_model_changed",
            ],
            "role": "Robustness interpretation",
        },
    ],
    "table_07_reproducibility_release": [
        {
            "source_id": "phase8_final_model_lock",
            "fields": [
                "selected_model_family",
                "sensitivity_profile_selections",
                "weighted_score_used",
                "pairwise_p_value_used_as_gate",
                "single_seed_checkpoint_selected",
            ],
            "role": "Selection reproducibility",
        },
        {
            "source_id": "phase9_artifact_lock",
            "fields": [
                "model_family",
                "canonical_seed",
                "class_labels_in_output_index_order",
                "source_bundle_checkpoint_exact",
                "source_bundle_outputs_exact",
                "serialization_roundtrip_exact",
                "validation_data_access",
                "test_data_access",
            ],
            "role": "Artifact reproducibility",
        },
        {
            "source_id": "phase10_release_lock",
            "fields": [
                "release_name",
                "payload_file_count",
                "archive_file_count",
                "release_directory_tree_sha256",
                "zip_archive_sha256",
                "deterministic_rebuild_exact",
                "raw_dataset_included",
                "processed_dataset_included",
                "nonselected_checkpoints_included",
            ],
            "role": "Release integrity",
        },
        {
            "source_id": "phase11_lock",
            "fields": [
                "model_retraining_performed",
                "test_used_for_model_selection",
                "final_model_changed",
                "phase11_locked",
            ],
            "role": "Post-selection robustness integrity",
        },
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
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    fieldnames = list(
        rows[0].keys()
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


def json_path_exists(
    value: Any,
    path: str,
) -> bool:
    current = value

    for token in path.split("."):
        if isinstance(current, dict):
            if token not in current:
                return False

            current = current[token]
        else:
            return False

    return True


for required_path in (
    BINDING_PATH,
    BINDING_LOCK_PATH,
    FIELD_INSPECTION_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(
            required_path
        )

for output_path in (
    OUTPUT_MAPPING_JSON,
    OUTPUT_MAPPING_CSV,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 13 manuscript column mapping "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

binding = read_json(
    BINDING_PATH
)

binding_lock = read_json(
    BINDING_LOCK_PATH
)

field_inspection = read_json(
    FIELD_INSPECTION_PATH
)

entry_checks = {
    "binding_locked": (
        binding.get("status")
        == "locked"
        and binding.get(
            "ready_for_asset_generation"
        )
        is True
        and binding.get(
            "all_checks_passed"
        )
        is True
    ),
    "binding_lock_valid": (
        binding_lock.get("status")
        == "locked"
        and binding_lock.get(
            "ready_for_asset_generation"
        )
        is True
        and binding_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "binding_hash_matches": (
        binding_lock[
            "binding"
        ][
            "sha256"
        ]
        == sha256_file(
            BINDING_PATH
        )
    ),
    "field_inspection_completed": (
        field_inspection.get("status")
        == "completed"
        and field_inspection.get(
            "ready_for_column_mapping"
        )
        is True
        and field_inspection.get(
            "all_checks_passed"
        )
        is True
    ),
    "mapping_asset_count_12": (
        len(COLUMN_MAPPING) == 12
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
        "Column-mapping entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

source_registry = binding[
    "source_registry"
]

planned_asset_ids = {
    asset["asset_id"]
    for asset in binding[
        "assets"
    ]
}

if set(
    COLUMN_MAPPING
) != planned_asset_ids:
    raise RuntimeError(
        "Column-mapping asset IDs do not "
        "match the locked asset set."
    )

validated_assets = []
flat_rows = []
validation_errors = []

for asset in binding["assets"]:
    asset_id = asset[
        "asset_id"
    ]

    mapping_entries = (
        COLUMN_MAPPING[
            asset_id
        ]
    )

    validated_entries = []

    for entry in mapping_entries:
        source_id = entry[
            "source_id"
        ]

        if source_id not in source_registry:
            validation_errors.append(
                {
                    "asset_id": asset_id,
                    "source_id": source_id,
                    "error": (
                        "source_id_not_in_registry"
                    ),
                }
            )
            continue

        source_record = source_registry[
            source_id
        ]

        source_path = Path(
            source_record[
                "path"
            ]
        )

        if not source_path.exists():
            validation_errors.append(
                {
                    "asset_id": asset_id,
                    "source_id": source_id,
                    "error": (
                        "source_file_missing"
                    ),
                }
            )
            continue

        if sha256_file(
            source_path
        ) != source_record[
            "sha256"
        ]:
            validation_errors.append(
                {
                    "asset_id": asset_id,
                    "source_id": source_id,
                    "error": (
                        "source_hash_changed"
                    ),
                }
            )
            continue

        suffix = source_path.suffix.lower()

        if suffix == ".csv":
            with source_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                reader = csv.DictReader(
                    handle
                )

                headers = set(
                    reader.fieldnames
                    or []
                )

            missing_fields = [
                field
                for field in entry[
                    "fields"
                ]
                if field not in headers
            ]

        elif suffix == ".json":
            source_json = read_json(
                source_path
            )

            missing_fields = [
                field
                for field in entry[
                    "fields"
                ]
                if not json_path_exists(
                    source_json,
                    field,
                )
            ]

        else:
            missing_fields = list(
                entry["fields"]
            )

        if missing_fields:
            validation_errors.append(
                {
                    "asset_id": asset_id,
                    "source_id": source_id,
                    "error": (
                        "required_fields_missing"
                    ),
                    "missing_fields": (
                        missing_fields
                    ),
                }
            )
            continue

        validated_entry = {
            "source_id": source_id,
            "relative_path": (
                source_record[
                    "relative_path"
                ]
            ),
            "sha256": (
                source_record[
                    "sha256"
                ]
            ),
            "source_type": suffix[
                1:
            ],
            "role": entry["role"],
            "fields": entry[
                "fields"
            ],
            "field_count": len(
                entry[
                    "fields"
                ]
            ),
            "validation_status": (
                "passed"
            ),
        }

        validated_entries.append(
            validated_entry
        )

        for field_order, field in enumerate(
            entry["fields"],
            start=1,
        ):
            flat_rows.append(
                {
                    "asset_order": asset[
                        "asset_order"
                    ],
                    "asset_id": asset_id,
                    "asset_type": asset[
                        "asset_type"
                    ],
                    "source_id": source_id,
                    "source_type": suffix[
                        1:
                    ],
                    "field_order": (
                        field_order
                    ),
                    "field": field,
                    "role": entry[
                        "role"
                    ],
                    "relative_path": (
                        source_record[
                            "relative_path"
                        ]
                    ),
                    "validation_status": (
                        "passed"
                    ),
                }
            )

    validated_assets.append(
        {
            "asset_order": asset[
                "asset_order"
            ],
            "asset_id": asset_id,
            "asset_type": asset[
                "asset_type"
            ],
            "title": asset[
                "title"
            ],
            "mapping_entries": (
                validated_entries
            ),
            "source_count": len(
                validated_entries
            ),
            "field_count": sum(
                entry[
                    "field_count"
                ]
                for entry in (
                    validated_entries
                )
            ),
            "mapping_status": (
                "validated"
            ),
        }
    )

if validation_errors:
    raise RuntimeError(
        "Column mapping validation failed:\n"
        + json.dumps(
            validation_errors,
            indent=2,
            ensure_ascii=True,
        )
    )

mapping = {
    "status": "validated",
    "phase": 13,
    "artifact_name": (
        "manuscript_column_mapping"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "validated_at_utc": utc_now(),
    "asset_count": len(
        validated_assets
    ),
    "mapping_row_count": len(
        flat_rows
    ),
    "assets": validated_assets,
    "binding": {
        "path": str(
            BINDING_PATH
        ),
        "sha256": sha256_file(
            BINDING_PATH
        ),
    },
    "field_inspection": {
        "path": str(
            FIELD_INSPECTION_PATH
        ),
        "sha256": sha256_file(
            FIELD_INSPECTION_PATH
        ),
    },
    "mapping_rules": {
        "exact_field_names_only": True,
        "locked_sources_only": True,
        "no_inferred_columns": True,
        "no_model_inference": True,
        "no_test_array_access": True,
        "no_source_mutation": True,
    },
    "asset_generation_performed": False,
    "model_inference_performed": False,
    "test_arrays_accessed": False,
    "source_files_mutated": False,
    "ready_for_asset_generation": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_MAPPING_JSON,
    mapping,
)

write_csv(
    OUTPUT_MAPPING_CSV,
    flat_rows,
)

preflight_checks = {
    "all_12_assets_mapped": (
        len(validated_assets) == 12
    ),
    "all_assets_have_sources": all(
        asset[
            "source_count"
        ]
        > 0
        for asset in (
            validated_assets
        )
    ),
    "all_assets_have_fields": all(
        asset[
            "field_count"
        ]
        > 0
        for asset in (
            validated_assets
        )
    ),
    "all_mapping_entries_validated": (
        len(validation_errors) == 0
    ),
    "smoke_test_sources_absent": all(
        "smoke"
        not in row[
            "relative_path"
        ].lower()
        for row in flat_rows
    ),
    "model_inference_not_performed": (
        mapping[
            "model_inference_performed"
        ]
        is False
    ),
    "test_arrays_not_accessed": (
        mapping[
            "test_arrays_accessed"
        ]
        is False
    ),
    "source_files_not_mutated": (
        mapping[
            "source_files_mutated"
        ]
        is False
    ),
    "asset_generation_not_performed": (
        mapping[
            "asset_generation_performed"
        ]
        is False
    ),
}

if not all(
    preflight_checks.values()
):
    failed = [
        name
        for name, passed
        in preflight_checks.items()
        if not passed
    ]

    raise RuntimeError(
        "Column-mapping preflight failed: "
        + ", ".join(
            failed
        )
    )

preflight = {
    "status": "passed",
    "phase": 13,
    "artifact_name": (
        "manuscript_column_mapping_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": entry_checks,
    "preflight_checks": (
        preflight_checks
    ),
    "mapping": {
        "path": str(
            OUTPUT_MAPPING_JSON
        ),
        "sha256": sha256_file(
            OUTPUT_MAPPING_JSON
        ),
    },
    "mapping_csv": {
        "path": str(
            OUTPUT_MAPPING_CSV
        ),
        "sha256": sha256_file(
            OUTPUT_MAPPING_CSV
        ),
    },
    "asset_count": 12,
    "mapping_row_count": len(
        flat_rows
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
        "manuscript_column_mapping"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "mapping": {
        "path": str(
            OUTPUT_MAPPING_JSON
        ),
        "sha256": sha256_file(
            OUTPUT_MAPPING_JSON
        ),
    },
    "mapping_csv": {
        "path": str(
            OUTPUT_MAPPING_CSV
        ),
        "sha256": sha256_file(
            OUTPUT_MAPPING_CSV
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
    "mapping_row_count": len(
        flat_rows
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

print("=" * 100)
print("PHASE 13 MANUSCRIPT COLUMN MAPPING")
print("=" * 100)
print(
    "Assets mapped                   : "
    f"{len(validated_assets)}"
)
print(
    "Validated field mappings        : "
    f"{len(flat_rows)}"
)
print(
    "Smoke-test sources used         : False"
)
print()

for asset in validated_assets:
    print(
        f"{asset['asset_order']:02d}. "
        f"{asset['asset_id']:<38} | "
        f"sources={asset['source_count']:<2} | "
        f"fields={asset['field_count']}"
    )

print()
print("=" * 100)
print("COLUMN MAPPING PREFLIGHT")
print("=" * 100)
print(
    "All 12 assets mapped            : True"
)
print(
    "All fields found exactly        : True"
)
print(
    "Only locked sources used        : True"
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
    "PHASE 13 MANUSCRIPT COLUMN MAPPING LOCKED"
)
