from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

AUDIT = ROOT / "results" / "v2" / "audit"

PHASE4_COMPLETION = (
    AUDIT
    / "phase4_classical_baselines_completed_v3_2.json"
)

PHASE4_LOCK = (
    AUDIT
    / "phase4_classical_baselines_lock_manifest_v3_2.json"
)

PREFLIGHT_JSON = (
    AUDIT
    / "phase5_preflight_inventory_v3_2.json"
)

PREFLIGHT_MATRIX = (
    AUDIT
    / "phase5_planned_configuration_matrix_v3_2.csv"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

FINAL_CACHE_MANIFEST = (
    FINAL_CACHE / "manifest.json"
)

FINAL_CACHE_VERIFICATION = (
    AUDIT
    / "tabular_baseline_final_cache_verification_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

MODEL_REGISTRY = (
    ROOT
    / "models"
    / "architecture"
    / "nbaiot_model_registry_v1.json"
)

LEGACY_FP32_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_fp32_baseline_protocol_v1.json"
)

LEGACY_PRUNING_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_structured_pruning_protocol_v1.json"
)

LEGACY_QUANTIZATION_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_quantization_protocol_v1.json"
)

LEGACY_PRUNING_QAT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_pruning_qat_protocol_v1.json"
)

PROTOCOL_DIRECTORY = (
    ROOT / "configs" / "protocols"
)

OUTPUT_PROTOCOL = (
    PROTOCOL_DIRECTORY
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

OUTPUT_MATRIX = (
    AUDIT
    / "phase5_locked_configuration_matrix_v3_2.csv"
)

OUTPUT_COMPARISONS = (
    AUDIT
    / "phase5_predeclared_comparisons_v3_2.csv"
)

OUTPUT_COMPLETION = (
    AUDIT
    / "phase5_protocol_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_protocol_lock_manifest_v3_2.json"
)

PHASE5_RESULTS_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
)

PROTOCOL_VERSION = (
    "phase5_fair_budget_compression_v3_2"
)

DATA_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

VARIANTS = (
    "B0",
    "FP32-FT",
    "P25-noFT",
    "P25",
    "P50-noFT",
    "P50",
    "DQ",
    "PTQ",
    "QAT",
    "P25-QAT",
    "P50-QAT",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

EXPECTED_CONFIGURATION_COUNT = (
    len(ARCHITECTURES)
    * len(VARIANTS)
    * len(SEEDS)
)

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_ORDER = (
    "benign",
    "gafgyt",
    "mirai",
)

SPLIT_COUNTS = {
    "train": {
        "fingerprints": 1_534_583,
        "represented_raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_797,
        "represented_raw_rows": 1_059_390,
    },
    "test": {
        "fingerprints": 371_796,
        "represented_raw_rows": 1_059_393,
    },
}

BASE_LEARNING_RATES = {
    "tinyml_mlp": 0.003,
    "compact_dnn": 0.001,
}

LOW_LR_FACTOR = 0.1

B0_BUDGET = {
    "optimizer": "AdamW",
    "batch_size": 4096,
    "max_epochs": 15,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 0.0002,
    "weight_decay": 0.0001,
    "num_workers": 0,
    "loss": "CrossEntropyLoss",
    "selection_metric": (
        "validation_fingerprint_macro_f1"
    ),
    "restore_best_checkpoint": True,
}

EQUAL_UPDATE_BUDGET = {
    "optimizer": "AdamW",
    "batch_size": 4096,
    "max_epochs": 8,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.0002,
    "weight_decay": 0.0001,
    "num_workers": 0,
    "loss": "CrossEntropyLoss",
    "learning_rate_factor_from_b0": (
        LOW_LR_FACTOR
    ),
    "restore_best_checkpoint": True,
}

ALIAS_RESOLUTION = {
    "P25": "P25-FP32-FT",
    "P50": "P50-FP32-FT",
}

PRUNING_RULES = {
    "type": (
        "physical_structured_hidden_unit_pruning"
    ),
    "mask_only_pruning_allowed": False,
    "pruning_ratios": [
        0.25,
        0.50,
    ],
    "source_policy": (
        "Each ratio is generated independently "
        "from the matching architecture-seed B0 "
        "checkpoint. P50 is never derived from P25."
    ),
    "retained_width_rule": (
        "max(1, round((1-r) * original_hidden_width))"
    ),
    "preserve_output_layer": True,
    "preserve_class_count": 3,
    "compact_following_dimensions": True,
    "recompute_parameters_and_macs": True,
    "pruning_engine_policy": (
        "Use the repository physical compaction "
        "implementation after it passes dimension, "
        "logit-shape, parameter-count, and MAC tests. "
        "The implementation file hash is recorded by "
        "every run manifest."
    ),
}

QUANTIZATION_RULES = {
    "DQ": {
        "source": "B0",
        "training_updates": False,
        "quantization": (
            "dynamic_int8_linear_layers"
        ),
        "validation_selection": False,
        "phase5_test_policy": (
            "evaluate_once_after_conversion"
        ),
    },
    "PTQ": {
        "source": "B0",
        "training_updates": False,
        "quantization": (
            "static_post_training_int8"
        ),
        "calibration_source": (
            "train_only"
        ),
        "phase5_calibration_role": (
            "conversion_smoke_and_validation_only"
        ),
        "phase5_test_policy": (
            "deferred_to_phase6_calibration_sensitivity"
        ),
        "reason_for_deferral": (
            "Phase 6 must compare calibration sizes "
            "4096, 16384, 65536, and 262144 using "
            "train-only balanced samples, select on "
            "validation, and access test only for the "
            "selected setting."
        ),
    },
    "QAT": {
        "source": "B0",
        "training_updates": True,
        "quantization": (
            "quantization_aware_training_int8"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
    },
    "P25-QAT": {
        "source": (
            "same_P25_noFT_state_used_by_P25"
        ),
        "training_updates": True,
        "quantization": (
            "physical_P25_then_QAT"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
    },
    "P50-QAT": {
        "source": (
            "same_P50_noFT_state_used_by_P50"
        ),
        "training_updates": True,
        "quantization": (
            "physical_P50_then_QAT"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
    },
}

PRIMARY_COMPARISONS = (
    {
        "comparison_id": "C01",
        "left_variant": "QAT",
        "right_variant": "FP32-FT",
        "purpose": (
            "Separate fake-quantization effect "
            "from equal additional training."
        ),
        "family": "equal_budget_quantization",
        "priority": "primary",
    },
    {
        "comparison_id": "C02",
        "left_variant": "P25",
        "right_variant": "FP32-FT",
        "purpose": (
            "Estimate P25 structured-pruning "
            "effect under an equal FP32 update budget."
        ),
        "family": "equal_budget_pruning",
        "priority": "primary",
    },
    {
        "comparison_id": "C03",
        "left_variant": "P50",
        "right_variant": "FP32-FT",
        "purpose": (
            "Estimate P50 structured-pruning "
            "effect under an equal FP32 update budget."
        ),
        "family": "equal_budget_pruning",
        "priority": "primary",
    },
    {
        "comparison_id": "C04",
        "left_variant": "P25-QAT",
        "right_variant": "P25",
        "purpose": (
            "Compare QAT with FP32 fine-tuning "
            "from the identical P25-noFT source."
        ),
        "family": "equal_budget_pruning_qat",
        "priority": "primary",
    },
    {
        "comparison_id": "C05",
        "left_variant": "P50-QAT",
        "right_variant": "P50",
        "purpose": (
            "Compare QAT with FP32 fine-tuning "
            "from the identical P50-noFT source."
        ),
        "family": "equal_budget_pruning_qat",
        "priority": "primary",
    },
    {
        "comparison_id": "C06",
        "left_variant": "P25-noFT",
        "right_variant": "P25",
        "purpose": (
            "Measure recovery attributable to "
            "post-pruning FP32 fine-tuning."
        ),
        "family": "fine_tuning_ablation",
        "priority": "primary",
    },
    {
        "comparison_id": "C07",
        "left_variant": "P50-noFT",
        "right_variant": "P50",
        "purpose": (
            "Measure recovery attributable to "
            "post-pruning FP32 fine-tuning."
        ),
        "family": "fine_tuning_ablation",
        "priority": "primary",
    },
    {
        "comparison_id": "C08",
        "left_variant": "DQ",
        "right_variant": "B0",
        "purpose": (
            "Measure standalone dynamic "
            "quantization effect."
        ),
        "family": "standalone_quantization",
        "priority": "secondary_locked",
    },
    {
        "comparison_id": "C09",
        "left_variant": "PTQ",
        "right_variant": "B0",
        "purpose": (
            "Measure standalone PTQ effect only "
            "after Phase 6 calibration selection."
        ),
        "family": "standalone_quantization",
        "priority": "secondary_locked_deferred",
    },
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
    chunk_size: int = (
        8 * 1024 * 1024
    ),
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            chunk_size
        ):
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


def file_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


def optional_file_record(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(path),
    }


required_paths = (
    PHASE4_COMPLETION,
    PHASE4_LOCK,
    PREFLIGHT_JSON,
    PREFLIGHT_MATRIX,
    FINAL_CACHE,
    FINAL_CACHE_MANIFEST,
    FINAL_CACHE_VERIFICATION,
    MODEL_SOURCE,
    MODEL_REGISTRY,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

output_paths = (
    OUTPUT_PROTOCOL,
    OUTPUT_MATRIX,
    OUTPUT_COMPARISONS,
    OUTPUT_COMPLETION,
    OUTPUT_LOCK_MANIFEST,
)

existing_outputs = [
    path
    for path in output_paths
    if path.exists()
]

if existing_outputs:
    raise FileExistsError(
        "Phase 5 protocol artifacts already "
        "exist; refusing to overwrite: "
        + ", ".join(
            str(path)
            for path in existing_outputs
        )
    )

if (
    PHASE5_RESULTS_ROOT.exists()
    and any(
        PHASE5_RESULTS_ROOT.iterdir()
    )
):
    raise RuntimeError(
        "Phase 5 result directory is not empty. "
        "Protocol must be locked before any run."
    )

phase4_completion = read_json(
    PHASE4_COMPLETION
)

phase4_lock = read_json(
    PHASE4_LOCK
)

preflight = read_json(
    PREFLIGHT_JSON
)

cache_manifest = read_json(
    FINAL_CACHE_MANIFEST
)

cache_verification = read_json(
    FINAL_CACHE_VERIFICATION
)

preflight_rows = read_csv(
    PREFLIGHT_MATRIX
)

preflight_architectures = {
    row["architecture"]
    for row in preflight_rows
}

preflight_variants = {
    row["variant"]
    for row in preflight_rows
}

preflight_seeds = {
    int(row["seed"])
    for row in preflight_rows
}

preflight_checks = {
    "phase4_completion_locked": (
        phase4_completion.get("status")
        == "locked"
    ),
    "phase4_completion_checks_passed": (
        phase4_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase4_ready_for_phase5": (
        phase4_completion.get(
            "next_phase_ready"
        )
        is True
        and int(
            phase4_completion.get(
                "next_phase"
            )
        )
        == 5
    ),
    "phase4_lock_manifest_locked": (
        phase4_lock.get("status")
        == "locked"
        and phase4_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "preflight_passed": (
        preflight.get("status")
        == "passed"
        and preflight.get(
            "all_checks_passed"
        )
        is True
    ),
    "preflight_phase_is_five": (
        int(preflight.get("phase"))
        == 5
    ),
    "preflight_matrix_count_is_110": (
        len(preflight_rows)
        == EXPECTED_CONFIGURATION_COUNT
    ),
    "preflight_architecture_set_matches": (
        preflight_architectures
        == set(ARCHITECTURES)
    ),
    "preflight_variant_set_matches": (
        preflight_variants
        == set(VARIANTS)
    ),
    "preflight_seed_set_matches": (
        preflight_seeds
        == set(SEEDS)
    ),
    "cache_manifest_completed": (
        cache_manifest.get("status")
        == "completed"
        and cache_manifest.get(
            "all_checks_passed"
        )
        is True
    ),
    "cache_protocol_matches": (
        cache_manifest.get(
            "protocol_version"
        )
        == DATA_PROTOCOL_VERSION
    ),
    "cache_verification_passed": (
        cache_verification.get("status")
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "model_source_hash_matches_preflight": (
        preflight[
            "neural_assets"
        ]["model_source"]["sha256"]
        == sha256_file(MODEL_SOURCE)
    ),
    "model_registry_hash_matches_preflight": (
        preflight[
            "neural_assets"
        ]["model_registry"]["sha256"]
        == sha256_file(MODEL_REGISTRY)
    ),
    "torch_importable": (
        preflight[
            "environment"
        ]["torch_runtime"][
            "importable"
        ]
        is True
    ),
    "configuration_count_formula_matches": (
        EXPECTED_CONFIGURATION_COUNT
        == 110
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
        "Phase 5 protocol lock preflight "
        "failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

variant_definitions = {
    "B0": {
        "canonical_name": "B0",
        "source_variant": None,
        "source_state": (
            "new_training_from_final_v3_2_cache"
        ),
        "pruning_ratio": 0.0,
        "physical_compaction": False,
        "fake_quantization": False,
        "training_mode": "train_from_scratch",
        "budget": "b0_budget",
        "selection_metric": (
            "fp32_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "FP32-FT": {
        "canonical_name": "FP32-FT",
        "source_variant": "B0",
        "source_state": (
            "matching_architecture_seed_B0_best_checkpoint"
        ),
        "pruning_ratio": 0.0,
        "physical_compaction": False,
        "fake_quantization": False,
        "training_mode": (
            "equal_budget_fp32_fine_tuning"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "fp32_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "P25-noFT": {
        "canonical_name": "P25-noFT",
        "source_variant": "B0",
        "source_state": (
            "independent_physical_pruning_from_B0"
        ),
        "pruning_ratio": 0.25,
        "physical_compaction": True,
        "fake_quantization": False,
        "training_mode": "no_updates_after_pruning",
        "budget": "zero_update_budget",
        "selection_metric": (
            "not_applicable_deterministic_transform"
        ),
        "phase5_test_evaluations": 1,
    },
    "P25": {
        "canonical_name": "P25-FP32-FT",
        "source_variant": "P25-noFT",
        "source_state": (
            "identical_P25_noFT_state_shared_with_P25_QAT"
        ),
        "pruning_ratio": 0.25,
        "physical_compaction": True,
        "fake_quantization": False,
        "training_mode": (
            "equal_budget_fp32_fine_tuning"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "fp32_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "P50-noFT": {
        "canonical_name": "P50-noFT",
        "source_variant": "B0",
        "source_state": (
            "independent_physical_pruning_from_B0"
        ),
        "pruning_ratio": 0.50,
        "physical_compaction": True,
        "fake_quantization": False,
        "training_mode": "no_updates_after_pruning",
        "budget": "zero_update_budget",
        "selection_metric": (
            "not_applicable_deterministic_transform"
        ),
        "phase5_test_evaluations": 1,
    },
    "P50": {
        "canonical_name": "P50-FP32-FT",
        "source_variant": "P50-noFT",
        "source_state": (
            "identical_P50_noFT_state_shared_with_P50_QAT"
        ),
        "pruning_ratio": 0.50,
        "physical_compaction": True,
        "fake_quantization": False,
        "training_mode": (
            "equal_budget_fp32_fine_tuning"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "fp32_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "DQ": {
        "canonical_name": "DQ",
        "source_variant": "B0",
        "source_state": (
            "matching_architecture_seed_B0_best_checkpoint"
        ),
        "pruning_ratio": 0.0,
        "physical_compaction": False,
        "fake_quantization": False,
        "training_mode": (
            "dynamic_int8_conversion_no_updates"
        ),
        "budget": "zero_update_budget",
        "selection_metric": (
            "not_applicable_deterministic_transform"
        ),
        "phase5_test_evaluations": 1,
    },
    "PTQ": {
        "canonical_name": "PTQ",
        "source_variant": "B0",
        "source_state": (
            "matching_architecture_seed_B0_best_checkpoint"
        ),
        "pruning_ratio": 0.0,
        "physical_compaction": False,
        "fake_quantization": False,
        "training_mode": (
            "static_int8_calibration_no_updates"
        ),
        "budget": "zero_update_budget",
        "selection_metric": (
            "phase6_validation_calibration_selection"
        ),
        "phase5_test_evaluations": 0,
    },
    "QAT": {
        "canonical_name": "QAT",
        "source_variant": "B0",
        "source_state": (
            "matching_architecture_seed_B0_best_checkpoint"
        ),
        "pruning_ratio": 0.0,
        "physical_compaction": False,
        "fake_quantization": True,
        "training_mode": (
            "equal_budget_quantization_aware_training"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "P25-QAT": {
        "canonical_name": "P25-QAT",
        "source_variant": "P25-noFT",
        "source_state": (
            "identical_P25_noFT_state_shared_with_P25"
        ),
        "pruning_ratio": 0.25,
        "physical_compaction": True,
        "fake_quantization": True,
        "training_mode": (
            "equal_budget_pruned_quantization_aware_training"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
    "P50-QAT": {
        "canonical_name": "P50-QAT",
        "source_variant": "P50-noFT",
        "source_state": (
            "identical_P50_noFT_state_shared_with_P50"
        ),
        "pruning_ratio": 0.50,
        "physical_compaction": True,
        "fake_quantization": True,
        "training_mode": (
            "equal_budget_pruned_quantization_aware_training"
        ),
        "budget": "equal_update_budget",
        "selection_metric": (
            "converted_int8_validation_fingerprint_macro_f1"
        ),
        "phase5_test_evaluations": 1,
    },
}

if set(variant_definitions) != set(VARIANTS):
    raise RuntimeError(
        "Variant definition set does not "
        "match the locked matrix."
    )

matrix_rows: list[dict[str, Any]] = []

for architecture in ARCHITECTURES:
    base_lr = BASE_LEARNING_RATES[
        architecture
    ]

    low_lr = (
        base_lr * LOW_LR_FACTOR
    )

    for variant in VARIANTS:
        definition = (
            variant_definitions[variant]
        )

        for seed in SEEDS:
            if definition["budget"] == (
                "b0_budget"
            ):
                learning_rate = base_lr
                max_epochs = B0_BUDGET[
                    "max_epochs"
                ]
                patience = B0_BUDGET[
                    "early_stopping_patience"
                ]
                min_delta = B0_BUDGET[
                    "early_stopping_min_delta"
                ]
            elif definition["budget"] == (
                "equal_update_budget"
            ):
                learning_rate = low_lr
                max_epochs = (
                    EQUAL_UPDATE_BUDGET[
                        "max_epochs"
                    ]
                )
                patience = (
                    EQUAL_UPDATE_BUDGET[
                        "early_stopping_patience"
                    ]
                )
                min_delta = (
                    EQUAL_UPDATE_BUDGET[
                        "early_stopping_min_delta"
                    ]
                )
            else:
                learning_rate = None
                max_epochs = 0
                patience = 0
                min_delta = None

            configuration_id = (
                f"{architecture}__"
                f"{variant.lower().replace('-', '_')}"
                f"__seed_{seed}"
            )

            output_directory = (
                PHASE5_RESULTS_ROOT
                / architecture
                / variant.lower().replace(
                    "-",
                    "_",
                )
                / f"seed_{seed}"
            )

            matrix_rows.append(
                {
                    "configuration_id": (
                        configuration_id
                    ),
                    "architecture": (
                        architecture
                    ),
                    "variant": variant,
                    "canonical_variant": (
                        definition[
                            "canonical_name"
                        ]
                    ),
                    "alias_resolution": (
                        ALIAS_RESOLUTION.get(
                            variant,
                            "",
                        )
                    ),
                    "seed": seed,
                    "source_variant": (
                        definition[
                            "source_variant"
                        ]
                        or ""
                    ),
                    "source_state_policy": (
                        definition[
                            "source_state"
                        ]
                    ),
                    "pruning_ratio": (
                        definition[
                            "pruning_ratio"
                        ]
                    ),
                    "physical_compaction": (
                        definition[
                            "physical_compaction"
                        ]
                    ),
                    "fake_quantization": (
                        definition[
                            "fake_quantization"
                        ]
                    ),
                    "training_mode": (
                        definition[
                            "training_mode"
                        ]
                    ),
                    "budget_name": (
                        definition["budget"]
                    ),
                    "optimizer": (
                        "AdamW"
                        if max_epochs > 0
                        else ""
                    ),
                    "learning_rate": (
                        learning_rate
                        if learning_rate
                        is not None
                        else ""
                    ),
                    "batch_size": (
                        4096
                        if max_epochs > 0
                        else ""
                    ),
                    "max_epochs": (
                        max_epochs
                    ),
                    "early_stopping_patience": (
                        patience
                    ),
                    "early_stopping_min_delta": (
                        min_delta
                        if min_delta
                        is not None
                        else ""
                    ),
                    "weight_decay": (
                        0.0001
                        if max_epochs > 0
                        else ""
                    ),
                    "selection_metric": (
                        definition[
                            "selection_metric"
                        ]
                    ),
                    "phase5_test_evaluation_count": (
                        definition[
                            "phase5_test_evaluations"
                        ]
                    ),
                    "test_policy": (
                        "deferred_to_phase6"
                        if variant == "PTQ"
                        else "exactly_once_after_lock"
                    ),
                    "output_directory": str(
                        output_directory
                    ),
                    "status": "locked_not_started",
                }
            )

if len(matrix_rows) != (
    EXPECTED_CONFIGURATION_COUNT
):
    raise RuntimeError(
        "Locked configuration matrix "
        "does not contain 110 rows."
    )

configuration_ids = {
    row["configuration_id"]
    for row in matrix_rows
}

if len(configuration_ids) != (
    EXPECTED_CONFIGURATION_COUNT
):
    raise RuntimeError(
        "Configuration IDs are not unique."
    )

matrix_checks = {
    "configuration_count_is_110": (
        len(matrix_rows) == 110
    ),
    "architecture_set_matches": (
        {
            row["architecture"]
            for row in matrix_rows
        }
        == set(ARCHITECTURES)
    ),
    "variant_set_matches": (
        {
            row["variant"]
            for row in matrix_rows
        }
        == set(VARIANTS)
    ),
    "seed_set_matches": (
        {
            int(row["seed"])
            for row in matrix_rows
        }
        == set(SEEDS)
    ),
    "P25_alias_resolved": (
        all(
            row["canonical_variant"]
            == "P25-FP32-FT"
            for row in matrix_rows
            if row["variant"] == "P25"
        )
    ),
    "P50_alias_resolved": (
        all(
            row["canonical_variant"]
            == "P50-FP32-FT"
            for row in matrix_rows
            if row["variant"] == "P50"
        )
    ),
    "equal_budget_max_epochs_are_eight": (
        all(
            int(row["max_epochs"]) == 8
            for row in matrix_rows
            if row["budget_name"]
            == "equal_update_budget"
        )
    ),
    "equal_budget_lr_factor_is_point_one": (
        all(
            math.isclose(
                float(
                    row["learning_rate"]
                ),
                BASE_LEARNING_RATES[
                    row["architecture"]
                ]
                * LOW_LR_FACTOR,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for row in matrix_rows
            if row["budget_name"]
            == "equal_update_budget"
        )
    ),
    "PTQ_test_is_deferred": (
        all(
            int(
                row[
                    "phase5_test_evaluation_count"
                ]
            )
            == 0
            and row["test_policy"]
            == "deferred_to_phase6"
            for row in matrix_rows
            if row["variant"] == "PTQ"
        )
    ),
    "non_PTQ_test_count_is_one": (
        all(
            int(
                row[
                    "phase5_test_evaluation_count"
                ]
            )
            == 1
            for row in matrix_rows
            if row["variant"] != "PTQ"
        )
    ),
    "P25_and_P25_QAT_share_source": (
        all(
            row["source_variant"]
            == "P25-noFT"
            for row in matrix_rows
            if row["variant"]
            in {"P25", "P25-QAT"}
        )
    ),
    "P50_and_P50_QAT_share_source": (
        all(
            row["source_variant"]
            == "P50-noFT"
            for row in matrix_rows
            if row["variant"]
            in {"P50", "P50-QAT"}
        )
    ),
}

failed_matrix_checks = [
    name
    for name, passed
    in matrix_checks.items()
    if not passed
]

if failed_matrix_checks:
    raise RuntimeError(
        "Phase 5 locked matrix "
        "validation failed: "
        + ", ".join(
            failed_matrix_checks
        )
    )

matrix_fieldnames = [
    "configuration_id",
    "architecture",
    "variant",
    "canonical_variant",
    "alias_resolution",
    "seed",
    "source_variant",
    "source_state_policy",
    "pruning_ratio",
    "physical_compaction",
    "fake_quantization",
    "training_mode",
    "budget_name",
    "optimizer",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "weight_decay",
    "selection_metric",
    "phase5_test_evaluation_count",
    "test_policy",
    "output_directory",
    "status",
]

atomic_csv(
    OUTPUT_MATRIX,
    matrix_rows,
    matrix_fieldnames,
)

comparison_rows: list[
    dict[str, Any]
] = []

for comparison in PRIMARY_COMPARISONS:
    for architecture in ARCHITECTURES:
        comparison_rows.append(
            {
                **comparison,
                "architecture": architecture,
                "paired_seeds": (
                    "42;123;2026;3407;8192"
                ),
                "analysis_unit": (
                    "paired_training_seed"
                ),
                "primary_endpoint": (
                    "test_fingerprint_macro_f1"
                ),
                "safety_endpoints": (
                    "gafgyt_fnr;"
                    "mirai_fnr;"
                    "benign_misclassification"
                ),
                "test_isolation_required": True,
                "status": (
                    "locked_deferred_phase6"
                    if comparison[
                        "left_variant"
                    ]
                    == "PTQ"
                    else "locked"
                ),
            }
        )

comparison_fieldnames = [
    "comparison_id",
    "architecture",
    "left_variant",
    "right_variant",
    "purpose",
    "family",
    "priority",
    "paired_seeds",
    "analysis_unit",
    "primary_endpoint",
    "safety_endpoints",
    "test_isolation_required",
    "status",
]

atomic_csv(
    OUTPUT_COMPARISONS,
    comparison_rows,
    comparison_fieldnames,
)

source_inventory = [
    file_record(
        PHASE4_COMPLETION
    ),
    file_record(
        PHASE4_LOCK
    ),
    file_record(
        PREFLIGHT_JSON
    ),
    file_record(
        PREFLIGHT_MATRIX
    ),
    file_record(
        FINAL_CACHE_MANIFEST
    ),
    file_record(
        FINAL_CACHE_VERIFICATION
    ),
    file_record(
        MODEL_SOURCE
    ),
    file_record(
        MODEL_REGISTRY
    ),
]

legacy_reference_inventory = [
    optional_file_record(path)
    for path in (
        LEGACY_FP32_PROTOCOL,
        LEGACY_PRUNING_PROTOCOL,
        LEGACY_QUANTIZATION_PROTOCOL,
        LEGACY_PRUNING_QAT_PROTOCOL,
    )
]

protocol = {
    "status": "locked",
    "protocol_name": (
        "Phase 5 fair-budget neural "
        "compression matrix"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "data_protocol_version": (
        DATA_PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "task": {
        "name": "family_3",
        "classes": list(
            EXPECTED_CLASS_ORDER
        ),
        "feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "primary_unit": (
            "fingerprint"
        ),
        "secondary_weighting": (
            "represented_raw_row_count"
        ),
    },
    "data": {
        "cache": str(FINAL_CACHE),
        "split_counts": SPLIT_COUNTS,
        "split_policy": (
            "fixed leakage-controlled "
            "grouped_float32 v3_2"
        ),
        "input_pipeline": (
            "canonical float32 -> "
            "train-only StandardScaler -> "
            "final float32"
        ),
        "scaler_policy": {
            "fit_split": "train_only",
            "fit_once_for_fixed_split": True,
            "validation_or_test_in_fit": False,
            "persist_mean_scale_and_hash": True,
        },
        "class_weight_policy": {
            "fit_split": "train_only",
            "scheme": (
                "inverse_square_root_frequency_mean1"
            ),
            "primary_training_unit": (
                "fingerprint"
            ),
            "raw_occurrence_weight_used_for_fit": False,
        },
    },
    "legacy_reuse_policy": {
        "legacy_checkpoints_directly_reused": False,
        "legacy_scaler_directly_reused": False,
        "legacy_test_results_mixed_with_v3_2": False,
        "architecture_code_reuse": (
            "allowed only under locked source "
            "and registry hashes"
        ),
        "reason": (
            "Phase 5 B0 checkpoints must be newly "
            "trained on the final v3_2 cache."
        ),
    },
    "architectures": {
        "tinyml_mlp": {
            "topology_reference": (
                "115->64->32->3"
            ),
            "b0_learning_rate": (
                BASE_LEARNING_RATES[
                    "tinyml_mlp"
                ]
            ),
            "equal_budget_learning_rate": (
                BASE_LEARNING_RATES[
                    "tinyml_mlp"
                ]
                * LOW_LR_FACTOR
            ),
        },
        "compact_dnn": {
            "topology_reference": (
                "115->128->64->32->3"
            ),
            "b0_learning_rate": (
                BASE_LEARNING_RATES[
                    "compact_dnn"
                ]
            ),
            "equal_budget_learning_rate": (
                BASE_LEARNING_RATES[
                    "compact_dnn"
                ]
                * LOW_LR_FACTOR
            ),
        },
        "model_source": (
            file_record(
                MODEL_SOURCE
            )
        ),
        "model_registry": (
            file_record(
                MODEL_REGISTRY
            )
        ),
    },
    "seeds": list(SEEDS),
    "variant_order": list(
        VARIANTS
    ),
    "alias_resolution": (
        ALIAS_RESOLUTION
    ),
    "alias_interpretation": (
        "P25 and P50 are the equal-budget "
        "FP32-fine-tuned pruned controls. "
        "Their canonical report names are "
        "P25-FP32-FT and P50-FP32-FT."
    ),
    "budgets": {
        "B0": {
            **B0_BUDGET,
            "learning_rates": (
                BASE_LEARNING_RATES
            ),
        },
        "equal_update_budget": {
            **EQUAL_UPDATE_BUDGET,
            "learning_rates": {
                architecture: (
                    learning_rate
                    * LOW_LR_FACTOR
                )
                for architecture, learning_rate
                in BASE_LEARNING_RATES.items()
            },
            "applies_to": [
                "FP32-FT",
                "P25",
                "P50",
                "QAT",
                "P25-QAT",
                "P50-QAT",
            ],
            "causal_fairness_rule": (
                "Branches in a direct comparison "
                "must start from the identical source "
                "state and receive the same maximum "
                "epoch, learning-rate, patience, "
                "min-delta, batch-size, optimizer, "
                "and weight-decay budget."
            ),
        },
    },
    "variant_definitions": (
        variant_definitions
    ),
    "pruning": PRUNING_RULES,
    "quantization": (
        QUANTIZATION_RULES
    ),
    "test_isolation": {
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
        "test_used_for_early_stopping": False,
        "test_used_for_thresholds": False,
        "test_used_for_calibration": False,
        "non_PTQ_test_evaluations_per_configuration": 1,
        "PTQ_phase5_test_evaluations_per_configuration": 0,
        "PTQ_final_test_access": (
            "exactly once after Phase 6 "
            "validation-only calibration selection"
        ),
    },
    "required_saved_outputs": [
        "best_checkpoint",
        "optimizer_and_training_history",
        "scaler_manifest",
        "class_weight_manifest",
        "source_checkpoint_hash",
        "model_source_hash",
        "variant_transform_manifest",
        "validation_logits_and_probabilities",
        "test_logits_and_probabilities_when_unlocked",
        "fingerprint_level_metrics",
        "raw_occurrence_weighted_metrics",
        "confusion_matrices",
        "per_class_precision_recall_f1_fnr",
        "MCC",
        "parameter_count",
        "MAC_count",
        "raw_tensor_bytes",
        "serialized_artifact_bytes",
        "test_access_counter",
        "run_manifest",
    ],
    "reporting": {
        "primary_endpoint": (
            "fingerprint_macro_f1"
        ),
        "secondary_endpoints": [
            "raw_occurrence_weighted_macro_f1",
            "accuracy",
            "balanced_accuracy",
            "MCC",
            "gafgyt_fnr",
            "mirai_fnr",
            "benign_misclassification",
            "log_loss",
        ],
        "probability_artifacts_required": True,
        "AUROC_or_AUPRC_reportable_only_if_recomputed_from_saved_scores": True,
        "five_seed_mean_and_standard_deviation": True,
        "single_best_seed_as_main_result": False,
    },
    "predeclared_comparisons": list(
        PRIMARY_COMPARISONS
    ),
    "matrix": {
        "architectures": list(
            ARCHITECTURES
        ),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "configuration_count": (
            EXPECTED_CONFIGURATION_COUNT
        ),
        "locked_matrix_csv": str(
            OUTPUT_MATRIX
        ),
        "phase5_immediate_test_configuration_count": (
            sum(
                int(
                    row[
                        "phase5_test_evaluation_count"
                    ]
                )
                for row in matrix_rows
            )
        ),
        "PTQ_deferred_configuration_count": (
            sum(
                1
                for row in matrix_rows
                if row["variant"] == "PTQ"
            )
        ),
    },
    "execution_order": [
        "generate_and_lock_train_only_scaler_and_class_weights",
        "train_B0_for_both_architectures_and_five_seeds",
        "independently_verify_and_lock_B0_checkpoints",
        "create_equal_budget_FP32_FT_and_pruning_noFT_sources",
        "run_P25_P50_QAT_and_pruning_QAT_branches",
        "run_DQ",
        "run_PTQ_conversion_validation_only",
        "complete_Phase6_PTQ_calibration_sensitivity_before_PTQ_test",
        "independently_verify_all_saved_artifacts",
        "aggregate_paired_seed_results",
    ],
    "source_inventory": (
        source_inventory
    ),
    "legacy_reference_inventory": (
        legacy_reference_inventory
    ),
    "preflight_checks": (
        preflight_checks
    ),
    "matrix_checks": (
        matrix_checks
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

completion = {
    "status": "locked",
    "phase": 5,
    "phase_name": (
        "fair_budget_compression_matrix"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "configuration_count": (
        EXPECTED_CONFIGURATION_COUNT
    ),
    "architecture_count": (
        len(ARCHITECTURES)
    ),
    "variant_count": (
        len(VARIANTS)
    ),
    "seed_count": len(SEEDS),
    "naming_ambiguity_resolved": True,
    "P25_canonical_name": (
        "P25-FP32-FT"
    ),
    "P50_canonical_name": (
        "P50-FP32-FT"
    ),
    "equal_update_budget_max_epochs": 8,
    "PTQ_test_deferred_to_phase6": True,
    "legacy_checkpoint_reuse_allowed": False,
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "matrix": str(
        OUTPUT_MATRIX
    ),
    "matrix_sha256": (
        sha256_file(
            OUTPUT_MATRIX
        )
    ),
    "comparisons": str(
        OUTPUT_COMPARISONS
    ),
    "comparisons_sha256": (
        sha256_file(
            OUTPUT_COMPARISONS
        )
    ),
    "next_action": (
        "Generate and independently verify "
        "the Phase 5 train-only scaler and "
        "class-weight artifacts."
    ),
    "ready_for_execution_artifact_build": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_COMPLETION,
    completion,
)

generated_inventory = [
    file_record(
        OUTPUT_PROTOCOL
    ),
    file_record(
        OUTPUT_MATRIX
    ),
    file_record(
        OUTPUT_COMPARISONS
    ),
    file_record(
        OUTPUT_COMPLETION
    ),
]

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": (
        source_inventory
    ),
    "legacy_reference_artifacts": (
        legacy_reference_inventory
    ),
    "generated_artifacts": (
        generated_inventory
    ),
    "configuration_count": 110,
    "immediate_phase5_test_count": (
        sum(
            int(
                row[
                    "phase5_test_evaluation_count"
                ]
            )
            for row in matrix_rows
        )
    ),
    "PTQ_test_count_deferred": 10,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "protocol_exists": (
        OUTPUT_PROTOCOL.exists()
    ),
    "matrix_exists": (
        OUTPUT_MATRIX.exists()
    ),
    "comparisons_exist": (
        OUTPUT_COMPARISONS.exists()
    ),
    "completion_exists": (
        OUTPUT_COMPLETION.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "completion_protocol_hash_matches": (
        read_json(
            OUTPUT_COMPLETION
        )["protocol_sha256"]
        == sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "completion_matrix_hash_matches": (
        read_json(
            OUTPUT_COMPLETION
        )["matrix_sha256"]
        == sha256_file(
            OUTPUT_MATRIX
        )
    ),
    "lock_manifest_locked": (
        read_json(
            OUTPUT_LOCK_MANIFEST
        ).get("status")
        == "locked"
    ),
}

failed_post_checks = [
    name
    for name, passed
    in post_checks.items()
    if not passed
]

if failed_post_checks:
    raise RuntimeError(
        "Phase 5 protocol post-lock "
        "verification failed: "
        + ", ".join(
            failed_post_checks
        )
    )

PHASE5_RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 92)
print("PHASE 5 FAIR-BUDGET COMPRESSION PROTOCOL LOCK SUMMARY")
print("=" * 92)
print(
    "Protocol version               : "
    f"{PROTOCOL_VERSION}"
)
print(
    "Architectures                  : "
    f"{list(ARCHITECTURES)}"
)
print(
    "Variants                       : "
    f"{len(VARIANTS)}"
)
print(
    "Seeds                          : "
    f"{list(SEEDS)}"
)
print(
    "Locked configurations          : "
    f"{len(matrix_rows)}"
)
print(
    "P25 canonical meaning          : "
    "P25-FP32-FT"
)
print(
    "P50 canonical meaning          : "
    "P50-FP32-FT"
)
print(
    "Equal update budget            : "
    "LR=0.1x B0, max_epochs=8, "
    "patience=3, min_delta=0.0002"
)
print(
    "Physical pruning               : "
    "True"
)
print(
    "P25/P50 independently from B0  : "
    "True"
)
print(
    "Legacy checkpoints reused      : "
    "False"
)
print(
    "Immediate Phase 5 test configs : "
    f"{sum(int(row['phase5_test_evaluation_count']) for row in matrix_rows)}"
)
print(
    "PTQ configs deferred to Phase 6: "
    "10"
)
print(
    "Protocol                       : "
    f"{OUTPUT_PROTOCOL}"
)
print(
    "Locked matrix                  : "
    f"{OUTPUT_MATRIX}"
)
print(
    "Comparisons                    : "
    f"{OUTPUT_COMPARISONS}"
)
print(
    "Lock manifest                  : "
    f"{OUTPUT_LOCK_MANIFEST}"
)
print(
    "All checks passed              : True"
)
print(
    "PHASE 5 FAIR-BUDGET COMPRESSION PROTOCOL LOCKED"
)
