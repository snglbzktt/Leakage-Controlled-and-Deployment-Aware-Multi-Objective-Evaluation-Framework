from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_LOCK = (
    AUDIT
    / "phase5_compression_locked_v3_2.json"
)

PHASE5_MASTER_MATRIX = (
    AUDIT
    / "phase5_compression_master_run_matrix_v3_2.csv"
)

PHASE8_FINAL_FAMILY_LOCK = (
    AUDIT
    / "phase8_final_model_family_locked_v3_2.json"
)

MODEL_SOURCE = (
    ROOT
    / "src"
    / "models"
    / "nbaiot_models.py"
)

PRUNING_ENGINE = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

QAT_ENGINE = (
    ROOT
    / "src"
    / "compression"
    / "phase5_qat_engine_v3_2.py"
)

SCALER_PATH = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

OUTPUT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase9_final_artifact_protocol_v3_2.json"
)

OUTPUT_REGISTRY = (
    AUDIT
    / "phase9_final_artifact_source_registry_v3_2.csv"
)

OUTPUT_PREFLIGHT = (
    AUDIT
    / "phase9_final_artifact_protocol_preflight_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase9_final_artifact_protocol_preflight_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase9_final_artifact_protocol_preflight_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase9_final_artifact_v3_2"
PHASE5_PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
PHASE8_PROTOCOL_VERSION = "phase8_multi_objective_decision_v3_2"

SELECTED_ARCHITECTURE = "tinyml_mlp"
SELECTED_VARIANT = "P50-QAT"
SELECTED_REPRESENTATION = "static_int8"

CANONICAL_ARTIFACT_SEED = 2026

EXPECTED_FEATURE_COUNT = 115
EXPECTED_CLASS_COUNT = 3

PLANNED_ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "final_artifact"
    / "tinyml_mlp_P50_QAT_seed2026_v3_2"
)

PLANNED_BUNDLE_FILES = (
    "model_int8_checkpoint.pt",
    "nbaiot_models.py",
    "phase5_physical_pruning_engine_v3_2.py",
    "phase5_qat_engine_v3_2.py",
    "train_only_standard_scaler_v3_2.npz",
    "inference_contract.json",
    "artifact_manifest.json",
    "artifact_lock.json",
)

WINDOWS_FILE_RETRY_COUNT = 40
WINDOWS_FILE_RETRY_DELAY_SECONDS = 0.25


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
        "Windows kept the destination file locked "
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
        ),
        encoding="utf-8",
    )

    replace_with_retry(
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

    replace_with_retry(
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


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


required_paths = (
    PHASE5_LOCK,
    PHASE5_MASTER_MATRIX,
    PHASE8_FINAL_FAMILY_LOCK,
    MODEL_SOURCE,
    PRUNING_ENGINE,
    QAT_ENGINE,
    SCALER_PATH,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_PROTOCOL,
    OUTPUT_REGISTRY,
    OUTPUT_PREFLIGHT,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 9 artifact-preflight output "
            "already exists; refusing to overwrite: "
            f"{output_path}"
        )

if PLANNED_ARTIFACT_ROOT.exists():
    raise FileExistsError(
        "Planned final artifact directory already exists; "
        "refusing to continue before the protocol is locked: "
        f"{PLANNED_ARTIFACT_ROOT}"
    )

phase5_lock = read_json(
    PHASE5_LOCK
)

phase8_lock = read_json(
    PHASE8_FINAL_FAMILY_LOCK
)

phase5_rows = read_csv(
    PHASE5_MASTER_MATRIX
)

entry_checks = {
    "phase5_locked": (
        phase5_lock.get("status")
        == "locked"
        and phase5_lock.get(
            "protocol_version"
        )
        == PHASE5_PROTOCOL_VERSION
        and phase5_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase5_master_hash_matches": (
        phase5_lock.get(
            "master_run_matrix_sha256"
        )
        == sha256_file(
            PHASE5_MASTER_MATRIX
        )
    ),
    "phase8_family_locked": (
        phase8_lock.get("status")
        == "locked"
        and phase8_lock.get(
            "protocol_version"
        )
        == PHASE8_PROTOCOL_VERSION
        and phase8_lock.get(
            "final_model_family_selected"
        )
        is True
        and phase8_lock.get(
            "single_seed_checkpoint_selected"
        )
        is False
        and phase8_lock.get(
            "ready_for_final_artifact_protocol"
        )
        is True
        and phase8_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "selected_family_matches_protocol": (
        phase8_lock[
            "selected_model_family"
        ]["architecture"]
        == SELECTED_ARCHITECTURE
        and phase8_lock[
            "selected_model_family"
        ]["variant"]
        == SELECTED_VARIANT
        and phase8_lock[
            "selected_model_family"
        ]["representation"]
        == SELECTED_REPRESENTATION
    ),
    "final_artifact_directory_absent": (
        not PLANNED_ARTIFACT_ROOT.exists()
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
        "Phase 9 final-artifact protocol entry gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

selected_rows = [
    row
    for row in phase5_rows
    if row["architecture"]
    == SELECTED_ARCHITECTURE
    and row["variant"]
    == SELECTED_VARIANT
    and int(row["seed"])
    == CANONICAL_ARTIFACT_SEED
]

if len(selected_rows) != 1:
    raise RuntimeError(
        "Canonical artifact source lookup did not "
        "return exactly one Phase 5 row."
    )

selected_row = selected_rows[0]

if selected_row["representation"] != SELECTED_REPRESENTATION:
    raise RuntimeError(
        "Canonical source representation mismatch."
    )

if int(
    selected_row[
        "test_evaluation_count"
    ]
) != 1:
    raise RuntimeError(
        "Canonical source test-evaluation count mismatch."
    )

if parse_bool(
    selected_row[
        "test_used_for_selection"
    ]
):
    raise RuntimeError(
        "Canonical source reports test use for selection."
    )

run_directory = Path(
    selected_row[
        "run_directory"
    ]
)

metrics_path = Path(
    selected_row[
        "metrics_path"
    ]
)

checkpoint_path = (
    run_directory
    / "best_int8_checkpoint.pt"
)

for path in (
    run_directory,
    metrics_path,
    checkpoint_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

metrics_sha256 = sha256_file(
    metrics_path
)

if (
    metrics_sha256
    != selected_row[
        "metrics_sha256"
    ]
):
    raise RuntimeError(
        "Canonical source metrics hash mismatch."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

checkpoint_checks = {
    "checkpoint_is_dictionary": (
        isinstance(checkpoint, dict)
    ),
    "int8_state_dict_present": (
        isinstance(
            checkpoint.get(
                "INT8_model_state_dict"
            ),
            dict,
        )
        and len(
            checkpoint[
                "INT8_model_state_dict"
            ]
        )
        > 0
    ),
    "qat_backend_present": (
        isinstance(
            checkpoint.get(
                "QAT_backend"
            ),
            str,
        )
        and len(
            checkpoint[
                "QAT_backend"
            ]
        )
        > 0
    ),
    "qat_backend_supported": (
        checkpoint.get(
            "QAT_backend"
        )
        in torch.backends.quantized.supported_engines
    ),
}

failed_checkpoint_checks = [
    name
    for name, passed
    in checkpoint_checks.items()
    if not passed
]

if failed_checkpoint_checks:
    raise RuntimeError(
        "Canonical QAT checkpoint preflight failed: "
        + ", ".join(
            failed_checkpoint_checks
        )
    )

source_registry_row = {
    "artifact_role": (
        "canonical_final_deployment_source"
    ),
    "selection_method": (
        "fixed_canonical_seed_no_metric_ranking"
    ),
    "architecture": (
        SELECTED_ARCHITECTURE
    ),
    "variant": (
        SELECTED_VARIANT
    ),
    "representation": (
        SELECTED_REPRESENTATION
    ),
    "seed": (
        CANONICAL_ARTIFACT_SEED
    ),
    "run_directory": str(
        run_directory
    ),
    "metrics_path": str(
        metrics_path
    ),
    "metrics_sha256": (
        metrics_sha256
    ),
    "checkpoint_path": str(
        checkpoint_path
    ),
    "checkpoint_sha256": (
        sha256_file(
            checkpoint_path
        )
    ),
    "checkpoint_size_bytes": int(
        checkpoint_path.stat().st_size
    ),
    "QAT_backend": (
        checkpoint[
            "QAT_backend"
        ]
    ),
    "test_evaluation_count": int(
        selected_row[
            "test_evaluation_count"
        ]
    ),
    "test_used_for_seed_selection": (
        False
    ),
    "validation_used_for_seed_selection": (
        False
    ),
    "deployment_metric_used_for_seed_selection": (
        False
    ),
    "single_checkpoint_selected_during_preflight": (
        False
    ),
}

atomic_csv(
    OUTPUT_REGISTRY,
    [
        source_registry_row,
    ],
    [
        "artifact_role",
        "selection_method",
        "architecture",
        "variant",
        "representation",
        "seed",
        "run_directory",
        "metrics_path",
        "metrics_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "QAT_backend",
        "test_evaluation_count",
        "test_used_for_seed_selection",
        "validation_used_for_seed_selection",
        "deployment_metric_used_for_seed_selection",
        "single_checkpoint_selected_during_preflight",
    ],
)

protocol = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact_protocol"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "upstream_selection": {
        "model_family": (
            "tinyml_mlp::P50-QAT"
        ),
        "architecture": (
            SELECTED_ARCHITECTURE
        ),
        "variant": (
            SELECTED_VARIANT
        ),
        "representation": (
            SELECTED_REPRESENTATION
        ),
        "primary_decision_profile": (
            "balanced"
        ),
    },
    "checkpoint_selection_policy": {
        "canonical_seed": (
            CANONICAL_ARTIFACT_SEED
        ),
        "method": (
            "fixed project canonical seed"
        ),
        "test_metric_ranking_used": False,
        "validation_metric_ranking_used": False,
        "deployment_metric_ranking_used": False,
        "best_seed_selection_used": False,
        "rationale": (
            "The model family was selected from aggregate "
            "five-seed evidence. The deployment checkpoint "
            "is fixed to seed 2026 without ranking seeds by "
            "test, validation, or deployment outcomes."
        ),
    },
    "artifact_construction_policy": {
        "planned_artifact_root": str(
            PLANNED_ARTIFACT_ROOT
        ),
        "copy_checkpoint_byte_for_byte": (
            True
        ),
        "copy_model_source_byte_for_byte": (
            True
        ),
        "copy_pruning_engine_byte_for_byte": (
            True
        ),
        "copy_QAT_engine_byte_for_byte": (
            True
        ),
        "copy_scaler_byte_for_byte": (
            True
        ),
        "planned_bundle_files": list(
            PLANNED_BUNDLE_FILES
        ),
        "overwrite_existing_artifact": (
            False
        ),
        "source_hash_verification_required": (
            True
        ),
        "bundle_hash_manifest_required": (
            True
        ),
    },
    "inference_contract": {
        "input_feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "input_dtype": "float32",
        "input_shape": (
            "[batch, 115]"
        ),
        "preprocessing": (
            "locked train-only StandardScaler"
        ),
        "output_class_count": (
            EXPECTED_CLASS_COUNT
        ),
        "output_shape": (
            "[batch, 3]"
        ),
        "output_semantics": (
            "logits; class-label order must be copied "
            "verbatim from a locked training artifact "
            "and hard-failed if unavailable"
        ),
        "device": "CPU",
        "quantization": (
            "eager static INT8 QAT"
        ),
        "batch_sizes_for_artifact_smoke": [
            1,
            32,
        ],
    },
    "verification_policy": {
        "construct_model_from_bundled_sources": (
            True
        ),
        "strict_state_dict_loading": (
            True
        ),
        "batch_1_inference_required": (
            True
        ),
        "batch_32_inference_required": (
            True
        ),
        "finite_output_required": True,
        "serialized_roundtrip_required": (
            True
        ),
        "source_checkpoint_output_match_required": (
            True
        ),
        "validation_data_access": False,
        "test_data_access": False,
        "independent_manifest_verification_required": (
            True
        ),
    },
    "source_artifacts": {
        "phase5_lock": file_record(
            PHASE5_LOCK
        ),
        "phase5_master_matrix": file_record(
            PHASE5_MASTER_MATRIX
        ),
        "phase8_final_family_lock": file_record(
            PHASE8_FINAL_FAMILY_LOCK
        ),
        "selected_metrics": file_record(
            metrics_path
        ),
        "selected_checkpoint": file_record(
            checkpoint_path
        ),
        "model_source": file_record(
            MODEL_SOURCE
        ),
        "pruning_engine": file_record(
            PRUNING_ENGINE
        ),
        "QAT_engine": file_record(
            QAT_ENGINE
        ),
        "scaler": file_record(
            SCALER_PATH
        ),
    },
    "environment": {
        "python_version": (
            sys.version
        ),
        "platform": (
            platform.platform()
        ),
        "torch_version": (
            torch.__version__
        ),
        "quantized_engine_current": (
            torch.backends.quantized.engine
        ),
        "quantized_engines_supported": list(
            torch.backends.quantized.supported_engines
        ),
    },
    "artifact_construction_performed": (
        False
    ),
    "model_inference_performed": False,
    "single_checkpoint_locked": False,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PROTOCOL,
    protocol,
)

preflight_checks = {
    "selected_family_tinyml_P50_QAT": (
        phase8_lock[
            "selected_model_family"
        ]["candidate_id"]
        == (
            "tinyml_mlp::P50-QAT"
        )
    ),
    "canonical_seed_2026": (
        CANONICAL_ARTIFACT_SEED
        == 2026
    ),
    "source_registry_row_count_1": (
        True
    ),
    "checkpoint_exists": (
        checkpoint_path.exists()
    ),
    "checkpoint_hash_recorded": (
        len(
            source_registry_row[
                "checkpoint_sha256"
            ]
        )
        == 64
    ),
    "QAT_backend_supported": (
        checkpoint_checks[
            "qat_backend_supported"
        ]
    ),
    "test_not_used_for_seed_selection": (
        source_registry_row[
            "test_used_for_seed_selection"
        ]
        is False
    ),
    "validation_not_used_for_seed_selection": (
        source_registry_row[
            "validation_used_for_seed_selection"
        ]
        is False
    ),
    "deployment_not_used_for_seed_selection": (
        source_registry_row[
            "deployment_metric_used_for_seed_selection"
        ]
        is False
    ),
    "artifact_directory_not_created": (
        not PLANNED_ARTIFACT_ROOT.exists()
    ),
    "artifact_construction_not_performed": (
        protocol[
            "artifact_construction_performed"
        ]
        is False
    ),
    "model_inference_not_performed": (
        protocol[
            "model_inference_performed"
        ]
        is False
    ),
    "single_checkpoint_not_locked": (
        protocol[
            "single_checkpoint_locked"
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
        "Phase 9 final-artifact protocol preflight failed: "
        + ", ".join(
            failed_preflight_checks
        )
    )

preflight = {
    "status": "passed",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact_protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "verified_at_utc": utc_now(),
    "entry_checks": (
        entry_checks
    ),
    "checkpoint_checks": (
        checkpoint_checks
    ),
    "preflight_checks": (
        preflight_checks
    ),
    "protocol": file_record(
        OUTPUT_PROTOCOL
    ),
    "source_registry": file_record(
        OUTPUT_REGISTRY
    ),
    "artifact_construction_performed": (
        False
    ),
    "model_inference_performed": False,
    "single_checkpoint_locked": False,
    "ready_for_artifact_construction": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_PREFLIGHT,
    preflight,
)

lock = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact_protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "selected_model_family": (
        "tinyml_mlp::P50-QAT"
    ),
    "canonical_artifact_seed": (
        CANONICAL_ARTIFACT_SEED
    ),
    "protocol": str(
        OUTPUT_PROTOCOL
    ),
    "protocol_sha256": (
        sha256_file(
            OUTPUT_PROTOCOL
        )
    ),
    "source_registry": str(
        OUTPUT_REGISTRY
    ),
    "source_registry_sha256": (
        sha256_file(
            OUTPUT_REGISTRY
        )
    ),
    "preflight_report": str(
        OUTPUT_PREFLIGHT
    ),
    "preflight_report_sha256": (
        sha256_file(
            OUTPUT_PREFLIGHT
        )
    ),
    "planned_artifact_root": str(
        PLANNED_ARTIFACT_ROOT
    ),
    "artifact_construction_performed": (
        False
    ),
    "model_inference_performed": False,
    "single_checkpoint_locked": False,
    "ready_for_artifact_construction": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 9,
    "artifact_name": (
        "final_deployment_artifact_protocol_preflight"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(PHASE5_LOCK),
        file_record(PHASE5_MASTER_MATRIX),
        file_record(PHASE8_FINAL_FAMILY_LOCK),
        file_record(metrics_path),
        file_record(checkpoint_path),
        file_record(MODEL_SOURCE),
        file_record(PRUNING_ENGINE),
        file_record(QAT_ENGINE),
        file_record(SCALER_PATH),
    ],
    "generated_artifacts": [
        file_record(OUTPUT_PROTOCOL),
        file_record(OUTPUT_REGISTRY),
        file_record(OUTPUT_PREFLIGHT),
        file_record(OUTPUT_LOCK),
    ],
    "selected_model_family": (
        "tinyml_mlp::P50-QAT"
    ),
    "canonical_artifact_seed": (
        CANONICAL_ARTIFACT_SEED
    ),
    "artifact_construction_performed": (
        False
    ),
    "model_inference_performed": False,
    "single_checkpoint_locked": False,
    "ready_for_artifact_construction": (
        True
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

print("=" * 92)
print("PHASE 9 FINAL DEPLOYMENT ARTIFACT PROTOCOL PREFLIGHT")
print("=" * 92)
print(
    "Final model family              : tinyml_mlp::P50-QAT"
)
print(
    "Representation                  : static_int8"
)
print(
    "Canonical artifact seed         : 2026"
)
print(
    "Seed selection method           : fixed canonical seed"
)
print(
    "Test metric seed ranking        : False"
)
print(
    "Validation metric seed ranking  : False"
)
print(
    "Deployment metric seed ranking  : False"
)
print(
    "Source checkpoint               : "
    f"{checkpoint_path}"
)
print(
    "Checkpoint size                 : "
    f"{checkpoint_path.stat().st_size} bytes"
)
print(
    "QAT backend                     : "
    f"{checkpoint['QAT_backend']}"
)
print(
    "Planned artifact directory      : "
    f"{PLANNED_ARTIFACT_ROOT}"
)
print(
    "Artifact construction performed : False"
)
print(
    "Model inference performed       : False"
)
print(
    "Single checkpoint locked        : False"
)
print(
    "Preflight status                : LOCKED"
)
print(
    "Ready for artifact construction : True"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 9 FINAL ARTIFACT PROTOCOL LOCKED"
)
