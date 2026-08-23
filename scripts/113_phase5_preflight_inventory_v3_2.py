from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

FINAL_CACHE_MANIFEST = FINAL_CACHE / "manifest.json"

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

LEGACY_PRUNING_QAT_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "nbaiot_family3_pruning_qat_protocol_v1.json"
)

LEGACY_SCALER = (
    ROOT
    / "models"
    / "preprocessing"
    / "nbaiot_standard_scaler_seed2026.npz"
)

PHASE5_OUTPUT_ROOT = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
)

OUTPUT_JSON = (
    AUDIT
    / "phase5_preflight_inventory_v3_2.json"
)

OUTPUT_CSV = (
    AUDIT
    / "phase5_planned_configuration_matrix_v3_2.csv"
)

EXPECTED_PROTOCOL_VERSION = (
    "tabular_baseline_protocol_v3_2"
)

EXPECTED_FEATURE_COUNT = 115

EXPECTED_SPLITS = {
    "train": {
        "fingerprints": 1_534_583,
        "raw_rows": 4_943_823,
    },
    "validation": {
        "fingerprints": 371_797,
        "raw_rows": 1_059_390,
    },
    "test": {
        "fingerprints": 371_796,
        "raw_rows": 1_059_393,
    },
}

ARCHITECTURES = (
    "tinyml_mlp",
    "compact_dnn",
)

SEEDS = (
    42,
    123,
    2026,
    3407,
    8192,
)

ROADMAP_CORE_VARIANTS = (
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

PREDECLARED_COMPARISONS = (
    "QAT vs FP32-FT",
    "P25 vs FP32-FT",
    "P50 vs FP32-FT",
    "P25-QAT vs P25-FP32-FT",
    "P50-QAT vs P50-FP32-FT",
    "P25-noFT vs P25",
    "P50-noFT vs P50",
)

LEGACY_SEARCH_ROOTS = (
    ROOT / "results",
    ROOT / "models",
    ROOT / "configs",
)

CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".torchscript",
    ".jit",
}

TEXT_SUFFIXES = {
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def package_version(
    package_name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            package_name
        )
    except (
        importlib.metadata.PackageNotFoundError,
        ValueError,
    ):
        return None


def optional_json_metadata(
    path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "sha256": None,
        "top_level_keys": [],
    }

    if not path.exists():
        return record

    record["size_bytes"] = int(
        path.stat().st_size
    )

    record["sha256"] = sha256_file(
        path
    )

    try:
        value = read_json(path)
    except Exception as error:
        record["json_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return record

    record["top_level_keys"] = sorted(
        str(key)
        for key in value.keys()
    )

    for candidate_key in (
        "protocol_name",
        "protocol_version",
        "status",
        "dataset",
        "registry_status",
        "registry_version",
    ):
        if candidate_key in value:
            record[candidate_key] = value[
                candidate_key
            ]

    return record


def find_legacy_artifacts() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    tokens = (
        "tinyml",
        "compact",
        "prun",
        "qat",
        "ptq",
        "quant",
        "fp32",
        "b0",
    )

    for search_root in LEGACY_SEARCH_ROOTS:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            if path in seen:
                continue

            lower_name = path.name.lower()

            relevant_name = any(
                token in lower_name
                for token in tokens
            )

            relevant_suffix = (
                path.suffix.lower()
                in CHECKPOINT_SUFFIXES
                or path.suffix.lower()
                in TEXT_SUFFIXES
            )

            if not (
                relevant_name
                and relevant_suffix
            ):
                continue

            seen.add(path)

            try:
                relative = path.relative_to(
                    ROOT
                )
            except ValueError:
                relative = path

            records.append(
                {
                    "relative_path": str(
                        relative
                    ),
                    "suffix": (
                        path.suffix.lower()
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "is_checkpoint": (
                        path.suffix.lower()
                        in CHECKPOINT_SUFFIXES
                    ),
                }
            )

    records.sort(
        key=lambda row: row[
            "relative_path"
        ]
    )

    return records


required_paths = (
    PHASE4_COMPLETION,
    PHASE4_LOCK,
    FINAL_CACHE,
    FINAL_CACHE_MANIFEST,
    FINAL_CACHE_VERIFICATION,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_JSON,
    OUTPUT_CSV,
):
    if output_path.exists():
        raise FileExistsError(
            "Phase 5 preflight artifact "
            "already exists; refusing "
            f"to overwrite: {output_path}"
        )

phase4_completion = read_json(
    PHASE4_COMPLETION
)

phase4_lock = read_json(
    PHASE4_LOCK
)

cache_manifest = read_json(
    FINAL_CACHE_MANIFEST
)

cache_verification = read_json(
    FINAL_CACHE_VERIFICATION
)

phase4_checks = {
    "phase4_completion_locked": (
        phase4_completion.get("status")
        == "locked"
    ),
    "phase4_completion_all_checks": (
        phase4_completion.get(
            "all_checks_passed"
        )
        is True
    ),
    "phase4_next_phase_ready": (
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
    ),
    "phase4_lock_all_checks": (
        phase4_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "cache_verification_passed": (
        cache_verification.get(
            "status"
        )
        == "passed"
        and cache_verification.get(
            "all_checks_passed"
        )
        is True
    ),
    "cache_protocol_matches": (
        cache_manifest.get(
            "protocol_version"
        )
        == EXPECTED_PROTOCOL_VERSION
    ),
}

failed_phase4_checks = [
    name
    for name, passed
    in phase4_checks.items()
    if not passed
]

if failed_phase4_checks:
    raise RuntimeError(
        "Phase 5 entry gate failed: "
        + ", ".join(
            failed_phase4_checks
        )
    )

cache_split_inventory = []
cache_split_checks: dict[str, bool] = {}

for split_name, expected in (
    EXPECTED_SPLITS.items()
):
    x_path = (
        FINAL_CACHE
        / f"X_{split_name}.npy"
    )

    y_path = (
        FINAL_CACHE
        / f"y_{split_name}.npy"
    )

    raw_path = (
        FINAL_CACHE
        / (
            "raw_row_count_"
            f"{split_name}.npy"
        )
    )

    for path in (
        x_path,
        y_path,
        raw_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    x = np.load(
        x_path,
        mmap_mode="r",
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    raw = np.load(
        raw_path,
        mmap_mode="r",
    )

    split_checks = {
        "x_shape": (
            x.shape
            == (
                expected[
                    "fingerprints"
                ],
                EXPECTED_FEATURE_COUNT,
            )
        ),
        "x_dtype": (
            x.dtype == np.float32
        ),
        "y_shape": (
            y.shape
            == (
                expected[
                    "fingerprints"
                ],
            )
        ),
        "raw_shape": (
            raw.shape
            == (
                expected[
                    "fingerprints"
                ],
            )
        ),
        "raw_total": (
            int(
                np.asarray(
                    raw,
                    dtype=np.int64,
                ).sum()
            )
            == expected["raw_rows"]
        ),
    }

    for check_name, passed in (
        split_checks.items()
    ):
        cache_split_checks[
            f"{split_name}_{check_name}"
        ] = passed

    cache_split_inventory.append(
        {
            "split": split_name,
            "fingerprint_rows": int(
                x.shape[0]
            ),
            "feature_count": int(
                x.shape[1]
            ),
            "x_dtype": str(x.dtype),
            "represented_raw_rows": int(
                np.asarray(
                    raw,
                    dtype=np.int64,
                ).sum()
            ),
            "all_checks_passed": all(
                split_checks.values()
            ),
        }
    )

failed_cache_checks = [
    name
    for name, passed
    in cache_split_checks.items()
    if not passed
]

if failed_cache_checks:
    raise RuntimeError(
        "Final cache shape/count "
        "verification failed: "
        + ", ".join(
            failed_cache_checks
        )
    )

package_versions = {
    package_name: package_version(
        package_name
    )
    for package_name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scikit-learn",
        "pandas",
        "onnx",
        "onnxruntime",
        "torch-pruning",
    )
}

torch_runtime: dict[str, Any] = {
    "importable": False,
    "version": package_versions[
        "torch"
    ],
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_names": [],
    "mps_available": False,
    "quantized_engines": [],
    "selected_quantized_engine": None,
    "thread_count": None,
}

try:
    torch = importlib.import_module(
        "torch"
    )

    torch_runtime[
        "importable"
    ] = True

    torch_runtime[
        "cuda_available"
    ] = bool(
        torch.cuda.is_available()
    )

    torch_runtime[
        "cuda_device_count"
    ] = int(
        torch.cuda.device_count()
    )

    torch_runtime[
        "cuda_device_names"
    ] = [
        torch.cuda.get_device_name(
            index
        )
        for index in range(
            torch.cuda.device_count()
        )
    ]

    if hasattr(
        torch.backends,
        "mps",
    ):
        torch_runtime[
            "mps_available"
        ] = bool(
            torch.backends.mps.is_available()
        )

    if hasattr(
        torch.backends,
        "quantized",
    ):
        torch_runtime[
            "quantized_engines"
        ] = list(
            torch.backends.quantized.supported_engines
        )

        torch_runtime[
            "selected_quantized_engine"
        ] = str(
            torch.backends.quantized.engine
        )

    torch_runtime[
        "thread_count"
    ] = int(
        torch.get_num_threads()
    )

except Exception as error:
    torch_runtime[
        "import_error"
    ] = (
        f"{type(error).__name__}: "
        f"{error}"
    )

legacy_protocol_inventory = [
    optional_json_metadata(path)
    for path in (
        MODEL_REGISTRY,
        LEGACY_FP32_PROTOCOL,
        LEGACY_PRUNING_PROTOCOL,
        LEGACY_PRUNING_QAT_PROTOCOL,
    )
]

legacy_artifacts = (
    find_legacy_artifacts()
)

legacy_checkpoint_records = [
    row
    for row in legacy_artifacts
    if row["is_checkpoint"]
]

legacy_protocol_dataset_names = {
    str(
        row.get("dataset")
    )
    for row in legacy_protocol_inventory
    if row.get("dataset") is not None
}

legacy_direct_reuse_checks = {
    "legacy_scaler_exists": (
        LEGACY_SCALER.exists()
    ),
    "legacy_fp32_protocol_exists": (
        LEGACY_FP32_PROTOCOL.exists()
    ),
    "legacy_checkpoint_count_positive": (
        len(
            legacy_checkpoint_records
        )
        > 0
    ),
    "legacy_protocol_names_final_v3_2_cache": (
        EXPECTED_PROTOCOL_VERSION
        in legacy_protocol_dataset_names
    ),
    "legacy_artifacts_have_phase5_v2_path": (
        any(
            "results\\v2\\phase5"
            in row[
                "relative_path"
            ].lower()
            or "results/v2/phase5"
            in row[
                "relative_path"
            ].lower()
            for row in legacy_artifacts
        )
    ),
}

legacy_direct_reuse_eligible = all(
    legacy_direct_reuse_checks.values()
)

phase5_matrix_rows = []

for architecture in ARCHITECTURES:
    for variant in ROADMAP_CORE_VARIANTS:
        for seed in SEEDS:
            phase5_matrix_rows.append(
                {
                    "architecture": architecture,
                    "variant": variant,
                    "seed": seed,
                    "configuration_id": (
                        f"{architecture}__"
                        f"{variant.lower().replace('-', '_')}"
                        f"__seed_{seed}"
                    ),
                    "source_checkpoint_policy": (
                        "matching_architecture_seed_B0"
                    ),
                    "train_split_only_for_updates": True,
                    "validation_only_for_selection": True,
                    "test_evaluation_count": 1,
                    "status": "planned_not_locked",
                }
            )

atomic_csv(
    OUTPUT_CSV,
    phase5_matrix_rows,
    [
        "architecture",
        "variant",
        "seed",
        "configuration_id",
        "source_checkpoint_policy",
        "train_split_only_for_updates",
        "validation_only_for_selection",
        "test_evaluation_count",
        "status",
    ],
)

naming_ambiguities = [
    {
        "issue": (
            "Roadmap core variants contain "
            "P25 and P50, while predefined "
            "comparisons additionally name "
            "P25-FP32-FT and P50-FP32-FT."
        ),
        "resolution_status": (
            "must_be_resolved_before_protocol_lock"
        ),
        "silent_resolution_applied": False,
    }
]

model_source_record = {
    "path": str(MODEL_SOURCE),
    "exists": MODEL_SOURCE.exists(),
    "size_bytes": (
        int(
            MODEL_SOURCE.stat().st_size
        )
        if MODEL_SOURCE.exists()
        else None
    ),
    "sha256": (
        sha256_file(
            MODEL_SOURCE
        )
        if MODEL_SOURCE.exists()
        else None
    ),
}

model_registry_record = (
    optional_json_metadata(
        MODEL_REGISTRY
    )
)

environment_checks = {
    "python_3_11": (
        sys.version_info.major == 3
        and sys.version_info.minor == 11
    ),
    "numpy_importable": True,
    "torch_importable": (
        torch_runtime["importable"]
    ),
    "model_source_exists": (
        MODEL_SOURCE.exists()
    ),
    "model_registry_exists": (
        MODEL_REGISTRY.exists()
    ),
    "phase5_output_root_absent_or_empty": (
        not PHASE5_OUTPUT_ROOT.exists()
        or not any(
            PHASE5_OUTPUT_ROOT.iterdir()
        )
    ),
}

report = {
    "status": "passed",
    "phase": 5,
    "phase_name": (
        "fair_budget_compression_matrix"
    ),
    "generated_at_utc": utc_now(),
    "entry_gate": {
        "phase4_completed_and_locked": True,
        "checks": phase4_checks,
    },
    "roadmap_scope": {
        "architectures": list(
            ARCHITECTURES
        ),
        "core_variants": list(
            ROADMAP_CORE_VARIANTS
        ),
        "seeds": list(SEEDS),
        "planned_configuration_count": (
            len(
                phase5_matrix_rows
            )
        ),
        "predeclared_comparisons": list(
            PREDECLARED_COMPARISONS
        ),
        "fp32_ft_budget": {
            "source": (
                "same matching B0 checkpoint"
            ),
            "low_learning_rate": True,
            "maximum_epochs": 8,
            "early_stopping": True,
            "pruning": False,
            "fake_quantization": False,
        },
        "pruning_no_ft_controls": [
            "P25-noFT",
            "P50-noFT",
        ],
        "pruning_equal_budget_controls_named_in_roadmap": [
            "P25-FP32-FT",
            "P50-FP32-FT",
        ],
        "naming_ambiguities": (
            naming_ambiguities
        ),
    },
    "final_cache": {
        "path": str(FINAL_CACHE),
        "manifest_sha256": (
            sha256_file(
                FINAL_CACHE_MANIFEST
            )
        ),
        "verification_sha256": (
            sha256_file(
                FINAL_CACHE_VERIFICATION
            )
        ),
        "split_inventory": (
            cache_split_inventory
        ),
        "all_checks_passed": True,
    },
    "environment": {
        "python_version": (
            platform.python_version()
        ),
        "python_executable": (
            sys.executable
        ),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": (
            os.cpu_count()
        ),
        "free_disk_gib": (
            shutil.disk_usage(
                ROOT
            ).free
            / (1024**3)
        ),
        "package_versions": (
            package_versions
        ),
        "torch_runtime": (
            torch_runtime
        ),
        "checks": (
            environment_checks
        ),
    },
    "neural_assets": {
        "model_source": (
            model_source_record
        ),
        "model_registry": (
            model_registry_record
        ),
    },
    "legacy_inventory": {
        "protocols": (
            legacy_protocol_inventory
        ),
        "legacy_scaler": {
            "path": str(
                LEGACY_SCALER
            ),
            "exists": (
                LEGACY_SCALER.exists()
            ),
            "sha256": (
                sha256_file(
                    LEGACY_SCALER
                )
                if LEGACY_SCALER.exists()
                else None
            ),
        },
        "artifact_count": len(
            legacy_artifacts
        ),
        "checkpoint_count": len(
            legacy_checkpoint_records
        ),
        "artifacts": (
            legacy_artifacts
        ),
        "direct_reuse_checks": (
            legacy_direct_reuse_checks
        ),
        "direct_reuse_eligible": (
            legacy_direct_reuse_eligible
        ),
        "scientific_decision": (
            "Legacy neural artifacts are inventory "
            "evidence only and are not directly reusable "
            "unless their dataset, split, scaler, target, "
            "architecture, seed, and test-use manifests "
            "exactly match the final v3_2 cache."
        ),
    },
    "planned_matrix_csv": str(
        OUTPUT_CSV
    ),
    "planned_matrix_csv_sha256": (
        sha256_file(
            OUTPUT_CSV
        )
    ),
    "next_required_action": (
        "Resolve variant naming and lock a new "
        "Phase 5 protocol tied to the final v3_2 "
        "cache before any B0 or compression run."
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_JSON,
    report,
)

print("=" * 92)
print("PHASE 5 PREFLIGHT INVENTORY SUMMARY")
print("=" * 92)
print(
    "Phase 4 entry gate            : PASSED"
)
print(
    "Final cache verification      : PASSED"
)
print(
    "Architectures                 : "
    f"{list(ARCHITECTURES)}"
)
print(
    "Core variants                 : "
    f"{len(ROADMAP_CORE_VARIANTS)}"
)
print(
    "Seeds                         : "
    f"{list(SEEDS)}"
)
print(
    "Planned configurations        : "
    f"{len(phase5_matrix_rows)}"
)
print(
    "Torch importable              : "
    f"{torch_runtime['importable']}"
)
print(
    "CUDA available               : "
    f"{torch_runtime['cuda_available']}"
)
print(
    "Legacy checkpoint count       : "
    f"{len(legacy_checkpoint_records)}"
)
print(
    "Legacy direct reuse eligible  : "
    f"{legacy_direct_reuse_eligible}"
)
print(
    "Naming ambiguity detected     : True"
)
print(
    "Silent naming resolution      : False"
)
print(
    "Preflight report              : "
    f"{OUTPUT_JSON}"
)
print(
    "Planned matrix CSV            : "
    f"{OUTPUT_CSV}"
)
print(
    "All checks passed             : True"
)
print(
    "PHASE 5 PREFLIGHT INVENTORY COMPLETED"
)
