from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path.cwd()
AUDIT = ROOT / "results" / "v2" / "audit"

PHASE5_PROTOCOL = (
    ROOT
    / "configs"
    / "protocols"
    / "phase5_fair_budget_compression_protocol_v3_2.json"
)

B0_PAIR_LOCK = (
    AUDIT / "phase5_B0_pair_locked_v3_2.json"
)

B0_CHECKPOINT_REGISTRY = (
    AUDIT / "phase5_B0_checkpoint_registry_v3_2.csv"
)

PRUNING_PREFLIGHT_LOCK = (
    AUDIT
    / "phase5_physical_pruning_preflight_locked_v3_2.json"
)

PRUNING_ENGINE_LOCK = (
    AUDIT
    / "phase5_physical_pruning_engine_locked_v3_2.json"
)

PRUNING_ENGINE_REPORT = (
    AUDIT
    / "phase5_physical_pruning_engine_verification_v3_2.json"
)

PRUNING_ENGINE_PATH = (
    ROOT
    / "src"
    / "compression"
    / "phase5_physical_pruning_engine_v3_2.py"
)

MODEL_SOURCE = (
    ROOT / "src" / "models" / "nbaiot_models.py"
)

FINAL_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "tabular_baseline_family3_v3_2"
)

X_TRAIN_PATH = FINAL_CACHE / "X_train.npy"

SCALER_NPZ = (
    ROOT
    / "results"
    / "v2"
    / "phase5_compression"
    / "shared"
    / "preprocessing"
    / "phase5_train_only_standard_scaler_v3_2.npz"
)

OUTPUT_REGISTRY = (
    AUDIT
    / "phase5_physical_pruning_source_checkpoint_registry_v3_2.csv"
)

OUTPUT_SUMMARY = (
    AUDIT
    / "phase5_physical_pruning_sources_summary_v3_2.json"
)

OUTPUT_LOCK = (
    AUDIT
    / "phase5_physical_pruning_sources_locked_v3_2.json"
)

OUTPUT_LOCK_MANIFEST = (
    AUDIT
    / "phase5_physical_pruning_sources_lock_manifest_v3_2.json"
)

PROTOCOL_VERSION = "phase5_fair_budget_compression_v3_2"
ENGINE_VERSION = "phase5_physical_pruning_engine_v3_2"

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

VARIANTS = {
    0.25: "P25-noFT",
    0.50: "P50-noFT",
}

DIRECTORY_NAMES = {
    "P25-noFT": "p25_noft",
    "P50-noFT": "p50_noft",
}

EXPECTED_WIDTHS = {
    "tinyml_mlp": {
        "P25-noFT": [
            [115, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 32],
            [32, 16],
            [16, 3],
        ],
    },
    "compact_dnn": {
        "P25-noFT": [
            [115, 96],
            [96, 48],
            [48, 24],
            [24, 3],
        ],
        "P50-noFT": [
            [115, 64],
            [64, 32],
            [32, 16],
            [16, 3],
        ],
    },
}

HIDDEN_DIMS = {
    "tinyml_mlp": [64, 32],
    "compact_dnn": [128, 64, 32],
}

EXPECTED_INPUT_FEATURES = 115
EXPECTED_OUTPUT_CLASSES = 3
SMOKE_BATCH_SIZE = 64
CPU_THREADS = 4


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


def atomic_torch_save(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        value,
        temporary,
    )

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


def load_module(
    path: Path,
    module_name: str,
) -> ModuleType:
    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


def callable_kwargs(
    candidate: Any,
    architecture: str,
) -> dict[str, Any]:
    signature = inspect.signature(
        candidate
    )

    kwargs: dict[str, Any] = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in {
            "self",
            "args",
            "kwargs",
        }:
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        normalized = "".join(
            character
            for character in name.lower()
            if character.isalnum()
        )

        if normalized in {
            "inputdim",
            "inputsize",
            "inputfeatures",
            "nfeatures",
            "numfeatures",
            "infeatures",
            "featurecount",
        }:
            kwargs[name] = (
                EXPECTED_INPUT_FEATURES
            )
            continue

        if normalized in {
            "numclasses",
            "nclasses",
            "classcount",
            "outputdim",
            "outputsize",
            "outfeatures",
        }:
            kwargs[name] = (
                EXPECTED_OUTPUT_CLASSES
            )
            continue

        if normalized in {
            "hiddendims",
            "hiddensizes",
            "hiddenlayers",
            "hiddenunits",
        }:
            kwargs[name] = list(
                HIDDEN_DIMS[
                    architecture
                ]
            )
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Unresolved model constructor "
                f"parameter: {name}"
            )

    return kwargs


def instantiate_model(
    model_module: ModuleType,
    model_symbol: str,
    architecture: str,
) -> nn.Module:
    if not hasattr(
        model_module,
        model_symbol,
    ):
        raise RuntimeError(
            f"Missing model symbol: {model_symbol}"
        )

    candidate = getattr(
        model_module,
        model_symbol,
    )

    model = candidate(
        **callable_kwargs(
            candidate,
            architecture,
        )
    )

    if not isinstance(
        model,
        nn.Module,
    ):
        raise RuntimeError(
            "Model constructor did not "
            "return torch.nn.Module."
        )

    return model


def extract_logits(
    output: Any,
) -> torch.Tensor:
    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):
        for value in output:
            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    if isinstance(
        output,
        dict,
    ):
        for key in (
            "logits",
            "output",
            "outputs",
            "predictions",
        ):
            value = output.get(key)

            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    raise RuntimeError(
        "Model output did not contain logits."
    )


def tensor_state_digest(
    model: nn.Module,
) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):
        contiguous = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )

        digest.update(
            str(
                contiguous.dtype
            ).encode("ascii")
        )

        digest.update(
            np.asarray(
                contiguous.shape,
                dtype=np.int64,
            ).tobytes()
        )

        digest.update(
            contiguous.numpy().tobytes()
        )

    return digest.hexdigest()


def branch_directory(
    architecture: str,
    variant: str,
    seed: int,
) -> Path:
    return (
        ROOT
        / "results"
        / "v2"
        / "phase5_compression"
        / architecture
        / "pruning_sources"
        / DIRECTORY_NAMES[
            variant
        ]
        / f"seed_{seed}"
    )


required_paths = (
    PHASE5_PROTOCOL,
    B0_PAIR_LOCK,
    B0_CHECKPOINT_REGISTRY,
    PRUNING_PREFLIGHT_LOCK,
    PRUNING_ENGINE_LOCK,
    PRUNING_ENGINE_REPORT,
    PRUNING_ENGINE_PATH,
    MODEL_SOURCE,
    X_TRAIN_PATH,
    SCALER_NPZ,
)

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

for output_path in (
    OUTPUT_REGISTRY,
    OUTPUT_SUMMARY,
    OUTPUT_LOCK,
    OUTPUT_LOCK_MANIFEST,
):
    if output_path.exists():
        raise FileExistsError(
            "Pruning-source aggregate artifact "
            "already exists; refusing to "
            f"overwrite: {output_path}"
        )

protocol = read_json(
    PHASE5_PROTOCOL
)

b0_pair_lock = read_json(
    B0_PAIR_LOCK
)

preflight_lock = read_json(
    PRUNING_PREFLIGHT_LOCK
)

engine_lock = read_json(
    PRUNING_ENGINE_LOCK
)

engine_report = read_json(
    PRUNING_ENGINE_REPORT
)

registry_rows = read_csv(
    B0_CHECKPOINT_REGISTRY
)

entry_checks = {
    "protocol_locked": (
        protocol.get("status")
        == "locked"
        and protocol.get(
            "protocol_version"
        )
        == PROTOCOL_VERSION
    ),
    "B0_pair_locked": (
        b0_pair_lock.get(
            "status"
        )
        == "locked"
        and b0_pair_lock.get(
            "ready_for_branch_source_generation"
        )
        is True
    ),
    "B0_registry_hash_matches": (
        b0_pair_lock.get(
            "checkpoint_registry_sha256"
        )
        == sha256_file(
            B0_CHECKPOINT_REGISTRY
        )
    ),
    "pruning_preflight_locked": (
        preflight_lock.get(
            "status"
        )
        == "locked"
        and preflight_lock.get(
            "ready_for_pruning_engine_build"
        )
        is True
    ),
    "pruning_engine_locked": (
        engine_lock.get(
            "status"
        )
        == "locked"
        and engine_lock.get(
            "ready_for_P25_noFT_and_P50_noFT_generation"
        )
        is True
        and engine_lock.get(
            "all_checks_passed"
        )
        is True
    ),
    "engine_source_hash_matches": (
        engine_lock.get(
            "engine_source_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_PATH
        )
    ),
    "engine_report_hash_matches": (
        engine_lock.get(
            "verification_report_sha256"
        )
        == sha256_file(
            PRUNING_ENGINE_REPORT
        )
    ),
    "registry_has_ten_B0_rows": (
        len(registry_rows)
        == 10
    ),
    "registry_architecture_seed_matrix_matches": (
        {
            (
                row["architecture"],
                int(row["seed"]),
            )
            for row in registry_rows
        }
        == {
            (
                architecture,
                seed,
            )
            for architecture
            in ARCHITECTURES
            for seed in SEEDS
        }
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
        "Pruning-source generation entry "
        "gate failed: "
        + ", ".join(
            failed_entry_checks
        )
    )

torch.set_num_threads(
    CPU_THREADS
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.use_deterministic_algorithms(
    True
)

x_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

with np.load(
    SCALER_NPZ
) as values:
    mean64 = np.array(
        values["mean_float64"],
        copy=True,
    )

    scale64 = np.array(
        values["scale_float64"],
        copy=True,
    )

smoke_numpy = (
    (
        np.asarray(
            x_train[
                :SMOKE_BATCH_SIZE
            ],
            dtype=np.float64,
        )
        - mean64
    )
    / scale64
).astype(np.float32)

if not np.isfinite(
    smoke_numpy
).all():
    raise RuntimeError(
        "Non-finite train-only smoke batch."
    )

smoke_batch = torch.from_numpy(
    smoke_numpy
)

model_module = load_module(
    MODEL_SOURCE,
    "phase5_pruning_source_model_module",
)

engine_module = load_module(
    PRUNING_ENGINE_PATH,
    "phase5_pruning_source_engine_module",
)

if (
    engine_module.ENGINE_VERSION
    != ENGINE_VERSION
):
    raise RuntimeError(
        "Pruning-engine version mismatch."
    )

registry_output_rows: list[
    dict[str, Any]
] = []

generated_count = 0
resumed_count = 0

sorted_B0_rows = sorted(
    registry_rows,
    key=lambda row: (
        row["architecture"],
        int(row["seed"]),
    ),
)

branch_index = 0

for B0_row in sorted_B0_rows:
    architecture = (
        B0_row["architecture"]
    )

    seed = int(
        B0_row["seed"]
    )

    model_symbol = (
        B0_row["model_symbol"]
    )

    B0_checkpoint_path = Path(
        B0_row["checkpoint_path"]
    )

    B0_checkpoint_hash = (
        B0_row["checkpoint_sha256"]
    )

    if sha256_file(
        B0_checkpoint_path
    ) != B0_checkpoint_hash:
        raise RuntimeError(
            "B0 checkpoint hash mismatch: "
            f"{B0_checkpoint_path}"
        )

    B0_checkpoint = torch.load(
        B0_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    B0_model = instantiate_model(
        model_module,
        model_symbol,
        architecture,
    )

    B0_model.load_state_dict(
        B0_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    B0_model.eval()

    B0_digest_before = tensor_state_digest(
        B0_model
    )

    for pruning_ratio, variant in (
        VARIANTS.items()
    ):
        branch_index += 1

        final_directory = branch_directory(
            architecture,
            variant,
            seed,
        )

        temporary_directory = (
            final_directory.with_name(
                final_directory.name
                + ".tmp_generation"
            )
        )

        checkpoint_path = (
            final_directory
            / "source_checkpoint.pt"
        )

        manifest_path = (
            final_directory
            / "source_manifest.json"
        )

        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory
            )

        if final_directory.exists():
            if not (
                checkpoint_path.exists()
                and manifest_path.exists()
            ):
                raise FileExistsError(
                    "Incomplete existing pruning-source "
                    f"directory: {final_directory}"
                )

            manifest = read_json(
                manifest_path
            )

            resume_checks = {
                "status_completed": (
                    manifest.get("status")
                    == "completed"
                ),
                "identity_matches": (
                    manifest.get(
                        "architecture"
                    )
                    == architecture
                    and manifest.get(
                        "variant"
                    )
                    == variant
                    and int(
                        manifest.get("seed")
                    )
                    == seed
                ),
                "B0_hash_matches": (
                    manifest.get(
                        "source_B0_checkpoint_sha256"
                    )
                    == B0_checkpoint_hash
                ),
                "engine_hash_matches": (
                    manifest.get(
                        "pruning_engine_sha256"
                    )
                    == sha256_file(
                        PRUNING_ENGINE_PATH
                    )
                ),
                "checkpoint_hash_matches": (
                    manifest.get(
                        "source_checkpoint_sha256"
                    )
                    == sha256_file(
                        checkpoint_path
                    )
                ),
                "validation_access_zero": (
                    int(
                        manifest.get(
                            "validation_access_count"
                        )
                    )
                    == 0
                ),
                "test_access_zero": (
                    int(
                        manifest.get(
                            "test_access_count"
                        )
                    )
                    == 0
                ),
            }

            failed_resume_checks = [
                name
                for name, passed
                in resume_checks.items()
                if not passed
            ]

            if failed_resume_checks:
                raise RuntimeError(
                    "Existing pruning-source resume "
                    f"check failed for {architecture} "
                    f"seed {seed} {variant}: "
                    + ", ".join(
                        failed_resume_checks
                    )
                )

            stored = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            metadata = stored[
                "pruning_metadata"
            ]

            resumed_count += 1
            status = "resumed_existing"

        else:
            temporary_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            (
                pruned_model,
                metadata,
            ) = engine_module.physically_prune_mlp(
                B0_model,
                pruning_ratio,
                source_checkpoint_sha256=(
                    B0_checkpoint_hash
                ),
            )

            pruned_model.eval()

            expected_widths = (
                EXPECTED_WIDTHS[
                    architecture
                ][variant]
            )

            with torch.inference_mode():
                logits = extract_logits(
                    pruned_model(
                        smoke_batch
                    )
                )

            generation_checks = {
                "expected_widths_match": (
                    metadata[
                        "compacted_linear_widths"
                    ]
                    == expected_widths
                ),
                "source_is_direct_B0": (
                    metadata[
                        "generated_directly_from_B0"
                    ]
                    is True
                ),
                "physical_compaction_true": (
                    metadata[
                        "physical_compaction"
                    ]
                    is True
                    and metadata[
                        "mask_only_pruning"
                    ]
                    is False
                ),
                "parameter_count_reduced": (
                    metadata[
                        "compacted_parameter_count"
                    ]
                    < metadata[
                        "source_parameter_count"
                    ]
                ),
                "MAC_count_reduced": (
                    metadata[
                        "compacted_linear_macs_per_sample"
                    ]
                    < metadata[
                        "source_linear_macs_per_sample"
                    ]
                ),
                "smoke_output_shape_matches": (
                    tuple(
                        logits.shape
                    )
                    == (
                        SMOKE_BATCH_SIZE,
                        EXPECTED_OUTPUT_CLASSES,
                    )
                ),
                "smoke_output_finite": (
                    bool(
                        torch.isfinite(
                            logits
                        ).all().item()
                    )
                ),
                "B0_model_unchanged": (
                    tensor_state_digest(
                        B0_model
                    )
                    == B0_digest_before
                ),
            }

            failed_generation_checks = [
                name
                for name, passed
                in generation_checks.items()
                if not passed
            ]

            if failed_generation_checks:
                raise RuntimeError(
                    f"{architecture} seed {seed} "
                    f"{variant} generation failed: "
                    + ", ".join(
                        failed_generation_checks
                    )
                )

            temporary_checkpoint = (
                temporary_directory
                / "source_checkpoint.pt"
            )

            atomic_torch_save(
                {
                    "artifact_type": (
                        "phase5_physical_pruning_source"
                    ),
                    "protocol_version": (
                        PROTOCOL_VERSION
                    ),
                    "engine_version": (
                        ENGINE_VERSION
                    ),
                    "architecture": (
                        architecture
                    ),
                    "variant": variant,
                    "seed": seed,
                    "model_symbol": (
                        model_symbol
                    ),
                    "source_B0_checkpoint_path": (
                        str(
                            B0_checkpoint_path
                        )
                    ),
                    "source_B0_checkpoint_sha256": (
                        B0_checkpoint_hash
                    ),
                    "pruning_engine_path": (
                        str(
                            PRUNING_ENGINE_PATH
                        )
                    ),
                    "pruning_engine_sha256": (
                        sha256_file(
                            PRUNING_ENGINE_PATH
                        )
                    ),
                    "model_state_dict": (
                        pruned_model.state_dict()
                    ),
                    "pruning_metadata": (
                        metadata
                    ),
                    "train_smoke_rows_used": (
                        SMOKE_BATCH_SIZE
                    ),
                    "validation_access_count": 0,
                    "test_access_count": 0,
                    "generated_at_utc": (
                        utc_now()
                    ),
                },
                temporary_checkpoint,
            )

            stored = torch.load(
                temporary_checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            roundtrip_checks = {
                "identity_matches": (
                    stored[
                        "architecture"
                    ]
                    == architecture
                    and stored[
                        "variant"
                    ]
                    == variant
                    and int(
                        stored["seed"]
                    )
                    == seed
                ),
                "metadata_matches": (
                    stored[
                        "pruning_metadata"
                    ]
                    == metadata
                ),
                "state_keys_match": (
                    set(
                        stored[
                            "model_state_dict"
                        ].keys()
                    )
                    == set(
                        pruned_model.state_dict().keys()
                    )
                ),
                "state_tensors_match": all(
                    torch.equal(
                        stored[
                            "model_state_dict"
                        ][key],
                        value,
                    )
                    for key, value
                    in pruned_model.state_dict().items()
                ),
            }

            failed_roundtrip_checks = [
                name
                for name, passed
                in roundtrip_checks.items()
                if not passed
            ]

            if failed_roundtrip_checks:
                raise RuntimeError(
                    f"{architecture} seed {seed} "
                    f"{variant} checkpoint roundtrip failed: "
                    + ", ".join(
                        failed_roundtrip_checks
                    )
                )

            temporary_manifest = (
                temporary_directory
                / "source_manifest.json"
            )

            atomic_json(
                temporary_manifest,
                {
                    "status": "completed",
                    "artifact_type": (
                        "phase5_physical_pruning_source"
                    ),
                    "protocol_version": (
                        PROTOCOL_VERSION
                    ),
                    "engine_version": (
                        ENGINE_VERSION
                    ),
                    "architecture": (
                        architecture
                    ),
                    "variant": variant,
                    "seed": seed,
                    "model_symbol": (
                        model_symbol
                    ),
                    "pruning_ratio": (
                        pruning_ratio
                    ),
                    "source_B0_checkpoint_path": (
                        str(
                            B0_checkpoint_path
                        )
                    ),
                    "source_B0_checkpoint_sha256": (
                        B0_checkpoint_hash
                    ),
                    "pruning_engine_path": (
                        str(
                            PRUNING_ENGINE_PATH
                        )
                    ),
                    "pruning_engine_sha256": (
                        sha256_file(
                            PRUNING_ENGINE_PATH
                        )
                    ),
                    "source_checkpoint_path": (
                        str(
                            final_directory
                            / "source_checkpoint.pt"
                        )
                    ),
                    "source_checkpoint_sha256": (
                        sha256_file(
                            temporary_checkpoint
                        )
                    ),
                    "source_checkpoint_size_bytes": (
                        int(
                            temporary_checkpoint
                            .stat()
                            .st_size
                        )
                    ),
                    "source_linear_widths": (
                        metadata[
                            "source_linear_widths"
                        ]
                    ),
                    "compacted_linear_widths": (
                        metadata[
                            "compacted_linear_widths"
                        ]
                    ),
                    "source_parameter_count": (
                        metadata[
                            "source_parameter_count"
                        ]
                    ),
                    "compacted_parameter_count": (
                        metadata[
                            "compacted_parameter_count"
                        ]
                    ),
                    "source_linear_macs_per_sample": (
                        metadata[
                            "source_linear_macs_per_sample"
                        ]
                    ),
                    "compacted_linear_macs_per_sample": (
                        metadata[
                            "compacted_linear_macs_per_sample"
                        ]
                    ),
                    "generated_directly_from_B0": (
                        True
                    ),
                    "physical_compaction": (
                        True
                    ),
                    "mask_only_pruning": (
                        False
                    ),
                    "train_smoke_rows_used": (
                        SMOKE_BATCH_SIZE
                    ),
                    "validation_access_count": 0,
                    "test_access_count": 0,
                    "generated_at_utc": (
                        utc_now()
                    ),
                    "all_checks_passed": True,
                },
            )

            os.replace(
                temporary_directory,
                final_directory,
            )

            checkpoint_path = (
                final_directory
                / "source_checkpoint.pt"
            )

            manifest_path = (
                final_directory
                / "source_manifest.json"
            )

            manifest = read_json(
                manifest_path
            )

            if (
                manifest[
                    "source_checkpoint_sha256"
                ]
                != sha256_file(
                    checkpoint_path
                )
            ):
                raise RuntimeError(
                    "Post-move source-checkpoint "
                    "hash mismatch."
                )

            generated_count += 1
            status = "generated_new"

            del pruned_model

        expected_widths = (
            EXPECTED_WIDTHS[
                architecture
            ][variant]
        )

        if (
            metadata[
                "compacted_linear_widths"
            ]
            != expected_widths
        ):
            raise RuntimeError(
                "Registered compacted widths "
                "do not match the locked plan."
            )

        registry_output_rows.append(
            {
                "architecture": architecture,
                "variant": variant,
                "seed": seed,
                "model_symbol": model_symbol,
                "pruning_ratio": (
                    pruning_ratio
                ),
                "source_B0_checkpoint_path": (
                    str(
                        B0_checkpoint_path
                    )
                ),
                "source_B0_checkpoint_sha256": (
                    B0_checkpoint_hash
                ),
                "pruning_engine_path": (
                    str(
                        PRUNING_ENGINE_PATH
                    )
                ),
                "pruning_engine_sha256": (
                    sha256_file(
                        PRUNING_ENGINE_PATH
                    )
                ),
                "source_checkpoint_path": (
                    str(
                        checkpoint_path
                    )
                ),
                "source_checkpoint_size_bytes": (
                    int(
                        checkpoint_path
                        .stat()
                        .st_size
                    )
                ),
                "source_checkpoint_sha256": (
                    sha256_file(
                        checkpoint_path
                    )
                ),
                "source_manifest_path": (
                    str(
                        manifest_path
                    )
                ),
                "source_manifest_sha256": (
                    sha256_file(
                        manifest_path
                    )
                ),
                "source_linear_widths": (
                    json.dumps(
                        metadata[
                            "source_linear_widths"
                        ],
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                ),
                "compacted_linear_widths": (
                    json.dumps(
                        metadata[
                            "compacted_linear_widths"
                        ],
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                ),
                "source_parameter_count": (
                    metadata[
                        "source_parameter_count"
                    ]
                ),
                "compacted_parameter_count": (
                    metadata[
                        "compacted_parameter_count"
                    ]
                ),
                "parameter_reduction_fraction": (
                    1.0
                    - metadata[
                        "compacted_parameter_count"
                    ]
                    / metadata[
                        "source_parameter_count"
                    ]
                ),
                "source_linear_macs_per_sample": (
                    metadata[
                        "source_linear_macs_per_sample"
                    ]
                ),
                "compacted_linear_macs_per_sample": (
                    metadata[
                        "compacted_linear_macs_per_sample"
                    ]
                ),
                "MAC_reduction_fraction": (
                    1.0
                    - metadata[
                        "compacted_linear_macs_per_sample"
                    ]
                    / metadata[
                        "source_linear_macs_per_sample"
                    ]
                ),
                "generated_directly_from_B0": True,
                "physical_compaction": True,
                "mask_only_pruning": False,
                "validation_access_count": 0,
                "test_access_count": 0,
                "generation_status": status,
                "registry_status": (
                    "locked_pruning_source_candidate"
                ),
            }
        )

        print(
            f"[{branch_index}/20] "
            f"{architecture} seed={seed} "
            f"{variant} | "
            f"widths={expected_widths} | "
            f"status={status} | "
            "validation=0 | test=0",
            flush=True,
        )

    del B0_model
    del B0_checkpoint

if len(
    registry_output_rows
) != 20:
    raise RuntimeError(
        "Expected exactly twenty pruning "
        "source checkpoints."
    )

registry_output_rows.sort(
    key=lambda row: (
        row["architecture"],
        row["variant"],
        int(row["seed"]),
    )
)

registry_checks = {
    "twenty_rows_present": (
        len(
            registry_output_rows
        )
        == 20
    ),
    "architecture_variant_seed_matrix_complete": (
        {
            (
                row["architecture"],
                row["variant"],
                int(row["seed"]),
            )
            for row in registry_output_rows
        }
        == {
            (
                architecture,
                variant,
                seed,
            )
            for architecture in ARCHITECTURES
            for variant in (
                "P25-noFT",
                "P50-noFT",
            )
            for seed in SEEDS
        }
    ),
    "all_checkpoint_hashes_unique": (
        len(
            {
                row[
                    "source_checkpoint_sha256"
                ]
                for row in registry_output_rows
            }
        )
        == 20
    ),
    "all_source_B0_hashes_match_registry": all(
        row[
            "source_B0_checkpoint_sha256"
        ]
        in {
            B0_row[
                "checkpoint_sha256"
            ]
            for B0_row in registry_rows
        }
        for row in registry_output_rows
    ),
    "all_generated_directly_from_B0": all(
        row[
            "generated_directly_from_B0"
        ]
        is True
        for row in registry_output_rows
    ),
    "all_physical_not_mask_only": all(
        row[
            "physical_compaction"
        ]
        is True
        and row[
            "mask_only_pruning"
        ]
        is False
        for row in registry_output_rows
    ),
    "all_validation_counts_zero": all(
        int(
            row[
                "validation_access_count"
            ]
        )
        == 0
        for row in registry_output_rows
    ),
    "all_test_counts_zero": all(
        int(
            row[
                "test_access_count"
            ]
        )
        == 0
        for row in registry_output_rows
    ),
}

failed_registry_checks = [
    name
    for name, passed
    in registry_checks.items()
    if not passed
]

if failed_registry_checks:
    raise RuntimeError(
        "Pruning-source registry checks "
        "failed: "
        + ", ".join(
            failed_registry_checks
        )
    )

atomic_csv(
    OUTPUT_REGISTRY,
    registry_output_rows,
    [
        "architecture",
        "variant",
        "seed",
        "model_symbol",
        "pruning_ratio",
        "source_B0_checkpoint_path",
        "source_B0_checkpoint_sha256",
        "pruning_engine_path",
        "pruning_engine_sha256",
        "source_checkpoint_path",
        "source_checkpoint_size_bytes",
        "source_checkpoint_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_linear_widths",
        "compacted_linear_widths",
        "source_parameter_count",
        "compacted_parameter_count",
        "parameter_reduction_fraction",
        "source_linear_macs_per_sample",
        "compacted_linear_macs_per_sample",
        "MAC_reduction_fraction",
        "generated_directly_from_B0",
        "physical_compaction",
        "mask_only_pruning",
        "validation_access_count",
        "test_access_count",
        "generation_status",
        "registry_status",
    ],
)

group_summaries: list[
    dict[str, Any]
] = []

for architecture in ARCHITECTURES:
    for variant in (
        "P25-noFT",
        "P50-noFT",
    ):
        rows = [
            row
            for row in registry_output_rows
            if row["architecture"]
            == architecture
            and row["variant"]
            == variant
        ]

        group_summaries.append(
            {
                "architecture": (
                    architecture
                ),
                "variant": variant,
                "run_count": len(rows),
                "seeds": list(SEEDS),
                "compacted_linear_widths": (
                    json.loads(
                        rows[0][
                            "compacted_linear_widths"
                        ]
                    )
                ),
                "source_parameter_count": (
                    int(
                        rows[0][
                            "source_parameter_count"
                        ]
                    )
                ),
                "compacted_parameter_count": (
                    int(
                        rows[0][
                            "compacted_parameter_count"
                        ]
                    )
                ),
                "parameter_reduction_fraction": (
                    float(
                        rows[0][
                            "parameter_reduction_fraction"
                        ]
                    )
                ),
                "source_linear_macs_per_sample": (
                    int(
                        rows[0][
                            "source_linear_macs_per_sample"
                        ]
                    )
                ),
                "compacted_linear_macs_per_sample": (
                    int(
                        rows[0][
                            "compacted_linear_macs_per_sample"
                        ]
                    )
                ),
                "MAC_reduction_fraction": (
                    float(
                        rows[0][
                            "MAC_reduction_fraction"
                        ]
                    )
                ),
            }
        )

summary = {
    "status": "completed",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "generated_at_utc": utc_now(),
    "checkpoint_count": 20,
    "newly_generated_count": (
        generated_count
    ),
    "resumed_existing_count": (
        resumed_count
    ),
    "architecture_count": 2,
    "variant_count": 2,
    "seeds": list(SEEDS),
    "groups": group_summaries,
    "source_policy": {
        "P25_generated_directly_from_B0": True,
        "P50_generated_directly_from_B0": True,
        "P50_generated_from_P25": False,
        "same_P25_source_reserved_for_P25_and_P25_QAT": True,
        "same_P50_source_reserved_for_P50_and_P50_QAT": True,
        "physical_compaction": True,
        "mask_only_pruning": False,
    },
    "data_access": {
        "train_smoke_rows_used_per_generation": (
            SMOKE_BATCH_SIZE
        ),
        "validation_access_count": 0,
        "test_access_count": 0,
    },
    "entry_checks": entry_checks,
    "registry_checks": (
        registry_checks
    ),
    "registry_csv": (
        file_record(
            OUTPUT_REGISTRY
        )
    ),
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_SUMMARY,
    summary,
)

lock = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "engine_version": (
        ENGINE_VERSION
    ),
    "locked_at_utc": utc_now(),
    "checkpoint_count": 20,
    "registry": str(
        OUTPUT_REGISTRY
    ),
    "registry_sha256": (
        sha256_file(
            OUTPUT_REGISTRY
        )
    ),
    "summary": str(
        OUTPUT_SUMMARY
    ),
    "summary_sha256": (
        sha256_file(
            OUTPUT_SUMMARY
        )
    ),
    "validation_access_count": 0,
    "test_access_count": 0,
    "ready_for_independent_pruning_source_verification": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK,
    lock,
)

lock_manifest = {
    "status": "locked",
    "phase": 5,
    "artifact_name": (
        "physical_pruning_source_checkpoints"
    ),
    "protocol_version": (
        PROTOCOL_VERSION
    ),
    "locked_at_utc": utc_now(),
    "source_artifacts": [
        file_record(
            PHASE5_PROTOCOL
        ),
        file_record(
            B0_PAIR_LOCK
        ),
        file_record(
            B0_CHECKPOINT_REGISTRY
        ),
        file_record(
            PRUNING_PREFLIGHT_LOCK
        ),
        file_record(
            PRUNING_ENGINE_LOCK
        ),
        file_record(
            PRUNING_ENGINE_REPORT
        ),
        file_record(
            PRUNING_ENGINE_PATH
        ),
        file_record(
            MODEL_SOURCE
        ),
        file_record(
            SCALER_NPZ
        ),
    ],
    "generated_artifacts": [
        file_record(
            OUTPUT_REGISTRY
        ),
        file_record(
            OUTPUT_SUMMARY
        ),
        file_record(
            OUTPUT_LOCK
        ),
    ],
    "checkpoint_count": 20,
    "ready_for_independent_pruning_source_verification": True,
    "all_checks_passed": True,
}

atomic_json(
    OUTPUT_LOCK_MANIFEST,
    lock_manifest,
)

post_checks = {
    "registry_exists": (
        OUTPUT_REGISTRY.exists()
    ),
    "summary_exists": (
        OUTPUT_SUMMARY.exists()
    ),
    "lock_exists": (
        OUTPUT_LOCK.exists()
    ),
    "lock_manifest_exists": (
        OUTPUT_LOCK_MANIFEST.exists()
    ),
    "registry_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "registry_sha256"
        ]
        == sha256_file(
            OUTPUT_REGISTRY
        )
    ),
    "summary_hash_matches_lock": (
        read_json(
            OUTPUT_LOCK
        )[
            "summary_sha256"
        ]
        == sha256_file(
            OUTPUT_SUMMARY
        )
    ),
    "ready_for_independent_verification": (
        read_json(
            OUTPUT_LOCK
        ).get(
            "ready_for_independent_pruning_source_verification"
        )
        is True
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
        "Pruning-source post-lock checks "
        "failed: "
        + ", ".join(
            failed_post_checks
        )
    )

print()
print("=" * 92)
print("PHASE 5 PHYSICAL PRUNING SOURCE GENERATION SUMMARY")
print("=" * 92)
print(
    "Source checkpoints completed    : 20"
)
print(
    "Newly generated                 : "
    f"{generated_count}"
)
print(
    "Resumed existing                : "
    f"{resumed_count}"
)
print(
    "Architectures                   : 2"
)
print(
    "Variants                        : P25-noFT, P50-noFT"
)
print(
    "Seeds                           : "
    f"{list(SEEDS)}"
)
print(
    "P25 directly from B0            : True"
)
print(
    "P50 directly from B0            : True"
)
print(
    "Physical compaction             : True"
)
print(
    "Validation access count         : 0"
)
print(
    "Test access count               : 0"
)
print(
    "Source registry status          : LOCKED"
)
print(
    "Ready for independent verify    : True"
)
print(
    "Registry                        : "
    f"{OUTPUT_REGISTRY}"
)
print(
    "Summary                         : "
    f"{OUTPUT_SUMMARY}"
)
print(
    "All checks passed               : True"
)
print(
    "PHASE 5 PHYSICAL PRUNING SOURCES GENERATED AND LOCKED"
)
